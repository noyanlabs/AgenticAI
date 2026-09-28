# The Backbone - precisely spinal cord
# Initially the input to the Central Backend comes from user (via server.py). It runs the chain: LLM -> tasks -> results -> LLM ... until final_answer.
# Communication with the server is done ONLY through two injected callables: emit(event) and wait_for_reply(request_id, timeout), so this file never imports the server.
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import re
import shutil
import threading
import time
import uuid
import sys

from llama_cpp import Llama, LlamaGrammar

from ProjectBackend import sandbox
from ProjectBackend import Interpreter
from ProjectBackend.Interpreter.sql_utils import run_select
from ProjectBackend.Interpreter.image_and_encoded_image_interpreter import prepare as prepare_image

BASE_DIR = Path(__file__).resolve().parent
STORAGE = (BASE_DIR / "LocalStorage").resolve()
HISTORY = BASE_DIR / "AgentHistory"
def _find_default_model() -> Path:
    # ori = BASE_DIR / "LLM" / "main.gguf"
    # if ori.is_file():
    #     return ori
    return BASE_DIR / "LLM" / "main.gguf"

MODEL_PATH = Path(os.environ.get("AGENTICAI_MODEL", _find_default_model()))
MMPROJ_PATH = Path(os.environ.get("AGENTICAI_MMPROJ", BASE_DIR / "LLM" / "main_vision.gguf"))
METADATA_FILE = ".metadata.json"
INTERNAL_DIRS = {".interpreted", ".packages"}          # hidden working folders the agent should not treat as user files
CONTEXT_WINDOW = int(os.environ.get("AGENTICAI_CTX", "50000"))
MIN_CONTEXT_WINDOW = 8192
MAX_OUTPUT_TOKENS = 5000
MAX_DECODE_RETRIES = 2      # how many times a full context reset is attempted after a KV-cache/decode failure before giving up
NUDGE_WINDOW_SECONDS = 8
PERMISSION_TIMEOUT_SECONDS = 300
READ_LIMIT_CHARS = 10000
METADATA_PROMPT_LIMIT = 6000                            # chars of metadata injected into the prompt each turn (prevents prompt bloat)
PARALLEL_ACTIONS = {"list_dir", "read_file", "read_document", "read_metadata", "search_files", "query_sql"}
DESTRUCTIVE_ACTIONS = {"delete_path", "move_path"}
MAX_PARALLEL = 6
CONTEXT_SAFETY_RATIO = 0.90                             # start trimming old tool results at this fraction of the window

# Actions whose raw output is bulky/exploratory and gets compressed into a short, task-aware summary before it
# ever touches sess.context or the user's screen. Everything NOT in this set (writes, sandbox runs, control
# actions) passes through unchanged - see run_task() / SUMMARIZE_OUTPUT_CHARS_SKIP below.
SUMMARIZE_ACTIONS = {"read_file", "read_document", "read_metadata", "search_files", "query_sql", "analyze_image", "analyze_video", "list_dir"}
SUMMARY_MAX_TOKENS = 1500   # bumped from 180 - that was too tight even as a floor, let alone for wordy sources like vision descriptions
SUMMARY_SKIP_CHARS = 220           # outputs already this short aren't worth a model round-trip; used as-is
SUMMARY_INPUT_CHARS = 20000        # how much of a huge raw output we actually feed to the summarizer call
WORKFLOWS_DIR = "workflows"   # folder inside LocalStorage holding the industry's SOP/workflow markdown files

SUMMARIZER_SYSTEM_PROMPT = """You compress a tool's raw output into a terse note for another AI agent's memory.
Not for humans. No grammar, no full sentences, no filler words (a/an/the/is/was). Fragments and keywords only, comma or semicolon separated.
Keep every concrete fact the agent would need later: numbers, names, paths, headings, counts, key values, structure.
Drop formatting, boilerplate, and anything irrelevant to WHY the tool was called (given to you as TASK REASON).
Output ONLY the compressed note, nothing else. Aim for the shortest note that loses no fact relevant to the task reason."""

SUMMARY_GRAMMAR = r'''
root      ::= "{" ws "\"summary\"" ws ":" ws string ws "}"
string    ::= "\"" ( [^"\\\x00-\x1f] | "\\" ["\\/bfnrtu] )* "\""
ws        ::= [ \t\n]*
'''

qwen = None
vision_enabled = False
vision_handler = None
model_lock = threading.Lock()
sessions = {}                                           # session_id -> Session

ACTION_NAMES = ["list_dir", "read_file", "write_file", "append_file", "delete_path", "move_path", "make_dir", "search_files", "read_document",
                "read_metadata", "write_metadata", "run_python", "run_shell", "pip_install", "analyze_image", "analyze_video", "query_sql",
                "network_request", "ask_user", "final_answer"]
ACTION_GRAMMAR = r'''
root      ::= "[" ws task ( ws "," ws task )* ws "]"
task      ::= "{" ws "\"action\"" ws ":" ws action ws "," ws "\"message\"" ws ":" ws string ( ws "," ws field )* ws "}"
action    ::= ''' + " | ".join('"\\"' + a + '\\""' for a in ACTION_NAMES) + r'''
field     ::= key ws ":" ws value
key       ::= "\"" [a-z_]+ "\""
value     ::= string | number | "true" | "false" | "null" | array | object
object    ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
member    ::= string ws ":" ws value
array     ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
number    ::= "-"? [0-9]+
string    ::= "\"" ( [^"\\\x00-\x1f] | "\\" ["\\/bfnrtu] )* "\""
ws        ::= [ \t\n]*
'''

NAMING_GRAMMAR = r'''
root      ::= "{" ws "\"name\"" ws ":" ws string ws "}"
string    ::= "\"" ( [^"\\\x00-\x1f] | "\\" ["\\/bfnrtu] )* "\""
ws        ::= [ \t\n]*
'''
SYSTEM_PROMPT = """You are AgenticAI, a careful, autonomous, long-horizon developer agent. You work ONLY inside a private workspace folder called LocalStorage.
Everything you do is shown live to the user, so every task needs a clear "message" telling the user what you are doing and why.

INDUSTRIAL CONTEXT: You are deployed inside an industrial organization (refinery / PSU / manufacturing unit). The data here is confidential and never leaves this machine. Users are engineers and officers who need real work products (approval notes, calculations, reports, presentations, spreadsheets, code), and mistakes can affect safety, compliance and money. Be precise, show your calculation steps, state units, and never invent values, limits or procedures.
The organization's own procedures are stored as markdown files in the "workflows" folder inside LocalStorage (for example, how to format an approval note, how to calculate remaining life, how an inspection is reviewed). These are the ground truth for how work is done HERE:
- For any task that produces or reviews a work product (note, report, calculation, inspection review, presentation, etc.), FIRST call list_dir on "workflows" in your first batch, and read_file every workflow file that matches the task. Follow its steps, structure, limits and formatting exactly.
- If workflow files conflict with your general knowledge, the workflow files win. If no workflow file covers the task, say so in your message, use a sensible standard approach, and tell the user which assumptions you made.
- Never edit or delete files inside "workflows" unless the user explicitly asks. Read-only by default.
- Cite which workflow file (and clause, if it has numbered clauses) you followed when you give the final answer.
- Greetings and simple questions do not need this: answer them immediately with final_answer.

OUTPUT FORMAT (strict): reply with ONE JSON array of task objects and nothing else:
[{"action": "...", "message": "...", ...fields}, {"action": "...", "message": "...", ...fields}]
Emit SEVERAL tasks in one array whenever they are independent (for example list several folders, or read several files at once). This gets you oriented fast.
Read-only tasks in the array run in parallel; tasks that change things run one after another in the order you wrote them.
Use "\\n" for newlines inside JSON strings. After every batch you receive the results and decide the next batch.

ACTIONS (the field names after "message"):
 list_dir        path                      list a folder (use "." for the workspace root)
 read_file       path                      read a plain text file (long files are truncated; use start/end line numbers as "start","end" to page)
 read_document   path, [pages]             read pptx/docx/xlsx/csv/pdf/zip via the interpreters. pages like "3-7" for slides/pages. Spreadsheets/CSVs become SQLite and you get the schema.
 search_files    pattern, [query]          find files by name glob (pattern, e.g. "*.pdf") and optionally text inside them (query)
 read_metadata                             read the workspace metadata index (what is inside every file)
 write_metadata  content                   write the FULL metadata index JSON: {"relative/path": {"summary": "...", "type": "...", "keywords": ["..."]}, ...}
 write_file      path, content             create/overwrite a text file
 append_file     path, content             append text to a file
 make_dir        path                      create a folder
 move_path       path, destination         move/rename
 delete_path     path                      delete a file or folder
 run_python      code                      run Python 3 in the sandbox (cwd is LocalStorage). Print what you want to see. Good for analysis, plotting, converting, generating files.
 run_shell       command                   run a shell command in the sandbox
 pip_install     packages (array)          install Python packages (needs user approval, internet)
 query_sql       db, query                 run a read-only SELECT on a SQLite database an interpreter created (db = database file name, e.g. "sales.sqlite")
 analyze_image   path, [question]          look at an image with the vision model (ALWAYS supply "path", e.g. "path": "name.png")
 analyze_video   path, [question], [frames]  look at frames sampled evenly from a video with the vision model (ALWAYS supply "path")
 network_request method, url, [headers], [body]   internet request (needs user approval; never put workspace data in it)
 ask_user        question                  ask the user something and wait for the answer
 final_answer    text (or message)         finish. Provide your answer in "text" or "message" to the user, using Markdown and LaTeX ($...$ inline, $$...$$ block). Emit final_answer ALONE in its array.

WORKSPACE RULES:
- All paths are relative to LocalStorage. Never use absolute paths or "..".
- When calling analyze_image, analyze_video, read_file, or read_document, ALWAYS explicitly include the "path" field with the exact file name (e.g. {"action": "analyze_image", "message": "...", "path": "photo.png"}). Never omit the "path" field.
- CONVERSATIONAL MESSAGES: If the user sends a greeting (e.g. "hello", "hi", "how are you") or a simple question that does not require inspecting files, reply IMMEDIATELY with final_answer. Do NOT explore the workspace, read files, or call analyze_image for such messages.
- For actual task requests (analyze files, write code, find info, etc.), start with a batch: list_dir "." plus read_metadata plus list_dir "workflows" (and list a few promising folders). Then read_file the matching workflow file(s) before doing the work.
- The workspace can hold thousands of files. NEVER read everything. The metadata index is NOT shown to you automatically - call read_metadata yourself when you want to know what a file contains before reading it, so you read only the few files that matter. The [WORKSPACE] line each turn only tells you whether an index exists, not what's in it.
- If read_metadata reports the index is missing or has gaps, ask the user with ask_user for permission to build it, and only then create it: read files through read_document/read_file, summarise each in 1-2 sentences, and write_metadata with the full index (keep existing entries).
- The output of read_file/read_document/read_metadata/search_files/query_sql/analyze_image/analyze_video/list_dir that you see in TOOL RESULTS is a compressed note, not the raw content - it is written in fragments to save space, not full prose. Trust the facts in it. If it lacks a specific detail you now need, re-read the same file with a narrower request (e.g. a line range or a specific page) rather than assuming the detail doesn't exist.
- Never read ppt/docx/xlsx/pdf/zip/images with read_file. Use read_document (or analyze_image/analyze_video) instead.
- If a pdf/document reports pages that need vision, call analyze_image on the rendered image paths it lists ONLY IF the user's task actually requires understanding those images.
- Do not guess file contents. Read first. If a result looks wrong or truncated, say so and try a better approach.
- Internet is OFF. Only pip_install and network_request can reach it, and the user must approve each. Never send file contents, file names or any workspace data through the network.
- Be efficient: batch independent tasks, avoid repeating a read you already have, and finish with final_answer once the question is truly answered, If the user says something simple which doesn't need any extra knowledge, then directly give out the final_answer without running a tool.
- final_answer must NEVER be combined with any other action in the same array. If you emit final_answer, it must be the ONLY object in the list.
- If the user sends a NUDGE while you work, treat it as an important correction or extra instruction and adapt immediately."""

SIMPLE_STRING_RULE = r'''string    ::= "\"" ( [^"\\] | "\\" ["\\/bfnrtu] )* "\""'''


_GRAMMAR_CACHE: dict[str, LlamaGrammar] = {}


def compact_context_history(context: list) -> list:
    """
    Tool outputs are now compressed to short task-aware notes AT THE SOURCE (see summarize_output /
    run_task), so past turns no longer contain raw dumps that need reactive shrinking here. The one thing
    still worth stripping from OLD turns is the per-turn [WORKSPACE] / [CAPABILITIES] reminder line - it's
    only useful on the turn it was written for, and it repeats verbatim every step. trim_context() below
    remains the real safety net for anything that still doesn't fit.
    """
    if len(context) <= 3:
        return context

    tool_msg_indices = [
        i for i, m in enumerate(context)
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].startswith("TOOL RESULTS:")
    ]
    if len(tool_msg_indices) <= 1:
        return context

    for idx in tool_msg_indices[:-1]:
        msg_text = context[idx]["content"]
        if "\n\n[WORKSPACE]" in msg_text:
            msg_text = msg_text.split("\n\n[WORKSPACE]")[0]
        context[idx]["content"] = msg_text.strip()

    return context


def load_grammar(text: str) -> LlamaGrammar:
    # PERF: LlamaGrammar.from_string() parses and builds a pushdown automaton from the grammar text - not free,
    # and this was being redone on EVERY single ask_model()/create_name() call even though ACTION_GRAMMAR and
    # NAMING_GRAMMAR never change after startup. Compile once per distinct grammar text, then reuse the object.
    cached = _GRAMMAR_CACHE.get(text)
    if cached is not None:
        return cached
    # Strict grammar first; if this llama.cpp build rejects the control-character range, fall back to the simpler string rule.
    try:
        grammar = LlamaGrammar.from_string(text, verbose=False)
    except Exception:
        relaxed = re.sub(r'^string\s+::=.*$', SIMPLE_STRING_RULE, text, flags=re.MULTILINE)
        grammar = LlamaGrammar.from_string(relaxed, verbose=False)
    _GRAMMAR_CACHE[text] = grammar
    return grammar


def load_model(context_window: int = CONTEXT_WINDOW):
    # Loads Qwen once. Attaches the vision projector if present. If memory runs out the context window is halved until it fits.
    global qwen, vision_enabled, vision_handler
    if qwen is not None:
        return
    handler, vision_enabled = None, False
    if MMPROJ_PATH.is_file():
        try:
            from llama_cpp.llama_chat_format import MTMDChatHandler
            handler = MTMDChatHandler(clip_model_path=str(MMPROJ_PATH), verbose=False)
        except Exception:
            handler = None
    n_ctx = context_window
    while n_ctx >= MIN_CONTEXT_WINDOW:
        try:
            try:
                qwen = Llama(model_path=str(MODEL_PATH), n_ctx=n_ctx, n_gpu_layers=-1, flash_attn=True, n_batch=1024, n_ubatch=512, n_threads=os.cpu_count() or 8, chat_handler=None, verbose=False)
            except Exception:
                qwen = Llama(model_path=str(MODEL_PATH), n_ctx=n_ctx, n_gpu_layers=-1, chat_handler=None, verbose=False)
            if handler is not None:
                try:
                    handler._init_mtmd_context(qwen)
                    vision_enabled = True
                    vision_handler = handler
                except Exception as ve:
                    print(f"[backend] Warning: vision projector '{MMPROJ_PATH.name}' is incompatible with model '{MODEL_PATH.name}': {ve}", file=sys.stderr)
                    vision_enabled = False
                    vision_handler = None
            else:
                vision_enabled = False
                vision_handler = None
            return
        except Exception as e:
            if n_ctx // 2 < MIN_CONTEXT_WINDOW:
                raise RuntimeError(f"could not load the model: {e}")
            n_ctx //= 2


def context_size() -> int:
    return qwen.n_ctx() if qwen is not None else CONTEXT_WINDOW


def _msg_tokens(content) -> int:
    # Token count of a SINGLE message's content. Kept cheap and separate so trim_context (below) never has to
    # retokenize the whole conversation just to check one message's contribution.
    text = content if isinstance(content, str) else json.dumps(content)[:2000]
    try:
        return len(qwen.tokenize(text.encode("utf-8"), add_bos=False))
    except Exception:
        return len(text) // 3


def count_tokens(messages: list) -> int:
    # Sum of per-message counts. Still available for callers that want a one-shot total, but trim_context no longer
    # calls this inside a loop - see the note there for why that was the main slowdown on every single turn.
    return sum(_msg_tokens(m["content"]) for m in messages)


def trim_context(context: list) -> list:
    limit = int(context_size() * CONTEXT_SAFETY_RATIO) - MAX_OUTPUT_TOKENS
    counts = [_msg_tokens(m["content"]) for m in context]
    total = sum(counts)
    if total <= limit:
        return context

    # 1. Truncate long tool result strings
    for i in range(2, len(context) - 4):
        if total <= limit:
            break
        msg = context[i]
        if msg["role"] == "user" and isinstance(msg["content"], str) and msg["content"].startswith("TOOL RESULTS") and len(msg["content"]) > 400:
            new_content = msg["content"][:300] + "\n...[older result removed to save memory]"
            context[i] = {"role": "user", "content": new_content}
            new_count = _msg_tokens(new_content)
            total += new_count - counts[i]
            counts[i] = new_count

    # 2. Hard drop oldest user/assistant turns if still exceeding context window
    while total > limit and len(context) > 4:
        del context[2]
        counts.pop(2)
        total = sum(counts)

    return context


def fallback_name(user_input: str) -> str:
    return re.sub(r"\s+", " ", user_input).strip()[:40] or "New session"


def create_name(user_input: str) -> str:
    # A separate tiny call so the naming prompt never pollutes the agent's own context.
    # NOTE: this still does a real model round-trip and is deliberately NOT called on the hot path anymore -
    # see rename_session_async below. Kept as a plain sync function so it's easy to call from a background thread.
    fallback = fallback_name(user_input)
    try:
        with model_lock:
            out = qwen.create_chat_completion(
                messages=[{"role": "system", "content": 'Reply with ONE JSON object: {"name": "<short, specific title of 2-5 words for this chat>"}'},
                          {"role": "user", "content": user_input[:1500]}],
                grammar=load_grammar(NAMING_GRAMMAR), temperature=0.1, max_tokens=60)
        name = json.loads(out["choices"][0]["message"]["content"])["name"].strip()
        return name[:60] or fallback
    except Exception:
        return fallback


def rename_session_async(sess, user_input: str) -> None:
    # PERF: create_name() used to be called INLINE, synchronously, before run_agent() started - meaning every
    # new chat paid a full extra LLM round-trip (its own grammar load, its own prefill/decode) as pure added
    # latency before the agent even looked at the user's message. The naming call doesn't block anything the
    # user can see except the session's *label*, so it now runs in a background thread: the session starts
    # immediately with the cheap fallback_name() title, and a "session" rename event follows once the real
    # name is ready (usually well before the agent's own first reply, but never blocking it).
    def worker():
        name = create_name(user_input)
        if sess.id in sessions:
            sess.name = name
            emit(sess, "session", id=sess.id, name=sess.name, autopilot=sess.autopilot)
            save_session(sess)
    threading.Thread(target=worker, daemon=True).start()


def history_path(session_id: str) -> Path:
    return HISTORY / f"{session_id}.json"


def save_session(sess) -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    data = {"id": sess.id, "name": sess.name, "created": sess.created, "updated": datetime.now().isoformat(timespec="seconds"),
            "autopilot": sess.autopilot, "context": sess.context, "events": sess.events, "network_audit": sess.network_audit}
    tmp = history_path(sess.id).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(history_path(sess.id))          # atomic write - a crash can never leave half a history file


def list_sessions() -> list:
    HISTORY.mkdir(parents=True, exist_ok=True)
    out = []
    for f in HISTORY.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append({"id": d["id"], "name": d["name"], "created": d["created"], "updated": d.get("updated", d["created"]), "messages": len([e for e in d.get("events", []) if e.get("type") in ("user", "final")])})
        except Exception:
            continue
    return sorted(out, key=lambda s: s["updated"], reverse=True)


def load_session_data(session_id: str) -> dict | None:
    f = history_path(session_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id) or not f.is_file():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def delete_session(session_id: str) -> bool:
    f = history_path(session_id)
    if re.fullmatch(r"[A-Za-z0-9_-]+", session_id) and f.is_file():
        f.unlink()
        sessions.pop(session_id, None)
        return True
    return False


class Session:
    # One running conversation. It holds the state that changes over time (context, events, flags) so the functions below stay flat and simple.
    def __init__(self, session_id: str, name: str, context: list, autopilot: bool = False, created: str = None, events: list = None, network_audit: list = None):
        self.id, self.name, self.context = session_id, name, context
        self.autopilot = autopilot
        self.created = created or datetime.now().isoformat(timespec="seconds")
        self.events = events or []
        self.network_audit = network_audit or []
        self.emit = lambda event: None
        self.replies = {}                       # request_id -> answer
        self.reply_cv = threading.Condition()
        self.nudges = []                        # text the user typed during a nudge window
        self.cancelled = threading.Event()
        self.running = False
        self.lock = threading.Lock()


def emit(sess: Session, event_type: str, **data) -> dict:
    event = {"type": event_type, "time": datetime.now().isoformat(timespec="seconds"), **data}
    if event_type not in ("thinking", "countdown", "tick"):
        sess.events.append(event)
    try:
        sess.emit(event)
    except Exception:
        pass
    return event


def submit_reply(session_id: str, request_id: str, answer) -> bool:
    sess = sessions.get(session_id)
    if sess is None:
        return False
    with sess.reply_cv:
        sess.replies[request_id] = answer
        sess.reply_cv.notify_all()
    return True


def submit_nudge(session_id: str, text: str) -> bool:
    sess = sessions.get(session_id)
    if sess is None or not text.strip():
        return False
    with sess.reply_cv:
        sess.nudges.append(text.strip())
        sess.reply_cv.notify_all()
    return True


def cancel_session(session_id: str) -> None:
    sess = sessions.get(session_id)
    if sess:
        sess.cancelled.set()
        with sess.reply_cv:
            sess.reply_cv.notify_all()


def set_autopilot(session_id: str, enabled: bool) -> bool:
    sess = sessions.get(session_id)
    if sess is None:
        return False
    sess.autopilot = enabled
    emit(sess, "autopilot", enabled=enabled)
    return True


def ask_user(sess: Session, kind: str, question: str, detail: str = "", options=("yes", "no")):
    # Blocks until the frontend answers (or the safety timeout passes -> treated as "no"). Used for permissions AND ask_user tasks.
    request_id = uuid.uuid4().hex[:10]
    emit(sess, "ask", request_id=request_id, kind=kind, question=question, detail=detail, options=list(options))
    deadline = time.time() + PERMISSION_TIMEOUT_SECONDS
    with sess.reply_cv:
        while request_id not in sess.replies and not sess.cancelled.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            sess.reply_cv.wait(timeout=min(remaining, 1.0))
        return sess.replies.pop(request_id, None)


def needs_permission(sess: Session, kind: str, question: str, detail: str = "") -> bool:
    answer = ask_user(sess, kind, question, detail)
    return isinstance(answer, str) and answer.strip().lower() in ("yes", "y", "allow", "approve", "true")


def nudge_window(sess: Session) -> list:
    # Gives the user NUDGE_WINDOW_SECONDS to type a correction. Any typing extends nothing; sending a message ends the wait right away.
    if sess.autopilot or sess.cancelled.is_set():
        return []
    emit(sess, "countdown", seconds=NUDGE_WINDOW_SECONDS)
    deadline = time.time() + NUDGE_WINDOW_SECONDS
    with sess.reply_cv:
        while not sess.nudges and not sess.cancelled.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            sess.reply_cv.wait(timeout=min(remaining, 0.25))
        taken, sess.nudges = sess.nudges[:], []
    if taken:
        emit(sess, "nudge", text="\n".join(taken))
    return taken


def _safe(rel: str) -> Path:
    # Every path the LLM gives is resolved against LocalStorage.
    if not rel:
        rel = "."

    # Clean string, quotes, and backslashes
    rel_str = str(rel).strip().strip("'\"`").replace("\\", "/")

    # Strip common LLM prefixes
    if rel_str.startswith("LocalStorage/"):
        rel_str = rel_str[len("LocalStorage/"):]
    elif rel_str.startswith("./"):
        rel_str = rel_str[2:]

    root = STORAGE.resolve()

    # If the LLM sent a full absolute path, check if it's actually inside LocalStorage
    if re.match(r"^([a-zA-Z]:|/|~)", rel_str):
        p = Path(rel_str).expanduser().resolve()
    else:
        p = (root / rel_str).resolve()

    # Security check: ensure path does not escape LocalStorage
    if p != root and root not in p.parents:
        raise PermissionError(f"path escapes LocalStorage: {rel}")

    return p


def _rel(p: Path) -> str:
    try:
        r = p.resolve().relative_to(STORAGE.resolve()).as_posix()
        return r or "."
    except ValueError:
        return p.name

NOISE_DIRS = {"Caches", "__pycache__", "node_modules", ".git", ".venv", "venv", "site-packages", ".Trash", "Trash", ".cache"}

def _visible_files() -> list:
    files = []
    root = STORAGE.resolve()
    for f in root.rglob("*"):
        try:
            # Ignore hidden files like .DS_Store
            if f.name.startswith("."):
                continue
            parts = f.relative_to(root).parts
            if f.is_file() and not any(part in INTERNAL_DIRS or part in NOISE_DIRS or part.startswith(".") for part in parts) and f.name != METADATA_FILE:
                files.append(f)
        except Exception:
            continue
    return files



def do_list_dir(t: dict) -> str:
    p = _safe(t.get("path", "."))
    if not p.is_dir():
        return f"ERROR: not a directory: {t.get('path')}"
    shown_path = _rel(p)
    entries = [e for e in sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower())) if e.name not in INTERNAL_DIRS]
    if not entries:
        return f"Contents of \"{shown_path}\": (empty directory)"
    # Every listed name is DIRECTLY inside `shown_path` - nothing here is nested deeper. Each line also spells
    # out the exact path to use with read_file/read_document/etc, so the model never has to reconstruct or
    # guess a path from a bare file name plus a folder name it saw elsewhere in the same listing (that
    # ambiguity is exactly what caused "ERROR: file not found" on files that were actually one level up).
    lines = [f"Contents of \"{shown_path}\" (all items below are directly inside this folder, not nested further):"]
    for e in entries[:500]:
        full = f"{shown_path}/{e.name}" if shown_path != "." else e.name
        if e.is_dir():
            lines.append(f'  [dir]  "{full}"')
        else:
            lines.append(f'        "{full}"  ({e.stat().st_size} bytes)')
    if len(entries) > 500:
        lines.append(f"  ...[{len(entries) - 500} more entries not shown; use search_files]")
    return "\n".join(lines)


def do_read_file(t: dict) -> str:
    # 1. Try standard keys
    raw_path = t.get("path") or t.get("file") or t.get("filepath") or t.get("target") or t.get("filename")

    # 2. FALLBACK: If path key is missing, extract backticked filename or standard filename from 'message'
    if not raw_path and t.get("message"):
        msg = t["message"]
        # Look for words ending in file extensions inside the message string (e.g., `README.md` or README.md)
        match = re.search(r'`?([a-zA-Z0-9_\-\/\.]+\.(md|py|dart|yaml|txt|json|html|css|js|csv))`?', msg)
        if match:
            raw_path = match.group(1)

    # 3. If still missing, return error showing available files
    if not raw_path or str(raw_path).strip() in ("", "None", "."):
        available = [_rel(f) for f in _visible_files()[:10]]
        avail_str = ", ".join(available) if available else "none"
        return f"ERROR: read_file requires a 'path' parameter (e.g. {{\"action\": \"read_file\", \"message\": \"...\", \"path\": \"README.md\"}}). Available files: [{avail_str}]"

    # 4. Execute read
    try:
        p = _safe(raw_path)
    except Exception as e:
        return f"ERROR: Invalid path '{raw_path}': {e}"

    if not p.is_file():
        available = [_rel(f) for f in _visible_files()[:10]]
        avail_str = ", ".join(available) if available else "none"
        return f"ERROR: file '{raw_path}' does not exist inside LocalStorage. Available files: [{avail_str}]"

    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return f"ERROR reading file '{raw_path}': {e}"

    start, end = max(int(t.get("start", 1) or 1), 1), int(t.get("end", 0) or 0) or len(lines)
    return "\n".join(lines[start - 1:end])

def do_write_file(t: dict) -> str:
    p = _safe(t.get("path"))
    if p.name == METADATA_FILE:
        return "ERROR: use write_metadata for the metadata index"
    p.parent.mkdir(parents=True, exist_ok=True)
    content = str(t.get("content", ""))
    p.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {_rel(p)}"


def do_append_file(t: dict) -> str:
    p = _safe(t.get("path"))
    p.parent.mkdir(parents=True, exist_ok=True)
    content = str(t.get("content", ""))
    with open(p, "a", encoding="utf-8") as f:
        f.write(content)
    return f"OK: appended {len(content)} chars to {_rel(p)}"


def do_make_dir(t: dict) -> str:
    p = _safe(t.get("path"))
    p.mkdir(parents=True, exist_ok=True)
    return f"OK: directory {_rel(p)} ready"


def do_move_path(t: dict) -> str:
    src, dst = _safe(t.get("path")), _safe(t.get("destination"))
    if not src.exists():
        return f"ERROR: not found: {t.get('path')}"
    if src == STORAGE.resolve():
        return "ERROR: cannot move the workspace root"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"OK: moved {_rel(src) if src.exists() else t.get('path')} -> {_rel(dst)}"


def do_delete_path(t: dict) -> str:
    p = _safe(t.get("path"))
    if p == STORAGE.resolve():
        return "ERROR: cannot delete the workspace root"
    if not p.exists():
        return f"ERROR: not found: {t.get('path')}"
    shutil.rmtree(p) if p.is_dir() else p.unlink()
    return f"OK: deleted {t.get('path')}"


def do_search_files(t: dict) -> str:
    pattern, query = str(t.get("pattern", "*") or "*"), str(t.get("query", "") or "")
    hits = []
    for f in _visible_files():
        rel = _rel(f)
        if not Path(rel).match(pattern) and not f.match(pattern):
            continue
        if not query:
            hits.append(rel)
            continue
        if Interpreter.interpreter_for(f) or f.stat().st_size > 5_000_000:
            continue
        try:
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if query.lower() in line.lower():
                    hits.append(f"{rel}:{n}: {line.strip()[:160]}")
                    if len(hits) >= 200:
                        break
        except Exception:
            continue
        if len(hits) >= 200:
            break
    return "\n".join(hits[:200]) if hits else "(no matches)"


def load_metadata() -> dict:
    f = STORAGE / METADATA_FILE
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metadata_gaps(files: list = None, meta: dict = None) -> list:
    # files that have no metadata entry, or changed after their entry was written.
    # Accepts an already-scanned file list / already-loaded metadata so callers that already have them
    # (see workspace_notes) don't trigger a second rglob() + stat() pass over LocalStorage.
    meta = load_metadata() if meta is None else meta
    gaps = []
    for f in (_visible_files() if files is None else files):
        rel = _rel(f)
        entry = meta.get(rel)
        if not entry:
            gaps.append(rel)
        elif isinstance(entry, dict) and entry.get("mtime") and f.stat().st_mtime > float(entry["mtime"]) + 1:
            gaps.append(rel)
    return sorted(gaps)


def do_read_metadata(t: dict) -> str:
    meta = load_metadata()
    gaps = metadata_gaps()
    if not meta:
        return f"NO METADATA INDEX EXISTS YET. {len(gaps)} file(s) are not indexed." + (" Ask the user for permission before building it." if gaps else "")
    text = json.dumps(meta, indent=1, ensure_ascii=False)
    if len(text) > READ_LIMIT_CHARS * 2:
        text = text[:READ_LIMIT_CHARS * 2] + "\n...[truncated]"
    return text + (f"\n\nMISSING/STALE: {len(gaps)} file(s) not covered: {gaps[:50]}" if gaps else "\n\nThe index covers every file.")


def do_write_metadata(t: dict) -> str:
    raw = t.get("content")
    if not raw:
        return "ERROR: write_metadata requires a 'content' object parameter containing metadata entries (e.g. {\"action\": \"write_metadata\", \"message\": \"...\", \"content\": {\"file.txt\": {\"summary\": \"...\"}}})."

    try:
        incoming = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(incoming, dict) or not incoming:
            raise ValueError("must be a non-empty JSON object mapping path -> entry")
    except Exception as e:
        return f"ERROR: invalid metadata JSON in 'content': {e}"

    merged = load_metadata()
    existing_files = {_rel(f): f for f in _visible_files()}
    added = 0
    for rel, entry in incoming.items():
        rel = str(rel).replace("\\", "/")
        if rel not in existing_files:
            continue
        entry = entry if isinstance(entry, dict) else {"summary": str(entry)}
        f = existing_files[rel]
        entry.update({"size": f.stat().st_size, "mtime": f.stat().st_mtime,
                      "type": entry.get("type") or f.suffix.lstrip(".") or "file"})
        merged[rel] = entry
        added += 1

    merged = {k: v for k, v in merged.items() if k in existing_files}
    (STORAGE / METADATA_FILE).write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
    return f"OK: metadata index now covers {len(merged)} file(s) ({added} written). Remaining gaps: {len(metadata_gaps())}"


def parse_page_range(spec) -> tuple | None:
    if not spec:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", str(spec))
    return (int(m.group(1)), int(m.group(2) or m.group(1))) if m else None


def format_interpreted(r: dict) -> str:
    if not r["ok"]:
        return f"ERROR: {r['error']}"
    parts = [f"SUMMARY: {r['summary']}", r["text"][:READ_LIMIT_CHARS] + (f"\n...[truncated, {len(r['text'])} chars total]" if len(r["text"]) > READ_LIMIT_CHARS else "")]
    for i, tbl in enumerate(r["tables"][:20], 1):
        rows = tbl["rows"][:30]
        parts.append(f"TABLE {i} ({tbl['location']}):\n" + "\n".join(" | ".join(row) for row in rows) + (f"\n...[{len(tbl['rows']) - 30} more rows]" if len(tbl["rows"]) > 30 else ""))
    if r["math"]:
        parts.append("EQUATIONS (LaTeX):\n" + "\n".join(f"  [{m['location']}] {m['latex']}" for m in r["math"][:60]))
    if r["images"]:
        parts.append("IMAGES (use analyze_image with the given path):\n" + "\n".join(f"  {i['path']}  [{i['location']}]" for i in r["images"][:40] if "data_uri" not in i or i.get("path")))
    return "\n\n".join(p for p in parts if p)


def do_read_document(t: dict) -> str:
    p = _safe(t.get("path"))
    if not p.is_file():
        # Same failure mode read_file already guards against: the model guessed a path (often nesting the
        # file under a folder it saw nearby in a list_dir result) that doesn't exist. Searching by bare file
        # name across the whole workspace usually finds the real location in one step instead of the model
        # retrying the same wrong path or giving up.
        matches = [_rel(f) for f in _visible_files() if f.name == p.name]
        if matches:
            return f"ERROR: no file at '{t.get('path')}'. A file with that exact name exists at: {matches}. Use that path instead."
        available = [_rel(f) for f in _visible_files()[:10]]
        return f"ERROR: file '{t.get('path')}' does not exist inside LocalStorage. Available files: [{', '.join(available) if available else 'none'}]"
    options = {}
    rng = parse_page_range(t.get("pages"))
    if rng:
        options["slide_range" if p.suffix.lower() == ".pptx" else "page_range"] = rng
    return format_interpreted(Interpreter.initialize(p, **options))


def do_query_sql(t: dict) -> str:
    db = str(t.get("db", ""))
    matches = list((STORAGE / ".interpreted" / "sql").glob(Path(db).name)) if db else []
    if not matches:
        return "ERROR: database not found. Call read_document on the spreadsheet/CSV first, then use the db file name it reports."
    return run_select(matches[0], str(t.get("query", "")))


def vision_ask(images: list, question: str) -> str:
    if not vision_enabled or vision_handler is None:
        return "ERROR: vision is not available. Place the mmproj file at LLM/main_vision.gguf and restart. See the README."
    content = [{"type": "image_url", "image_url": {"url": uri}} for uri in images] + [{"type": "text", "text": question or "Describe this in detail, including any text, numbers, charts and tables you can see."}]
    try:
        with model_lock:
            out = vision_handler(llama=qwen, messages=[{"role": "system", "content": "You are a precise visual analyst. Describe only what you can actually see."},
                                                       {"role": "user", "content": content}], temperature=0.1, max_tokens=1500)
        return out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: vision model failed: {e}"


def _resolve_image_file(target) -> Path | None:
    if isinstance(target, Path):
        return target if target.is_file() else None
    if not target or not str(target).strip():
        return None
    raw = str(target).strip().strip("'\"`")
    if raw.startswith("LocalStorage/"):
        raw = raw[len("LocalStorage/"):]
    elif raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        return None
    if Path(raw).is_absolute():
        cand = Path(raw).resolve()
        if (cand == STORAGE.resolve() or STORAGE.resolve() in cand.parents) and cand.is_file():
            return cand
    try:
        cand = _safe(raw)
        if cand.is_file():
            return cand
    except Exception:
        pass
    cand = (STORAGE / raw).resolve()
    if (cand == STORAGE.resolve() or STORAGE.resolve() in cand.parents) and cand.is_file():
        return cand
    return None


def _extract_image_path(t: dict) -> Path | None:
    for key in ("path", "image", "image_path", "file", "file_path", "filename", "name", "target", "img"):
        val = t.get(key)
        if val:
            resolved = _resolve_image_file(val)
            if resolved:
                return resolved

    message = str(t.get("message", ""))
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
    existing_images = [
        f for f in STORAGE.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        and not any(part in NOISE_DIRS or (part.startswith(".") and part != ".interpreted") for part in f.parts[:-1])
    ]

    for img in existing_images:
        rel = _rel(img)
        if img.name in message or rel in message:
            return img

    msg_lower = message.lower()
    for img in existing_images:
        rel = _rel(img).lower()
        if img.name.lower() in msg_lower or rel in msg_lower:
            return img

    quoted_matches = re.findall(r"['\"`]?([a-zA-Z0-9_\- .]+\.(?:png|jpe?g|webp|bmp|gif|tiff))['\"`]?", message, re.IGNORECASE)
    for match in quoted_matches:
        resolved = _resolve_image_file(match)
        if resolved:
            return resolved

    if len(existing_images) == 1 and not t.get("path"):
        return existing_images[0]

    return None


def _image_uri_from_path(path_text) -> str:
    resolved = _resolve_image_file(path_text)
    if not resolved:
        raise PermissionError(f"image not found inside LocalStorage: {path_text}")
    return prepare_image(resolved)["data_uri"]


def do_analyze_image(t: dict) -> str:
    img_path = _extract_image_path(t)
    if not img_path:
        IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
        existing_images = [
            f.name for f in STORAGE.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            and not any(part in NOISE_DIRS or (part.startswith(".") and part != ".interpreted") for part in f.parts[:-1])
        ]
        raw_val = t.get("path") or t.get("image") or t.get("file")
        if raw_val:
            return f"ERROR: image '{raw_val}' not found inside LocalStorage. Available images: {', '.join(existing_images) if existing_images else 'none'}."
        else:
            return f"ERROR: analyze_image requires a 'path' parameter specifying which image to analyze (e.g. 'path': '{existing_images[0] if existing_images else 'image.png'}'). Available images: {', '.join(existing_images) if existing_images else 'none'}."
    try:
        return vision_ask([prepare_image(img_path)["data_uri"]], t.get("question", ""))
    except Exception as e:
        return f"ERROR: {e}"


def do_analyze_video(t: dict) -> str:
    raw = t.get("path") or t.get("video") or t.get("file")
    if not raw:
        return "ERROR: analyze_video requires a 'path' parameter specifying the video file."
    try:
        p = _safe(str(raw).strip().strip("'\"`"))
    except Exception as e:
        return f"ERROR: invalid video path: {e}"
    if not p.is_file():
        return f"ERROR: video not found inside LocalStorage: {raw}"
    r = Interpreter.initialize(p, num_frames=int(t.get("frames", 8) or 8))
    if not r["ok"]:
        return f"ERROR: {r['error']}"
    stamps = ", ".join(i["location"] for i in r["images"])
    answer = vision_ask([i["data_uri"] for i in r["images"]], f"These are {len(r['images'])} frames sampled evenly from a video ({stamps}). {t.get('question', 'Describe what happens in the video.')}")
    return f"{r['summary']}\n\n{answer}"


def do_run_python(t: dict, sess: Session) -> str:
    return format_sandbox(sandbox.initialize("python", str(t.get("code", "")), timeout=int(t.get("timeout", 60) or 60)))


def do_run_shell(t: dict, sess: Session) -> str:
    return format_sandbox(sandbox.initialize("shell", str(t.get("command", "")), timeout=int(t.get("timeout", 60) or 60)))


def format_sandbox(r: dict) -> str:
    out = f"[sandbox: {r.get('mode', '?')} | exit code {r['exit_code']}]"
    if r["stdout"]:
        out += f"\nSTDOUT:\n{r['stdout']}"
    if r["stderr"]:
        out += f"\nSTDERR:\n{r['stderr']}"
    return out


def record_network(sess: Session, audit: dict) -> None:
    sess.network_audit.append(audit)
    emit(sess, "network", audit=audit, total=len(sess.network_audit), sent=sum(1 for a in sess.network_audit if a.get("sent")))


def do_pip_install(t: dict, sess: Session) -> str:
    packages = t.get("packages", [])
    packages = [packages] if isinstance(packages, str) else [str(p) for p in packages]
    if not packages:
        return "ERROR: no packages given"
    bad = [p for p in packages if not sandbox.SAFE_PACKAGE.match(p)]
    if bad:
        return f"ERROR: invalid package name(s) {bad}. Use plain names like 'numpy' or 'numpy==1.26.4'."
    audit = {"time": datetime.now().isoformat(timespec="seconds"), "method": "PIP", "url": "https://pypi.org (packages: " + ", ".join(packages) + ")", "outbound_bytes": 0,
             "outbound_sha256": sandbox.payload_fingerprint(" ".join(packages))["sha256"], "leak_scan": {"clean": True, "hits": [], "scanned_files": 0}}
    if not needs_permission(sess, "network", f"Install Python packages: {', '.join(packages)}?", "This connects to the internet (pypi.org). Only the package names are sent. No files from LocalStorage leave your computer."):
        audit.update({"sent": False, "note": "DENIED by user"})
        record_network(sess, audit)
        return "DENIED: the user refused the installation."
    r = sandbox.initialize("pip", packages)
    audit.update({"sent": True, "note": "package names only; nothing from LocalStorage was transmitted"})
    record_network(sess, audit)
    return format_sandbox(r)


def do_network_request(t: dict, sess: Session) -> str:
    method, url = str(t.get("method", "GET")).upper(), str(t.get("url", ""))
    if not re.match(r"^https?://", url):
        return "ERROR: url must start with http:// or https://"
    headers = t.get("headers") if isinstance(t.get("headers"), dict) else {}
    body = t.get("body") if isinstance(t.get("body"), str) else None
    outbound = f"{url}\n{json.dumps(headers)}\n{body or ''}"
    scan = sandbox.scan_for_leak(outbound)
    fp = sandbox.payload_fingerprint(outbound)
    audit = {"time": datetime.now().isoformat(timespec="seconds"), "method": method, "url": url, "outbound_sha256": fp["sha256"], "outbound_bytes": fp["bytes"], "leak_scan": scan}
    if not scan["clean"]:
        audit.update({"sent": False, "note": "BLOCKED before sending: payload contained LocalStorage data"})
        record_network(sess, audit)
        return "BLOCKED: the request contained data from LocalStorage and was NOT sent. Never put workspace data in network requests."
    detail = f"{method} {url}\nPayload: {fp['bytes']} bytes, SHA-256 {fp['sha256'][:16]}...\nLeak scan: clean ({scan['scanned_files']} workspace files checked)"
    if not needs_permission(sess, "network", f"Allow internet request to {url}?", detail):
        audit.update({"sent": False, "note": "DENIED by user"})
        record_network(sess, audit)
        return "DENIED: the user refused this network request."
    result = sandbox.initialize("network", {"method": method, "url": url, "headers": headers, "body": body})
    record_network(sess, result["audit"])
    return f"HTTP {result['audit'].get('status')}\n{result['body']}"


def do_ask_user(t: dict, sess: Session) -> str:
    if sess.autopilot:
        return "AUTOPILOT is on: the user is not available. Make the best reasonable assumption and continue."
    answer = ask_user(sess, "question", str(t.get("question", "")), options=[])
    return f"USER ANSWER: {answer}" if answer else "USER DID NOT ANSWER (timed out). Continue with the best assumption."


def confirm_if_needed(sess: Session, t: dict) -> str | None:
    action = t["action"]
    # Workflow files are the organization's procedures: never modified silently, not even in autopilot.
    if action in ("write_file", "append_file", "delete_path", "move_path"):
        try:
            targets = [_safe(t.get("path"))] + ([_safe(t.get("destination"))] if t.get("destination") else [])
        except Exception:
            targets = []
        wf_root = (STORAGE / WORKFLOWS_DIR).resolve()
        if any(p == wf_root or wf_root in p.parents for p in targets):
            if not needs_permission(sess, "permission",
                                    f"This changes the organization's workflow file: {t.get('path')}. Allow?",
                                    "Workflow files define how work is done here. Only approve if you intend to edit them."):
                return f"DENIED: the user refused to modify workflow file {t.get('path')}."
            return None
    if sess.autopilot:
        return None
    question = None
    if action in DESTRUCTIVE_ACTIONS:
        question, detail = f"Allow {action.replace('_', ' ')}: {t.get('path')}" + (f" -> {t.get('destination')}" if action == "move_path" else "") + "?", "This changes or removes files in LocalStorage."
    elif action == "write_file" and _safe(t.get("path")).exists():
        question, detail = f"Overwrite existing file {t.get('path')}?", "The current content will be replaced."
    elif action == "write_metadata":
        question, detail = "Write the metadata index for your files?", "A hidden file '.metadata.json' will be created/updated in LocalStorage so future searches are fast."
    elif action in ("run_python", "run_shell"):
        kind = "python" if action == "run_python" else "shell"
        code = t.get("code" if kind == "python" else "command", "")
        if sandbox.detect_network_use(kind, str(code)):
            question, detail = f"This {kind} code looks like it uses the internet. Run it with the network BLOCKED?", "It will run with internet disabled; if it needs the internet it will fail safely."
        else:
            return None
    if question and not needs_permission(sess, "permission", question, detail):
        return f"DENIED: the user refused this {action}."
    return None


EXECUTORS = {"list_dir": do_list_dir, "read_file": do_read_file, "write_file": do_write_file, "append_file": do_append_file, "make_dir": do_make_dir, "move_path": do_move_path,
             "delete_path": do_delete_path, "search_files": do_search_files, "read_document": do_read_document, "read_metadata": do_read_metadata, "write_metadata": do_write_metadata,
             "query_sql": do_query_sql, "analyze_image": do_analyze_image, "analyze_video": do_analyze_video}
SESSION_EXECUTORS = {"run_python": do_run_python, "run_shell": do_run_shell, "pip_install": do_pip_install, "network_request": do_network_request, "ask_user": do_ask_user}


def execute(sess: Session, task: dict) -> str:
    action = task["action"]
    try:
        denied = confirm_if_needed(sess, task)
        if denied:
            return denied
        if action in SESSION_EXECUTORS:
            return SESSION_EXECUTORS[action](task, sess)
        if action in EXECUTORS:
            return EXECUTORS[action](task)
        return f"ERROR: unknown action {action}"
    except PermissionError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def run_task(sess: Session, index: int, task: dict) -> tuple:
    emit(sess, "task", index=index, action=task["action"], message=task.get("message", ""), params={k: v for k, v in task.items() if k not in ("action", "message")})
    started = time.time()
    output = execute(sess, task)
    action = task["action"]
    # Bulky/exploratory reads get compressed into a short, task-aware note (see summarize_output). The note is
    # what the user sees live AND what lands in context - the raw output is never shown and never stored.
    # Sandbox runs (run_python/run_shell) and short "OK: ..." writes are untouched, exactly as-is.
    context_output = summarize_output(task.get("message", ""), action, output) if action in SUMMARIZE_ACTIONS else output
    emit(sess, "result", index=index, action=action, output=context_output[:READ_LIMIT_CHARS], seconds=round(time.time() - started, 2))
    return index, task, context_output


def run_batch(sess: Session, tasks: list) -> list:
    # Read-only tasks that sit next to each other run in parallel; anything that changes state runs alone, in order.
    results, i = [], 0
    while i < len(tasks) and not sess.cancelled.is_set():
        if tasks[i]["action"] in PARALLEL_ACTIONS:
            j = i
            while j < len(tasks) and tasks[j]["action"] in PARALLEL_ACTIONS:
                j += 1
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, j - i)) as pool:
                futures = [pool.submit(run_task, sess, k, tasks[k]) for k in range(i, j)]
                results.extend(f.result() for f in futures)
            i = j
        else:
            results.append(run_task(sess, i, tasks[i]))
            i += 1
    return results


MAX_GAP_PATHS_SHOWN = 10   # kept only for do_read_metadata's own gap listing, not for the per-turn note anymore


def workflow_files() -> list:
    d = STORAGE / WORKFLOWS_DIR
    if not d.is_dir():
        return []
    return sorted(f.name for f in d.iterdir() if f.is_file() and not f.name.startswith("."))


def workspace_notes() -> str:
    meta = load_metadata()
    if meta:
        note = f"[WORKSPACE] metadata index present ({len(meta)} entries, may not cover every file). Call read_metadata if you need it - do not assume it's complete or incomplete."
    else:
        note = "[WORKSPACE] no metadata index yet. Call read_metadata if you want to check, or read_file/read_document directly if you already know what you need."
    wf = workflow_files()
    if wf:
        note += f" [WORKFLOWS] {WORKFLOWS_DIR}/: {', '.join(wf[:20])}. Read the ones matching the task before working."
    else:
        note += f" [WORKFLOWS] no '{WORKFLOWS_DIR}' folder found - proceed with standard practice and state your assumptions."
    note += f" [CAPABILITIES] vision: {'available' if vision_enabled else 'NOT available'}."
    return note

def summarize_output(task_message: str, action: str, raw_output: str) -> str:
    # Compresses ONE tool's raw output into a short, task-aware note before it ever reaches sess.context.
    # This is the piece that replaces "dump everything, then shrink later" with "compress at the source".
    # Kept out of the main agent loop's context entirely - it's its own tiny call, so it never pollutes
    # or gets polluted by the agent's own running conversation.
    if raw_output.startswith("ERROR") or raw_output.startswith("DENIED"):
        return raw_output    # errors are already short and the agent needs the exact text to react correctly
    if len(raw_output) <= SUMMARY_SKIP_CHARS:
        return raw_output    # not worth a model round-trip
    clipped = raw_output[:SUMMARY_INPUT_CHARS]
    user_msg = f"TASK REASON: {task_message or action}\nACTION: {action}\nRAW OUTPUT:\n{clipped}"
    # BUGFIX: a fixed 180-token budget was too tight for wordy sources (vision descriptions especially) -
    # the grammar-constrained call was getting cut off with finish_reason "length" before it could close
    # the JSON object, json.loads() then threw, and the previous code silently swallowed that exception,
    # so every such failure showed up to the user as a mid-sentence truncation with no way to tell why.
    # Scale the budget with input size (capped) and actually try again once before giving up.
    token_budget = min(SUMMARY_MAX_TOKENS * 2, max(SUMMARY_MAX_TOKENS, len(clipped) // 20))
    for attempt in range(2):
        try:
            with model_lock:
                out = qwen.create_chat_completion(
                    messages=[{"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                    grammar=load_grammar(SUMMARY_GRAMMAR), temperature=0.1, max_tokens=token_budget)
            content = out["choices"][0]["message"]["content"]
            summary = json.loads(content)["summary"].strip()
            return summary or raw_output[:SUMMARY_SKIP_CHARS]
        except Exception as e:
            print(f"[summarize_output] attempt {attempt + 1} failed for action={action!r}: {type(e).__name__}: {e}", file=sys.stderr)
            token_budget = min(token_budget * 2, MAX_OUTPUT_TOKENS)   # give the retry more room, in case it was a length cutoff
    # Both attempts failed - fall back to a hard truncation of the raw text, but now we've LOGGED why on
    # the server's stderr, so this is diagnosable instead of a silent mystery in the transcript.
    return raw_output[:SUMMARY_SKIP_CHARS] + "...[summary failed, truncated]"


def ask_model(context: list, temperature: float) -> list:
    # One constrained LLM call. The grammar guarantees a valid JSON array of tasks, so parsing never fails on syntax.
    with model_lock:
        out = qwen.create_chat_completion(messages=context, grammar=load_grammar(ACTION_GRAMMAR), temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)
    raw = out["choices"][0]["message"]["content"]
    context.append({"role": "assistant", "content": raw})
    tasks = json.loads(raw)
    finish = out["choices"][0].get("finish_reason")
    if finish == "length":
        raise ValueError("model output was cut off (too long); ask for smaller steps")
    return tasks


def build_results_message(results: list, nudges: list, sess: Session) -> str:
    text = "TOOL RESULTS:\n" + "\n\n".join(f"--- task {i + 1}: {t['action']} ---\n{out}" for i, t, out in sorted(results, key=lambda r: r[0]))
    if nudges:
        text += "\n\nUSER NUDGE (sent while you were working - this is important, follow it): " + " | ".join(nudges)
    text += "\n\n" + workspace_notes() + f"\n[AUTOPILOT: {'ON - the user is away' if sess.autopilot else 'off'}]"
    return text


def run_agent(sess: Session, user_input: str, max_steps: int, temperature: float) -> None:
    STORAGE.mkdir(parents=True, exist_ok=True)
    emit(sess, "user", text=user_input)
    sess.context.append({"role": "user",
                         "content": user_input + "\n\n" + workspace_notes() + f"\n[AUTOPILOT: {'ON - the user is away' if sess.autopilot else 'off'}]"})
    steps_since_error = 0
    decode_failures = 0
    for step in range(1, max_steps + 1):
        if sess.cancelled.is_set():
            emit(sess, "cancelled")
            break
        emit(sess, "thinking", step=step)

        # Apply context history compaction and window trimming
        sess.context = compact_context_history(sess.context)
        sess.context = trim_context(sess.context)

        try:
            tasks = ask_model(sess.context, temperature)
        except Exception as e:
            is_decode_error = "-3" in str(e) or "decode" in str(e).lower()
            emit(sess, "error", text=f"The model produced an unusable answer: {e}")
            if is_decode_error:
                decode_failures += 1
                if decode_failures > MAX_DECODE_RETRIES:
                    emit(sess, "final",
                         text=f"I stopped because the model kept running out of context space ({e}), even after shrinking the conversation.")
                    break
                sess.context = sess.context[:2]
                sess.context.append({"role": "user",
                                     "content": f"(Context was reset after a decode error - continuing.) {user_input}\n\n" + workspace_notes()})
                continue
            steps_since_error += 1
            if steps_since_error >= 3:
                emit(sess, "final", text=f"I stopped because the model failed {steps_since_error} times in a row: {e}")
                break
            if sess.context and sess.context[-1]["role"] == "assistant":
                sess.context.pop()
            sess.context = trim_context(sess.context)
            sess.context.append({"role": "user",
                                 "content": f"Your last output could not be used ({e}). Reply again with a valid, shorter JSON array."})
            continue

        steps_since_error = 0
        final = next((t for t in tasks if t["action"] == "final_answer"), None)
        if final:
            t_val = str(final.get("text") or "").strip()
            m_val = str(final.get("message") or "").strip()
            final_text = t_val if (t_val and len(t_val) >= len(m_val)) else (m_val or t_val)
            emit(sess, "final", text=final_text, message=m_val or final_text)
            break

        results = run_batch(sess, tasks)
        if sess.cancelled.is_set():
            emit(sess, "cancelled")
            break
        nudges = nudge_window(sess)
        sess.context.append({"role": "user", "content": build_results_message(results, nudges, sess)})
        save_session(sess)
    else:
        emit(sess, "final",
             text=f"I reached the step limit ({max_steps}) before finishing. Send another message to continue from here.")
    emit(sess, "done")
    save_session(sess)


def initialize(user_input: str, emit_callback, session_id: str = None, max_steps: int = 40, temp: float = 0.2, autopilot: bool = False) -> str:
    # Entry point (called by server.py in a worker thread). New session if session_id is None, otherwise the old conversation continues.
    load_model()
    HISTORY.mkdir(parents=True, exist_ok=True)
    sess = sessions.get(session_id) if session_id else None
    if sess is None and session_id:
        data = load_session_data(session_id)
        if data:
            sess = Session(data["id"], data["name"], data["context"], data.get("autopilot", False), data["created"], data.get("events", []), data.get("network_audit", []))
    is_new = sess is None
    if sess is None:
        sess = Session(datetime.now().strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4], fallback_name(user_input), [{"role": "system", "content": SYSTEM_PROMPT}], autopilot)
    with sess.lock:
        if sess.running:
            return sess.id
        sess.running = True
    sess.emit = emit_callback
    sess.cancelled.clear()
    sessions[sess.id] = sess
    sess.autopilot = autopilot or sess.autopilot
    emit(sess, "session", id=sess.id, name=sess.name, autopilot=sess.autopilot)
    if is_new:
        rename_session_async(sess, user_input)   # real title arrives a little later via its own "session" event; never blocks the agent
    try:
        run_agent(sess, user_input, max_steps, temp)
    finally:
        sess.running = False
    return sess.id