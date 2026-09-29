# The Terminal Face - run `python cli.py` and that's it. No arguments. It starts the backend server by itself if it is not running.
# Only needs: rich, websockets, httpx.
from pathlib import Path
import asyncio
import json
import os
import subprocess
import sys
import time

import httpx
import websockets
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "ProjectBackend"
TOKEN_FILE = BACKEND_DIR / ".server_token"
PORT = int(os.environ.get("AGENTICAI_PORT", "8765"))
HTTP = f"http://127.0.0.1:{PORT}"
WS_URL = f"ws://127.0.0.1:{PORT}/ws"
NUDGE_SECONDS = 8
STARTUP_TIMEOUT = 600                       # loading a 9B model can take a while on a slow disk
TYPEWRITER_SECONDS = 1.2                    # any answer, however long, is revealed within about this time
TYPEWRITER_MIN_CHUNK = 3
console = Console()
state = {"token": None, "session_id": None, "autopilot": False, "net_total": 0, "net_sent": 0, "net_blocked": 0, "net_denied": 0}
BANNER = r"""
 _____      ___  ____  
| ____|__ _/ _ \/ ___| 
|  _| / _` | | | \___ \ 
| |__| (_| | |_| |___) |
|_____\__,_|\___/|____/ 
"""
MATH_SYMBOLS = {r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\theta": "θ", r"\lambda": "λ", r"\mu": "μ", r"\pi": "π", r"\sigma": "σ", r"\omega": "ω",
                r"\Delta": "Δ", r"\Sigma": "Σ", r"\Omega": "Ω", r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
                r"\infty": "∞", r"\sum": "Σ", r"\int": "∫", r"\prod": "Π", r"\partial": "∂", r"\nabla": "∇", r"\to": "→", r"\rightarrow": "→", r"\in": "∈", r"\sqrt": "√"}
ACTION_ICONS = {"list_dir": "▤", "read_file": "▯", "read_document": "▣", "write_file": "✎", "append_file": "✎", "delete_path": "✖", "move_path": "⇄", "make_dir": "▤",
                "search_files": "⌕", "read_metadata": "◈", "write_metadata": "◈", "run_python": "▶", "run_shell": "▶", "pip_install": "⇩", "network_request": "🌐",
                "analyze_image": "◉", "analyze_video": "◉", "query_sql": "▦", "ask_user": "?", "final_answer": "✔"}


def read_key(timeout: float):
    # Non-blocking single key read: returns a character or None. Works on Windows, macOS and Linux; degrades to "no key" if stdin is not a terminal.
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return None
    if os.name == "nt":
        import msvcrt
        end = time.time() + timeout
        while time.time() < end:
            if msvcrt.kbhit():
                return msvcrt.getwch()
            time.sleep(0.02)
        return None
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def flush_pending_keys() -> None:
    # drop anything typed while the agent was busy so it cannot leak into the next prompt
    if os.name == "nt":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    elif sys.stdin.isatty():
        import termios
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except Exception:
            pass


def show_banner() -> None:
    console.clear()
    console.print(Text(BANNER, style="bold white"), highlight=False)
    console.print(Text("  local  ·  private  ·  autonomous", style="dim"), highlight=False)
    console.print(Rule(style="dim"))


def render_math(text: str) -> str:
    # Terminals cannot typeset LaTeX, so common commands become Unicode and $...$ markers are dropped for readability.
    import re
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", text)
    text = re.sub(r"\^\{([^{}]*)\}", lambda m: "^(" + m.group(1) + ")" if len(m.group(1)) > 1 else "^" + m.group(1), text)
    text = re.sub(r"_\{([^{}]*)\}", lambda m: "_(" + m.group(1) + ")" if len(m.group(1)) > 1 else "_" + m.group(1), text)
    for cmd in sorted(MATH_SYMBOLS, key=len, reverse=True):
        text = re.sub(re.escape(cmd) + r"(?![a-zA-Z])", MATH_SYMBOLS[cmd], text)
    text = re.sub(r"\\(text|mathrm|mathbf|left|right)\s*\{?([^{}]*)\}?", r"\2", text)
    text = text.replace("\\,", " ").replace("\\ ", " ").replace("\\;", " ")
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: "\n\n    " + m.group(1).strip() + "\n\n", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", lambda m: m.group(1).strip(), text)
    return text


def typewriter(markdown_text: str, title: str = "AgenticAI") -> None:
    # The model answers in one piece (structure matters), so this only ANIMATES the reveal. Long answers are revealed in bigger chunks so it stays fast.
    text = render_math(markdown_text).strip()
    total = len(text)
    if total == 0:
        console.print(Panel(Text("(empty response)", style="dim"), title=f"[bold]{title}[/]", border_style="white", box=box.ROUNDED, padding=(1, 2)))
        return
    duration = min(TYPEWRITER_SECONDS, 0.15 + total / 900)      # short replies finish almost instantly, long ones cap at TYPEWRITER_SECONDS
    chunk = max(TYPEWRITER_MIN_CHUNK, int(total / (duration * 40)))
    delay = duration / max(total / chunk, 1)
    with Live(console=console, refresh_per_second=30, transient=False) as live:
        for i in range(0, total + chunk, chunk):
            live.update(Panel(Markdown(text[:i] + ("▌" if i < total else "")), title=f"[bold]{title}[/]", border_style="white", box=box.ROUNDED, padding=(1, 2)))
            time.sleep(delay)
        live.update(Panel(Markdown(text), title=f"[bold]{title}[/]", border_style="white", box=box.ROUNDED, padding=(1, 2)))


def token_from_file() -> str | None:
    try:
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def server_health() -> dict | None:
    try:
        r = httpx.get(f"{HTTP}/health", timeout=1.5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def start_server_process() -> None:
    # Detached background process so closing the CLI does not kill an answer in progress; output goes to a log file, never into the UI.
    log = open(BACKEND_DIR / "server.log", "ab")
    kwargs = {"cwd": str(BACKEND_DIR), "stdout": log, "stderr": log, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([sys.executable, "server.py"], **kwargs)


def ensure_backend() -> bool:
    # Shows a spinner while the server starts and the model loads. The server answers /health within a second, so the CLI never looks frozen.
    health = server_health()
    with console.status("[bold]Starting AgenticAI backend…", spinner="dots") as status:
        if health is None:
            start_server_process()
            deadline = time.time() + 30
            while health is None and time.time() < deadline:
                time.sleep(0.4)
                health = server_health()
            if health is None:
                console.print("[bold red]The backend server did not start.[/] See ProjectBackend/server.log")
                return False
        started = time.time()
        while not health.get("model_loaded"):
            status.update(f"[bold]Loading the language model… {int(time.time() - started)}s[/] [dim](first start can take a minute)[/]")
            if time.time() - started > STARTUP_TIMEOUT:
                console.print("[bold red]Model loading timed out.[/] Check LLM/main.gguf and ProjectBackend/server.log")
                return False
            time.sleep(1)
            health = server_health() or health
    state["token"] = token_from_file()
    if not state["token"]:
        console.print("[bold red]Could not read the server token.[/]")
        return False
    state["vision"] = health.get("vision", False)
    return True


def api(method: str, path: str):
    r = httpx.request(method, f"{HTTP}{path}", headers={"x-agent-token": state["token"]}, timeout=10)
    r.raise_for_status()
    return r.json()


def confirm_autopilot() -> bool:
    console.print(Panel(Text.from_markup(
        "[bold red]AUTOPILOT WARNING[/]\n\n"
        "In autopilot the agent will, [bold]without asking you[/]:\n"
        "  • overwrite, move and DELETE files in LocalStorage\n"
        "  • run any Python code and shell commands it writes\n"
        "  • skip the 8-second window to correct it\n"
        "  • skip its own questions to you\n\n"
        "Internet access will [bold]still[/] ask for your approval.\n"
        "A wrong step can destroy data that has no backup.\n\n"
        "Type [bold]I ACCEPT THE RISK[/] to enable it."), border_style="red", box=box.HEAVY, title="[bold red]⚠  DANGER  ⚠[/]"))
    return console.input("[bold red]> [/]").strip() == "I ACCEPT THE RISK"


def network_badge() -> Text:
    if state["net_total"] == 0:
        return Text("● NETWORK: 0 requests — no internet activity", style="green")
    style = "yellow" if state["net_sent"] else "green"
    return Text(f"● NETWORK: {state['net_total']} attempted · {state['net_sent']} sent · {state['net_blocked']} blocked · {state['net_denied']} denied", style=style)


def show_network_proof(audit: list) -> None:
    table = Table(box=box.SIMPLE_HEAD, title="Network audit (every attempt is logged)", title_style="bold")
    for col in ("Time", "Request", "Bytes", "SHA-256", "Leak scan", "Outcome"):
        table.add_column(col, overflow="fold")
    for a in audit:
        scan = a.get("leak_scan", {})
        outcome = "sent" if a.get("sent") else ("blocked" if "BLOCKED" in a.get("note", "") else "denied/failed")
        table.add_row(a.get("time", "")[11:], f"{a.get('method', '')} {a.get('url', '')}"[:60], str(a.get("outbound_bytes", 0)), a.get("outbound_sha256", "")[:12] + "…",
                      "clean" if scan.get("clean", True) else "[red]LEAK[/]", outcome)
    if not audit:
        table.add_row("—", "no network activity in this session", "0", "—", "—", "—")
    console.print(table)
    console.print(Text("The leak scan compares outbound data against your workspace files. It is a best-effort check, not a mathematical guarantee.", style="dim"))


def update_network_state(audit: dict) -> None:
    state["net_total"] += 1
    note = audit.get("note", "")
    if audit.get("sent"):
        state["net_sent"] += 1
    elif "BLOCKED" in note:
        state["net_blocked"] += 1
    else:
        state["net_denied"] += 1


def show_task(event: dict) -> None:
    icon = ACTION_ICONS.get(event["action"], "•")
    params = event.get("params") or {}
    packages = ", ".join(params.get("packages")) if isinstance(params.get("packages"), list) else params.get("packages")
    detail = params.get("path") or params.get("query") or params.get("command") or params.get("url") or params.get("question") or params.get("pattern") or packages or ""
    header = Text.assemble((f" {icon} ", "bold"), (event["action"], "bold"), (f"  {str(detail)[:70]}" if detail else "", "dim"))
    console.print(header)
    if event.get("message"):
        console.print(Text("   " + event["message"], style="italic"))
    if event["action"] in ("run_python", "run_shell") and (params.get("code") or params.get("command")):
        from rich.syntax import Syntax
        code = params.get("code") or params.get("command")
        console.print(Panel(Syntax(str(code)[:1500], "python" if params.get("code") else "bash", theme="ansi_dark", word_wrap=True), border_style="dim", box=box.ROUNDED))


def show_result(event: dict) -> None:
    out = event.get("output", "")
    lines = out.splitlines()
    shown = "\n".join(lines[:14]) + (f"\n… ({len(lines) - 14} more lines)" if len(lines) > 14 else "")
    style = "red" if out.startswith(("ERROR", "DENIED", "BLOCKED")) else "dim"
    console.print(Panel(Text(shown[:1800] or "(no output)", style=style), title=f"[dim]result · {event.get('seconds', 0)}s[/]", border_style="dim", box=box.ROUNDED, title_align="left"))


async def countdown_and_nudge(ws) -> None:
    # The 8-second window. Press any key to start typing a correction; press Enter to send it. Otherwise the agent continues by itself.
    loop = asyncio.get_running_loop()
    console.print()
    end = time.time() + NUDGE_SECONDS
    first = None
    while time.time() < end and first is None:
        remaining = end - time.time()
        bar = "█" * int(remaining / NUDGE_SECONDS * 24) + "░" * (24 - int(remaining / NUDGE_SECONDS * 24))
        console.print(Text(f"\r ⏱  {remaining:3.0f}s  {bar}  press any key, then type your message", style="cyan"), end="")
        first = await loop.run_in_executor(None, read_key, 0.25)
    console.print("\r" + " " * 90 + "\r", end="")
    if first is None:
        return
    flush_pending_keys()
    console.print(Text("  Timer paused. Type your message to the agent (Enter to send, empty to cancel):", style="cyan"))
    text = (await loop.run_in_executor(None, lambda: console.input("[bold cyan]  you › [/]"))).strip()
    if text:
        await ws.send(json.dumps({"type": "nudge", "session_id": state["session_id"], "text": text}))
        console.print(Text("  ✔ message will be delivered to the agent", style="green"))


async def answer_ask(ws, event: dict) -> None:
    loop = asyncio.get_running_loop()
    kind = event["kind"]
    if kind == "network":
        border, label = "blue", "🌐 INTERNET ACCESS REQUEST"
    elif kind == "question":
        border, label = "magenta", "? THE AGENT ASKS YOU"
    else:
        border, label = "yellow", "⚠  PERMISSION NEEDED"
    body = Text(event["question"], style="bold")
    if event.get("detail"):
        body.append("\n\n" + event["detail"], style="dim")
    console.print(Panel(body, title=f"[bold]{label}[/]", border_style=border, box=box.HEAVY))
    flush_pending_keys()
    if kind == "question":
        answer = await loop.run_in_executor(None, lambda: console.input("[bold magenta]  answer › [/]"))
    else:
        raw = await loop.run_in_executor(None, lambda: console.input(f"[bold {border}]  allow? (y/N) › [/]"))
        answer = "yes" if raw.strip().lower() in ("y", "yes") else "no"
    await ws.send(json.dumps({"type": "reply", "session_id": state["session_id"], "request_id": event["request_id"], "answer": answer}))


async def handle_event(ws, event: dict, spinner: dict) -> bool:
    # returns True when the turn is finished
    kind = event["type"]
    if kind == "session":
        state["session_id"] = event["id"]
        console.print(Text(f" session · {event['name']}", style="dim"))
    elif kind == "thinking":
        spinner["status"].update(f"[bold]Thinking… step {event['step']}")
        spinner["status"].start()
    elif kind == "user":
        pass
    else:
        spinner["status"].stop()
        if kind == "task":
            show_task(event)
        elif kind == "result":
            show_result(event)
        elif kind == "countdown":
            await countdown_and_nudge(ws)
        elif kind == "nudge":
            console.print(Text(f"  ➜ your message reached the agent: {event['text']}", style="cyan"))
        elif kind == "ask":
            await answer_ask(ws, event)
        elif kind == "network":
            update_network_state(event["audit"])
            console.print(network_badge())
        elif kind == "error":
            console.print(Panel(Text(event.get("text", "error")), border_style="red", title="error"))
        elif kind == "final":
            console.print()
            t_val = str(event.get("text") or "").strip()
            m_val = str(event.get("message") or "").strip()
            final_text = (t_val if len(t_val) >= len(m_val) else m_val) if (t_val and m_val) else (t_val or m_val)
            typewriter(final_text)
        elif kind == "cancelled":
            console.print(Text(" ■ stopped", style="bold"))
        elif kind == "autopilot":
            state["autopilot"] = event["enabled"]
        elif kind == "done":
            return True
    return False


async def run_turn(text: str, session_id: str | None) -> None:
    # One user message -> stream of events until "done". Ctrl+C asks the agent to stop instead of killing the CLI.
    async with websockets.connect(f"{WS_URL}?token={state['token']}", max_size=None, ping_interval=20) as ws:
        await ws.send(json.dumps({"type": "send", "text": text, "session_id": session_id, "autopilot": state["autopilot"]}))
        spinner = {"status": console.status("[bold]Thinking…", spinner="dots")}
        spinner["status"].start()
        try:
            async for raw in ws:
                if await handle_event(ws, json.loads(raw), spinner):
                    break
        except KeyboardInterrupt:
            await ws.send(json.dumps({"type": "cancel", "session_id": state["session_id"]}))
            console.print(Text("\n ■ stopping after the current step…", style="bold"))
        finally:
            spinner["status"].stop()
    console.print(network_badge())
    console.print(Rule(style="dim"))


def show_history_table(sessions: list) -> None:
    table = Table(box=box.SIMPLE_HEAD, title="Previous chats", title_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Name")
    table.add_column("Last used", style="dim")
    table.add_column("Msgs", justify="right", style="dim")
    for i, s in enumerate(sessions, 1):
        table.add_row(str(i), s["name"], s["updated"].replace("T", " "), str(s["messages"]))
    console.print(table)


def replay_session(session_id: str) -> None:
    data = api("GET", f"/api/sessions/{session_id}")
    console.print(Panel(Text(data["name"], style="bold"), border_style="white", subtitle=f"created {data['created'].replace('T', ' ')}"))
    state["net_total"] = state["net_sent"] = state["net_blocked"] = state["net_denied"] = 0
    for a in data.get("network_audit", []):
        update_network_state(a)
    for e in data["events"]:
        if e["type"] == "user":
            console.print(Panel(Text(e["text"]), title="[bold]You[/]", border_style="dim", box=box.ROUNDED, title_align="left"))
        elif e["type"] == "task":
            show_task(e)
        elif e["type"] == "final":
            t_val = str(e.get("text") or "").strip()
            m_val = str(e.get("message") or "").strip()
            final_text = (t_val if len(t_val) >= len(m_val) else m_val) if (t_val and m_val) else (t_val or m_val)
            console.print(Panel(Markdown(render_math(final_text)), title="[bold]AgenticAI[/]", border_style="white", box=box.ROUNDED, padding=(1, 2)))
    console.print(network_badge())


def chat_loop(session_id: str | None) -> None:
    state["session_id"] = session_id
    console.print(Text("  Type your message. Commands: /new  /history  /autopilot  /network  /help  /quit", style="dim"))
    while True:
        badge = "[bold red] AUTOPILOT[/]" if state["autopilot"] else ""
        try:
            text = console.input(f"\n[bold]you{badge} › [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            raise SystemExit(0)
        if text == "/new":
            return
        if text == "/history":
            return
        if text == "/help":
            console.print(Text("  /new  start a new chat      /history  switch chat      /autopilot  toggle autopilot\n  /network  show network proof      /quit  exit      Ctrl+C  stop the agent", style="dim"))
            continue
        if text == "/network":
            audit = api("GET", f"/api/sessions/{state['session_id']}")["network_audit"] if state["session_id"] else []
            show_network_proof(audit)
            continue
        if text == "/autopilot":
            if state["autopilot"]:
                state["autopilot"] = False
                console.print(Text("  autopilot OFF", style="green"))
            elif confirm_autopilot():
                state["autopilot"] = True
                console.print(Text("  AUTOPILOT ON — the agent will act without asking", style="bold red"))
            else:
                console.print(Text("  not enabled", style="dim"))
            continue
        asyncio.run(run_turn(text, state["session_id"]))


def main_menu() -> str:
    show_banner()
    console.print(network_badge())
    console.print()
    console.print("  [bold]1[/]  New chat")
    console.print("  [bold]2[/]  Previous chats")
    console.print("  [bold]3[/]  Delete a chat")
    console.print("  [bold]4[/]  Quit")
    return console.input("\n[bold]  choose › [/]").strip()


def initialize() -> None:
    # Start here: backend -> menu -> chat loop.
    show_banner()
    if not ensure_backend():
        return
    while True:
        choice = main_menu()
        if choice == "1":
            state["net_total"] = state["net_sent"] = state["net_blocked"] = state["net_denied"] = 0
            show_banner()
            chat_loop(None)
        elif choice in ("2", "3"):
            sessions = api("GET", "/api/sessions")
            if not sessions:
                console.print("  [dim]No previous chats yet.[/]")
                time.sleep(1.2)
                continue
            show_history_table(sessions)
            pick = console.input("\n[bold]  number (Enter to go back) › [/]").strip()
            if not pick.isdigit() or not 1 <= int(pick) <= len(sessions):
                continue
            chosen = sessions[int(pick) - 1]
            if choice == "3":
                if console.input(f"  Delete '{chosen['name']}'? (y/N) › ").strip().lower() == "y":
                    api("DELETE", f"/api/sessions/{chosen['id']}")
                    console.print("  [dim]deleted[/]")
                    time.sleep(0.8)
                continue
            show_banner()
            replay_session(chosen["id"])
            chat_loop(chosen["id"])
        elif choice in ("4", "q", "quit"):
            console.print("  [dim]Goodbye.[/]")
            return


if __name__ == "__main__":
    try:
        initialize()
    except KeyboardInterrupt:
        console.print("\n  [dim]Goodbye.[/]")
    except SystemExit:
        pass
