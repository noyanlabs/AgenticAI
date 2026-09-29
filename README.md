# EcOS
A **marine-intelligence** AI agent designed for **safety-critical**, **real-time decisions at sea**. It runs on a **multi-agent** core that plans, fetches and correlates **satellite, weather and ocean data**, and replies in the fisherman's own **regional language**, giving **explainable recommendations**, **proactive alerts** and **drift-aware geofencing** even in **low-connectivity** waters.

## Project Structure
```
GitHub Repository
EcOS/
├── Blueprint/
│   ├── blueprint_implementation.ipynb
│   └── sample_qwen_output.json
├── ProjectBackend/
│   ├── Interpreter/                   
│   │   ├── ppt_interpreter.py
│   │   ├── docx_interpreter.py
│   │   ├── xcel_interpreter.py
│   │   ├── csv_interpreter.py
│   │   ├── pdf_interpreter.py
│   │   ├── zip_interpreter.py
│   │   ├── image_and_encoded_image_interpreter.py
│   │   └── video_interpreter.py
│   ├── LLM/                           
│   │   └── Qwen3.5-9B-Q4_K_M.gguf
│   ├── LocalStorage/
│   ├── ToolCall/
│   │   └── web_requests.py
│   ├── central_backend.py             
│   ├── server.py
│   └── sandbox.py                   
├── GUI/                      ------> Flutter Web app for locally hosted LLMs or the centres (An open-source customisable tool for researchers)
├── EcOS_UI/                  ------> Flutter Web app for local Users (Workers, etc) using 3rd party API calls to LLMs
│   ├── lib/
│   │   ├── main.dart
│   │   └── dashboard.dart
│   └── assets/
├── cli.py                            
└── README.md
```


## How to use this Agent
> For Researchers or Hosting Centers
1. Pull this Github Repo on your computer.
2. Download your desired LLM in `gguf` format from Hugging Face into the `LLM` dirrectory inside the `ProjectBackend` directory, and name the model as `main.gguf` and the vision model as `main_vision.gguf`.
3. Navigate to the AgenticAI directory in terminal.
4. Run this command:
```bash
python3 start.py --gui
```
5. It will open the AgenticAI in your Local Browser.
6. 🎉 Enjoy the Automation, by adding your required files in the `LocalStorage` Directory

> For Local Workers
1. Open the Link :
```link
https://google.com/
```
2. 🎉 Enjoy the Automation
_😊 We hope that you will definitely feel our work..._

`User (CLI, GUI) <---> Server (Python) <---> Central Backend (Python)`

The EcOS Agent can be accessed in two ways:
1. **CLI**
2. **GUI**

### To Do List:
- [x] **Central Backend** - Acts as the primary execution engine (Rule based or Deep Learning Based)
- [x] **Collection of LLMs & API Access** - Acting as different parts of a brain to process various types of received data. (Managing which LLM to use when whithout needing to swap LLMs)
- [x] **Interpreter** - It interprets meaning from the documents, images and videos. (long xcel and csv data dealing, video dealing, the most important thing needed for EcOS)
- [x] **Sandbox** - It helps executing code to do a variety of tasks like creating PPT, DOCUMENT, PDF, XCEL, CSV, etc and doing some calculations. (Second most important thing)
- [x] **CLI Application**
- [x] **GUI Application** (Web Based or mobile based one for all to be made on Flutter)
- [x] **Testing and Improvements**
- [x] **Finalizing**

### Workload Distribution:

|Member|Work|
|---|---|
|**Aradhya**|Tool Call|
|**Asjad**|Central Backend, Collection of LLMs and API Access, Interpreter, Sandbox, CLI Application, GUI Application, Tool Call|
|**Jairaj**|Interpreter|
|**Mujtaba**|Interpreter|

100% Progress ====================
