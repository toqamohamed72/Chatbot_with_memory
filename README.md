# Custom AI Chatbot with Memory

A command-line chatbot that remembers the conversation, built step by step on the OpenAI Python SDK. It works with the OpenAI API or with a free local model served by [Ollama](https://ollama.com) through the same OpenAI-compatible interface.

## Features

- **Conversation memory.** The API is stateless, so the full `messages` list is re-sent with every request.
- **System prompt** that defines the assistant's persona and rules.
- **History pruning.** A sliding window (`max_turns`) keeps the size of every request bounded.
- **Chat commands:** `/help`, `/history`, `/clear`, `/exit`.
- **Friendly error messages** that say what to fix: invalid key, unknown model, no credits, server unreachable, timeout.
- **Secrets in `.env`**, validated at startup and never printed.
- **Offline tests.** The test suite uses a fake client, so it needs no API key, no network and no model.

## Project structure

```
custom-ai-chatbot-with-memory/
├── src/
│   ├── config.py           # loads and validates settings from .env
│   ├── chatbot.py          # ChatBot class: history, system prompt, pruning, error handling
│   ├── phase4_chat.py      # terminal chat interface
│   ├── phase2_basic.py     # first API request, step by step
│   ├── phase2_tests.py     # prompt experiments and API error demos
│   ├── list_models.py      # lists the models available on the server
│   └── main.py             # configuration smoke test (written for the OpenAI API)
├── tests/
│   └── test_chatbot.py     # offline pytest suite (20 tests)
├── notebooks/
│   └── chatbot_demo.ipynb  # the whole project in a single notebook
├── .env.example            # template for your local .env
├── pytest.ini
├── requirements.txt
└── requirements-dev.txt
```

## Requirements

- Python 3.10 or newer
- **Either** [Ollama](https://ollama.com/download) with a pulled model (free, runs locally) **or** an OpenAI API key with available credits

## Setup

```powershell
git clone https://github.com/YOUR-USERNAME/custom-ai-chatbot-with-memory.git
cd custom-ai-chatbot-with-memory

python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create your local configuration file from the template (macOS/Linux: `cp .env.example .env`):

```powershell
Copy-Item .env.example .env
```

### Option A: local model with Ollama

```powershell
ollama pull llama3.2:3b
```

`.env`:

```
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1/
OPENAI_MODEL=llama3.2:3b
```

The key is required by the SDK but ignored by Ollama, so any non-empty value works.

### Option B: OpenAI API

`.env`:

```
OPENAI_API_KEY=sk-...your key...
OPENAI_MODEL=<a model name listed by python src/list_models.py>
```

Leave `OPENAI_BASE_URL` out. Model names change over time, so list the models available to your key instead of copying a name from a tutorial.

> `.env` is listed in `.gitignore`. Never commit it, and never paste a real key into an issue, a notebook or a screenshot.

## Usage

```powershell
python src/phase4_chat.py
```

| Command | Effect |
|---|---|
| `/history` | Show the stored messages and the size of the last request |
| `/clear` | Forget the conversation (the system prompt is kept) |
| `/help` | Show the commands |
| `/exit` | Quit |

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

The suite covers message order, memory across turns, `clear_conversation()`, history pruning (including a bound on the size of every request) and error handling.

## Notebook

`notebooks/chatbot_demo.ipynb` walks through the same project in five phases in one file, with explanations: setup, the first request, memory, the `ChatBot` class, and tests. Select the `.venv` environment as the kernel, then run **Restart & Run All**.

## How the memory works

A chat model keeps nothing between requests. Every turn appends the user message to a list, sends the whole list, and appends the model's answer. Two consequences drive the design:

1. The request grows with every turn, and so does the cost on paid APIs.
2. The context window is finite.

`ChatBot.prune_history()` keeps the system message and the last `max_turns` user/assistant pairs, so the request size stays bounded.

## Limitations

- Memory lives in RAM and is lost when the program exits.
- Pruning counts messages, not tokens, and drops old messages silently (a fact mentioned early is forgotten after `max_turns` turns).
- Small local models (a 3B model, in testing) give noticeably weaker answers in Arabic than in English.

## Roadmap

- Persist conversations (JSON file or SQLite)
- Summarise old turns instead of dropping them
- Stream answers token by token
- Web interface (Streamlit or Gradio) reusing the `ChatBot` class
- Evaluate models on a fixed prompt set, including Arabic
