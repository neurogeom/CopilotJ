# AGENTS.md

CopilotJ is a conversational multi-agent system for bioimage analysis. It translates natural-language requests into executable workflows by coordinating the Fiji/ImageJ ecosystem, Python scientific libraries, and deep-learning models.

Three runtime components work together:

- **Bridge server** (`copilotj/server/`) — aiohttp-based HTTP/WebSocket server. Serves the REST API, manages chat threads, and relays commands to the ImageJ plugin via WebSocket.
- **ImageJ plugin** (`plugin/`) — Java Maven project running inside Fiji. Executes ImageJ commands and reports state back via WebSocket.
- **Web frontend** (`web/`) — Vue 3 + PrimeVue + Tailwind CSS SPA. Communicates with the bridge server via REST (NDJSON streaming for chat).

## Architecture

### Python Backend (`copilotj/`)

The core agent framework follows a layered architecture:

**`copilotj/core/`** — Foundation abstractions:

- `Agent` / `ChatAgent` — Base agent classes. `ChatAgent` adds LLM streaming via `ModelClient` with abort support.
- `Runtime` — Shared logging/UI dispatch layer. Every agent in a pattern shares one runtime.
- `Pattern` — Orchestrator that groups agents, manages their shared runtime, and routes messages.
- `ModelClient` — Abstract LLM interface with concrete implementations for OpenAI (chat completions + responses API), Gemini, Ollama, and DeepSeek. Uses `ReActChatCompletionClient` wrapper for ReAct-format output parsing.
- `Tool` / `FunctionTool` — Tools are pydantic models with auto-generated JSON schemas from function signatures. `HandoffFunctionTool` emits UI handoff events when called.
- `UI` / `CLI` — Abstract UI event system. Events (`UIEventPost`, `UIEventToolCall`, etc.) are sent as NDJSON to the frontend. `CLI` is the terminal fallback.
- `Config` — Environment-based configuration.
- `lifecycle` — Thread-safe cleanup registry (`register_cleanup` / `run_cleanup`). Used by resources that need cleanup on shutdown or signal but may be created on the daemon thread in managed mode.

**`copilotj/multiagent/`** — Agent orchestration:

- `LeaderDriven` (Pattern) — Main multi-agent pattern. Creates a `LeaderAgent` and loads specialized `Executor` agents from TOML configs.
- `LeaderAgent` (ChatAgent) — ReAct-style leader that reasons, calls tools (ImageJ perception, macro execution, Python scripts, knowledge bank), and delegates to specialized agents. Manages dialog-level conversation history with summarization.
- `Executor` (ChatAgent) — Generic specialized agent with its own system prompt and tools, loaded from TOML. Runs ReAct loops with tool retry and error recovery.
- `agent_configs/` — TOML files defining specialized agents. Active configs: `tool_agent.toml`, `research_agent.toml`, `success_case.toml`, `imagej_macro_help.toml`. Disabled configs have `.disabled.toml` suffix.
- `agent_loader.py` — Dynamically loads agent classes and tool functions from TOML configs using `importlib`.

**`copilotj/server/`** — HTTP layer:

- `Server` — aiohttp application with CORS. Routes: `/api/threads` (CRUD), `/api/threads/{id}/posts` (NDJSON chat stream), `/api/plugins` (WebSocket bridge), `/api/config`.
- `Bridge` — WebSocket hub. Manages connected ImageJ plugin clients with ID negotiation. Forwards events between the Python backend and the Java plugin.
- `Threads` — Thread-per-conversation manager. Each `_Thread` creates a `LeaderDriven` pattern, runs the agent as a background task, and streams `UIEvent`s back as NDJSON.

**`copilotj/plugin/`** — Python-side ImageJ interface:

- `PluginAPI` / `HTTPPluginAPI` / `BridgePluginAPI` — Client wrappers that send typed request/response messages to the Java plugin via the bridge. Provides methods like `take_snapshot()`, `run_script()`, `capture_image()`, `call_action()`.
- `awt/` — AWT widget tree model mirroring Java-side widget types (buttons, checkboxes, sliders, etc.) for UI interaction.

#### Key Environment Variables

- `COPILOTJ_MANAGED`: Set `1` when run under Appose managed mode
- `COPILOTJ_LLM_API_KEY`: API key for the primary model
- `COPILOTJ_LLM_BASE_URL`: Override API endpoint
- `COPILOTJ_VISION_ENABLED`: Set `1` to enable vision features (disabled by default for privacy)
- `COPILOTJ_VLM_MODEL`: Vision model for image analysis (requires `COPILOTJ_VISION_ENABLED=1`)
- `COPILOTJ_VLM_API_KEY`: API key for the vision model
- `COPILOTJ_LLM_PROXY`: HTTP proxy
- `CIJ_PROXY`: Global download proxy for all backend downloads (HuggingFace, model DB, bioimage model zoo, research/Tavily/DDGS, cellpose/stardist); also the fallback proxy for the LLM/VLM clients when their own proxy is unset. Passed explicitly to each download client (no global proxy env vars are set)
- `COPILOTJ_KB_AUTOSAVE`: Set `1` to auto-ingest dialog summaries into knowledge bank
- `LANGFUSE_SECRET_KEY`/`LANGFUSE_PUBLIC_KEY`: Optional Langfuse observability

### Managed Mode Threading Constraints

When running under Appose managed mode, the Python process has a specific thread architecture:

```
Main thread: start_server() (init script) → Appose stdin/stdout blocking loop
  └── Daemon thread: asyncio event loop (aiohttp server)
```

The daemon thread cannot call `signal.signal()` or any API requiring the main thread. Code running on the daemon thread (agent tools, HTTP handlers) must:

1. Avoid `signal`, `tkinter`, `curses`, and any other API that requires the main thread. Minimize direct use of stdin and stdout, including `input()` and `print()`, because they may interfere with the Appose stdin/stdout loop.
2. Register cleanup callbacks via `copilotj.core.lifecycle.register_cleanup()` instead of `atexit.register()` or signal handlers directly.

### Java Plugin

Maven project using SciJava/ImageJ2 APIs. Key classes: `DefaultCopilotJBridgeService` (WebSocket client), `SnapshotManager` (UI state snapshots), `ScriptRunner` (macro/script execution), `ImagejListener` (operation history tracking). Java AWT widget tree is mirrored on the Python side in `copilotj/plugin/awt/`.

### Web Frontend

Vue 3 SPA with PrimeVue components, Tailwind CSS, Pinia stores, and Vue Router. Views: `Chat.vue` (main chat), `Manual.vue`, `About.vue`, `Home.vue`. API layer in `web/src/apis/` handles thread management and NDJSON parsing.

### Knowledge Bank (`knowledge_bank/`)

TOML-based RAG system with `macro/` (ImageJ macro snippets) and `research/` (dialog-derived insights) subdirectories. Tools in `kb_tools.py` handle retrieval and auto-ingestion (enabled by `COPILOTJ_KB_AUTOSAVE=1`).

## Build & Run Commands

Task runner is [just](https://github.com/casey/just). Run `just` or `just --list` to see all commands.

| Common Command      | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `just dev-server`   | Start the Python bridge server             |
| `just dev-plugin`   | Run the ImageJ plugin in debug mode        |
| `just dev-web`      | Start the Vue frontend dev server          |
| `just build-plugin` | Build the plugin JAR via Maven             |
| `just test`         | Run Python tests via pytest (with doctest) |
| `just lint`         | Run all linters (Python + frontend)        |
| `just format`       | Auto-format all code (Python + frontend)   |

### Coding Conventions

- Prefer module-level imports over local imports inside function bodies. Avoid placing import statements inside functions unless required to address a heavy dependency or an unavoidable circular dependency.
- Ruff config is in `pyproject.toml`: line-length 120, target Python 3.12, Google docstring convention.
- The project uses `flake.nix` with `uv2nix` for reproducible Python environments. Direnv (`.envrc`) loads the nix shell automatically. The Python virtualenv is at `.venv/`.
- Frontend icons: use **`@tabler/icons-vue`** — the only icon dependency in `web/`. Render Tabler icons as `<IconX size="16" />` (named component imports, e.g. `IconChevronRight`); color comes from the parent `text-*` class since icons use `currentColor`.

## Compact Instructions

When compressing, preserve in priority order:

1. Architecture decisions (NEVER summarize)
2. Modified files and their key changes
3. Current verification status (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can delete, keep pass/fail only)
