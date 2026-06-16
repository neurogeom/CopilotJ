# CopilotJ Development

This project contains three parts:

1. **Agent**: The core logic that understands user requests, interacts with AI models, and orchestrates tasks. It
   communicates with the ImageJ plugin via the Bridge server to control ImageJ.
2. **ImageJ plugin**: A Java-based plugin running within ImageJ. It listens for commands from the Agent (relayed by the
   Bridge server), executes these commands using ImageJ's APIs, and sends results or state changes back.
3. **Bridge server**: A communication hub that facilitates interaction between the Python-based Agent and the
   Java-based ImageJ plugin. It typically uses WebSockets to relay messages and commands.

## Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/)
- **Java 8+** and **Maven 3.x**
- **Node.js 22+** with [pnpm v11](https://pnpm.io/installation)
- [just](https://github.com/casey/just) (command runner)

## Quick start with just

The project uses [just](https://github.com/casey/just) as a task runner.
Run `just` or `just --list` to see all available commands.

You can also review the `justfile` to understand how each task is defined and configured, or refer to it directly if you prefer not to install [just](https://github.com/casey/just).

| Command             | Description                    |
| ------------------- | ------------------------------ |
| `just dev-server`   | Run the bridge server          |
| `just dev-plugin`   | Run the ImageJ plugin (debug)  |
| `just dev-web`      | Run the web frontend           |
| `just test`         | Run Python tests               |
| `just test-cov`     | Run tests with coverage report |
| `just build-plugin` | Build the plugin JAR           |
| `just build-web`    | Build the web frontend         |

## Running the components

### Bridge server

```bash
just dev-server
```

Alternatively: `./bin/run-backend.sh`

### Fiji plugin

```bash
just dev-plugin
```

If you make changes to the Java code, restart the plugin to apply the updates.

### Web frontend

```bash
cd web && pnpm install
just dev-web
```

Alternatively: `./bin/run-frontend.sh`

Once everything is running, look for the message `Bridge WebSocket connection established` in the server console to
confirm that the connection is active.

For production deployment, we strongly recommend using the provided Docker-based setup, which includes a preconfigured frontend build and reverse proxy.

## External server mode

By default, the CopilotJ plugin manages the Python backend automatically via the Managed Server tab. For development or advanced setups, you can run the backend server separately and connect via the External Server tab.

### Docker deployment

CopilotJ can be deployed with [Docker](https://docker.com/) and [Docker Compose](https://docs.docker.com/compose/).
This launches the backend, frontend, and reverse proxy in a unified environment.

```bash
git clone https://github.com/neurogeom/CopilotJ.git
cd CopilotJ
```

Create your `.env.local` in the repository root, then:

```bash
# Build the images locally
docker compose build

# Start the full stack
docker compose up -d
```

The default Compose setup exposes the web interface on `http://localhost:8786`.

For GPU passthrough (requires NVIDIA Docker support):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

To rebuild after source changes:

```bash
docker compose up -d --build
```

### Local development server

```bash
git clone https://github.com/neurogeom/CopilotJ.git
cd CopilotJ
uv sync
```

Create your `.env.local` in the repository root, then:

```bash
uv run python -m copilotj.server --host 127.0.0.1 --port 8786
```

### Connecting from the plugin

1. Open the CopilotJ dialog (**Plugins -> CopilotJ**).
2. Switch to the **External Server** tab.
3. Enter the server URL (e.g., `http://localhost:8786`).
4. Click **(Re)Connect**.

In debug mode (`just dev-plugin`), the plugin auto-connects to `http://127.0.0.1:8786`.

## Environment Configuration

CopilotJ supports direct configuration through the frontend, as well as server-side configuration via .env.local environment variables. The .env.local approach is mainly intended for developer convenience: it allows you to configure model credentials and other options once, so you don’t need to reconfigure them every time the server starts.

It can also serve as a server-level default when CopilotJ is hosted for other users. If the server provides a model and API key, end users can simply connect through the web interface without configuring anything themselves.

CopilotJ loads configuration from `.env` and `.env.local` files. In managed mode,
these files are located in the CopilotJ home directory:

- **macOS / Linux:** `~/.local/state/copilotj/.env.local`
- **Windows:** `%LOCALAPPDATA%\copilotj\.env.local`

For external server mode, place `.env.local` in the repository root. Sensitive information such as API keys must be
stored locally and should never be committed to version control.

### Background: models, providers, and API keys

CopilotJ requires at least one **language model** (LLM) to function. A language model is a remote AI service that
understands and generates text; CopilotJ sends your instructions to the model, which reasons about what to do and
orchestrates CopilotJ's tools accordingly.

Models are provided by **AI providers** — companies that operate the model servers. Each provider requires you to create
an account and authenticate with an **API key**: a secret credential you include in your `.env.local`. Every request your
session sends to a model runs on that provider's remote servers and is billed to your account in units called **tokens**
(roughly corresponding to words). Most providers require you to add a payment method and purchase credits before API
requests will succeed; a free or evaluation-tier account will typically return an error on the first request.

CopilotJ uses **two separate model slots**:

- **`COPILOTJ_LLM_MODEL`**: the main reasoning model, used for planning, tool orchestration, and conversation. This is the
  most important setting and must always be configured.
- **`COPILOTJ_VLM_MODEL`**: an optional vision-language model (VLM) used when CopilotJ needs to interpret image content
  directly. All current models from the three recommended providers (OpenAI, Anthropic, Google) support image input. If
  omitted, image understanding is disabled.

### Provider quick reference

| Provider  | API endpoint                                               | Buy credits                                                                          | Manage API keys                                                               | Available models                                                                  |
| --------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| OpenAI    | `https://api.openai.com/v1`                                | [OpenAI Billing](https://platform.openai.com/settings/organization/billing/overview) | [OpenAI API keys](https://platform.openai.com/settings/organization/api-keys) | [OpenAI models](https://developers.openai.com/api/docs/models)                    |
| Anthropic | `https://api.anthropic.com/v1`                             | [Anthropic Billing](https://platform.claude.com/settings/billing)                    | [Anthropic API keys](https://platform.claude.com/settings/keys)               | [Claude models](https://platform.claude.com/docs/en/about-claude/models/overview) |
| Google    | `https://generativelanguage.googleapis.com/v1beta/openai/` | [Google AI Billing](https://aistudio.google.com/billing)                             | [Google API keys](https://aistudio.google.com/api-keys)                       | [Gemini models](https://ai.google.dev/gemini-api/docs/models)                     |
| Ollama    | `http://localhost:11434`                                   | free (local)                                                                         | n/a                                                                           | [Ollama model library](https://ollama.com/library)                                |

### OpenAI

```env
COPILOTJ_LLM_MODEL=gpt-5.4
COPILOTJ_LLM_API_KEY=sk-proj-xxxxxxxx

# Optional: vision model (can reuse the same key)
COPILOTJ_VLM_MODEL=gpt-5.4
COPILOTJ_VLM_API_KEY=sk-proj-xxxxxxxx
```

### Anthropic (Claude)

```env
COPILOTJ_LLM_MODEL=claude-sonnet-4-6
COPILOTJ_LLM_API_KEY=sk-ant-api03-xxxxxxxx

# Optional: vision model (can reuse the same key)
COPILOTJ_VLM_MODEL=claude-sonnet-4-6
COPILOTJ_VLM_API_KEY=sk-ant-api03-xxxxxxxx
```

### Google Gemini

```env
COPILOTJ_LLM_MODEL=gemini-2.5-flash
COPILOTJ_LLM_API_KEY=AIza-xxxxxxxx

# Optional: vision model (can reuse the same key)
COPILOTJ_VLM_MODEL=gemini-2.5-flash
COPILOTJ_VLM_API_KEY=AIza-xxxxxxxx
```

### Ollama (local, offline)

```env
COPILOTJ_LLM_MODEL=ollama/qwen3:30b
COPILOTJ_LLM_BASE_URL=http://localhost:11434
```

Note: Ollama models generally do not support image input. If image understanding is needed, configure
`COPILOTJ_VLM_MODEL` separately using a cloud provider.

### All configuration variables

| Variable                  | Description                                                                     |
| ------------------------- | ------------------------------------------------------------------------------- |
| `COPILOTJ_LLM_MODEL`      | Main LLM model name (required)                                                  |
| `COPILOTJ_LLM_API_KEY`    | API key for the main model (required)                                           |
| `COPILOTJ_LLM_BASE_URL`   | Override API endpoint for the main model                                        |
| `COPILOTJ_LLM_PROVIDER`   | Explicit LLM provider selection (optional; auto-detected if not set)            |
| `COPILOTJ_VLM_MODEL`      | Vision-language model name (optional)                                           |
| `COPILOTJ_VLM_API_KEY`    | API key for the VLM (falls back to `COPILOTJ_LLM_API_KEY`)                      |
| `COPILOTJ_VLM_BASE_URL`   | Override API endpoint for the VLM (falls back to `COPILOTJ_LLM_BASE_URL`)       |
| `COPILOTJ_VLM_PROVIDER`   | Explicit VLM provider (optional; falls back to `COPILOTJ_LLM_PROVIDER`)         |
| `COPILOTJ_LLM_PROXY`      | HTTP/HTTPS proxy for LLM outbound API requests                                  |
| `COPILOTJ_VLM_PROXY`      | HTTP/HTTPS proxy for VLM outbound requests (falls back to `COPILOTJ_LLM_PROXY`) |
| `COPILOTJ_TAVILY_API_KEY` | Tavily API key for live web search                                              |
| `COPILOTJ_KB_AUTOSAVE`    | Set `1` to auto-ingest dialog summaries into the knowledge bank                 |
| `COPILOTJ_DEV`            | Development mode (presence-based flag)                                          |
| `LANGFUSE_SECRET_KEY`     | Langfuse secret key for observability                                           |
| `LANGFUSE_PUBLIC_KEY`     | Langfuse public key for observability                                           |
| `LANGFUSE_HOST`           | Langfuse host URL (e.g. `https://us.cloud.langfuse.com`)                        |

A complete `.env.local` template:

```env
# LLM configuration (text-based reasoning) — choose one provider
COPILOTJ_LLM_MODEL=gpt-4.1
COPILOTJ_LLM_API_KEY=sk-xxxxxxxx
#COPILOTJ_LLM_BASE_URL=http://localhost:11434
#COPILOTJ_LLM_PROXY=http://PATH_TO_YOUR_PROXY
#COPILOTJ_LLM_PROVIDER=openai

# Vision-language model (image understanding) — optional, choose one provider
#COPILOTJ_VLM_MODEL=gemini-2.5-flash
#COPILOTJ_VLM_API_KEY=AI-xxxxxxxx
#COPILOTJ_VLM_PROVIDER=openai

# External search tool (web search)
#COPILOTJ_TAVILY_API_KEY=tvly-xxxxxxxxx

# Knowledge bank settings (1 to enable, 0 to disable)
COPILOTJ_KB_AUTOSAVE=0

## [Optional] Observability and tracing (Langfuse)
#LANGFUSE_SECRET_KEY=<secret key>
#LANGFUSE_PUBLIC_KEY=<public key>
#LANGFUSE_HOST="https://us.cloud.langfuse.com"
```

### Agent configuration (advanced)

CopilotJ uses a configurable multi-agent architecture. Agent configuration files are located in
`copilotj/multiagent/agent_configs/`. Each configuration file defines an agent's system prompt, role description,
and optional constraints.

**Customizing existing agents:** Modify prompt files in `agent_configs/` to adjust reasoning style, constrain
responsibilities, or tune domain-specific behavior. Changes take effect after restarting the server.

**Adding new agents:** Copy an existing configuration file, define a unique agent name and role, write a system prompt,
and register any custom tools.

## Testing

```bash
just test        # Run Python tests
just test-cov    # Run tests with coverage report (HTML + XML)
```

## Observability (optional)

CopilotJ integrates [Langfuse](https://langfuse.com/) for developers who want to observe and debug LLM usage
(API calls, prompts/responses, latency, caching). It is enabled automatically when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` environment variables are set. This is entirely optional — end users do not need it.

### Building the plugin from source

After cloning the repository, build the plugin with:

```bash
cd plugin
mvn clean package
mvn dependency:copy-dependencies -DoutputDirectory=target/deps
```

Locate the generated `.jar` file in `target/` and copy it along with all JARs from `target/deps/` into Fiji's `jars/` directory. Then restart Fiji.

Alternatively, Maven can install directly into a Fiji installation:

```bash
cd plugin && mvn clean install -Dscijava.app.directory=/path/to/Fiji
```

This copies the plugin JAR and all dependency JARs into the specified Fiji installation. Fiji comes bundled with many of CopilotJ's dependencies; the [SciJava infrastructure](https://github.com/scijava/scijava-maven-plugin/) keeps only the newer version of each dependency JAR.
