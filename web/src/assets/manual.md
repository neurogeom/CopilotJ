# User manual

## Installation

### A. System requirements

- **Operating systems:** macOS, Linux, Windows
- **Software:**
  - [Fiji](https://fiji.sc/#download): Stable and Latest versions are both supported
  - [Python 3.12](https://www.python.org/downloads/): not required when using Docker
- **Hardware:**
  - RAM: `>= 16 GB` (`32 GB` recommended)
  - GPU: optional, only required for deep-learning models

### B. Installation overview

CopilotJ consists of three required components:

- **CopilotJ Bridge Plugin** for Fiji
- **CopilotJ Core Server**
- **CopilotJ frontend server**

### C. Install the CopilotJ Bridge plugin

#### Option C1. download prebuilt JAR files

Download the precompiled JAR files for the CopilotJ Bridge plugin:

- [`CopilotJBridge-0.1.0-SNAPSHOT.jar`](https://copilotj.cvcd.xyz/software/precompiled_plugin/CopilotJBridge-0.1.0-SNAPSHOT.jar)
- [`jackson-datatype-jsr310-2.16.1.jar`](https://repo1.maven.org/maven2/com/fasterxml/jackson/datatype/jackson-datatype-jsr310/2.16.1/jackson-datatype-jsr310-2.16.1.jar)
- [`Java-WebSocket-1.5.2.jar`](https://repo1.maven.org/maven2/org/java-websocket/Java-WebSocket/1.5.2/Java-WebSocket-1.5.2-sources.jar.sha1)

**Install into Fiji:**

Install the CopilotJ Bridge plugin by placing the required JAR files in the appropriate `jars/` directory. The exact steps may vary slightly depending on your operating system and Fiji installation.

1. **Windows:** Open the Fiji installation directory (e.g., `D:\Fiji.app\`).
2. **macOS:** Locate `Fiji.app` in Finder, then right-click and select **Show Package Contents**.
3. Copy the files `CopilotJBridge-0.1.0-SNAPSHOT.jar`, `jackson-datatype-jsr310-2.16.1.jar` and `Java-WebSocket-1.5.2.jar` into `jars/`.

**Verify plugin installation**

1. Restart Fiji.
2. Confirm that **Plugins -> CopilotJ** appears in the menu.
3. Click it and verify that the configuration window opens successfully.

#### Option C2. Build from source

After cloning the repository, build the plugin with:

```bash
cd plugin
mvn clean package
mvn dependency:copy-dependencies -DoutputDirectory=target/deps
```

After building, locate the generated `.jar` file in the `target/` directory, then follow the same installation steps in **Option C1** to copy the built plugin JAR and required dependency JARs into Fiji.

### D. Install the CopilotJ core server

#### Option D1. Docker installation (recommended)

CopilotJ can be deployed with [Docker](https://docker.com/) and [Docker Compose](https://docs.docker.com/compose/).
This approach is recommended for most users, as it launches the backend, frontend, and reverse proxy in a unified environment.

Recommended workflow:

1. Clone the repository from GitHub.

   ```bash
   git clone https://github.com/neurogeom/CopilotJ.git
   cd CopilotJ
   ```

2. Create your local configuration file. Refer to the section below for detailed instructions.
3. Build the required images locally.

   ```bash
   # Build the images locally
   docker compose build
   ```

4. Start the full stack with `docker compose up`.

   ```bash
   # Start the full stack
   docker compose up -d
   ```

   The default Compose setup exposes the web interface on `http://localhost:8786`.

   If GPU passthrough is available on your machine and Docker is configured with NVIDIA support, start the GPU-enabled stack with:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
   ```

   To rebuild the images after updating the source code, run:

   ```bash
   docker compose up -d --build
   ```

#### Option D2. local installation

1. Clone the repository from GitHub.

   ```bash
   git clone https://github.com/neurogeom/CopilotJ.git
   cd CopilotJ
   ```

2. Install dependencies with [uv](https://docs.astral.sh/uv/).

   ```bash
   uv sync
   ```

3. Create your local configuration file. Refer to the section below for detailed instructions.
4. Start the core server.

   ```bash
   uv run python -m copilotj.server --host 127.0.0.1 --port 8786
   ```

By default, the local development server listens at `http://127.0.0.1:8786`. If you change the host or port, make sure to update all related configurations accordingly so that other components can still communicate with the server.

### E. Install the CopilotJ frontend server

If you use the recommended Docker Compose deployment (**Option D1**), the CopilotJ frontend server starts automatically together with the backend and the Nginx proxy.
No separate frontend installation is required.

For source-based development (**Option D2**), the frontend can also be run independently from the `web/` directory.

1. Install [Node.js](https://nodejs.org/en/download) 22 or later, then enable `pnpm` through Corepack.

   ```bash
   corepack enable pnpm
   ```

2. Install frontend dependencies.

   ```bash
   cd web
   pnpm install
   ```

3. Start the Vite development server.

   ```bash
   pnpm run dev
   ```

   If the backend runs on a different origin, add `VITE_API_BASE_URL` to `web/.env.development.local`.

   ```env
   VITE_API_BASE_URL=http://127.0.0.1:8786
   ```

4. (Advanced) For production deployment, we strongly recommend using the provided Docker-based setup, which includes a preconfigured frontend build and reverse proxy.

   If you need to integrate CopilotJ into your own infrastructure (e.g., behind a custom reverse proxy), you can refer to the `web/Dockerfile`, which demonstrates the standard production workflow: building the Vue application with `pnpm run build` and serving the generated `dist/` directory using an `nginx:alpine` image.

## Configuration

### A. Environment configuration

CopilotJ is configured through a local environment file named `.env.local` in the root directory of the repository. For Docker Compose deployments, this file is mounted into the container at runtime. Sensitive information such as API keys must be stored locally and should never be committed to version control.

#### Background: models, providers, and API keys

CopilotJ requires at least one **language model** (LLM) to function. A language model is a remote AI service that understands and generates text; CopilotJ sends your instructions to the model, which reasons about what to do and orchestrates CopilotJ's tools accordingly.

Models are provided by **AI providers** — companies that operate the model servers. Each provider requires you to create an account and authenticate with an **API key**: a secret credential you include in your `.env.local`. Every request your session sends to a model runs on that provider's remote servers and is billed to your account in units called **tokens** (roughly corresponding to words). Most providers require you to add a payment method and purchase credits before API requests will succeed; a free or evaluation-tier account will typically return an error on the first request.

CopilotJ uses **two separate model slots**:

- **`COPILOTJ_MODEL`**: the main reasoning model, used for planning, tool orchestration, and conversation. This is the most important setting and must always be configured.
- **`COPILOTJ_VLM_MODEL`**: an optional vision-language model (VLM) used when CopilotJ needs to interpret image content directly. All current models from the three recommended providers (OpenAI, Anthropic, Google) support image input. If omitted, image understanding is disabled.

After updating `.env.local`, restart the CopilotJ core server for changes to take effect.

#### Provider quick reference

| Provider | API endpoint | Buy credits | Manage API keys | Available models |
|---|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | [OpenAI Billing](https://platform.openai.com/settings/organization/billing/overview) | [OpenAI API keys](https://platform.openai.com/settings/organization/api-keys) | [OpenAI models](https://developers.openai.com/api/docs/models) |
| Anthropic | `https://api.anthropic.com/v1` | [Anthropic Billing](https://platform.claude.com/settings/billing) | [Anthropic API keys](https://platform.claude.com/settings/keys) | [Claude models](https://platform.claude.com/docs/en/about-claude/models/overview) |
| Google | `https://generativelanguage.googleapis.com/v1beta/openai/` | [Google AI Billing](https://aistudio.google.com/billing) | [Google API keys](https://aistudio.google.com/api-keys) | [Gemini models](https://ai.google.dev/gemini-api/docs/models) |
| Ollama | `http://localhost:11434` | free (local) | n/a | [Ollama model library](https://ollama.com/library) |

#### Option A1: OpenAI

OpenAI's GPT models work reliably with CopilotJ. `gpt-5.4` is the current flagship; `gpt-5.4-mini` costs less but may produce weaker results on complex workflows.

1. Create an account at [platform.openai.com](https://platform.openai.com).
2. Add credits via [OpenAI Billing](https://platform.openai.com/settings/organization/billing/overview).
3. Create an API key at [OpenAI API keys](https://platform.openai.com/settings/organization/api-keys).

```env
COPILOTJ_MODEL=gpt-5.4
COPILOTJ_API_KEY=sk-proj-xxxxxxxx

# Optional: vision model (can reuse the same key)
COPILOTJ_VLM_MODEL=gpt-5.4
COPILOTJ_VLM_API_KEY=sk-proj-xxxxxxxx
```

#### Option A2: Anthropic (Claude)

Claude models are recommended for their strong multi-step reasoning and tool use. `claude-sonnet-4-6` offers a good balance of capability and cost; `claude-opus-4-6` is the most capable option.

1. Create an account at [console.anthropic.com](https://console.anthropic.com).
2. Add credits via [Anthropic Billing](https://platform.claude.com/settings/billing).
3. Create an API key at [Anthropic API keys](https://platform.claude.com/settings/keys).

```env
COPILOTJ_MODEL=claude-sonnet-4-6
COPILOTJ_API_KEY=sk-ant-api03-xxxxxxxx

# Optional: vision model (can reuse the same key)
COPILOTJ_VLM_MODEL=claude-sonnet-4-6
COPILOTJ_VLM_API_KEY=sk-ant-api03-xxxxxxxx
```

#### Option A3: Google Gemini

Gemini models are well-supported and competitively priced. `gemini-2.5-flash` is fast and inexpensive; `gemini-2.5-pro` provides stronger reasoning.

1. Visit [Google AI Studio](https://aistudio.google.com) and sign in with a Google account.
2. Enable billing if required via [Google AI Billing](https://aistudio.google.com/billing).
3. Create an API key at [Google API keys](https://aistudio.google.com/api-keys).

```env
COPILOTJ_MODEL=gemini-2.5-flash
COPILOTJ_API_KEY=AIza-xxxxxxxx

# Optional: vision model (can reuse the same key)
COPILOTJ_VLM_MODEL=gemini-2.5-flash
COPILOTJ_VLM_API_KEY=AIza-xxxxxxxx
```

#### Option A4: Ollama (local, offline)

[Ollama](https://ollama.com) runs models locally on your own hardware, with no data sent to external servers and no per-request cost. CopilotJ supports Ollama models via the `ollama/` prefix.

**Important caveat:** CopilotJ's agentic workflows require strong multi-step reasoning and reliable tool-calling. As of early 2026, locally available Ollama models — including large models — have not proven capable enough to reliably complete CopilotJ's workflows out of the box. They tend to fail at tool orchestration, lose context across steps, or produce malformed tool calls. Ollama support is functional, but results will be significantly worse than with a frontier cloud model. This is a limitation of current local model capability, not of CopilotJ itself. With careful model selection and prompt tuning it may be possible to get acceptable results; if you experiment, the most capable models for tool-use tasks as of early 2026   include **qwen3** (Alibaba), **gemma3** (Google), **llama3.3** (Meta), and **phi4** (Microsoft) — prefer the largest variant your hardware can run. You can check the [Ollama model library](https://ollama.com/library) for new releases.

**Prerequisites:** Before configuring CopilotJ to use an Ollama model, make sure Ollama itself is installed and running, and that you have pulled the desired model:

```bash
# Install Ollama from https://ollama.com if not already installed
ollama pull qwen3:30b   # or whichever model you want to use
```

Ollama support in CopilotJ also requires an additional Python package not installed by default. Run this once before starting the server:

```bash
uv sync --group all
```

Then configure `.env.local`. No API key is needed:

```env
COPILOTJ_MODEL=ollama/qwen3:30b
COPILOTJ_BASE_URL=http://localhost:11434
```

Note: Ollama models generally do not support image input. If image understanding is needed, configure `COPILOTJ_VLM_MODEL` separately using a cloud provider from the options above.

#### Other configuration variables

The following variables are optional and relate to specific CopilotJ features.

**`COPILOTJ_BASE_URL`**

Overrides the default API endpoint for the main model. Use this when connecting to a non-default server, such as a local Ollama instance (`http://localhost:11434`) or a custom inference server. When not set, each provider uses its standard public endpoint. See the provider-specific sections above for when this is needed.

**`COPILOTJ_PROXY`**

Routes all outbound model API requests through an HTTP/HTTPS [proxy server](https://en.wikipedia.org/wiki/Proxy_server) — an intermediary between your machine and the internet. Commonly required in institutional or corporate networks where all traffic must pass through a central gateway. If you are connecting directly to the internet, you do not need this. Example value: `http://proxy.example.com:8080`.

**`COPILOTJ_RAG_API_KEY`**

Enables [retrieval-augmented generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation), a technique that allows the model to search a document collection before generating a response, rather than relying solely on what it learned during training. This can ground CopilotJ's answers in specific literature, protocols, or your own notes. RAG requires a separate **embedding model** to convert text into numerical vectors for similarity search; `COPILOTJ_RAG_API_KEY` is the credential for that embedding service. If you are already using OpenAI for `COPILOTJ_MODEL`, the same key often works here. For most users getting started, RAG is not required.

**`COPILOTJ_TAVILY_API_KEY`**

Enables live web search during CopilotJ sessions via [Tavily](https://app.tavily.com/), a search API designed for use with language models. Because LLMs have a training cutoff and no built-in internet access, web search allows CopilotJ to look up current documentation, papers, or tool usage examples in real time. Without this key, CopilotJ relies only on what its model already knows. Obtain a key from the [Tavily dashboard](https://app.tavily.com/).

**`COPILOTJ_KB_AUTOSAVE`**

Controls CopilotJ's **knowledge bank** — a persistent store of summaries from past sessions. When set to `1`, CopilotJ automatically summarizes completed dialogues and saves them so that information from previous sessions can be recalled in future ones. Useful if you run many sessions and want CopilotJ to build up knowledge about your data, workflows, and preferences over time. Disabled (`0`) by default; summaries can also be saved manually from the chat interface.

**`LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_HOST`**

Enable execution tracing via [Langfuse](https://langfuse.com), an open-source observability platform for LLM applications. Tracing records a structured log of every model call, tool invocation, token count, latency, and cost for each session, viewable in the Langfuse dashboard. Useful for debugging unexpected agent behavior or understanding where tokens are being spent. Not needed for normal use. Obtain keys from the [Langfuse dashboard](https://cloud.langfuse.com/).

A complete `.env.local` template with all options:

```env
# Set DEV mode for better logging
COPILOTJ_DEV=1

# LLM configuration (text-based reasoning) — choose one provider
COPILOTJ_MODEL=gpt-4.1
COPILOTJ_API_KEY=sk-xxxxxxxx
#COPILOTJ_BASE_URL=http://localhost:11434
#COPILOTJ_PROXY=http://PATH_TO_YOUR_PROXY

# Vision-language model (image understanding) — optional, choose one provider
#COPILOTJ_VLM_MODEL=gemini-2.5-flash
#COPILOTJ_VLM_API_KEY=AI-xxxxxxxx

# Retrieval-augmented generation (RAG)
#COPILOTJ_RAG_API_KEY=sk-xxxxxxxx

# External search tool (web search)
#COPILOTJ_TAVILY_API_KEY=tvly-xxxxxxxxx

# Knowledge bank settings (1 to enable, 0 to disable)
COPILOTJ_KB_AUTOSAVE=0

## [Optional] Observability and tracing (Langfuse)
#LANGFUSE_SECRET_KEY=<secret key>
#LANGFUSE_PUBLIC_KEY=<public key>
#LANGFUSE_HOST="https://us.cloud.langfuse.com"
```

### B. Agent settings (advanced, optional)

CopilotJ uses a configurable multi-agent architecture. Each agent is defined by prompt templates and behavioral parameters stored in the core server repository.

Agent configuration files are located in `copilotj/multiagent/agent_configs/`

Each configuration file defines an agent's system prompt, role description, and optional constraints. These prompt templates determine how an agent reasons, interacts with tools, and communicates with other agents.

1. **Customizing existing agents**

   Advanced users may modify prompt files in `agent_configs/` to:
   - adjust an agent's reasoning style or verbosity
   - constrain or expand an agent's responsibilities
   - tune domain-specific behavior, such as bioimage-analysis rules

   Changes take effect after restarting the CopilotJ core server.

2. **Adding new agents**

   New agents can be introduced by creating a new configuration file that follows the existing template.

   Typical workflow:
   1. Copy an existing agent configuration file.
   2. Define a unique agent name and role description.
   3. Write a system prompt that specifies the agent's responsibilities and boundaries.
   4. Develop and register any custom tools required by the new agent.

   This makes CopilotJ extensible without modifying the core execution logic.

## Using CopilotJ

### A. Starting CopilotJ

To use CopilotJ, first start the core server and then connect from the Fiji interface.

1. **Start the CopilotJ core server**
   - Ensure the core server is running locally or via Docker.
   - In standard local setups, the default server endpoint is sufficient.

2. **Launch Fiji and connect through the plugin**
   - In Fiji, navigate to **Plugins -> CopilotJ -> Connect**.
   - In typical local setups, the default server URL does not need modification.
     If you have changed the server endpoint, enter only the root URL (do not append additional paths).
   - Click **Connect** to establish the connection between the plugin and the core server.

3. **Open the CopilotJ web interface**
   - Open a web browser and navigate to the deployed CopilotJ frontend: if using Docker (**Option D1**), this will be `http://127.0.0.1:8786` by default&mdash;or if running from source (**Option D2**), the default will be `http://127.0.0.1:5173`.
   - This web interface provides the chat-based frontend for interacting with CopilotJ.
   - Once the connection is established, you can start a conversational session and issue analysis instructions by clicking the "Chat" button at the top right of the web interface.

4. **Open an image for analysis**
   - Use Fiji to open the image or image stack to be analyzed.
   - Example datasets used in the study can be found in supplementary data for testing and reproducibility.

### B. Issuing analysis instructions

Users interact with CopilotJ through natural-language instructions, for example:

```text
Segment nuclei and measure mean intensity in channel 2.
```

CopilotJ will automatically:

1. Interpret the user request.
2. Construct an analysis workflow.
3. Execute ImageJ and Python-based operations.
4. Return processed images, measurements, and logs.

### C. Workflows

CopilotJ treats each analysis session as a structured workflow that can be executed, saved, and shared across datasets and users.

1. **Saving workflows**

   Completed workflows can be saved directly from the conversational interface.

   Example user command:

   ```text
   Save dialog 1 as a workflow named: XXX
   ```

   Saved workflows record:
   - all analysis steps and execution order
   - parameters and tool versions
   - agent decisions and execution logs

   Supported save formats:
   - **Markdown**: human-readable documentation of the analysis
   - **JSON**: machine-readable workflow specification for re-execution

2. **Workflow execution**

   Workflows can be queried and executed with natural-language commands, for example:

   ```text
   Show my workflows
   I want to execute the workflow 1
   I want to execute the workflow named: XXX
   ```

   CopilotJ translates the selected workflow into an ordered sequence of operations, including ImageJ commands, Python scripts, and external tool calls.
   - Execution status and intermediate results are streamed to the web interface in real time.
   - ImageJ state, including open images, ROIs, and errors, is continuously monitored.
   - If a failure occurs, CopilotJ may automatically revise parameters or execution order and retry the workflow.

3. **Sharing and reusing workflows**

   Saved workflows can be shared with collaborators or re-applied to new datasets.

   Supported export formats:
   - **Markdown workflows** for supplementary documentation or lab protocols
   - **JSON workflows** that can be reloaded into CopilotJ, for example under `<project_dir>/temp/workflows`
   - **ZIP archives** bundling workflows with images, scripts, and metadata

   This workflow-centric design supports reproducibility, transparency, and efficient reuse of analysis pipelines.

### D. File locations and temporary folder

All files generated during execution are stored in a designated temporary working directory, referred to as the `<project_dir>/temp` folder.

The `<project_dir>/temp` folder serves as a centralized location for artifacts generated during an analysis session, including:

- processed images and intermediate image results
- measurement tables such as CSV files
- generated reports and logs
- saved workflows in Markdown or JSON
- optional ZIP bundles containing workflows, data, and metadata

The temporary folder is managed by the CopilotJ core server and updates in real time as workflows execute. Unless explicitly cleaned or overwritten, it preserves outputs from the current session for debugging, validation, and reproducibility.
