# User manual

## Installation

CopilotJ consists of three components:

- **CopilotJ Core Server** — a Python application that manages the language model, reasoning, and tool orchestration.
- **CopilotJ Bridge Plugin** - a Fiji plugin for communication between the Core Server and ImageJ.
- **CopilotJ Web Frontend** — a Vue-based web application that provides the conversational interface for users to interact with CopilotJ.

For the convenience of end users, the CopilotJ Bridge plugin supports a **managed server** mode where the Core Server is installed, configured, and launched automatically in the background.
This provides a seamless experience with minimal setup. For advanced users, an **external server** mode allows connecting to a manually managed Core Server instance, for example running on a remote machine or in a Docker container.
We also provide a hosted version of the web frontend for users who prefer not to run it locally, accessible at [copilotj.chat](https://copilotj.chat).

**System requirements**:

- **Operating systems**: macOS, Linux, Windows
- **[Fiji](https://fiji.sc/#download)**: Stable and Latest versions are both supported
- **Hardware:**
  - RAM: `>= 16 GB` (`32 GB` recommended)
  - GPU: optional, only required for deep-learning models
- **Network:** required for downloading the Python environment on first use and for LLM API access

### Install the CopilotJ Bridge plugin

**Download prebuilt JAR files**:

Download the precompiled JAR files for the CopilotJ Bridge plugin:

- [`CopilotJBridge-0.1.0-SNAPSHOT.jar`](https://copilotj.cvcd.xyz/software/precompiled_plugin/CopilotJBridge-0.1.0-SNAPSHOT.jar)
- [`jackson-datatype-jsr310-2.16.1.jar`](https://repo1.maven.org/maven2/com/fasterxml/jackson/datatype/jackson-datatype-jsr310/2.16.1/jackson-datatype-jsr310-2.16.1.jar)
- [`Java-WebSocket-1.5.2.jar`](https://repo1.maven.org/maven2/org/java-websocket/Java-WebSocket/1.5.2/Java-WebSocket-1.5.2-sources.jar)

**Install into Fiji**:

Install the CopilotJ Bridge plugin by placing the required JAR files in the appropriate `jars/` directory. The exact steps may vary slightly depending on your operating system and Fiji installation.

1. **Windows:** Open the Fiji installation directory (e.g., `D:\Fiji.app\`).
2. **macOS:** Locate `Fiji.app` in Finder, then right-click and select **Show Package Contents**.
3. Copy the files `CopilotJBridge-0.1.0-SNAPSHOT.jar`, `jackson-datatype-jsr310-2.16.1.jar`, `Java-WebSocket-1.5.2.jar`, `appose-0.11.0.jar` and `groovy-4.0.18.jar` into `jars/`.

**Verify plugin installation**:

1. Restart Fiji.
2. Confirm that **Plugins -> CopilotJ** appears in the menu.
3. Click it and verify that the configuration dialog opens with **Managed Server** and **External Server** tabs.

### Install the Python environment

1. In the dialog's **Managed Server** tab, click **Install**. The plugin downloads Python, creates a virtual environment, and installs all dependencies automatically.
2. Wait for the installation to complete. This may take **5–10 minutes** depending on your network speed. Progress is shown in the **Progress Log** area.
3. Once the status shows **Ready**, the installation is complete and does not need to be repeated.

## Configuration

### A. Environment configuration

CopilotJ is configured through a local environment file named `.env.local`. In managed mode, this file is located in the CopilotJ home directory:

- **macOS / Linux:** `~/.local/state/copilotj/.env.local`
- **Windows:** `%LOCALAPPDATA%\copilotj\.env.local`

Create the file if it does not already exist. Sensitive information such as API keys must be stored locally and should never be committed to version control. After updating `.env.local`, restart the managed server from the plugin dialog for changes to take effect.

#### Background: models, providers, and API keys

CopilotJ requires at least one **language model** (LLM) to function. A language model is a remote AI service that understands and generates text; CopilotJ sends your instructions to the model, which reasons about what to do and orchestrates CopilotJ's tools accordingly.

Models are provided by **AI providers** — companies that operate the model servers. Each provider requires you to create an account and authenticate with an **API key**: a secret credential you include in your `.env.local`. Every request your session sends to a model runs on that provider's remote servers and is billed to your account in units called **tokens** (roughly corresponding to words). Most providers require you to add a payment method and purchase credits before API requests will succeed; a free or evaluation-tier account will typically return an error on the first request.

CopilotJ uses **two separate model slots**:

- **`COPILOTJ_MODEL`**: the main reasoning model, used for planning, tool orchestration, and conversation. This is the most important setting and must always be configured.
- **`COPILOTJ_VLM_MODEL`**: an optional vision-language model (VLM) used when CopilotJ needs to interpret image content directly. All current models from the three recommended providers (OpenAI, Anthropic, Google) support image input. If omitted, image understanding is disabled.

#### Provider quick reference

| Provider  | API endpoint                                               | Buy credits                                                                          | Manage API keys                                                               | Available models                                                                  |
| --------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| OpenAI    | `https://api.openai.com/v1`                                | [OpenAI Billing](https://platform.openai.com/settings/organization/billing/overview) | [OpenAI API keys](https://platform.openai.com/settings/organization/api-keys) | [OpenAI models](https://developers.openai.com/api/docs/models)                    |
| Anthropic | `https://api.anthropic.com/v1`                             | [Anthropic Billing](https://platform.claude.com/settings/billing)                    | [Anthropic API keys](https://platform.claude.com/settings/keys)               | [Claude models](https://platform.claude.com/docs/en/about-claude/models/overview) |
| Google    | `https://generativelanguage.googleapis.com/v1beta/openai/` | [Google AI Billing](https://aistudio.google.com/billing)                             | [Google API keys](https://aistudio.google.com/api-keys)                       | [Gemini models](https://ai.google.dev/gemini-api/docs/models)                     |
| Ollama    | `http://localhost:11434`                                   | free (local)                                                                         | n/a                                                                           | [Ollama model library](https://ollama.com/library)                                |

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

**Important caveat:** CopilotJ's agentic workflows require strong multi-step reasoning and reliable tool-calling. As of early 2026, locally available Ollama models — including large models — have not proven capable enough to reliably complete CopilotJ's workflows out of the box. They tend to fail at tool orchestration, lose context across steps, or produce malformed tool calls. Ollama support is functional, but results will be significantly worse than with a frontier cloud model. This is a limitation of current local model capability, not of CopilotJ itself. With careful model selection and prompt tuning it may be possible to get acceptable results; if you experiment, the most capable models for tool-use tasks as of early 2026 include **qwen3** (Alibaba), **gemma3** (Google), **llama3.3** (Meta), and **phi4** (Microsoft) — prefer the largest variant your hardware can run. You can check the [Ollama model library](https://ollama.com/library) for new releases.

**Prerequisites:** Before configuring CopilotJ to use an Ollama model, make sure Ollama itself is installed and running, and that you have pulled the desired model:

```bash
# Install Ollama from https://ollama.com if not already installed
ollama pull qwen3:30b   # or whichever model you want to use
```

Ollama support in CopilotJ also requires an additional Python package. In managed mode, this dependency is installed automatically. For external server setups, run `uv sync --group all` once before starting the server.

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

**`COPILOTJ_VLM_BASE_URL`**

Overrides the API endpoint for the VLM only. Some LLMs, especially local or smaller models, do not support image input. If you need vision features, configure `COPILOTJ_VLM_MODEL` and `COPILOTJ_VLM_API_KEY` with a model that supports image input, and set `COPILOTJ_VLM_BASE_URL` when that VLM is served by a different provider or endpoint from the main LLM. If omitted and no VLM-specific model/key is configured, CopilotJ reuses `COPILOTJ_BASE_URL` for vision tasks.

**`COPILOTJ_PROXY`**

Routes all outbound model API requests through an HTTP/HTTPS [proxy server](https://en.wikipedia.org/wiki/Proxy_server) — an intermediary between your machine and the internet. Commonly required in institutional or corporate networks where all traffic must pass through a central gateway. If you are connecting directly to the internet, you do not need this. Example value: `http://proxy.example.com:8080`.

**`COPILOTJ_TAVILY_API_KEY`**

Enables live web search during CopilotJ sessions via [Tavily](https://app.tavily.com/), a search API designed for use with language models. Because LLMs have a training cutoff and no built-in internet access, web search allows CopilotJ to look up current documentation, papers, or tool usage examples in real time. Without this key, CopilotJ relies only on what its model already knows. Obtain a key from the [Tavily dashboard](https://app.tavily.com/).

**`COPILOTJ_KB_AUTOSAVE`**

Controls CopilotJ's **knowledge bank** — a persistent store of summaries from past sessions. When set to `1`, CopilotJ automatically summarizes completed dialogues and saves them so that information from previous sessions can be recalled in future ones. Useful if you run many sessions and want CopilotJ to build up knowledge about your data, workflows, and preferences over time. Disabled (`0`) by default; summaries can also be saved manually from the chat interface.

**`LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_HOST`**

Enable execution tracing via [Langfuse](https://langfuse.com), an open-source observability platform for LLM applications. Tracing records a structured log of every model call, tool invocation, token count, latency, and cost for each session, viewable in the Langfuse dashboard. Useful for debugging unexpected agent behavior or understanding where tokens are being spent. Not needed for normal use. Obtain keys from the [Langfuse dashboard](https://cloud.langfuse.com/).

A complete `.env.local` template with all options:

```env
# LLM configuration (text-based reasoning) — choose one provider
COPILOTJ_MODEL=gpt-4.1
COPILOTJ_API_KEY=sk-xxxxxxxx
#COPILOTJ_BASE_URL=http://localhost:11434
#COPILOTJ_PROXY=http://PATH_TO_YOUR_PROXY

# Vision-language model (image understanding) — optional, choose one provider
#COPILOTJ_VLM_MODEL=gemini-2.5-flash
#COPILOTJ_VLM_API_KEY=AI-xxxxxxxx

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

   Changes take effect after restarting the managed server from the plugin dialog.

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

1. **Configure API keys** — Create a `.env.local` file in the CopilotJ data directory (see the [Configuration](#a-environment-configuration) section above) with your LLM provider credentials.

2. **Open the CopilotJ plugin dialog**
   - In Fiji, navigate to **Plugins -> CopilotJ**.
   - The dialog opens with the **Managed Server** tab selected.

3. **Start the server**
   - Click **Start**. The backend starts and connects automatically.
   - The server URL (e.g., `http://127.0.0.1:12345`) is shown next to the **Server** label once startup completes. The port is chosen automatically and persisted across restarts.
   - The environment is synced on each start, so updates are applied automatically.

4. **Open the web interface**
   - Open a web browser and navigate to [copilotj.chat](https://copilotj.chat), then click **Chat**.
   - The web frontend connects to the managed server automatically.

5. **Open an image for analysis**
   - Use Fiji to open the image or image stack to be analyzed.
   - Example datasets used in the study can be found in supplementary data for testing and reproducibility.

> **Advanced:** To connect to an externally running server instead, switch to the **External Server** tab in the dialog, enter the server URL, and click **(Re)Connect**. For advanced usage such as Docker deployment, local development servers, and building the plugin from source, see the [development documentation](https://github.com/neurogeom/CopilotJ/blob/main/DEVELOPMENT.md).

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

All files generated during execution are stored in a designated temporary working directory, referred to as the `<copilotj_home>/temp` folder.

The `<copilotj_home>/temp` folder serves as a centralized location for artifacts generated during an analysis session, including:

- processed images and intermediate image results
- measurement tables such as CSV files
- generated reports and logs
- saved workflows in Markdown or JSON
- optional ZIP bundles containing workflows, data, and metadata

The temporary folder is managed by the CopilotJ core server and updates in real time as workflows execute. Unless explicitly cleaned or overwritten, it preserves outputs from the current session for debugging, validation, and reproducibility.
