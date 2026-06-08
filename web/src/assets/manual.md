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

When you first open the CopilotJ web interface, a setup wizard guides you through connecting to your server and configuring the AI model. You can revisit these settings at any time from the ⚙️ icon in the chat interface.

### Background: models, providers, and API keys

CopilotJ requires at least one **language model** (LLM) to function. A language model is a remote AI service that understands and generates text; CopilotJ sends your instructions to the model, which reasons about what to do and orchestrates CopilotJ's tools accordingly.

Models are provided by **AI providers** — companies that operate the model servers. Each provider requires you to create an account and authenticate with an **API key**: a secret credential that authorizes requests on your behalf. Every request your session sends to a model runs on that provider's remote servers and is billed to your account in units called **tokens** (roughly corresponding to words). Most providers require you to add a payment method and purchase credits before API requests will succeed; a free or evaluation-tier account will typically return an error on the first request.

### Provider quick reference

| Provider  | Get API key                                                                   | Buy credits                                                                          | Recommended models                                               |
| --------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| OpenAI    | [OpenAI API keys](https://platform.openai.com/settings/organization/api-keys) | [OpenAI Billing](https://platform.openai.com/settings/organization/billing/overview) | `gpt-5.4` (flagship), `gpt-5.4-mini` (lower cost)                |
| Anthropic | [Anthropic API keys](https://platform.claude.com/settings/keys)               | [Anthropic Billing](https://platform.claude.com/settings/billing)                    | `claude-sonnet-4-6` (balanced), `claude-opus-4-6` (most capable) |
| Google    | [Google API keys](https://aistudio.google.com/api-keys)                       | [Google AI Billing](https://aistudio.google.com/billing)                             | `gemini-2.5-flash` (fast), `gemini-2.5-pro` (stronger reasoning) |
| Ollama    | n/a (local)                                                                   | free                                                                                 | See [Ollama model library](https://ollama.com/library)           |

#### OpenAI

OpenAI's GPT models work reliably with CopilotJ. `gpt-5.4` is the current flagship; `gpt-5.4-mini` costs less but may produce weaker results on complex workflows.

1. Create an account at [platform.openai.com](https://platform.openai.com).
2. Add credits via [OpenAI Billing](https://platform.openai.com/settings/organization/billing/overview).
3. Create an API key at [OpenAI API keys](https://platform.openai.com/settings/organization/api-keys).

#### Anthropic (Claude)

Claude models are recommended for their strong multi-step reasoning and tool use. `claude-sonnet-4-6` offers a good balance of capability and cost; `claude-opus-4-6` is the most capable option.

1. Create an account at [console.anthropic.com](https://console.anthropic.com).
2. Add credits via [Anthropic Billing](https://platform.claude.com/settings/billing).
3. Create an API key at [Anthropic API keys](https://platform.claude.com/settings/keys).

#### Google Gemini

Gemini models are well-supported and competitively priced. `gemini-2.5-flash` is fast and inexpensive; `gemini-2.5-pro` provides stronger reasoning.

1. Visit [Google AI Studio](https://aistudio.google.com) and sign in with a Google account.
2. Enable billing if required via [Google AI Billing](https://aistudio.google.com/billing).
3. Create an API key at [Google API keys](https://aistudio.google.com/api-keys).

#### Ollama (local, offline)

[Ollama](https://ollama.com) runs models locally on your own hardware, with no data sent to external servers and no per-request cost.

**Model recommendations:** When using Ollama with CopilotJ, we recommend choosing models with strong multi-step reasoning, vision capability, and reliable tool use.
As of early 2026, capable options for agentic tasks include *gemma4* (Google), *qwen3.6* (Alibaba), and *kimi-k2.6* (Kimi).
In general, prefer the largest variant your hardware can comfortably run, as larger models tend to perform better on complex workflows.
You can also check the [Ollama model library](https://ollama.com/library) for newer releases.
If a smaller model lacks sufficient reasoning, vision, or tool-calling capability, it may not be able to reliably support complex agentic applications.

**Prerequisites:** Before using an Ollama model with CopilotJ, make sure Ollama is installed and running, and that you have pulled the desired model:

```bash
# Install Ollama from https://ollama.com if not already installed
ollama pull qwen3:30b   # or whichever model you want to use
```

Note: Ollama models generally do not support image input. If image understanding is needed, configure a separate vision model using a cloud provider from the options above.

### First-time setup wizard

The first time you open the CopilotJ chat page, a setup wizard appears. It has five steps:

#### Step 1 — Connection

Enter the URL of your CopilotJ server and click **Connect**.

- In managed mode, this is the server URL shown in the Fiji plugin dialog (e.g. `http://127.0.0.1:12345`).
- If you are using an external server, enter its URL (e.g. `http://localhost:8786`).

The wizard tests the connection and shows a success or error message. You must connect successfully before proceeding.

#### Step 2 — Model

Choose the primary language model for your conversations.

- **Model** — Select from the autocomplete list, which groups models by provider (Anthropic, OpenAI, Google, Ollama). You can also type a custom model name.
- **API Key** — Enter your provider API key. This field is hidden automatically for Ollama models (which don't require a key).
- **Base URL** — Override the default API endpoint if needed (e.g. for a custom inference server). Most users can leave this empty.

#### Step 3 — Vision

Configure the vision model used for image analysis tasks.

By default, **Use main model for vision** is enabled — vision tasks will use the same model and API key you configured in Step 2. This works well because current models from OpenAI, Anthropic, and Google all support image input.

If you prefer a separate model for vision (e.g. a cheaper model for image tasks), disable the toggle and configure:

- **Model** — Select a vision-capable model.
- **API Key** — Enter its API key.
- **Base URL** — Override the endpoint if needed.

#### Step 4 — Advanced (optional)

Optional settings that you can skip or configure later:

- **HTTP Proxy** — Route API requests through a proxy server. Commonly needed in institutional or corporate networks (e.g. `http://proxy.example.com:8080`).
- **Tavily API Key** — Enable live web search during sessions via [Tavily](https://app.tavily.com/). Without this, CopilotJ relies only on what its model already knows.
- **Auto-save to Knowledge Bank** — When enabled, CopilotJ automatically summarizes completed dialogues and saves them so information from past sessions can be recalled in future ones.
- **Auto-scroll to Bottom** — Automatically scroll the chat to the latest message.

You can click **Skip** to proceed without changing any advanced settings.

#### Step 5 — Review

Review all your settings before finishing. Once you click **Start Using CopilotJ**, the configuration is saved and you are ready to begin.

### Changing settings after setup

You can update your configuration at any time by clicking the ⚙️ (settings) icon in the chat interface. The settings panel has four tabs:

- **Model** — Change the default model, API key, or base URL.
- **Vision** — Update the vision model configuration.
- **Integrations** — Modify the proxy or Tavily API key.
- **Preferences** — Adjust the API server URL, knowledge bank autosave, and auto-scroll behavior.

### Per-thread model override

Each conversation thread can use a different model from the default. Click the floating model button in the chat toolbar to open the thread-specific model dialog, where you can select a different model or API key for that thread only.

## Using CopilotJ

### A. Starting CopilotJ

1. **Open the CopilotJ plugin dialog**
   - In Fiji, navigate to **Plugins -> CopilotJ**.
   - The dialog opens with the **Managed Server** tab selected.

2. **Start the server**
   - Click **Start**. The backend starts and connects automatically.
   - The server URL (e.g., `http://127.0.0.1:12345`) is shown next to the **Server** label once startup completes. The port is chosen automatically and persisted across restarts.

3. **Open the web interface**
   - Open a web browser and navigate to [copilotj.chat](https://copilotj.chat), then click **Chat**.
   - On first use, the setup wizard will guide you through connecting to the server and configuring your model and API key (see the [Configuration](#configuration) section above).
   - The web frontend connects to the managed server automatically.

4. **Open an image for analysis**
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

   Saved workflows can also be run from terminal using an interactive script. First, start the CopilotJ bridge server and ImageJ/Fiji plugin, then run either of the following:

   ```bash
   sh scripts/run-workflow.sh
   ```

   or:

   ```bash
   just run-workflow
   ```

   The script asks which workflow to run, the input file or folder, the output folder, and any optional `name=value` parameters.

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
