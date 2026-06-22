# User manual

## Installation

CopilotJ consists of three components:

- **CopilotJ Core Server** — a Python application that manages the language model, reasoning, and tool orchestration.
- **CopilotJ Bridge Plugin** - a Fiji plugin for communication between the Core Server and ImageJ.
- **CopilotJ Web Frontend** — a Vue-based web application that provides the conversational interface for users to interact with CopilotJ.

For the convenience of end users, the CopilotJ Bridge plugin supports a **managed server** mode where the Core Server is installed, configured, and launched automatically in the background.
This provides a seamless experience with minimal setup. For advanced users, a **standalone server** mode allows connecting to a manually managed Core Server instance, for example running on a remote machine or in a Docker container.
We also provide a hosted version of the web frontend for users who prefer not to run it locally, accessible at [copilotj.chat](https://copilotj.chat).

**System requirements**:

- **Operating systems**: macOS, Linux, Windows
- **[Fiji](https://fiji.sc/#download)**: Stable and Latest versions are both supported
- **Hardware:**
  - RAM: `>= 16 GB` (`32 GB` recommended)
  - GPU: optional, only required for deep-learning models
- **Network:** required for downloading the Python environment on first use and for LLM API access

### Install the CopilotJ Bridge plugin

The ImageJ Updater is the recommended way to install and update the CopilotJ plugin. Simply add the CopilotJ update site (https://sites.imagej.net/CopilotJ/), and the plugin will be installed automatically.

<details>
<summary>Step-by-step instructions for using the ImageJ Updater</summary>

1. Open Fiji and go to `Help` -> `Update...` to open the Fiji Updater.
2. Click `Manage Update Sites` in the lower-left corner.
3. Click “Add Unlisted Site”, add a new update site named CopilotJ with the URL https://sites.imagej.net/CopilotJ/, and then make sure the checkbox on the left is selected.
4. Click `Apply and Close` to close the `Manage Update Sites` window.
5. Click `Apply Changes` in the main `Fiji Updater` window to install the plugin.
6. Restart Fiji, you should now see _Plugins -> CopilotJ_ in the menu.

</details>

### Install the Python environment

1. In the dialog's **Managed Server** tab, click **Install**. The plugin downloads Python, creates a virtual environment, and installs all dependencies automatically.
2. Wait for the installation to complete. This may take **5–10 minutes** depending on your network speed. Progress is shown in the **Progress Log** area.
3. Once the status shows **Ready**, the installation is complete and does not need to be repeated.

## Getting started

1. **Open the CopilotJ plugin dialog**
   - In Fiji, navigate to **Plugins -> CopilotJ**.
   - The dialog opens with the **Managed Server** tab selected.

2. **Start the server**
   - Click **Start**. The backend starts and connects automatically.
   - The server URL (e.g., `http://127.0.0.1:12345`) is shown next to the **Server** label once startup completes. The port is chosen automatically and persisted across restarts.

3. **Open the web interface**
   - Click **Open copilotj.chat** in the dialog's **Managed Server** tab to open the chat interface in your default browser.
   - On first use, the setup wizard will guide you through connecting to the server and configuring your model and API key (see the [First-time setup wizard](#first-time-setup-wizard) below).
   - The web frontend connects to the managed server automatically.

4. **Open an image for analysis**
   - Use Fiji to open the image or image stack to be analyzed.
   - Example datasets used in the study can be found in supplementary data for testing and reproducibility.

> **Advanced:** To connect to a standalone server instead, switch to the **Standalone Server** tab in the dialog, enter the server URL, and click **(Re)Connect**. For advanced usage such as Docker deployment, local development servers, and building the plugin from source, see the [development documentation](https://github.com/neurogeom/CopilotJ/blob/main/DEVELOPMENT.md).

### Models, providers, and API keys

CopilotJ requires at least one **language model** (LLM) to function. A language model is a remote AI service that understands and generates text; CopilotJ sends your instructions to the model, which reasons about what to do and orchestrates CopilotJ's tools accordingly.

Models are provided by **AI providers** — companies that operate the model servers. Each provider requires you to create an account and authenticate with an **API key**: a secret credential that authorizes requests on your behalf. Every request your session sends to a model runs on that provider's remote servers and is billed to your account in units called **tokens** (roughly corresponding to words). Most providers require you to add a payment method and purchase credits before API requests will succeed; a free or evaluation-tier account will typically return an error on the first request.

### Provider quick reference

::: tabs

=== OpenAI

OpenAI's GPT models work reliably with CopilotJ. `gpt-5.4` is the current flagship; `gpt-5.4-mini` costs less but may produce weaker results on complex workflows.

1. Create an account at [platform.openai.com](https://platform.openai.com).
2. Add credits via [OpenAI Billing](https://platform.openai.com/settings/organization/billing/overview).
3. Create an API key at [OpenAI API keys](https://platform.openai.com/settings/organization/api-keys).

=== Google Gemini

Gemini models are well-supported and competitively priced. `gemini-2.5-flash` is fast and inexpensive; `gemini-2.5-pro` provides stronger reasoning.

1. Visit [Google AI Studio](https://aistudio.google.com) and sign in with a Google account.
2. Enable billing if required via [Google AI Billing](https://aistudio.google.com/billing).
3. Create an API key at [Google API keys](https://aistudio.google.com/api-keys).

=== Anthropic (Claude)

Claude models are recommended for their strong multi-step reasoning and tool use. `claude-sonnet-4-6` offers a good balance of capability and cost; `claude-opus-4-6` is the most capable option.

1. Create an account at [console.anthropic.com](https://console.anthropic.com).
2. Add credits via [Anthropic Billing](https://platform.claude.com/settings/billing).
3. Create an API key at [Anthropic API keys](https://platform.claude.com/settings/keys).

=== DeepSeek

[DeepSeek](https://platform.deepseek.com) offers capable reasoning models at low cost through an OpenAI-compatible API.

1. Create an account and add credits at [platform.deepseek.com](https://platform.deepseek.com).
2. Create an API key at [DeepSeek API keys](https://platform.deepseek.com/api_keys).

In CopilotJ, select **DeepSeek** as the provider (the base URL `https://api.deepseek.com` is filled in automatically), enter your API key, and choose a model such as `deepseek-chat` (general) or `deepseek-reasoner` (reasoning).

=== OpenRouter

[OpenRouter](https://openrouter.ai) is a unified API gateway: one account, one API key, and one OpenAI-compatible endpoint give access to models from many providers — including OpenAI, Anthropic, and Google. It is convenient when you want to compare models from different providers without maintaining separate accounts, and it accepts prepaid credits with a low minimum (currently $10).

1. Create an account at [openrouter.ai](https://openrouter.ai).
2. Add credits via [OpenRouter Credits](https://openrouter.ai/credits).
3. Create an API key at [OpenRouter Keys](https://openrouter.ai/keys).

To use OpenRouter with CopilotJ, select **OpenRouter** as the provider, enter your OpenRouter API key, and type the model slug from the [OpenRouter model library](https://openrouter.ai/models) — these are prefixed with the provider, for example `openai/gpt-5.2`, `google/gemini-2.5-pro` or `anthropic/claude-sonnet-4.5`.

=== Ollama (local, offline)

[Ollama](https://ollama.com) runs models locally on your own hardware, with no data sent to external servers and no per-request cost.

**Model recommendations:** When using Ollama with CopilotJ, we recommend choosing models with strong multi-step reasoning, vision capability, and reliable tool use.
As of early 2026, capable options for agentic tasks include _gemma4_ (Google), _qwen3.6_ (Alibaba), and _kimi-k2.6_ (Kimi).
In general, prefer the largest variant your hardware can comfortably run, as larger models tend to perform better on complex workflows.
You can also check the [Ollama model library](https://ollama.com/library) for newer releases.
If a smaller model lacks sufficient reasoning, vision, or tool-calling capability, it may not be able to reliably support complex agentic applications.

**Prerequisites:** Before using an Ollama model with CopilotJ, make sure Ollama is installed and running, and that you have pulled the desired model:

```bash
# Install Ollama from https://ollama.com if not already installed
ollama pull qwen3:30b   # or whichever model you want to use
```

Note: Ollama models generally do not support image input. If image understanding is needed, configure a separate vision model using a cloud provider from the options above.

=== OpenAI-compatible server

If your model is served by any other OpenAI-compatible endpoint — for example a local inference server such as [LM Studio](https://lmstudio.ai), [vLLM](https://docs.vllm.ai), [llama.cpp](https://github.com/ggml-org/llama.cpp), or [Ollama's OpenAI-compatible API](https://github.com/ollama/ollama/blob/main/docs/openai.md) — you can connect CopilotJ to it directly.

To do so, select **OpenAI** as the provider, set the **Base URL** to your server's endpoint (for example, `http://localhost:1234/v1` for LM Studio), enter whatever API key the server requires (use a placeholder such as `sk-1234` for servers that do not enforce authentication), and type the model name exactly as the server exposes it.

Consult your server's documentation for the correct base URL and the list of available model names.

:::

### First-time setup wizard

The first time you open the CopilotJ chat page, a setup wizard appears. It has six steps — click each to expand:

<details id="wizard-notice">
<summary>Step 1 — Notice</summary>

Before anything else, CopilotJ shows a **Privacy & Data Handling notice**. It explains that your task text — and, if you enable Vision, image snapshots of the ImageJ interface — is sent to your model provider, and that how that data is handled is governed by the provider's own policies.

- Tick _I have read and understood the above information and agree to use CopilotJ under these conditions_ to proceed. You cannot continue until you agree.
- Optionally tick _I choose to enable Vision support_ to turn on image understanding. This is **recommended** when your model supports it, and you can also change it later. Enabling Vision here adds the Vision step (Step 4) to the wizard; if you leave it off, that step is skipped.

</details>

<details id="wizard-connection">
<summary>Step 2 — Connection</summary>

Enter the URL of your CopilotJ server and click **Connect**.

- In managed mode, this is the server URL shown in the Fiji plugin dialog (e.g. `http://127.0.0.1:12345`).
- If you are using a standalone server, enter its URL (e.g. `http://localhost:8786`).

The wizard tests the connection and shows a success or error message. You must connect successfully before proceeding.

</details>

<details id="wizard-model">
<summary>Step 3 — Model</summary>

First choose a **Provider** from the dropdown — OpenAI, Anthropic, Google Gemini, DeepSeek, OpenRouter, SiliconFlow, Ollama (local), or OpenAI-compatible. The fields then adapt to the provider:

- **Cloud providers** (OpenAI, Anthropic, Google Gemini, DeepSeek, SiliconFlow, OpenRouter) — a **Model** autocomplete plus an **API Key**, with an **Advanced settings** collapsible for a custom **Base URL** (pre-filled per provider).
- **Ollama** — a **Base URL** (default `http://localhost:11434`) and a **Model** list loaded live from your Ollama server; no API key is required.
- **OpenAI-compatible** — a **Base URL**, a **Model** name you type, and an **API Key**.

See the [Provider quick reference](#provider-quick-reference) below for how to obtain each provider's API key.

</details>

<details id="wizard-vision">
<summary>Step 4 — Vision</summary>

This step only appears if you enabled Vision in the **Notice** step (Step 1). CopilotJ **auto-detects** whether your main model supports image input, using a built-in model-capability database (for Ollama, a known list that includes llava, moondream, minicpm-v, llama3.2-vision, gemma3, qwen2.5-vl, and pixtral):

- If your model supports vision, **Use main model for vision** is enabled and on — vision tasks reuse your main model and API key.
- If it does not, the toggle is disabled and you are asked to configure a **separate vision model** (provider, model, and API key), or to leave vision off.

If you don't need image understanding, simply leave vision disabled.

</details>

<details id="wizard-preferences">
<summary>Step 5 — Preferences</summary>

Optional settings that you can skip or configure later:

- **HTTP Proxy** — Route API requests through a proxy server. Commonly needed in institutional or corporate networks (e.g. `http://proxy.example.com:8080`).
- **Tavily API Key** — Enable live web search during sessions via [Tavily](https://app.tavily.com/). Without this, CopilotJ relies only on what its model already knows.
- **Auto-save to Knowledge Bank** — When enabled (off by default), CopilotJ automatically summarizes completed dialogues and saves them so information from past sessions can be recalled in future ones.
- **Auto-scroll to Bottom** — Automatically scroll the chat to the latest message (on by default).

You can click **Skip** to proceed without changing any advanced settings.

</details>

<details id="wizard-finish">
<summary>Step 6 — Finish</summary>

Review your server, model, vision, and preference settings. Once you click **Start Using CopilotJ**, the configuration is saved and you are ready to begin. You can revise any of these settings later from the ⚙️ icon in the chat interface.

</details>

## Using CopilotJ

With setup complete, the sections below describe everyday use.

### Issuing analysis instructions

Users interact with CopilotJ through natural-language instructions, for example:

```text
Segment nuclei and measure mean intensity in channel 2.
```

CopilotJ will automatically:

1. Interpret the user request.
2. Construct an analysis workflow.
3. Execute ImageJ and Python-based operations.
4. Return processed images, measurements, and logs.

### Workflows

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

### File locations and temporary folder

All files generated during execution are stored in a designated temporary working directory, referred to as the `<copilotj_home>/temp` folder.

> **Tip:** Click **Open Resources** in the dialog's **Managed Server** tab to open `<copilotj_home>` (the CopilotJ home directory) directly in your file manager — the `temp/` folder lives inside it.

The `<copilotj_home>/temp` folder serves as a centralized location for artifacts generated during an analysis session, including:

- processed images and intermediate image results
- measurement tables such as CSV files
- generated reports and logs
- saved workflows in Markdown or JSON
- optional ZIP bundles containing workflows, data, and metadata

The temporary folder is managed by the CopilotJ core server and updates in real time as workflows execute. Unless explicitly cleaned or overwritten, it preserves outputs from the current session for debugging, validation, and reproducibility.

## FAQ

<details id="faq-jar-conflict">
<summary>After installing or updating other plugins, CopilotJ fails to load</summary>

Fiji loads **all plugins through a single shared set of Java libraries** (the contents of its `jars/` folder). There is no isolation between plugins: if CopilotJ depends on one version of a shared library (for example the JSON or WebSocket library it uses) but another plugin you installed ships an **older or incompatible** version of that same library, only one copy can win. Whichever version ends up on disk, some plugin is left calling a method that no longer exists, and at runtime this surfaces as a **`NoSuchMethodError`**, **`NoClassDefFoundError`**, **`IncompatibleClassChangeError`**, or simply a plugin that will not start. The ImageJ community refers to this as _version skew_.

It is almost always caused by one of the following:

- A third-party update site or plugin pinned an older version of a shared library that overwrote the newer one CopilotJ needs.
- A JAR was dropped manually into the `jars/` or `plugins/` folder.
- Fiji has not been updated for a long time, so the locally installed copies drifted out of sync with the central update site.

**Fix 1 — recommended: start from a clean, fully-updated Fiji.** Version skew is a property of a particular installation, so the most reliable cure is to remove it:

1. Download a fresh Fiji from [fiji.sc](https://fiji.sc/).
2. Start it and run `Help` -> `Update...` to bring everything to the latest version.
3. Re-add the CopilotJ update site (`https://sites.imagej.net/CopilotJ/`) — and any other update sites you need — then `Apply Changes` and restart Fiji.
4. Confirm CopilotJ loads **before** adding more plugins. If you re-enable several third-party update sites, enable them **one at a time**, testing after each, so you can tell which site reintroduces the conflict.

**Fix 2 — only if Fix 1 is not possible: remove or correct the offending JAR by hand.**

1. Find which JAR is actually being loaded. In Fiji run `Plugins` -> `Utilities` -> `Find Jar For Class...` and type the class named in the error; it reports the JAR that supplies it — often a stale copy that another plugin placed under `jars/` or `plugins/<some-plugin>/`.
2. Quit Fiji and **rename** the conflicting older JAR (renaming rather than deleting lets you undo the change), then restart Fiji and run `Help` -> `Update...` so the Fiji Updater restores the correct version from the central site.
3. If the Updater does not own that file, either disable the third-party update site that ships it, or fall back to a clean Fiji (Fix 1).

> Avoid simply _replacing_ a JAR with an arbitrary version downloaded from elsewhere — choosing a version by hand tends to create a new conflict for the next plugin. Starting fresh (Fix 1) is the safe, repeatable fix.

See the ImageJ guide on [`NoSuchMethodError` / `NoClassDefFoundError` ("version skew")](https://imagej.net/learn/troubleshooting) for background.

</details>

<details id="faq-rate-limited-429">
<summary>I see a "Rate limited — retrying" message, or a 429 / TPM error from the model</summary>

LLM providers cap how much you can send them at once: a **requests-per-minute** limit and, more often the real bottleneck, a **tokens-per-minute (TPM)** limit on the total text and image data moving in and out. When you exceed that cap — common during peak hours, with a long conversation history, or on a low-tier plan — the provider replies with an HTTP **429 "Too Many Requests"** (often reported as a TPM error). This is temporary and is not a bug in CopilotJ.

**What CopilotJ does automatically.** When the model returns a 429 before it has produced any output, CopilotJ backs off and retries on its own — **up to five attempts** — waiting a little longer between each try and honoring the provider's `Retry-After` hint when one is given. While it waits, an amber **"Rate limited — retrying (n/5, waiting Xs)…"** bar with a live countdown appears under the message. In most cases you do not need to do anything — just let the countdown run and the request resumes on its own.

**If the retries are exhausted and it still fails:**

- **Wait and resend.** Peak-hour limits usually clear within seconds to a few minutes.
- **Reduce the load per turn.** A very long thread or several open images raise the token count per request; starting a fresh thread or closing images you no longer need lowers your TPM usage.
- **Switch to a model or provider with a higher limit.** Tiers and models differ widely in their TPM allowances — see the [Provider quick reference](#provider-quick-reference). A local [Ollama](https://ollama.com) model, which runs on your own hardware, has no provider rate limit at all.
- **Raise your account quota.** Most providers let you increase your usage tier / TPM from the account dashboard once billing is configured.

When a model error does occur, CopilotJ surfaces it directly in the chat so you can see exactly what went wrong.

</details>

<details id="faq-open-chat-button">
<summary>The "Open copilotj.chat" button doesn't do anything, or the chat page won't load</summary>

The **Open Chat** button launches the hosted CopilotJ web frontend at [copilotj.chat/#/chat](https://copilotj.chat/#/chat) in your default browser. A few things can get in the way:

- **Nothing happens when you click it.** Your platform may not support opening a browser from Fiji (for example, a headless or some Linux environments). Just open [https://copilotj.chat/#/chat](https://copilotj.chat/#/chat) manually in any browser.
- **The page opens but shows a connection error.** The button opens the _frontend_, which still needs your local server running. In the **Managed Server** tab, make sure the server status is **Running**, then use the setup wizard (or the **Preferences** in the chat UI) to point the frontend at the server URL shown next to **Server** (for example, `http://127.0.0.1:12345`).

</details>

<details id="faq-open-resources">
<summary>The "Open Resources" button doesn't open the folder</summary>

The **Open Resources** button opens the CopilotJ home directory in your file manager:

- **Windows:** `%LOCALAPPDATA%\copilotj\.env.local`
- **macOS / Linux:** `~/.local/state/copilotj/.env.local`

If the button doesn't work:

- **Nothing happens, or you see "Opening a folder is not supported".** Your platform may not support this from Fiji. Open the folder manually at the path above.
- **The folder is empty, or there is no `temp/` folder.** Subfolders such as `temp/`, `assets/`, and `knowledge_bank/` are created when the Python environment is installed and a session runs. Click **Install** (then **Start**) in the **Managed Server** tab first.
- **You see a "Failed to open resource directory" error.** Open the path above manually in your file manager.

</details>

<details id="faq-general-ai">
<summary>Can I use general AI applications to interact with Fiji?</summary>

Yes. The CopilotJ Bridge plugin exposes an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server. Any MCP-compatible AI client — for example Claude Code, Codex, or other tools that speak MCP — can connect to your running Fiji instance and drive ImageJ operations directly, without going through the CopilotJ web frontend.

</details>

<details id="why-is-my-mcp-server-not-available">
<summary>Why is my MCP server not available?</summary>

MCP only works with **Fiji-Latest**. The MCP libraries require Java 17 or newer; **Fiji-Latest** ships Java 21, while **Fiji-Stable** still ships Java 8. On Fiji-Stable the MCP tab in the CopilotJ dialog shows a "not available" notice, while the rest of CopilotJ keeps working normally. See [fiji#413](https://github.com/fiji/fiji/issues/413) for background on Fiji's Java versions.

To use MCP, upgrade to **Fiji-Latest** — download it from [fiji.sc](https://fiji.sc/).

</details>

<details id="faq-run-workflow-cli">
<summary>Can I run saved workflows without LLM participation?</summary>

Yes. In CopilotJ, the saved workflow is designed with this principle in mind: each workflow is a structured, executable record that may contain both Fiji macro commands and Python code, saved in a JSON format together with the necessary execution metadata. During replay, CopilotJ executes these commands AS-IS; once the workflow is launched, the LLM is no longer involved, and no additional LLM credits are consumed.

Saved workflows can also be run from the terminal using an interactive script. First, start the CopilotJ bridge server in standalone mode and the ImageJ/Fiji plugin, then run either of the following:

```bash
sh scripts/run-workflow.sh
```

or:

```bash
just run-workflow
```

The script asks which workflow to run, the input file or folder, the output folder, and any optional `name=value` parameters.

</details>
