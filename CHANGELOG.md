# Changelog

All notable changes to CopilotJ are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-30

### Added

- **Managed Server mode**: A brand-new **Managed Server** mode lets the CopilotJ Bridge plugin install, configure, and launch the Core Server automatically in the background. Getting CopilotJ running is now just a few clicks from inside Fiji — no manual Python server to set up, dramatically simplifying installation.
  - In the plugin dialog's **Managed Server** tab, **Install** downloads Python, creates a virtual environment, and installs all dependencies automatically; **Start** launches the server.
  - Open the chat UI straight from the dialog (**Open copilotj.chat**) and reach your CopilotJ home folder via **Open Resources**.
  - Removing the managed environment offers a **keep-data vs delete-all** choice, so you can uninstall without losing your data.
- **Install the plugin from the ImageJ Updater**: The CopilotJ Bridge plugin is now published on a Fiji/ImageJ update site, so it installs and updates through the ImageJ Updater — no manual JAR placement.
  - Add the CopilotJ update site (`https://sites.imagej.net/CopilotJ/`) under `Help → Update… → Manage Update Sites`, then `Apply Changes`.
  - Smaller download — StarDist model weights and other heavy assets are fetched on demand instead of being bundled in the plugin.
- **Web frontend & onboarding**: [copilotj.chat](https://copilotj.chat) is now the recommended hosted frontend (no need to run it yourself), and a new **first-time setup wizard** lets users start using it directly in the browser and point it at their local CopilotJ server.
  - Use CopilotJ directly at [copilotj.chat](https://copilotj.chat) — the hosted frontend is now the recommended path instead of self-hosting.
  - A new **setup wizard** walks first-time users through configuring the server connection and their model provider/keys; the wizard and Settings dialogs share unified components.
- **Agent runtime & reliability**: Reliability improvements to the multi-agent run loop.
  - **Auto-retry on 429 rate limits** with visible frontend feedback, so transient rate limits no longer fail a run.
  - **Stop now aborts in-flight runs promptly**, including mid-retry.
  - **Expanded provider support**: Added support for more providers, including Anthropic via the native SDK, Google Gemini via the native SDK, OpenRouter with OpenAI-compatible APIs, and more. Also introduced a unified model discovery API and a revamped provider settings UI.
  - Stronger **knowledge-bank execution** validation and error checks.
  - **Script/macro execution timeout** is enforced via a new `ScriptRequest` timeout field — protection against runaway scripts.
- **Privacy & consent**:
  - Added a **User Agreement** and **Vision notice** so users consent before any vision/VLM feature is used.
  - A `vision_enabled` flag disables vision features; the preference no longer resets to the server default on reload.
- **Java-based MCP server**: CopilotJ can expose itself as an MCP server so external MCP clients (e.g. Claude Desktop, Cline) can drive ImageJ; loaded via an isolated child-first ClassLoader, ensuring compatibility with both Java 8 and Java 17+ Fiji.
- **Batch QC**: new batch quality-control capability integrated with the workflow tools.
- **Usage logging**: per-request token usage (prompt, completion, and cache-hit tokens) is now logged for every provider, so operators can monitor consumption and verify that prompt caching is actually hitting.

### Changed

- **Prompt caching**: enabled for Anthropic and Qwen to reduce latency and cost.
- **Docs**: the user manual was restructured with troubleshooting FAQs and is rendered at build time via a Vite plugin.

### Fixed

- **Notable fixes**: Fiji-Quit hang from a leaked EDT context ClassLoader; emoji-encoding crash on GBK stdout in managed mode; broken MCP-unavailable manual deep-link.

### Security

- Merged a large batch of dependency updates (~50) across the Python, Java, JavaScript, and CI stacks — including security-relevant libraries (tornado, cryptography, aiohttp, urllib3) — to keep CopilotJ on current, patched versions.

## [1.0.0] - 2026-03-20

### Added

First public release of CopilotJ — a conversational multi-agent system that turns natural-language requests into executable, verifiable bioimage-analysis workflows by coordinating ImageJ/Fiji, Python scientific libraries, and deep-learning models.

- **Three components**: a **web frontend** (browser-based chat UI that streams responses from the backend over a REST API); a **multi-agent backend** (orchestrates agents and tools, integrates Python and deep-learning environments, and connects to multiple LLM providers); and the **CopilotJ Bridge** — an ImageJ/Fiji plugin that talks to the backend over a bidirectional WebSocket.
- **Installation**: install the plugin by downloading the prebuilt JARs (or building from source) and copying them into Fiji's `plugins/` and `jars/` directories; run the backend stack with Docker Compose.
- The Docker image is distributed separately due to size; see the user manual.
