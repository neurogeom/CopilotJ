# User manual

## Installation

### A. System requirements

- **Operating systems:** macOS, Linux, Windows
- **Software:**
  - [ImageJ / Fiji](https://imagej.net/downloads): latest stable version recommended
  - [Python 3.12](https://www.python.org/downloads/): not required when using Docker
- **Hardware:**
  - RAM: `>= 16 GB` (`32 GB` recommended)
  - GPU: optional, only required for deep-learning models

### B. Installation overview

CopilotJ consists of three required components:

- **CopilotJ Bridge Plugin** for ImageJ / Fiji
- **CopilotJ Core Server**
- **CopilotJ frontend server**

### C. Install the CopilotJ Bridge plugin

#### Option C1. download prebuilt JAR files

Download the precompiled JAR files for the CopilotJ Bridge plugin:

- [`CopilotJBridge-0.1.0-SNAPSHOT.jar`](https://copilotj.cvcd.xyz/software/precompiled_plugin/CopilotJBridge-0.1.0-SNAPSHOT.jar)
- [`jackson-datatype-jsr310-2.9.8.jar`](https://copilotj.cvcd.xyz/software/precompiled_plugin/jackson-datatype-jsr310-2.9.8.jar)
- [`Java-WebSocket-1.5.2.jar`](https://copilotj.cvcd.xyz/software/precompiled_plugin/Java-WebSocket-1.5.2.jar)

**Install into ImageJ / Fiji:**

Install the CopilotJ Bridge plugin by placing the required JAR files in the appropriate `plugins/` and `jars/` directories. The exact steps may vary slightly depending on your operating system and ImageJ / Fiji installation.

1. **Windows:** Open the ImageJ / Fiji installation directory (e.g., `D:\Fiji.app\`).
2. **macOS:** Locate `Fiji.app` in Finder, then right-click and select **Show Package Contents**.
3. Copy the plugin JAR `CopilotJBridge-0.1.0-SNAPSHOT.jar` into `plugins/`.
4. Copy the dependency JARs `jackson-datatype-jsr310-2.9.8.jar` and `Java-WebSocket-1.5.2.jar` into `jars/`.

**Verify plugin installation**

1. Restart ImageJ / Fiji.
2. Confirm that **Plugins -> CopilotJ** appears in the menu.
3. Click it and verify that the configuration window opens successfully.

#### Option C2. Build from source

After cloning the repository, build the plugin with:

```bash
cd plugin
mvn clean package
mvn dependency:copy-dependencies -DoutputDirectory=target/deps
```

After building, locate the generated `.jar` file in the `target/` directory, then follow the same installation steps in **Option C1** to copy the built plugin JAR and required dependency JARs into ImageJ / Fiji.

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

CopilotJ is configured through a local environment file in the root directory of the repository.

It is recommended to create a file named `.env.local`.
For Docker Compose deployments, it will be mounted into the container as `.env.local` at runtime.

Customize this file with your own API keys and model preferences.
Sensitive information such as API keys must be stored locally and should never be committed to version control.

Create the local environment file with content such as:

```env
# Set DEV mode for better logging
COPILOTJ_DEV=1

# LLM configuration (text-based reasoning)
COPILOTJ_MODEL=gpt-4.1-mini
COPILOTJ_API_KEY=sk-xxxxxxxx
#COPILOTJ_BASE_URL=http://localhost:11434
#COPILOTJ_PROXY=http://PATH_TO_YOUR_PROXY

# Vision-language model (image understanding)
COPILOTJ_VLM_MODEL=gemini-2.5-flash
COPILOTJ_VLM_API_KEY=AI-xxxxxxxx

# Retrieval-augmented generation (RAG)
COPILOTJ_RAG_API_KEY=sk-xxxxxxxx

# External search tool (web search)
COPILOTJ_TAVILY_API_KEY=tvly-xxxxxxxxx

# Knowledge bank settings (1 to enable, 0 to disable)
COPILOTJ_KB_AUTOSAVE=0

## [Optional] Observability and tracing (Langfuse)
#LANGFUSE_SECRET_KEY=<secret key>
#LANGFUSE_PUBLIC_KEY=<public key>
#LANGFUSE_HOST="https://us.cloud.langfuse.com"
```

1. **How to obtain API keys**
   - **`COPILOTJ_API_KEY`**: Obtain it from the provider of the selected LLM, such as OpenAI or Google Gemini.
   - **`COPILOTJ_VLM_API_KEY`**: Obtain it from the selected vision-language model provider.
   - **`COPILOTJ_RAG_API_KEY`**: Obtain it from the embedding or retrieval service provider used in your setup.
   - **`COPILOTJ_TAVILY_API_KEY`**: Obtain it from the Tavily dashboard after registering an account.
   - **`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`**: Obtain them from the Langfuse dashboard when creating a project.

   Useful links: [OpenAI API keys](https://platform.openai.com/api-keys), [Google Gemini API keys](https://aistudio.google.com/app/apikey), [Tavily dashboard](https://app.tavily.com/) and [Langfuse dashboard](https://cloud.langfuse.com/auth/sign-in)

2. **Purpose of each configuration**
   - **`COPILOTJ_MODEL`**: Selects the text-based LLM used for reasoning, planning, and tool orchestration.
   - **`COPILOTJ_VLM_MODEL`**: Selects the vision-language model used for image understanding and multimodal reasoning.
   - **`COPILOTJ_RAG_API_KEY`**: Enables retrieval-augmented generation through external knowledge services.
   - **`COPILOTJ_PROXY`**: Routes outbound network traffic through a proxy when required.
   - **`COPILOTJ_BASE_URL`**: Specifies the endpoint for local or remote model backends, such as Ollama or SiliconFlow.
   - **`COPILOTJ_KB_AUTOSAVE`**: Controls whether dialogue summaries are automatically stored in the CopilotJ knowledge bank.
   - **Langfuse keys**: Enable execution tracing, debugging, and performance monitoring.

   Multiple model backends are supported. For example:

   ```env
   COPILOTJ_MODEL=gemini-2.5-pro
   COPILOTJ_MODEL=ollama/llama3.1:latest
   COPILOTJ_MODEL=gpt-4.1
   ```

   After updating the local environment file, restart the CopilotJ core server for the changes to take effect.

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

To use CopilotJ, first start the core server and then connect from the ImageJ / Fiji interface.

1. **Start the CopilotJ core server**
   - Ensure the core server is running locally or via Docker.
   - In standard local setups, the default server endpoint is sufficient.

2. **Launch ImageJ / Fiji and connect through the plugin**
   - In ImageJ / Fiji, navigate to **Plugins -> CopilotJ -> Connect**.
   - In typical local setups, the default server URL does not need modification.
     If you have changed the server endpoint, enter only the root URL (do not append additional paths).
   - Click **Connect** to establish the connection between the plugin and the core server.

3. **Open the CopilotJ web interface**
   - Open a web browser and navigate to the deployed CopilotJ frontend, for example `http://127.0.0.1:8786`.
   - The web interface provides the chat-based frontend for interacting with CopilotJ.
   - Once the connection is established, you can start a conversational session and issue analysis instructions.

4. **Open an image for analysis**
   - Use ImageJ / Fiji to open the image or image stack to be analyzed.
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
