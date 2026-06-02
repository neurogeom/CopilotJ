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
- **Node.js 22+** with [pnpm](https://pnpm.io/installation)
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
