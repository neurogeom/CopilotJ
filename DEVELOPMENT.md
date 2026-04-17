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

## Testing

```bash
just test        # Run Python tests
just test-cov    # Run tests with coverage report (HTML + XML)
```

## Observability (optional)

CopilotJ integrates [Langfuse](https://langfuse.com/) for developers who want to observe and debug LLM usage
(API calls, prompts/responses, latency, caching). It is enabled automatically when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` environment variables are set. This is entirely optional — end users do not need it.

### Manual Plugin Deployment

If you prefer to manually deploy the plugin to your Fiji installation:

1. **Build and install the plugin and its dependencies:**
   Navigate to the `plugin` directory and build the package:

   ```bash
   cd plugin && mvn clean install -Dscijava.app.directory=/path/to/Fiji
   ```

   where `/path/to/Fiji` is the file path to your Fiji installation folder.
   This will create the plugin JAR file (e.g., `CopilotJBridge-1.0.0.jar`)
   in the `plugin/target/` directory, then copy it along with all of its
   dependency JAR files into the specified Fiji installation.

   Note that Fiji comes bundled with many of CopilotJ's dependencies, but the
   [SciJava infrastructure](https://github.com/scijava/scijava-maven-plugin/)
   takes care to keep only the newer version of each dependency JAR when
   copying them.

2. **Restart Fiji:**
   After the build is complete, restart Fiji for the changes to take effect.
