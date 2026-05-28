# Appose Integration: Managed Python Process for CopilotJ

## Overview

This document describes a design for embedding the CopilotJ Python server as a
managed subprocess within the Fiji/ImageJ Java plugin using the
[Appose](https://github.com/apposed/appose-java) inter-process communication
framework and [uv](https://docs.astral.sh/uv/) for Python environment
management.

The goal is to **eliminate the requirement for users to manually install and
run the Python server**. Instead, the Fiji plugin creates the environment,
starts the server, and manages its lifecycle automatically.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Fiji / ImageJ (JVM)                                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  CopilotJBridgeService                               │    │
│  │                                                      │    │
│  │  ┌─────────────┐    ┌──────────────────────────┐     │    │
│  │  │ EnvManager  │    │  Appose Service          │     │    │
│  │  │             │    │  (process lifecycle)      │     │    │
│  │  │ • uv setup  │───>│  • start / stop / kill   │     │    │
│  │  │ • deps sync │    │  • port discovery (task) │     │    │
│  │  └─────────────┘    └──────────────────────────┘     │    │
│  │                                                      │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  Connection (WebSocket Client)                │    │    │
│  │  │  • existing bridge protocol                   │    │    │
│  │  │  • bidirectional event forwarding             │    │    │
│  │  │  • connects to dynamic port from Appose task  │    │    │
│  │  └────────────┬─────────────────────────────────┘    │    │
│  └───────────────┼──────────────────────────────────────┘    │
└──────────────────┼───────────────────────────────────────────┘
                   │ WebSocket (ws://127.0.0.1:<port>/api/plugins)
                   │   port discovered via Appose task response
┌──────────────────┼───────────────────────────────────────────┐
│  Python Process  │                                           │
│  (managed by Appose)                                         │
│                                                              │
│  ┌───────────────┴─────────────────────────────────────┐    │
│  │  CopilotJ Server (aiohttp)                          │    │
│  │  • binds port 0 → OS assigns dynamic port           │    │
│  │  • returns port to Java via Appose task response    │    │
│  │  • REST API (/api/threads, /api/...)                │    │
│  │  • WebSocket endpoint (/api/plugins)                │    │
│  │  • Health check via HTTP GET /api/ping              │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Key design principle**: Appose manages the **process lifecycle** (create
environment, start process, stop process) and provides the **port discovery**
channel (Python binds port 0, returns the assigned port to Java via Appose task
response). The existing **WebSocket bridge** handles all **data communication**.
Health checking uses HTTP `GET /api/ping` on the discovered port. These concerns
are cleanly separated.

## 1. Environment Creation with uv

### Directory layout

```
Fiji.app/                          (Fiji-Stable)
  or Fiji/                         (Fiji-Latest)
  copilotj_<arch>/                 ← root directory for CopilotJ env
    .venv/                         ← uv-managed virtual environment
      bin/python                   ← Python 3.12+ interpreter
      lib/python3.12/
        site-packages/             ← all dependencies
    .env                           ← user configuration
    uv.lock                        ← lockfile for reproducibility
```

The arch suffix matches the convention used by SAMJ's Appose integration
(`appose_<arch>`). The directory is resolved at runtime:

```java
String arch = PlatformDetection.getArch();
// Rosetta on Apple Silicon: use arm64
if (PlatformDetection.isMacOS() && PlatformDetection.isUsingRosseta()) {
    arch = PlatformDetection.ARCH_ARM64;
}
File envRoot = new File(fijiDir, "copilotj_" + arch);
```

### Installation steps

```java
public class CopilotJEnvManager {

    private final File envRoot;

    public void installEverything() throws IOException {
        if (!checkUvInstalled())     installUv();
        if (!checkVenvCreated())     createVenv();
        if (!checkDepsInstalled())   installDeps();
        if (!checkConfigPresent())   writeDefaultConfig();
    }

    /** Run an external process, draining stdout/stderr to prevent deadlock. */
    private int runProcess(ProcessBuilder pb) throws IOException, InterruptedException {
        Process p = pb.start();
        // Drain stdout and stderr in background threads to prevent deadlock
        // when child output exceeds the OS pipe buffer (~64KB).
        Thread stdoutDrainer = new Thread(() -> {
            try (var is = p.getInputStream()) { is.transferTo(OutputStream.nullOutputStream()); }
            catch (IOException ignored) {}
        }, "uv-stdout-drain");
        Thread stderrDrainer = new Thread(() -> {
            try (var is = p.getErrorStream()) { is.transferTo(OutputStream.nullOutputStream()); }
            catch (IOException ignored) {}
        }, "uv-stderr-drain");
        stdoutDrainer.start();
        stderrDrainer.start();
        int exitCode = p.waitFor();
        stdoutDrainer.join();
        stderrDrainer.join();
        return exitCode;
    }
}
```

**Step 1 — Download uv**: Download the `uv` binary for the current platform
into `copilotj_<arch>/bin/` (or `copilotj_<arch>/` on Windows). uv is a single
static binary (~20 MB), making distribution simple.

Official install URLs:
- Linux/macOS: `https://github.com/astral-sh/uv/releases/latest/download/uv-<platform>.tar.gz`
- Windows: `https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip`

**Step 2 — Create virtual environment**:

```java
// uv venv .venv --python 3.12
ProcessBuilder pb = new ProcessBuilder(
    uvBinary.getAbsolutePath(), "venv", ".venv", "--python", "3.12"
);
pb.directory(envRoot);
int exitCode = runProcess(pb);
```

**Step 3 — Install dependencies**: Use the `pyproject.toml` bundled in the
plugin JAR as the dependency source. Extract it to `envRoot/` and sync:

```java
// Extract pyproject.toml AND uv.lock from JAR resources
extractResource("/copilotj-env/pyproject.toml", envRoot);
extractResource("/copilotj-env/uv.lock", envRoot);

// uv sync --frozen (uses lockfile for reproducibility)
ProcessBuilder pb = new ProcessBuilder(
    uvBinary.getAbsolutePath(), "sync", "--frozen"
);
pb.directory(envRoot);
int exitCode = runProcess(pb);
```

Alternatively, for faster installs without a lockfile:

```java
// uv pip install -e /path/to/copilotj/source
// Or: uv pip install copilotj (if published to PyPI)
```

**Step 4 — Write default config**: Create a `.env` file in `envRoot/` with
placeholder values. The user edits this file to configure API keys.

### Idempotency

Each `check*()` method verifies whether the corresponding step has already been
completed (e.g., `checkDepsInstalled()` verifies that `aiohttp` and `openai`
are importable). This makes `installEverything()` safe to call multiple times.

## 2. Process Lifecycle with Appose

### Appose dependency

Add to the Fiji plugin's `pom.xml`:

```xml
<dependency>
    <groupId>org.apposed</groupId>
    <artifactId>appose-java</artifactId>
    <version>0.8.0</version>
</dependency>
```

Note: the SAMJ project accesses Appose via the `dl-modelrunner` dependency.
CopilotJ can depend on `appose-java` directly since it does not need the
model-runner abstractions.

### Starting the server

```java
public class CopilotJProcessManager {

    private Service pythonService;
    private Connection wsConnection;

    public void start() throws IOException {
        // 1. Create Appose Environment pointing to the venv
        File venvDir = new File(envRoot, ".venv");
        Environment env = new Environment() {
            @Override
            public String base() {
                return venvDir.getAbsolutePath();
            }
        };

        // 2. Read .env file and pass config as env vars (single source of truth)
        Map<String, String> envVars = readDotEnv(new File(envRoot, ".env"));

        // 3. Create the Appose Service with env vars
        pythonService = env.python(envVars);

        // 4. Send init task — Python binds port 0, returns assigned port
        Task initTask = pythonService.task(
            "import copilotj.appose_worker as w\n"
            + "w.start_server()\n"
        );
        initTask.waitFor();

        // 5. Extract port from task response
        int port = (int) initTask.outputs.get("port");
        String host = "127.0.0.1";
        log.info("copilotj: Python server started on port " + port);

        // 6. Connect via WebSocket (existing Connection class)
        wsConnection = new Connection(
            "http://" + host + ":" + port,
            eventHandler, logService, maxRetry
        );
        wsConnection.connect();
    }
}
```

### Python-side adapter

A thin adapter module (`copilotj/appose_worker.py`) bridges the Appose task
protocol with the existing `Server` class. It properly handles Appose's
stdin/stdout JSON task protocol for port discovery:

```python
"""Appose worker adapter for CopilotJ server.

Handles the Appose task protocol on stdin/stdout while running the aiohttp
server on a dynamically assigned port. The port is returned to Java via the
task response so Java can connect its WebSocket client.
"""
import asyncio
import json
import sys
import threading

from copilotj.core import load_env
from copilotj.server import Server

_server: Server | None = None
_runner = None  # aiohttp.web.AppRunner


def start_server():
    """Start the aiohttp server. Binds port 0 for dynamic allocation.

    This function is invoked by the Appose init task. It starts the server
    and writes the port number to task.outputs so Java can discover it.
    """
    global _server, _runner

    load_env()
    _server = Server()

    loop = asyncio.new_event_loop()

    # Start the server on port 0 (OS assigns a free port)
    app = _server._create_app()

    async def _start():
        global _runner
        import aiohttp.web as web
        _runner = web.AppRunner(app)
        await _runner.setup()
        site = web.TCPSite(_runner, "127.0.0.1", 0)
        await site.start()
        # Get the actual port assigned by the OS
        return site._server.sockets[0].getsockname()[1]

    port = loop.run_until_complete(_start())

    # Return port to Java via Appose task protocol (stdout JSON)
    _respond({"port": port})

    # Start stdin monitor thread for shutdown + future tasks
    monitor = threading.Thread(target=_stdin_loop, args=(loop,), daemon=True)
    monitor.start()

    # Keep the event loop alive until shutdown
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(_runner.cleanup())
        loop.close()


def _respond(outputs: dict):
    """Write a task response to stdout in Appose JSON format."""
    json.dump({"outputs": outputs}, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _stdin_loop(loop: asyncio.AbstractEventLoop):
    """Read Appose task JSON from stdin; on EOF, trigger graceful shutdown."""
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                _handle_task(task, loop)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    finally:
        # stdin closed → Appose is shutting us down
        loop.call_soon_threadsafe(loop.stop)


def _handle_task(task: dict, loop: asyncio.AbstractEventLoop):
    """Handle incoming Appose tasks (extensible for future commands)."""
    # Future tasks can be dispatched here (e.g., status queries)
    pass
```

### Stopping the server

The `CopilotJBridgeService` interface gains a `stop()` method, called from
SciJava's `dispose()` lifecycle to prevent orphan Python processes:

```java
// CopilotJBridgeService.java (interface addition)
public interface CopilotJBridgeService extends Service {
    // ... existing methods ...
    void stop();
}

// DefaultCopilotJBridgeService.java
@Override
public void stop() {
    // 1. Close WebSocket connection first
    if (wsConnection != null) {
        wsConnection.close();
        wsConnection = null;
    }

    // 2. Gracefully stop the Python process via Appose
    if (pythonService != null) {
        pythonService.close();  // closes stdin → triggers graceful shutdown
        pythonService = null;
    }
}

@Override
public void dispose() {
    stop();
}

public void forceStop() {
    // For unresponsive processes
    if (pythonService != null) {
        pythonService.kill();  // SIGKILL / process.destroyForcibly()
        pythonService = null;
    }
}
```

### Health checking

Use HTTP `GET /api/ping` on the discovered port for health checking. The port
is known from the Appose task response, so no separate discovery is needed:

```java
// Periodic health check using HTTP
URL url = new URL("http://127.0.0.1:" + port + "/api/ping");
HttpURLConnection conn = (HttpURLConnection) url.openConnection();
conn.setRequestMethod("GET");
conn.setConnectTimeout(5000);
int status = conn.getResponseCode();  // 200 = healthy
```

## 3. Build Process

### Minimal build pipeline

```
┌─────────────────────────────────────────────┐
│  Build time (CI / developer machine)        │
│                                             │
│  1. Build Python package (uv build)         │
│  2. Generate uv.lock                        │
│  3. Build Java plugin (mvn package)         │
│  4. Bundle into Fiji update site            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Install time (user's machine, first run)   │
│                                             │
│  1. Java plugin calls installEverything()   │
│  2. Downloads uv binary (~20 MB)            │
│  3. Creates .venv with Python 3.12          │
│  4. Installs CopilotJ + dependencies        │
│  5. Writes default .env config              │
│                                             │
│  Total: ~5-10 GB (torch, tensorflow, etc.)  │
│  Time: 10-30 min (depends on bandwidth)     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Run time (every launch)                    │
│                                             │
│  1. Appose starts Python process            │
│  2. Python starts aiohttp server            │
│  3. Java connects via WebSocket             │
│  4. Normal CopilotJ operation               │
└─────────────────────────────────────────────┘
```

### Bundling resources into the JAR

The plugin JAR should bundle these files as classpath resources:

| Resource | Purpose |
|---|---|
| `copilotj-env/pyproject.toml` | Python dependency specification |
| `copilotj-env/uv.lock` | Lockfile for reproducible installs |
| `copilotj-env/.env.template` | Default config template |

During `installEverything()`, `pyproject.toml` and `uv.lock` are extracted to
`copilotj_<arch>/` before running `uv sync --frozen`. Note: the lockfile is
generated on the build machine; platform-specific wheels (torch, tensorflow)
may require a lockfile per platform. This is deferred to a follow-up.

### Maven configuration

Add to `plugin/pom.xml`:

```xml
<dependencies>
    <!-- Appose for process management -->
    <dependency>
        <groupId>org.apposed</groupId>
        <artifactId>appose-java</artifactId>
        <version>0.8.0</version>
    </dependency>

    <!-- Platform detection (for arch resolution) -->
    <dependency>
        <groupId>io.bioimage</groupId>
        <artifactId>dl-modelrunner</artifactId>
        <version>0.6.3</version>
    </dependency>
</dependencies>
```

Note: the `dl-modelrunner` dependency provides `PlatformDetection` and `Mamba`
utilities. If CopilotJ only needs `PlatformDetection`, consider extracting that
class to avoid the full dependency.

## 4. Key API Calls

### Java side — Environment setup

```java
// Check if environment exists
File venvPython = new File(envRoot, ".venv/bin/python");
boolean installed = venvPython.canExecute();

// Create venv (with I/O draining to prevent deadlock)
ProcessBuilder pb = new ProcessBuilder(
    uvPath, "venv", ".venv", "--python", "3.12"
);
pb.directory(envRoot);
int exitCode = runProcess(pb);

// Install dependencies
pb = new ProcessBuilder(
    uvPath, "sync", "--frozen", "--no-dev"
);
pb.directory(envRoot);
exitCode = runProcess(pb);
```

### Java side — Appose process management

```java
// Create environment
Environment env = new Environment() {
    @Override
    public String base() { return venvDir.getAbsolutePath(); }
};

// Read .env config and pass as env vars
Map<String, String> envVars = readDotEnv(new File(envRoot, ".env"));

// Create service with env vars (lazy — process starts on first task)
Service service = env.python(envVars);

// Send initialization task (triggers process start + server boot)
// Python binds port 0, returns assigned port in task response
Task init = service.task(
    "from copilotj.appose_worker import start_server\n"
    + "start_server()\n"
);
init.waitFor();
int port = (int) init.outputs.get("port");

// Health check via HTTP /api/ping (not Appose task)
URL pingUrl = new URL("http://127.0.0.1:" + port + "/api/ping");

// Graceful shutdown
service.close();  // closes stdin → Python side detects and shuts down

// Force shutdown (if grace period expires)
service.kill();   // process.destroyForcibly()
```

### Java side — WebSocket bridge (unchanged)

```java
// Existing Connection class works as-is, using the dynamic port
int port = ...; // from Appose task response
Connection conn = new Connection(
    "http://127.0.0.1:" + port,
    eventHandler, logService, maxRetry
);
conn.connect();

// Send event to Python server
conn.send(eventJson);

// Close
conn.close();
```

### Python side — Appose worker adapter

```python
# copilotj/appose_worker.py (new file)

def start_server():
    """Start aiohttp server on port 0. Returns port via Appose task."""
    # ... (see Section 2 for full implementation)
```

## 5. Error Handling

| Scenario | Detection | Recovery |
|---|---|---|
| uv download fails | HTTP error / checksum mismatch | Retry with backoff |
| venv creation fails | Non-zero exit code | Clean up partial venv, retry |
| Dependency install fails | `pip check` or import test | Log error, prompt user |
| Python process crashes | Appose `monitorLoop` → CRASH | Restart process, reconnect WS |
| WebSocket disconnect | `Connection.onClose` | Existing reconnect logic in `Connection.java` |
| Server not responding | HTTP /api/ping timeout | Kill + restart |
| HTTPPluginAPI port mismatch | Env var `COPILOTJ_PORT` not set | Set env var before Python start |
| Stale environment | Version mismatch check | Re-run `uv sync` |

### Crash recovery sequence

```java
void onProcessCrash() {
    // 1. Close stale WebSocket
    wsConnection.close();

    // 2. Kill any remaining Python process
    pythonService.kill();

    // 3. Restart (discovers new port via Appose task)
    start();
}
```

## 6. Fiji Version Compatibility

| Fiji Version | Java | Appose | Status |
|---|---|---|---|
| Fiji-Stable | Java 8 | Compatible | Tested by SAMJ CI |
| Fiji-Latest | Java 21 | Compatible | Tested by SAMJ CI |

**Known issue**: JNA version conflicts. Fiji's updater may leave old JNA jars
(e.g., `jna-3.2.7.jar`) alongside the required version. Appose's shared memory
feature requires JNA 5.14.0+. However, since CopilotJ's Appose integration only
uses process lifecycle management (not shared memory), this may not be an issue.

If shared memory is needed in the future (e.g., for efficient image transfer),
the plugin should verify JNA versions at startup and warn the user.

## 7. Future Considerations

- **Lighter dependency set**: The full `pyproject.toml` installs ~70 packages
  including `torch` and `tensorflow` (~5-10 GB). Consider offering a "lite"
  mode that only installs the agent/LLM dependencies without the heavy
  segmentation libraries.

- **Shared memory for images**: Once the basic Appose integration is stable,
  consider using Appose's `SharedMemory` + `NDArray` for zero-copy image
  transfer between Java and Python, bypassing the base64-over-WebSocket path.

- **Environment versioning**: Embed a version marker in `copilotj_<arch>/` so
  the plugin can detect when the environment needs updating (e.g., after a
  plugin upgrade that requires new Python dependencies).

- **Fiji update site integration**: Package the uv binary and dependency
  specification as part of the Fiji update site so users get everything
  through the standard Fiji updater.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES_OPEN | 9 issues, 2 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

UNRESOLVED: 0 unresolved decisions (all 17 decisions resolved)
VERDICT: ENG REVIEW passed with issues — 3 P1 fixes required before implementation

### Key decisions from review

1. **Appose task protocol**: Use Appose tasks for port discovery. Python binds port 0, returns port via task response.
2. **Stop lifecycle**: Add stop() to CopilotJBridgeService, called from SciJava dispose(). Prevents orphan processes.
3. **ProcessBuilder I/O**: Drain stdout/stderr in background threads to prevent deadlock during uv commands.
4. **Config delivery**: Java reads .env, passes to Python via env vars. Single source of truth.
5. **Bundled resources**: Extract both pyproject.toml AND uv.lock from JAR.
6. **HTTPPluginAPI**: Read port from env var instead of hardcoding 8786.
7. **Test suite**: Full coverage (28 paths) for env manager, process manager, and appose worker.
8. **Dynamic port**: Python selects port (bind port 0), returns to Java via Appose task.
9. **Platform lockfile**: Deferred to follow-up.
