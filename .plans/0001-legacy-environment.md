# Replace LegacyInjector.preinit() with LegacyEnvironment

## Problem

`mvn exec:java` fails with `No _hooks field found in ij.IJ`. The
`static { LegacyInjector.preinit(); }` in `DefaultCopilotJBridgeService` calls
`preinit()` _after_ `ij.IJ` is already loaded — on Java 21 with module strong
encapsulation, Javassist cannot retroactively patch the class, so `preinit()`
itself triggers `ij.IJ` loading and then discovers it wasn't patched.

This affects **both** modes:

1. **`dev-plugin`** (Java 8, inside Fiji): works because Fiji's launcher
   controls class-loading order.
2. **`dev-plugin-mcp`** (Java 21, standalone): fails because Maven's
   `exec:java` does not control class-loading order.

The root cause is not MCP-specific — it's that the latest Fiji dependencies
(`ij1-patcher` 1.2.6) + Java 21's module system make the static-block
approach unreliable outside Fiji's launcher.

## Approach: Use LegacyEnvironment

`LegacyEnvironment` (from `ij1-patcher`) creates an isolated
`LegacyClassLoader` that patches `ij.IJ` inside its own classloader. This
doesn't require `-javaagent` and doesn't depend on class-loading ordering.

However, `LegacyEnvironment` creates an **isolated** classloader — the `ij.*`
classes inside it are different class definitions from the ones in the parent
classloader. This means `ij.ImagePlus` from the legacy env cannot be assigned
to an `ij.ImagePlus` variable in the calling code.

This is a fundamental constraint. The plan must work within it.

### Key observation

The MCP layer (`copilotj.mcp.*`) never imports `ij.*` directly — it delegates
all ImageJ interaction through `EventHandler`, which dispatches to
`ScriptRunner`, `Summerizer`, `ScreenCapturer`, etc.

These downstream classes (`ScriptRunner`, `Summerizer`, `ScreenCapturer`,
`ImagejListener`, `CopilotJBridgeActionToolInstaller`, `IjImageHelper`) are
where `ij.*` imports live. They are tightly coupled to ImageJ1 APIs.

For standalone execution, we need these `ij.*` calls to run inside a
`LegacyEnvironment` so they use the patched classloader.

## Plan

### Step 1: Create `McpMain` entry point

New file: `src/main/java/copilotj/mcp/McpMain.java`

```java
// Standalone entry point (no Fiji launcher, no javaagent needed)
public class McpMain {
    public static void main(String... args) throws Exception {
        // Create isolated legacy environment (patches ij.IJ in its own classloader)
        var legacyEnv = new LegacyEnvironment(null, /* headless */ false);
        legacyEnv.runMacro("print(\"Legacy environment initialized.\");");

        // Build a minimal SciJava context (no LegacyService, no UIService)
        // and start the MCP server
        // ...
    }
}
```

This replaces `DefaultCopilotJBridgeService.main()` as the entry point for
the MCP profile.

### Step 2: Build lightweight SciJava context for standalone mode

`McpMain` creates a `Context` with only the services the server needs:

- `LogService`
- `EventService` (if needed)

It does **not** initialize `LegacyService` or `UIService`, which are the
services that drag in the full ImageJ2 legacy bridge and trigger the
`LegacyInjector` issues.

### Step 3: Wrap ImageJ interactions through LegacyEnvironment

The core challenge: `ScriptRunner`, `Summerizer`, `ScreenCapturer` etc. use
`ij.*` classes directly. In standalone mode, these calls must go through
`LegacyEnvironment.runMacro()` or `LegacyEnvironment.runScript()` instead.

Two sub-approaches:

**Option A: Run everything as macros/scripts through LegacyEnvironment**

Since `LegacyEnvironment.runMacro()` and `LegacyEnvironment.runScript()` already
handle the classloader isolation, the MCP tools can delegate to these methods
directly instead of using `EventHandler` → `ScriptRunner`.

This works for:

- `run_macro` → `legacyEnv.runMacro(script)`
- `run_script` → `legacyEnv.runScript(language, script)`
- `fiji_environment` → run a small macro that dumps env info
- `capture_screen` → run a macro that captures and returns base64
- `folder_summary` → run a macro that lists files

The MCP tool classes (`RunMacroTool`, `FijiEnvironmentTool`, etc.) would get
a `LegacyEnvironment` reference instead of an `EventHandler` reference in
standalone mode.

**Option B: Use LegacyEnvironment's classloader directly**

Load `ScriptRunner`, `Summerizer`, etc. through `LegacyEnvironment`'s
classloader so their `ij.*` references resolve to the patched versions.

This is more invasive and fragile — not recommended.

**Recommendation: Option A.** The MCP tools already encode their interactions
as events that map to high-level operations. Re-routing these through
`legacyEnv.runMacro()` keeps the change localized and testable.

### Step 4: Update justfile and pom.xml

- Change `dev-plugin-mcp` recipe to use `McpMain` as the main class
- The `DefaultCopilotJBridgeService` remains unchanged for Fiji mode

### Step 5: Handle return values from macros

Macros return strings, not structured data. The current MCP tools expect
JSON responses from `EventHandler`. For standalone mode, we need:

- Macros that output structured results (JSON strings via `print()`)
- A parser in `McpMain` to capture macro output and convert to MCP responses

`LegacyEnvironment.runMacro()` returns a `String` (the macro's output), so
we can parse that.

## Files to modify

| File                                      | Change                                                                |
| ----------------------------------------- | --------------------------------------------------------------------- |
| `src/main/java/copilotj/mcp/McpMain.java` | **New** — standalone entry point                                      |
| `pom.xml`                                 | Update MCP profile `exec.mainClass` default to `copilotj.mcp.McpMain` |
| `justfile`                                | Update `dev-plugin-mcp` to use new main class                         |

## Files unchanged

| File                                | Reason                                                         |
| ----------------------------------- | -------------------------------------------------------------- |
| `DefaultCopilotJBridgeService.java` | Fiji mode — keep `LegacyInjector.preinit()`                    |
| `McpModule.java`                    | May need minor wiring changes for standalone context           |
| `copilotj/mcp/tools/*`              | May need dual-mode support (EventHandler vs LegacyEnvironment) |

## Open questions

1. **Headless vs GUI**: `McpMain` likely runs headless (no Fiji window). Should
   `LegacyEnvironment` be created with `headless=true`? This depends on whether
   the MCP server needs to interact with a running Fiji GUI instance or works
   standalone.

2. **Connection to running Fiji**: The current architecture has MCP tools
   communicating with a running Fiji instance via `EventHandler`. If MCP runs
   standalone with its own `LegacyEnvironment`, it would be an independent
   ImageJ instance — is that the intended behavior?

3. **Scope of rewrite**: Option A (macro-based delegation) may require
   reimplementing some `EventHandler` logic as macros. Need to assess which
   events are trivially expressible as macros vs which need direct Java API
   access.
