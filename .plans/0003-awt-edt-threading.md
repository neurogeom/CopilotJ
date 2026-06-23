# Plan: Route Fiji UI mutation through the AWT Event Dispatch Thread

> Status: **RFC — deferred from the `awt-refs` branch.** Surfaced by the
> `/review` security pass (EDT violation at the MCP `call_action` trust
> boundary). Not folded into this PR because the fix is cross-cutting (every
> Fiji-touching event shares the pattern) and an inconsistent single-path wrap
> would be worse than none. This document captures the analysis so it is not lost.

## Context

AWT's single-thread rule requires all component reads/writes and event dispatch
to happen on the **Event Dispatch Thread (EDT)**. CopilotJ's control paths
violate this: every Fiji-touching event (`run_action`, `run_macro`, `run_script`,
`capture_screen`, `capture_image`) executes on the **Jetty worker thread** (MCP)
or the **WebSocket/daemon thread** (bridge), never on the EDT.

This is pre-existing — the macro/script paths have always done this — but the
AWT ref-model branch (`awt-refs`) makes it more prominent by exposing structured
component mutation (`Checkbox.setState`, `Scrollbar.setValue`, `Choice.select`,
`TextField.setText`, `List.select`) plus synthesized AWT events (`ItemEvent`,
`AdjustmentEvent`, `TextEvent`, `ActionEvent`) directly to caller-supplied input.

**Current call chain (`run_action`), all off-EDT:**

```
Jetty worker
  → McpModule.callEvent            (plugin/.../mcp/McpModule.java:185)
  → EventHandler.handle            (plugin/.../EventHandler.java:158)
  → case "run_action"              (EventHandler.java:247)
  → SnapshotManager.runAction      (SnapshotManager.java:107)
  → new Snapshot(...).runAction(ref, action, params)
  → node.runAction(action, params) (Snapshot.java:236)
  → e.g. CheckboxNode.setState     (CheckboxNode.java:87)
  → component.setState(...) + fire ItemEvent   ← off-EDT AWT mutation ❌
```

**Why it mostly works today:** ImageJ's own macro engine, `IJ.run(...)`, and most
SciJava/ImageJ2 listeners are themselves EDT-lax and tolerate off-EDT calls. The
MCP server binds `127.0.0.1`, so the only caller is the user's own local client.
No field data has been lost yet.

**Why it is still a landmine:**

1. Deadlock with `RepaintManager` if a listener triggers a synchronous repaint
   while the EDT is mid-paint (lock inversion).
2. Listeners that assume the EDT (e.g. call `SwingUtilities`-gated APIs, or
   `ModalContext`) throw on the Jetty thread — surfaces to the client as a
   generic `InvocationTargetException` / MCP `isError`.
3. Torn state: model updated off-EDT while EDT reads → flicker / inconsistent UI.

## Decision log

| Decision                     | Choice                                                                        | Why                                                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Fix in the AWT ref-model PR? | **No — defer**                                                                | Too large / cross-cutting; `run_macro`/`run_script`/`capture` share the pattern; an inconsistent single-path fix is worse than none |
| Scope of the fix             | **Holistic — all Fiji-touching events**                                       | Consistency; do EDT once, across every path that mutates (or reads) AWT                                                             |
| Wrapper primitive            | `EventQueue.invokeAndWait` (preferred)                                        | Actions need the result synchronously to build the response; `invokeLater` would require a callback/future shape                    |
| Where to wrap                | **One chokepoint in `EventHandler.handle`** (via an `onEdt(Callable)` helper) | Avoids scattering EDT logic across nodes; `Snapshot`/`ScriptRunner`/`ScreenCapturer` stay thread-agnostic                           |
| Called _from_ the EDT        | Detect & run inline (`isDispatchThread()` short-circuit)                      | `invokeAndWait` throws `Error` when invoked on the EDT itself                                                                       |

## Scope — paths that touch AWT off-EDT

All dispatch through `EventHandler.handle` (plugin/.../EventHandler.java:204):

| Event               | AWT touch                                                      | EDT-critical?                                     |
| ------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| `run_action`        | `component.setState/setValue/select/setText`, fires AWT events | **Yes** (component mutation + listener callbacks) |
| `run_script`        | `IJ.run(...)`, macro engine touches UI                         | **Yes**                                           |
| `take_snapshot`     | builds Snapshot tree (reads AWT components)                    | Read-only but reads AWT state                     |
| `capture_screen`    | `Window.getWindows()`, screenshots                             | Read-only                                         |
| `capture_image`     | `ImagePlus.getBufferedImage()`                                 | Read-only                                         |
| `compare_snapshots` | builds Snapshots                                               | Read-only                                         |

The mutation paths (`run_action`, `run_script`) are the hard requirement; the
read paths should be wrapped too for consistency so "is this on the EDT?" has
one answer.

## Proposed approach

### 1. A single EDT chokepoint

New helper (e.g. `copilotj.util.AwtEventQueue`, or inline in `EventHandler`):

```java
/** Runs a Callable on the EDT and returns its result; runs inline if already on the EDT. */
static <T> T onEdt(final java.util.concurrent.Callable<T> task) {
  if (java.awt.EventQueue.isDispatchThread()) {
    try { return task.call(); }
    catch (RuntimeException e) { throw e; }
    catch (Exception e) { throw new RuntimeException(e); }
  }
  final Object[] holder = new Object[1];
  try {
    java.awt.EventQueue.invokeAndWait(() -> {
      try { holder[0] = task.call(); }
      catch (RuntimeException e) { holder[0] = e; }
      catch (Exception e) { holder[0] = new RuntimeException(e); }
    });
  } catch (java.lang.reflect.InvocationTargetException e) {
    final Throwable c = e.getCause();
    if (c instanceof RuntimeException re) throw re;
    throw new RuntimeException(c);
  } catch (InterruptedException e) {
    Thread.currentThread().interrupt();
    throw new RuntimeException("EDT task interrupted", e);
  }
  if (holder[0] instanceof RuntimeException re) throw re;
  @SuppressWarnings("unchecked") final T result = (T) holder[0];
  return result;
}
```

### 2. Wrap the event cases in `EventHandler.handle`

```java
case "run_action": {
  final SnapshotManager.ActionRequest request = objectMapper.convertValue(payload.data,
      SnapshotManager.ActionRequest.class);
  final Action.Response result = onEdt(() -> snapshotManager.runAction(request));
  return objectMapper.valueToTree(result);
}
```

Do the same for `run_script`, `take_snapshot`, `capture_screen`, `capture_image`,
`compare_snapshots`.

### 3. Leave node / Snapshot / ScriptRunner code as-is

They stay thread-agnostic — they run on whatever thread calls them. The EDT
discipline lives only in `EventHandler`. No per-node changes.

## Risks / tradeoffs

- **Blocks the Jetty/WS thread on the EDT** (`invokeAndWait` blocks until the EDT
  runs the task). Acceptable for a local single-user server; the existing
  per-request timeout (`ScriptRequest.timeout`) still bounds wall clock.
- **UI freeze during long actions.** Because the EDT is single-threaded, a long
  `run_action`/`run_script` on the EDT freezes the whole Fiji UI for its
  duration. Today the UI can repaint during a long action; after the fix it
  cannot. This is _correct_ EDT behavior but a UX change. **Mitigation**: scope
  the first cut to short actions; for genuinely long work, design an async
  variant (`SwingWorker` / `invokeLater` + polling) — a larger follow-up.
- **`invokeAndWait` from the EDT throws `Error`** — mitigated by the
  `isDispatchThread()` short-circuit. **Audit** whether any ImageJ listener ever
  re-enters `EventHandler` synchronously (see Open questions).
- **Managed mode** (`COPILOTJ_MANAGED=1`): `EventHandler.handle` runs on the
  daemon asyncio thread, not Jetty — the same `onEdt(...)` applies; Fiji owns the
  EDT regardless, so it resolves.

## Files to modify

| File                                                             | Change                                                                                                                             |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `plugin/.../util/AwtEventQueue.java` (new) **or** `EventHandler` | `onEdt(Callable)` helper                                                                                                           |
| `plugin/.../EventHandler.java`                                   | wrap `run_action` (+ `run_script`, `take_snapshot`, `capture_screen`, `capture_image`, `compare_snapshots`) bodies in `onEdt(...)` |

No changes to `SnapshotManager`, `Snapshot`, the component `*Node.runAction`,
`ScriptRunner`, or `ScreenCapturer`.

## Verification

1. `just build-plugin` — compiles.
2. `SwingUtilities.isEventDispatchThread()` probe: temporarily assert inside one
   node's `runAction` that `EventQueue.isDispatchThread()` is true → passes after
   the wrap, fails before.
3. Smoke: `python3 scripts/mcp_smoke_test.py` — all checks pass; the
   round-trip / stale-ref checks now exercise the EDT path. Watch for deadlocks
   under `call_action` on Threshold / Brightness·Contrast dialogs (their
   listeners repaint).
4. Stress: rapid `run_macro` + `call_action` interleaving under a dialog with
   active listeners — no deadlock, no partial repaint.
5. Managed mode: run under `COPILOTJ_MANAGED=1`; confirm `onEdt` still resolves
   the Fiji-owned EDT.

## Open questions

- Does any ImageJ listener synchronously re-enter `EventHandler` (which would
  put us on the EDT mid-`handle`)? The `isDispatchThread()` short-circuit
  handles it, but audit `ImagejListener` / `WindowListener` callbacks.
- Wrap the read-paths (`take_snapshot` / `capture_*`) on the EDT too
  (consistency) or leave them off (cheap, read-only)? Recommendation: wrap them,
  so the EDT question has one answer everywhere.
- Long-action UX: accept the UI freeze for actions >1s (EDT-correct), or design
  an async variant for known-long actions up front?

## Trigger

Pick this RFC up when any of:

- A deadlock or partial repaint is observed during `call_action` on a dialog with
  active listeners (Threshold, Brightness·Contrast, Color Picker), **or**
- The Python-bridge `call_action` is wired to a real agent (more call volume →
  more exposure), **or**
- A new Fiji-touching event makes the off-EDT inconsistency costly.

Until then, ImageJ's EDT-lax behavior tolerates the current off-EDT mutation.
