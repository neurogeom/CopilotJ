# Plan: Port the YAML snapshot "access format" to the Java MCP server

> Status: **RFC — deferred from the `awt-refs` branch.** The `awt-refs` branch added a
> playwright-mcp-style ref model on both the Python and Java sides, but the two consumers
> of the Fiji snapshot present it in **different formats**: the Python path renders a compact
> YAML-like text tree, while the Java MCP `take_snapshot` tool returns raw JSON. This RFC
> captures the design for making them symmetric. Not folded into the resource-cleanup change
> because it is a non-trivial renderer + test-infrastructure effort, and the raw-JSON output
> is correct (if verbose) in the meantime.

## Context

CopilotJ has two consumers of the Fiji AWT widget snapshot:

1. **Python multi-agent path** — `take_snapshot` over the WebSocket bridge → JSON →
   deserialized into typed pydantic models (`copilotj/plugin/awt/...`) → rendered as a
   compact **YAML-like text tree** via `Response._describe()`. This is the "YAML-like access
   format" the LLM reads to pick a ref, e.g.:

   ```
   Snapshot #1 (image: blobs.tif, screen: 1920x1080, scale: 1.00)
   - window "Brightness/Contrast" [ref=e1] (id=1):
     - scrollbar "Minimum" [ref=e3] (setValue): value=0
     - label: 99.33 %
     - button "Apply" [ref=e9] (click)
   ```

2. **Java MCP path** (`plugin/.../mcp/`) — `take_snapshot` tool → `McpModule.callEvent` →
   `EventHandler.handle` → `objectMapper.valueToTree(snapshot)` → **pretty-printed JSON** of
   the raw AWT tree. An MCP client (LLM) gets JSON, not the compact YAML.

**The asymmetry:** Python renders the compact YAML; Java MCP serves raw JSON. Goal: port the
YAML rendering to Java so the MCP `take_snapshot` emits the same compact, ref-annotated tree.

**Two further gaps in the Java output (beyond the format):**

- Java has no equivalent of Python's per-window **flattening** (`IjContrastAdjuster`,
  `IjThresholdAdjuster`, `IjGenericDialog` merge adjacent labels into widget names, inline
  button panels as a transparent `Buttons` group, drop redundant mirrored text fields). Java
  renders the raw nested AWT tree for these dialogs.
- Java `describe()` (`ComponentNode`) is a generic single-line stub (`"Button: label=Apply"`,
  `"Container"`), used only by `print()` debugging — not by the MCP tool.

## Decision log

| Decision                 | Choice                                                                             | Why                                                                                                                                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Port now or defer?       | **Defer (this RFC)**                                                               | Renderer + new JUnit test infra is a sizable, self-contained effort; raw JSON is correct meanwhile                                                                                                     |
| Where to render YAML     | **New `SnapshotFormatter` class** (pure functions over the node tree)              | Mirrors Python's data/view split; keeps `ComponentNode` pure data; trivially unit-testable without live AWT; do **not** overload `describe()`                                                          |
| The MCP seam             | **Inside `TakeSnapshotTool`**, not `EventHandler`                                  | `EventHandler`'s `take_snapshot` case serves **both** the Python bridge (needs JSON — it discriminates window subclass by the `type` field) and the MCP tool. YAML applies only on the MCP path.       |
| Get the snapshot         | `handler.getSnapshotManager().capture()` (new public accessor)                     | `capture()` already `deactivate()`s (SnapshotManager.java:72) and returns a snapshot whose **captured** fields are intact; `runAction()` uses a fresh live snapshot independently. No JSON round-trip. |
| Format exposure          | **`take_snapshot` tool returns YAML** (resource layer removed in the prior change) | `take_snapshot`/`fiji_environment` tools already exist; the redundant `fiji://windows`/`fiji://environment` resources were removed. The tool is the LLM-facing view.                                   |
| Programmatic JSON access | **Gone from the tool**                                                             | The smoke test is the only programmatic consumer; it is rewritten to **parse the YAML** (the real contract the LLM sees).                                                                              |
| Flattening               | **Phase 1 generic only; flatteners deferred**                                      | Generic renderer ships the symmetric interaction mode for all windows now; per-dialog flatteners are pure polish, byte-match later.                                                                    |
| Java verification        | **Add JUnit 5 + `SnapshotFormatterTest`**                                          | Formatter is pure; mirror `test_awt_snapshot.py` case-by-case for byte-level grammar verification without running Fiji.                                                                                |

## Scope

The MCP `take_snapshot` tool today (`plugin/.../mcp/tools/TakeSnapshotTool.java`) calls
`McpModule.callEvent(handler, "take_snapshot", null)` and wraps the JSON in `TextContent`.
The change moves the tool to render YAML and leaves every other MCP tool, the bridge, and the
Python side untouched.

| Path                                         | Today           | After                              |
| -------------------------------------------- | --------------- | ---------------------------------- |
| MCP `take_snapshot` tool                     | raw JSON        | **YAML text**                      |
| MCP `fiji_environment` tool                  | JSON            | JSON (unchanged)                   |
| `EventHandler` `take_snapshot` case (bridge) | JSON            | JSON (unchanged — Python needs it) |
| Python `Response._describe()` rendering      | YAML            | YAML (unchanged)                   |
| `scripts/mcp_smoke_test.py` snapshot helpers | parse tool JSON | parse tool **YAML**                |

## The YAML grammar (the spec)

Authoritative source: `copilotj/test/plugin/test_awt_snapshot.py` + `copilotj/plugin/awt/_base.py`.
The Java renderer must reproduce these exactly.

- **Snapshot header:** `Snapshot #<id> (image: <currentImage|N/A>, screen: <w>x<h>, scale: <guiScale>)`
  then each window's lines. No windows → `no window opened`.
- **Node line:** `- <role> "<name>" [ref=eN] <extras> <actions>: <state>` — any segment may be omitted:
  - `role`: button|checkbox|choice|scrollbar|list|textfield|textarea|label|canvas|unknown|container|window
  - `"name"`: quoted — button/checkbox→label, window→title (IjImage/IjTextWindow) or forced "ImageJ"/"ImageJ Console", unknown→name; others→omitted.
  - `[ref=eN]`: only when ref non-null (labels & intermediate containers have none — `isRefEligible()=false`).
  - `extras`: windows emit `(id=<id>)`; others none.
  - `actions`: `(id1,id2,…)` comma-joined **short** ids. Omit if none. Reuse `Action.shortId()` (`lastIndexOf('.')`).
  - `: state`: scrollbar→`value=<v>` (orientation is **not** emitted); checkbox→`checked=<true|false>`; choice/list→`selected=<str_or_empty> items=[…]`; textfield/textarea→`text=<str_or_empty>`; label→raw sanitized text (→space, truncate 300+`…`, **no quotes**); button/canvas/unknown/window-leaf→none.
- **Containers:** render `- <head>:` (head + colon, no inline state) then each child subtree indented **2 spaces**. Verbosity LOW hides children (window renders as just `- window … (id=N)`).
- **Windows:** head includes `(id=N)` and name=title; render head + `:` then indented children — except `IjImage`/`IjTextWindow`, which render bespoke detail lines (see below).
- Helpers: `strOrEmpty(null|"")→"<empty>"`, else quoted+escaped (`\n→\\n`, `\r→\\r`, `\t→\\t`), max 300 then `"…"`. `formatItems` max 8, quoted, trailing `, ...`. Label text is raw (separate helper: no quotes, whitespace→space, truncate 300+`…`).

Bespoke windows (mirror `copilotj/plugin/awt/window/ij_image.py`, `ij_text_window.py`, `ij_imagej.py`, `ij2_console.py`):

- `IjImage`: head+`:`, then `type: <typee>, size: <size>, path: <path|N/A>` (`typee = bitDepth+"-bit "+imageType` unless already contained); `dimension: …`, `stack: … (channels, slices, frames)`; if calibrated: `calibrated: …`; `resolution: …` (+`, zoom=…` when ≠1); roi one-liner. LOW truncates after the type/size/path line.
- `IjTextWindow`: head+`:`, then `Results Table: <title> (size: N, headings: …)` or `no results table`.
- `IjImageJ` / `Ij2ConsoleWindow`: header line only, forced names "ImageJ" / "ImageJ Console".

## Proposed approach

### 1. NEW `plugin/src/main/java/copilotj/awt/SnapshotFormatter.java`

Pure functions over the existing node tree (no AWT, no Jackson). Methods (mirror Python's
`_describe` family; return `List<String>` internally, join with `"\n"` at the top):
`render(Snapshot)` / `render(Snapshot, Verbosity)`, `describeNode(...)`, container path,
`describeChildren(...)` (Phase-2 flattener hook — no-op in Phase 1), `yamlHead`, `yamlLine`,
`stateInline`, `nodeName` (class→role/name map; force "ImageJ"/"ImageJ Console"),
`headExtras` (windows→`(id=N)`), `actionSegment`, the four bespoke window methods, and
helpers `strOrEmpty` / `formatItems` / `sanitizeLabelText`.

### 2. MODIFY `TakeSnapshotTool.handle`

```java
Snapshot snap = handler.getSnapshotManager().capture();   // already deactivate()d; captured fields intact
String yaml = SnapshotFormatter.render(snap, Verbosity.NORMAL);
return CallToolResult(TextContent(yaml));
```

No `deactivate()` needed. Input schema unchanged (no params). Update `definition()` description
to note YAML rendering (keep substring "ref" so the smoke-test metadata check still passes).

### 3. MODIFY `EventHandler.java`

Add a public accessor (cross-package caller in `copilotj.mcp.tools`):

```java
public SnapshotManager getSnapshotManager() { return snapshotManager; }
```

Do **not** touch the `"take_snapshot"` case.

### 4. MODIFY `scripts/mcp_smoke_test.py`

Rewrite the snapshot helpers to parse the YAML text from `take_snapshot` (regular grammar):
track the current window from `- window "<title>" …(id=N):` lines; a node-line regex extracts
role / `"<name>"` / `[ref=e(\d+)]` / `(actions)` / trailing `: <state>`.

- `find_action_ref(yaml, action_short_id, window_title=None, label=None)` — match on the short id (e.g. `setState`, not the `.setState` suffix) and optional name.
- `checkbox_state_by_ref(yaml, ref)` — parse `checked=<true|false>` on that ref's line.
  The `take_snapshot` tool check asserts the text starts with `Snapshot #`.

### 5. ADD JUnit 5 + `plugin/src/test/java/copilotj/awt/SnapshotFormatterTest.java`

Add `org.junit.jupiter:junit-jupiter` (test scope) to `plugin/pom.xml`; mirror
`test_awt_snapshot.py` with lightweight `ComponentNode`/`ContainerNode` stubs (only
`getType/getName/getRef/getActions/isRefEligible/getChildren` needed): button, choice, label,
checkbox, scrollbar, window-tree-two-space-indent, low-verbosity-hides-children. Add a
`just test-plugin` target (`cd plugin && mvn test`); keep `just build-plugin` as
`mvn package -DskipTests`.

## Phases

- **Phase 1 (this RFC, when picked up):** generic renderer + bespoke IjImage/IjTextWindow/ImageJ/Console. Every window renders correctly with all refs/actions; specialized dialogs show the raw nested tree (not byte-identical to Python).
- **Phase 2 (follow-up):** per-window flatteners (ContrastAdjuster, ThresholdAdjuster, GenericDialog) + transparent `Buttons` group, so Java YAML byte-matches Python. Likely introduces a `RenderedNode` IR the formatter renders, with flatteners as `List<ComponentNode> → List<RenderedNode>` transforms (keeps JSON/bridge output raw). Also capture dialog/window titles for `IjGenericDialog`/`UnknownWindow` (today they rely on AWT `getName()`, often null).

## Risks / tradeoffs

- **Loss of structured JSON from the tool.** The smoke test is the only known programmatic consumer and is rewritten to parse YAML. Any future consumer that wants structured data must parse the YAML (regular, low-risk) or be re-examined. Trade: the tool now returns exactly what the LLM consumes.
- **YAML parsing in the smoke test is more brittle than JSON.** Mitigated by the grammar being regular and exhaustively specified by `test_awt_snapshot.py`; the smoke-test parser is small.
- **EDT.** `capture()` reads AWT state off the Jetty thread — pre-existing (today's MCP `take_snapshot` already does this via `callEvent`). See `.plans/0003-awt-edt-threading.md`. This RFC does not change threading.
- **`IjImageJ`/`Ij2ConsoleWindow` name special-casing.** Their AWT `getName()` is not "ImageJ"/"ImageJ Console"; the formatter forces these names to match Python.
- **Verbosity.** Phase 1 hardcodes NORMAL. The only in-scope verbosity effects are IjImage's LOW truncation and the generic container's child-hiding at LOW.

## Files to modify (when implemented)

| File                                                                                    | Change                                                              |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `plugin/.../awt/SnapshotFormatter.java` (new)                                           | The YAML renderer                                                   |
| `plugin/.../mcp/tools/TakeSnapshotTool.java`                                            | `handle()` renders YAML via `SnapshotFormatter`; description tweak  |
| `plugin/.../EventHandler.java`                                                          | Add `public SnapshotManager getSnapshotManager()`                   |
| `plugin/pom.xml` + `plugin/src/test/java/copilotj/awt/SnapshotFormatterTest.java` (new) | JUnit 5 + grammar tests                                             |
| `scripts/mcp_smoke_test.py`                                                             | Parse YAML; drop resource checks (already done in the prior change) |
| `justfile`                                                                              | `test-plugin` target                                                |

No changes to `SnapshotManager`, `Snapshot`, the `*Node` classes, `ComponentNode.describe()`,
the Python side, or the bridge `EventHandler` cases.

## Verification

1. **Java unit tests:** `cd plugin && mvn test` — all `SnapshotFormatterTest` cases pass (byte-level grammar, no Fiji).
2. **Build:** `just build-plugin`.
3. **Python regression:** `just test` — `test_awt_snapshot.py` and the rest stay green (untouched).
4. **Manual MCP smoke (live Fiji):** `just dev-plugin` then `python3 scripts/mcp_smoke_test.py`:
   - `tools/list` → 9 tools, no resources.
   - `take_snapshot` → YAML starting with `Snapshot #`.
   - `call_action` stability / round-trip / unknown-action / stale-ref all PASS/SKIP (refs parsed from YAML).

## Open questions

- Byte-match vs. good-enough for Phase 1? Recommendation: good-enough (generic renderer); byte-match is Phase 2.
- Should the smoke-test YAML parser live in the script only, or also as a reusable helper elsewhere? Recommendation: script-only for now.
- When Phase 2 lands, move flattening fully to Java and drop Python's (single source of truth), or keep both? Recommendation: keep both; they serve different consumers (Python bridge vs. MCP).

## Trigger

Pick this RFC up when any of:

- An MCP client needs the compact ref-annotated tree and the raw JSON is too verbose for the LLM context window, **or**
- The Java MCP `take_snapshot` output is observed to confuse an LLM (e.g. it fails to find refs in the JSON), **or**
- Byte-parity with the Python snapshot rendering becomes a requirement (e.g. cross-path consistency tests).

Until then, the raw-JSON `take_snapshot` is correct and the ref loop works; only the
presentation differs from the Python path.
