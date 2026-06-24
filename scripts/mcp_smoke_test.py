#!/usr/bin/env python3
"""Manual smoke test for the CopilotJ MCP Streamable HTTP server.

Runs against a live server (start Fiji, then `just dev-plugin` which boots the
MCP server at 127.0.0.1:3001). It walks the MCP handshake and exercises every
tool, resource and prompt, printing [PASS]/[FAIL]/[SKIP] per check.

    python3 scripts/mcp_smoke_test.py [--host 127.0.0.1] [--port 3001] \\
        [--timeout 30] [--no-fiji] [--no-color]

Pure standard library — no third-party dependencies (the project is nix-managed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3001
CLIENT_INFO = {"name": "copilotj-smoke", "version": "0.1.0"}

EXPECTED_TOOLS = {
    "run_macro",
    "run_script",
    "capture_fiji_screen",
    "capture_image",
    "take_snapshot",
    "call_action",
    "fiji_environment",
    "list_operations",
    "folder_summary",
}
EXPECTED_RESOURCES = set()  # windows/environment are now tools (take_snapshot, fiji_environment)
EXPECTED_PROMPTS = {"analyze_bioimage", "debug_macro"}

# Substrings produced by the Java side when Fiji is not reachable (stable across
# McpModule.callEvent and the resource fallbacks) — used to downgrade a tool
# error to SKIP rather than FAIL.
FIJI_DOWN_MARKERS = ("No response from Fiji", "Fiji not connected")


# --------------------------------------------------------------------------- #
# ANSI styling
# --------------------------------------------------------------------------- #


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.enabled else s

    def green(self, s: str) -> str:
        return self._wrap("32", s)

    def red(self, s: str) -> str:
        return self._wrap("31", s)

    def yellow(self, s: str) -> str:
        return self._wrap("33", s)

    def bold(self, s: str) -> str:
        return self._wrap("1", s)

    def dim(self, s: str) -> str:
        return self._wrap("2", s)


# --------------------------------------------------------------------------- #
# MCP client (Streamable HTTP)
# --------------------------------------------------------------------------- #


class McpError(Exception):
    """Raised for transport or JSON-RPC level failures."""


def parse_sse(body: str, expected_id: int) -> dict:
    """Parse an SSE text body, returning the JSON-RPC message with id == expected_id."""
    matches: list[dict] = []
    for raw_event in body.replace("\r\n", "\n").split("\n\n"):
        data_lines: list[str] = []
        for line in raw_event.split("\n"):
            if line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]  # spec: a single leading space is stripped
                data_lines.append(value)
        if not data_lines:
            continue
        try:
            msg = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        if msg.get("id") == expected_id:
            matches.append(msg)
    if matches:
        return matches[-1]
    raise McpError(f"SSE stream contained no message with id={expected_id}")


class McpClient:
    """Minimal MCP Streamable HTTP client over the `/mcp` endpoint."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._id = 0
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self._last_headers: dict[str, str] = {}

    def _header(self, name: str) -> str | None:
        lower = name.lower()
        for key, value in self._last_headers.items():
            if key.lower() == lower:
                return value
        return None

    def _post(self, payload: dict) -> tuple[int, dict[str, str], bytes]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        req = urllib.request.Request(self.base_url + "/mcp", data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, dict(resp.getheaders()), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers.items()), e.read()
        except urllib.error.URLError as e:
            raise McpError(f"connection error: {e.reason}") from e

    def call(
        self,
        method: str,
        params: dict | None = None,
        *,
        is_notification: bool = False,
    ) -> dict | None:
        payload: dict = {"jsonrpc": "2.0", "method": method}
        expected_id: int | None = None
        if not is_notification:
            self._id += 1
            expected_id = self._id
            payload["id"] = expected_id
        if params is not None:
            payload["params"] = params

        status, headers, body = self._post(payload)
        self._last_headers = headers

        if is_notification:
            if not (200 <= status < 300):
                raise McpError(f"notification {method}: expected 2xx, got {status}: {body[:200]!r}")
            return None

        ctype = headers.get("Content-Type", "")
        if not body or status == 202:
            raise McpError(f"{method}: empty body / status {status} for a request")
        if "application/json" in ctype:
            msg = json.loads(body)
        elif "text/event-stream" in ctype:
            msg = parse_sse(body.decode("utf-8", "replace"), expected_id)  # type: ignore[arg-type]
        else:
            raise McpError(f"{method}: unexpected Content-Type {ctype!r} (status {status})")

        if "error" in msg:
            raise McpError(f"{method}: JSON-RPC error {msg['error']}")
        return msg.get("result")

    def initialize(self) -> dict:
        result = self.call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
        )
        self.session_id = self._header("Mcp-Session-Id")
        self.protocol_version = (result or {}).get("protocolVersion", PROTOCOL_VERSION)
        return result or {}

    def notify_initialized(self) -> None:
        self.call("notifications/initialized", is_notification=True)


# --------------------------------------------------------------------------- #
# Results tracking
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    name: str
    status: str  # PASS / FAIL / SKIP
    detail: str = ""


class Results:
    def __init__(self, style: Style) -> None:
        self.style = style
        self.checks: list[Check] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))
        tag = {
            "PASS": self.style.green("[PASS]"),
            "FAIL": self.style.red("[FAIL]"),
            "SKIP": self.style.yellow("[SKIP]"),
        }[status]
        line = f"{tag} {name}"
        if detail:
            line += f" — {detail}"
        print(line)

    def counts(self) -> dict[str, int]:
        out = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        for check in self.checks:
            out[check.status] += 1
        return out

    def exit_code(self) -> int:
        return 1 if any(c.status == "FAIL" for c in self.checks) else 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def first_text(result: dict) -> str:
    return "\n".join(
        block.get("text", "")
        for block in result.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


def has_image(result: dict) -> bool:
    return any(isinstance(b, dict) and b.get("type") == "image" for b in result.get("content", []))


def image_b64_len(result: dict) -> int:
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "image":
            return len(block.get("data", ""))
    return 0


def fiji_down(text: str) -> bool:
    low = text.lower()
    return any(marker.lower() in low for marker in FIJI_DOWN_MARKERS)


def call_tool(
    client: McpClient,
    name: str,
    args: dict,
    results: Results,
    *,
    fiji_independent: bool = False,
) -> None:
    try:
        res = client.call("tools/call", {"name": name, "arguments": args})
    except McpError as e:
        results.add(name, "FAIL", str(e)[:200])
        return

    res = res or {}
    is_error = bool(res.get("isError", False))
    text = first_text(res).strip()

    if is_error:
        if not fiji_independent and fiji_down(text):
            results.add(name, "SKIP", f"Fiji unreachable: {text[:140]}")
        else:
            results.add(name, "FAIL", f"tool error: {text[:200]}")
        return

    if has_image(res):
        results.add(name, "PASS", f"image content block ({image_b64_len(res)} bytes b64)")
        return

    summary = text.replace("\n", " ")[:140]
    results.add(name, "PASS", summary or "ok")


def call_tool_quiet(client: McpClient, name: str, args: dict) -> str:
    """Invoke a tool without recording a check.

    Returns the first text block of the result (stripped), or "" on transport
    or tool error. Used for setup steps that only put Fiji into a state the
    following check can act on (e.g. opening a window before call_action).
    """
    try:
        res = client.call("tools/call", {"name": name, "arguments": args})
    except McpError:
        return ""
    return first_text(res or {}).strip()


def _walk_components(snap_text: str):
    """Yield ``(node, window)`` for every component dict in a take_snapshot tree.

    ``window`` is the top-level window node containing ``node`` (the window node
    itself on the first yield of each window). All snapshot-parsing helpers build
    on this single traversal (DRY) rather than each re-walking the tree.
    """
    if not snap_text:
        return
    try:
        snap = json.loads(snap_text)
    except json.JSONDecodeError:
        return

    def walk(node):
        if isinstance(node, dict):
            yield node
            for child in node.get("children") or []:
                yield from walk(child)

    for window in snap.get("windows") or []:
        for node in walk(window):
            yield node, window


def find_action_ref(
    snap_text: str,
    type_suffix: str,
    *,
    window_title: str | None = None,
    label: str | None = None,
) -> tuple[str | None, str | None]:
    """Locate the first component whose action ``type`` ends with ``type_suffix``.

    Returns ``(ref, action_short_id)``, where ``action_short_id`` is the substring
    after the last dot of the action type (e.g. ``java.awt.Checkbox.setState`` ->
    ``setState``). Optional ``window_title`` (substring of the window's title/name)
    and ``label`` (exact component label) scope the search so the target is
    deterministic rather than "the first match anywhere in the tree". Returns
    ``(None, None)`` if no match is found or the snapshot can't be parsed.
    """
    for node, window in _walk_components(snap_text):
        ref = node.get("ref")
        if ref is None:
            continue
        if label is not None and node.get("label") != label:
            continue
        if window_title is not None:
            wtitle = ""
            if isinstance(window, dict):
                wtitle = window.get("title") or window.get("name") or ""
            if window_title not in wtitle:
                continue
        for action in node.get("actions") or []:
            if isinstance(action, dict) and str(action.get("type", "")).endswith(type_suffix):
                return ref, str(action.get("type", "")).rsplit(".", 1)[-1]
    return None, None


def checkbox_state_by_ref(snap_text: str, ref: str) -> bool | None:
    """Return the ``state`` of the checkbox identified by ``ref``, or ``None``."""
    for node, _ in _walk_components(snap_text):
        if node.get("ref") == ref and "state" in node:
            return bool(node["state"])
    return None


def expect_tool_error(
    client: McpClient,
    results: Results,
    name: str,
    args: dict,
    needle: str,
) -> None:
    """Assert a tools/call returns ``isError`` whose text contains ``needle``.

    Used for the ref-model's error contracts (stale/bogus ref -> "not found";
    unknown action -> "unknown action"). A Fiji-unreachable result downgrades to
    SKIP; anything else is a FAIL.
    """
    try:
        res = client.call("tools/call", {"name": "call_action", "arguments": args})
    except McpError as e:
        results.add(name, "FAIL", str(e)[:200])
        return
    res = res or {}
    text = first_text(res).strip()
    if res.get("isError") and needle in text.lower():
        results.add(name, "PASS", text[:140])
    elif fiji_down(text):
        results.add(name, "SKIP", "Fiji unreachable")
    else:
        results.add(name, "FAIL", f"expected isError + '{needle}', got: {text[:200]}")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def finish(results: Results, style: Style) -> int:
    counts = results.counts()
    print()
    print(
        style.bold("Summary") + f": {style.green(str(counts['PASS']))} PASS, "
        f"{style.red(str(counts['FAIL']))} FAIL, "
        f"{style.yellow(str(counts['SKIP']))} SKIP "
        f"({len(results.checks)} checks)"
    )
    return results.exit_code()


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual smoke test for the CopilotJ MCP server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=30, help="per-request timeout (seconds)")
    parser.add_argument(
        "--no-fiji",
        action="store_true",
        help="run only Fiji-independent checks (handshake + folder_summary)",
    )
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    style = Style(use_color)
    results = Results(style)

    print(style.bold(f"CopilotJ MCP smoke test → http://{args.host}:{args.port}/mcp\n"))

    client = McpClient(args.host, args.port, args.timeout)

    # 0/1. Connectivity + initialize handshake.
    try:
        init_result = client.initialize()
    except McpError as e:
        msg = str(e)
        if "connection error" in msg.lower():
            print(style.red(f"[FAIL] initialize — {msg}"))
            print(
                style.dim(
                    "\nIs the MCP server running? Start Fiji, then `just dev-plugin` "
                    f"(default {DEFAULT_HOST}:{DEFAULT_PORT})."
                )
            )
            return 2
        results.add("initialize", "FAIL", msg[:200])
        return results.exit_code()

    server_info = init_result.get("serverInfo", {})
    if server_info.get("name") != "CopilotJ":
        results.add(
            "initialize",
            "FAIL",
            f"serverInfo.name={server_info.get('name')!r}, expected 'CopilotJ'",
        )
    else:
        results.add(
            "initialize",
            "PASS",
            f"serverInfo={server_info.get('name')}/{server_info.get('version')}, protocol={client.protocol_version}",
        )
    if client.session_id:
        results.add("session", "PASS", f"Mcp-Session-Id captured ({client.session_id[:12]}…)")
    else:
        results.add("session", "FAIL", "no Mcp-Session-Id header on initialize response")

    # 2. notifications/initialized.
    try:
        client.notify_initialized()
        results.add("notifications/initialized", "PASS", "accepted")
    except McpError as e:
        results.add("notifications/initialized", "FAIL", str(e)[:200])

    # 3. tools/list — expect exactly the 9 tools.
    tools: list[dict] = []
    try:
        res = client.call("tools/list")
        tools = (res or {}).get("tools", [])
        names = {t.get("name") for t in tools}
        if names == EXPECTED_TOOLS:
            results.add("tools/list", "PASS", f"{len(names)} tools")
        else:
            results.add(
                "tools/list",
                "FAIL",
                f"missing={sorted(EXPECTED_TOOLS - names)} extra={sorted(names - EXPECTED_TOOLS)}",
            )
    except McpError as e:
        results.add("tools/list", "FAIL", str(e)[:200])

    # 3b. Tool metadata — descriptions/schemas must teach the snapshot->ref->action
    # loop, otherwise the metadata change can regress while every other check still
    # passes (codex #5).
    tools_by_name = {t.get("name"): t for t in tools if isinstance(t, dict)}
    ts_desc = (tools_by_name.get("take_snapshot") or {}).get("description", "")
    ca = tools_by_name.get("call_action") or {}
    ca_desc = ca.get("description", "")
    ca_params = ((ca.get("inputSchema") or {}).get("properties") or {}).get("parameters", {})
    ca_params_desc = ca_params.get("description", "") if isinstance(ca_params, dict) else ""
    md_problems: list[str] = []
    if "ref" not in ts_desc:
        md_problems.append("take_snapshot description lacks 'ref'")
    if "ref" not in ca_desc:
        md_problems.append("call_action description lacks 'ref'")
    if not ca_params_desc or ca_params_desc.strip() == "Action parameters":
        md_problems.append("call_action parameters description is still the bare 'Action parameters'")
    if md_problems:
        results.add("tool metadata", "FAIL", "; ".join(md_problems))
    else:
        results.add("tool metadata", "PASS", "descriptions + parameters schema teach the ref loop")

    # 4. resources/list — no resources are exposed (windows/environment are tools).
    try:
        res = client.call("resources/list")
        uris = {r.get("uri") for r in (res or {}).get("resources", [])}
        if uris == EXPECTED_RESOURCES:
            results.add("resources/list", "PASS", "no resources (converted to tools)")
        else:
            results.add(
                "resources/list",
                "FAIL",
                f"got={sorted(uris)} expected={sorted(EXPECTED_RESOURCES)}",
            )
    except McpError as e:
        results.add("resources/list", "FAIL", str(e)[:200])

    # 5. prompts/list.
    try:
        res = client.call("prompts/list")
        names = {p.get("name") for p in (res or {}).get("prompts", [])}
        if names == EXPECTED_PROMPTS:
            results.add("prompts/list", "PASS", f"{len(names)} prompts")
        else:
            results.add(
                "prompts/list",
                "FAIL",
                f"got={sorted(names)} expected={sorted(EXPECTED_PROMPTS)}",
            )
    except McpError as e:
        results.add("prompts/list", "FAIL", str(e)[:200])

    # 6. folder_summary — the only Fiji-independent tool.
    tmpdir = tempfile.mkdtemp(prefix="copilotj_smoke_")
    with open(os.path.join(tmpdir, "sample.tif"), "wb") as fh:
        fh.write(b"placeholder")
    call_tool(client, "folder_summary", {"folder_path": tmpdir}, results, fiji_independent=True)

    if args.no_fiji:
        print(style.dim("\n--no-fiji: skipping Fiji-dependent tools, resources and prompts."))
        return finish(results, style)

    # 7-14. Fiji-dependent tools.
    call_tool(client, "fiji_environment", {}, results)
    call_tool(client, "run_macro", {"script": 'print("smoke test");'}, results)
    call_tool(client, "run_script", {"language": "macro", "script": 'print("py smoke");'}, results)
    call_tool(client, "take_snapshot", {}, results)
    # call_action + the ref-model invariants need an interactable widget. A bare Fiji
    # has none, so open ROI Manager (its "Show All" checkbox exposes a safe
    # checkbox.setState action) and target it deterministically by label (codex #3).
    call_tool_quiet(client, "run_macro", {"script": 'run("ROI Manager...");'})
    snap_a = call_tool_quiet(client, "take_snapshot", {})
    ref, action_name = find_action_ref(snap_a, ".setState", label="Show All")
    if ref is None:
        # Fall back to any setState if the named label isn't present.
        ref, action_name = find_action_ref(snap_a, ".setState")

    # Check 1 — ref stability: the same widget keeps the same ref across snapshots
    # that don't change the UI (scoped to the targeted checkbox; codex #4).
    snap_b = call_tool_quiet(client, "take_snapshot", {})
    ref_b, _ = find_action_ref(snap_b, ".setState", label="Show All") if ref else (None, None)
    if ref is None or action_name is None:
        results.add("call_action (stability)", "SKIP", "no checkbox action (ROI Manager unavailable?)")
    elif ref_b == ref:
        results.add("call_action (stability)", "PASS", f"{ref} stable across 2 snapshots")
    else:
        results.add("call_action (stability)", "FAIL", f"{ref} -> {ref_b} (expected stable)")

    if ref is not None and action_name is not None:
        # Check 3 — round-trip: flip the checkbox, re-snapshot, verify, restore.
        before = checkbox_state_by_ref(snap_a, ref)
        if before is None:
            results.add("call_action (round-trip)", "SKIP", "could not read checkbox state")
        else:
            target = not before
            call_tool_quiet(
                client,
                "call_action",
                {"ref": ref, "action": action_name, "parameters": [target]},
            )
            after = checkbox_state_by_ref(call_tool_quiet(client, "take_snapshot", {}), ref)
            call_tool_quiet(
                client,
                "call_action",
                {"ref": ref, "action": action_name, "parameters": [before]},  # restore
            )
            if after == target:
                results.add("call_action (round-trip)", "PASS", f"state {before}->{after} (restored)")
            else:
                results.add("call_action (round-trip)", "FAIL", f"expected {target}, got {after}")

        # Check 4 — unknown action id on a valid ref -> distinct error contract.
        expect_tool_error(
            client,
            results,
            "call_action (unknown action)",
            {"ref": ref, "action": "bogusAction", "parameters": []},
            "unknown action",
        )

        # Check 2 — stale ref: close the window, then the ref must be "not found"
        # (the real production invariant — a previously-valid ref goes stale; codex #1).
        call_tool_quiet(
            client,
            "run_macro",
            {"script": 'selectWindow("ROI Manager"); run("Close");'},
        )
        expect_tool_error(
            client,
            results,
            "call_action (stale ref)",
            {"ref": ref, "action": action_name, "parameters": [False]},
            "not found",
        )

    # Cheap extra: a fabricated ref also yields "not found" (stable baseline that
    # doesn't depend on the window-close succeeding).
    expect_tool_error(
        client,
        results,
        "call_action (bogus ref)",
        {"ref": "e999999", "action": "setState", "parameters": [False]},
        "not found",
    )
    since = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    call_tool(client, "list_operations", {"since": since}, results)
    call_tool(client, "capture_image", {}, results)
    call_tool(client, "capture_fiji_screen", {}, results)

    # 16-17. prompts/get for both prompts (debug_macro requires both args).
    try:
        res = client.call(
            "prompts/get",
            {"name": "analyze_bioimage", "arguments": {"task": "segment objects"}},
        )
        n = len((res or {}).get("messages", []))
        results.add("prompts/get analyze_bioimage", "PASS", f"{n} message(s)")
    except McpError as e:
        results.add("prompts/get analyze_bioimage", "FAIL", str(e)[:200])
    try:
        res = client.call(
            "prompts/get",
            {
                "name": "debug_macro",
                "arguments": {"error_message": "x", "original_script": "y"},
            },
        )
        n = len((res or {}).get("messages", []))
        results.add("prompts/get debug_macro", "PASS", f"{n} message(s)")
    except McpError as e:
        results.add("prompts/get debug_macro", "FAIL", str(e)[:200])

    return finish(results, style)


if __name__ == "__main__":
    sys.exit(main())
