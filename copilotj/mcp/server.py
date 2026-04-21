# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp.web as web
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent

from copilotj.plugin.api import BridgePluginAPI, ClientPluginAPI
from copilotj.server.bridge import DEV_CLIENT_ID, Bridge
from copilotj.util.base64 import extract_base64_image

__all__ = ["mcp"]

_log = logging.getLogger("copilotj.mcp")

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "CopilotJ",
    instructions=(
        "CopilotJ provides Fiji/ImageJ2 bioimage analysis tools. "
        "Use capture_fiji_screen to see the current Fiji state, "
        "run_macro to execute ImageJ macros, and capture_image to "
        "inspect specific images. Start by checking the environment "
        "with fiji_environment."
    ),
)

# ---------------------------------------------------------------------------
# Bridge lifecycle
# ---------------------------------------------------------------------------

_bridge: Bridge | None = None
_api: ClientPluginAPI | None = None
_bridge_runner: _BridgeRunner | None = None

_DEFAULT_BRIDGE_HOST = "0.0.0.0"
_DEFAULT_BRIDGE_PORT = 8786


class _BridgeRunner:
    """Manages the embedded WebSocket bridge as a background aiohttp server."""

    def __init__(self, bridge: Bridge, host: str, port: int):
        self.bridge = bridge
        self.host = host
        self.port = port
        self._app = web.Application()
        self._app.router.add_get("/api/plugins", bridge.on_plugin_connect)
        self._app.router.add_post("/api/plugins/events", bridge.on_forward_event)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        _log.info("Bridge listening on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        await self.bridge.close()


async def _ensure_bridge() -> None:
    """Lazily start the embedded bridge if not already running."""
    global _bridge, _api, _bridge_runner
    if _bridge_runner is not None:
        return

    host = os.environ.get("COPILOTJ_BRIDGE_HOST", _DEFAULT_BRIDGE_HOST)
    port = int(os.environ.get("COPILOTJ_BRIDGE_PORT", str(_DEFAULT_BRIDGE_PORT)))

    _bridge = Bridge()
    inner = BridgePluginAPI(handler=_bridge)
    _api = inner.with_client(DEV_CLIENT_ID)

    _bridge_runner = _BridgeRunner(_bridge, host, port)
    await _bridge_runner.start()


def _detect_timeout(script: str) -> int:
    """Auto-detect timeout based on script content."""
    lower = script.lower()
    if any(kw in lower for kw in ("batch", "for(", "for (", "getfilelist", "while")):
        return 180
    return 15


async def _get_api() -> ClientPluginAPI:
    """Return the API client, starting the bridge if needed.

    Raises ToolError when Fiji is not connected.
    """
    await _ensure_bridge()
    assert _api is not None
    assert _bridge is not None
    assert _bridge_runner is not None

    if not _bridge._clients:
        raise ToolError(
            "Fiji is not connected. Start Fiji with the CopilotJBridge plugin "
            f"and ensure it connects to ws://{_bridge_runner.host}:{_bridge_runner.port}/api/plugins"
        )
    return _api


# ---------------------------------------------------------------------------
# MCP Tools — Tier 1 (Core Fiji tools)
# ---------------------------------------------------------------------------


@mcp.tool
async def run_macro(
    script: str,
    timeout: int | None = None,
) -> str:
    """Execute an ImageJ macro script in the running Fiji instance.

    Timeout is auto-detected: 15s for normal scripts, 180s for batch/loop scripts.
    Set timeout explicitly to override auto-detection.
    """
    api = await _get_api()
    if timeout is None:
        timeout = _detect_timeout(script)

    try:
        script_with_marker = script + '\nprint("Macro executed.");'
        response = await api.run_script("macro", script_with_marker, timeout=timeout, with_snapshot=False)
    except asyncio.TimeoutError:
        raise ToolError(
            f"Script execution timeout ({timeout}s). "
            "For batch processing, consider breaking down the script or setting a longer timeout."
        )
    except Exception as e:
        raise ToolError(f"Bridge error: {e}")

    if response.err or "Error" in str(response):
        raise ToolError(f"Error during execution: {response}")

    return f"Macro executed successfully.\n{response}"


@mcp.tool
async def run_script(
    language: str,
    script: str,
    timeout: int | None = None,
) -> str:
    """Execute a script in Fiji.

    Supported languages: macro, JavaScript, Python, etc.
    Returns script output or error message.
    """
    api = await _get_api()
    if timeout is None:
        timeout = _detect_timeout(script)

    try:
        response = await api.run_script(language, script, timeout=timeout, with_snapshot=False)
    except asyncio.TimeoutError:
        raise ToolError(f"Script execution timeout ({timeout}s).")
    except Exception as e:
        raise ToolError(f"Bridge error: {e}")

    if response.err:
        raise ToolError(f"Script error: {response.err}")

    outputs = response.outputs or {}  # TODO: format outputs in a more readable way
    return str(response)


@mcp.tool
async def capture_fiji_screen() -> Image:
    """Capture the current Fiji screen as an image.

    Returns a screenshot showing all open Fiji windows and their state.
    The client model can analyze the image directly (no separate VLM needed).
    """
    api = await _get_api()

    try:
        response = await api.capture_screen()
    except Exception as e:
        raise ToolError(f"Failed to capture screen: {e}")

    if not response.screenshots:
        raise ToolError("No screenshots captured. Fiji may not have any windows open.")

    # Return the first (usually only) screenshot as an Image
    screenshot = response.screenshots[0]
    image_bytes = base64.b64decode(screenshot.image)
    return Image(data=image_bytes, format="png")


@mcp.tool
async def capture_image(title: str | None = None) -> ToolResult:
    """Capture the current active Fiji image with metadata.

    Returns the image content along with metadata: dimensions, bit depth, histogram.
    Optionally specify a window title to capture a specific image.
    """
    api = await _get_api()

    try:
        response = await api.capture_image(title=title)
    except Exception as e:
        raise ToolError(f"Failed to capture image: {e}")

    content: list = []

    # Metadata as text content
    metadata: dict[str, Any] = {}
    if response.info:
        metadata["info"] = response.info.model_dump(mode="json")
    if response.histogram:
        metadata["histogram"] = response.histogram.model_dump(mode="json")
    if metadata:
        content.append(TextContent(type="text", text=json.dumps(metadata, indent=2)))

    # Image as proper MCP image content block
    if response.image:
        raw_b64 = extract_base64_image(response.image)
        image_bytes = base64.b64decode(raw_b64)
        content.append(Image(data=image_bytes, format="png"))

    if not content:
        raise ToolError("No image data or metadata returned.")

    return ToolResult(content=content)


@mcp.tool
async def take_snapshot() -> dict[str, Any]:
    """Get a structured snapshot of the current Fiji UI state.

    Returns: open windows, available actions, current image name, screen dimensions.
    Use this to understand what's open before running commands.
    """
    api = await _get_api()

    try:
        response = await api.take_snapshot()
    except Exception as e:
        raise ToolError(f"Failed to take snapshot: {e}")

    return response.model_dump(mode="json")


@mcp.tool
async def call_action(
    snapshot_id: int,
    action_id: int,
    parameters: list[Any] | None = None,
) -> dict[str, Any]:
    """Execute a UI action from a previous snapshot.

    First call take_snapshot() to get available actions and their IDs,
    then call this with the snapshot_id and action_id.
    """
    api = await _get_api()

    try:
        response = await api.call_action(snapshot_id=snapshot_id, action_id=action_id, parameters=parameters)
    except Exception as e:
        raise ToolError(f"Failed to execute action: {e}")

    return response.model_dump(mode="json")


@mcp.tool
async def fiji_environment() -> dict[str, Any]:
    """Get Fiji/ImageJ2 environment information.

    Returns: ImageJ home, Java version, installed plugins, and other system details.
    """
    api = await _get_api()

    try:
        response = await api.summarise_environment()
    except Exception as e:
        raise ToolError(f"Failed to get environment: {e}")

    return response.model_dump(mode="json")


@mcp.tool
async def list_operations(since: str | None = None) -> dict[str, Any]:
    """Get recent Fiji operation history.

    Returns list of operations performed since the given datetime (ISO 8601 format).
    If no datetime is provided, returns operations since the last call.
    """
    api = await _get_api()

    dt = None
    if since is not None:
        try:
            dt = datetime.fromisoformat(since)
        except ValueError:
            raise ToolError(f"Invalid datetime format: {since}. Use ISO 8601 format (e.g., '2026-04-15T10:00:00').")

    try:
        response = await api.get_operation_history(since=dt)
    except Exception as e:
        raise ToolError(f"Failed to get operation history: {e}")

    return response.model_dump(mode="json")


@mcp.tool
async def folder_summary(folder_path: str) -> str:
    """List files in a directory on the local filesystem.

    Useful for discovering image files to open in Fiji.
    Returns up to 300 items with relative paths.
    """
    if folder_path.strip() == ".":
        raise ToolError("The current directory is not a valid folder path. Please provide a specific folder path.")

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ToolError(f"The path '{folder_path}' is not a valid directory.")

    items: list[str] = []
    max_files = 300

    for path in folder.rglob("*"):
        if len(items) >= max_files:
            break
        relative = path.relative_to(folder)
        if path.is_file():
            items.append(str(relative))
        elif path.is_dir():
            items.append(f"Directory: {relative}")

    total_msg = ""
    if len(items) >= max_files:
        total_msg = f" (Showing first {max_files} items, more files may exist)"

    file_list = "\n".join(f"  {i}. {item}" for i, item in enumerate(items, 1))
    return f"Folder: {folder_path}{total_msg}\n{file_list}"


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------


@mcp.resource("fiji://environment")
async def get_environment_resource() -> dict[str, Any]:
    """Fiji/ImageJ2 environment information."""
    api = await _get_api()
    try:
        response = await api.summarise_environment()
        return response.model_dump(mode="json")
    except Exception:
        return {"error": "Fiji not connected"}


@mcp.resource("fiji://windows")
async def get_windows_resource() -> dict[str, Any]:
    """Currently open Fiji windows."""
    api = await _get_api()
    try:
        response = await api.take_snapshot()
        return response.model_dump(mode="json")
    except Exception:
        return {"error": "Fiji not connected"}


# ---------------------------------------------------------------------------
# MCP Prompts
# ---------------------------------------------------------------------------


@mcp.prompt
def analyze_bioimage(task: str = "segment objects") -> str:
    """Template for bioimage analysis workflows in Fiji."""
    return (
        "I want to analyze bioimages in Fiji/ImageJ2. "
        f"My goal is: {task}\n\n"
        "Please help me by:\n"
        "1. First, check the Fiji environment with fiji_environment\n"
        "2. Capture the current screen with capture_fiji_screen to see what's open\n"
        "3. If no image is open, guide me to open one or use run_macro to open a sample\n"
        "4. Develop a step-by-step ImageJ macro workflow for my task\n"
        "5. Execute the macro with run_macro and verify results with capture_fiji_screen\n"
    )


@mcp.prompt
def debug_macro(error_message: str, original_script: str) -> str:
    """Template for debugging ImageJ macro errors."""
    return (
        "I got an error running an ImageJ macro. Please help me debug it.\n\n"
        f"Error message:\n{error_message}\n\n"
        f"Original script:\n```\n{original_script}\n```\n\n"
        "Please:\n"
        "1. Analyze the error and identify the likely cause\n"
        "2. Check the current Fiji state with take_snapshot\n"
        "3. Propose a corrected macro\n"
        "4. Test the corrected macro with run_macro\n"
    )
