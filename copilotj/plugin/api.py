# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
import uuid
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast, override

import aiohttp
import pydantic
import ujson
from pydantic import TypeAdapter

from copilotj.core.config import SINGLE_CLIENT_ID
from copilotj.plugin._base import Request, Response
from copilotj.plugin.awt import SnapshotDifference, SnapshotSummary, TakeSnapshotRequest
from copilotj.plugin.awt.action import TypedActionResponse
from copilotj.plugin.image_capturer import (
    CaptureImageRequest,
    CaptureScreenRequest,
    IjImagePreviewWithInfoResponse,
    ScreenPreviews,
)
from copilotj.plugin.imagej_listener import GetOperationHistoryRequest, HistoryResponse
from copilotj.plugin.script_runner import ScriptRequest, ScriptResult
from copilotj.plugin.snapshot_manager import ActionRequest, CompareSnapshotRequest
from copilotj.plugin.summarizer import EnvironmentSummary, SummariseEnvironmentRequest

if TYPE_CHECKING:
    from copilotj.server import Bridge

_log = logging.getLogger(__name__)

__all__ = [
    "PluginAPI",
    "HTTPPluginAPI",
    "BridgePluginAPI",
    "ClientPluginAPI",
    "PluginRequestError",
    "PluginNotConnectedError",
    "CLIENT_NOT_FOUND_ERR",
]

# Bridge error string returned when no ImageJ plugin client is connected. Shared with
# ``copilotj.server.bridge`` so the producer and the consumer agree on the literal.
CLIENT_NOT_FOUND_ERR = "Client not found"


class PluginRequestError(RuntimeError):
    """Any failure of a plugin request (no client connected, timeout, plugin-side error).

    Catch this base class where plugin failures should be tolerated (e.g. optional
    ImageJ window-info lookups). Catch :class:`PluginNotConnectedError` where the
    "plugin not connected" case specifically should short-circuit.
    """


class PluginNotConnectedError(PluginRequestError):
    """No ImageJ plugin client is connected to the bridge.

    Carries a curated, user-facing message (``DEFAULT_MESSAGE``) so the agent/UI layer
    can surface it directly without leaking internal detail.
    """

    DEFAULT_MESSAGE = (
        "ImageJ is not connected. Please start Fiji with the CopilotJ plugin "
        "installed and wait for it to connect to the server, then resend your request."
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)


##########################################################


class BridgeRequest(pydantic.BaseModel):
    client_id: uuid.UUID
    event: str
    data: Any | None = None
    timeout: float | None = None  # seconds


class BridgeResponse(pydantic.BaseModel):
    data: Any | None = None
    err: str | None = None


##########################################################


class ClientPluginAPI:
    _inner: "PluginAPI"
    _client_id: uuid.UUID

    def __init__(self, inner: "PluginAPI", client_id: uuid.UUID) -> None:
        super().__init__()
        self._inner = inner
        self._client_id = client_id

    async def call_action(
        self, snapshot_id: int, action_id: int, parameters: list[Any] | None = None
    ) -> TypedActionResponse:
        return await self._request(ActionRequest(snapshot_id=snapshot_id, action_id=action_id, parameters=parameters))

    async def capture_image(self, title: str | None = None) -> IjImagePreviewWithInfoResponse:
        """Captures the current image."""
        return await self._request(CaptureImageRequest(title=title))

    async def capture_screen(self) -> ScreenPreviews:
        """Captures the current screen."""
        return await self._request(CaptureScreenRequest())

    async def compare_snapshots(self, snapshot_early: int, snapshot_later: int | None = None) -> SnapshotDifference:
        """Compares two snapshots."""
        return await self._request(CompareSnapshotRequest(id_early=snapshot_early, id_later=snapshot_later))

    async def get_operation_history(self, since: datetime | None = None) -> HistoryResponse:
        """Retrieves the operation history.

        Args:
            since: datetime, optional
                If provided, only retrieve operations since this datetime.
                If None, only retrieve operations since last call.

        Returns:
            HistoryResponse: The operation history.
        """

        if since is None:
            warnings.warn("The 'since' parameter should be provided explicitly", DeprecationWarning)
            since = self._inner._t_last_history

        self._inner._t_last_history = datetime.now()

        req = GetOperationHistoryRequest(since=since)
        return await self._request(req)

    async def run_script(
        self, language: str, script: str, *, with_snapshot: bool = True, timeout: int | None = None
    ) -> ScriptResult:
        """Runs a script in ImageJ.

        Args:
            language: The scripting language to use (e.g., "JavaScript", "Python").
            script: The script to execute.
            with_snapshot: Whether to take a snapshot of the context before and after running the script.

        Returns:
            ScriptResult: The result of running the script.
        """
        snapshot1 = None
        if with_snapshot:
            warnings.warn("The automatic snapshot parameter is deprecated", DeprecationWarning)
            snapshot1 = await self.take_snapshot()

        timeout = timeout or 600
        result = await self._request(
            ScriptRequest(language=language, script=script, timeout=max(timeout - 3, 3)), timeout=timeout
        )
        if snapshot1 is not None:
            result.snapshot = await self.compare_snapshots(snapshot1.id)

        return result

    async def summarise_context(self) -> SnapshotSummary:
        warnings.warn("The 'summarise_context' class was renamed to `take_snapshot`", DeprecationWarning)
        return await self.take_snapshot()

    async def summarise_environment(self) -> EnvironmentSummary:
        """Summarizes the ImageJ environment.

        Returns:
            EnvironmentSummary: A summary of the ImageJ environment.
        """
        return await self._request(SummariseEnvironmentRequest())

    async def take_snapshot(self) -> SnapshotSummary:
        """Take snapshot of current ImageJ."""
        return await self._request(TakeSnapshotRequest())

    async def _request[R: Response](self, data: Request[R], *, timeout: float | None = None) -> R:
        return await self._inner._request(self._client_id, data, timeout=timeout)


class PluginAPI(abc.ABC):
    """Protocol defining the ImageJ plugin API interface."""

    def __init__(self) -> None:
        super().__init__()
        self._t_last_history = datetime.now()

    def with_client(self, client_id: uuid.UUID) -> ClientPluginAPI:
        """Creates a client plugin API instance with the given client ID."""
        return ClientPluginAPI(self, client_id)

    def attach_single_client(self) -> ClientPluginAPI:
        """Attaches to the first client that connects to the server."""
        return self.with_client(SINGLE_CLIENT_ID)

    @abc.abstractmethod
    async def _request[R: Response](
        self, client_id: uuid.UUID, data: Request[R], *, timeout: float | None = None
    ) -> R: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


class HTTPPluginAPI(PluginAPI):
    """HTTP-based implementation of the PluginAPI protocol."""

    def __init__(self, server: str):
        super().__init__()
        self.server = server.rstrip("/")
        self.session = aiohttp.ClientSession(self.server, json_serialize=ujson.dumps)

    @override
    async def _request[R: Response](self, client_id: uuid.UUID, data: Request[R], *, timeout: float | None = None) -> R:
        timeout = timeout or data._timeout
        request = BridgeRequest(client_id=client_id, event=data.event, data=data, timeout=timeout).model_dump_json()
        async with self.session.post(
            "/api/plugins/events", data=bytes(request, "utf-8"), headers={"Content-Type": "application/json"}
        ) as resp:
            respStr = await resp.text()
            # The bridge returns HTTP 500 + a BridgeResponse body (err=...) for plugin-side
            # failures such as "Client not found". Parse the body even on >=400 so those map
            # to the typed plugin errors instead of a generic Exception (the status check
            # used to fire first, hiding the not-connected mapping).
            result: BridgeResponse | None
            try:
                result = BridgeResponse.model_validate_json(respStr)
            except Exception:
                result = None
            err = result.err if result is not None else None
            if err == CLIENT_NOT_FOUND_ERR:
                raise PluginNotConnectedError()
            if resp.status >= 400:
                # Body may be plugin/proxy-controlled (echoed paths, script text) — log it
                # server-side only, keep the exception message free of raw response text so
                # it can't flow into the LLM retry observation or the UI.
                _log.warning("Plugin API request failed: status=%s body=%s", resp.status, respStr)
                raise PluginRequestError(f"Plugin API request failed with HTTP {resp.status}")
            if err:
                _log.warning("Plugin request returned error: %s", err)
                raise PluginRequestError("Plugin request failed")
            if result is None:
                raise PluginRequestError("Plugin API returned a malformed response")
            return cast(R, data.response_type.model_validate(result.data))

    @override
    async def close(self):
        await self.session.close()


class BridgePluginAPI(PluginAPI):
    """Direct bridge-based implementation of the PluginAPI protocol."""

    def __init__(self, handler: "Bridge"):
        super().__init__()
        self.default_timeout = 4
        self.handler = handler

    @override
    async def _request[R: Response](self, client_id: uuid.UUID, data: Request[R], *, timeout: float | None = None) -> R:
        timeout = timeout or data._timeout or self.default_timeout
        response = await self.handler.send_event(
            BridgeRequest(client_id=client_id, event=data.event, data=data, timeout=timeout)
        )
        if response.err:
            if response.err == CLIENT_NOT_FOUND_ERR:
                raise PluginNotConnectedError()
            _log.warning("Plugin request failed: %s", response.err)
            raise PluginRequestError("Plugin request failed")
        adapter = TypeAdapter(data.response_type)
        return cast(R, adapter.validate_python(response.data))

    @override
    async def close(self) -> None:
        pass  # do nothing, the bridge will be closed otherwise
