# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the plugin-error hierarchy and the not-connected mapping.

Covers:
  A.  BridgePluginAPI._request maps ``err="Client not found"`` -> PluginNotConnectedError
      (with the curated DEFAULT_MESSAGE); any other err -> PluginRequestError.
  A2. HTTPPluginAPI._request parses the body even on HTTP >=400 (the bridge returns
      500 + body for not-found), so the mapping is reachable.
  A3. A plugin timeout (raised by the bridge as PluginRequestError) propagates as
      PluginRequestError through BridgePluginAPI.
  B.  Bridge.send_event returns ``err == CLIENT_NOT_FOUND_ERR`` when no client matches.

Drives async code with ``asyncio.run`` (no pytest-asyncio), matching the suite.
"""

import asyncio
import uuid

import pytest

from copilotj.plugin.api import (
    CLIENT_NOT_FOUND_ERR,
    BridgePluginAPI,
    BridgeRequest,
    BridgeResponse,
    HTTPPluginAPI,
    PluginNotConnectedError,
    PluginRequestError,
)
from copilotj.plugin.awt import TakeSnapshotRequest
from copilotj.server.bridge import Bridge


class _StubBridge:
    """Bridge stand-in whose send_event returns a canned response or raises."""

    def __init__(self, response: BridgeResponse | None = None, raise_exc: BaseException | None = None) -> None:
        self._response = response
        self._raise_exc = raise_exc

    async def send_event(self, req: BridgeRequest) -> BridgeResponse:  # noqa: ARG002
        if self._raise_exc is not None:
            raise self._raise_exc
        assert self._response is not None
        return self._response


class _FakeResp:
    """aiohttp response stand-in usable as an async context manager."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> "_FakeResp":
        return self

    async def __aexit__(self, *_a: object) -> bool:
        return False


class _FakeSession:
    """aiohttp ClientSession stand-in returning a canned response for .post()."""

    def __init__(self, status: int, body: str) -> None:
        self._resp = _FakeResp(status, body)

    def post(self, url: str, **_kwargs: object) -> _FakeResp:  # noqa: ARG002
        return self._resp

    async def close(self) -> None:
        pass


def _http_api(status: int, body: str) -> HTTPPluginAPI:
    api = HTTPPluginAPI.__new__(HTTPPluginAPI)  # bypass __init__ (avoids a real aiohttp session)
    api.session = _FakeSession(status, body)  # type: ignore[attr-defined]
    return api


# --- A. BridgePluginAPI mapping ------------------------------------------------===========


def test_bridge_plugin_api_maps_client_not_found():
    api = BridgePluginAPI(_StubBridge(response=BridgeResponse(err=CLIENT_NOT_FOUND_ERR)))

    with pytest.raises(PluginNotConnectedError) as exc_info:
        asyncio.run(api._request(uuid.uuid4(), TakeSnapshotRequest()))

    assert str(exc_info.value) == PluginNotConnectedError.DEFAULT_MESSAGE


def test_bridge_plugin_api_other_err_is_request_error_not_subclass():
    api = BridgePluginAPI(_StubBridge(response=BridgeResponse(err="boom")))

    with pytest.raises(PluginRequestError) as exc_info:
        asyncio.run(api._request(uuid.uuid4(), TakeSnapshotRequest()))

    assert not isinstance(exc_info.value, PluginNotConnectedError)
    # Raw bridge text must NOT be echoed into the message (it flows into the LLM retry
    # observation and the UI) — kept in server logs only.
    assert "boom" not in str(exc_info.value)


# --- A2. HTTPPluginAPI mapping (>=400 body parsing) =========================================


def test_http_plugin_api_maps_not_found_on_500():
    body = BridgeResponse(err=CLIENT_NOT_FOUND_ERR).model_dump_json()
    api = _http_api(500, body)

    with pytest.raises(PluginNotConnectedError):
        asyncio.run(api._request(uuid.uuid4(), TakeSnapshotRequest()))


def test_http_plugin_api_other_err_on_500():
    body = BridgeResponse(err="boom").model_dump_json()
    api = _http_api(500, body)

    with pytest.raises(PluginRequestError) as exc_info:
        asyncio.run(api._request(uuid.uuid4(), TakeSnapshotRequest()))

    assert not isinstance(exc_info.value, PluginNotConnectedError)
    assert "boom" not in str(exc_info.value)


def test_http_plugin_api_malformed_2xx_is_request_error():
    # A 2xx response with a non-JSON body must raise PluginRequestError, not crash with
    # AssertionError (the old `assert result is not None` was stripped under `python -O`).
    api = _http_api(200, "<html>not json</html>")

    with pytest.raises(PluginRequestError):
        asyncio.run(api._request(uuid.uuid4(), TakeSnapshotRequest()))


def test_http_plugin_api_non_json_500_is_request_error():
    api = _http_api(500, "<html>gateway error</html>")

    with pytest.raises(PluginRequestError):
        asyncio.run(api._request(uuid.uuid4(), TakeSnapshotRequest()))


# --- A3. Timeout normalization ==============================================================


def test_bridge_plugin_api_propagates_timeout_as_request_error():
    # After bridge normalization, a plugin timeout surfaces as PluginRequestError.
    api = BridgePluginAPI(_StubBridge(raise_exc=PluginRequestError("Timeout waiting for response")))

    with pytest.raises(PluginRequestError):
        asyncio.run(api._request(uuid.uuid4(), TakeSnapshotRequest()))


# --- B. Bridge constant ====================================================================


def test_bridge_send_event_returns_client_not_found_constant():
    bridge = Bridge()

    resp = asyncio.run(bridge.send_event(BridgeRequest(client_id=uuid.uuid4(), event="x")))

    assert resp.err == CLIENT_NOT_FOUND_ERR


# --- G (lean). Real Bridge (no clients) + real BridgePluginAPI end-to-end ==================


def test_real_bridge_with_real_bridge_plugin_api_raises_not_connected():
    # Integration guard: the real Bridge produces the constant that the real
    # BridgePluginAPI maps — the whole source-to-error chain with no stubs between them.
    api = BridgePluginAPI(Bridge())

    with pytest.raises(PluginNotConnectedError) as exc_info:
        asyncio.run(api._request(uuid.uuid4(), TakeSnapshotRequest()))

    assert str(exc_info.value) == PluginNotConnectedError.DEFAULT_MESSAGE
