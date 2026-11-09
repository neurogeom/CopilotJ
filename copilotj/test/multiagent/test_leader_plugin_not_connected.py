# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the leader's plugin-not-connected handling.

C. LeaderDriven.run() short-circuits on PluginNotConnectedError at the tool-exec
   boundary: it calls leader_agent.print_error with the curated message and breaks
   (no retries). A generic PluginRequestError (e.g. timeout) does NOT short-circuit.
F. LeaderAgent._safe_window_info() tolerates any PluginRequestError (incl. the
   PluginNotConnectedError subclass and timeouts) and returns "".
"""

import asyncio
from types import SimpleNamespace

from copilotj.multiagent.leader_multiagent import LeaderAgent, LeaderDriven
from copilotj.plugin.api import PluginNotConnectedError, PluginRequestError


# --------------------------------------------------------------------------------------------
# F. _safe_window_info
# --------------------------------------------------------------------------------------------


class _FakePluginTools:
    def __init__(self, *, window_info: str = "info", exc: BaseException | None = None) -> None:
        self._window_info = window_info
        self._exc = exc

    async def imagej_windowInfo(self) -> str:
        if self._exc is not None:
            raise self._exc
        return self._window_info


def _leader_with_plugin_tools(exc: BaseException | None) -> LeaderAgent:
    agent = LeaderAgent.__new__(LeaderAgent)
    agent.plugin_tools = _FakePluginTools(exc=exc)
    return agent


def test_safe_window_info_tolerates_not_connected():
    agent = _leader_with_plugin_tools(PluginNotConnectedError())
    assert asyncio.run(agent._safe_window_info()) == ""


def test_safe_window_info_tolerates_timeout():
    agent = _leader_with_plugin_tools(PluginRequestError("Timeout waiting for response"))
    assert asyncio.run(agent._safe_window_info()) == ""


def test_safe_window_info_returns_info_when_ok():
    agent = _leader_with_plugin_tools(None)
    assert asyncio.run(agent._safe_window_info()) == "info"


def test_safe_window_info_returns_empty_when_no_plugin_tools():
    # A stripped-down leader without plugin_tools degrades to "" (no AttributeError).
    agent = LeaderAgent.__new__(LeaderAgent)
    assert asyncio.run(agent._safe_window_info()) == ""


# --------------------------------------------------------------------------------------------
# C. run() tool-exec short-circuit
# --------------------------------------------------------------------------------------------


class _FakeLeaderAgent:
    """Leader stand-in: begin_dialog yields one tool call; _call_tool raises on demand."""

    def __init__(self, *, call_tool_exc: BaseException | None) -> None:
        self._call_tool_exc = call_tool_exc
        self.print_errors: list[str] = []
        self.name = "leader"

        tool_call = SimpleNamespace(
            tool=SimpleNamespace(name="take_snapshot"), args=SimpleNamespace(model_dump=lambda: {})
        )
        self._first_response = SimpleNamespace(content=None, reasoning_content="thinking", tool_calls=[tool_call])
        self._final_response = SimpleNamespace(content="done", reasoning_content=None, tool_calls=[])

    async def begin_dialog(self, _task: str, *, trace_ctx: object = None) -> SimpleNamespace:  # noqa: ARG002
        return self._first_response

    async def _call_tool(self, _tool_call: object) -> str:  # noqa: ARG002
        if self._call_tool_exc is not None:
            raise self._call_tool_exc
        return "ok"

    async def continue_dialog(
        self,
        prior_response: object,
        observation: str,
        *,
        trace_ctx: object = None,  # noqa: ARG002
    ) -> SimpleNamespace:
        # On the generic-retry path, end the dialog with a final answer so the loop exits.
        return self._final_response

    async def print_error(self, message: str) -> None:
        self.print_errors.append(message)


def _make_pattern(leader: _FakeLeaderAgent) -> tuple[LeaderDriven, list[dict]]:
    """Build a LeaderDriven bypassing __init__, wired for run()."""
    pattern = LeaderDriven.__new__(LeaderDriven)
    pattern.leader_agent = leader
    pattern._summarize_task = None
    pattern.dialog_counter = 0

    captured: list[dict] = []

    async def _bg_summarize(_dialog_id: int, _task: str, dialog_context: dict) -> None:
        captured.append(dialog_context)

    async def _dialog_changed(_dialog_id: int, _state: str) -> None:  # noqa: ARG001
        pass

    pattern._background_summarize_and_store = _bg_summarize  # type: ignore[method-assign]
    pattern.dialog_changed = _dialog_changed  # type: ignore[method-assign]
    pattern.log_info = lambda *_a, **_k: None  # type: ignore[method-assign]
    pattern.log_error = lambda *_a, **_k: None  # type: ignore[method-assign]
    return pattern, captured


def test_run_short_circuits_on_plugin_not_connected():
    leader = _FakeLeaderAgent(call_tool_exc=PluginNotConnectedError())
    pattern, captured = _make_pattern(leader)

    asyncio.run(pattern.run("analyze this image"))

    # The curated message was surfaced exactly once via print_error (no retries).
    assert len(leader.print_errors) == 1
    assert leader.print_errors[0] == PluginNotConnectedError.DEFAULT_MESSAGE
    # Background summarize captured the failed status (best-effort: the task may not
    # have run before loop close, so only assert when it did).
    if captured:
        assert captured[0]["status"] == "failed"


def test_run_does_not_short_circuit_on_generic_plugin_request_error():
    # A timeout (PluginRequestError, not the subclass) must NOT short-circuit — it
    # falls through to the generic retry path, so print_error is never called.
    leader = _FakeLeaderAgent(call_tool_exc=PluginRequestError("Timeout waiting for response"))
    pattern, _captured = _make_pattern(leader)

    # The generic retry path eventually exhausts max_tool_retry and breaks with a
    # "Too many failed tool calls" status — but no curated print_error.
    asyncio.run(pattern.run("analyze this image"))

    assert leader.print_errors == []
