# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Executor propagation of PluginNotConnectedError.

When a delegated agent's tool fails because no plugin is connected, the Executor
must re-raise PluginNotConnectedError out of ``run()`` (skipping its retry loop)
so the leader's tool-exec short-circuit can surface the curated message. A generic
tool error still goes through the existing retry path and returns a string.
"""

import asyncio
from types import SimpleNamespace

import pytest

from copilotj.multiagent.Executor import Executor
from copilotj.plugin.api import PluginNotConnectedError


def _make_executor(*, call_tool_exc: BaseException) -> Executor:
    """An Executor bypassing __init__, with _create/_call_tool stubbed."""
    exe = Executor.__new__(Executor)
    exe.name = "test_executor"
    exe.tools = []
    exe.system_prompt = ""
    exe.max_iterations = 15
    exe.tool_retry_counter = 0
    exe.max_tool_retry = 3

    fake_tool_call = SimpleNamespace(tool=SimpleNamespace(name="take_snapshot"), args=SimpleNamespace())
    fake_response = SimpleNamespace(content=None, reasoning_content=None, tool_calls=[fake_tool_call])

    async def _create(*_a: object, **_k: object) -> SimpleNamespace:
        return fake_response

    async def _call_tool(_tool_call: object) -> str:
        raise call_tool_exc

    exe._create = _create  # type: ignore[method-assign]
    exe._call_tool = _call_tool  # type: ignore[method-assign]
    exe._build_execution_context = lambda _ctx, _i: ""  # type: ignore[method-assign]
    exe._is_task_complete = lambda _text: False  # type: ignore[method-assign]
    exe._suggest_tool_based_on_context = lambda _thought, _task: ""  # type: ignore[method-assign]
    exe._generate_final_summary = lambda _ctx: "summary"  # type: ignore[method-assign]
    exe.log_info = lambda *_a, **_k: None  # type: ignore[method-assign]
    exe.log_error = lambda *_a, **_k: None  # type: ignore[method-assign]
    return exe


def test_plugin_not_connected_propagates_out_of_run():
    exe = _make_executor(call_tool_exc=PluginNotConnectedError())

    with pytest.raises(PluginNotConnectedError):
        asyncio.run(exe.run("do something"))


def test_generic_tool_error_does_not_propagate():
    # A non-plugin error stays on the retry path and returns a string (existing behaviour).
    exe = _make_executor(call_tool_exc=ValueError("not a plugin error"))

    result = asyncio.run(exe.run("do something"))

    assert isinstance(result, str)
