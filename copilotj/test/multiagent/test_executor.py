# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Executor.run`` — the append-only ReAct loop.

Drives async code with ``asyncio.run`` (no pytest-asyncio), mirroring
``test_agent_retry.py``. A scripted ``ModelClient`` pops one canned response
per ``create_stream`` call and records the messages it was handed, so we can
assert both control flow and the prefix-cache contract.
"""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

import anthropic
import pytest

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client import ModelClient, ModelResponseChunk, ModelSyntaxError, ToolCall
from copilotj.core.model_client.anthropic import AnthropicChatCompletionClient
from copilotj.core.tool import FunctionTool, Tool
from copilotj.multiagent.Executor import Executor

# Sentinel script: raise a malformed-ReAct error on this create_stream call.
RAISE_SYNTAX = object()


# --------------------------------------------------------------------- helpers


def _thought(text: str) -> ModelResponseChunk:
    return ModelResponseChunk(reasoning_content=text, content=None, finish_reason=None)


def _final(text: str) -> ModelResponseChunk:
    return ModelResponseChunk(reasoning_content=None, content=text, finish_reason="stop")


def _make_tool(*, fail_remaining: int = 0) -> tuple[FunctionTool, dict[str, int]]:
    """Build a fake ``lookup(query)`` tool with observable, mutable state.

    ``fail_remaining`` makes the next N calls raise, then succeed — for the
    retry / cross-task-reset tests.
    """
    state = {"calls": 0, "fail": fail_remaining}

    def fn(query: str) -> str:
        state["calls"] += 1
        if state["fail"] > 0:
            state["fail"] -= 1
            raise RuntimeError("tool boom")
        return f"ok:{query}"

    return FunctionTool(fn, "test lookup tool", name="lookup"), state


def _tc(tool: FunctionTool, query: str = "cells") -> ToolCall:
    return ToolCall(id="tc1", tool=tool, args=tool.args_type()(query=query))


class _ScriptedClient(ModelClient):
    """Pops one canned script per ``create_stream`` call; records all messages.

    A script is either the ``RAISE_SYNTAX`` sentinel or a list of
    ``ModelResponseChunk`` / ``ToolCall`` items to yield in order.
    """

    def __init__(self, scripts: list[Any]) -> None:
        self._scripts = list(scripts)
        self._idx = 0
        self.recorded: list[list[TextMessage | ImageMessage]] = []

    @override
    def get_model(self) -> str:
        return "scripted"

    @override
    def get_api_key(self) -> str | None:
        return None

    @override
    async def create(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError

    @override
    async def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk | ToolCall, None]:
        self.recorded.append(list(messages))
        script = self._scripts[self._idx]
        self._idx += 1
        if script is RAISE_SYNTAX:
            raise ModelSyntaxError("malformed ReAct output")
        for chunk in script:
            yield chunk


class _RecordingRuntime:
    """Duck-typed Runtime: no-ops everywhere, records tool results."""

    def __init__(self) -> None:
        self.tool_results: list[tuple[str, str]] = []

    async def update_current_agent(self, agent: str) -> None:  # noqa: ARG002
        pass

    async def print_chat(self, agent: str, message: Any) -> None:  # noqa: ARG002
        pass

    async def print_info(self, agent: str, message: str) -> None:  # noqa: ARG002
        pass

    async def print_error(self, agent: str, message: str) -> None:  # noqa: ARG002
        pass

    async def print_retry(self, agent: str, info: Any) -> None:  # noqa: ARG002
        pass

    async def print_tool_called(self, agent: str, tool_call_id: str) -> None:  # noqa: ARG002
        pass

    async def print_tool_call_result(self, agent: str, tool_call_id: str, status: str, result: str) -> None:  # noqa: ARG002
        self.tool_results.append((status, str(result)))

    async def print_handoff(self, agent: str, handoff: Any) -> None:  # noqa: ARG002
        pass

    def log_info(self, message: str) -> None:  # noqa: ARG002
        pass

    def log_error(self, message: str) -> None:  # noqa: ARG002
        pass


def _make_executor(client: _ScriptedClient, tools: list[Tool]) -> Executor:
    ex = Executor(
        name="Tool Agent", description="desc", prompt="You are a test agent.", tools=tools, model_client=client
    )
    ex._set_runtime(_RecordingRuntime())  # noqa: SLF001
    return ex


def _content_sig(messages: Sequence[TextMessage | ImageMessage]) -> str:
    """Content signature of the provider-formatted payload (Anthropic format).

    Runs the messages through ``AnthropicChatCompletionClient._format_messages``
    and concatenates system + each message's text. Cache-control markers are
    ignored (only ``text`` is read), so a stable content prefix is detectable
    even though the trailing breakpoint marker moves each call.
    """
    system, msgs = AnthropicChatCompletionClient._format_messages(messages)  # noqa: SLF001
    parts: list[str] = []
    if system is not anthropic.NOT_GIVEN:
        for block in system:
            parts.append("SYSTEM:" + block.get("text", ""))
    for m in msgs:
        text = "".join(b.get("text", "") for b in m["content"] if b.get("type") == "text")
        parts.append(f"{m['role']}:{text}")
    return "\n".join(parts)


# --------------------------------------------------------------------- tests


def test_happy_path_tool_then_final():
    tool, state = _make_tool()
    client = _ScriptedClient([[_thought("need info"), _tc(tool)], [_final("Final Answer: 42")]])
    ex = _make_executor(client, [tool])

    result = asyncio.run(ex.run("count the cells"))

    assert result == "Final Answer: 42"
    assert state["calls"] == 1  # tool executed exactly once


def test_append_only_cache_contract():
    """The post-format payload prefix must be byte-stable across iterations.

    Codex #7: asserting Python list-append is not enough — the provider
    formatter merges consecutive same-role turns, so we verify stability at the
    formatted boundary.
    """
    tool, _ = _make_tool()
    client = _ScriptedClient(
        [
            [_thought("step 1"), _tc(tool)],
            [_thought("step 2"), _tc(tool)],
            [_final("Final Answer: done")],
        ]
    )
    ex = _make_executor(client, [tool])

    asyncio.run(ex.run("multi-step task"))

    assert len(client.recorded) == 3
    # Raw TextMessage list is append-only: each call's prefix is frozen.
    assert client.recorded[0] == client.recorded[1][: len(client.recorded[0])]
    assert client.recorded[1] == client.recorded[2][: len(client.recorded[1])]
    # Formatted content prefix is stable (system text identical + growing prefix).
    sigs = [_content_sig(m) for m in client.recorded]
    assert sigs[1].startswith(sigs[0])
    assert sigs[2].startswith(sigs[1])
    # System text is identical across all calls.
    assert sigs[0].split("\n")[0] == sigs[1].split("\n")[0] == sigs[2].split("\n")[0]


def test_tool_error_then_retry_then_success():
    tool, state = _make_tool(fail_remaining=1)  # first call raises, second succeeds
    client = _ScriptedClient(
        [
            [_thought("try once"), _tc(tool)],  # tool raises
            [_thought("try again"), _tc(tool)],  # tool succeeds
            [_final("Final Answer: recovered")],
        ]
    )
    ex = _make_executor(client, [tool])

    result = asyncio.run(ex.run("flaky task"))

    assert result == "Final Answer: recovered"
    assert state["calls"] == 2


def test_exhaustion_returns_summary():
    """Loop exhaustion renders the summary without crashing (F2 fix).

    Without the fix, ``_generate_final_summary`` calls ``.get('name')`` on the
    string ``action_summary`` and raises AttributeError.
    """
    tool, _ = _make_tool()
    client = _ScriptedClient([[_thought("a"), _tc(tool)], [_thought("b"), _tc(tool)]])
    ex = _make_executor(client, [tool])
    ex.max_iterations = 2  # noqa: SLF001 — force exhaustion after 2 steps

    result = asyncio.run(ex.run("never finishes"))

    assert "Task Summary" in result
    assert "❌" not in result  # not the top-level error path


def test_retry_counter_reset_across_runs():
    """C1: tool_retry_counter must reset per run() — executors are reused.

    Run 1 exhausts retries (counter left at 3 in the old code). Run 2 must not
    be poisoned: its first failure should NOT immediately abort.

    Both runs share one event loop because the instance-level ``_abort_event``
    (an ``asyncio.Event``) is loop-bound — matching the real lifecycle where
    ``LeaderAgent.delegate_task`` reuses the executor on a single loop.
    """
    tool, _ = _make_tool(fail_remaining=99)  # run 1: always fails
    client = _ScriptedClient(
        [
            [_thought("a"), _tc(tool)],
            [_thought("b"), _tc(tool)],
            [_thought("c"), _tc(tool)],
        ]
    )
    ex = _make_executor(client, [tool])

    # Run 2: tool fails once, then succeeds, then final answer.
    tool2, _ = _make_tool(fail_remaining=1)
    client2 = _ScriptedClient([[_thought("x"), _tc(tool2)], [_final("Final Answer: ok")]])

    async def _both() -> tuple[str, str]:
        r1 = await ex.run("doomed task")
        # Swap in run 2's client/tool; system_prompt depends on the tool set.
        ex._client = client2  # noqa: SLF001
        ex.tools = [tool2]  # noqa: SLF001
        ex.system_prompt = ex._build_enhanced_system_prompt("You are a test agent.")  # noqa: SLF001
        r2 = await ex.run("recovering task")
        return r1, r2

    r1, r2 = asyncio.run(_both())

    assert "failed after 3 attempts" in r1  # run 1 exhausted retries (counter now 3)
    assert r2 == "Final Answer: ok"  # run 2 not poisoned by run 1's counter


def test_syntax_budget_aborts():
    """C2: repeated malformed ReAct aborts after max_syntax_errors, not 15."""
    client = _ScriptedClient([RAISE_SYNTAX] * 6)
    ex = _make_executor(client, [])

    result = asyncio.run(ex.run("stuck task"))

    assert "aborted: too many invalid ReAct responses" in result
    assert len(client.recorded) == ex.max_syntax_errors  # stopped at 3, not 15


def test_syntax_budget_resets_on_success():
    """C2 refinement: a successful turn resets the budget, so 3 SCATTERED slips
    (not 3 consecutive) do not abort a long run. Without the reset, the 3rd
    malformed turn below would abort despite the recovery in between."""
    tool, _ = _make_tool()
    client = _ScriptedClient(
        [
            RAISE_SYNTAX,
            RAISE_SYNTAX,
            [_thought("recovered"), _tc(tool)],  # success → resets budget to 0
            RAISE_SYNTAX,
            [_final("Final Answer: ok")],
        ]
    )
    ex = _make_executor(client, [tool])

    result = asyncio.run(ex.run("task"))

    assert result == "Final Answer: ok"  # not aborted despite 3 total malformed turns


def test_syntax_error_correction_recovers():
    """A malformed response appends a correction turn, then the model recovers."""
    tool, _ = _make_tool()
    client = _ScriptedClient(
        [
            RAISE_SYNTAX,
            [_thought("fixed"), _tc(tool)],
            [_final("Final Answer: good")],
        ]
    )
    ex = _make_executor(client, [tool])

    result = asyncio.run(ex.run("task"))

    assert result == "Final Answer: good"
    # A correction user-turn was appended before the recovery call.
    assert any("not valid ReAct format" in getattr(m, "text", "") for m in client.recorded[1])


def test_reflection_no_action_path():
    """A thought without an action appends a reflection turn, then finishes."""
    tool, _ = _make_tool()
    client = _ScriptedClient(
        [
            [_thought("just thinking, no action yet")],  # no tool_calls, no content
            [_final("Final Answer: decided")],
        ]
    )
    ex = _make_executor(client, [tool])

    result = asyncio.run(ex.run("task"))

    assert result == "Final Answer: decided"
    # A reflection user-turn was appended before the final-answer call.
    assert any("reflect on your progress" in getattr(m, "text", "") for m in client.recorded[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
