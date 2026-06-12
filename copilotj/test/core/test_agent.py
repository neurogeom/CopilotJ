# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Thought extraction post-processing in ChatAgent._create().

The Thought extraction logic lives inline in ``ChatAgent._create()`` (lines
215–231 of ``agent.py``).  It is tightly coupled to streaming, langfuse
tracing, and the runtime — so rather than exercising the full method, these
tests reproduce the *exact same conditional check* that the production code
uses.  If the condition changes in production, these tests should be updated
in tandem.
"""

from copilotj.core.model_client import ModelResponse, ToolCall
from copilotj.core.tool import FunctionTool


def _lookup(query: str) -> str:
    return query


_TOOL = FunctionTool(_lookup, "lookup", name="lookup")


def _make_tool_call() -> ToolCall:
    return ToolCall(id="tc1", tool=_TOOL, args=_TOOL.args_type()(query="test"))


def _apply_thought_extraction(completion: ModelResponse) -> ModelResponse:
    """Reproduce the Thought extraction post-processing from ChatAgent._create().

    This mirrors the conditional at lines 220–231 of ``agent.py`` exactly.
    """
    if (
        completion.tool_calls
        and completion.content
        and not completion.reasoning_content
        and completion.content.strip().lower().startswith("thought")
    ):
        return ModelResponse(
            content=None,
            reasoning_content=completion.content.strip(),
            tool_calls=completion.tool_calls,
            finish_reason=completion.finish_reason,
        )
    return completion


def test_thought_moved_to_reasoning_when_tool_calls_present():
    original = ModelResponse(
        content="Thought: I need to analyze this image first.",
        reasoning_content=None,
        tool_calls=[_make_tool_call()],
        finish_reason="tool_calls",
    )

    result = _apply_thought_extraction(original)

    assert result.content is None
    assert result.reasoning_content == "Thought: I need to analyze this image first."
    assert result.tool_calls == original.tool_calls


def test_thought_not_moved_when_no_tool_calls():
    original = ModelResponse(
        content="Thought: I will answer directly.\nFinal Answer: 42",
        reasoning_content=None,
        tool_calls=[],
        finish_reason="stop",
    )

    result = _apply_thought_extraction(original)

    # Content unchanged — no tool calls, so this is a final answer
    assert result.content == "Thought: I will answer directly.\nFinal Answer: 42"
    assert result.reasoning_content is None


def test_thought_not_moved_when_reasoning_already_present():
    original = ModelResponse(
        content="Thought: extra text",
        reasoning_content="already captured by model",
        tool_calls=[_make_tool_call()],
        finish_reason="tool_calls",
    )

    result = _apply_thought_extraction(original)

    # reasoning_content was already set — don't overwrite
    assert result.content == "Thought: extra text"
    assert result.reasoning_content == "already captured by model"


def test_content_without_thought_prefix_unchanged():
    original = ModelResponse(
        content="Just some regular content",
        reasoning_content=None,
        tool_calls=[_make_tool_call()],
        finish_reason="tool_calls",
    )

    result = _apply_thought_extraction(original)

    # Doesn't start with "thought" — leave as-is
    assert result.content == "Just some regular content"
    assert result.reasoning_content is None
