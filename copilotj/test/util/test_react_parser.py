# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client import ModelClient, ModelResponse, ModelResponseChunk, ModelSyntaxError, ToolCall
from copilotj.core.tool import FunctionTool, Tool
from copilotj.util.react_parser import ReActChatCompletionClient, _build_last_line_prefix_regex


def lookup(query: str) -> str:
    return query


class _StubModelClient(ModelClient):
    def __init__(
        self,
        *,
        response: ModelResponse | None = None,
        stream_chunks: list[ModelResponseChunk | ToolCall] | None = None,
    ) -> None:
        self._response = response or ModelResponse(
            reasoning_content=None,
            content=None,
            tool_calls=None,
            finish_reason="unknown",
        )
        self._stream_chunks = stream_chunks or []

    @override
    def get_model(self) -> str:
        return "stub"

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
    ) -> ModelResponse:
        return self._response

    @override
    async def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk | ToolCall, None]:
        for chunk in self._stream_chunks:
            yield chunk


def test_create_parses_standard_react_response():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content="""\
Thought: need more information
Action:
{"name": "lookup", "args": {"query": "cells"}}
Final Answer: found it
""",
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    response = asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[tool]))

    assert response.reasoning_content == "need more information"
    assert response.content == "found it"
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool.name == "lookup"
    assert response.tool_calls[0].args.query == "cells"
    assert response.finish_reason == "stop"


def test_create_raises_for_plain_text_without_react_keywords():
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content="Just answer directly.",
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    try:
        asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[]))
    except ModelSyntaxError as exc:
        assert "Failed to extract action from text" in exc.message
        assert "Just answer directly." in exc.message
    else:
        raise AssertionError("Expected ModelSyntaxError for plain text content")


def test_create_raises_for_unknown_tool():
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content='Action: {"name": "missing_tool", "args": {}}',
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    try:
        asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[]))
    except ModelSyntaxError as exc:
        assert "Tool 'missing_tool' not found" in exc.message
    else:
        raise AssertionError("Expected ModelSyntaxError for unknown tool")


def test_create_stream_parses_standard_react_response():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            stream_chunks=[
                ModelResponseChunk(reasoning_content=None, content="Thought: inspect\n", finish_reason=None),
                ModelResponseChunk(
                    reasoning_content=None,
                    content='Action: {"name": "lookup", "args": {"query": "cells"}}\nFinal Answer: done',
                    finish_reason=None,
                ),
                ModelResponseChunk(reasoning_content=None, content=None, finish_reason="stop"),
            ]
        )
    )

    items = [
        item
        for item in asyncio.run(_collect_stream(client, [tool]))
        if not (isinstance(item, ModelResponseChunk) and item.reasoning_content == "")
    ]

    assert len(items) == 4
    assert isinstance(items[0], ModelResponseChunk)
    assert items[0].reasoning_content == "inspect\n"
    assert isinstance(items[1], ToolCall)
    assert items[1].tool.name == "lookup"
    assert items[1].args.query == "cells"
    assert isinstance(items[2], ModelResponseChunk)
    assert items[2].content == "done"
    assert isinstance(items[3], ModelResponseChunk)
    assert items[3].finish_reason == "stop"


def test_build_last_line_prefix_regex_matches_action_and_final_prefixes():
    pattern = _build_last_line_prefix_regex("Action", "Final Answer")

    assert pattern.search("A")
    assert pattern.search("Actio")
    assert pattern.search("Final")
    assert pattern.search("Final Answe")
    assert not pattern.search("Observation")


async def _collect_stream(client: ReActChatCompletionClient, tools: list[Tool]) -> list[ModelResponseChunk | ToolCall]:
    items: list[ModelResponseChunk | ToolCall] = []
    async for item in client.create_stream([TextMessage(role="user", text="help")], tools=tools):
        items.append(item)
    return items
