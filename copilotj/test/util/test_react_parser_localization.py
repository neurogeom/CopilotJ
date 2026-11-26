# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client import ModelClient, ModelResponse, ModelResponseChunk, ModelSyntaxError, ToolCall
from copilotj.core.tool import FunctionTool, Tool
from copilotj.util.react_parser import ReActChatCompletionClient


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


async def _collect_stream(client: ReActChatCompletionClient, tools: list[Tool]) -> list[ModelResponseChunk | ToolCall]:
    items: list[ModelResponseChunk | ToolCall] = []
    async for item in client.create_stream([TextMessage(role="user", text="help")], tools=tools):
        items.append(item)
    return items


def test_create_parses_localized_react_keywords():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content=(
                    "思考：需要更多信息\n"
                    '行动：{"name": "lookup", "args": {"query": "cells"}}\n'
                    "最终答案：找到了"
                ),
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    response = asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[tool]))

    assert response.reasoning_content == "需要更多信息"
    assert response.content == "找到了"
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool.name == "lookup"
    assert response.tool_calls[0].args.query == "cells"


def test_create_stream_parses_localized_react_keywords():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            stream_chunks=[
                ModelResponseChunk(reasoning_content=None, content="思考：查看\n", finish_reason=None),
                ModelResponseChunk(
                    reasoning_content=None,
                    content='行动：{"name": "lookup", "args": {"query": "cells"}}\n最终答案：完成',
                    finish_reason=None,
                ),
                ModelResponseChunk(reasoning_content=None, content=None, finish_reason="stop"),
            ]
        )
    )

    items = asyncio.run(_collect_stream(client, [tool]))

    tool_calls = [item for item in items if isinstance(item, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool.name == "lookup"
    assert tool_calls[0].args.query == "cells"
    assert any(isinstance(item, ModelResponseChunk) and item.content == "完成" for item in items)


def test_create_parses_dsml_tool_call():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content=(
                    "<｜｜DSML｜｜ calls>\n"
                    '<｜｜DSML｜｜ invoke name="lookup">\n'
                    '<｜｜DSML｜｜ parameter name="args" string="false">{"query": "cells"}</｜｜DSML｜｜ parameter>\n'
                    "</｜｜DSML｜｜ invoke>\n"
                    "</｜｜DSML｜｜ calls>"
                ),
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    response = asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[tool]))

    assert response.content is None
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool.name == "lookup"
    assert response.tool_calls[0].args.query == "cells"


def test_create_parses_dsml_named_parameters():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content=(
                    "<｜｜DSML｜｜ invoke name=\"lookup\">"
                    '<｜｜DSML｜｜ parameter name="query" string="true">cells</｜｜DSML｜｜ parameter>'
                    "</｜｜DSML｜｜ invoke>"
                ),
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    response = asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[tool]))

    assert response.tool_calls is not None
    assert response.tool_calls[0].args.query == "cells"


def test_create_stream_parses_dsml_tool_call():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    dsml = (
        "<｜｜DSML｜｜ calls>\n"
        '<｜｜DSML｜｜ invoke name="lookup">\n'
        '<｜｜DSML｜｜ parameter name="args" string="false">{"query": "cells"}</｜｜DSML｜｜ parameter>\n'
        "</｜｜DSML｜｜ invoke>\n"
        "</｜｜DSML｜｜ calls>"
    )
    # Split the markup arbitrarily to make sure buffering works.
    chunks = [dsml[:40], dsml[40:120], dsml[120:]]
    client = ReActChatCompletionClient(
        _StubModelClient(
            stream_chunks=[
                *(ModelResponseChunk(reasoning_content=None, content=c, finish_reason=None) for c in chunks),
                ModelResponseChunk(reasoning_content=None, content=None, finish_reason="stop"),
            ]
        )
    )

    items = asyncio.run(_collect_stream(client, [tool]))

    tool_calls = [item for item in items if isinstance(item, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool.name == "lookup"
    assert tool_calls[0].args.query == "cells"


def test_create_raises_on_truncated_dsml():
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content='<｜｜DSML｜｜ invoke name="lookup"><｜｜DSML｜｜ parameter name="args">{"query": "ce',
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    try:
        asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[]))
    except ModelSyntaxError:
        pass
    else:
        raise AssertionError("Expected ModelSyntaxError for truncated DSML")


def test_create_stream_raises_on_truncated_dsml():
    client = ReActChatCompletionClient(
        _StubModelClient(
            stream_chunks=[
                ModelResponseChunk(
                    reasoning_content=None,
                    content='<｜｜DSML｜｜ invoke name="lookup"><｜｜DSML｜｜ parameter name="args">{"query": "ce',
                    finish_reason=None,
                ),
                ModelResponseChunk(reasoning_content=None, content=None, finish_reason="stop"),
            ]
        )
    )

    try:
        asyncio.run(_collect_stream(client, []))
    except ModelSyntaxError:
        pass
    else:
        raise AssertionError("Expected ModelSyntaxError for truncated DSML")


def test_create_does_not_treat_action_like_text_as_final_answer():
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content='I will continue with {"name": "lookup", "args": {"query": "cells"}}',
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    try:
        asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[]))
    except ModelSyntaxError:
        pass
    else:
        raise AssertionError("Expected ModelSyntaxError for unparsable action")
