# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import pydantic

__all__ = [
    "TextMessage",
    "ImageMessage",
    "HandoffMessage",
    "ToolCallRecord",
    "ToolCallMessage",
    "ToolResultMessage",
]


class TextMessage(pydantic.BaseModel):
    role: Literal["assistant", "system", "user"]
    text: str


class ImageMessage(pydantic.BaseModel):
    role: Literal["assistant", "system", "user"]
    image: str


class HandoffMessage(pydantic.BaseModel):
    target: str
    message: TextMessage | ImageMessage


class ToolCallRecord(pydantic.BaseModel):
    """Serialisable record of a single tool call for conversation history."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolCallMessage(pydantic.BaseModel):
    """Assistant message containing one or more tool calls.

    Used in conversation history for both native and ReAct modes.  The client
    layer converts this to the appropriate API format (native tool_calls or
    reconstructed ReAct text).
    """

    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCallRecord]
    reasoning_content: str | None = None


class ToolResultMessage(pydantic.BaseModel):
    """Tool role message carrying the execution result.

    In native mode this maps to ``{"role": "tool", ...}``.  In ReAct mode the
    client layer converts it to a user message with ``Observation:`` prefix.
    """

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
