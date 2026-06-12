# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client import OpenAIChatCompletionClient


def test_format_single_text_message():
    text_msg = TextMessage(role="user", text="Hello")
    result = OpenAIChatCompletionClient._format_messages([text_msg])

    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["type"] == "text"  # type: ignore
    assert result[0]["content"][0]["text"] == "Hello"  # type: ignore


def test_format_single_image_message():
    img_msg = ImageMessage(role="user", image="data:image/jpeg;base64,abc123")
    result = OpenAIChatCompletionClient._format_messages([img_msg])

    assert result[0]["content"][0]["type"] == "image_url"  # type: ignore
    assert (
        result[0]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"  # type: ignore
    )


def test_format_mixed_messages_same_role():
    text_msg = TextMessage(role="user", text="Hello")
    img_msg = ImageMessage(role="user", image="abc123")
    result = OpenAIChatCompletionClient._format_messages([text_msg, img_msg])

    assert len(result[0]["content"]) == 2  # type: ignore
    assert result[0]["content"][0]["type"] == "text"  # type: ignore
    assert result[0]["content"][1]["type"] == "image_url"  # type: ignore


def test_format_messages_different_roles():
    system_msg = TextMessage(role="system", text="Be helpful")
    user_msg = TextMessage(role="user", text="Hello")
    result = OpenAIChatCompletionClient._format_messages([system_msg, user_msg])

    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"


def test_merge_single_text_message():
    text_msg = TextMessage(role="user", text="Hello")
    result = OpenAIChatCompletionClient._merge_messages([text_msg])

    assert result["role"] == "user"
    assert result["content"][0]["type"] == "text"  # type: ignore
    assert result["content"][0]["text"] == "Hello"  # type: ignore


def test_merge_single_image_message():
    img_msg = ImageMessage(role="user", image="data:image/jpeg;base64,abc123")
    result = OpenAIChatCompletionClient._merge_messages([img_msg])

    assert result["content"][0]["type"] == "image_url"  # type: ignore
    assert result["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"  # type: ignore


def test_merge_mixed_messages():
    text_msg = TextMessage(role="user", text="Hello")
    img_msg = ImageMessage(role="user", image="abc123")
    result = OpenAIChatCompletionClient._merge_messages([text_msg, img_msg])

    assert len(result["content"]) == 2  # type: ignore
    assert result["content"][0]["type"] == "text"  # type: ignore
    assert result["content"][1]["type"] == "image_url"  # type: ignore


# ---------------------------------------------------------------------------
# Gemini native client tests — skipped when google-genai is not installed.
# ---------------------------------------------------------------------------
pytest.importorskip("google.genai", reason="google-genai not installed")

from copilotj.core.model_client.gemini import GeminiChatCompletionClient  # noqa: E402
from copilotj.core.tool import FunctionTool  # noqa: E402

_DUMMY_KEY = "test-api-key"


# -- Message conversion -----------------------------------------------------


def test_gemini_system_message_extraction():
    system = TextMessage(role="system", text="Be helpful.")
    user = TextMessage(role="user", text="Hello")
    contents, instruction = GeminiChatCompletionClient._convert_messages([system, user])

    assert instruction == "Be helpful."
    assert len(contents) == 1
    assert contents[0].role == "user"


def test_gemini_system_message_not_in_contents():
    s1 = TextMessage(role="system", text="Part one.")
    s2 = TextMessage(role="system", text="Part two.")
    user = TextMessage(role="user", text="Hi")
    contents, instruction = GeminiChatCompletionClient._convert_messages([s1, s2, user])

    assert instruction == "Part one.\n\nPart two."
    # Only the user message should appear in contents.
    assert len(contents) == 1
    assert contents[0].role == "user"


def test_gemini_mid_conversation_system_demoted_to_user():
    """Non-leading system messages should be demoted to user role, preserving position."""
    s1 = TextMessage(role="system", text="You are an assistant.")
    u1 = TextMessage(role="user", text="Hello")
    s2 = TextMessage(role="system", text="Please speak Chinese.")
    u2 = TextMessage(role="user", text="How is the weather?")

    contents, instruction = GeminiChatCompletionClient._convert_messages([s1, u1, s2, u2])

    # Only the leading system message is extracted.
    assert instruction == "You are an assistant."

    # The remaining messages are: user, demoted-system(as user), user — all
    # consecutive "user" role so they merge into a single Content with 3 parts.
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 3  # "Hello" + "Please speak Chinese." + "How is the weather?"


def test_gemini_role_mapping_assistant():
    assistant = TextMessage(role="assistant", text="Hi there")
    contents, _ = GeminiChatCompletionClient._convert_messages([assistant])

    assert contents[0].role == "model"


def test_gemini_role_mapping_user():
    user = TextMessage(role="user", text="Hello")
    contents, _ = GeminiChatCompletionClient._convert_messages([user])

    assert contents[0].role == "user"


def test_gemini_message_grouping():
    m1 = TextMessage(role="user", text="A")
    m2 = TextMessage(role="user", text="B")
    m3 = TextMessage(role="assistant", text="C")
    contents, _ = GeminiChatCompletionClient._convert_messages([m1, m2, m3])

    assert len(contents) == 2
    # First group: two user messages merged.
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 2
    # Second group: assistant message.
    assert contents[1].role == "model"


def test_gemini_image_url():
    img = ImageMessage(role="user", image="https://example.com/photo.png")
    contents, _ = GeminiChatCompletionClient._convert_messages([img])

    assert len(contents) == 1
    part = contents[0].parts[0]
    # URL-based images use from_uri; the part should have file_data with file_uri set.
    assert hasattr(part, "file_data") and part.file_data.file_uri


def test_gemini_image_data_uri():
    data = base64.b64encode(b"fake-png-bytes").decode()
    img = ImageMessage(role="user", image=f"data:image/png;base64,{data}")
    contents, _ = GeminiChatCompletionClient._convert_messages([img])

    part = contents[0].parts[0]
    # Data-URI images use from_bytes; the part should have inline_data set.
    assert hasattr(part, "inline_data") and part.inline_data is not None


def test_gemini_no_system_messages():
    user = TextMessage(role="user", text="Hello")
    contents, instruction = GeminiChatCompletionClient._convert_messages([user])

    assert instruction is None
    assert len(contents) == 1


# -- Tool conversion --------------------------------------------------------


def test_gemini_convert_tools():
    def get_weather(location: str) -> str:
        """Get weather for a city."""
        return "sunny"

    tool = FunctionTool(func=get_weather, description="Get weather")
    result = GeminiChatCompletionClient._convert_tools([tool])

    assert result is not None
    assert len(result) == 1
    assert len(result[0].function_declarations) == 1
    decl = result[0].function_declarations[0]
    assert decl.name == "get_weather"
    assert decl.description == "Get weather"
    assert decl.parameters_json_schema is not None


def test_gemini_convert_tools_none():
    assert GeminiChatCompletionClient._convert_tools(None) is None
    assert GeminiChatCompletionClient._convert_tools([]) is None


# -- Finish reason mapping --------------------------------------------------


def test_gemini_finish_reason_stop():
    assert GeminiChatCompletionClient._parse_finish_reason("STOP") == "stop"
    assert GeminiChatCompletionClient._parse_finish_reason("stop") == "stop"


def test_gemini_finish_reason_tool_calls():
    assert GeminiChatCompletionClient._parse_finish_reason("TOOL_CALLS") == "tool_calls"


def test_gemini_finish_reason_unknown():
    assert GeminiChatCompletionClient._parse_finish_reason("SAFETY") == "unknown"
    assert GeminiChatCompletionClient._parse_finish_reason("WHATEVER") == "unknown"


def test_gemini_finish_reason_enum_like():
    # The SDK may return enum-style strings like "FinishReason.STOP".
    assert GeminiChatCompletionClient._parse_finish_reason("FinishReason.STOP") == "stop"
    assert GeminiChatCompletionClient._parse_finish_reason("FinishReason.MAX_TOKENS") == "stop"
