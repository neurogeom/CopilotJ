# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

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
