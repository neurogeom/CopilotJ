# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from copilotj.core.model_client.openai_chat_completion import OpenAIChatCompletionClient
from copilotj.core.model_client.openai_response import OpenAIResponseClient

__all__ = [
    "GeminiChatCompletionClient",
    "GeminiResponseClient",
]


class GeminiChatCompletionClient(OpenAIChatCompletionClient):
    def __init__(self, model: str, api_key: str, *, proxy: str | None = None):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        super().__init__(model, api_key, proxy=proxy, base_url=base_url)


class GeminiResponseClient(OpenAIResponseClient):
    def __init__(self, model: str, api_key: str, *, proxy: str | None = None):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        super().__init__(model, api_key, proxy=proxy, base_url=base_url)
