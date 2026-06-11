# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from copilotj.core.model_client.openai import OpenAIChatCompletionClient

__all__ = [
    "AnthropicChatCompletionClient",
]


class AnthropicChatCompletionClient(OpenAIChatCompletionClient):
    """Anthropic/Claude client using the OpenAI-compatible API.

    This is a placeholder that uses Anthropic's OpenAI-compatible endpoint.
    It will be replaced with a native ``anthropic`` SDK implementation in the future.
    """

    def __init__(self, model: str, api_key: str, *, base_url: str | None = None, proxy: str | None = None):
        url = base_url or "https://api.anthropic.com/v1"
        super().__init__(model, api_key, base_url=url, proxy=proxy)
