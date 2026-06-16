# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, override

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client.openai_chat_completion import OpenAIChatCompletionClient

__all__ = ["OpenRouterChatCompletionClient"]

# Ephemeral prompt-cache breakpoint (5-minute TTL). Tagged onto the last system
# block (caches tools+system) and the last message content block (caches the
# growing conversation prefix) in _format_messages. Two breakpoints, within the
# Anthropic 4-breakpoint-per-request limit.
_CACHE_CONTROL = {"type": "ephemeral"}

# OpenRouter model prefixes whose provider requires explicit ``cache_control``
# breakpoints. Both Anthropic and Alibaba Qwen demand explicit per-block
# breakpoints (identical syntax) while OpenAI / DeepSeek / Gemini-2.5 cache
# implicitly and must not receive them. See
# https://openrouter.ai/docs/guides/best-practices/prompt-caching
_EXPLICIT_CACHE_PREFIXES = ("anthropic/", "qwen/")


def _supports_explicit_cache(model: str) -> bool:
    """Return True if *model* needs explicit ``cache_control`` on OpenRouter.

    A leading ``~`` (which pins a specific provider on OpenRouter) is ignored.
    """
    return model.removeprefix("~").startswith(_EXPLICIT_CACHE_PREFIXES)


class OpenRouterChatCompletionClient(OpenAIChatCompletionClient):
    """OpenRouter client over the OpenAI-compatible chat-completions API.

    Adds explicit ``cache_control`` breakpoints for ``anthropic/*`` and
    ``qwen/*`` models, mirroring :class:`AnthropicChatCompletionClient`:
    the last system block and the last message block. Other models are passed
    through unchanged (their providers cache implicitly).
    """

    def __init__(self, model: str, api_key: str, *, base_url: str | None = None, proxy: str | None = None):
        super().__init__(model, api_key, base_url=base_url, proxy=proxy)
        self._explicit_cache = _supports_explicit_cache(model)

    @override
    def _format_messages(self, messages: Sequence[TextMessage | ImageMessage]):
        openai_messages = super()._format_messages(messages)
        if not self._explicit_cache or not openai_messages:
            return openai_messages
        # Tag the last system block and the last message content block as cache
        # breakpoints. The system breakpoint caches tools+system (stable across a
        # run); the message breakpoint caches the growing conversation prefix.
        # Two breakpoints, within the 4-breakpoint limit.
        # PERF: It should be added at the AI level, not at the API level.
        for msg in openai_messages:
            if msg["role"] == "system" and isinstance(msg["content"], list) and msg["content"]:
                msg["content"][-1]["cache_control"] = _CACHE_CONTROL  # type: ignore[typeddict-unknown-key]
                break
        last = openai_messages[-1]
        if isinstance(last["content"], list) and last["content"]:
            last["content"][-1]["cache_control"] = _CACHE_CONTROL  # type: ignore[typeddict-unknown-key]
        return openai_messages
