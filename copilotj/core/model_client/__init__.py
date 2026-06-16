# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import logging

from copilotj.core.config import Config, load_config

# Base types
from copilotj.core.model_client._types import (
    _VALID_PROVIDERS,
    FinishReasons,
    ModelClient,
    ModelProviderError,
    ModelResponse,
    ModelResponseChunk,
    ModelSyntaxError,
    ToolCall,
)

# Provider clients
from copilotj.core.model_client.anthropic import AnthropicChatCompletionClient
from copilotj.core.model_client.gemini import GeminiChatCompletionClient
from copilotj.core.model_client.ollama import OllamaChatCompletionClient
from copilotj.core.model_client.openai_chat_completion import OpenAIChatCompletionClient
from copilotj.core.model_client.openai_response import OpenAIResponseClient
from copilotj.core.model_client.openrouter import OpenRouterChatCompletionClient

logger = logging.getLogger(__name__)

__all__ = [
    # Types / base
    "FinishReasons",
    "ToolCall",
    "ModelResponse",
    "ModelResponseChunk",
    "ModelClient",
    "ModelSyntaxError",
    "ModelProviderError",
    # Providers
    "OpenAIChatCompletionClient",
    "OpenAIResponseClient",
    "GeminiChatCompletionClient",
    "OllamaChatCompletionClient",
    "AnthropicChatCompletionClient",
    "OpenRouterChatCompletionClient",
    # Factory functions
    "new_model_client",
    "new_vlm_model_client",
]


def new_model_client(cfg: Config) -> ModelClient:
    return _new_model_client(
        cfg.llm_model,
        cfg.llm_api_key,
        proxy=cfg.llm_proxy,
        base_url=cfg.llm_base_url,
        cfg=cfg,
        provider=cfg.llm_provider,
    )


def new_vlm_model_client(cfg: Config) -> ModelClient:
    return _new_model_client(
        cfg.vlm_model,
        cfg.vlm_api_key,
        proxy=cfg.llm_proxy,
        base_url=cfg.vlm_base_url,
        cfg=cfg,
        provider=cfg.vlm_provider or cfg.llm_provider,
    )


def _strip_provider_prefix(model: str) -> tuple[str | None, str]:
    """Strip known provider prefixes from model name.

    Returns (provider_hint, stripped_model_name).
    """
    for prefix, provider in (("ollama/", "ollama"), ("deepseek/", "deepseek")):
        if model.startswith(prefix):
            return provider, model.split("/", 1)[1]
    return None, model


def _resolve_client(
    provider: str,
    model: str,
    api_key: str,
    *,
    proxy: str | None,
    base_url: str | None,
) -> ModelClient:
    """Create a ModelClient for a known provider string."""
    if provider not in _VALID_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid providers: {', '.join(_VALID_PROVIDERS)}")

    match provider:
        case "openai":
            return OpenAIChatCompletionClient(model, api_key, base_url=base_url, proxy=proxy)
        case "openai-responses":
            return OpenAIResponseClient(model, api_key, base_url=base_url, proxy=proxy)
        case "anthropic":
            return AnthropicChatCompletionClient(model, api_key, base_url=base_url, proxy=proxy)
        case "gemini":
            return GeminiChatCompletionClient(model, api_key, proxy=proxy, base_url=base_url)
        case "ollama":
            return OllamaChatCompletionClient(model=model, base_url=base_url)
        case "deepseek":
            url = base_url or "https://api.deepseek.com"
            return OpenAIChatCompletionClient(model=model, api_key=api_key, base_url=url, proxy=proxy)
        case "siliconflow":
            url = base_url or "https://api.siliconflow.cn/v1"
            return OpenAIChatCompletionClient(model, api_key, base_url=url, proxy=proxy)
        case "openrouter":
            url = base_url or "https://openrouter.ai/api/v1"
            return OpenRouterChatCompletionClient(model=model, api_key=api_key, base_url=url, proxy=proxy)
        case "openai-compatible":
            return OpenAIChatCompletionClient(model, api_key, base_url=base_url, proxy=proxy)


def _new_model_client(
    model: str,
    api_key: str,
    *,
    proxy: str | None,
    base_url: str | None = None,
    cfg: Config | None = None,
    provider: str | None = None,
) -> ModelClient:
    cfg = cfg or load_config()
    proxy = proxy or cfg.llm_proxy

    # If provider is explicitly given, use it directly.
    if provider is not None:
        # Strip any known prefix from model name (e.g. "ollama/llama3" -> "llama3")
        _, model = _strip_provider_prefix(model)
        return _resolve_client(provider=provider, model=model, api_key=api_key, proxy=proxy, base_url=base_url)

    # Backward-compatible prefix-based detection
    logger.warning(
        "Auto-detecting provider from model name '%s'. "
        "Set COPILOTJ_LLM_PROVIDER explicitly to remove this warning. "
        "Valid values: %s",
        model,
        ", ".join(_VALID_PROVIDERS),
    )
    if model.startswith("ollama/"):
        model_name = model.split("/", 1)[1]
        return OllamaChatCompletionClient(model=model_name)

    elif model.startswith("deepseek/"):
        model_name = model.split("/", 1)[1]
        url = base_url or "https://api.deepseek.com"
        return OpenAIChatCompletionClient(model=model_name, api_key=api_key, base_url=url, proxy=proxy)

    elif model.startswith("gemini-"):
        return GeminiChatCompletionClient(model, api_key, proxy=proxy, base_url=base_url)

    elif model.startswith("claude-"):
        return AnthropicChatCompletionClient(model, api_key, base_url=base_url, proxy=proxy)

    elif model.startswith("gpt-"):
        if base_url:
            return OpenAIChatCompletionClient(model, api_key, base_url=base_url, proxy=proxy)
        else:
            return OpenAIResponseClient(model, api_key, proxy=proxy)

    elif model.startswith("zai-org/") or model.startswith("Pro/"):
        url = base_url or "https://api.siliconflow.cn/v1"
        return OpenAIChatCompletionClient(model, api_key, base_url=url, proxy=proxy)

    return OpenAIChatCompletionClient(model, api_key, base_url=base_url, proxy=proxy)
