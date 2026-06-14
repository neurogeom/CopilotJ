# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Provider model discovery for the frontend model picker.

Cloud providers (OpenAI / Anthropic / Gemini / …) are resolved from the cached
LiteLLM catalog (:func:`copilotj.core.model_info.list_catalog_models`); Ollama
is queried live at ``{base_url}/api/tags`` since only the locally installed
models are relevant.  The dispatcher :func:`list_provider_models` returns a
uniform ``{provider, source, models}`` shape consumed by the ``/api/models``
endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from copilotj.core.model_info import CatalogModel, get_model_capabilities, list_catalog_models

__all__ = [
    "DEFAULT_OLLAMA_URL",
    "list_ollama_models",
    "list_provider_models",
]

_log = logging.getLogger("copilotj.core.model_listing")

DEFAULT_OLLAMA_URL = "http://localhost:11434"


async def _fetch_ollama_tags(base_url: str, *, timeout: float) -> dict[str, Any] | None:
    """GET ``{base_url}/api/tags`` and return the parsed JSON, or ``None``.

    Separated from :func:`list_ollama_models` so tests can stub the network
    hop without spinning up a real Ollama instance.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)
    except Exception as exc:  # noqa: BLE001 - any failure means "unreachable"
        _log.debug("Ollama model listing failed for %s: %s", url, exc)
        return None


async def list_ollama_models(base_url: str, *, timeout: float = 2.0) -> list[CatalogModel] | None:
    """List models installed on a local Ollama instance.

    Performs a live ``GET {base_url}/api/tags``.  Ollama runs locally, so we
    request on demand without caching.  Returns ``None`` when Ollama is
    unreachable (not running / connection refused / timeout), or the list of
    installed models otherwise (empty when reachable but nothing is pulled).
    """
    data = await _fetch_ollama_tags(base_url, timeout=timeout)
    if not isinstance(data, dict):
        return None

    models: list[CatalogModel] = []
    for entry in data.get("models", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        caps = get_model_capabilities(f"ollama/{name}")
        models.append(
            CatalogModel(
                id=name,
                label=name,
                provider="ollama",
                supports_vision=caps.supports_vision,
                supports_function_calling=caps.supports_function_calling,
                context_window=caps.context_window,
            )
        )
    models.sort(key=lambda m: m.id)
    return models


def _model_to_dict(m: CatalogModel) -> dict[str, Any]:
    """Serialize a :class:`CatalogModel` to the ``/api/models`` item shape."""
    return {
        "id": m.id,
        "label": m.label,
        "supports_vision": m.supports_vision,
        "context_window": m.context_window,
    }


async def list_provider_models(provider: str, *, base_url: str | None = None) -> dict[str, Any]:
    """Resolve available models for *provider* into a uniform dict.

    - ``ollama`` → live ``/api/tags`` at ``base_url`` (default
      :data:`DEFAULT_OLLAMA_URL`); ``source`` is ``"live"`` when Ollama is
      reachable (empty list when no models are pulled), or ``"unreachable"``
      when the request fails (Ollama not running / refused / timeout).
    - any other provider → filtered LiteLLM catalog; ``source`` is
      ``"catalog"`` (empty list when nothing matches).

    Never raises — callers can ``asyncio.gather`` over providers safely.
    """
    if provider == "ollama":
        models = await list_ollama_models(base_url or DEFAULT_OLLAMA_URL)
        if models is None:
            return {"provider": provider, "source": "unreachable", "models": []}
        return {
            "provider": provider,
            "source": "live",
            "models": [_model_to_dict(m) for m in models],
        }

    models = list_catalog_models(provider)
    return {"provider": provider, "source": "catalog", "models": [_model_to_dict(m) for m in models]}
