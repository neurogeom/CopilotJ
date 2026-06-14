# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for copilotj.core.model_listing and the catalog listing helper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from copilotj.core.model_info import list_catalog_models
from copilotj.core.model_listing import (
    DEFAULT_OLLAMA_URL,
    list_ollama_models,
    list_provider_models,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CATALOG: dict[str, dict] = {
    "gpt-4o": {
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_vision": True,
        "max_input_tokens": 128000,
    },
    "ft:gpt-4o": {  # fine-tune placeholder -> dropped
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-3.5-turbo": {
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "text-embedding-3-small": {  # embedding mode -> dropped
        "litellm_provider": "openai",
        "mode": "embedding",
    },
    "claude-opus-4-5": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_vision": True,
        "max_input_tokens": 200000,
    },
    "gemini/gemini-2.5-pro": {  # provider/ prefix -> stripped
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_vision": True,
        "max_input_tokens": 1048576,
    },
    "gemini/gemini-2.5-flash": {
        "litellm_provider": "gemini",
        "mode": "chat",
    },
    "openrouter/anthropic/claude-3-haiku": {  # openrouter/ prefix stripped; nested "/" kept
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_vision": True,
        "max_input_tokens": 200000,
    },
    "deepseek/deepseek-chat": {  # native deepseek entry; supplement adds the v4 family
        "litellm_provider": "deepseek",
        "mode": "chat",
        "supports_function_calling": True,
        "max_input_tokens": 131072,
    },
    "deepseek/deepseek-v3.2": {
        "litellm_provider": "deepseek",
        "mode": "chat",
        "supports_function_calling": True,
        "max_input_tokens": 163840,
    },
}


@pytest.fixture(autouse=True)
def _reset_db_cache():
    """Reset the module-level DB cache between tests."""
    import copilotj.core.model_info as mod

    mod._db_cache = None
    yield
    mod._db_cache = None


# ---------------------------------------------------------------------------
# list_catalog_models
# ---------------------------------------------------------------------------


class TestListCatalogModels:
    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_CATALOG)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_openai_filters_ft_and_embedding(self, _dl, _load):
        models = list_catalog_models("openai")
        ids = [m.id for m in models]
        assert ids == ["gpt-3.5-turbo", "gpt-4o"]  # sorted; ft: and embedding excluded
        assert not any(i.startswith("ft:") for i in ids)

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_CATALOG)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_anthropic(self, _dl, _load):
        models = list_catalog_models("anthropic")
        assert [m.id for m in models] == ["claude-opus-4-5"]
        assert models[0].supports_vision is True
        assert models[0].context_window == 200000

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_CATALOG)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_gemini_prefix_stripped(self, _dl, _load):
        ids = [m.id for m in list_catalog_models("gemini")]
        assert ids == ["gemini-2.5-flash", "gemini-2.5-pro"]
        assert not any(i.startswith("gemini/") for i in ids)

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_CATALOG)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_unknown_provider_empty(self, _dl, _load):
        assert list_catalog_models("ollama") == []
        assert list_catalog_models("nonexistent") == []

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_CATALOG)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_openrouter_nested_slash_kept(self, _dl, _load):
        models = list_catalog_models("openrouter")
        assert [m.id for m in models] == ["anthropic/claude-3-haiku"]
        assert models[0].supports_vision is True
        assert models[0].context_window == 200000

    @patch("copilotj.core.model_info._load_db", return_value={})
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_empty_db(self, _dl, _load):
        assert list_catalog_models("openai") == []


# ---------------------------------------------------------------------------
# _SUPPLEMENTAL_MODELS merge
# ---------------------------------------------------------------------------


class TestSupplementalModels:
    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_CATALOG)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_deepseek_v4_appended_when_absent(self, _dl, _load):
        ids = [m.id for m in list_catalog_models("deepseek")]
        # catalog contributes the native entries...
        assert "deepseek-chat" in ids
        assert "deepseek-v3.2" in ids
        # ...the supplement fills the v4 gap the upstream catalog hasn't shipped
        assert "deepseek-v4" in ids
        assert "deepseek-v4-pro" in ids
        assert "deepseek-v4-flash" in ids
        # result stays sorted, no duplicates
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_CATALOG)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_supplement_yields_to_catalog(self, _dl, _load):
        # Simulate the upstream catalog finally shipping deepseek-v4: the catalog
        # entry must win and no duplicate may appear.
        with patch(
            "copilotj.core.model_info._load_db",
            return_value={
                **SAMPLE_CATALOG,
                "deepseek/deepseek-v4": {
                    "litellm_provider": "deepseek",
                    "mode": "chat",
                    "supports_vision": True,
                    "max_input_tokens": 128000,
                },
            },
        ):
            models = list_catalog_models("deepseek")
        v4 = [m for m in models if m.id == "deepseek-v4"]
        assert len(v4) == 1  # catalog wins, supplement dropped — no duplicate
        assert v4[0].supports_vision is True  # catalog value, not supplement default
        assert v4[0].context_window == 128000

    @patch("copilotj.core.model_info._load_db", return_value={})
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_supplement_surfaces_when_catalog_unavailable(self, _dl, _load):
        # With no catalog at all, the deepseek picker still shows the v4 family.
        ids = [m.id for m in list_catalog_models("deepseek")]
        assert ids == ["deepseek-v4", "deepseek-v4-flash", "deepseek-v4-pro"]

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_CATALOG)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_no_supplement_for_other_providers(self, _dl, _load):
        # Providers without a supplement are untouched (exact list unchanged).
        assert [m.id for m in list_catalog_models("anthropic")] == ["claude-opus-4-5"]


# ---------------------------------------------------------------------------
# list_ollama_models
# ---------------------------------------------------------------------------


class TestListOllamaModels:
    def test_success(self):
        async def run():
            with (
                patch("copilotj.core.model_info._load_db", return_value={}),
                patch("copilotj.core.model_info._download_db_sync", return_value={}),
                patch(
                    "copilotj.core.model_listing._fetch_ollama_tags",
                    new=AsyncMock(
                        return_value={
                            "models": [
                                {"name": "llama3:8b"},
                                {"name": "llava:13b"},
                                {"model": "qwen2"},  # falls back to "model" key
                            ]
                        }
                    ),
                ),
            ):
                return await list_ollama_models("http://localhost:11434")

        models = asyncio.run(run())
        assert [m.id for m in models] == ["llama3:8b", "llava:13b", "qwen2"]
        assert next(m for m in models if m.id == "llava:13b").supports_vision is True  # heuristic
        assert next(m for m in models if m.id == "qwen2").supports_vision is False

    def test_failure_returns_none(self):
        async def run():
            with patch("copilotj.core.model_listing._fetch_ollama_tags", new=AsyncMock(return_value=None)):
                return await list_ollama_models("http://localhost:11434")

        assert asyncio.run(run()) is None

    def test_reachable_but_empty(self):
        async def run():
            with patch("copilotj.core.model_listing._fetch_ollama_tags", new=AsyncMock(return_value={"models": []})):
                return await list_ollama_models("http://localhost:11434")

        assert asyncio.run(run()) == []


# ---------------------------------------------------------------------------
# list_provider_models
# ---------------------------------------------------------------------------


class TestListProviderModels:
    def test_ollama_live(self):
        async def run():
            with (
                patch("copilotj.core.model_info._load_db", return_value={}),
                patch("copilotj.core.model_info._download_db_sync", return_value={}),
                patch(
                    "copilotj.core.model_listing._fetch_ollama_tags",
                    new=AsyncMock(return_value={"models": [{"name": "llama3"}]}),
                ),
            ):
                return await list_provider_models("ollama")

        d = asyncio.run(run())
        assert d["provider"] == "ollama"
        assert d["source"] == "live"
        assert d["models"][0]["id"] == "llama3"

    def test_ollama_unreachable(self):
        async def run():
            with patch("copilotj.core.model_listing._fetch_ollama_tags", new=AsyncMock(return_value=None)):
                return await list_provider_models("ollama")

        d = asyncio.run(run())
        assert d["source"] == "unreachable"
        assert d["models"] == []

    def test_ollama_reachable_but_empty(self):
        async def run():
            with patch(
                "copilotj.core.model_listing._fetch_ollama_tags",
                new=AsyncMock(return_value={"models": []}),
            ):
                return await list_provider_models("ollama")

        d = asyncio.run(run())
        assert d["source"] == "live"
        assert d["models"] == []

    def test_catalog_provider(self):
        async def run():
            with (
                patch("copilotj.core.model_info._load_db", return_value=SAMPLE_CATALOG),
                patch("copilotj.core.model_info._download_db_sync", return_value={}),
            ):
                return await list_provider_models("anthropic")

        d = asyncio.run(run())
        assert d["source"] == "catalog"
        assert [m["id"] for m in d["models"]] == ["claude-opus-4-5"]

    def test_default_ollama_url(self):
        assert DEFAULT_OLLAMA_URL == "http://localhost:11434"
