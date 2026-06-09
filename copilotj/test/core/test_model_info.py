# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for copilotj.core.model_info."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from copilotj.core.model_info import (
    _normalize_model_name,
    _lookup_model,
    _ollama_vision_heuristic,
    get_model_capabilities,
    supports_vision,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DB: dict[str, dict] = {
    "gpt-4o": {
        "supports_vision": True,
        "supports_function_calling": True,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "openai/gpt-4o-mini": {
        "supports_vision": True,
        "supports_function_calling": True,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
    },
    "gpt-3.5-turbo": {
        "supports_vision": False,
        "supports_function_calling": True,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
    },
    "ollama/llava": {
        "supports_vision": True,
        "supports_function_calling": False,
        "max_input_tokens": 4096,
    },
    "deepseek/deepseek-chat": {
        "supports_vision": False,
        "supports_function_calling": True,
        "max_input_tokens": 65536,
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
# _normalize_model_name
# ---------------------------------------------------------------------------


class TestNormalizeModelName:
    def test_bare_name(self):
        assert _normalize_model_name("gpt-4o") == (None, "gpt-4o")

    def test_openai_prefix(self):
        assert _normalize_model_name("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_ollama_prefix(self):
        assert _normalize_model_name("ollama/llava") == ("ollama", "llava")

    def test_deepseek_prefix(self):
        assert _normalize_model_name("deepseek/deepseek-chat") == ("deepseek", "deepseek-chat")

    def test_nested_slash(self):
        # Only the first slash is split
        assert _normalize_model_name("provider/model/sub") == ("provider", "model/sub")


# ---------------------------------------------------------------------------
# _lookup_model
# ---------------------------------------------------------------------------


class TestLookupModel:
    def test_exact_match(self):
        entry = _lookup_model(SAMPLE_DB, "gpt-4o")
        assert entry is not None
        assert entry["supports_vision"] is True

    def test_bare_name_match(self):
        """Provider-prefixed query → bare name match in DB."""
        entry = _lookup_model(SAMPLE_DB, "openai/gpt-4o")
        assert entry is not None
        assert entry["supports_vision"] is True

    def test_provider_prefix_match(self):
        """Bare name query → provider-prefixed match in DB."""
        entry = _lookup_model(SAMPLE_DB, "gpt-4o-mini")
        assert entry is not None
        # Should find "openai/gpt-4o-mini" via common prefix search
        assert entry["supports_vision"] is True

    def test_unknown_model(self):
        entry = _lookup_model(SAMPLE_DB, "nonexistent-model")
        assert entry is None

    def test_deepseek_with_prefix(self):
        entry = _lookup_model(SAMPLE_DB, "deepseek-chat")
        assert entry is not None
        assert entry["supports_vision"] is False


# ---------------------------------------------------------------------------
# _ollama_vision_heuristic
# ---------------------------------------------------------------------------


class TestOllamaVisionHeuristic:
    def test_known_vision_model(self):
        assert _ollama_vision_heuristic("ollama/llava") is True

    def test_known_vision_model_with_tag(self):
        assert _ollama_vision_heuristic("ollama/llava:13b") is True

    def test_known_gemma3(self):
        assert _ollama_vision_heuristic("ollama/gemma3:12b") is True

    def test_unknown_ollama_model(self):
        assert _ollama_vision_heuristic("ollama/llama3") is False

    def test_non_ollama_ignored(self):
        # The heuristic strips the prefix, so plain names are checked too
        assert _ollama_vision_heuristic("llava") is True


# ---------------------------------------------------------------------------
# get_model_capabilities
# ---------------------------------------------------------------------------


class TestGetModelCapabilities:
    def test_empty_model(self):
        caps = get_model_capabilities("")
        assert caps.supports_vision is False
        assert caps.source == "unknown"

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_known_vision_model(self, _mock_dl, _mock_load):
        caps = get_model_capabilities("gpt-4o")
        assert caps.supports_vision is True
        assert caps.supports_function_calling is True
        assert caps.context_window == 128000
        assert caps.max_output_tokens == 16384
        assert caps.source == "litellm_db"

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_known_non_vision_model(self, _mock_dl, _mock_load):
        caps = get_model_capabilities("gpt-3.5-turbo")
        assert caps.supports_vision is False
        assert caps.supports_function_calling is True
        assert caps.source == "litellm_db"

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_unknown_model_returns_conservative(self, _mock_dl, _mock_load):
        caps = get_model_capabilities("my-custom-model-v1")
        assert caps.supports_vision is False
        assert caps.source == "unknown"

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_ollama_vision_model(self, _mock_dl, _mock_load):
        # "ollama/llava" is in the sample DB → litellm_db source
        caps = get_model_capabilities("ollama/llava")
        assert caps.supports_vision is True
        assert caps.source == "litellm_db"

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_ollama_unknown_model_heuristic(self, _mock_dl, _mock_load):
        caps = get_model_capabilities("ollama/llama3")
        assert caps.supports_vision is False
        assert caps.source == "heuristic"

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_ollama_known_vision_heuristic(self, _mock_dl, _mock_load):
        # "ollama/moondream2" is NOT in SAMPLE_DB, but is in the heuristic set
        caps = get_model_capabilities("ollama/moondream2")
        assert caps.supports_vision is True
        assert caps.source == "heuristic"


# ---------------------------------------------------------------------------
# supports_vision
# ---------------------------------------------------------------------------


class TestSupportsVision:
    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_vision_model(self, _mock_dl, _mock_load):
        assert supports_vision("gpt-4o") is True

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_non_vision_model(self, _mock_dl, _mock_load):
        assert supports_vision("gpt-3.5-turbo") is False

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_unknown_model(self, _mock_dl, _mock_load):
        assert supports_vision("unknown-model") is False


# ---------------------------------------------------------------------------
# resolve_vision_config
# ---------------------------------------------------------------------------


class TestResolveVisionConfig:
    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_detects_vision_capability(self, _mock_dl, _mock_load):
        from copilotj.core.config import Config, resolve_vision_config

        cfg = Config(llm_model="gpt-4o", llm_api_key="key")
        resolved = resolve_vision_config(cfg)
        assert resolved.llm_supports_vision is True
        assert resolved.vlm_configured is False
        # vision_enabled stays False — no auto-enable
        assert resolved.vision_enabled is False

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_no_vision_no_vlm(self, _mock_dl, _mock_load):
        from copilotj.core.config import Config, resolve_vision_config

        cfg = Config(llm_model="gpt-3.5-turbo", llm_api_key="key")
        resolved = resolve_vision_config(cfg)
        assert resolved.llm_supports_vision is False
        assert resolved.vlm_configured is False

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_separate_vlm_detected(self, _mock_dl, _mock_load):
        from copilotj.core.config import Config, resolve_vision_config

        cfg = Config(
            llm_model="gpt-3.5-turbo",
            llm_api_key="key",
            vlm_model="gpt-4o",
            vlm_api_key="vlm-key",
        )
        resolved = resolve_vision_config(cfg)
        assert resolved.llm_supports_vision is False
        assert resolved.vlm_configured is True

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_same_model_no_key_means_no_vlm(self, _mock_dl, _mock_load):
        """If vlm_model == llm_model, it's not a separate VLM."""
        from copilotj.core.config import Config, resolve_vision_config

        cfg = Config(llm_model="gpt-3.5-turbo", llm_api_key="key", vlm_model="gpt-3.5-turbo", vlm_api_key="key")
        resolved = resolve_vision_config(cfg)
        assert resolved.vlm_configured is False

    @patch("copilotj.core.model_info._load_db", return_value=SAMPLE_DB)
    @patch("copilotj.core.model_info._download_db_sync", return_value={})
    def test_vision_enabled_unaffected_by_capability(self, _mock_dl, _mock_load):
        """vision_enabled is an independent switch, never auto-set."""
        from copilotj.core.config import Config, resolve_vision_config

        # Model supports vision, but vision is NOT enabled
        cfg = Config(llm_model="gpt-4o", llm_api_key="key", vision_enabled=False)
        resolved = resolve_vision_config(cfg)
        assert resolved.vision_enabled is False
        assert resolved.llm_supports_vision is True

        # Model supports vision, and vision IS enabled
        cfg2 = Config(llm_model="gpt-4o", llm_api_key="key", vision_enabled=True)
        resolved2 = resolve_vision_config(cfg2)
        assert resolved2.vision_enabled is True
        assert resolved2.llm_supports_vision is True
