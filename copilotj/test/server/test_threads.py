# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for copilotj.server.threads._resolve_config.

Covers the explicit ``use_server`` / explicit-model union: a ``use_server`` slot
keeps the server's env config, while an explicit model is applied verbatim (a
null ``api_key`` means "no key" — it no longer borrows the server's key).
"""

from __future__ import annotations

from unittest.mock import patch

from copilotj.core.config import Config
from copilotj.server.threads import _ConfigQuery, _resolve_config


def _base_cfg() -> Config:
    return Config(
        llm_model="server-model",
        llm_api_key="server-key",
        llm_base_url="https://server.example/v1",
        llm_provider="openai",
        vlm_model="server-vlm",
        vlm_api_key="server-vlm-key",
        vlm_base_url="https://vlm.example/v1",
        vlm_provider="openai",
    )


def _resolve(cfg: Config, query: dict) -> Config:
    """Resolve a query, bypassing vision capability lookup (network/model DB)."""
    with patch("copilotj.server.threads.resolve_vision_config", side_effect=lambda c: c):
        return _resolve_config(cfg, _ConfigQuery.model_validate(query))


def test_use_server_keeps_server_config() -> None:
    resolved = _resolve(_base_cfg(), {"model": {"use_server": True}})
    assert resolved.llm_model == "server-model"
    assert resolved.llm_api_key == "server-key"
    assert resolved.llm_base_url == "https://server.example/v1"


def test_explicit_null_api_key_does_not_borrow_server_key() -> None:
    """A user model with no key must NOT silently use the server's key."""
    resolved = _resolve(
        _base_cfg(),
        {"model": {"name": "user-model", "api_key": None, "base_url": None, "provider": "openai"}},
    )
    assert resolved.llm_model == "user-model"
    assert resolved.llm_api_key is None  # not "server-key"
    assert resolved.llm_base_url is None


def test_explicit_model_applied_verbatim() -> None:
    resolved = _resolve(
        _base_cfg(),
        {
            "model": {
                "name": "user-model",
                "api_key": "user-key",
                "base_url": "https://user.example/v1",
                "provider": "anthropic",
            }
        },
    )
    assert resolved.llm_model == "user-model"
    assert resolved.llm_api_key == "user-key"
    assert resolved.llm_base_url == "https://user.example/v1"
    assert resolved.llm_provider == "anthropic"


def test_vlm_use_server_keeps_server_vlm() -> None:
    resolved = _resolve(_base_cfg(), {"vlm": {"use_server": True}})
    assert resolved.vlm_model == "server-vlm"
    assert resolved.vlm_api_key == "server-vlm-key"


def test_vlm_explicit_applied_verbatim() -> None:
    resolved = _resolve(
        _base_cfg(),
        {"vlm": {"name": "user-vlm", "api_key": None, "base_url": None, "provider": "openai"}},
    )
    assert resolved.vlm_model == "user-vlm"
    assert resolved.vlm_api_key is None


def test_none_model_keeps_server_config() -> None:
    """Omitting the model slot (legacy) still means 'use the server's config'."""
    resolved = _resolve(_base_cfg(), {"vision_enabled": True})
    assert resolved.llm_model == "server-model"
    assert resolved.llm_api_key == "server-key"
