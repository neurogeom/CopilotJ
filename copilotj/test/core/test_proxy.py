# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``CIJ_PROXY`` download-proxy behavior in :mod:`copilotj.core`.

Covers :func:`Config.cij_proxy` loading (with no ``os.environ`` mutation) and the
LLM/VLM client proxy precedence, including the ``vlm_proxy`` regression
(``new_vlm_model_client`` previously ignored ``cfg.vlm_proxy``).
"""

from __future__ import annotations

import os

from copilotj.core.config import Config, load_config
from copilotj.core.model_client import new_model_client, new_vlm_model_client

_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def test_cij_proxy_loaded_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("COPILOTJ_HOME", str(tmp_path))  # isolate from any real .env
    monkeypatch.setenv("CIJ_PROXY", "http://127.0.0.1:8080")
    cfg = load_config()
    assert cfg.cij_proxy == "http://127.0.0.1:8080"


def test_cij_proxy_unset_defaults_none(monkeypatch, tmp_path):
    monkeypatch.setenv("COPILOTJ_HOME", str(tmp_path))
    monkeypatch.delenv("CIJ_PROXY", raising=False)
    cfg = load_config()
    assert cfg.cij_proxy is None


def test_load_config_does_not_mutate_proxy_env(monkeypatch, tmp_path):
    """The explicit approach must never write proxy env vars."""
    for key in _PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("COPILOTJ_HOME", str(tmp_path))
    monkeypatch.setenv("CIJ_PROXY", "http://127.0.0.1:8080")
    load_config()
    for key in _PROXY_ENV_KEYS:
        assert key not in os.environ


# --------------------------------------------------------------------------- #
# LLM / VLM client proxy precedence
# --------------------------------------------------------------------------- #


def _capture_resolve(monkeypatch, captured: dict):
    """Monkeypatch ``_resolve_client`` to record the proxy kwarg and return a dummy."""

    def fake_resolve(provider, model, api_key, *, proxy, base_url):  # noqa: ARG001
        captured["proxy"] = proxy
        captured["provider"] = provider
        return object()

    monkeypatch.setattr("copilotj.core.model_client._resolve_client", fake_resolve)


def test_llm_client_prefers_llm_proxy(monkeypatch):
    captured: dict = {}
    _capture_resolve(monkeypatch, captured)
    cfg = Config(llm_model="gpt-4o", llm_provider="openai", llm_proxy="http://llm:1", cij_proxy="http://cij:1")
    new_model_client(cfg)
    assert captured["proxy"] == "http://llm:1"


def test_llm_client_falls_back_to_cij_proxy(monkeypatch):
    captured: dict = {}
    _capture_resolve(monkeypatch, captured)
    cfg = Config(llm_model="gpt-4o", llm_provider="openai", cij_proxy="http://cij:1")
    new_model_client(cfg)
    assert captured["proxy"] == "http://cij:1"


def test_vlm_client_prefers_vlm_proxy(monkeypatch):
    """Regression: ``cfg.vlm_proxy`` must win (previously ignored — fell back to llm_proxy)."""
    captured: dict = {}
    _capture_resolve(monkeypatch, captured)
    cfg = Config(
        llm_model="llm",
        vlm_model="gpt-4o",
        vlm_provider="openai",
        vlm_proxy="http://vlm:1",
        llm_proxy="http://llm:1",
        cij_proxy="http://cij:1",
    )
    new_vlm_model_client(cfg)
    assert captured["proxy"] == "http://vlm:1"


def test_vlm_client_excludes_llm_proxy(monkeypatch):
    """VLM proxy precedence is ``vlm_proxy > cij_proxy`` — ``llm_proxy`` is never used."""
    captured: dict = {}
    _capture_resolve(monkeypatch, captured)
    # vlm_proxy unset, both llm_proxy and cij_proxy set → cij_proxy wins
    cfg = Config(
        llm_model="llm", vlm_model="gpt-4o", vlm_provider="openai", llm_proxy="http://llm:1", cij_proxy="http://cij:1"
    )
    new_vlm_model_client(cfg)
    assert captured["proxy"] == "http://cij:1"

    captured.clear()
    # vlm_proxy and cij_proxy unset, llm_proxy set → no proxy (llm_proxy excluded)
    cfg = Config(llm_model="llm", vlm_model="gpt-4o", vlm_provider="openai", llm_proxy="http://llm:1")
    new_vlm_model_client(cfg)
    assert captured["proxy"] is None
