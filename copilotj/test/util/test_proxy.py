# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the proxy helpers in :mod:`copilotj.util.env`.

Covers :func:`proxy_dict` and the scoped :func:`temporary_proxy` context manager
(the mechanism used for the env-only cellpose/stardist weight downloads).
"""

from __future__ import annotations

import os

import pytest

from copilotj.core.config import Config
from copilotj.util import proxy_dict, temporary_proxy

_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@pytest.fixture
def clean_proxy_env(monkeypatch):
    """Remove any pre-existing proxy env vars so tests start from a clean baseline."""
    for key in _PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------- #
# proxy_dict
# --------------------------------------------------------------------------- #


def test_proxy_dict_returns_dict_when_set():
    proxies = proxy_dict(Config(cij_proxy="http://127.0.0.1:8080"))
    assert proxies == {"http": "http://127.0.0.1:8080", "https": "http://127.0.0.1:8080"}


def test_proxy_dict_none_when_unset():
    assert proxy_dict(Config()) is None


# --------------------------------------------------------------------------- #
# temporary_proxy (scoped env — the cellpose/stardist path)
# --------------------------------------------------------------------------- #


def test_temporary_proxy_sets_env_from_cij(monkeypatch, clean_proxy_env):
    cfg = Config(cij_proxy="http://127.0.0.1:8080")
    with temporary_proxy(cfg):
        assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:8080"
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:8080"
        assert os.environ["ALL_PROXY"] == "http://127.0.0.1:8080"
    # Restored on exit
    for key in _PROXY_ENV_KEYS:
        assert key not in os.environ


def test_temporary_proxy_restores_prior_value(monkeypatch, clean_proxy_env):
    monkeypatch.setenv("HTTP_PROXY", "http://pre-existing:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://pre-existing:1")
    cfg = Config(cij_proxy="http://127.0.0.1:8080")
    with temporary_proxy(cfg):
        assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:8080"
    assert os.environ["HTTP_PROXY"] == "http://pre-existing:1"


def test_temporary_proxy_noop_without_proxy(clean_proxy_env):
    with temporary_proxy(Config()):
        pass
    for key in _PROXY_ENV_KEYS:
        assert key not in os.environ
