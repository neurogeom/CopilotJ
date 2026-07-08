# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os

from copilotj.core.config import Config

__all__ = ["proxy_dict", "temporary_proxy"]


def proxy_dict(cfg: Config) -> dict[str, str] | None:
    """Return an explicit ``{"http":..,"https":..}`` proxies dict from ``cfg.cij_proxy``.

    Returns ``None`` when no download proxy is configured, so callers can pass the
    result directly to clients that accept ``proxies=None`` (e.g. ``requests``,
    ``TavilyClient``).

    >>> from copilotj.core.config import Config
    >>> proxy_dict(Config(cij_proxy="http://127.0.0.1:8080"))
    {'http': 'http://127.0.0.1:8080', 'https': 'http://127.0.0.1:8080'}
    >>> proxy_dict(Config()) is None
    True
    """
    return {"http": cfg.cij_proxy, "https": cfg.cij_proxy} if cfg.cij_proxy else None


@contextlib.contextmanager
def temporary_proxy(cfg: Config, default_value: str | None = None):
    """Set a temporary proxy for the duration of the context

    Notes: not thread-safe, use with caution in multi-threaded environments.
    """
    proxy = default_value or cfg.llm_proxy or cfg.cij_proxy
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]

    old_env = {k: os.environ.get(k) for k in keys + [k.lower() for k in keys]}

    if proxy:
        for key in keys:
            os.environ[key] = proxy
            os.environ[key.lower()] = proxy

    try:
        yield
    finally:
        for key, val in old_env.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)
