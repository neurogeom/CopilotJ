# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os

from copilotj.core.config import Config

__all__ = ["temporary_proxy"]


@contextlib.contextmanager
def temporary_proxy(cfg: Config, default_value: str | None = None):
    """Set a temporary proxy for the duration of the context

    Notes: not thread-safe, use with caution in multi-threaded environments.
    """
    proxy = default_value or cfg.llm_proxy
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
