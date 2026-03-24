# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import nullcontext
from typing import Any

import langfuse as _langfuse
import langfuse.openai as _langfuse_openai
import openai


class NoOpLangfuse:
    def start_as_current_observation(self, **_: Any):
        return nullcontext()


LangfuseClient = _langfuse.Langfuse | NoOpLangfuse


def _is_enabled(value: str | None) -> bool:
    if value is None:
        return False

    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_langfuse_enabled() -> bool:
    return (
        _is_enabled(os.getenv("LANGFUSE_ENABLED"))
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


def get_langfuse_client() -> LangfuseClient:
    if is_langfuse_enabled():
        return _langfuse.get_client()

    return NoOpLangfuse()


def new_langfuse_client() -> LangfuseClient:
    if is_langfuse_enabled():
        return _langfuse.Langfuse()

    return NoOpLangfuse()


def new_async_openai_client(*, api_key: str, http_client: Any = None, base_url: str | None = None):
    client_kwargs = {"api_key": api_key, "http_client": http_client, "base_url": base_url}
    if is_langfuse_enabled():
        return _langfuse_openai.AsyncOpenAI(**client_kwargs)

    return openai.AsyncOpenAI(**client_kwargs)
