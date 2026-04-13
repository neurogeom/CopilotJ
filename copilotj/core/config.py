# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import os

import dotenv

__all__ = ["load_env", "is_dev", "get_llm_and_key", "get_llm_base_url", "get_vlm_and_key", "get_proxy"]


def load_env() -> None:
    dotenv.load_dotenv(".env")
    dotenv.load_dotenv(".env.local")


def is_dev() -> bool:
    return os.getenv("COPILOTJ_DEV") is not None


def get_llm_and_key(model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    model = model or os.getenv("COPILOTJ_MODEL", "")
    api_key = api_key or os.getenv("COPILOTJ_API_KEY", "") or ""
    return model, api_key


def get_llm_base_url() -> str | None:
    return os.getenv("COPILOTJ_BASE_URL", None)


def get_vlm_and_key(model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    model = model or os.getenv("COPILOTJ_VLM_MODEL") or os.getenv("COPILOTJ_MODEL", "")
    api_key = api_key or os.getenv("COPILOTJ_VLM_API_KEY", os.getenv("COPILOTJ_API_KEY", "")) or ""
    return model, api_key


def get_proxy(default_value: str | None = None) -> str | None:
    return default_value or os.getenv("COPILOTJ_PROXY", None)
