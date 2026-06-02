# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from pathlib import Path

import dotenv

__all__ = [
    "SINGLE_CLIENT_ID",
    "get_home",
    "load_env",
    "is_dev",
    "is_managed",
    "is_single_client",
    "get_llm_and_key",
    "get_llm_base_url",
    "get_vlm_and_key",
    "get_vlm_base_url",
    "get_proxy",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_home() -> Path:
    """Return the CopilotJ home directory.

    Priority:
    1. COPILOTJ_HOME env var (explicit override or set by Java/Appose)
    2. Project root derived from this file's location (CLI dev mode)
    """
    explicit = os.getenv("COPILOTJ_HOME")
    if explicit:
        return Path(explicit)
    return _PROJECT_ROOT


def load_env() -> None:
    home = get_home()
    dotenv.load_dotenv(home / ".env", override=False)
    dotenv.load_dotenv(home / ".env.local", override=False)


def is_dev() -> bool:
    return os.getenv("COPILOTJ_DEV") is not None


def is_managed() -> bool:
    return os.getenv("COPILOTJ_MANAGED") is not None


SINGLE_CLIENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def is_single_client() -> bool:
    explicit = os.getenv("COPILOTJ_SINGLE_CLIENT")
    if explicit is not None:
        return explicit.lower() not in ("0", "false", "no")
    return is_dev() or is_managed()


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


def get_vlm_base_url() -> str | None:
    return os.getenv("COPILOTJ_VLM_BASE_URL", None) or os.getenv("COPILOTJ_BASE_URL", None)


def get_proxy(default_value: str | None = None) -> str | None:
    return default_value or os.getenv("COPILOTJ_PROXY", None)
