# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
from pathlib import Path

import dotenv

__all__ = ["load_env", "is_dev", "get_llm_and_key", "get_llm_base_url", "get_vlm_and_key", "get_proxy"]


def load_env() -> None:
    dotenv.load_dotenv(".env")
    dotenv.load_dotenv(".env.local")


def is_dev() -> bool:
    return os.getenv("COPILOTJ_DEV") is not None


def _load_oauth_token() -> str | None:
    """Load API key from OAuth authentication file as fallback."""
    try:
        auth_file = Path.home() / ".chatimej" / "auth.json"
        if auth_file.exists():
            auth_data = json.loads(auth_file.read_text())
            return auth_data.get("OPENAI_API_KEY")
    except Exception:
        pass
    return None


def get_llm_and_key(model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    model = model or os.getenv("COPILOTJ_MODEL", "")
    api_key = api_key or os.getenv("COPILOTJ_API_KEY") or _load_oauth_token() or ""
    return model, api_key


def get_llm_base_url() -> str | None:
    return os.getenv("COPILOTJ_BASE_URL", None)


def get_vlm_and_key(model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    model = model or os.getenv("COPILOTJ_VLM_MODEL") or os.getenv("COPILOTJ_MODEL", "")
    api_key = api_key or os.getenv("COPILOTJ_VLM_API_KEY") or os.getenv("COPILOTJ_API_KEY", "") or _load_oauth_token() or ""
    return model, api_key


def get_proxy(default_value: str | None = None) -> str | None:
    return default_value or os.getenv("COPILOTJ_PROXY", None)
