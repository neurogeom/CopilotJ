# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import dotenv

__all__ = [
    "Config",
    "load_config",
    "SINGLE_CLIENT_ID",
    "get_home",
    "is_dev",
    "is_managed",
    "is_single_client",
    "load_managed_config",
    "save_managed_config",
    "get_llm_and_key",
    "get_llm_base_url",
    "get_vlm_and_key",
    "get_vlm_base_url",
    "get_proxy",
    "bootstrap_assets",
    "resolve_vision_config",
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


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables.

    Created once at server startup via ``load_config()`` and passed through
    the constructor chain (Server → Threads → LeaderDriven → …).
    Each component reads only the fields it needs.
    """

    # LLM
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str | None = None

    # VLM (already resolved: VLM_* falls back to generic *)
    vlm_model: str = ""
    vlm_api_key: str = ""
    vlm_base_url: str | None = None

    # Network
    proxy: str | None = None

    # Tool keys
    tavily_api_key: str | None = None

    # Feature flags
    kb_autosave: bool = False
    dev: bool = False
    vision_enabled: bool = False

    # Auto-detected model capabilities (set by resolve_vision_config, info-only)
    llm_supports_vision: bool = False  # Whether the LLM model supports vision (from model DB)
    vlm_configured: bool = False  # Whether a separate VLM is explicitly configured

    # BioImage Model Zoo
    bioimage_model_zoo_url: str = "https://bioimage-io.github.io/collection-bioimage-io/collection.json"
    bioimage_model_zoo_cache: str = ""
    bioimage_model_zoo_cache_ttl: int = 86400

    @property
    def vision_available(self) -> bool:
        """Whether vision can actually work (main model supports it or separate VLM configured)."""
        return self.llm_supports_vision or self.vlm_configured


def load_config() -> Config:
    """Load .env files and return a Config object.

    Called once at server startup (or CLI entry).  Returns the populated
    Config directly — no global state is stored.
    """
    home = get_home()
    dotenv.load_dotenv(home / ".env", override=False)
    dotenv.load_dotenv(home / ".env.local", override=False)

    return Config(
        llm_model=os.getenv("COPILOTJ_MODEL", ""),
        llm_api_key=os.getenv("COPILOTJ_API_KEY", "") or "",
        llm_base_url=os.getenv("COPILOTJ_BASE_URL", None),
        vlm_model=os.getenv("COPILOTJ_VLM_MODEL") or os.getenv("COPILOTJ_MODEL", ""),
        vlm_api_key=os.getenv("COPILOTJ_VLM_API_KEY", os.getenv("COPILOTJ_API_KEY", "")) or "",
        vlm_base_url=os.getenv("COPILOTJ_VLM_BASE_URL", None) or os.getenv("COPILOTJ_BASE_URL", None),
        proxy=os.getenv("COPILOTJ_PROXY", None),
        tavily_api_key=os.getenv("COPILOTJ_TAVILY_API_KEY", None),
        kb_autosave=os.getenv("COPILOTJ_KB_AUTOSAVE", "0") == "1",
        dev=os.getenv("COPILOTJ_DEV") is not None,
        vision_enabled=os.getenv("COPILOTJ_VISION_ENABLED", "0") == "1",
        bioimage_model_zoo_url=os.getenv(
            "BIOIMAGE_MODEL_ZOO_URL",
            "https://bioimage-io.github.io/collection-bioimage-io/collection.json",
        ),
        bioimage_model_zoo_cache=os.getenv(
            "BIOIMAGE_MODEL_ZOO_CACHE",
            str(Path(__file__).resolve().parent.parent.parent / "temp" / "bioimage_model_zoo"),
        ),
        bioimage_model_zoo_cache_ttl=int(os.getenv("BIOIMAGE_MODEL_ZOO_CACHE_TTL", "86400")),
    )


_log = logging.getLogger("copilotj.core.config")


def resolve_vision_config(cfg: Config) -> Config:
    """Detect model capabilities and populate info fields.

    Called after ``load_config()`` at server startup.  Uses the LiteLLM
    model database to check whether the configured LLM supports vision.

    This does **not** change ``vision_enabled`` — that remains the user's
    explicit opt-in switch.  The detected info fields are used by the
    frontend for UX guidance (e.g. suggesting the user enable vision
    when the model supports it, or warning when vision is enabled but
    the model lacks vision support).
    """
    from copilotj.core.model_info import get_model_capabilities

    # Detect main model vision capability
    main_caps = get_model_capabilities(cfg.llm_model) if cfg.llm_model else None
    llm_supports_vision = bool(main_caps and main_caps.supports_vision)

    # Check if a separate VLM is explicitly configured
    vlm_configured = bool(cfg.vlm_model and cfg.vlm_model != cfg.llm_model)

    if llm_supports_vision:
        _log.info("Model %s supports vision (source=%s)", cfg.llm_model, main_caps.source)
    else:
        _log.info("Model %s does not support vision", cfg.llm_model)

    # Warn if vision is enabled but the model can't handle it and no VLM fallback exists
    if cfg.vision_enabled and not llm_supports_vision and not vlm_configured:
        _log.warning(
            "Vision is enabled but model %s does not support vision and no separate VLM is configured. "
            "Set COPILOTJ_VLM_MODEL to configure a vision-capable model.",
            cfg.llm_model,
        )

    return replace(cfg, llm_supports_vision=llm_supports_vision, vlm_configured=vlm_configured)


# ---------------------------------------------------------------------------
# Legacy helpers — still used by bridge.py, plugin/api.py, appose_worker.py
# ---------------------------------------------------------------------------


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


def load_managed_config() -> dict:
    path = get_home() / "config.json"
    if path.is_file():
        return json.loads(path.read_text("utf-8"))
    return {}


def save_managed_config(data: dict) -> None:
    if not is_managed():
        return
    path = get_home() / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), "utf-8")


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


def bootstrap_assets() -> None:
    """Copy assets/ from project source to COPILOTJ_HOME if missing."""
    home = get_home()
    source_root = Path(__file__).resolve().parent.parent.parent
    source_assets = source_root / "assets"
    target_assets = home / "assets"

    if not source_assets.exists():
        return
    if target_assets.exists() and any(target_assets.iterdir()):
        return

    shutil.copytree(source_assets, target_assets, dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Private helpers — used internally by model_client.py.
# External code should read fields directly from the Config object.
# ---------------------------------------------------------------------------


def _get_llm_and_key(cfg: Config, model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    return model or cfg.llm_model, api_key or cfg.llm_api_key


def _get_llm_base_url(cfg: Config) -> str | None:
    return cfg.llm_base_url


def _get_vlm_and_key(cfg: Config, model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    return model or cfg.vlm_model, api_key or cfg.vlm_api_key


def _get_vlm_base_url(cfg: Config) -> str | None:
    return cfg.vlm_base_url


def _get_proxy(cfg: Config, default_value: str | None = None) -> str | None:
    return default_value or cfg.proxy


def _is_dev(cfg: Config) -> bool:
    return cfg.dev
