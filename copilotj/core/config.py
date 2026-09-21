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

_logger = logging.getLogger(__name__)

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
    "bootstrap_dir_if_empty",
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
    llm_proxy: str | None = None
    llm_provider: str | None = None

    # VLM (falls back to LLM fields)
    vlm_model: str = ""
    vlm_api_key: str = ""
    vlm_base_url: str | None = None
    vlm_proxy: str | None = None
    vlm_provider: str | None = None

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

    # LLM env vars — new names with deprecated fallback
    llm_model_raw = os.getenv("COPILOTJ_LLM_MODEL", "")
    if not llm_model_raw:
        old = os.getenv("COPILOTJ_MODEL")
        if old:
            _logger.warning("COPILOTJ_MODEL is deprecated, use COPILOTJ_LLM_MODEL instead")
            llm_model_raw = old

    llm_api_key_raw = os.getenv("COPILOTJ_LLM_API_KEY", "") or ""
    if not llm_api_key_raw:
        old = os.getenv("COPILOTJ_API_KEY")
        if old:
            _logger.warning("COPILOTJ_API_KEY is deprecated, use COPILOTJ_LLM_API_KEY instead")
            llm_api_key_raw = old

    llm_base_url_raw = os.getenv("COPILOTJ_LLM_BASE_URL")
    if llm_base_url_raw is None:
        old = os.getenv("COPILOTJ_BASE_URL")
        if old:
            _logger.warning("COPILOTJ_BASE_URL is deprecated, use COPILOTJ_LLM_BASE_URL instead")
            llm_base_url_raw = old

    llm_proxy_raw = os.getenv("COPILOTJ_LLM_PROXY")
    if llm_proxy_raw is None:
        old = os.getenv("COPILOTJ_PROXY")
        if old:
            _logger.warning("COPILOTJ_PROXY is deprecated, use COPILOTJ_LLM_PROXY instead")
            llm_proxy_raw = old

    return Config(
        llm_model=llm_model_raw,
        llm_api_key=llm_api_key_raw,
        llm_base_url=llm_base_url_raw,
        vlm_model=os.getenv("COPILOTJ_VLM_MODEL") or llm_model_raw,
        vlm_api_key=os.getenv("COPILOTJ_VLM_API_KEY", llm_api_key_raw) or "",
        vlm_base_url=os.getenv("COPILOTJ_VLM_BASE_URL", None) or llm_base_url_raw,
        llm_proxy=llm_proxy_raw,
        llm_provider=os.getenv("COPILOTJ_LLM_PROVIDER", None),
        vlm_proxy=os.getenv("COPILOTJ_VLM_PROXY", None),
        vlm_provider=os.getenv("COPILOTJ_VLM_PROVIDER", None),
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
# Runtime helpers — environment-based checks used during startup and by the
# bridge / plugin / appose-worker layers.
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


def bootstrap_dir_if_empty(src: Path, dst: Path) -> bool:
    """Copy a seed directory tree into ``dst`` only when ``dst`` is missing or empty.

    Used to seed user-data dirs (``assets`` and ``knowledge_bank``) from the bundled
    source on first run. Note: the ``agents`` dir does NOT use this — it is user-editable
    and uses the richer dpkg-style refresh in ``copilotj.multiagent.agent_loader``.
    No-op (returns False) when:

    - ``src`` does not exist (no seed available),
    - ``src`` and ``dst`` resolve to the same path (dev mode, where the home dir
      IS the source tree, so copying would recurse into itself),
    - ``dst`` already exists and is non-empty (user data present; never clobber).

    Otherwise copies ``src`` onto ``dst`` (``dirs_exist_ok=True``) and returns True.
    """
    if not src.exists():
        return False
    if src.resolve() == dst.resolve():
        return False
    if dst.exists() and any(dst.iterdir()):
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def bootstrap_assets() -> None:
    """Copy assets/ from project source to COPILOTJ_HOME if missing."""
    source_root = Path(__file__).resolve().parent.parent.parent
    bootstrap_dir_if_empty(source_root / "assets", get_home() / "assets")


def _is_dev(cfg: Config) -> bool:
    return cfg.dev
