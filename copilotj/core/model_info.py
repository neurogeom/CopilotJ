# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Model capability lookup using the LiteLLM model database.

Downloads ``model_prices_and_context_window.json`` from the LiteLLM repository,
caches it under ``COPILOTJ_HOME/cache/``, and provides capability queries
(e.g. ``supports_vision``) for any model name.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from copilotj.core.config import get_home

__all__ = [
    "CatalogModel",
    "ModelCapabilities",
    "ensure_model_db_async",
    "get_model_capabilities",
    "list_catalog_models",
    "supports_vision",
]

_log = logging.getLogger("copilotj.core.model_info")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LITELLM_DB_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
DB_FILENAME = "model_prices_and_context_window.json"
CACHE_SUBDIR = "cache"  # relative to COPILOTJ_HOME
CACHE_TTL = 7 * 24 * 3600  # 7 days
_DOWNLOAD_TIMEOUT = 30  # seconds

# Known Ollama models that support vision.
_OLLAMA_VISION_MODELS: frozenset[str] = frozenset(
    {
        "llava",
        "llava-llama3",
        "llava-v1.6",
        "bakllava",
        "moondream",
        "moondream2",
        "minicpm-v",
        "minicpm-v2.6",
        "llama3.2-vision",
        "llama3.2-vision:11b",
        "llama3.2-vision:90b",
        "gemma3",
        "gemma3:1b",
        "gemma3:4b",
        "gemma3:12b",
        "gemma3:27b",
        "qwen2.5-vl",
        "qwen2-vl",
        "mistral-small3.1",
        "pixtral",
        "llama4",
        "llama4:scout",
        "llama4:maverick",
    }
)

# Module-level cache of the parsed DB to avoid re-reading from disk on every call.
_db_cache: dict[str, Any] | None = None
_db_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities detected for a specific model."""

    model: str
    supports_vision: bool
    supports_function_calling: bool
    context_window: int | None
    max_output_tokens: int | None
    source: str  # "litellm_db" | "heuristic" | "unknown"


@dataclass(frozen=True)
class CatalogModel:
    """A model entry surfaced for provider-based selection (e.g. dropdowns).

    Unlike :class:`ModelCapabilities` (a per-name lookup), this is a flat
    listing entry carrying the bare model id a provider expects.
    """

    id: str
    label: str
    provider: str
    supports_vision: bool
    supports_function_calling: bool
    context_window: int | None


# ---------------------------------------------------------------------------
# Locally maintained model entries
# ---------------------------------------------------------------------------
#
# Newest models that are not yet in the upstream LiteLLM catalog. These are
# merged into the catalog listing by :func:`list_catalog_models` and silently
# dropped once the catalog catches up (deduped by model id, catalog wins). Keep
# this minimal — only models the catalog is missing — and prune entries as the
# upstream catalog adds them. Capabilities left as conservative defaults
# (vision=False, function_calling=True, context_window=None) rather than
# guessing unverified specs.
_SUPPLEMENTAL_MODELS: dict[str, list[CatalogModel]] = {
    # DeepSeek-V4 (Preview, released 2026-04-24) is absent from the upstream
    # catalog as of 2026-06 — see BerriAI/litellm#26709, #28309 and the unmerged
    # PRs #26380 / #27056. Remove these once `deepseek/deepseek-v4*` ships there.
    "deepseek": [
        CatalogModel(
            id="deepseek-v4",
            label="deepseek-v4",
            provider="deepseek",
            supports_vision=False,
            supports_function_calling=True,
            context_window=None,
        ),
        CatalogModel(
            id="deepseek-v4-pro",
            label="deepseek-v4-pro",
            provider="deepseek",
            supports_vision=False,
            supports_function_calling=True,
            context_window=None,
        ),
        CatalogModel(
            id="deepseek-v4-flash",
            label="deepseek-v4-flash",
            provider="deepseek",
            supports_vision=False,
            supports_function_calling=True,
            context_window=None,
        ),
    ],
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path() -> Path:
    """Return the path to the cached model database file."""
    return get_home() / CACHE_SUBDIR / DB_FILENAME


def _is_cache_stale(path: Path) -> bool:
    """Return ``True`` if the cache file is missing or older than ``CACHE_TTL``."""
    if not path.exists():
        return True
    import time

    return (time.time() - path.stat().st_mtime) > CACHE_TTL


def _load_db() -> dict[str, Any]:
    """Read the cached JSON from disk.  Returns an empty dict if missing."""
    global _db_cache
    if _db_cache is not None:
        return _db_cache

    with _db_lock:
        # Double-checked locking: another thread may have loaded while we waited.
        if _db_cache is not None:
            return _db_cache

        path = _cache_path()
        if not path.exists():
            _log.debug("Model DB cache not found at %s", path)
            return {}

        try:
            text = path.read_text(encoding="utf-8")
            _db_cache = json.loads(text)
            _log.debug("Loaded model DB from %s (%d entries)", path, len(_db_cache))
            return _db_cache  # type: ignore[return-value]
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("Failed to read model DB cache %s: %s", path, exc)
            return {}


def _store_db(data: dict[str, Any]) -> None:
    """Write the DB dict to the cache file and update the module-level cache."""
    global _db_cache

    with _db_lock:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file in the same directory, then rename.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            Path(tmp_path).rename(path)
        except BaseException:
            # Clean up temp file on any error.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        _db_cache = data
        _log.info("Model DB cached at %s (%d entries)", path, len(data))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


async def download_db(*, force: bool = False) -> Path | None:
    """Download the model database from GitHub.

    Skips the download if the cache is fresh (unless *force* is ``True``).
    Thread-safe: ``_store_db`` holds the lock when writing; the download
    itself runs without the lock so it doesn't block other threads.
    Returns the cache path on success, or ``None`` on failure.
    """
    path = _cache_path()

    # Fast path: cache is fresh, skip download.
    if not force and not _is_cache_stale(path):
        _log.debug("Model DB cache is fresh, skipping download")
        return path

    _log.info("Downloading model DB from %s", LITELLM_DB_URL)

    try:
        timeout = aiohttp.ClientTimeout(total=_DOWNLOAD_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(LITELLM_DB_URL) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

        if not isinstance(data, dict) or len(data) < 100:
            _log.warning("Downloaded model DB looks invalid (%d entries)", len(data) if isinstance(data, dict) else 0)
            return path if path.exists() else None

        _store_db(data)  # thread-safe: acquires _db_lock internally
        return path

    except Exception as exc:
        _log.warning("Failed to download model DB: %s", exc)
        return path if path.exists() else None


def _download_db_sync() -> dict[str, Any]:
    """Synchronous fallback download using ``urllib`` (for CLI / first-run)."""
    _log.info("Downloading model DB (sync) from %s", LITELLM_DB_URL)
    try:
        req = urllib.request.Request(LITELLM_DB_URL)
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if isinstance(data, dict) and len(data) >= 100:
            _store_db(data)
            return data

        _log.warning("Downloaded model DB looks invalid")
        return {}
    except Exception as exc:
        _log.warning("Failed to download model DB (sync): %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Model name normalization & lookup
# ---------------------------------------------------------------------------


def _normalize_model_name(model: str) -> tuple[str | None, str]:
    """Strip a known provider prefix from *model*.

    Returns ``(provider, bare_name)``.  Ollama models keep the prefix
    because the LiteLLM DB uses ``"ollama/llava"`` keys.
    """
    slash = model.find("/")
    if slash == -1:
        return None, model

    provider = model[:slash]
    bare = model[slash + 1 :]
    return provider, bare


def _lookup_model(db: dict[str, Any], model: str) -> dict[str, Any] | None:
    """Look up *model* in the DB with several normalization strategies.

    Tries in order:

    1. Exact key match.
    2. Bare name (strip provider prefix).
    3. Common provider-prefixed variants for known providers.
    4. LiteLLM-style AWS Bedrock prefix (``anthropic.``).
    """
    # 1. Exact match
    entry = db.get(model)
    if entry is not None:
        return entry

    provider, bare = _normalize_model_name(model)

    if provider is None:
        # Bare name — try common prefixes
        for prefix in ("openai/", "anthropic/", "gemini/", "cohere/", "mistral/", "deepseek/", "ollama/"):
            entry = db.get(f"{prefix}{model}")
            if entry is not None:
                return entry
        # Also try the name as-is in case it's already a DB key
        return db.get(model)
    else:
        # 2. Bare name match
        entry = db.get(bare)
        if entry is not None:
            return entry

        # 3. Try other provider prefixes
        provider_map = {
            "openai": "openai/",
            "anthropic": "anthropic/",
            "gemini": "gemini/",
            "google": "gemini/",
            "cohere": "cohere_command/",
            "mistral": "mistral/",
            "deepseek": "deepseek/",
            "ollama": "ollama/",
        }
        mapped_prefix = provider_map.get(provider)
        if mapped_prefix:
            entry = db.get(f"{mapped_prefix}{bare}")
            if entry is not None:
                return entry

        # 4. Try LiteLLM-style AWS Bedrock / Vertex AI prefixes
        for prefix in ("anthropic.", "bedrock-", "vertex-"):
            entry = db.get(f"{prefix}{bare}")
            if entry is not None:
                return entry

    return None


def _ollama_vision_heuristic(model: str) -> bool:
    """Check if an Ollama model is known to support vision.

    Strips the ``ollama/`` prefix and compares against a known set,
    also checking base names for tag variants (e.g. ``llava:13b`` → ``llava``).
    """
    _, bare = _normalize_model_name(model)
    # Strip Ollama tag (e.g. "llava:13b" → "llava")
    tag_sep = bare.find(":")
    base = bare[:tag_sep] if tag_sep != -1 else bare
    return base in _OLLAMA_VISION_MODELS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model_capabilities(model: str) -> ModelCapabilities:
    """Return capabilities for *model*.

    Loads the cached model database, looks up the model (with normalisation),
    and returns a ``ModelCapabilities`` instance.  For unknown models,
    returns conservative defaults (``supports_vision=False``).
    """
    if not model:
        return ModelCapabilities(
            model=model,
            supports_vision=False,
            supports_function_calling=False,
            context_window=None,
            max_output_tokens=None,
            source="unknown",
        )

    db = _load_db()
    if not db:
        # No cache available — try a blocking download on first use.
        db = _download_db_sync()

    entry = _lookup_model(db, model) if db else None

    if entry is not None:
        return ModelCapabilities(
            model=model,
            supports_vision=bool(entry.get("supports_vision", False)),
            supports_function_calling=bool(entry.get("supports_function_calling", False)),
            context_window=entry.get("max_input_tokens") or entry.get("max_tokens"),
            max_output_tokens=entry.get("max_output_tokens"),
            source="litellm_db",
        )

    # Not in DB — check Ollama heuristic
    provider, _bare = _normalize_model_name(model)
    if provider == "ollama":
        return ModelCapabilities(
            model=model,
            supports_vision=_ollama_vision_heuristic(model),
            supports_function_calling=False,
            context_window=None,
            max_output_tokens=None,
            source="heuristic",
        )

    # Unknown model — conservative defaults
    return ModelCapabilities(
        model=model,
        supports_vision=False,
        supports_function_calling=False,
        context_window=None,
        max_output_tokens=None,
        source="unknown",
    )


def supports_vision(model: str) -> bool:
    """Return ``True`` if *model* is known to support vision / image input."""
    return get_model_capabilities(model).supports_vision


def _merge_supplements(provider: str, catalog_models: list[CatalogModel]) -> list[CatalogModel]:
    """Merge locally-maintained entries (see :data:`_SUPPLEMENTAL_MODELS`) into *catalog_models*.

    Catalog entries win: a supplemental entry whose id already exists in the
    catalog is dropped, so this becomes a no-op for a provider once the upstream
    catalog catches up. Result is sorted by id.
    """
    extras = _SUPPLEMENTAL_MODELS.get(provider)
    if not extras:
        return catalog_models
    seen = {m.id for m in catalog_models}
    merged = list(catalog_models)
    for model in extras:
        if model.id not in seen:
            merged.append(model)
            seen.add(model.id)
    merged.sort(key=lambda m: m.id)
    return merged


def list_catalog_models(provider: str) -> list[CatalogModel]:
    """Return chat models from the cached LiteLLM catalog for *provider*.

    Filters entries whose ``litellm_provider`` equals *provider* and whose
    ``mode`` is ``chat`` or ``completion``, drops fine-tune placeholders
    (keys starting with ``ft:``), strips a leading ``{provider}/`` prefix
    from the key to form the bare model id (e.g. ``gemini/gemini-2.5-pro``
    -> ``gemini-2.5-pro``), and returns the result sorted by id with
    duplicates removed.  Locally maintained entries for models the upstream
    catalog has not yet added (see :data:`_SUPPLEMENTAL_MODELS`) are merged
    in, with catalog entries taking precedence.  Returns only those
    supplemental entries when the catalog is unavailable.
    """
    db = _load_db()
    if not db:
        db = _download_db_sync()
    if not db:
        return _merge_supplements(provider, [])

    prefix = f"{provider}/"
    seen: set[str] = set()
    models: list[CatalogModel] = []
    for key, entry in db.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("litellm_provider") != provider:
            continue
        if entry.get("mode") not in ("chat", "completion"):
            continue
        if key.startswith("ft:"):
            continue
        model_id = key[len(prefix) :] if key.startswith(prefix) else key
        if model_id in seen:
            continue
        seen.add(model_id)
        models.append(
            CatalogModel(
                id=model_id,
                label=model_id,
                provider=provider,
                supports_vision=bool(entry.get("supports_vision", False)),
                supports_function_calling=bool(entry.get("supports_function_calling", False)),
                context_window=entry.get("max_input_tokens") or entry.get("max_tokens"),
            )
        )

    models.sort(key=lambda m: m.id)
    return _merge_supplements(provider, models)


async def ensure_model_db_async() -> bool:
    """Ensure the model DB is available (download if stale).

    Called at server startup as a background task.  Returns ``True``
    if the DB is available (fresh or stale).
    """
    path = await download_db()
    return path is not None and path.exists()
