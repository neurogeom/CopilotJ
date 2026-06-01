# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Knowledge base utilities: JSONL export/import and FAISS index rebuild."""

import asyncio
import gzip
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from copilotj.core.config import get_home
from copilotj.core.embedding import get_embeddings

__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_INDEX_DIR",
    "INDEX_NAME",
    "ensure_faiss_index",
    "ensure_faiss_index_async",
    "export_jsonl",
    "init_knowledge_base",
    "load_jsonl_docs",
    "save_rebuild_hash",
]

_log = logging.getLogger(__name__)


DEFAULT_INDEX_DIR = get_home() / "assets" / "knowledge_base"
DEFAULT_DATA_DIR = get_home() / "data"
INDEX_NAME = "knowledge_base"

_METADATA_SKIP_KEYS = frozenset({"file_path", "trapped", "modDate", "creationDate"})

_rebuild_lock = threading.Lock()


def _clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if v not in ("", None) and k not in _METADATA_SKIP_KEYS}


def export_jsonl(docs: Sequence[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for doc in docs:
            entry = {"content": doc.page_content, "metadata": _clean_metadata(doc.metadata)}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _log.info("Exported %d docs to %s (%.1f MB)", len(docs), path, path.stat().st_size / 1024 / 1024)


def load_jsonl_docs(path: Path) -> list[Document]:
    docs: list[Document] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            docs.append(Document(page_content=entry["content"], metadata=entry.get("metadata", {})))
    _log.info("Loaded %d docs from %s", len(docs), path)
    return docs


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_rebuild_hash(index_dir: Path) -> str | None:
    hash_path = index_dir / f"{INDEX_NAME}.rebuild.hash"
    if not hash_path.exists():
        return None
    return hash_path.read_text().strip()


def save_rebuild_hash(index_dir: Path, hash_value: str) -> None:
    hash_path = index_dir / f"{INDEX_NAME}.rebuild.hash"
    hash_path.write_text(hash_value)


def _find_jsonl_path(index_dir: Path) -> Path | None:
    pattern = f"{INDEX_NAME}.jsonl.gz"
    candidate = index_dir / pattern
    if candidate.exists():
        return candidate
    return None


def _rebuild(index_dir: Path, jsonl_path: Path) -> None:
    docs = load_jsonl_docs(jsonl_path)
    if not docs:
        raise ValueError(f"No documents found in {jsonl_path}")

    _log.info("Rebuilding FAISS index from %d documents...", len(docs))
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir), index_name=INDEX_NAME)
    save_rebuild_hash(index_dir, _file_hash(jsonl_path))
    _log.info("FAISS index rebuilt and saved to %s", index_dir)


def ensure_faiss_index(index_dir: Path = DEFAULT_INDEX_DIR) -> bool:
    jsonl_path = _find_jsonl_path(index_dir)
    if jsonl_path is None:
        _log.warning("No JSONL found in %s", index_dir)
        return False

    faiss_path = index_dir / f"{INDEX_NAME}.faiss"
    jsonl_hash = _file_hash(jsonl_path)
    cached_hash = _load_rebuild_hash(index_dir)

    if faiss_path.exists() and cached_hash == jsonl_hash:
        return True

    with _rebuild_lock:
        if faiss_path.exists() and _load_rebuild_hash(index_dir) == jsonl_hash:
            return True
        if not faiss_path.exists():
            _log.info("FAISS index missing, rebuilding from %s ...", jsonl_path.name)
        else:
            _log.info("JSONL changed (hash mismatch), rebuilding FAISS index ...")
        _rebuild(index_dir, jsonl_path)
    return True


async def ensure_faiss_index_async(index_dir: Path = DEFAULT_INDEX_DIR) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, ensure_faiss_index, index_dir)


async def init_knowledge_base(index_dir: Path = DEFAULT_INDEX_DIR) -> bool:
    return await ensure_faiss_index_async(index_dir)
