#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and inspect the ImageJ FAISS knowledge base.

The index is rebuilt as one coherent artifact. Incremental updates are intentionally
not supported because FAISS deletion/replacement must keep the docstore, index, and
manifest in lockstep.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from copilotj.core import load_env  # noqa: E402
from copilotj.core.embedding import get_embeddings  # noqa: E402

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "assets" / "knowledge_base"
INDEX_NAME = "knowledge_base"
MANIFEST_FILE = "rag_manifest.json"
SUPPORTED_EXTENSIONS = (".md", ".markdown", ".pdf", ".txt")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HASH_READ_SIZE = 65536


def get_file_hash(filepath: Path) -> str:
    hasher = hashlib.md5()
    with filepath.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_READ_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def to_relative_path(filepath: Path) -> str:
    try:
        return str(filepath.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(filepath.resolve())


def get_supported_files(data_dir: Path) -> list[Path]:
    files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not _is_hidden(path)
    ]
    return sorted(files)


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def load_documents_from_file(filepath: Path) -> list[Document]:
    loader = _new_loader(filepath)
    documents = loader.load()
    for document in documents:
        document.metadata["source"] = to_relative_path(filepath)
        document.metadata["filename"] = filepath.name
        document.metadata["extension"] = filepath.suffix.lower()
    return documents


def _new_loader(filepath: Path):
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        return PyMuPDFLoader(str(filepath))
    if suffix == ".txt":
        return TextLoader(str(filepath), encoding="utf-8")
    if suffix in {".md", ".markdown"}:
        return UnstructuredMarkdownLoader(str(filepath))
    raise ValueError(f"Unsupported file type: {filepath}")


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_rag(data_dir: Path = DEFAULT_DATA_DIR, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
    files = get_supported_files(data_dir)
    if not files:
        raise ValueError(f"No supported files found in {data_dir}")

    documents = load_documents(files)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError(f"No text chunks created from {data_dir}")

    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, get_embeddings())
    vector_store.save_local(str(index_dir), index_name=INDEX_NAME)
    save_manifest(index_dir, build_manifest(files, len(chunks)))


def load_documents(files: list[Path]) -> list[Document]:
    documents = []
    for filepath in files:
        documents.extend(load_documents_from_file(filepath))
    return documents


def build_manifest(files: list[Path], total_chunks: int) -> dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "version": "1.0",
        "created": now,
        "last_updated": now,
        "files": {to_relative_path(filepath): get_file_metadata(filepath) for filepath in files},
        "stats": {"total_files": len(files), "total_chunks": total_chunks},
    }


def get_file_metadata(filepath: Path) -> dict[str, Any]:
    stat = filepath.stat()
    return {
        "path": to_relative_path(filepath),
        "hash": get_file_hash(filepath),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "mtime_str": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def save_manifest(index_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = index_dir / MANIFEST_FILE
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def load_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / MANIFEST_FILE
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def show_status(data_dir: Path = DEFAULT_DATA_DIR, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
    paths = index_paths(index_dir)
    print("RAG System Status")
    print("=================")
    print(f"Index Directory: {index_dir}")
    print(f"  FAISS index: {_format_exists(paths['faiss'])}")
    print(f"  Document store: {_format_exists(paths['pkl'])}")
    print(f"  Manifest: {_format_exists(paths['manifest'])}")

    if paths["manifest"].exists():
        manifest = load_manifest(index_dir)
        print_manifest(manifest)
        print_change_summary(data_dir, manifest)


def index_paths(index_dir: Path) -> dict[str, Path]:
    return {
        "faiss": index_dir / f"{INDEX_NAME}.faiss",
        "pkl": index_dir / f"{INDEX_NAME}.pkl",
        "manifest": index_dir / MANIFEST_FILE,
    }


def _format_exists(path: Path) -> str:
    return "exists" if path.exists() else "missing"


def print_manifest(manifest: dict[str, Any]) -> None:
    stats = manifest.get("stats", {})
    print("Manifest Info:")
    print(f"  Version: {manifest.get('version', 'unknown')}")
    print(f"  Created: {manifest.get('created', 'unknown')}")
    print(f"  Last Updated: {manifest.get('last_updated', 'unknown')}")
    print(f"  Total Files: {stats.get('total_files', 'unknown')}")
    print(f"  Total Chunks: {stats.get('total_chunks', 'unknown')}")


def print_change_summary(data_dir: Path, manifest: dict[str, Any]) -> None:
    changes = detect_changes(data_dir, manifest)
    print(f"Data Directory: {data_dir}")
    print(f"  Supported files found: {len(get_supported_files(data_dir))}")
    print(f"  New: {len(changes['new'])}")
    print(f"  Modified: {len(changes['modified'])}")
    print(f"  Deleted: {len(changes['deleted'])}")


def detect_changes(data_dir: Path, manifest: dict[str, Any]) -> dict[str, list[str]]:
    current_files = get_supported_files(data_dir)
    current_paths = {to_relative_path(path): path for path in current_files}
    manifest_files = manifest.get("files", {})

    return {
        "new": sorted(set(current_paths) - set(manifest_files)),
        "modified": sorted(_modified_paths(current_paths, manifest_files)),
        "deleted": sorted(set(manifest_files) - set(current_paths)),
    }


def _modified_paths(current_paths: dict[str, Path], manifest_files: dict[str, Any]) -> list[str]:
    modified = []
    for relative_path, filepath in current_paths.items():
        if relative_path not in manifest_files:
            continue
        if get_file_hash(filepath) != manifest_files[relative_path].get("hash"):
            modified.append(relative_path)
    return modified


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Rebuild the RAG index from the data directory")
    parser.add_argument("--status", action="store_true", help="Show index and data directory status")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args(argv)

    if not args.build and not args.status:
        args.status = True
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    apply_embedding_overrides(args.embedding_model, args.device)

    if args.build:
        build_rag(data_dir=args.data_dir, index_dir=args.index_dir)
        return 0

    show_status(data_dir=args.data_dir, index_dir=args.index_dir)
    return 0


def apply_embedding_overrides(model: str | None, device: str | None) -> None:
    if model:
        os.environ["COPILOTJ_EMBEDDING_MODEL"] = model
    if device:
        os.environ["COPILOTJ_EMBEDDING_DEVICE"] = device


if __name__ == "__main__":
    raise SystemExit(main())
