# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import logging

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings: Embeddings | None = None

__all__ = [
    "RAG_EMBEDDING_MODEL",
    "get_embeddings",
    "new_local_embeddings",
]


def new_local_embeddings() -> Embeddings:
    """Create the local embedding model used by the bundled FAISS index."""
    from langchain_huggingface import HuggingFaceEmbeddings

    device = _detect_embedding_device()
    logger.info("Loading local embedding model: %s on device: %s", RAG_EMBEDDING_MODEL, device)

    return HuggingFaceEmbeddings(
        model_name=RAG_EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings() -> Embeddings:
    """Get the local embeddings used for ImageJ RAG."""
    global _embeddings
    if _embeddings is None:
        _embeddings = new_local_embeddings()
    return _embeddings


def _detect_embedding_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
