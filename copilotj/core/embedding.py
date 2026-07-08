# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import logging

from langchain_core.embeddings import Embeddings

from copilotj.core.config import Config

logger = logging.getLogger(__name__)

RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings: Embeddings | None = None

__all__ = [
    "RAG_EMBEDDING_MODEL",
    "configure_download_proxy",
    "get_embeddings",
    "new_local_embeddings",
]


def new_local_embeddings() -> Embeddings:
    """Create the local embedding model used by the bundled FAISS index.

    The HuggingFace download proxy is configured once at process startup via
    :func:`configure_download_proxy`, so this function needs no ``Config``.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    device = _detect_embedding_device()
    logger.info("Loading local embedding model: %s on device: %s", RAG_EMBEDDING_MODEL, device)

    return HuggingFaceEmbeddings(
        model_name=RAG_EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def configure_download_proxy(cfg: Config) -> None:
    """Route HuggingFace downloads through ``CIJ_PROXY`` (explicit, no env vars).

    ``huggingface_hub`` (httpx-based) downloads weights via a client produced by its
    client factory. Installing a factory that returns a proxied ``httpx.Client`` makes
    sentence-transformers/huggingface_hub fetch the embedding weights through the
    proxy without touching ``os.environ``. Call once at process startup (server /
    appose_worker / rag_builder) before the first HF download; a no-op when no
    download proxy is configured.
    """
    proxy = cfg.cij_proxy
    if not proxy:
        return
    import httpx
    from huggingface_hub import set_client_factory

    set_client_factory(lambda: httpx.Client(proxy=proxy))


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
