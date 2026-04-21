"""Embedding model factory — provider-agnostic, singleton-cached.

Branches on `config.EMBEDDING_PROVIDER` to return the right embedding
instance. All three providers expose the LlamaIndex `BaseEmbedding`
interface (`.get_text_embedding(text)`, `.get_text_embedding_batch([...])`),
so call sites are provider-agnostic.

Providers:
    openai       — OpenAI SaaS (requires OPENAI_API_KEY). Dim depends on
                   EMBEDDING_MODEL (text-embedding-3-small = 1536).
    airtel_bge   — Airtel-hosted BGE-M3 via OpenAI-compatible endpoint
                   (requires VPN). Dim 1024.
    local_hf     — HuggingFace sentence-transformers on-device. Default
                   model BAAI/bge-small-en-v1.5 → 384-dim, ~130MB, fully
                   offline after first download.

`verify_collection_dim(collection)` is a cheap idempotent guard: on the
first call per-process-per-collection it compares the model's output dim
to whatever's already stored in Chroma. A mismatch raises with a clear
remediation message so we don't silently mix embeddings from different
providers.

Usage:
    from query.embed_factory import get_embed_model, verify_collection_dim
    em = get_embed_model()
    vec = em.get_text_embedding("hello")
    verify_collection_dim(chroma_collection)   # once per collection
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_lock = Lock()
_cached: Any | None = None
_model_dim: int | None = None
_verified_collection_names: set[str] = set()


def get_embed_model():
    """Return the configured embedding model (singleton-cached)."""
    global _cached
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is not None:
            return _cached
        from config import (
            AIRTEL_EMBEDDING_API_KEY,
            AIRTEL_EMBEDDING_BASE_URL,
            EMBEDDING_MODEL,
            EMBEDDING_PROVIDER,
            LOCAL_EMBEDDING_MODEL,
            OPENAI_API_KEY,
        )
        provider = (EMBEDDING_PROVIDER or "local_hf").lower()
        if provider == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding

            model = OpenAIEmbedding(
                model=EMBEDDING_MODEL or "text-embedding-3-small",
                api_key=OPENAI_API_KEY,
            )
            logger.info("Embedding provider: openai (%s)", EMBEDDING_MODEL)
        elif provider == "airtel_bge":
            from llama_index.embeddings.openai import OpenAIEmbedding

            model = OpenAIEmbedding(
                model=EMBEDDING_MODEL or "bge-m3",
                api_key=AIRTEL_EMBEDDING_API_KEY,
                api_base=AIRTEL_EMBEDDING_BASE_URL,
            )
            logger.info(
                "Embedding provider: airtel_bge (%s at %s)",
                EMBEDDING_MODEL, AIRTEL_EMBEDDING_BASE_URL,
            )
        elif provider == "local_hf":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            name = LOCAL_EMBEDDING_MODEL or "BAAI/bge-small-en-v1.5"
            logger.info(
                "Loading local HuggingFace embedding model: %s "
                "(first run downloads ~130MB to ~/.cache/huggingface/)",
                name,
            )
            model = HuggingFaceEmbedding(model_name=name, device="cpu")
            logger.info("Loaded local HuggingFace embedding model: %s", name)
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER: {provider!r}. "
                f"Must be one of: openai, airtel_bge, local_hf."
            )
        _cached = model
        return _cached


def get_model_dim() -> int:
    """Return the dimension the current embedding model produces (cached)."""
    global _model_dim
    if _model_dim is None:
        _model_dim = len(get_embed_model().get_text_embedding("dim probe"))
    return _model_dim


def verify_collection_dim(collection) -> None:
    """Raise if the Chroma collection's stored embedding dim doesn't match the model's.

    Idempotent per (process, collection name). No-op on empty collections.
    """
    name = getattr(collection, "name", None) or "<unknown>"
    if name in _verified_collection_names:
        return
    try:
        count = collection.count()
    except Exception as exc:
        logger.warning("Could not query collection count for %s: %s", name, exc)
        return
    if count == 0:
        _verified_collection_names.add(name)
        return

    try:
        sample = collection.get(include=["embeddings"], limit=1)
    except Exception as exc:
        logger.warning("Could not peek collection %s for dim check: %s", name, exc)
        return
    embeddings = sample.get("embeddings") or []
    if not embeddings or embeddings[0] is None:
        _verified_collection_names.add(name)
        return

    stored_dim = len(embeddings[0])
    model_dim = get_model_dim()
    if stored_dim != model_dim:
        from config import EMBEDDING_PROVIDER

        raise RuntimeError(
            f"Embedding dimension mismatch: collection {name!r} has "
            f"{stored_dim}-dim vectors, current model "
            f"(EMBEDDING_PROVIDER={EMBEDDING_PROVIDER}) produces {model_dim}-dim. "
            f"Either (a) delete ./chroma_db/ and re-ingest with the new provider, "
            f"or (b) set EMBEDDING_PROVIDER back to match the existing collection."
        )
    _verified_collection_names.add(name)


def reset_embed_model() -> None:
    """Drop cached model + dim + verification set. Mainly for tests."""
    global _cached, _model_dim
    with _lock:
        _cached = None
        _model_dim = None
        _verified_collection_names.clear()
