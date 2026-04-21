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
from typing import Any, List

import requests
from llama_index.core.bridge.pydantic import Field
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Minimal embedding client for any OpenAI-compatible /v1/embeddings endpoint.

    Unlike llama_index's OpenAIEmbedding, this does NOT validate the
    model name against a hardcoded enum of OpenAI-branded models — so it
    works with custom deployments that expose non-OpenAI models
    (bge-m3, jina, nomic, etc.) behind the standard OpenAI embeddings
    contract. Used for the airtel_bge provider.
    """

    api_base: str = Field(description="Base URL, e.g. http://host:port/v1")
    api_key: str = Field(description="Bearer token value")
    model_name: str = Field(description="Model name sent in request body")
    timeout_seconds: int = Field(default=30)

    def _post(self, inputs: List[str]) -> List[List[float]]:
        url = f"{self.api_base.rstrip('/')}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.model_name, "input": inputs}
        resp = requests.post(
            url, headers=headers, json=payload, timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in items]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._post([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._post([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._post(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

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
            # llama_index's OpenAIEmbedding validates the model against a
            # hardcoded enum of OpenAI-branded models and rejects "bge-m3"
            # even when api_base is set — so we use our own minimal client.
            model = OpenAICompatibleEmbedding(
                api_base=AIRTEL_EMBEDDING_BASE_URL,
                api_key=AIRTEL_EMBEDDING_API_KEY,
                model_name=EMBEDDING_MODEL or "bge-m3",
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
