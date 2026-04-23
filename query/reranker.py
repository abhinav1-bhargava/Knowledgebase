"""Cross-encoder re-ranker for second-stage retrieval refinement.

The first-stage vector retriever (Airtel BGE-M3) casts a wide net (k=50).
A cross-encoder scores each (question, chunk) pair directly — more
expensive per pair but much better at picking the truly relevant ones —
and we keep the top-k by that score.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — 22M params, ~23MB on
disk, good balance of quality vs. latency. Downloaded on first use and
cached under ~/.cache/huggingface.

Usage:
    from query.reranker import rerank
    top10 = rerank("what's the HDO target?", candidates_from_vector, top_k=10)
"""

from __future__ import annotations

import logging
from threading import Lock

logger = logging.getLogger(__name__)

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_lock = Lock()
_reranker = None


def get_reranker():
    """Return the cached CrossEncoder instance; load on first call.

    First load downloads ~23MB from HuggingFace and typically takes
    5–10 seconds. Subsequent calls are instant.
    """
    global _reranker
    if _reranker is not None:
        return _reranker
    with _lock:
        if _reranker is not None:
            return _reranker
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading cross-encoder %s (first run downloads ~23MB)",
            _RERANKER_MODEL,
        )
        _reranker = CrossEncoder(_RERANKER_MODEL)
        logger.info("Cross-encoder loaded.")
        return _reranker


def rerank(
    question: str,
    chunks: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """Re-rank a list of candidate chunks via cross-encoder scoring.

    Each input chunk is expected to be a dict with at least `text` and
    `metadata` (and `score`, the original retrieval similarity). A copy
    of each chunk is returned with `rerank_score` attached; the returned
    list is sorted by rerank_score desc and truncated to `top_k`. The
    original `score` is preserved so downstream confidence math (mean
    similarity) stays comparable across retrieval strategies.
    """
    if not chunks:
        return []
    model = get_reranker()
    pairs = [(question, c.get("text") or "") for c in chunks]
    scores = model.predict(pairs)

    ranked: list[dict] = []
    for chunk, score in zip(chunks, scores):
        c = dict(chunk)
        c["rerank_score"] = float(score)
        ranked.append(c)
    ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


def reset_reranker() -> None:
    """Drop the cached CrossEncoder. Mainly for tests."""
    global _reranker
    with _lock:
        _reranker = None
