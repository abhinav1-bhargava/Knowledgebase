"""Chunking strategy factory for document ingestion.

Provides a single entry point — `build_splitter(strategy, embed_model=None)` —
that returns a LlamaIndex node parser configured for one of four strategies:

    fixed_512   SentenceSplitter(chunk_size=512,  chunk_overlap=50)
    fixed_768   SentenceSplitter(chunk_size=768,  chunk_overlap=75)
    fixed_1024  SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    semantic    SemanticSplitterNodeParser — variable chunk sizes,
                splits on topic boundaries detected by cosine drops in
                sentence-level embedding similarity.

Keeping the strategy → parameters mapping here (rather than inside
`ingestion/docs_ingest.py`) lets the RAG Lab UI and the ingestion CLI
reference the same canonical list without duplication.

Usage:
    from ingestion.chunking_strategies import build_splitter
    splitter = build_splitter("fixed_768")
    # or:
    splitter = build_splitter("semantic")   # pulls embed model lazily

Semantic chunking needs an embedding model at split time (it embeds
each sentence to find topic boundaries). The factory lazy-loads
`query.embed_factory.get_embed_model()` when no model is passed in, so
the strategy stays interchangeable at the call site.
"""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)

SUPPORTED_CHUNKING_STRATEGIES: tuple[str, ...] = (
    "fixed_512",
    "fixed_768",
    "fixed_1024",
    "semantic",
)

# Canonical per-strategy (chunk_size, chunk_overlap) for the fixed variants.
FIXED_STRATEGY_PARAMS: dict[str, tuple[int, int]] = {
    "fixed_512": (512, 50),
    "fixed_768": (768, 75),
    "fixed_1024": (1024, 100),
}

# Canonical collection name per strategy. The 512 default keeps existing
# installations pointing at whatever collection name they've been using
# (env-configurable in callers); the new strategies land in fresh
# collections so the 24-combo RAG Lab can run side-by-side comparisons.
DEFAULT_COLLECTION_BY_STRATEGY: dict[str, str] = {
    "fixed_512": "pm_onboarding",
    "fixed_768": "kb_768",
    "fixed_1024": "kb_1024",
    "semantic": "kb_semantic",
}


def build_splitter(strategy: str, embed_model: Any | None = None):
    """Return a configured node parser for the named chunking strategy.

    Raises ValueError for unknown strategies.
    """
    if strategy in FIXED_STRATEGY_PARAMS:
        chunk_size, chunk_overlap = FIXED_STRATEGY_PARAMS[strategy]
        logger.info(
            "Building SentenceSplitter (strategy=%s, chunk_size=%d, overlap=%d)",
            strategy, chunk_size, chunk_overlap,
        )
        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
        )

    if strategy == "semantic":
        # Imported lazily — SemanticSplitterNodeParser is heavier and only
        # needed for this branch.
        from llama_index.core.node_parser import SemanticSplitterNodeParser

        if embed_model is None:
            from query.embed_factory import get_embed_model

            embed_model = get_embed_model()
        logger.info(
            "Building SemanticSplitterNodeParser "
            "(buffer_size=1, breakpoint_percentile_threshold=95)"
        )
        return SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )

    raise ValueError(
        f"Unknown chunking strategy {strategy!r}. "
        f"Supported: {SUPPORTED_CHUNKING_STRATEGIES}"
    )
