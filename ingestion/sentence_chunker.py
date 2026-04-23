"""Sentence-level chunking for the sentence-window retrieval strategy.

Two entry points:

    chunk_by_sentence(text) -> list[dict]
        Plain helper that tokenizes into sentences and returns
        [{text, sentence_index, total_sentences}, ...]. Useful for
        scripts or tests that want raw sentence data.

    SentenceIndexNodeParser
        LlamaIndex NodeParser subclass that hooks into IngestionPipeline.
        Emits one TextNode per sentence with sentence_index and
        total_sentences added to the inherited document metadata
        (doc_hash, pod, filename, etc. are preserved end-to-end).

NLTK's punkt / punkt_tab tokenizer is lazy-downloaded on first use and
cached under ~/nltk_data; subsequent runs are offline. We probe for
`punkt_tab` first (newer NLTK default) and fall back to `punkt`.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import nltk
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, TextNode

logger = logging.getLogger(__name__)


def _ensure_punkt() -> None:
    """Idempotent NLTK punkt bootstrap. Safe to call on every tokenize."""
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            return
        except LookupError:
            continue
    # Neither found — try downloading the newer first, fall back to older.
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.download(pkg, quiet=True)
            logger.info("Downloaded NLTK %s", pkg)
            return
        except Exception as exc:
            logger.warning("NLTK download of %s failed: %s", pkg, exc)


def chunk_by_sentence(text: str) -> list[dict]:
    """Split `text` into sentences; return one dict per sentence."""
    _ensure_punkt()
    sentences = nltk.sent_tokenize(text or "")
    total = len(sentences)
    return [
        {"text": s, "sentence_index": i, "total_sentences": total}
        for i, s in enumerate(sentences)
    ]


class SentenceIndexNodeParser(NodeParser):
    """Emit one TextNode per sentence with sentence_index / total_sentences metadata.

    Runs inside LlamaIndex's IngestionPipeline in place of SentenceSplitter.
    Inherited Document metadata (doc_hash, pod, filename, ingestion_date,
    etc.) is merged onto each emitted node so Chroma metadata filters
    (`doc_hash == X AND sentence_index between …`) work at query time.
    """

    @classmethod
    def class_name(cls) -> str:
        return "SentenceIndexNodeParser"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
        _ensure_punkt()
        out: list[BaseNode] = []
        for node in nodes:
            text = node.get_content() or ""
            sentences = nltk.sent_tokenize(text)
            total = len(sentences)
            base_meta = dict(node.metadata or {})
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                meta = {
                    **base_meta,
                    "sentence_index": i,
                    "total_sentences": total,
                }
                out.append(TextNode(text=sentence, metadata=meta))
        return out
