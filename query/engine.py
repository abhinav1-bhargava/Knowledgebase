"""Retrieval + generation engine.

Wraps a llama-index query engine over the ChromaDB vector store, applies
metadata filters (e.g. by pod), retrieves the top-K most relevant chunks,
generates an answer via the configured LLM with mandatory citations, and
returns a structured result including a confidence score.

Usage:
    from query.engine import answer_query
    result = answer_query("What metrics does the Recharge pod own?", pod="recharge")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured response returned by the query engine."""

    answer: str
    citations: list[dict]
    confidence: float
    is_gap: bool


def get_query_engine(pod: str | None = None):
    """Construct and return a llama-index query engine, optionally pod-filtered."""
    raise NotImplementedError


def answer_query(question: str, pod: str | None = None) -> QueryResult:
    """Answer a user question. Logs the query and flags low-confidence as a gap."""
    raise NotImplementedError
