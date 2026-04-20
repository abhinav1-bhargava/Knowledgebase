"""Append-only query log (JSONL).

Records every user query along with the assistant's response metadata:
timestamp, pod, question, confidence, is_gap, citation_count, latency_ms.
Used by the contributor UI for analytics (top unanswered questions, low-
confidence trends) and by gap detection.

Usage:
    from storage.query_log import log_query, read_log
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_query(
    question: str,
    pod: str | None,
    confidence: float,
    is_gap: bool,
    citation_count: int,
    latency_ms: int,
) -> None:
    """Append a query record to the log."""
    raise NotImplementedError


def read_log(limit: int | None = None) -> list[dict]:
    """Read query log records, most-recent first, up to `limit`."""
    raise NotImplementedError
