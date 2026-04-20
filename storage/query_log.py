"""Append-only query log (JSONL).

Records every user query along with the assistant's response metadata so
the contributor UI can surface top unanswered questions, low-confidence
trends, and recent activity.

Record shape:
    timestamp (UTC ISO-8601), question, pod, answer, confidence,
    low_confidence, citations (list of {type, identifier, ...}).

Usage:
    from storage.query_log import log_query
    log_query(question="...", pod="recharge", answer="...",
              confidence=0.82, low_confidence=False, citations=[...])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

QUERY_LOG_PATH = Path("query_log.jsonl")


def log_query(
    question: str,
    pod: str | None,
    answer: str,
    confidence: float,
    low_confidence: bool,
    citations: list[dict] | None = None,
) -> None:
    """Append a query record to query_log.jsonl.

    Never raises: failures to write the log are logged and swallowed so an
    analytics-write failure cannot break the user-facing answer path.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "pod": pod,
        "answer": answer,
        "confidence": confidence,
        "low_confidence": low_confidence,
        "citations": citations or [],
    }
    try:
        with QUERY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to write query log: %s", exc)


def read_log(limit: int | None = None) -> list[dict]:
    """Read query log records, most-recent first, up to `limit`."""
    raise NotImplementedError
