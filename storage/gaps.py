"""Knowledge gap store (JSONL).

A "gap" is a question the assistant could not answer with confidence.
`log_gap` records the gap at query time (append-only); the contributor UI
will later consume these records to let pod members write verified
answers or route the question to an SME.

Record shape:
    timestamp (UTC ISO-8601), question, pod, confidence,
    retrieval_scores (list of floats), status ("open").

Usage:
    from storage.gaps import log_gap
    log_gap(question="...", pod="recharge", confidence=0.42,
            retrieval_scores=[0.55, 0.40, 0.31])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

GAPS_LOG_PATH = Path("gaps.jsonl")


def log_gap(
    question: str,
    pod: str | None,
    confidence: float,
    retrieval_scores: list[float],
) -> None:
    """Append an open gap record to gaps.jsonl. Never raises."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "pod": pod,
        "confidence": confidence,
        "retrieval_scores": list(retrieval_scores),
        "status": "open",
    }
    try:
        with GAPS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to write gaps log: %s", exc)


def record_gap(question: str, pod: str | None) -> str:
    """Append (or increment) a gap record. Returns the gap id."""
    raise NotImplementedError


def list_open_gaps(pod: str | None = None) -> list[dict]:
    """Return all open gaps, optionally filtered by pod."""
    raise NotImplementedError


def resolve_gap(gap_id: str, resolved_by: str) -> None:
    """Mark a gap as resolved."""
    raise NotImplementedError
