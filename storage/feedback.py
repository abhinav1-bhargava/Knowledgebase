"""User feedback store (JSONL).

Captures a PM's reaction to an individual assistant answer. Each record
binds to a specific turn via the `query_id` UUID assigned when the turn
was created, so contributors can later trace which source or prompt
produced a "wrong" / "outdated" judgement.

Record shape:
    timestamp (UTC ISO-8601), query_id, question, answer, feedback_type,
    user, pod.

Valid feedback_type values: "helpful", "wrong", "outdated", "wrong_source".

Usage:
    from storage.feedback import log_feedback
    log_feedback(
        query_id="...", question="...", answer="...",
        feedback_type="helpful", user="Asha M.", pod="recharge",
    )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

FEEDBACK_LOG_PATH = Path("feedback.jsonl")
VALID_FEEDBACK_TYPES = {"helpful", "wrong", "outdated", "wrong_source"}


def log_feedback(
    query_id: str,
    question: str,
    answer: str,
    feedback_type: str,
    user: str,
    pod: str | None,
) -> None:
    """Append a feedback record to feedback.jsonl. Never raises.

    Unknown `feedback_type` values are ignored with a warning so a typo in
    the caller cannot silently corrupt analytics.
    """
    if feedback_type not in VALID_FEEDBACK_TYPES:
        logger.warning("Ignoring unknown feedback_type %r", feedback_type)
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": query_id,
        "question": question,
        "answer": answer,
        "feedback_type": feedback_type,
        "user": user,
        "pod": pod,
    }
    try:
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to write feedback log: %s", exc)
