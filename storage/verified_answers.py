"""Verified answers store (JSONL, append-only, latest-entry-per-answer-id wins).

Authoritative Q/A pairs written by pod contributors via the contributor
UI. Every mutation (save, update, retire, mark-reviewed) appends a fresh
record; `list_answers` / `resolve_by_answer_id` collapse the history by
keeping the most recent entry per `answer_id`.

Record shape (all fields optional except `answer_id`):
    answer_id, question, pod, answer, source_references (list[str]),
    sme, status (Draft|Published|Retired), review_due_date (ISO date),
    author, authored_date (UTC ISO-8601), last_edited_date,
    last_edited_by.

Usage:
    from storage.verified_answers import save_answer, list_answers, update_answer
    aid = save_answer({"question": "...", "answer": "...", ...})
    update_answer(aid, {"status": "Published"}, edited_by="Ravi K.")
    for row in list_answers(pod="recharge", status="Published"):
        ...
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from storage._jsonl import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

VERIFIED_ANSWERS_PATH = Path("verified_answers.jsonl")


def _latest_per_answer_id() -> dict[str, dict]:
    """Collapse the append-only log into latest-per-answer_id view."""
    latest: dict[str, dict] = {}
    for r in read_jsonl(VERIFIED_ANSWERS_PATH):
        aid = r.get("answer_id")
        if aid:
            latest[aid] = r
    return latest


def save_answer(fields: dict) -> str:
    """Persist a new verified answer. Returns the assigned answer_id."""
    answer_id = fields.get("answer_id") or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "answer_id": answer_id,
        "question": fields.get("question", ""),
        "pod": fields.get("pod"),
        "answer": fields.get("answer", ""),
        "source_references": fields.get("source_references", []),
        "sme": fields.get("sme"),
        "status": fields.get("status", "Draft"),
        "review_due_date": fields.get("review_due_date"),
        "author": fields.get("author"),
        "authored_date": fields.get("authored_date") or now,
        "last_edited_date": now,
    }
    append_jsonl(VERIFIED_ANSWERS_PATH, record)
    return answer_id


def list_answers(
    pod: str | None = None,
    status: str | None = None,
    overdue_only: bool = False,
) -> list[dict]:
    """Return the latest view of each answer, with optional filters."""
    answers = list(_latest_per_answer_id().values())
    if pod:
        answers = [a for a in answers if a.get("pod") == pod]
    if status:
        answers = [a for a in answers if a.get("status") == status]
    if overdue_only:
        today_iso = date.today().isoformat()
        answers = [
            a for a in answers
            if (a.get("review_due_date") or "9999-12-31") < today_iso
        ]
    return answers


def update_answer(answer_id: str, patches: dict, edited_by: str | None = None) -> None:
    """Append an update record. Later entries take precedence on read."""
    current = _latest_per_answer_id().get(answer_id)
    if current is None:
        logger.warning("update_answer: no existing answer for %s — creating placeholder", answer_id)
        base = {"answer_id": answer_id}
    else:
        base = dict(current)
    base.update(patches)
    base["answer_id"] = answer_id
    base["last_edited_date"] = datetime.now(timezone.utc).isoformat()
    if edited_by:
        base["last_edited_by"] = edited_by
    # Preserve authored_date across updates.
    if current and current.get("authored_date"):
        base["authored_date"] = current["authored_date"]
    append_jsonl(VERIFIED_ANSWERS_PATH, base)


def resolve_by_answer_id(answer_id: str) -> dict:
    """Return the latest record for `answer_id`, or {} if not found."""
    return _latest_per_answer_id().get(answer_id, {})
