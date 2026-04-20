"""Verified answers store (JSONL).

Authoritative Q/A pairs written by pod contributors. Each record carries:
id, question, answer, pod, author, created_at, source_links (optional).
The verified_ingest module reads from here to push entries into ChromaDB
with source_type='verified'.

Usage:
    from storage.verified_answers import add_answer, list_answers, delete_answer
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def add_answer(
    question: str,
    answer: str,
    pod: str,
    author: str,
    source_links: list[str] | None = None,
) -> str:
    """Persist a new verified answer. Returns the answer id."""
    raise NotImplementedError


def list_answers(pod: str | None = None) -> list[dict]:
    """Return all verified answers, optionally filtered by pod."""
    raise NotImplementedError


def delete_answer(answer_id: str) -> None:
    """Delete a verified answer by id."""
    raise NotImplementedError
