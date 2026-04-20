"""Knowledge gap store (JSONL).

A "gap" is a question the assistant could not answer with confidence. The
contributor UI surfaces open gaps so pod members can write a verified answer
or point at the right source. Records: id, question, pod, first_seen_at,
last_seen_at, occurrence_count, status (open/resolved), resolved_by.

Usage:
    from storage.gaps import record_gap, list_open_gaps, resolve_gap
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def record_gap(question: str, pod: str | None) -> str:
    """Append (or increment) a gap record. Returns the gap id."""
    raise NotImplementedError


def list_open_gaps(pod: str | None = None) -> list[dict]:
    """Return all open gaps, optionally filtered by pod."""
    raise NotImplementedError


def resolve_gap(gap_id: str, resolved_by: str) -> None:
    """Mark a gap as resolved."""
    raise NotImplementedError
