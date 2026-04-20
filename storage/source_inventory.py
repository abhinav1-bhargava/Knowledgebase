"""Source inventory + health aggregation.

Combines four signals into a single source-centric view used by the
contributor UI's Sources tab:

    1. Chroma metadata — distinct (source_type, identifier) combos with
       chunk count and latest ingestion_date per source.
    2. query_log.jsonl — count of queries in the window whose citations
       include this identifier (hit count).
    3. feedback.jsonl + query_log.jsonl — count of negative feedback
       (wrong / outdated / wrong_source) on answers that cited this
       identifier. Feedback is cross-referenced to query_log via the
       (question, answer) pair because query_log doesn't yet carry a
       query_id (pre-Phase-D decision).
    4. source_status.jsonl — latest status (active/flagged/deprecated)
       written by this module's flag/deprecate/unflag calls.

TODO (retrieval-side honouring of source status):
    query/engine.py::_retrieve should post-filter or downrank chunks
    whose (relative_path/issue_key, source_type) resolves to `flagged`
    or `deprecated` via `get_source_status`. That change is out of scope
    for Phase F — the contributor UI records status changes, but the
    consumer-side ranking doesn't yet honour them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage._jsonl import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

SOURCE_STATUS_PATH = Path("source_status.jsonl")
_NEGATIVE_FEEDBACK = {"wrong", "outdated", "wrong_source"}


def _source_key(identifier: str, source_type: str) -> str:
    return f"{source_type}::{identifier}"


# --- Chroma aggregation ----------------------------------------------------


def _read_chroma_sources(pod: str | None) -> list[dict]:
    """Aggregate distinct sources from Chroma with chunk count + latest ingestion date."""
    try:
        import chromadb

        from config import CHROMA_PATH
    except Exception:
        return []
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_or_create_collection("pm_onboarding")
        if col.count() == 0:
            return []
        result = col.get(include=["metadatas"])
    except Exception as exc:
        logger.warning("Failed to read Chroma metadata: %s", exc)
        return []

    agg: dict[tuple[str, str], dict] = {}
    for md in result.get("metadatas") or []:
        if not md:
            continue
        if pod and md.get("pod") != pod:
            continue
        source_type = md.get("source_type", "") or ""
        if source_type == "jira":
            identifier = md.get("issue_key") or md.get("filename")
        elif source_type == "verified":
            identifier = md.get("verified_id") or md.get("filename")
        else:
            identifier = md.get("filename") or md.get("relative_path")
        if not identifier:
            continue
        key = (source_type, identifier)
        entry = agg.setdefault(key, {
            "identifier": identifier,
            "source_type": source_type,
            "pod": md.get("pod"),
            "chunks": 0,
            "ingested": md.get("ingestion_date"),
        })
        entry["chunks"] += 1
        ing = md.get("ingestion_date")
        if ing and (not entry["ingested"] or ing > entry["ingested"]):
            entry["ingested"] = ing
    return list(agg.values())


# --- Hit / thumbs-down aggregation ------------------------------------------


def _read_hits(window_days: int) -> dict[str, int]:
    """Count query_log citations per identifier in the window."""
    from storage.query_log import QUERY_LOG_PATH

    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    counts: dict[str, int] = {}
    for r in read_jsonl(QUERY_LOG_PATH):
        if r.get("timestamp", "") < cutoff:
            continue
        for c in r.get("citations", []) or []:
            ident = c.get("identifier") if isinstance(c, dict) else None
            if ident:
                counts[ident] = counts.get(ident, 0) + 1
    return counts


def _read_thumbs_down(window_days: int) -> dict[str, int]:
    """Count negative feedback per cited identifier in the window.

    Cross-refs feedback.jsonl → query_log.jsonl via (question, answer)
    because query_log has no query_id. Ties broken arbitrarily if the
    same (question, answer) pair appears twice in query_log.
    """
    from pathlib import Path as _P

    from storage.query_log import QUERY_LOG_PATH

    feedback_records = read_jsonl(_P("feedback.jsonl"))
    query_records = read_jsonl(QUERY_LOG_PATH)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    q_by_qa: dict[tuple[str, str], dict] = {}
    for q in query_records:
        q_by_qa[(q.get("question", ""), q.get("answer", ""))] = q

    counts: dict[str, int] = {}
    for fb in feedback_records:
        if fb.get("timestamp", "") < cutoff:
            continue
        if fb.get("feedback_type") not in _NEGATIVE_FEEDBACK:
            continue
        q = q_by_qa.get((fb.get("question", ""), fb.get("answer", "")))
        if not q:
            continue
        for c in q.get("citations", []) or []:
            ident = c.get("identifier") if isinstance(c, dict) else None
            if ident:
                counts[ident] = counts.get(ident, 0) + 1
    return counts


# --- Public API -------------------------------------------------------------


def _read_all_statuses() -> dict[str, str]:
    latest: dict[str, str] = {}
    for r in read_jsonl(SOURCE_STATUS_PATH):
        key = _source_key(r.get("identifier", ""), r.get("source_type", ""))
        latest[key] = r.get("status", "active")
    return latest


def list_sources(pod: str | None = None, window_days: int = 30) -> list[dict]:
    """Return the merged per-source view."""
    sources = _read_chroma_sources(pod=pod)
    hits = _read_hits(window_days)
    thumbs = _read_thumbs_down(window_days)
    statuses = _read_all_statuses()
    for s in sources:
        ident = s["identifier"]
        s["hits"] = hits.get(ident, 0)
        s["thumbs_down"] = thumbs.get(ident, 0)
        s["status"] = statuses.get(_source_key(ident, s["source_type"]), "active")
    return sources


def flag_source(identifier: str, source_type: str, user: str, reason: str) -> None:
    append_jsonl(SOURCE_STATUS_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "identifier": identifier,
        "source_type": source_type,
        "status": "flagged",
        "flagged_by": user,
        "reason": reason,
    })


def deprecate_source(identifier: str, source_type: str, user: str) -> None:
    append_jsonl(SOURCE_STATUS_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "identifier": identifier,
        "source_type": source_type,
        "status": "deprecated",
        "deprecated_by": user,
    })


def unflag_source(identifier: str, source_type: str, user: str) -> None:
    append_jsonl(SOURCE_STATUS_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "identifier": identifier,
        "source_type": source_type,
        "status": "active",
        "unflagged_by": user,
    })


def get_source_status(identifier: str, source_type: str) -> str:
    return _read_all_statuses().get(_source_key(identifier, source_type), "active")
