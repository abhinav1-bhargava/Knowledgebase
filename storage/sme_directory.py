"""SME directory (JSONL, append-only, latest-entry-per-sme-id wins).

Record shape:
    sme_id (uuid), name, email (optional), topics (list[str]),
    pods (list[str]), added_by, added_at, last_edited_at,
    last_edited_by, status (active|inactive).

Usage:
    from storage.sme_directory import list_smes, add_sme, deactivate_sme
    new_id = add_sme("Ravi K.", "ravi@acme", ["payments"], ["recharge"], "admin")
    for sme in list_smes(pod="recharge", topic="payment"):
        ...
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from storage._jsonl import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

SME_DIRECTORY_PATH = Path("sme_directory.jsonl")


def _latest_per_sme_id() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for r in read_jsonl(SME_DIRECTORY_PATH):
        sid = r.get("sme_id")
        if sid:
            latest[sid] = r
    return latest


def list_smes(
    pod: str | None = None,
    topic: str | None = None,
    active_only: bool = True,
) -> list[dict]:
    """Return the latest view of each SME, with optional filters."""
    smes = list(_latest_per_sme_id().values())
    if active_only:
        smes = [s for s in smes if s.get("status", "active") == "active"]
    if pod:
        smes = [s for s in smes if pod in (s.get("pods") or [])]
    if topic:
        needle = topic.lower()
        smes = [
            s for s in smes
            if any(needle in (t or "").lower() for t in (s.get("topics") or []))
        ]
    return smes


def add_sme(
    name: str,
    email: str | None,
    topics: list[str],
    pods: list[str],
    added_by: str,
) -> str:
    """Append a new SME record. Returns the assigned sme_id."""
    sme_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    append_jsonl(SME_DIRECTORY_PATH, {
        "sme_id": sme_id,
        "name": name,
        "email": email,
        "topics": list(topics),
        "pods": list(pods),
        "added_by": added_by,
        "added_at": now,
        "last_edited_at": now,
        "status": "active",
    })
    return sme_id


def update_sme(sme_id: str, patches: dict, edited_by: str) -> None:
    """Append an updated SME record. No-op if the sme_id is unknown."""
    current = _latest_per_sme_id().get(sme_id)
    if current is None:
        logger.warning("update_sme: unknown sme_id %s", sme_id)
        return
    base = dict(current)
    base.update(patches)
    base["sme_id"] = sme_id
    base["last_edited_at"] = datetime.now(timezone.utc).isoformat()
    base["last_edited_by"] = edited_by
    append_jsonl(SME_DIRECTORY_PATH, base)


def deactivate_sme(sme_id: str, by_user: str) -> None:
    """Soft-delete via a status=inactive update."""
    update_sme(sme_id, {"status": "inactive"}, by_user)
