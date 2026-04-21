"""Filesystem helpers + audit log for contributor-side uploads.

Kept separate from app/contributor.py so the save + log paths can be
unit-tested without importing the Streamlit UI (whose module-level
script runs `main()` at import time and would pollute Streamlit's
internal state when imported outside a proper runtime).

Layout on disk:
    uploads/<pod>/<unix_ts>_<sanitized_original_filename>   (file bytes)
    upload_log.jsonl                                        (audit trail at repo root)

upload_log.jsonl is append-only. Each *upload event* gets a UUID
`upload_id` and is written at least twice: once with
`ingest_status="pending"` right after the bytes hit disk, and once more
with the final status (`success`, `failed`, or `skipped_dedup`) after
the ingest pipeline returns. Latest entry per `upload_id` wins on read
(same convention as storage/verified_answers.py), so consumers of the
log can treat it as a cheap mutable table.

Record shape:
    upload_id, filename (original), saved_path (relative to repo root),
    pod, uploader, label, uploaded_at (UTC ISO-8601), file_size,
    ingest_status (pending|success|failed|skipped_dedup),
    chunks_created (int; 0 when pending/failed), error_msg (str|null).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from storage._jsonl import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

UPLOADS_ROOT = Path("uploads")
UPLOAD_LOG_PATH = Path("upload_log.jsonl")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB — triggers a UI warning, not a hard block

VALID_INGEST_STATUSES = {"pending", "success", "failed", "skipped_dedup"}


class UploadedFileLike(Protocol):
    """The streamlit UploadedFile shape we actually depend on."""

    name: str
    size: int

    def getvalue(self) -> bytes: ...


def save_uploaded_file(uploaded_file: UploadedFileLike, pod: str) -> Path:
    """Save an uploaded-file-like object to uploads/<pod>/<ts>_<name>.

    Returns the destination Path (relative to cwd). Sanitizes the original
    name by replacing `/` and `\\` so a malicious filename cannot escape
    the pod directory.
    """
    pod_dir = UPLOADS_ROOT / pod
    pod_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    dest = pod_dir / f"{ts}_{safe_name}"
    dest.write_bytes(uploaded_file.getvalue())
    return dest


def log_upload(
    upload_id: str,
    filename: str,
    saved_path: str,
    pod: str,
    uploader: str,
    label: str | None,
    file_size: int,
    ingest_status: str,
    chunks_created: int = 0,
    error_msg: str | None = None,
) -> None:
    """Append one entry to upload_log.jsonl. Never raises.

    Call once with `ingest_status="pending"` after the bytes land on disk,
    and again with the final status after the ingest pipeline returns.
    """
    if ingest_status not in VALID_INGEST_STATUSES:
        logger.warning("Ignoring unknown ingest_status %r", ingest_status)
        return
    append_jsonl(UPLOAD_LOG_PATH, {
        "upload_id": upload_id,
        "filename": filename,
        "saved_path": saved_path,
        "pod": pod,
        "uploader": uploader,
        "label": label or None,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_size": file_size,
        "ingest_status": ingest_status,
        "chunks_created": chunks_created,
        "error_msg": error_msg,
    })


def latest_per_upload_id() -> dict[str, dict]:
    """Collapse the append-only log into latest-per-upload_id view."""
    latest: dict[str, dict] = {}
    for r in read_jsonl(UPLOAD_LOG_PATH):
        uid = r.get("upload_id")
        if uid:
            latest[uid] = r
    return latest
