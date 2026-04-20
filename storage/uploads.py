"""Filesystem helpers for contributor-side uploads.

Kept separate from app/contributor.py so the save + log paths can be
unit-tested without importing the Streamlit UI (whose module-level
script runs `main()` at import time and would pollute Streamlit's
internal state when imported outside a proper runtime).

Layout on disk:
    uploads/<pod>/<unix_ts>_<sanitized_original_filename>
    uploads/upload_log.jsonl        (one JSON object per upload event)

upload_log.jsonl record:
    {filename, pod, uploader, uploaded_at (UTC ISO-8601),
     label, file_size, status}

`status` is a free-text string: "saved" on success, or "save_error: ..."
on failure, so downstream tools can audit what actually made it to disk.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from storage._jsonl import append_jsonl

logger = logging.getLogger(__name__)

UPLOADS_ROOT = Path("uploads")
UPLOAD_LOG_PATH = UPLOADS_ROOT / "upload_log.jsonl"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB — triggers a UI warning, not a hard block


class UploadedFileLike(Protocol):
    """Streamlit's UploadedFile shape we actually depend on."""

    name: str
    size: int

    def getvalue(self) -> bytes: ...


def log_upload(
    filename: str,
    pod: str,
    uploader: str,
    label: str | None,
    file_size: int,
    status: str,
) -> None:
    """Append an audit record to uploads/upload_log.jsonl."""
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    append_jsonl(UPLOAD_LOG_PATH, {
        "filename": filename,
        "pod": pod,
        "uploader": uploader,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "label": label or None,
        "file_size": file_size,
        "status": status,
    })


def save_uploaded_file(uploaded_file: UploadedFileLike, pod: str) -> Path:
    """Save an uploaded-file-like object to uploads/<pod>/<ts>_<name>.

    Returns the destination Path. Sanitizes the original name by replacing
    `/` and `\\` so a malicious filename cannot escape the pod directory.
    """
    pod_dir = UPLOADS_ROOT / pod
    pod_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    dest = pod_dir / f"{ts}_{safe_name}"
    dest.write_bytes(uploaded_file.getvalue())
    return dest
