"""Shared append-only JSONL helpers.

Every storage module in this package uses the same append/read contract:
records are written one JSON object per line, reads are tolerant of
missing files and malformed lines, and write failures are logged rather
than raised so an analytics/storage hiccup cannot kill the UI path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file. Missing file → empty list."""
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line in %s", path)
    except OSError as exc:
        logger.error("Failed to read %s: %s", path, exc)
    return records


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single record. Never raises — logs and swallows on failure."""
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to append to %s: %s", path, exc)
