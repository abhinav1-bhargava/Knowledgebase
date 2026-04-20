"""Ingest contributor-curated verified answers into ChromaDB.

Verified answers are short, authoritative Q/A pairs written by pod
contributors via the contributor UI. They are stored in JSONL on disk
(storage/verified_answers) and indexed with source_type='verified' and a
high trust weight so the retrieval layer can prefer them over raw docs
when relevance is comparable.

Usage:
    python -m ingestion.verified_ingest [--pod <pod>]
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def ingest_verified(pod: str | None = None) -> int:
    """Index all verified answers (optionally filtered by pod). Returns count."""
    raise NotImplementedError


def main() -> None:
    """CLI entry point."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
