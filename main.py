"""Top-level CLI entry point.

Convenience wrapper that exposes ingestion and maintenance commands in one
place so contributors don't need to remember module paths.

Usage:
    python main.py ingest-docs --pod <pod>
    python main.py ingest-jira --project <KEY>
    python main.py ingest-verified
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Dispatch to the requested subcommand."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
