"""Ingest Jira tickets into ChromaDB via the Atlassian REST API.

Pulls issues from the configured Jira project (JIRA_PROJECT_KEY by default),
extracts summary, description, status, assignee, and recent comments, and
indexes them with metadata (pod, source_type='jira', issue_key, status,
updated_at) so they are filterable alongside docs.

Usage:
    python -m ingestion.jira_ingest --project <PROJECT_KEY> [--pod <pod>]
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def fetch_issues(project_key: str, jql: str | None = None) -> Iterable[dict]:
    """Yield issues from Jira matching the project (and optional JQL filter)."""
    raise NotImplementedError


def issue_to_document(issue: dict, pod: str) -> dict:
    """Convert a raw Jira issue payload into a llama-index Document-shaped dict."""
    raise NotImplementedError


def ingest_project(project_key: str, pod: str) -> int:
    """Ingest all issues for a project. Returns number of issues indexed."""
    raise NotImplementedError


def main() -> None:
    """CLI entry point."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
