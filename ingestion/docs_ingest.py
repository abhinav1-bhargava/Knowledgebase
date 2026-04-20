"""Ingest local documents (Confluence HTML exports, PDFs, DOCX) into ChromaDB.

Walks a docs directory (organized by pod), parses each supported file format,
chunks the text using llama-index node parsers, embeds the chunks with the
configured OpenAI embedding model, and persists them in the ChromaDB
collection with metadata (pod, source_type, source_path, title, ingested_at).

Usage:
    python -m ingestion.docs_ingest --pod <pod_name> [--docs-dir ./docs]
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_html(path: str) -> str:
    """Parse a Confluence HTML export and return clean text."""
    raise NotImplementedError


def load_pdf(path: str) -> str:
    """Extract text content from a PDF file."""
    raise NotImplementedError


def load_docx(path: str) -> str:
    """Extract text content from a DOCX file."""
    raise NotImplementedError


def ingest_pod(pod: str, docs_dir: str = "./docs") -> int:
    """Ingest all supported documents for a pod. Returns number of chunks indexed."""
    raise NotImplementedError


def main() -> None:
    """CLI entry point."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
