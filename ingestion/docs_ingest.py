"""Ingest local documents (Confluence HTML exports, PDFs, DOCX, Markdown, text)
into the persistent ChromaDB collection used by the retrieval layer.

Walks a docs directory (organized by pod) recursively, parses each supported
file format with a format-specific loader, chunks the resulting text with
LlamaIndex's SentenceSplitter (preferring paragraph and heading boundaries),
embeds each chunk with the configured OpenAI embedding model, and persists
chunks in the `pm_onboarding` Chroma collection with a consistent metadata
schema.

Metadata attached to every chunk:
    filename, relative_path, source_type, page_number (PDFs only),
    ingestion_date (UTC ISO-8601), doc_hash (SHA256 of file bytes), pod.

Deduplication:
    - If a chunk with the same `doc_hash` already exists in the collection,
      the file is skipped entirely (no re-embedding, no re-insertion).
    - Otherwise, any existing chunks matching (relative_path, pod) are
      deleted before the new chunks are inserted, so updated files cleanly
      replace stale ones.

Usage:
    python -m ingestion.docs_ingest --pod <pod_name> [--docs-dir ./docs]

Example:
    python -m ingestion.docs_ingest --pod recharge --docs-dir ./docs
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import chromadb
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from pypdf import PdfReader

from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pm_onboarding"

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}


@dataclass
class IngestionStats:
    """Outcome counters for a single ingest run."""

    files_processed: int = 0
    files_skipped_dedup: int = 0
    files_replaced: int = 0
    chunks_created: int = 0
    errors: int = 0
    error_files: list[str] = field(default_factory=list)


# --- File hashing & loaders -------------------------------------------------


def _file_hash(path: Path) -> str:
    """Return a SHA256 hex digest of the file's bytes (streamed in 64KB blocks)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pdf(path: Path) -> list[Document]:
    """Return one Document per page (1-indexed page_number metadata)."""
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(Document(text=text, metadata={"page_number": page_num}))
    return docs


def load_docx(path: Path) -> list[Document]:
    """Return a single Document containing all non-empty paragraphs."""
    doc = DocxDocument(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n\n".join(paragraphs).strip()
    return [Document(text=text)] if text else []


def load_html(path: Path) -> list[Document]:
    """Return a single Document with heading hierarchy preserved as markdown."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = html_to_markdown_text(raw)
    return [Document(text=text)] if text else []


def html_to_markdown_text(html: str) -> str:
    """Convert HTML to plain text, preserving h1..h6 as markdown headings.

    Strips <script>, <style>, <nav>, and <footer> entirely. Other block
    elements are followed by a newline so paragraphs survive into chunking.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    for level in range(1, 7):
        for h in list(soup.find_all(f"h{level}")):
            heading_text = h.get_text(" ", strip=True)
            if heading_text:
                h.replace_with(soup.new_string(f"\n\n{'#' * level} {heading_text}\n\n"))
            else:
                h.decompose()
    for tag in soup.find_all(["p", "li", "br", "div", "tr"]):
        tag.append(soup.new_string("\n"))
    text = soup.get_text()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def load_markdown(path: Path) -> list[Document]:
    """Return a single Document with the file's text verbatim."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Document(text=text)] if text else []


def load_text(path: Path) -> list[Document]:
    """Return a single Document with the file's text verbatim."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Document(text=text)] if text else []


LOADERS: dict[str, Callable[[Path], list[Document]]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
}


# --- Chroma helpers ---------------------------------------------------------


def _get_collection():
    """Return (creating if needed) the persistent pm_onboarding collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _doc_hash_exists(collection, doc_hash: str) -> bool:
    """True if any chunk with this doc_hash is already in the collection."""
    res = collection.get(where={"doc_hash": doc_hash}, limit=1)
    return bool(res.get("ids"))


def _delete_chunks_for_path(collection, relative_path: str, pod: str) -> int:
    """Delete all chunks tagged with this (relative_path, pod). Returns count."""
    res = collection.get(
        where={"$and": [{"relative_path": relative_path}, {"pod": pod}]},
    )
    ids = res.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


# --- Per-file ingestion -----------------------------------------------------


def _iter_files(docs_dir: Path) -> list[Path]:
    """Sorted list of supported files anywhere under docs_dir."""
    return sorted(
        p
        for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _ingest_file(
    file_path: Path,
    docs_dir: Path,
    pod: str,
    collection,
    pipeline: IngestionPipeline,
    stats: IngestionStats,
) -> None:
    """Hash, dedup, parse, chunk, embed, and insert a single file."""
    rel_path = str(file_path.relative_to(docs_dir))
    suffix = file_path.suffix.lower()
    source_type = SUPPORTED_EXTENSIONS[suffix]

    try:
        doc_hash = _file_hash(file_path)
    except OSError as exc:
        logger.error("Failed to read %s: %s", rel_path, exc)
        stats.errors += 1
        stats.error_files.append(rel_path)
        return

    if _doc_hash_exists(collection, doc_hash):
        logger.info("Skipping %s — identical content already indexed (doc_hash match)", rel_path)
        stats.files_skipped_dedup += 1
        return

    deleted = _delete_chunks_for_path(collection, rel_path, pod)
    if deleted:
        logger.info("Replacing %d stale chunk(s) for %s (content changed)", deleted, rel_path)
        stats.files_replaced += 1

    loader = LOADERS[suffix]
    try:
        documents = loader(file_path)
    except Exception as exc:
        logger.exception("Failed to parse %s: %s", rel_path, exc)
        stats.errors += 1
        stats.error_files.append(rel_path)
        return

    if not documents:
        logger.warning("No extractable text in %s — skipping", rel_path)
        return

    base_metadata = {
        "filename": file_path.name,
        "relative_path": rel_path,
        "source_type": source_type,
        "ingestion_date": datetime.now(timezone.utc).isoformat(),
        "doc_hash": doc_hash,
        "pod": pod,
    }
    for doc in documents:
        merged = {**base_metadata, **doc.metadata}
        # Chroma rejects None metadata values; drop them so absence == null.
        doc.metadata = {k: v for k, v in merged.items() if v is not None}

    try:
        nodes = pipeline.run(documents=documents, show_progress=False)
    except Exception as exc:
        logger.exception("Failed to embed/index %s: %s", rel_path, exc)
        stats.errors += 1
        stats.error_files.append(rel_path)
        return

    stats.files_processed += 1
    stats.chunks_created += len(nodes)
    logger.info("Indexed %s -> %d chunk(s)", rel_path, len(nodes))


# --- Public API -------------------------------------------------------------


def ingest_pod(pod: str, docs_dir: str = "./docs") -> IngestionStats:
    """Ingest all supported documents under docs_dir for the given pod.

    Creates docs_dir if it does not exist. Returns an IngestionStats with
    counters describing the run; does not raise on per-file errors.
    """
    docs_path = Path(docs_dir).resolve()
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        logger.info("Created docs directory at %s", docs_path)

    files = _iter_files(docs_path)
    stats = IngestionStats()

    if not files:
        logger.warning("No supported files found under %s", docs_path)
        return stats

    logger.info("Found %d candidate file(s) under %s", len(files), docs_path)

    collection = _get_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",
    )
    embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    pipeline = IngestionPipeline(
        transformations=[splitter, embed_model],
        vector_store=vector_store,
    )

    for file_path in files:
        _ingest_file(file_path, docs_path, pod, collection, pipeline, stats)

    return stats


# --- CLI --------------------------------------------------------------------


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _print_summary(pod: str, stats: IngestionStats) -> None:
    print("\n=== Ingestion Summary ===")
    print(f"  pod:                   {pod}")
    print(f"  files processed:       {stats.files_processed}")
    print(f"  files skipped (dedup): {stats.files_skipped_dedup}")
    print(f"  files replaced:        {stats.files_replaced}")
    print(f"  chunks created:        {stats.chunks_created}")
    print(f"  errors:                {stats.errors}")
    if stats.error_files:
        print("  failing files:")
        for ef in stats.error_files:
            print(f"    - {ef}")


def main() -> None:
    """CLI entry point. Exits non-zero only when every file errored."""
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Ingest local documents into the pm_onboarding ChromaDB collection.",
    )
    parser.add_argument("--pod", required=True, help="Pod name to tag the ingested chunks with")
    parser.add_argument("--docs-dir", default="./docs", help="Root directory to scan recursively")
    args = parser.parse_args()

    stats = ingest_pod(pod=args.pod, docs_dir=args.docs_dir)
    _print_summary(args.pod, stats)

    failed_only = stats.errors > 0 and stats.files_processed == 0
    sys.exit(1 if failed_only else 0)


if __name__ == "__main__":
    main()
