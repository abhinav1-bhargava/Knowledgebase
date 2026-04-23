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
from llama_index.vector_stores.chroma import ChromaVectorStore
from pypdf import PdfReader

from config import (
    CHROMA_PATH,
)
from ingestion.chunking_strategies import (
    DEFAULT_COLLECTION_BY_STRATEGY,
    SUPPORTED_CHUNKING_STRATEGIES,
    build_splitter,
)
from query.embed_factory import get_embed_model, verify_collection_dim

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pm_onboarding"  # default / back-compat; overridable via ingest_pod(collection_name=...)
DEFAULT_CHUNKING_STRATEGY = "fixed_512"

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
    """Outcome counters + per-file status lists for a single ingest run.

    The list-valued fields let callers (e.g. the contributor UI's upload
    flow) attribute a final per-file status back to their own upload
    records without re-deriving it from file system state.
    """

    files_processed: int = 0
    files_skipped_dedup: int = 0
    files_replaced: int = 0
    chunks_created: int = 0
    errors: int = 0
    error_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    processed_files: list[str] = field(default_factory=list)
    chunks_per_file: dict[str, int] = field(default_factory=dict)


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


def _get_collection(collection_name: str = COLLECTION_NAME):
    """Return (creating if needed) the named persistent collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=collection_name)


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
        stats.skipped_files.append(rel_path)
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
    stats.processed_files.append(rel_path)
    stats.chunks_per_file[rel_path] = len(nodes)
    logger.info("Indexed %s -> %d chunk(s)", rel_path, len(nodes))


# --- Public API -------------------------------------------------------------


def ingest_pod(
    pod: str,
    docs_dir: str = "./docs",
    chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY,
    collection_name: str | None = None,
) -> IngestionStats:
    """Ingest all supported documents under `docs_dir` into the named collection.

    - `chunking_strategy` picks which node parser to build (see
      ingestion/chunking_strategies.SUPPORTED_CHUNKING_STRATEGIES). Default
      is "fixed_512" which reproduces the pre-factory behaviour
      (SentenceSplitter, 512/50) for back-compat with existing ingests.
    - `collection_name` defaults to the strategy's canonical collection
      (fixed_512 → `pm_onboarding`, fixed_768 → `kb_768`, etc.) — callers
      can override to experiment with arbitrary collection names.

    Creates `docs_dir` if missing. Returns an IngestionStats; does not
    raise on per-file errors (those are logged and accumulated).
    """
    if chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise ValueError(
            f"Unsupported chunking_strategy {chunking_strategy!r}. "
            f"Must be one of {SUPPORTED_CHUNKING_STRATEGIES}."
        )
    if collection_name is None:
        collection_name = DEFAULT_COLLECTION_BY_STRATEGY[chunking_strategy]

    docs_path = Path(docs_dir).resolve()
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        logger.info("Created docs directory at %s", docs_path)

    files = _iter_files(docs_path)
    stats = IngestionStats()

    if not files:
        logger.warning("No supported files found under %s", docs_path)
        return stats

    logger.info(
        "Found %d candidate file(s) under %s (strategy=%s, collection=%s)",
        len(files), docs_path, chunking_strategy, collection_name,
    )

    collection = _get_collection(collection_name)
    verify_collection_dim(collection)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = get_embed_model()
    splitter = build_splitter(chunking_strategy, embed_model=embed_model)
    pipeline = IngestionPipeline(
        transformations=[splitter, embed_model],
        vector_store=vector_store,
    )

    for file_path in files:
        _ingest_file(file_path, docs_path, pod, collection, pipeline, stats)

    # Auto-rebuild the BM25 keyword index for this collection if anything
    # actually changed. Hybrid search goes stale against fresh uploads
    # otherwise; rebuild is cheap (whole-corpus re-tokenize + re-index,
    # seconds for ~1000 docs). Failures here must NOT break ingestion —
    # the vector side is already persisted; log and continue.
    if stats.files_processed > 0 or stats.files_replaced > 0:
        try:
            from query.bm25_index import build_bm25_index

            build_bm25_index(collection_name)
            logger.info(
                "BM25 index rebuilt for collection %s (post-ingest)", collection_name,
            )
        except Exception as exc:
            logger.warning(
                "BM25 index rebuild failed for collection %s (ingestion itself was OK): %s",
                collection_name, exc,
            )

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
        description="Ingest local documents into a ChromaDB collection.",
    )
    parser.add_argument("--pod", required=True, help="Pod name to tag the ingested chunks with")
    parser.add_argument("--docs-dir", default="./docs", help="Root directory to scan recursively")
    parser.add_argument(
        "--chunking",
        default=DEFAULT_CHUNKING_STRATEGY,
        choices=list(SUPPORTED_CHUNKING_STRATEGIES),
        help=(
            "Chunking strategy. fixed_512/768/1024 are SentenceSplitter variants; "
            "semantic uses SemanticSplitterNodeParser (topic-boundary splits)."
        ),
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help=(
            "Target Chroma collection. Defaults to the strategy's canonical "
            "collection (fixed_512→pm_onboarding, fixed_768→kb_768, "
            "fixed_1024→kb_1024, semantic→kb_semantic)."
        ),
    )
    args = parser.parse_args()

    stats = ingest_pod(
        pod=args.pod,
        docs_dir=args.docs_dir,
        chunking_strategy=args.chunking,
        collection_name=args.collection_name,
    )
    _print_summary(args.pod, stats)

    failed_only = stats.errors > 0 and stats.files_processed == 0
    sys.exit(1 if failed_only else 0)


if __name__ == "__main__":
    main()
