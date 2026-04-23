"""BM25 keyword index — build, persist, search.

Hybrid search needs a keyword signal alongside the vector one. rank_bm25's
BM25Okapi is built once per Chroma collection and pickled under
`./bm25_indices/<collection>.pkl`; search loads the pickle, tokenizes the
query the same way, and returns `[(doc_text, score, metadata)]` ordered
by descending BM25 score.

Design notes:
    - Tokenization is deliberately whitespace + lowercase so it matches
      the spec and stays obvious. Fancier tokenizers (stemming,
      stop-words, ngrams) are a follow-up if relevance plateaus.
    - Search auto-builds the index on first use if the pickle doesn't
      exist, so callers don't need to remember to prime it. A manual
      rebuild is still available via the CLI after bulk uploads (the
      index doesn't watch Chroma for changes).
    - Empty collections are handled: we persist `{"bm25": None}` so
      `search_bm25` can return [] cleanly instead of raising.

Usage:
    from query.bm25_index import search_bm25, build_bm25_index
    hits = search_bm25("productivity target", "pm_onboarding", top_k=50)

CLI:
    python -m query.bm25_index --collection pm_onboarding
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from config import CHROMA_PATH

logger = logging.getLogger(__name__)

BM25_DIR = Path("./bm25_indices")


def _index_path(collection_name: str) -> Path:
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    return BM25_DIR / f"{collection_name}.pkl"


def _tokenize(text: str) -> list[str]:
    """Whitespace + lowercase. Matches the spec; kept simple on purpose."""
    return (text or "").lower().split()


def build_bm25_index(collection_name: str) -> Path:
    """Pull every chunk from `collection_name` and persist a fresh BM25 pickle.

    Returns the path to the written index file. Callers should treat the
    returned path as stable — it lives under ./bm25_indices/ (gitignored).
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(name=collection_name)
    count = col.count()

    if count == 0:
        logger.warning(
            "Collection %s is empty — writing an empty BM25 stub "
            "so search_bm25 can return [] cleanly", collection_name,
        )
        path = _index_path(collection_name)
        with path.open("wb") as f:
            pickle.dump({"bm25": None, "docs": [], "metadatas": []}, f)
        return path

    result = col.get(include=["documents", "metadatas"])
    docs = list(result.get("documents") or [])
    metadatas = list(result.get("metadatas") or [])
    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)

    path = _index_path(collection_name)
    with path.open("wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs, "metadatas": metadatas}, f)
    logger.info("BM25 index written: %s (%d chunks)", path, len(docs))
    return path


def search_bm25(
    question: str,
    collection_name: str,
    top_k: int = 50,
    auto_build: bool = True,
) -> list[tuple[str, float, dict]]:
    """Return top-k `(doc_text, bm25_score, metadata)` tuples.

    When `auto_build` is True (default) and the pickle is missing, the
    index is built on demand — first hit for a new collection pays the
    build cost; subsequent calls are cheap.
    """
    path = _index_path(collection_name)
    if not path.exists():
        if not auto_build:
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run "
                f"`python -m query.bm25_index --collection {collection_name}` "
                f"or call build_bm25_index({collection_name!r}) first."
            )
        logger.info("BM25 index for %s missing — auto-building now", collection_name)
        build_bm25_index(collection_name)

    with path.open("rb") as f:
        data = pickle.load(f)

    bm25 = data.get("bm25")
    if bm25 is None:
        return []

    docs: list[str] = data.get("docs") or []
    metadatas: list[dict] = data.get("metadatas") or []
    tokens = _tokenize(question)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        (docs[i], float(scores[i]), (metadatas[i] if i < len(metadatas) else {}) or {})
        for i in order
    ]


def main() -> None:
    """CLI entry: rebuild a BM25 index for a given Chroma collection."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build or rebuild a BM25 keyword index over a Chroma collection.",
    )
    parser.add_argument(
        "--collection", required=True, help="Chroma collection name (e.g. pm_onboarding)",
    )
    args = parser.parse_args()
    path = build_bm25_index(args.collection)
    print(f"BM25 index written to: {path}")


if __name__ == "__main__":
    main()
