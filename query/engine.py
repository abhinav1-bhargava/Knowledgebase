"""Retrieval + generation engine.

Wraps a LlamaIndex retriever over the ChromaDB vector store, applies an
optional pod metadata filter, asks the configured OpenAI LLM to answer
using only the retrieved context (per SYSTEM_PROMPT), then parses the
response into a structured QueryResult including dedup'd citations,
mean-similarity confidence, and follow-up questions.

Side effects:
    - Every call appends a record to query_log.jsonl.
    - Calls with confidence < CONFIDENCE_THRESHOLD also append to
      gaps.jsonl so contributors can resolve them later.

OpenAI clients are instantiated lazily (singletons) so importing this
module is side-effect free — tests can swap in fakes before the first
call by assigning to the module-level _collection / _embed_model / _llm
globals (or by calling reset_clients()).

CLI:
    python -m query.engine "your question here" [--pod <pod>] [--top-k N]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, LLMMetadata, MessageRole
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


class _OpenAICompatibleLLM(OpenAI):
    """OpenAI subclass that bypasses llama_index's hardcoded model-name
    enums for non-OpenAI-branded deployments (Airtel's gpt-oss-120b).

    The parent's `metadata` property looks up context window and chat
    capability from tables that only contain OpenAI-branded names and
    raises ValueError for unknown models. We return sensible hardcoded
    values — gpt-oss-120b is a chat model with a 128K context window.
    """

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
        )
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    AIRTEL_LLM_API_KEY,
    AIRTEL_LLM_BASE_URL,
    CHROMA_PATH,
    CONFIDENCE_THRESHOLD,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    RETRIEVAL_TOP_K,
)
from query.embed_factory import get_embed_model, verify_collection_dim
from query.prompts import SYSTEM_PROMPT, build_user_prompt
from storage.gaps import log_gap
from storage.query_log import log_query

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pm_onboarding"
FOLLOWUP_MARKER = "Suggested follow-ups:"
SNIPPET_MAX_CHARS = 200


@dataclass
class Citation:
    """A single deduplicated source cited by the answer."""

    type: str  # "document" | "jira" | "verified_answer"
    identifier: str
    snippet: str
    url: Optional[str] = None


@dataclass
class QueryResult:
    """Structured response returned by `query()`."""

    answer: str
    citations: list[Citation]
    confidence: float
    low_confidence: bool
    follow_ups: list[str]
    retrieved_chunks: list[dict] = field(default_factory=list)


# --- Lazy singletons --------------------------------------------------------

_collection = None
_embed_model = None
_llm = None
_index = None

# Per-collection caches, populated on demand when retrieval targets a
# non-default collection (the RAG Lab compares across kb_768, kb_1024,
# kb_semantic alongside the default pm_onboarding).
_collections_by_name: dict[str, object] = {}
_indexes_by_name: dict[str, "VectorStoreIndex"] = {}


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def _get_collection_by_name(name: str):
    if name not in _collections_by_name:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collections_by_name[name] = client.get_or_create_collection(name=name)
    return _collections_by_name[name]


def _get_embed_model():
    """Delegate to the provider-agnostic embedding factory (cached there)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = get_embed_model()
    return _embed_model


def _get_llm():
    """Build and cache the LLM client per LLM_PROVIDER.

    airtel → hits the Airtel-hosted OpenAI-compatible endpoint; the
    llama_index OpenAI class doesn't validate the model name at
    construction time, so custom model ids (e.g. gpt-oss-120b) work
    with api_base set.
    openai → unchanged SaaS path.
    """
    global _llm
    if _llm is None:
        provider = (LLM_PROVIDER or "openai").lower()
        if provider == "airtel":
            _llm = _OpenAICompatibleLLM(
                model=LLM_MODEL,
                api_key=AIRTEL_LLM_API_KEY,
                api_base=AIRTEL_LLM_BASE_URL,
            )
            logger.info(
                "LLM provider: airtel (model=%s at %s)",
                LLM_MODEL, AIRTEL_LLM_BASE_URL,
            )
        else:
            _llm = OpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
            logger.info("LLM provider: openai (model=%s)", LLM_MODEL)
    return _llm


def _get_index() -> VectorStoreIndex:
    global _index
    if _index is None:
        vector_store = ChromaVectorStore(chroma_collection=_get_collection())
        _index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=_get_embed_model(),
        )
    return _index


def _get_index_for(collection_name: str) -> VectorStoreIndex:
    if collection_name not in _indexes_by_name:
        col = _get_collection_by_name(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=col)
        _indexes_by_name[collection_name] = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=_get_embed_model(),
        )
    return _indexes_by_name[collection_name]


def reset_clients() -> None:
    """Drop cached Chroma/embedding/LLM/index instances. Mainly for tests."""
    global _collection, _embed_model, _llm, _index
    _collection = None
    _embed_model = None
    _llm = None
    _index = None
    _collections_by_name.clear()
    _indexes_by_name.clear()


# --- Retrieval --------------------------------------------------------------


def _retrieve(
    question: str,
    pod: str | None,
    top_k: int,
    collection_name: str | None = None,
) -> list[dict]:
    """Retrieve top_k chunks, optionally filtered by pod and targeting a
    named collection. `collection_name=None` falls through to the default
    singleton collection for back-compat."""
    if collection_name is None:
        col = _get_collection()
        index = _get_index()
    else:
        col = _get_collection_by_name(collection_name)
        index = _get_index_for(collection_name)
    verify_collection_dim(col)
    filters = None
    if pod:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="pod", value=pod, operator=FilterOperator.EQ)],
        )
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    nodes = retriever.retrieve(question)
    return [
        {
            "text": n.get_content(),
            "score": float(n.score) if n.score is not None else 0.0,
            "metadata": dict(n.metadata or {}),
        }
        for n in nodes
    ]


# --- Citation building & response parsing -----------------------------------


def _citation_identifier(metadata: dict) -> tuple[str, str]:
    """Return (citation_type, identifier) for a retrieved chunk's metadata."""
    source_type = metadata.get("source_type", "")
    if source_type == "jira":
        return "jira", metadata.get("issue_key") or metadata.get("filename") or "UNKNOWN-JIRA"
    if source_type == "verified":
        ident = (
            metadata.get("verified_id")
            or metadata.get("filename")
            or "verified_answer"
        )
        return "verified_answer", ident
    ident = metadata.get("filename") or metadata.get("relative_path") or "UNKNOWN"
    return "document", ident


def _build_citations(chunks: list[dict]) -> list[Citation]:
    """Build citations from retrieved chunks, deduped by (type, identifier)."""
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for c in chunks:
        ctype, ident = _citation_identifier(c["metadata"])
        if (ctype, ident) in seen:
            continue
        seen.add((ctype, ident))
        snippet = " ".join((c.get("text") or "").split())
        if len(snippet) > SNIPPET_MAX_CHARS:
            snippet = snippet[: SNIPPET_MAX_CHARS - 3] + "..."
        url = c["metadata"].get("url") if isinstance(c.get("metadata"), dict) else None
        out.append(Citation(type=ctype, identifier=ident, snippet=snippet, url=url))
    return out


def _split_answer_and_followups(text: str) -> tuple[str, list[str]]:
    """Split an LLM response at 'Suggested follow-ups:' into (answer, items)."""
    idx = text.find(FOLLOWUP_MARKER)
    if idx == -1:
        return text.strip(), []
    answer = text[:idx].rstrip()
    tail = text[idx + len(FOLLOWUP_MARKER) :]
    follow_ups: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*", "•")):
            follow_ups.append(stripped.lstrip("-*• ").strip())
        elif re.match(r"^\d+[.)]\s+", stripped):
            follow_ups.append(re.sub(r"^\d+[.)]\s+", "", stripped))
    return answer, follow_ups


# --- LLM call ---------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _llm_chat(llm, system_prompt: str, user_prompt: str) -> str:
    """Invoke the LLM with a system + user message. Retries up to 3 times."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]
    resp = llm.chat(messages)
    return str(resp.message.content or "")


# --- Public entry point -----------------------------------------------------


def _generate_from_chunks(
    question: str,
    retrieved: list[dict],
    pod: str | None,
    start: float | None = None,
) -> QueryResult:
    """Generate the QueryResult from already-retrieved chunks.

    Shared by every retrieval strategy so confidence/LLM/citation/logging
    behaviour stays identical across naive, rewriting, hybrid, reranked,
    agentic, and sentence-window.
    """
    if start is None:
        start = time.time()

    if not retrieved:
        answer = "No sources found for this question."
        result = QueryResult(
            answer=answer,
            citations=[],
            confidence=0.0,
            low_confidence=True,
            follow_ups=[],
            retrieved_chunks=[],
        )
        log_query(
            question=question,
            pod=pod,
            answer=answer,
            confidence=0.0,
            low_confidence=True,
            citations=[],
        )
        log_gap(question=question, pod=pod, confidence=0.0, retrieval_scores=[])
        return result

    confidence = sum(c["score"] for c in retrieved) / len(retrieved)
    low_confidence = confidence < CONFIDENCE_THRESHOLD

    user_prompt = build_user_prompt(question, retrieved)

    try:
        raw = _llm_chat(_get_llm(), SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        logger.exception("LLM call failed after retries: %s", exc)
        raw = ""

    if not raw:
        answer = "The LLM did not return a response. Please retry."
        follow_ups: list[str] = []
    else:
        answer, follow_ups = _split_answer_and_followups(raw)

    citations = _build_citations(retrieved)

    result = QueryResult(
        answer=answer,
        citations=citations,
        confidence=confidence,
        low_confidence=low_confidence,
        follow_ups=follow_ups,
        retrieved_chunks=retrieved,
    )

    log_query(
        question=question,
        pod=pod,
        answer=answer,
        confidence=confidence,
        low_confidence=low_confidence,
        citations=[asdict(c) for c in citations],
    )
    if low_confidence:
        log_gap(
            question=question,
            pod=pod,
            confidence=confidence,
            retrieval_scores=[c["score"] for c in retrieved],
        )

    latency_ms = int((time.time() - start) * 1000)
    logger.info(
        "query finished in %d ms (confidence=%.3f, low=%s, chunks=%d)",
        latency_ms,
        confidence,
        low_confidence,
        len(retrieved),
    )
    return result


def query(
    question: str,
    pod: str | None = None,
    top_k: int | None = None,
    collection_name: str | None = None,
) -> QueryResult:
    """Naive RAG: embed → retrieve top-K → generate.

    Retrieves top-K chunks (optionally filtered by pod, optionally
    targeting a non-default collection), asks the LLM to answer using
    only the retrieved context, parses follow-ups, and logs the query
    (plus a gap record if confidence is below CONFIDENCE_THRESHOLD).
    """
    start = time.time()
    effective_top_k = top_k or RETRIEVAL_TOP_K
    retrieved = _retrieve(question, pod, effective_top_k, collection_name=collection_name)
    return _generate_from_chunks(question, retrieved, pod, start=start)


# Alias exposing the naive path under the naming convention the RAG Lab
# uses across all six strategies (query_naive / query_with_rewriting /
# query_hybrid / query_reranked / query_agentic / query_sentence_window).
def query_naive(
    question: str,
    pod: str | None = None,
    collection_name: str | None = None,
    top_k: int | None = None,
) -> QueryResult:
    """Naive RAG — same as `query`, with RAG-Lab-shaped signature."""
    return query(question, pod=pod, top_k=top_k, collection_name=collection_name)


def query_with_rewriting(
    question: str,
    pod: str | None = None,
    collection_name: str | None = None,
    top_k: int | None = None,
) -> QueryResult:
    """Retrieve with rephrased query variants and merge results.

    Strategy: expand the original question into up to 3 phrasings via
    the LLM (original + 2 rewrites), retrieve top-K chunks for each, then
    deduplicate on chunk text keeping the highest score per chunk. The
    deduped, re-sorted top-K is passed to the same generation path the
    naive query uses, so downstream logging and citation shape stay
    identical.

    Expect ~3× the embedding / retrieval cost of naive, plus one LLM
    call for the rewrite. Falls back to naive if the rewrite LLM fails
    (rewrite_query returns [original] only).
    """
    from query.rewriting import rewrite_query

    start = time.time()
    effective_top_k = top_k or RETRIEVAL_TOP_K

    variants = rewrite_query(question)
    logger.info("Rewriting produced %d variant(s)", len(variants))

    all_hits: list[dict] = []
    for variant in variants:
        hits = _retrieve(variant, pod, effective_top_k, collection_name=collection_name)
        all_hits.extend(hits)

    # Dedupe by chunk text, keep the highest score seen for each.
    seen: dict[str, dict] = {}
    for hit in all_hits:
        key = hit.get("text") or ""
        if not key:
            continue
        if key not in seen or hit["score"] > seen[key]["score"]:
            seen[key] = hit

    deduped = sorted(seen.values(), key=lambda c: c["score"], reverse=True)[:effective_top_k]
    return _generate_from_chunks(question, deduped, pod, start=start)


# --- CLI --------------------------------------------------------------------


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _cli_print(result: QueryResult) -> None:
    print("\n=== Answer ===")
    print(result.answer or "(no answer)")

    level = "low" if result.low_confidence else "high"
    print(f"\nConfidence: {result.confidence:.3f} ({level})")

    if result.citations:
        print("\nCitations:")
        for c in result.citations:
            url = f"  <{c.url}>" if c.url else ""
            print(f"  [{c.type}] {c.identifier}{url}")
    else:
        print("\nCitations: (none)")

    if result.follow_ups:
        print("\nFollow-ups:")
        for fu in result.follow_ups:
            print(f"  - {fu}")


def main() -> None:
    """CLI entry point."""
    _configure_logging()
    parser = argparse.ArgumentParser(description="Query the PM onboarding assistant.")
    parser.add_argument("question", type=str, help="Natural-language question")
    parser.add_argument("--pod", default=None, help="Optional pod metadata filter")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-K")
    args = parser.parse_args()

    try:
        result = query(args.question, pod=args.pod, top_k=args.top_k)
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        sys.exit(2)

    _cli_print(result)


if __name__ == "__main__":
    main()
