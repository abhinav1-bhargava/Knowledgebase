"""RAG Experimentation Lab — 3-column comparison across 24 combinations.

Lets a user run the same question through three different
(retrieval strategy × chunking strategy × top-k) configurations
side-by-side and compare confidence, latency, retrieved-chunk count,
token usage, estimated cost, and the generated answers.

Retrieval strategies (6): naive, rewriting, hybrid, reranked, agentic,
sentence-window.
Chunking strategies (4): fixed 512 / 768 / 1024 chars, semantic.
(The sentence-window retrieval strategy routes to the dedicated
kb_sentence_window collection regardless of the chunking radio,
because its ingestion format is special.)

Run:
    streamlit run app/rag_lab.py --server.port 8503
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

# `streamlit run app/rag_lab.py` only puts app/ on sys.path; the project
# root isn't added, so sibling-package imports would fail without this.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from design.styles import apply_design_system, render_badge

st.set_page_config(
    page_title="RAG Experimentation Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_design_system()

logger = logging.getLogger(__name__)


# --- Mappings (per the approved Part 7 spec) --------------------------------

CHUNKING_TO_COLLECTION: dict[str, str] = {
    "Fixed 512 chars": "pm_onboarding",
    "Fixed 768 chars": "kb_768",
    "Fixed 1024 chars": "kb_1024",
    "Semantic": "kb_semantic",
}

CHUNKING_TO_STRATEGY_KEY: dict[str, str] = {
    "Fixed 512 chars": "fixed_512",
    "Fixed 768 chars": "fixed_768",
    "Fixed 1024 chars": "fixed_1024",
    "Semantic": "semantic",
}

RETRIEVAL_LABELS: tuple[str, ...] = (
    "Naive RAG",
    "Query Rewriting",
    "Hybrid Search",
    "Re-ranked",
    "Agentic RAG",
    "Sentence Window",
)

# Lazy strategy-function lookup — the query_engine module carries heavy
# imports (chroma client, llama-index, embed factory); don't touch it
# until the user actually runs a comparison.

def _get_query_function(label: str) -> Callable:
    from query.engine import (
        query_agentic,
        query_hybrid,
        query_naive,
        query_reranked,
        query_sentence_window,
        query_with_rewriting,
    )
    return {
        "Naive RAG": query_naive,
        "Query Rewriting": query_with_rewriting,
        "Hybrid Search": query_hybrid,
        "Re-ranked": query_reranked,
        "Agentic RAG": query_agentic,
        "Sentence Window": query_sentence_window,
    }[label]


# The sentence-window strategy ingests with a special one-sentence-per-
# chunk format; it has its own collection regardless of the chunking
# radio the user picked. Everyone else respects the radio selection.
SENTENCE_WINDOW_COLLECTION = "kb_sentence_window"

# Airtel gpt-oss-120b price quoted in the spec.
TOKEN_COST_INR = 0.000034

# Variable-size chunkers don't honour a fixed overlap; grey out the slider.
VARIABLE_CHUNKING = {"Semantic"}


# --- Session state ----------------------------------------------------------


def _init_state() -> None:
    st.session_state.setdefault("pm_lab_results", [None, None, None])
    st.session_state.setdefault("pm_lab_question", "")
    st.session_state.setdefault("pm_lab_pod", "All pods")


# --- Collection inventory sidebar -------------------------------------------


@st.cache_data(ttl=15)
def _collection_inventory() -> dict[str, dict]:
    """Return {chunking_label: {collection, exists, count}} for every chunking."""
    import chromadb

    try:
        from config import CHROMA_PATH

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        existing = {c.name: c for c in client.list_collections()}
    except Exception as exc:
        logger.exception("Failed to list Chroma collections: %s", exc)
        existing = {}

    inventory: dict[str, dict] = {}
    for label, collection_name in CHUNKING_TO_COLLECTION.items():
        col = existing.get(collection_name)
        inventory[label] = {
            "collection": collection_name,
            "exists": col is not None,
            "count": col.count() if col is not None else 0,
        }
    # Sentence-window is shared by the Sentence Window strategy regardless
    # of chunking; include it in the sidebar summary.
    col = existing.get(SENTENCE_WINDOW_COLLECTION)
    inventory["(sentence-window strategy)"] = {
        "collection": SENTENCE_WINDOW_COLLECTION,
        "exists": col is not None,
        "count": col.count() if col is not None else 0,
    }
    return inventory


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Collections")
        inventory = _collection_inventory()
        for label, info in inventory.items():
            if info["exists"]:
                st.success(f"{label} → `{info['collection']}` · {info['count']} chunks")
            else:
                st.warning(f"{label} → `{info['collection']}` · not built")
                strategy_key = _strategy_key_for_sidebar_label(label)
                if strategy_key is not None:
                    btn_key = f"pm_lab_reingest_{info['collection']}"
                    if st.button(
                        f"Re-ingest into `{info['collection']}`",
                        key=btn_key,
                    ):
                        _run_reingest(strategy_key, info["collection"])

        st.divider()
        st.caption(
            "Fixed strategies use SentenceSplitter at 512/768/1024 char "
            "chunks. Semantic splits on topic boundaries. Sentence-window "
            "chunks per sentence and expands ±2 sentences at retrieval."
        )


def _strategy_key_for_sidebar_label(label: str) -> str | None:
    """Map sidebar labels to the ingestion strategy key."""
    if label in CHUNKING_TO_STRATEGY_KEY:
        return CHUNKING_TO_STRATEGY_KEY[label]
    if label == "(sentence-window strategy)":
        return "sentence_window"
    return None


def _run_reingest(chunking_strategy: str, collection_name: str) -> None:
    with st.spinner(
        f"Re-ingesting ./docs into `{collection_name}` "
        f"with strategy `{chunking_strategy}`… (may take a few minutes)"
    ):
        try:
            from ingestion.docs_ingest import ingest_pod

            stats = ingest_pod(
                pod="all",
                docs_dir="./docs",
                chunking_strategy=chunking_strategy,
                collection_name=collection_name,
            )
            st.success(
                f"Ingested {stats.files_processed} file(s) · "
                f"{stats.chunks_created} chunk(s) · errors: {stats.errors}"
            )
            _collection_inventory.clear()
            st.rerun()
        except Exception as exc:
            logger.exception("Re-ingest failed")
            st.error(f"Re-ingest failed: {exc}")


# --- Pod options ------------------------------------------------------------


@st.cache_data(ttl=30)
def _discover_pods() -> list[str]:
    """Pull distinct pod tags from pm_onboarding (the fixed-512 default)."""
    try:
        import chromadb

        from config import CHROMA_PATH

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_or_create_collection("pm_onboarding")
        if col.count() == 0:
            return []
        r = col.get(include=["metadatas"])
        pods: set[str] = set()
        for md in r.get("metadatas") or []:
            pod = (md or {}).get("pod")
            if pod:
                pods.add(str(pod))
        return sorted(pods)
    except Exception:
        logger.exception("Pod discovery failed")
        return []


# --- Per-config render ------------------------------------------------------


def _render_config_controls(column_idx: int) -> dict[str, Any]:
    label = chr(ord("A") + column_idx)
    st.subheader(f"Config {label}")

    retrieval = st.selectbox(
        "Retrieval strategy",
        options=RETRIEVAL_LABELS,
        index=0,
        key=f"pm_lab_retrieval_{column_idx}",
    )
    chunking = st.radio(
        "Chunking",
        options=list(CHUNKING_TO_COLLECTION.keys()),
        index=0,
        key=f"pm_lab_chunking_{column_idx}",
        horizontal=False,
    )
    top_k_default = [8, 16, 24][column_idx]
    top_k = st.slider(
        "Top-K",
        min_value=8,
        max_value=50,
        value=top_k_default,
        step=1,
        key=f"pm_lab_topk_{column_idx}",
    )
    overlap_disabled = chunking in VARIABLE_CHUNKING
    overlap = st.slider(
        "Overlap (chars)",
        min_value=0,
        max_value=200,
        value=50,
        step=5,
        key=f"pm_lab_overlap_{column_idx}",
        disabled=overlap_disabled,
        help=(
            "Variable-size chunker — overlap not applicable."
            if overlap_disabled else
            "Applied at ingest time. Changing this here is informational "
            "unless you re-ingest."
        ),
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key=f"pm_lab_temp_{column_idx}",
        help=(
            "Currently informational — per-query temperature requires "
            "threading the value through _llm_chat. Deferred."
        ),
    )

    return {
        "label": label,
        "retrieval": retrieval,
        "chunking": chunking,
        "top_k": top_k,
        "overlap": overlap,
        "temperature": temperature,
    }


def _resolve_collection(retrieval: str, chunking: str) -> str:
    if retrieval == "Sentence Window":
        return SENTENCE_WINDOW_COLLECTION
    return CHUNKING_TO_COLLECTION[chunking]


def _estimate_tokens(answer: str | None) -> int:
    if not answer:
        return 0
    # The spec's heuristic: word count * 1.3 ≈ tokens.
    return int(len(answer.split()) * 1.3)


def _run_one_config(
    question: str,
    pod_filter: str | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute one query and return a dict of result fields for display."""
    retrieval = config["retrieval"]
    chunking = config["chunking"]
    top_k = int(config["top_k"])
    collection = _resolve_collection(retrieval, chunking)

    query_fn = _get_query_function(retrieval)
    start = time.time()
    error = None
    result = None
    try:
        result = query_fn(
            question,
            pod=pod_filter,
            collection_name=collection,
            top_k=top_k,
        )
    except Exception as exc:
        logger.exception("Config %s failed", config.get("label"))
        error = str(exc)
    latency = time.time() - start

    return {
        "config": config,
        "collection": collection,
        "result": result,
        "error": error,
        "latency": latency,
    }


def _render_result(col, payload: dict[str, Any]) -> None:
    config = payload["config"]
    collection = payload["collection"]
    result = payload["result"]
    error = payload.get("error")
    latency = payload.get("latency", 0.0)

    with col:
        st.caption(
            f"Strategy **{config['retrieval']}** · chunking "
            f"**{config['chunking']}** · collection `{collection}`"
        )
        if error:
            st.error(f"Query failed: {error}")
            return
        if result is None:
            st.info("No result.")
            return

        # Confidence
        pct = int(round((result.confidence or 0.0) * 100))
        if result.confidence >= 0.8:
            variant = "success"
        elif result.confidence >= 0.5:
            variant = "warn"
        else:
            variant = "danger"
        st.markdown(
            f"**Confidence** &nbsp;"
            f"{render_badge(f'{pct}%', variant=variant)}",
            unsafe_allow_html=True,
        )

        # Key metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency (s)", f"{latency:.2f}")
        m2.metric("Chunks", len(result.citations or []))
        tokens = _estimate_tokens(result.answer)
        m3.metric("Tokens", f"~{tokens}")
        cost = tokens * TOKEN_COST_INR
        m4.metric("Est. cost", f"₹{cost:.4f}")

        # Answer + sources
        with st.expander("Answer", expanded=True):
            st.markdown(result.answer or "_(empty)_")
        if result.citations:
            with st.expander(f"Sources ({len(result.citations)})"):
                for i, cit in enumerate(result.citations, 1):
                    st.markdown(f"**[{i}]** `{cit.identifier}` &nbsp;*({cit.type})*")
                    if cit.snippet:
                        st.caption(cit.snippet)


# --- Main -------------------------------------------------------------------


def main() -> None:
    _init_state()
    _render_sidebar()

    st.title("RAG Experimentation Lab")
    st.caption(
        "Compare retrieval strategies and chunking approaches side-by-side. "
        "Six retrieval × four chunking = up to 24 combinations."
    )

    shared_c1, shared_c2 = st.columns([3, 1])
    question = shared_c1.text_input(
        "Test question",
        key="pm_lab_question",
        placeholder="What's the HDO fleet's productivity target?",
    )
    pod_options = ["All pods"] + _discover_pods()
    pod_choice = shared_c2.selectbox(
        "Pod filter",
        options=pod_options,
        index=0,
        key="pm_lab_pod",
    )
    pod_filter = None if pod_choice == "All pods" else pod_choice

    cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
    configs = []
    for i, col in enumerate((cfg_c1, cfg_c2, cfg_c3)):
        with col:
            configs.append(_render_config_controls(i))

    run = st.button(
        "Run comparison",
        type="primary",
        key="pm_lab_run",
        disabled=not question.strip(),
    )

    if run:
        with st.spinner("Running three configurations sequentially…"):
            payloads = [
                _run_one_config(question.strip(), pod_filter, cfg)
                for cfg in configs
            ]
        st.session_state.pm_lab_results = payloads

    payloads = st.session_state.get("pm_lab_results") or [None, None, None]
    if any(p is not None for p in payloads):
        st.divider()
        st.markdown("### Results")
        res_c1, res_c2, res_c3 = st.columns(3)
        for col, payload in zip((res_c1, res_c2, res_c3), payloads):
            if payload is None:
                continue
            _render_result(col, payload)


if __name__ == "__main__":
    main()
