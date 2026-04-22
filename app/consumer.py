"""Consumer chat UI (Streamlit).

A chat interface for product managers to query the pod knowledge base.
Reads distinct pods from ChromaDB metadata for the sidebar selector,
calls `query.engine.query` on submit, and renders each assistant turn
with a traffic-light confidence badge, an expandable Sources list,
clickable follow-up chips that re-submit on click, and a feedback row
whose presses append to feedback.jsonl.

Slash commands:
    /whats-tried <topic>     → rewrites to a Jira-focused prompt and queries
    /pod-overview, /people   → stubs (surfaced as info banners)

Safe to launch even when the OpenAI key is missing or still a placeholder:
the page renders with a warning banner and queries simply fail gracefully.

Run:
    streamlit run app/consumer.py
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

# `streamlit run app/consumer.py` only puts app/ on sys.path; the project root
# isn't added, so `from storage...` / `from config` would fail. Insert the
# project root up front so sibling packages import cleanly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import re

import streamlit as st

from design.styles import (
    apply_design_system,
    render_badge,
    render_citation_marker,
    render_source,
)

st.set_page_config(
    page_title="PM Onboarding — Consumer",
    page_icon=None,
    layout="wide" if __file__.endswith("contributor.py") else "centered",
    initial_sidebar_state="expanded",
)
apply_design_system()

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pm_onboarding"

# --- Safe config probing ----------------------------------------------------
# Import config defensively so a missing .env doesn't prevent the page from
# rendering — the user should see the warning banner, not a stack trace.
_CONFIG_ERROR: Optional[str] = None
_LLM_CONFIGURED: bool = False
_CHROMA_PATH: Optional[str] = None
try:
    from config import CHROMA_PATH as _CFG_CHROMA_PATH
    from config import OPENAI_API_KEY as _CFG_OPENAI_API_KEY

    _CHROMA_PATH = _CFG_CHROMA_PATH
    _LLM_CONFIGURED = bool(_CFG_OPENAI_API_KEY) and not _CFG_OPENAI_API_KEY.startswith(
        "sk-placeholder"
    )
except Exception as exc:
    _CONFIG_ERROR = str(exc)

from storage.feedback import log_feedback


# --- Engine invocation (deferred import) ------------------------------------


def _run_query(question: str, pod: Optional[str], top_k: Optional[int] = None):
    """Import query.engine lazily so the UI module loads even with a broken config."""
    from query.engine import query as _query

    return _query(question, pod=pod, top_k=top_k)


# --- Chroma discovery -------------------------------------------------------


@st.cache_data(ttl=30)
def _discover_pods() -> tuple[list[str], int]:
    """Return (sorted distinct pods, total chunk count) from Chroma.

    Cached for 30s so sidebar rendering doesn't hammer Chroma on every rerun.
    """
    if not _CHROMA_PATH:
        return [], 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        col = client.get_or_create_collection(name=COLLECTION_NAME)
        count = col.count()
        if count == 0:
            return [], 0
        result = col.get(include=["metadatas"])
        pods: set[str] = set()
        for md in result.get("metadatas") or []:
            pod = (md or {}).get("pod")
            if pod:
                pods.add(str(pod))
        return sorted(pods), count
    except Exception:
        logger.exception("Failed to read pods from Chroma")
        return [], 0


# --- Session state ----------------------------------------------------------


def _init_state() -> None:
    st.session_state.setdefault("pm_messages", [])
    st.session_state.setdefault("pm_user", "")
    st.session_state.setdefault("pm_selected_pod", None)
    st.session_state.setdefault("pm_pending_question", None)


# --- Submission dispatch (slash commands + plain queries) -------------------

SLASH_STUBS = {"/pod-overview", "/people"}


def _handle_submission(question: str, pod: Optional[str]) -> dict[str, Any]:
    """Route a user submission to the engine or a slash-command handler.

    Returns a message dict with at most one of: result, error, info.
    """
    query_id = str(uuid.uuid4())
    msg: dict[str, Any] = {
        "query_id": query_id,
        "question": question,
        "result": None,
        "error": None,
        "info": None,
    }

    if question.startswith("/"):
        parts = question.split(maxsplit=1)
        cmd = parts[0]
        rest = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/whats-tried":
            if not rest:
                msg["info"] = (
                    "Usage: `/whats-tried <topic>` — describe the topic to dig "
                    "into prior attempts, outcomes, and Jira references."
                )
                return msg
            rewritten = (
                f"What has been tried or attempted regarding: {rest}? "
                "Include outcomes, resolutions, and ticket references."
            )
            try:
                msg["result"] = _run_query(rewritten, pod, top_k=12)
            except Exception as exc:
                logger.exception("Slash /whats-tried failed")
                msg["error"] = str(exc)
            return msg

        if cmd in SLASH_STUBS:
            msg["info"] = f"`{cmd}` — slash command coming in a later phase."
            return msg

        msg["info"] = f"Unknown slash command: `{cmd}`"
        return msg

    try:
        msg["result"] = _run_query(question, pod)
    except Exception as exc:
        logger.exception("Query failed")
        msg["error"] = str(exc)
    return msg


# --- Rendering helpers ------------------------------------------------------


_CITATION_PATTERN = re.compile(r"\[([^\[\]]+?)\]")


def _renumber_answer_citations(
    answer_text: str, citations: list
) -> tuple[str, list]:
    """Rewrite LLM-produced ``[identifier]`` markers in the answer into
    sequentially numbered ``[1]``/``[2]`` citation links and return the
    corresponding ordered list of Citation objects.

    Identifiers that don't match any known citation pass through unchanged
    (e.g., the prompt's rule-3 placeholder ``[topic area]``).
    """
    ident_to_cit = {c.identifier: c for c in citations}
    order: list[str] = []
    ident_to_num: dict[str, int] = {}

    def replace(match: "re.Match[str]") -> str:
        ident = match.group(1).strip()
        if ident not in ident_to_cit:
            return match.group(0)
        if ident not in ident_to_num:
            order.append(ident)
            ident_to_num[ident] = len(order)
        n = ident_to_num[ident]
        return render_citation_marker(n, ident)

    rewritten = _CITATION_PATTERN.sub(replace, answer_text)
    numbered = [ident_to_cit[i] for i in order]
    # Any citations not referenced inline still get rendered below, after
    # the ones that are, so every retrieved source remains visible.
    referenced = set(order)
    trailing = [c for c in citations if c.identifier not in referenced]
    return rewritten, numbered + trailing


def _citation_meta_line(citation) -> str:
    """Short metadata string for the source card header area."""
    bits = [citation.type]
    if citation.url:
        bits.append(citation.url)
    return " · ".join(bits)


def _render_feedback_row(msg: dict[str, Any]) -> None:
    qid = msg["query_id"]
    pod = st.session_state.get("pm_selected_pod")
    user = st.session_state.get("pm_user") or "anonymous"
    answer = (msg["result"].answer if msg.get("result") else "") or ""

    cols = st.columns([1, 1, 1, 1.2, 4])
    buttons = [
        (cols[0], "👍 Helpful", "helpful"),
        (cols[1], "👎 Wrong", "wrong"),
        (cols[2], "📅 Outdated", "outdated"),
        (cols[3], "📎 Wrong source", "wrong_source"),
    ]
    for col, label, ftype in buttons:
        if col.button(label, key=f"fb_{ftype}_{qid}"):
            log_feedback(
                query_id=qid,
                question=msg["question"],
                answer=answer,
                feedback_type=ftype,
                user=user,
                pod=pod,
            )
            st.toast(f"Feedback logged: {ftype}", icon="✅")


def _render_assistant_turn(msg: dict[str, Any]) -> None:
    if msg.get("info"):
        st.info(msg["info"])
        return
    if msg.get("error"):
        st.error(f"Query failed: {msg['error']}")
        return

    result = msg.get("result")
    if result is None:
        st.error("No result returned.")
        return

    if result.low_confidence:
        st.warning("Low confidence — verify with an SME before relying on this.")

    answer_text = result.answer or "_(empty answer)_"
    rewritten, ordered_citations = _renumber_answer_citations(
        answer_text, list(result.citations or [])
    )
    st.markdown(rewritten, unsafe_allow_html=True)

    pct = int(round((result.confidence or 0.0) * 100))
    if result.confidence >= 0.8:
        variant = "success"
    elif result.confidence >= 0.5:
        variant = "warn"
    else:
        variant = "danger"
    st.markdown(
        render_badge(f"{pct}% confidence", variant=variant),
        unsafe_allow_html=True,
    )

    if ordered_citations:
        st.markdown(
            f'<h4 style="margin-top: 20px;">Sources ({len(ordered_citations)})</h4>',
            unsafe_allow_html=True,
        )
        for n, cit in enumerate(ordered_citations, 1):
            st.markdown(
                render_source(
                    num=n,
                    title=cit.identifier,
                    meta=_citation_meta_line(cit),
                    snippet=cit.snippet or "",
                ),
                unsafe_allow_html=True,
            )

    if result.follow_ups:
        st.markdown("**Follow-ups:**")
        cols = st.columns(min(len(result.follow_ups), 3))
        for i, fu in enumerate(result.follow_ups):
            col = cols[i % len(cols)]
            if col.button(fu, key=f"fu_{msg['query_id']}_{i}"):
                st.session_state.pm_pending_question = fu
                st.rerun()

    _render_feedback_row(msg)


# --- Sidebar ----------------------------------------------------------------


def _render_sidebar(pods: list[str], total_chunks: int) -> Optional[str]:
    with st.sidebar:
        st.header("Settings")
        st.text_input("Your name", key="pm_user", placeholder="e.g. Asha M.")
        options = ["All pods"] + pods
        selected = st.selectbox(
            "Pod",
            options=options,
            index=0,
            help="Filter retrieval by pod metadata.",
        )
        selected_pod = None if selected == "All pods" else selected
        st.session_state.pm_selected_pod = selected_pod
        st.metric("Queries this session", len(st.session_state.pm_messages))
        st.caption(f"Indexed chunks: {total_chunks}")
    return selected_pod


# --- Banners ----------------------------------------------------------------


def _render_banners(total_chunks: int) -> None:
    if _CONFIG_ERROR:
        st.error(f"Configuration error — {_CONFIG_ERROR}")
        return
    if not _LLM_CONFIGURED:
        st.warning(
            "LLM not configured. Queries will fail until `OPENAI_API_KEY` is set in `.env`."
        )
    if total_chunks == 0:
        st.info(
            "No documents indexed yet. Run "
            "`python -m ingestion.docs_ingest --pod <name>` first."
        )


# --- Main -------------------------------------------------------------------


def _prewarm_embeddings() -> None:
    """Trigger the embedding factory once per session so the first query
    doesn't freeze on a 130MB first-run HuggingFace download."""
    if st.session_state.get("pm_embed_prewarmed"):
        return
    try:
        with st.spinner(
            "Loading embedding model (first run may download ~130MB)..."
        ):
            from query.embed_factory import get_embed_model, get_model_dim

            get_embed_model()
            get_model_dim()
        st.session_state.pm_embed_prewarmed = True
    except Exception as exc:
        logger.warning("Embedding prewarm failed: %s", exc)


_EMPTY_STATE_EXAMPLES = [
    "What does the recharge flow do?",
    "What metrics does platform engineering track?",
    "What's been tried recently with notification delivery?",
]


def _render_empty_state() -> None:
    st.markdown(
        '<div class="kb-card" style="text-align:center; padding: 48px 24px; margin-top: 24px;">'
        '<div style="font-size: 24px; font-weight: 600; color: var(--ink-100); '
        'letter-spacing: -0.01em; margin-bottom: 8px;">'
        "Ask about products, features, or technical systems"
        "</div>"
        '<div style="font-size: 15px; color: var(--ink-500);">'
        "Get grounded answers with sources across the division."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
    cols = st.columns(len(_EMPTY_STATE_EXAMPLES))
    for i, q in enumerate(_EMPTY_STATE_EXAMPLES):
        if cols[i].button(q, key=f"pm_example_q_{i}"):
            st.session_state.pm_pending_question = q
            st.rerun()


def main() -> None:
    _init_state()
    _prewarm_embeddings()

    st.title("PM Onboarding Assistant")
    st.caption(
        "Ask about your pod's architecture, features, metrics, or history."
    )

    pods, total_chunks = _discover_pods()
    selected_pod = _render_sidebar(pods, total_chunks)
    _render_banners(total_chunks)

    # Empty state — first load, no questions yet.
    if not st.session_state.pm_messages:
        _render_empty_state()

    # Replay chat history
    for past in st.session_state.pm_messages:
        with st.chat_message("user"):
            st.markdown(past["question"])
        with st.chat_message("assistant"):
            _render_assistant_turn(past)

    typed = st.chat_input("Ask a question about your pod...")
    pending = st.session_state.pop("pm_pending_question", None)
    question = typed or pending
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            msg = _handle_submission(question, selected_pod)
        _render_assistant_turn(msg)

    st.session_state.pm_messages.append(msg)


if __name__ == "__main__":
    main()
