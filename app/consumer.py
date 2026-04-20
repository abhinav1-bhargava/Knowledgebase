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
import uuid
from typing import Any, Optional

import streamlit as st

st.set_page_config(
    page_title="PM Onboarding Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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


def _confidence_badge(confidence: float) -> str:
    if confidence >= 0.8:
        return f"🟢 **Confidence: {confidence:.2f}** (high)"
    if confidence >= 0.7:
        return f"🟡 **Confidence: {confidence:.2f}** (medium)"
    return f"🔴 **Confidence: {confidence:.2f}** (low)"


_CITATION_ICON = {
    "document": "📄",
    "jira": "🎫",
    "verified_answer": "✅",
}


def _render_citation(citation) -> None:
    icon = _CITATION_ICON.get(citation.type, "📄")
    if citation.type == "jira" and citation.url:
        st.markdown(f"{icon} [{citation.identifier}]({citation.url})")
    else:
        st.markdown(f"{icon} **{citation.identifier}**")
    if citation.snippet:
        st.caption(citation.snippet)


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
        st.warning("⚠️ Low confidence — verify with an SME before relying on this.")

    st.markdown(result.answer or "_(empty answer)_")
    st.markdown(_confidence_badge(result.confidence))

    if result.citations:
        with st.expander(f"Sources ({len(result.citations)})", expanded=False):
            for cit in result.citations:
                _render_citation(cit)

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
        st.divider()
        st.markdown(
            "**Contributor view** — run "
            "`streamlit run app/contributor.py --server.port 8502`."
        )
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


def main() -> None:
    _init_state()

    st.title("PM Onboarding Assistant")
    st.caption(
        "Ask about your pod's architecture, features, metrics, or history."
    )

    pods, total_chunks = _discover_pods()
    selected_pod = _render_sidebar(pods, total_chunks)
    _render_banners(total_chunks)

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


main()
