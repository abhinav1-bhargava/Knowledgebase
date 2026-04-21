"""Contributor curation UI (Streamlit).

Six tabs that let pod leads and SMEs curate gaps into verified answers,
track source health, manage SMEs, and view pod knowledge coverage:

    1. Gap Queue       — clustered open gaps with Claim / Write Answer actions
    2. Answer Editor   — single form for draft/publish, auto-prefills when
                         launched from a gap or a verified-answer Edit
    3. Sources         — per-source hits, thumbs-down, status, flag actions
    4. Verified Answers — latest view per answer_id, edit / retire / reviewed
    5. SME Directory   — CRUD over storage/sme_directory.jsonl
    6. Pod Health      — 7d/30d metrics and charts from query/feedback/gap logs

Safe to launch even with a broken config: the page mounts with a banner,
the query-engine import stays deferred, and storage helpers treat missing
JSONL files as empty so first-run never blows up.

Run:
    streamlit run app/contributor.py --server.port 8502
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# `streamlit run app/contributor.py` only puts app/ on sys.path; the project
# root isn't added, so `from storage...` / `from config` would fail. Insert
# the project root up front so sibling packages import cleanly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="PM Knowledge Base — Contributor",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pm_onboarding"

# --- Safe config probing ----------------------------------------------------
_CONFIG_ERROR: Optional[str] = None
_CHROMA_PATH: Optional[str] = None
try:
    from config import CHROMA_PATH as _CFG_CHROMA_PATH

    _CHROMA_PATH = _CFG_CHROMA_PATH
except Exception as exc:
    _CONFIG_ERROR = str(exc)

from storage.gaps import (
    GapCluster,
    claim_gap,
    cluster_gaps,
    resolve_gap,
)
from storage.uploads import (
    MAX_UPLOAD_BYTES,
    UPLOADS_ROOT,
    log_upload,
    save_uploaded_file,
)
import uuid
from storage.sme_directory import add_sme, deactivate_sme, list_smes
from storage.source_inventory import (
    deprecate_source,
    flag_source,
    list_sources,
    unflag_source,
)
from storage.verified_answers import (
    list_answers,
    save_answer,
    update_answer,
)


# --- Session state ----------------------------------------------------------


def _init_state() -> None:
    st.session_state.setdefault("pm_contrib_user", "")
    st.session_state.setdefault("pm_contrib_selected_pod", None)
    st.session_state.setdefault("pm_contrib_authoring_for", None)
    st.session_state.setdefault("pm_contrib_editing_answer_id", None)


# --- Chroma pod discovery ---------------------------------------------------


@st.cache_data(ttl=30)
def _discover_pods() -> tuple[list[str], int]:
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


# --- Sidebar ----------------------------------------------------------------


def _render_sidebar(pods: list[str], total_chunks: int) -> Optional[str]:
    with st.sidebar:
        st.header("Settings")
        options = ["All pods"] + pods
        selected = st.selectbox("Pod", options=options, index=0)
        selected_pod = None if selected == "All pods" else selected
        st.session_state.pm_contrib_selected_pod = selected_pod
        st.caption(f"Indexed chunks: {total_chunks}")
        st.divider()
        user = st.session_state.get("pm_contrib_user") or "(not set)"
        st.caption(f"Signed in as: {user}")
        st.markdown(
            "**Consumer view:** `streamlit run app/consumer.py`"
        )
    return selected_pod


# --- Tab 1: Gap Queue -------------------------------------------------------


def _render_gap_queue(pod: Optional[str]) -> None:
    st.subheader("Gap Queue")
    c1, c2 = st.columns([1, 1])
    status_filter = c1.selectbox(
        "Status",
        ["open", "claimed", "resolved", "all"],
        index=0,
        key="pm_contrib_gap_status_filter",
    )
    days_back = c2.number_input(
        "Days back",
        min_value=1,
        max_value=365,
        value=30,
        key="pm_contrib_gap_days",
    )
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(days_back))).isoformat()

    clusters: list[GapCluster] = cluster_gaps(pod=pod)
    if status_filter != "all":
        clusters = [c for c in clusters if c.latest_status == status_filter]
    clusters = [c for c in clusters if c.last_asked >= cutoff]
    # Frequency desc, then recency desc (tuple sort with reverse=True).
    clusters.sort(key=lambda c: (c.frequency, c.last_asked), reverse=True)

    if not clusters:
        st.info("No gaps match your filters.")
        return

    header = st.columns([5, 1, 2, 1.5, 1, 1.5])
    for col, label in zip(
        header,
        ["Question", "Freq", "Last asked", "Status", "Claim", "Write Answer"],
    ):
        col.markdown(f"**{label}**")

    user = st.session_state.get("pm_contrib_user") or "anonymous"
    for cluster in clusters:
        row = st.columns([5, 1, 2, 1.5, 1, 1.5])
        row[0].write(cluster.representative_question)
        row[1].write(cluster.frequency)
        row[2].write((cluster.last_asked or "")[:10])
        row[3].write(cluster.latest_status)
        if row[4].button("Claim", key=f"claim_{cluster.cluster_id}"):
            claim_gap(cluster.cluster_id, user)
            st.toast(f"Claimed cluster {cluster.cluster_id[:8]}", icon="✅")
            st.rerun()
        if row[5].button("Write Answer", key=f"write_{cluster.cluster_id}"):
            st.session_state.pm_contrib_authoring_for = cluster.cluster_id
            st.session_state.pm_contrib_editing_answer_id = None
            st.toast("Opened in Answer Editor tab — click to continue.", icon="📝")


# --- Tab 2: Answer Editor ---------------------------------------------------


def _find_cluster(cluster_id: str) -> Optional[GapCluster]:
    for c in cluster_gaps(pod=None):
        if c.cluster_id == cluster_id:
            return c
    return None


def _render_answer_editor(selected_pod: Optional[str], pods: list[str]) -> None:
    st.subheader("Answer Editor")

    authoring_id = st.session_state.get("pm_contrib_authoring_for")
    editing_id = st.session_state.get("pm_contrib_editing_answer_id")

    gap_cluster: Optional[GapCluster] = None
    prefill: dict = {}
    if authoring_id:
        gap_cluster = _find_cluster(authoring_id)
        if gap_cluster is None:
            st.warning(
                "Cluster no longer present (may have been resolved). Starting a fresh form."
            )
            st.session_state.pm_contrib_authoring_for = None
            authoring_id = None
    if editing_id and not authoring_id:
        for a in list_answers():
            if a.get("answer_id") == editing_id:
                prefill = a
                break

    if authoring_id:
        st.caption(f"Authoring for gap cluster `{authoring_id}`")
    elif editing_id:
        st.caption(f"Editing answer `{editing_id[:8]}…`")

    default_question = (
        gap_cluster.representative_question
        if gap_cluster
        else prefill.get("question", "")
    )
    pod_options = ["(none)"] + pods
    default_pod_value = prefill.get("pod") or selected_pod or "(none)"
    if default_pod_value not in pod_options:
        pod_options.append(default_pod_value)
    pod_idx = pod_options.index(default_pod_value)

    default_sources = "\n".join(prefill.get("source_references") or [])
    default_status_idx = 0 if (prefill.get("status") or "Draft") != "Published" else 1
    default_due_str = prefill.get("review_due_date")
    try:
        default_due = date.fromisoformat(default_due_str[:10]) if default_due_str else (date.today() + timedelta(days=180))
    except ValueError:
        default_due = date.today() + timedelta(days=180)

    with st.form("pm_contrib_answer_form", clear_on_submit=False):
        question_val = st.text_input(
            "Question",
            value=default_question,
            disabled=bool(gap_cluster),
            key="pm_contrib_ae_question",
        )
        pod_val = st.selectbox("Pod", pod_options, index=pod_idx, key="pm_contrib_ae_pod")
        answer_val = st.text_area(
            "Answer",
            value=prefill.get("answer", ""),
            height=300,
            help="Markdown supported. Be concise and cite specific sources.",
            key="pm_contrib_ae_answer",
        )
        sources_val = st.text_area(
            "Source references",
            value=default_sources,
            help="One per line. Format: type:identifier — e.g. document:architecture.pdf or jira:PDS-115",
            key="pm_contrib_ae_sources",
        )
        sme_val = st.text_input(
            "SME to consult",
            value=prefill.get("sme") or "",
            key="pm_contrib_ae_sme",
        )
        status_val = st.radio(
            "Status",
            ["Draft", "Published"],
            index=default_status_idx,
            horizontal=True,
            key="pm_contrib_ae_status",
        )
        due_val = st.date_input(
            "Review due date",
            value=default_due,
            key="pm_contrib_ae_due",
        )
        submitted = st.form_submit_button("Save")

    if not submitted:
        return

    # Resolve the actual question (read-only disabled field keeps the prefill).
    effective_question = default_question if gap_cluster else question_val.strip()
    effective_pod = None if pod_val == "(none)" else pod_val
    sources = [s.strip() for s in sources_val.splitlines() if s.strip()]

    errors: list[str] = []
    if not effective_question:
        errors.append("Question is required.")
    if not answer_val.strip():
        errors.append("Answer is required.")
    if status_val == "Published":
        if not effective_pod:
            errors.append("Pod is required when publishing.")
        if not sources:
            errors.append("At least one source reference is required when publishing.")
    if errors:
        for err in errors:
            st.error(err)
        return

    user = st.session_state.get("pm_contrib_user") or "anonymous"
    fields = {
        "question": effective_question,
        "pod": effective_pod,
        "answer": answer_val.strip(),
        "source_references": sources,
        "sme": sme_val.strip() or None,
        "status": status_val,
        "review_due_date": due_val.isoformat(),
        "author": user,
    }

    if editing_id:
        update_answer(editing_id, fields, edited_by=user)
        answer_id = editing_id
    else:
        answer_id = save_answer(fields)

    if status_val == "Published":
        # TODO (Phase G): replace this log line with an actual call to
        # reindex_verified(answer_id) so published answers land in Chroma
        # with source_type='verified' and high trust weight.
        logger.info("Would trigger reindex_verified() for answer_id=%s", answer_id)
        if authoring_id:
            resolve_gap(authoring_id, answer_id)

    st.session_state.pm_contrib_authoring_for = None
    st.session_state.pm_contrib_editing_answer_id = None
    st.toast(f"Saved {status_val.lower()} answer {answer_id[:8]}", icon="✅")
    st.rerun()


# --- Tab 3: Sources ---------------------------------------------------------


def _status_update(status_obj, **kwargs) -> None:
    """Wrap st.status().update() so bare-mode callers (tests) don't crash
    when st.status returns None outside a Streamlit runtime."""
    if status_obj is not None and hasattr(status_obj, "update"):
        try:
            status_obj.update(**kwargs)
        except Exception:
            pass


def _classify_ingest_status(
    rel_name: str, stats,
) -> tuple[str, int, str | None]:
    """Map a saved file's name against IngestionStats → (status, chunks, error_msg)."""
    if rel_name in stats.error_files:
        return "failed", 0, "ingest pipeline reported a parse/embed error"
    if rel_name in stats.skipped_files:
        return "skipped_dedup", 0, None
    if rel_name in stats.processed_files:
        return "success", stats.chunks_per_file.get(rel_name, 0), None
    # Ingester never saw the file (shouldn't normally happen).
    return "failed", 0, "ingester did not visit this file"


def _process_uploads(
    uploaded_files: list,
    pod: str,
    label: str | None,
    uploader: str,
) -> None:
    """Two-phase upload: save + `pending` log; then ingest + final-status log.

    If `ingest_pod` raises (e.g. dim mismatch, VPN down, misconfig), saved
    files stay on disk and every pending record flips to `failed` with the
    exception message, so a later CLI run can retry against ./uploads/.
    """
    pod_dir = UPLOADS_ROOT / pod
    # Per-upload bookkeeping so we can emit the final record later.
    records: list[dict] = []

    with st.status("Processing uploads...", expanded=True) as status:
        for uploaded in uploaded_files:
            upload_id = str(uuid.uuid4())
            size = int(getattr(uploaded, "size", 0) or 0)
            try:
                dest = save_uploaded_file(uploaded, pod)
                log_upload(
                    upload_id=upload_id,
                    filename=uploaded.name,
                    saved_path=str(dest),
                    pod=pod,
                    uploader=uploader,
                    label=label,
                    file_size=size,
                    ingest_status="pending",
                )
                records.append({
                    "upload_id": upload_id,
                    "filename": uploaded.name,
                    "saved_path": str(dest),
                    "rel_name": dest.name,  # key into IngestionStats.*_files
                    "file_size": size,
                })
                st.write(f"⏳ Saved `{dest}` (pending ingest)")
            except Exception as exc:
                logger.exception("Failed to save %s", uploaded.name)
                log_upload(
                    upload_id=upload_id,
                    filename=uploaded.name,
                    saved_path="",
                    pod=pod,
                    uploader=uploader,
                    label=label,
                    file_size=size,
                    ingest_status="failed",
                    error_msg=f"save_error: {exc}",
                )
                st.write(f"❌ Save failed for `{uploaded.name}`: {exc}")

        if not records:
            _status_update(status,label="No files saved", state="error")
            return

        st.write(f"Running ingestion over `{pod_dir}` (pod={pod})...")
        try:
            from ingestion.docs_ingest import ingest_pod

            stats = ingest_pod(pod=pod, docs_dir=str(pod_dir))
        except Exception as exc:
            # Embedding / dim-mismatch / VPN-down / misconfig — treat every
            # pending record as failed and preserve the files on disk.
            logger.exception("ingest_pod raised")
            err_text = str(exc)
            for rec in records:
                log_upload(
                    upload_id=rec["upload_id"],
                    filename=rec["filename"],
                    saved_path=rec["saved_path"],
                    pod=pod,
                    uploader=uploader,
                    label=label,
                    file_size=rec["file_size"],
                    ingest_status="failed",
                    error_msg=err_text,
                )
            _status_update(status,label=f"Ingestion error: {err_text}", state="error")
            st.warning(
                f"Files saved to `./uploads/{pod}/`. Ingestion deferred — "
                f"fix embeddings config and run: "
                f"`python -m ingestion.docs_ingest --pod {pod} --docs-dir ./uploads/{pod}`"
            )
            st.toast("Ingestion failed — files preserved, see warning", icon="❌")
            return

        # Phase 2 — per-file final status from IngestionStats.
        for rec in records:
            status_label, chunks, err = _classify_ingest_status(rec["rel_name"], stats)
            log_upload(
                upload_id=rec["upload_id"],
                filename=rec["filename"],
                saved_path=rec["saved_path"],
                pod=pod,
                uploader=uploader,
                label=label,
                file_size=rec["file_size"],
                ingest_status=status_label,
                chunks_created=chunks,
                error_msg=err,
            )
            icon = {"success": "✅", "skipped_dedup": "⏭️", "failed": "❌"}[status_label]
            st.write(
                f"{icon} `{rec['filename']}` → **{status_label}**"
                + (f" ({chunks} chunks)" if chunks else "")
            )

        is_complete = stats.errors == 0
        _status_update(status,
            label=(
                f"Ingested {stats.files_processed} file(s), "
                f"{stats.files_skipped_dedup} skipped, "
                f"{stats.chunks_created} chunk(s)"
            ),
            state="complete" if is_complete else "error",
        )
        st.toast(
            f"{len(records)} upload(s) • {stats.files_processed} indexed • "
            f"{stats.files_skipped_dedup} dedup • {stats.chunks_created} chunk(s)",
            icon="✅" if is_complete else "⚠️",
        )


def _render_upload_section(selected_pod: Optional[str], pods: list[str]) -> None:
    with st.expander("📤 Upload new documents", expanded=False):
        pod_options = pods[:]
        default_idx = 0
        if selected_pod and selected_pod in pod_options:
            default_idx = pod_options.index(selected_pod)
        elif not pod_options:
            pod_options = ["(type pod below)"]

        with st.form("pm_contrib_upload_form", clear_on_submit=False):
            uploaded = st.file_uploader(
                "Choose files (PDF / DOCX / Markdown / TXT / HTML)",
                type=["pdf", "docx", "md", "markdown", "txt", "html", "htm"],
                accept_multiple_files=True,
                key="pm_contrib_src_uploader",
            )
            c1, c2 = st.columns([1, 2])
            upload_pod_choice = c1.selectbox(
                "Tag with pod",
                pod_options,
                index=default_idx,
                key="pm_contrib_src_upload_pod",
            )
            pod_text_override = c2.text_input(
                "...or new pod name (wins if set)",
                key="pm_contrib_src_upload_pod_new",
                placeholder="e.g. recharge",
            )
            st.text_input(
                "Source label (optional)",
                placeholder="e.g. Q2 PRD v3",
                key="pm_contrib_src_upload_label",
            )
            submitted = st.form_submit_button(
                "Ingest uploaded files", type="primary"
            )

        if not submitted:
            return

        # Validate submission.
        if not uploaded:
            st.error("Upload at least one file before clicking Ingest.")
            return
        effective_pod = (pod_text_override or "").strip() or upload_pod_choice
        if effective_pod == "(type pod below)":
            effective_pod = ""
        if not effective_pod:
            st.error("Select a pod from the dropdown or enter one in the text input.")
            return

        # Large-file warning (informational; doesn't block).
        large = [f for f in uploaded if getattr(f, "size", 0) > MAX_UPLOAD_BYTES]
        if large:
            st.warning(
                "Large file(s) (>50MB): "
                + ", ".join(f.name for f in large)
                + " — ingestion embedding cost/time scales with size."
            )

        uploader = st.session_state.get("pm_contrib_user") or "anonymous"
        label = st.session_state.get("pm_contrib_src_upload_label") or None
        _process_uploads(uploaded, effective_pod, label, uploader)
        # Cached pod list now stale — fresh pod shows up in sidebar on rerun.
        _discover_pods.clear()
        st.rerun()


def _render_sources(pod: Optional[str]) -> None:
    st.subheader("Sources")
    pods, _total = _discover_pods()
    _render_upload_section(pod, pods)
    st.divider()
    sort_by = st.radio(
        "Sort by",
        ["Hits (30d)", "Thumbs-down (30d)"],
        horizontal=True,
        key="pm_contrib_src_sort",
    )
    sources = list_sources(pod=pod, window_days=30)
    if sort_by == "Hits (30d)":
        sources.sort(key=lambda s: s["hits"], reverse=True)
    else:
        sources.sort(key=lambda s: s["thumbs_down"], reverse=True)

    if not sources:
        st.info("No indexed sources yet.")
        return

    header = st.columns([3, 1, 1, 1, 1.5, 1, 1, 1.5])
    for col, label in zip(
        header,
        ["Identifier", "Type", "Pod", "Chunks", "Ingested", "Hits", "👎", "Status"],
    ):
        col.markdown(f"**{label}**")

    user = st.session_state.get("pm_contrib_user") or "anonymous"
    for s in sources:
        ident = s["identifier"]
        stype = s["source_type"]
        row = st.columns([3, 1, 1, 1, 1.5, 1, 1, 1.5])
        row[0].write(ident)
        row[1].write(stype)
        row[2].write(s.get("pod") or "-")
        row[3].write(s["chunks"])
        row[4].write((s.get("ingested") or "")[:10] or "-")
        row[5].write(s["hits"])
        row[6].write(s["thumbs_down"])
        current_status = s.get("status", "active")
        row[7].write(f"_{current_status}_")

        with st.expander(f"Actions — {ident}"):
            if current_status == "flagged":
                if st.button("Unflag", key=f"unflag_{stype}_{ident}"):
                    unflag_source(ident, stype, user)
                    st.toast("Unflagged", icon="✅")
                    st.rerun()
            else:
                reason = st.text_input(
                    "Reason",
                    key=f"flagreason_{stype}_{ident}",
                    placeholder="Why is this being flagged?",
                )
                if st.button("Flag for review", key=f"flag_{stype}_{ident}"):
                    flag_source(ident, stype, user, reason or "")
                    st.toast("Flagged for review", icon="🚩")
                    st.rerun()

            confirm = st.checkbox(
                "I confirm deprecation (retrieval will stop preferring this source — see TODO in query/engine.py)",
                key=f"dep_confirm_{stype}_{ident}",
            )
            if st.button(
                "Mark deprecated",
                key=f"dep_{stype}_{ident}",
                disabled=not confirm,
            ):
                deprecate_source(ident, stype, user)
                st.toast("Marked deprecated", icon="🗑️")
                st.rerun()


# --- Tab 4: Verified Answers ------------------------------------------------


def _render_verified_answers(pod: Optional[str]) -> None:
    st.subheader("Verified Answers")

    c1, c2 = st.columns([1, 1])
    status_filter = c1.selectbox(
        "Status",
        ["All", "Draft", "Published", "Retired"],
        index=0,
        key="pm_contrib_va_status",
    )
    overdue_only = c2.checkbox("Overdue review only", key="pm_contrib_va_overdue")

    answers = list_answers(
        pod=pod,
        status=None if status_filter == "All" else status_filter,
        overdue_only=overdue_only,
    )
    answers.sort(key=lambda a: (a.get("review_due_date") or "9999-12-31"))

    if not answers:
        st.info("No verified answers match your filters.")
        return

    header = st.columns([3, 4, 1.5, 1.5, 1, 1.5, 2.5])
    for col, label in zip(
        header,
        ["Question", "Answer snippet", "Author", "Authored", "Status", "Review due", "Actions"],
    ):
        col.markdown(f"**{label}**")

    today_str = date.today().isoformat()
    user = st.session_state.get("pm_contrib_user") or "anonymous"
    for a in answers:
        aid = a["answer_id"]
        row = st.columns([3, 4, 1.5, 1.5, 1, 1.5, 2.5])
        row[0].write((a.get("question") or "")[:120])
        row[1].write((a.get("answer") or "")[:100])
        row[2].write(a.get("author") or "-")
        row[3].write((a.get("authored_date") or "")[:10])
        row[4].write(a.get("status", "Draft"))
        due = (a.get("review_due_date") or "")[:10]
        is_overdue = bool(due) and due < today_str
        row[5].write(f"⚠️ {due}" if is_overdue else due or "-")

        btns = row[6].columns(3)
        if btns[0].button("Edit", key=f"va_ed_{aid}"):
            st.session_state.pm_contrib_editing_answer_id = aid
            st.session_state.pm_contrib_authoring_for = None
            st.toast("Loaded in Answer Editor tab — click to continue.", icon="📝")
        if btns[1].button("Retire", key=f"va_rt_{aid}"):
            update_answer(aid, {"status": "Retired"}, edited_by=user)
            st.toast("Retired", icon="🗂️")
            st.rerun()
        if btns[2].button("Reviewed", key=f"va_rv_{aid}"):
            update_answer(
                aid,
                {
                    "review_due_date": (date.today() + timedelta(days=180)).isoformat(),
                    "reviewer_name": user,
                },
                edited_by=user,
            )
            st.toast("Marked reviewed (+180d)", icon="✅")
            st.rerun()


# --- Tab 5: SME Directory ---------------------------------------------------


def _render_sme_directory(pod: Optional[str]) -> None:
    st.subheader("SME Directory")

    topic_filter = st.text_input(
        "Topic contains",
        placeholder="optional substring",
        key="pm_contrib_sme_topic",
    )
    smes = list_smes(pod=pod, topic=topic_filter or None, active_only=True)

    if smes:
        header = st.columns([2, 2.5, 3, 2, 1.5])
        for col, label in zip(
            header, ["Name", "Email", "Topics", "Pods", "Actions"]
        ):
            col.markdown(f"**{label}**")
        user = st.session_state.get("pm_contrib_user") or "anonymous"
        for sme in smes:
            row = st.columns([2, 2.5, 3, 2, 1.5])
            row[0].write(sme.get("name") or "-")
            row[1].write(sme.get("email") or "-")
            row[2].write(", ".join(sme.get("topics") or []) or "-")
            row[3].write(", ".join(sme.get("pods") or []) or "-")
            if row[4].button("Deactivate", key=f"sme_deact_{sme['sme_id']}"):
                deactivate_sme(sme["sme_id"], user)
                st.toast("Deactivated", icon="👋")
                st.rerun()
    else:
        st.info("No active SMEs match your filters.")

    st.divider()
    st.markdown("### Add a new SME")
    with st.form("pm_contrib_sme_form", clear_on_submit=True):
        name = st.text_input("Name")
        email = st.text_input("Email (optional)")
        topics_text = st.text_input("Topics (comma-separated)")
        pods_text = st.text_input("Pods (comma-separated)")
        submitted = st.form_submit_button("Add")

    if submitted:
        if not name.strip():
            st.error("Name is required.")
            return
        topics = [t.strip() for t in topics_text.split(",") if t.strip()]
        pod_list = [p.strip() for p in pods_text.split(",") if p.strip()]
        user = st.session_state.get("pm_contrib_user") or "anonymous"
        add_sme(name.strip(), email.strip() or None, topics, pod_list, user)
        st.toast(f"Added SME: {name.strip()}", icon="✅")
        st.rerun()


# --- Tab 6: Pod Health ------------------------------------------------------


def _render_pod_health(pod: Optional[str]) -> None:
    import pandas as pd
    from storage._jsonl import read_jsonl
    from storage.query_log import QUERY_LOG_PATH

    st.subheader(f"Pod Health — {pod or 'All pods'}")

    now = datetime.now(timezone.utc)
    cutoff_7d = (now - timedelta(days=7)).isoformat()
    cutoff_30d = (now - timedelta(days=30)).isoformat()

    queries_all = read_jsonl(QUERY_LOG_PATH)
    if pod:
        queries_all = [q for q in queries_all if q.get("pod") == pod]
    queries_7d = [q for q in queries_all if q.get("timestamp", "") >= cutoff_7d]
    queries_30d = [q for q in queries_all if q.get("timestamp", "") >= cutoff_30d]

    clusters = cluster_gaps(pod=pod)
    open_gaps = [c for c in clusters if c.latest_status == "open"]
    published_answers = list_answers(pod=pod, status="Published")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries (7d)", len(queries_7d))
    high = sum(1 for q in queries_7d if not q.get("low_confidence", False))
    pct = (high / len(queries_7d) * 100.0) if queries_7d else 0.0
    c2.metric("% high-confidence (7d)", f"{pct:.0f}%")
    c3.metric("Open gaps", len(open_gaps))
    c4.metric("Verified answers (published)", len(published_answers))

    st.divider()

    if not queries_30d:
        st.info("No query data for this pod in the last 30 days.")
    else:
        df = pd.DataFrame(queries_30d)
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        df = df.dropna(subset=["date"])
        volume = df.groupby("date").size().rename("queries")
        st.markdown("### Daily query volume (30d)")
        st.line_chart(volume)

        df["low"] = df.get("low_confidence", False).fillna(False).astype(int)
        low_pct = df.groupby("date")["low"].mean().mul(100.0).rename("% low-confidence")
        st.markdown("### Daily % low-confidence (30d)")
        st.line_chart(low_pct)

    if open_gaps:
        try:
            import altair as alt

            top = sorted(open_gaps, key=lambda c: c.frequency, reverse=True)[:10]
            top_df = pd.DataFrame([
                {"question": g.representative_question[:80], "frequency": g.frequency}
                for g in top
            ])
            st.markdown("### Top 10 open gaps by frequency")
            chart = (
                alt.Chart(top_df)
                .mark_bar()
                .encode(x="frequency:Q", y=alt.Y("question:N", sort="-x"))
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            logger.exception("Failed to render gaps chart")

    citation_counts: dict[str, int] = {}
    for q in queries_30d:
        for c in q.get("citations", []) or []:
            ident = c.get("identifier") if isinstance(c, dict) else None
            if ident:
                citation_counts[ident] = citation_counts.get(ident, 0) + 1
    if citation_counts:
        try:
            import altair as alt

            top = sorted(citation_counts.items(), key=lambda x: -x[1])[:10]
            src_df = pd.DataFrame([{"source": k, "hits": v} for k, v in top])
            st.markdown("### Top 10 most-cited sources (30d)")
            chart = (
                alt.Chart(src_df)
                .mark_bar()
                .encode(x="hits:Q", y=alt.Y("source:N", sort="-x"))
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            logger.exception("Failed to render sources chart")

    contrib = [
        a for a in published_answers
        if a.get("authored_date", "") >= cutoff_30d
    ]
    if contrib:
        by_author: dict[str, int] = {}
        for a in contrib:
            k = a.get("author") or "unknown"
            by_author[k] = by_author.get(k, 0) + 1
        leader = pd.DataFrame(
            sorted(
                ({"author": k, "published": v} for k, v in by_author.items()),
                key=lambda d: -d["published"],
            )
        )
        st.markdown("### Contribution leaderboard (30d)")
        st.dataframe(leader, use_container_width=True, hide_index=True)


# --- Main -------------------------------------------------------------------


def _prewarm_embeddings() -> None:
    """Trigger the embedding factory once per session so the first upload
    doesn't freeze on a 130MB first-run HuggingFace download."""
    if st.session_state.get("pm_embed_prewarmed"):
        return
    try:
        with st.spinner(
            "Loading embedding model (first run may download ~130MB)..."
        ):
            from query.embed_factory import get_embed_model, get_model_dim

            get_embed_model()
            get_model_dim()  # second call is cached after this
        st.session_state.pm_embed_prewarmed = True
    except Exception as exc:
        # Don't block rendering; downstream try/except paths will surface this
        # again with more actionable context.
        logger.warning("Embedding prewarm failed: %s", exc)


def main() -> None:
    _init_state()
    _prewarm_embeddings()

    st.title("PM Knowledge Base — Contributor")
    st.text_input(
        "Your name",
        key="pm_contrib_user",
        placeholder="e.g. Ravi K. (free text, no auth)",
    )

    pods, total_chunks = _discover_pods()
    selected_pod = _render_sidebar(pods, total_chunks)

    if _CONFIG_ERROR:
        st.error(f"Configuration error — {_CONFIG_ERROR}")

    tabs = st.tabs(
        [
            "Gap Queue",
            "Answer Editor",
            "Sources",
            "Verified Answers",
            "SME Directory",
            "Pod Health",
        ]
    )
    with tabs[0]:
        _render_gap_queue(selected_pod)
    with tabs[1]:
        _render_answer_editor(selected_pod, pods)
    with tabs[2]:
        _render_sources(selected_pod)
    with tabs[3]:
        _render_verified_answers(selected_pod)
    with tabs[4]:
        _render_sme_directory(selected_pod)
    with tabs[5]:
        _render_pod_health(selected_pod)


# Guarded so `from app.contributor import _process_uploads` doesn't run the
# whole Streamlit script (which would open forms in bare mode and pollute
# internal state for subsequent AppTest runs). Both `streamlit run` and
# AppTest execute scripts with __name__ == "__main__".
if __name__ == "__main__":
    main()
