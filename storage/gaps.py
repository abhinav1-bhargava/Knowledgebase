"""Knowledge gap store (JSONL) + clustering.

A "gap" is a question the assistant could not answer with confidence.
`log_gap` records each gap at query time (append-only). `cluster_gaps`
groups semantically similar questions into GapCluster records so the
contributor UI shows one row per *question*, not one per ask.

Clustering:
    - Primary: cosine similarity on OpenAI embeddings (threshold 0.85)
    - Fallback: difflib.SequenceMatcher ratio (threshold 0.80) when the
      OPENAI_API_KEY is missing or still a placeholder.
    - Greedy: questions are sorted by timestamp ASC; each new question
      joins the first cluster where it matches any existing member,
      else it starts a new cluster. This gives deterministic cluster
      assignment across runs for the same input.

Cluster id = SHA-1 of the earliest-timestamp question in the cluster,
so downstream status updates (claim_gap / resolve_gap) can reference a
stable id even as new matches accumulate.

Status model (gap_status.jsonl, append-only, latest-entry-per-cluster
wins on read):
    open      - default, no status record yet
    claimed   - a contributor picked it up
    resolved  - a verified answer has been published for it
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from storage._jsonl import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

GAPS_LOG_PATH = Path("gaps.jsonl")
GAP_STATUS_PATH = Path("gap_status.jsonl")

EMBED_SIM_THRESHOLD = 0.85
FUZZY_SIM_THRESHOLD = 0.80


@dataclass
class GapCluster:
    """One semantically-grouped gap row displayed in the Gap Queue."""

    cluster_id: str
    representative_question: str
    all_questions: list[str]
    frequency: int
    last_asked: str
    latest_status: str


# --- Gap log write path -----------------------------------------------------


def log_gap(
    question: str,
    pod: str | None,
    confidence: float,
    retrieval_scores: list[float],
) -> None:
    """Append an open gap record to gaps.jsonl. Never raises."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "pod": pod,
        "confidence": confidence,
        "retrieval_scores": list(retrieval_scores),
        "status": "open",
    }
    append_jsonl(GAPS_LOG_PATH, record)


# --- Clustering -------------------------------------------------------------


def _cluster_hash(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def _has_live_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "") or ""
    return bool(key) and not key.startswith("sk-placeholder")


def _fuzzy_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _compute_embeddings(questions: list[str]) -> dict[str, list[float]]:
    """Batch-embed unique questions. Returns empty on any failure (caller falls back to fuzzy)."""
    try:
        from config import EMBEDDING_MODEL, OPENAI_API_KEY
        from llama_index.embeddings.openai import OpenAIEmbedding

        model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        vectors = model.get_text_embedding_batch(questions)
        return dict(zip(questions, vectors))
    except Exception as exc:
        logger.warning(
            "Embedding computation failed, falling back to fuzzy similarity: %s", exc,
        )
        return {}


def cluster_gaps(pod: str | None = None) -> list[GapCluster]:
    """Group gap records into semantically similar clusters."""
    entries = read_jsonl(GAPS_LOG_PATH)
    if pod is not None:
        entries = [e for e in entries if e.get("pod") == pod]
    entries = [e for e in entries if e.get("question")]
    entries.sort(key=lambda e: e.get("timestamp", ""))
    if not entries:
        return []

    unique_qs = sorted({e["question"] for e in entries})
    emb_map: dict[str, list[float]] = {}
    if _has_live_openai_key():
        emb_map = _compute_embeddings(unique_qs)
    use_embeddings = bool(emb_map)
    threshold = EMBED_SIM_THRESHOLD if use_embeddings else FUZZY_SIM_THRESHOLD

    clusters: list[list[dict]] = []
    for entry in entries:
        q = entry["question"]
        placed = False
        for cluster in clusters:
            for existing in cluster:
                eq = existing["question"]
                if use_embeddings and q in emb_map and eq in emb_map:
                    sim = _cosine(emb_map[q], emb_map[eq])
                else:
                    sim = _fuzzy_similarity(q, eq)
                if sim >= threshold:
                    cluster.append(entry)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([entry])

    statuses = _read_all_latest_statuses()
    results: list[GapCluster] = []
    for cluster in clusters:
        earliest = cluster[0]  # entries sorted by timestamp ASC
        cluster_id = _cluster_hash(earliest["question"])
        rep = max(cluster, key=lambda e: len(e["question"]))["question"]
        last_asked = max(e.get("timestamp", "") for e in cluster)
        latest_status = statuses.get(cluster_id, "open")
        results.append(
            GapCluster(
                cluster_id=cluster_id,
                representative_question=rep,
                all_questions=[e["question"] for e in cluster],
                frequency=len(cluster),
                last_asked=last_asked,
                latest_status=latest_status,
            )
        )
    return results


# --- Status write path ------------------------------------------------------


def _read_all_latest_statuses() -> dict[str, str]:
    records = read_jsonl(GAP_STATUS_PATH)
    latest: dict[str, str] = {}
    for r in records:
        cid = r.get("cluster_id")
        if cid:
            latest[cid] = r.get("status", "open")
    return latest


def claim_gap(cluster_id: str, user: str) -> None:
    """Append a claimed-status record for a cluster."""
    append_jsonl(GAP_STATUS_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cluster_id": cluster_id,
        "status": "claimed",
        "claimed_by": user,
    })


def resolve_gap(cluster_id: str, answer_id: str) -> None:
    """Append a resolved-status record tying a cluster to its verified answer."""
    append_jsonl(GAP_STATUS_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cluster_id": cluster_id,
        "status": "resolved",
        "answer_id": answer_id,
    })


def get_latest_status(cluster_id: str) -> str:
    """Return the latest status for a cluster, defaulting to 'open'."""
    return _read_all_latest_statuses().get(cluster_id, "open")
