"""Prompt templates for the query engine.

Centralizes the system prompt (role, tone, citation requirements, refusal
policy when context is insufficient) and the user prompt builder that
formats retrieved context chunks with explicit source identifiers so the
LLM has exact handles to cite in brackets.

Keeping prompts here makes them easy to iterate on without touching the
engine plumbing.
"""

from __future__ import annotations

SYSTEM_PROMPT: str = """You are an onboarding assistant for product managers joining or moving between pods in an Indian telecom organization. You help them understand what already exists in their pod — applications, architecture, features shipped, metrics, problems tried — using only the retrieved context below.

Rules:
1. Answer using ONLY the retrieved context. Do not use outside knowledge.
2. Always cite sources. For document sources, cite by filename. For Jira tickets, cite by ticket key (e.g., PROJ-1234). Place citations inline in square brackets after the relevant claim.
3. If retrieval is weak, explicitly say: "I don't have strong sources for this — consider asking an SME for [topic area]." Do not guess or fabricate.
4. Prioritize sources by type:
   - "What does X do?" / "How does X work?" → prefer docs (Confluence/PDF/DOCX)
   - "What was tried?" / "Why did we not do X?" → prefer Jira sources
   - "What's our policy/metric?" → prefer verified_answer > docs > other
5. When a verified_answer is the primary basis, prepend: "Based on a verified answer by [author] ([date]):"
6. Be concise. 3-5 sentences unless complexity demands more.
7. Never fabricate ticket keys, filenames, dates, author names, metric values, or URLs.
8. End every answer with:

Suggested follow-ups:
- <question 1>
- <question 2>
- <question 3>
"""


def _identifier_for_metadata(metadata: dict) -> str:
    """Pick the citation handle the LLM should use for a chunk."""
    source_type = metadata.get("source_type", "")
    if source_type == "jira":
        return metadata.get("issue_key") or metadata.get("filename") or "UNKNOWN-JIRA"
    if source_type == "verified":
        return (
            metadata.get("verified_id")
            or metadata.get("filename")
            or "verified_answer"
        )
    return (
        metadata.get("filename")
        or metadata.get("relative_path")
        or "UNKNOWN"
    )


def _format_meta_bits(metadata: dict) -> str:
    """Render a short metadata tail for a source header (type, pod, etc.)."""
    source_type = metadata.get("source_type") or "unknown"
    bits = [f"type={source_type}"]
    pod = metadata.get("pod")
    if pod:
        bits.append(f"pod={pod}")
    if source_type == "verified":
        if metadata.get("author"):
            bits.append(f"author={metadata['author']}")
        date = metadata.get("created_at") or metadata.get("date")
        if date:
            bits.append(f"date={date}")
    if source_type == "jira" and metadata.get("status"):
        bits.append(f"status={metadata['status']}")
    if metadata.get("page_number") is not None:
        bits.append(f"page={metadata['page_number']}")
    return ", ".join(bits)


def build_user_prompt(question: str, context_chunks: list[dict]) -> str:
    """Render the user prompt with retrieved context injected.

    Each chunk becomes a labelled block of the form:

        Source [<identifier>] (<meta>):
        <chunk text>

    where <identifier> is the handle the LLM should use for citations
    (filename for docs, issue key for jira, verified_id for verified
    answers). Blocks are separated by a divider so long chunks don't blur
    together.
    """
    blocks: list[str] = []
    for chunk in context_chunks:
        metadata = chunk.get("metadata", {}) or {}
        identifier = _identifier_for_metadata(metadata)
        meta_bits = _format_meta_bits(metadata)
        text = (chunk.get("text") or "").strip()
        blocks.append(f"Source [{identifier}] ({meta_bits}):\n{text}")
    context_text = "\n\n---\n\n".join(blocks)
    return (
        "Retrieved context:\n\n"
        f"{context_text}\n\n"
        f"Question: {question}"
    )
