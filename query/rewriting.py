"""Query-rewriting helper.

Generates up to 3 rephrased variants of a user question so the retrieval
layer can cast a wider net (same intent, different vocabulary). Used by
`query.engine.query_with_rewriting`.

Strategy:
    - Ask the configured LLM to produce three rephrasings.
    - Parse one-per-line, skip blanks.
    - Return [original, variant_1, variant_2] — keep the original query
      in the mix so the rewrite path never loses ground relative to the
      naive path even if the LLM's rephrasings drift off-topic.

Defensive: any LLM failure falls back to returning just [original] so a
transient outage can't take down the whole retrieval call.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = (
    "Rephrase this question in 3 different ways to improve retrieval "
    "from a knowledge base. Use different vocabulary and phrasing while "
    "keeping the same intent.\n\n"
    "Original: {question}\n\n"
    "Return ONLY the 3 rephrased versions, one per line, no numbering "
    "or extra text."
)


def rewrite_query(question: str) -> list[str]:
    """Return [original, up-to-two-LLM-variants]. Never raises."""
    from query.engine import _get_llm  # lazy to avoid import cycle

    question = (question or "").strip()
    if not question:
        return [question]

    try:
        llm = _get_llm()
        response = llm.complete(_REWRITE_PROMPT.format(question=question))
        raw = (response.text or "").strip() if response is not None else ""
    except Exception as exc:
        logger.warning("Query rewrite failed — falling back to original only: %s", exc)
        return [question]

    variants: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        # Strip common numbering patterns the LLM sometimes emits even when
        # asked not to ("1. ...", "1) ...", "- ...").
        for prefix in ("-", "*", "•"):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
        if s and s[0].isdigit():
            # e.g. "1. Foo?" / "1) Foo?"
            head, sep, rest = s.partition(" ")
            if sep and head.rstrip(".)").isdigit():
                s = rest.strip()
        if s and s != question:
            variants.append(s)

    return [question] + variants[:2]
