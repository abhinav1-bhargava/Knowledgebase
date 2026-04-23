"""LLM-based question classifier for agentic RAG routing.

classify_question() asks the configured LLM to place a user question
into one of four canonical categories, then the engine-level
query_agentic() dispatches to the retrieval strategy best suited to
each:

    metrics       → numbers/KPIs/targets. Keyword-heavy; hybrid wins.
    procedural    → how-to/steps/workflow. Keyword-heavy too.
    debugging     → why-did-X-fail. Semantic nuance across many
                    near-miss chunks; reranker picks the right one.
    conceptual    → what-is-X/definitions. Vanilla semantic match.

The classifier is deliberately defensive — any LLM failure or
unparseable output falls back to `conceptual`, which routes to the
naive vector path. Worst case, agentic degrades to naive; it never
breaks.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SUPPORTED_CATEGORIES: tuple[str, ...] = (
    "metrics",
    "conceptual",
    "procedural",
    "debugging",
)
DEFAULT_CATEGORY = "conceptual"

_CLASSIFIER_PROMPT = (
    "Classify this question into ONE category:\n\n"
    "- metrics: Asks for numbers, KPIs, targets, percentages "
    "(e.g., \"What's the conversion rate?\")\n"
    "- conceptual: Asks what something is, how it works, definitions "
    "(e.g., \"What is the HDO fleet?\")\n"
    "- procedural: Asks how to do something, steps, workflows "
    "(e.g., \"How do I file a ticket?\")\n"
    "- debugging: Asks why something failed, error troubleshooting "
    "(e.g., \"Why did the deployment fail?\")\n\n"
    "Question: {question}\n\n"
    "Return ONLY the category name (metrics/conceptual/procedural/"
    "debugging), no explanation."
)


def classify_question(question: str) -> str:
    """Return one of SUPPORTED_CATEGORIES. Never raises.

    If the LLM call fails or returns something unparseable, falls back
    to `conceptual` (the safest default — routes to naive vector
    retrieval, which works reasonably for most query shapes).
    """
    question = (question or "").strip()
    if not question:
        return DEFAULT_CATEGORY

    try:
        from query.engine import _get_llm  # lazy to avoid import cycle

        llm = _get_llm()
        response = llm.complete(_CLASSIFIER_PROMPT.format(question=question))
        raw = (response.text or "").strip().lower() if response is not None else ""
    except Exception as exc:
        logger.warning(
            "Question classification failed, defaulting to %s: %s",
            DEFAULT_CATEGORY, exc,
        )
        return DEFAULT_CATEGORY

    # Robust match: an LLM asked to "return ONLY the category" sometimes
    # still surrounds the word with quotes/punctuation/yapping. Pick the
    # first supported category the response mentions.
    for cat in SUPPORTED_CATEGORIES:
        if cat in raw:
            logger.info("Classified question as %s (raw=%r)", cat, raw[:60])
            return cat

    logger.warning(
        "Could not parse category from LLM output %r — defaulting to %s",
        raw[:60], DEFAULT_CATEGORY,
    )
    return DEFAULT_CATEGORY
