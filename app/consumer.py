"""Consumer chat UI (Streamlit).

A chat interface for product managers to ask natural-language questions
about their pod. Calls query.engine.answer_query, renders the answer with
inline citations, and shows a "this looks like a gap" affordance when
confidence is below threshold so the user can flag it for contributors.

Run:
    streamlit run app/consumer.py
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Render the Streamlit chat UI."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
