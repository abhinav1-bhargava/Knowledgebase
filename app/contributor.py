"""Contributor curation UI (Streamlit).

For pod contributors to: review and resolve open knowledge gaps, write and
manage verified answers, browse recent queries by pod, and trigger
re-ingestion runs. Reads/writes the JSONL stores under storage/.

Run:
    streamlit run app/contributor.py --server.port 8502
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Render the Streamlit contributor UI."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
