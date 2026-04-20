"""Storage layer.

Lightweight JSONL-backed stores for state that lives outside the vector
index: detected knowledge gaps, contributor-curated verified answers, and
the query log used for analytics and gap detection.

Submodules:
    gaps              - Detected knowledge gaps awaiting contributor input
    verified_answers  - Curated authoritative Q/A pairs
    query_log         - Append-only log of user queries and outcomes
"""
