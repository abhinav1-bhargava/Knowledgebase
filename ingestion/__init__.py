"""Ingestion layer.

Loads source content (Confluence HTML exports, PDFs, DOCX files, Jira tickets,
and curated verified answers) into the ChromaDB vector store with consistent
metadata so the retrieval layer can filter by pod, source type, and recency.

Submodules:
    docs_ingest      - Local file ingestion (HTML / PDF / DOCX)
    jira_ingest      - Jira API ingestion
    verified_ingest  - Contributor-curated verified answers
"""
