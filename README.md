cat > README.md << 'EOF'
# Knowledgebase — PM Onboarding Assistant

RAG-based knowledge assistant helping product managers onboard into new pods. Indexes Confluence docs, Jira tickets, and PDFs; serves queries via a chat interface with gap detection and contributor curation.

## Problem
New or moved PMs struggle to understand what already exists in their pod: applications, architecture, metrics, features shipped, problems tried. This tool surfaces existing knowledge via natural language queries and flags gaps for contributors to fill.

## Architecture
- Ingestion: Confluence HTML/PDF, Jira tickets (via API), verified answers
- Retrieval: ChromaDB vector store with hybrid metadata filtering
- Generation: LLM-backed answers with mandatory citations
- Interfaces: Consumer chat UI (Streamlit) and Contributor curation UI

## Setup
1. Clone repo and cd into it
2. python -m venv venv && source venv/bin/activate
3. pip install -r requirements.txt
4. Copy .env.example to .env, fill credentials
5. Place source docs in ./docs/
6. Run: python -m ingestion.docs_ingest --pod <pod_name>
7. Run: python -m ingestion.jira_ingest --project <PROJECT_KEY>
8. Launch consumer: streamlit run app/consumer.py
9. Launch contributor: streamlit run app/contributor.py --server.port 8502

## Status
Prototype — pilot pod only. Not for production use.
EOF
# 
Knowledgebase
Solving the problem of Knowledge Base for Product managers
