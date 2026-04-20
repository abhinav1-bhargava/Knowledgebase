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
7. Jira ingestion is a stub in this prototype; see ingestion/jira_ingest.py for implementation plan.
8. Launch consumer: streamlit run app/consumer.py
9. Launch contributor: streamlit run app/contributor.py --server.port 8502

## Status
Prototype — pilot pod only. Not for production use.
Pilot corpus: PDFs + Markdown. Jira ingestion pending IT access (tracker: <TBD>).

## Known limitations
- Jira ingestion is a stub pending API access — SIT Atlassian returns 404 on tickets the browser can see; IT ticket raised. See ingestion/jira_ingest.py for the full implementation plan.
- OpenAI embeddings key (OPENAI_API_KEY in .env) is still required for both ingestion and query. With a placeholder key the consumer UI renders but queries will 401.
- Single-pod focus in this pilot. Cross-pod discovery and multi-tenant metadata filtering are out of scope until the first pod is validated.
EOF
# 
Knowledgebase
Solving the problem of Knowledge Base for Product managers
