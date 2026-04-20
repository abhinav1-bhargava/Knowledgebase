"""Query layer.

Builds the retrieval + generation pipeline over the ChromaDB index, runs
user queries, attaches citations, and computes a confidence signal used by
the consumer UI to either return an answer or surface a knowledge gap.

Submodules:
    engine   - Retrieval + LLM orchestration
    prompts  - System / user prompt templates with citation enforcement
"""
