"""Prompt templates for the query engine.

Centralizes the system prompt (role, tone, citation requirements, refusal
policy when context is insufficient) and the user prompt template that
injects retrieved context chunks. Keeping prompts here makes them easy to
iterate on without touching engine plumbing.
"""

from __future__ import annotations

SYSTEM_PROMPT: str = ""  # populated in Phase B

USER_PROMPT_TEMPLATE: str = ""  # populated in Phase B


def build_user_prompt(question: str, context_chunks: list[str]) -> str:
    """Render the user prompt with retrieved context injected."""
    raise NotImplementedError
