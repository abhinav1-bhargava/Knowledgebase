"""Centralized configuration for the PM Onboarding Assistant.

Loads environment variables from .env, validates that required secrets/URLs
are present, and exposes typed module-level constants used across ingestion,
retrieval, storage, and UI layers.

Required environment variables (validated at import time):
    OPENAI_API_KEY, JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY

Optional environment variables (with defaults):
    EMBEDDING_MODEL  (default: text-embedding-3-small)
    LLM_MODEL        (default: gpt-4o-mini)
    CHROMA_PATH      (default: ./chroma_db)
    CHUNK_SIZE       (default: 512)
    CHUNK_OVERLAP    (default: 50)
    RETRIEVAL_TOP_K  (default: 8)
    CONFIDENCE_THRESHOLD (default: 0.7)

Usage:
    from config import OPENAI_API_KEY, CHROMA_PATH, CHUNK_SIZE
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project root (the directory containing this file).
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


def _require(name: str) -> str:
    """Return the value of a required env var, or raise ConfigError."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigError(
            f"Required environment variable {name!r} is missing or empty. "
            f"Add it to .env (see .env.example)."
        )
    return value.strip()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {name!r} must be an integer, got {raw!r}."
        ) from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {name!r} must be a float, got {raw!r}."
        ) from exc


# --- Required secrets / endpoints -------------------------------------------
OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
JIRA_URL: str = _require("JIRA_URL")
JIRA_EMAIL: str = _require("JIRA_EMAIL")
JIRA_API_TOKEN: str = _require("JIRA_API_TOKEN")
JIRA_PROJECT_KEY: str = _require("JIRA_PROJECT_KEY")

# --- Model selection --------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Vector store -----------------------------------------------------------
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")

# --- Chunking & retrieval ---------------------------------------------------
CHUNK_SIZE: int = _int_env("CHUNK_SIZE", 512)
CHUNK_OVERLAP: int = _int_env("CHUNK_OVERLAP", 50)
RETRIEVAL_TOP_K: int = _int_env("RETRIEVAL_TOP_K", 8)
CONFIDENCE_THRESHOLD: float = _float_env("CONFIDENCE_THRESHOLD", 0.7)


__all__ = [
    "ConfigError",
    "OPENAI_API_KEY",
    "JIRA_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_PROJECT_KEY",
    "EMBEDDING_MODEL",
    "LLM_MODEL",
    "CHROMA_PATH",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_TOP_K",
    "CONFIDENCE_THRESHOLD",
]
