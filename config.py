"""Centralized configuration for the PM Onboarding Assistant.

Loads environment variables from .env, validates that required secrets/URLs
are present given the selected embedding provider, and exposes typed
module-level constants used across ingestion, retrieval, storage, and UI.

Required environment variables:
    JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY — always.
    OPENAI_API_KEY                      — required only when
                                           EMBEDDING_PROVIDER=openai
                                           (still loaded when empty so
                                           the LLM path can fall back
                                           to a clear runtime error).
    AIRTEL_EMBEDDING_BASE_URL,
    AIRTEL_EMBEDDING_API_KEY            — required only when
                                           EMBEDDING_PROVIDER=airtel_bge.

Embedding provider selection:
    EMBEDDING_PROVIDER  — one of {openai, airtel_bge, local_hf};
                          defaults to local_hf (runs entirely on-device
                          after a one-time ~130MB model download).

Optional environment variables (with defaults):
    EMBEDDING_MODEL       — API model name for openai/airtel_bge paths.
                            Defaults per provider: text-embedding-3-small
                            for openai, bge-m3 for airtel_bge.
    LOCAL_EMBEDDING_MODEL — HuggingFace model id for local_hf. Default:
                            BAAI/bge-small-en-v1.5 (384-dim, ~130MB).
    LLM_MODEL             — default: gpt-4o-mini
    CHROMA_PATH           — default: ./chroma_db
    CHUNK_SIZE            — default: 512
    CHUNK_OVERLAP         — default: 50
    RETRIEVAL_TOP_K       — default: 8
    CONFIDENCE_THRESHOLD  — default: 0.7

Usage:
    from config import OPENAI_API_KEY, CHROMA_PATH, EMBEDDING_PROVIDER
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


def _require(name: str) -> str:
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


_SUPPORTED_PROVIDERS = {"openai", "airtel_bge", "local_hf"}

# --- Embedding provider (drives which other vars are required) -------------
EMBEDDING_PROVIDER: str = (os.getenv("EMBEDDING_PROVIDER") or "local_hf").strip().lower()
if EMBEDDING_PROVIDER not in _SUPPORTED_PROVIDERS:
    raise ConfigError(
        f"Invalid EMBEDDING_PROVIDER {EMBEDDING_PROVIDER!r}. "
        f"Must be one of: {sorted(_SUPPORTED_PROVIDERS)}."
    )

# --- Jira creds (always required) ------------------------------------------
JIRA_URL: str = _require("JIRA_URL")
JIRA_EMAIL: str = _require("JIRA_EMAIL")
JIRA_API_TOKEN: str = _require("JIRA_API_TOKEN")
JIRA_PROJECT_KEY: str = _require("JIRA_PROJECT_KEY")

# --- Keys / endpoints (loaded unconditionally; validated per-provider) -----
OPENAI_API_KEY: str = (os.getenv("OPENAI_API_KEY") or "").strip()
AIRTEL_EMBEDDING_BASE_URL: str = (os.getenv("AIRTEL_EMBEDDING_BASE_URL") or "").strip()
AIRTEL_EMBEDDING_API_KEY: str = (os.getenv("AIRTEL_EMBEDDING_API_KEY") or "").strip()

# --- Embedding model names --------------------------------------------------
_DEFAULT_EMBEDDING_MODEL = {
    "openai": "text-embedding-3-small",
    "airtel_bge": "bge-m3",
    "local_hf": "text-embedding-3-small",  # unused for local_hf; kept for symmetry
}[EMBEDDING_PROVIDER]
EMBEDDING_MODEL: str = (os.getenv("EMBEDDING_MODEL") or _DEFAULT_EMBEDDING_MODEL).strip()
LOCAL_EMBEDDING_MODEL: str = (
    os.getenv("LOCAL_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
).strip()

# --- Provider-specific validation ------------------------------------------
if EMBEDDING_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise ConfigError(
            "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY in .env."
        )
elif EMBEDDING_PROVIDER == "airtel_bge":
    if not AIRTEL_EMBEDDING_BASE_URL:
        raise ConfigError(
            "EMBEDDING_PROVIDER=airtel_bge requires AIRTEL_EMBEDDING_BASE_URL in .env."
        )
    if not AIRTEL_EMBEDDING_API_KEY:
        raise ConfigError(
            "EMBEDDING_PROVIDER=airtel_bge requires AIRTEL_EMBEDDING_API_KEY in .env "
            "(use the literal string 'EMPTY' for the SIT endpoint)."
        )
# local_hf: no keys required; the model downloads on first use.

# --- LLM / vector store / chunking ----------------------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
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
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "LOCAL_EMBEDDING_MODEL",
    "AIRTEL_EMBEDDING_BASE_URL",
    "AIRTEL_EMBEDDING_API_KEY",
    "LLM_MODEL",
    "CHROMA_PATH",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_TOP_K",
    "CONFIDENCE_THRESHOLD",
]
