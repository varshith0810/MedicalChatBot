"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DATA_DIR = "data/books"
DEFAULT_SQLITE_DB_PATH = "data/medical_chatbot.sqlite3"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "medical_books"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_RETRIEVAL_K = 3
DEFAULT_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LLM_MODEL = "writer/palmyra-med-70b-32k"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for ingestion and retrieval components."""

    data_dir: Path = Path(DEFAULT_DATA_DIR)
    sqlite_db_path: Path = Path(DEFAULT_SQLITE_DB_PATH)
    qdrant_url: str = DEFAULT_QDRANT_URL
    qdrant_collection: str = DEFAULT_QDRANT_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def load_settings(env_file: str | os.PathLike[str] | None = ".env") -> Settings:
    """Load settings from .env/environment variables with safe defaults."""

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR)),
        sqlite_db_path=Path(os.getenv("SQLITE_DB_PATH", DEFAULT_SQLITE_DB_PATH)),
        qdrant_url=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_dim=_int_from_env("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM),
        chunk_size=_int_from_env("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_int_from_env("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        retrieval_k=_int_from_env("RETRIEVAL_K", DEFAULT_RETRIEVAL_K),
        llm_base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
    )
