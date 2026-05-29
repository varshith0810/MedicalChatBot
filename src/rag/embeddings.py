"""Embedding model wrapper used by ingestion and retrieval."""

from __future__ import annotations

from functools import cached_property
from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_EMBEDDING_MODEL


class EmbeddingModel:
    """Lazy SentenceTransformers encoder."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
