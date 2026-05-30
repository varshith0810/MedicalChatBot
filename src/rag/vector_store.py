"""Qdrant vector-store integration."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: list[float]
    payload: dict


class QdrantVectorStore:
    """Small Qdrant adapter for creating collections and upserting chunks."""

    def __init__(
        self, *, url: str, collection_name: str, vector_size: int, api_key: str | None = None
    ):
        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.collection_name = collection_name
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self.ensure_collection()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload,
                )
                for record in records
            ],
        )
