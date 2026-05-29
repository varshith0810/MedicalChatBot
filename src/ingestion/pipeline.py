"""End-to-end book ingestion pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings
from src.ingestion.chunking import chunk_pages
from src.ingestion.dedup import deduplicate_chunks
from src.ingestion.loaders import load_books
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import QdrantVectorStore, VectorRecord
from src.storage.db import connect, initialize_database
from src.storage.repositories import BookRepository, ChunkRepository


@dataclass
class IngestionStats:
    books_seen: int = 0
    books_ingested: int = 0
    pages_seen: int = 0
    chunks_created: int = 0
    chunks_deduplicated: int = 0
    chunks_stored: int = 0
    vectors_stored: int = 0


def vector_id_for_hash(digest: str) -> str:
    """Create a deterministic Qdrant-compatible UUID for a chunk hash."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"medical-chatbot/chunk/{digest}"))


class IngestionPipeline:
    """Coordinates PDF loading, SQLite writes, embeddings, and Qdrant upserts."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_model: EmbeddingModel | None = None,
        vector_store: QdrantVectorStore | None = None,
    ):
        self.settings = settings
        self.embedding_model = embedding_model or EmbeddingModel(settings.embedding_model)
        self.vector_store = vector_store or QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            vector_size=settings.embedding_dim,
        )

    def run(self) -> IngestionStats:
        stats = IngestionStats()
        books = load_books(self.settings.data_dir)
        stats.books_seen = len(books)

        with connect(self.settings.sqlite_db_path) as connection:
            initialize_database(connection)
            book_repo = BookRepository(connection)
            chunk_repo = ChunkRepository(connection)
            existing_hashes = chunk_repo.content_hashes()

            for book in books:
                stats.pages_seen += book.page_count
                source_path = str(Path(book.source_path))
                book_id = book_repo.upsert_book(
                    source_path=source_path,
                    title=book.title,
                    checksum=book.checksum,
                    page_count=book.page_count,
                )
                stats.books_ingested += 1

                chunks = chunk_pages(
                    book.pages,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
                stats.chunks_created += len(chunks)
                unique_chunks, duplicates = deduplicate_chunks(chunks, existing_hashes)
                stats.chunks_deduplicated += duplicates

                vector_records: list[VectorRecord] = []
                texts_to_embed: list[str] = []
                stored_payloads: list[tuple[int, str, str, int, int]] = []

                for chunk, digest in unique_chunks:
                    vector_id = vector_id_for_hash(digest)
                    stored = chunk_repo.insert_chunk(
                        book_id=book_id,
                        chunk_index=chunk.chunk_index,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        text=chunk.text,
                        content_hash=digest,
                        vector_id=vector_id,
                    )
                    if stored is None:
                        stats.chunks_deduplicated += 1
                        existing_hashes.add(digest)
                        continue
                    stats.chunks_stored += 1
                    existing_hashes.add(digest)
                    texts_to_embed.append(chunk.text)
                    stored_payloads.append(
                        (stored.id, vector_id, source_path, chunk.page_start, chunk.page_end)
                    )

                vectors = self.embedding_model.embed_documents(texts_to_embed)
                for vector, (chunk_id, vector_id, path, page_start, page_end), text in zip(
                    vectors, stored_payloads, texts_to_embed
                ):
                    vector_records.append(
                        VectorRecord(
                            id=vector_id,
                            vector=vector,
                            payload={
                                "chunk_id": chunk_id,
                                "book_id": book_id,
                                "source_path": path,
                                "page_start": page_start,
                                "page_end": page_end,
                                "text": text,
                            },
                        )
                    )

                self.vector_store.upsert(vector_records)
                stats.vectors_stored += len(vector_records)

            connection.commit()

        return stats
