"""Repository layer for persisted books and chunks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredChunk:
    id: int
    vector_id: str
    content_hash: str


class BookRepository:
    """Persistence operations for source books."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def upsert_book(
        self, *, source_path: str, title: str | None, checksum: str, page_count: int
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO books (source_path, title, checksum, page_count, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source_path) DO UPDATE SET
                title = excluded.title,
                checksum = excluded.checksum,
                page_count = excluded.page_count,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (source_path, title, checksum, page_count),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to upsert book {source_path}")
        return int(row["id"])


class ChunkRepository:
    """Persistence operations for text chunks."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def content_hashes(self) -> set[str]:
        rows = self.connection.execute("SELECT content_hash FROM chunks").fetchall()
        return {str(row["content_hash"]) for row in rows}

    def insert_chunk(
        self,
        *,
        book_id: int,
        chunk_index: int,
        page_start: int,
        page_end: int,
        text: str,
        content_hash: str,
        vector_id: str,
    ) -> StoredChunk | None:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO chunks (
                book_id, chunk_index, page_start, page_end, text, content_hash, vector_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (book_id, chunk_index, page_start, page_end, text, content_hash, vector_id),
        )
        if cursor.rowcount == 0:
            return None
        return StoredChunk(
            id=int(cursor.lastrowid), vector_id=vector_id, content_hash=content_hash
        )
