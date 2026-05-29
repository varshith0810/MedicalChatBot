"""Chunk deduplication helpers."""

from __future__ import annotations

import hashlib
import re

from src.ingestion.chunking import TextChunk

_NORMALIZE_RE = re.compile(r"\s+")


def content_hash(text: str) -> str:
    """Return a SHA-256 hash for normalized chunk text."""

    normalized = _NORMALIZE_RE.sub(" ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_chunks(
    chunks: list[TextChunk], existing_hashes: set[str] | None = None
) -> tuple[list[tuple[TextChunk, str]], int]:
    """Remove duplicate chunks within the batch and against persisted hashes."""

    seen = set(existing_hashes or set())
    unique: list[tuple[TextChunk, str]] = []
    duplicate_count = 0
    for chunk in chunks:
        digest = content_hash(chunk.text)
        if digest in seen:
            duplicate_count += 1
            continue
        seen.add(digest)
        unique.append((chunk, digest))
    return unique, duplicate_count
