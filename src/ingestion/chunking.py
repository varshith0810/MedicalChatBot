"""Page-aware text chunking."""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.cleaners import clean_text
from src.ingestion.loaders import LoadedPage


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    page_start: int
    page_end: int
    text: str


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        split_at = end
        if end < len(text):
            paragraph = text.rfind("\n\n", start, end)
            sentence = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            whitespace = text.rfind(" ", start, end)
            split_at = max(paragraph, sentence, whitespace)
            if split_at <= start:
                split_at = end
            elif split_at == sentence:
                split_at += 1
        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        if split_at >= len(text):
            break
        start = max(split_at - chunk_overlap, start + 1)
    return chunks


def chunk_pages(
    pages: list[LoadedPage], *, chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:
    """Clean and split each page into chunks with source page metadata."""

    output: list[TextChunk] = []
    for page in pages:
        cleaned = clean_text(page.text)
        for text in _split_text(cleaned, chunk_size, chunk_overlap):
            output.append(
                TextChunk(
                    chunk_index=len(output),
                    page_start=page.page_number,
                    page_end=page.page_number,
                    text=text,
                )
            )
    return output
