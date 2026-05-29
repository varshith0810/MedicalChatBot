#!/usr/bin/env python3
"""Ingest PDFs from DATA_DIR into SQLite and Qdrant."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.ingestion.pipeline import IngestionPipeline


def main() -> None:
    settings = load_settings()
    stats = IngestionPipeline(settings).run()
    print("Ingestion complete")
    print(f"Books discovered: {stats.books_seen}")
    print(f"Books ingested: {stats.books_ingested}")
    print(f"Pages processed: {stats.pages_seen}")
    print(f"Chunks created: {stats.chunks_created}")
    print(f"Chunks deduplicated: {stats.chunks_deduplicated}")
    print(f"Chunks stored in SQLite: {stats.chunks_stored}")
    print(f"Vectors stored in Qdrant: {stats.vectors_stored}")


if __name__ == "__main__":
    main()
