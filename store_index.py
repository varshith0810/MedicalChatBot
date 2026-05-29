"""Backward-compatible ingestion entrypoint.

The project now ingests medical-book chunks into SQLite and Qdrant instead of
Pinecone. Prefer running `python scripts/ingest_books.py` directly.
"""

from scripts.ingest_books import main


if __name__ == "__main__":
    main()
