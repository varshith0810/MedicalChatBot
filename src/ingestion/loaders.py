"""Document loaders for ingestion sources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


@dataclass(frozen=True)
class LoadedPage:
    source_path: Path
    page_number: int
    text: str


@dataclass(frozen=True)
class LoadedBook:
    source_path: Path
    title: str | None
    checksum: str
    pages: list[LoadedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)


def iter_pdf_paths(data_dir: str | Path) -> Iterable[Path]:
    """Yield PDF files under a directory in deterministic order."""

    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.pdf") if path.is_file())


def file_checksum(path: Path) -> str:
    """Calculate a stable SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pdf(path: str | Path) -> LoadedBook:
    """Load text from each page in a PDF file."""

    source_path = Path(path)
    reader = PdfReader(str(source_path))
    metadata_title = None
    if reader.metadata and reader.metadata.title:
        metadata_title = str(reader.metadata.title)

    pages = [
        LoadedPage(
            source_path=source_path,
            page_number=index + 1,
            text=page.extract_text() or "",
        )
        for index, page in enumerate(reader.pages)
    ]
    return LoadedBook(
        source_path=source_path,
        title=metadata_title or source_path.stem,
        checksum=file_checksum(source_path),
        pages=pages,
    )


def load_books(data_dir: str | Path) -> list[LoadedBook]:
    """Load all PDFs from a data directory."""

    return [load_pdf(path) for path in iter_pdf_paths(data_dir)]
