"""Text normalization utilities for medical-book ingestion."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"[ \t]+")
_LINEBREAK_RE = re.compile(r"\n{3,}")
_HYPHENATED_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalize extracted PDF text while preserving paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHENATED_LINEBREAK_RE.sub(r"\1\2", text)
    text = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = _LINEBREAK_RE.sub("\n\n", text)
    return text.strip()
