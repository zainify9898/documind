"""PDF ingestion for DocuMind.

Extracts text from a PDF and splits it into overlapping word-based chunks,
keeping the page number of every chunk so answers can cite their sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pypdf import PdfReader

DEFAULT_CHUNK_WORDS = 500
DEFAULT_OVERLAP_WORDS = 100


@dataclass
class Chunk:
    """One retrievable piece of a document."""

    text: str
    page: int
    index: int

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(index={self.index}, page={self.page}, text={preview!r}...)"


def _clean(text: str) -> str:
    """Collapse whitespace and strip control characters."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return ``(page_number, text)`` pairs for every page of a PDF.

    Page numbers are 1-based, matching what a reader sees in a PDF viewer.
    """
    reader = PdfReader(pdf_path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        pages.append((i, _clean(raw)))
    return pages


def chunk_words(words: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    """Split a word list into overlapping windows.

    Windows advance by ``chunk_size - overlap`` words, so consecutive chunks
    share ``overlap`` words of context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    step = chunk_size - overlap
    chunks: list[list[str]] = []
    start = 0
    while start < len(words):
        chunks.append(words[start : start + chunk_size])
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def ingest_pdf(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Extract a PDF and return its text as overlapping, page-tagged chunks.

    Chunks never cross a page boundary, so every chunk has exactly one page
    number and citations are always unambiguous.
    """
    chunks: list[Chunk] = []
    for page_number, page_text in extract_pages(pdf_path):
        words = page_text.split()
        if not words:
            continue
        for window in chunk_words(words, chunk_size, overlap):
            chunks.append(
                Chunk(
                    text=" ".join(window),
                    page=page_number,
                    index=len(chunks),
                )
            )
    return chunks


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    result = ingest_pdf(path)
    print(f"{path}: {len(result)} chunks")
    for chunk in result[:3]:
        print(chunk)
