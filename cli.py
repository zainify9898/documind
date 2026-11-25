"""Command-line interface for DocuMind.

Usage:
    python cli.py --pdf data/sample.pdf --question "What does the document say about AI in healthcare?"
    python cli.py --pdf report.pdf --question "Summarise the methodology." --top-k 3

The first run on a PDF builds a FAISS index (saved under .documind_cache/)
and reuses it on later runs. Set OPENAI_API_KEY to enable generative answers;
without it, DocuMind answers extractively from the retrieved passages.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embeddings import EMBEDDING_DIM, Embedder
from src.ingest import ingest_pdf
from src.qa import DocuMindQA
from src.vectorstore import VectorStore

CACHE_DIR = ".documind_cache"


def _cache_key(pdf_path: str) -> str:
    """Hash of the file contents, so a changed PDF is re-indexed."""
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def build_or_load(pdf_path: str, embedder: Embedder) -> VectorStore:
    """Load a cached index for this PDF, or build one from scratch."""
    key = _cache_key(pdf_path)
    cache_path = os.path.join(CACHE_DIR, key)
    if os.path.exists(os.path.join(cache_path, "index.faiss")):
        print(f"Loaded cached index for {pdf_path}")
        return VectorStore.load(cache_path, EMBEDDING_DIM)

    print(f"Indexing {pdf_path} ...")
    chunks = ingest_pdf(pdf_path)
    print(f"  {len(chunks)} chunks extracted")
    store = VectorStore(EMBEDDING_DIM)
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    store.save(cache_path)
    print(f"  index saved ({len(store)} vectors)")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind — Chat with Your PDFs")
    parser.add_argument("--pdf", required=True, help="Path to the PDF to ask about")
    parser.add_argument("--question", required=True, help="Question about the document")
    parser.add_argument("--top-k", type=int, default=5, help="Passages to retrieve (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        parser.error(f"PDF not found: {args.pdf}")

    embedder = Embedder()
    store = build_or_load(args.pdf, embedder)
    qa = DocuMindQA(store, embedder)
    answer = qa.ask(args.question, top_k=args.top_k)

    print(f"\nQ: {args.question}\n")
    print(f"Mode: {answer.mode}")
    print(answer.answer)
    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            print(f"  - page {source.page} (similarity {source.score:.2f})")
    print(f"\n{answer.note}")


if __name__ == "__main__":
    main()
