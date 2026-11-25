"""DocuMind — Chat with Your PDFs. Offline RAG research assistant."""

from .embeddings import EMBEDDING_DIM, MODEL_NAME, Embedder
from .ingest import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, Chunk, ingest_pdf
from .qa import Answer, DocuMindQA, Source
from .vectorstore import VectorStore

__all__ = [
    "Answer",
    "Chunk",
    "DEFAULT_CHUNK_WORDS",
    "DEFAULT_OVERLAP_WORDS",
    "DocuMindQA",
    "EMBEDDING_DIM",
    "Embedder",
    "MODEL_NAME",
    "Source",
    "VectorStore",
    "ingest_pdf",
]

__version__ = "1.0.0"
