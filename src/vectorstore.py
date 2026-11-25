"""FAISS vector store for DocuMind.

Stores chunk embeddings in a FAISS inner-product index. Because the
embeddings are L2-normalised, inner product is cosine similarity, so the
nearest neighbours are the most semantically similar chunks. The index is
saved to disk next to the chunk metadata so a document only needs to be
indexed once.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from .ingest import Chunk

INDEX_FILENAME = "index.faiss"
META_FILENAME = "chunks.json"


class VectorStore:
    """FAISS-backed store of document chunks and their embeddings."""

    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self.faiss = faiss
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the index."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("chunks and embeddings must have the same length")
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"expected embedding dim {self.dim}, got {embeddings.shape[1]}"
            )
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the ``top_k`` most similar chunks with their scores.

        Scores are cosine similarities in [-1, 1]; higher is more similar.
        """
        query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
        scores, indices = self.index.search(query, k)
        return [
            (self.chunks[int(i)], float(scores[0][rank]))
            for rank, i in enumerate(indices[0])
            if i >= 0
        ]

    def save(self, directory: str) -> None:
        """Persist the index and chunk metadata to ``directory``."""
        os.makedirs(directory, exist_ok=True)
        self.faiss.write_index(self.index, os.path.join(directory, INDEX_FILENAME))
        with open(os.path.join(directory, META_FILENAME), "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str, dim: int) -> "VectorStore":
        """Load a previously saved store."""
        import faiss

        store = cls(dim)
        store.index = faiss.read_index(os.path.join(directory, INDEX_FILENAME))
        with open(os.path.join(directory, META_FILENAME), encoding="utf-8") as f:
            store.chunks = [Chunk(**item) for item in json.load(f)]
        return store

    def __len__(self) -> int:
        return self.index.ntotal
