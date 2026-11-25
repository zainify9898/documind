"""Local embeddings for DocuMind.

Uses the sentence-transformers model ``all-MiniLM-L6-v2`` — a small, fast,
free model that runs entirely on CPU. No API key, no network calls at query
time, and no document text ever leaves the machine.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class Embedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalised float32 embeddings.

        Normalised embeddings make cosine similarity equal to the dot
        product, which is what the FAISS inner-product index computes.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Encode a single text (e.g. a user question)."""
        return self.embed([text])[0]
