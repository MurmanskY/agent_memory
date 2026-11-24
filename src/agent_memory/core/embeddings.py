from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.model = model or os.environ.get(
            "OPENAI_EMBED_MODEL", "text-embedding-3-small"
        )
        self._client = OpenAI()
        self.dim = 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class HashEmbedder:
    """Deterministic hashing embedder for offline smoke tests (no API needed)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        return vecs


class STEmbedder:
    """sentence-transformers embedder (e.g. all-MiniLM-L6-v2, used by A-MEM).

    Requires the optional dependency: pip install sentence-transformers
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, convert_to_numpy=True), dtype=np.float32
        )


def cosine_topk(
    query: np.ndarray, matrix: np.ndarray, top_k: int
) -> list[tuple[int, float]]:
    """Return (index, score) of the top_k rows of `matrix` closest to `query`."""
    if matrix.size == 0:
        return []
    q = query / (np.linalg.norm(query) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    scores = m @ q
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx]


def get_embedder(name: str = "openai", **kwargs) -> Embedder:
    if name == "openai":
        return OpenAIEmbedder(**kwargs)
    if name == "hash":
        return HashEmbedder(**kwargs)
    if name in ("sentence_transformers", "st"):
        return STEmbedder(**kwargs)
    raise ValueError(f"unknown embedder: {name}")
