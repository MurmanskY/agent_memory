from __future__ import annotations

from typing import Any

import numpy as np

from ..core.base import BaseMemory, MemoryItem
from ..core.embeddings import Embedder, cosine_topk, get_embedder


class RAGMemory(BaseMemory):
    """Naive RAG baseline: embed each memory, retrieve by cosine similarity."""

    name = "rag"

    def __init__(self, config: dict[str, Any] | None = None, embedder: Embedder | None = None):
        super().__init__(config)
        self.embedder = embedder or get_embedder(
            self.config.get("embedder", "openai"),
            **self.config.get("embedder_kwargs", {}),
        )
        self._items: list[MemoryItem] = []
        self._matrix: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        vec = self.embedder.embed([content])[0]
        item = MemoryItem(content=content, metadata=metadata or {}, embedding=vec.tolist())
        self._items.append(item)
        self._matrix = np.vstack([self._matrix, vec[None, :]])
        return item.id

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        q = self.embedder.embed([query])[0]
        return [self._items[i] for i, _ in cosine_topk(q, self._matrix, top_k)]

    def delete(self, memory_id: str) -> None:
        keep = [i for i, it in enumerate(self._items) if it.id != memory_id]
        self._items = [self._items[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else np.zeros(
            (0, self.embedder.dim), dtype=np.float32
        )

    def reset(self) -> None:
        self._items = []
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._items)
