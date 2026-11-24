from __future__ import annotations

from ..core.base import BaseMemory
from ..core.embeddings import Embedder
from .full_context import FullContextMemory
from .rag import RAGMemory
from .memgpt import MemGPTMemory
from .amem import AMemMemory
from .langmem import LangMemMemory

BASELINES: dict[str, type[BaseMemory]] = {
    "full_context": FullContextMemory,
    "rag": RAGMemory,
    "memgpt": MemGPTMemory,
    "amem": AMemMemory,
    "langmem": LangMemMemory,
}


def get_baseline(
    name: str,
    config: dict | None = None,
    llm=None,
    embedder: Embedder | None = None,
) -> BaseMemory:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline: {name} (have: {sorted(BASELINES)})")
    cls = BASELINES[name]
    if cls is RAGMemory:
        return cls(config, embedder=embedder)
    if cls is AMemMemory:
        return cls(config, llm=llm, embedder=embedder)
    return cls(config)


__all__ = [
    "BASELINES",
    "get_baseline",
    "FullContextMemory",
    "RAGMemory",
    "MemGPTMemory",
    "AMemMemory",
    "LangMemMemory",
]
