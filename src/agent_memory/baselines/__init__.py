from __future__ import annotations

from ..core.base import BaseMemory
from ..core.embeddings import Embedder
from .full_context import FullContextMemory
from .rag import RAGMemory
from .memgpt import MemGPTMemory
from .amem import AMemMemory
from .secom import SeComMemory
from .langmem import LangMemMemory
from .mem0 import Mem0Memory
from .memoryr1 import MemoryR1Memory
from .promem import ProMemMemory

BASELINES: dict[str, type[BaseMemory]] = {
    "full_context": FullContextMemory,
    "rag": RAGMemory,
    "memgpt": MemGPTMemory,
    "amem": AMemMemory,
    "secom": SeComMemory,
    "langmem": LangMemMemory,
    "mem0": Mem0Memory,
    "memoryr1": MemoryR1Memory,
    "promem": ProMemMemory,
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
    if cls is SeComMemory:
        return cls(config, llm=llm, embedder=embedder)
    if cls is MemoryR1Memory:
        return cls(config, llm=llm, embedder=embedder)
    if cls is Mem0Memory:
        return cls(config, llm=llm, embedder=embedder)
    if cls is ProMemMemory:
        return cls(config, llm=llm, embedder=embedder)
    return cls(config)


__all__ = [
    "BASELINES",
    "get_baseline",
    "FullContextMemory",
    "RAGMemory",
    "MemGPTMemory",
    "AMemMemory",
    "SeComMemory",
    "LangMemMemory",
    "Mem0Memory",
    "MemoryR1Memory",
    "ProMemMemory",
]
