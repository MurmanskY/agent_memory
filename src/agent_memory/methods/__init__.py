from __future__ import annotations

from ..baselines import BASELINES
from ..core.base import BaseMemory
from ..core.embeddings import Embedder
from ..core.llm import LLMClient
from .ours.method import OurMemory

# Our methods take precedence over baselines on name collision.
METHODS: dict[str, type[BaseMemory]] = {**BASELINES, "ours": OurMemory}


def get_method(
    name: str,
    config: dict | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
):
    if name == "agemem":
        # agent-style method (not a BaseMemory): has run_task(), driven by
        # benchmarks with a custom run() (e.g. hotpotqa)
        from ..baselines.agemem import AgeMemAgent

        return AgeMemAgent(config, llm=llm, embedder=embedder)
    if name not in METHODS:
        raise ValueError(f"unknown method: {name} (have: {sorted(METHODS)})")
    cls = METHODS[name]
    if cls is OurMemory:
        return cls(config, llm=llm, embedder=embedder)
    from ..baselines import get_baseline

    return get_baseline(name, config, llm=llm, embedder=embedder)


__all__ = ["METHODS", "get_method", "OurMemory"]
