from __future__ import annotations

from typing import Any

from ...core.base import BaseMemory, MemoryItem
from ...core.embeddings import Embedder, get_embedder
from ...core.llm import LLMClient, get_llm


class OurMemory(BaseMemory):
    """Our memory method — placeholder only.

    The class is wired into the method registry ("ours") and the evaluation
    pipeline so the framework is complete, but no algorithm is implemented
    yet. Fill in `add` / `retrieve` (and optionally `update` / `delete`)
    once the design is settled; add submodules in this package as needed.
    """

    name = "ours"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
    ):
        super().__init__(config)
        self.llm = llm or get_llm(
            self.config.get("llm", "openai"), **self.config.get("llm_kwargs", {})
        )
        self.embedder = embedder or get_embedder(self.config.get("embedder", "openai"))

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        raise NotImplementedError("our method is not implemented yet")

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        raise NotImplementedError("our method is not implemented yet")

    def reset(self) -> None:
        raise NotImplementedError("our method is not implemented yet")
