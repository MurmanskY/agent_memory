from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory


class LangMemMemory(BaseMemory):
    """LangMem (LangGraph) baseline. Skeleton — TODO.

    Reference: https://github.com/langchain-ai/langmem
    Wrap LangMem's semantic/episodic memory tools behind BaseMemory.
    """

    name = "langmem"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        raise NotImplementedError("LangMem baseline is a skeleton; see class docstring.")
