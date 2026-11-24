from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryItem:
    """A single unit stored in a memory system."""

    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    """Unified interface that every memory system (baseline or ours) implements.

    The evaluation runner only relies on this interface, so any new method can be
    benchmarked against all baselines without extra glue code.
    """

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a piece of memory (one dialogue turn, observation, ...).

        Returns the id of the created MemoryItem.
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """Fetch the memories most relevant to `query`."""

    def update(self, memory_id: str, content: str) -> None:
        raise NotImplementedError(f"{self.name} does not support update")

    def delete(self, memory_id: str) -> None:
        raise NotImplementedError(f"{self.name} does not support delete")

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memories (called between conversations)."""

    def render(self, item: MemoryItem) -> str:
        """How a retrieved memory is presented in the QA prompt."""
        return item.content

    def __len__(self) -> int:
        return 0
