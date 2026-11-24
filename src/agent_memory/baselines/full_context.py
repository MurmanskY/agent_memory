from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory, MemoryItem


class FullContextMemory(BaseMemory):
    """No memory management: keep everything, return the most recent items.

    Serves as the "unlimited context" upper reference and the truncation baseline.
    """

    name = "full_context"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_items: int = int(self.config.get("max_items", 10_000))
        self._items: list[MemoryItem] = []

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        item = MemoryItem(content=content, metadata=metadata or {})
        self._items.append(item)
        return item.id

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        # Query is ignored: full-context always sees (the tail of) everything.
        return self._items[-self.max_items :]

    def reset(self) -> None:
        self._items = []

    def __len__(self) -> int:
        return len(self._items)
