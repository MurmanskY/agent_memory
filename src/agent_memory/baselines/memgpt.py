from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory, MemoryItem


class MemGPTMemory(BaseMemory):
    """MemGPT baseline (paged virtual-context memory). Skeleton — TODO.

    Reference: Packer et al., "MemGPT: Towards LLMs as Operating Systems".
    Repo: https://github.com/cpacker/MemGPT

    Implementation plan:
      - core (in-context) memory blocks: persona / human, edited by LLM tool calls;
      - archival memory: embedding-searchable store (reuse RAGMemory internals);
      - recall memory: full conversation log with paging;
      - memory functions exposed as tools: core_memory_append/replace,
        archival_memory_insert/search, conversation_search.
    """

    name = "memgpt"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        raise NotImplementedError("MemGPT baseline is a skeleton; see class docstring.")
