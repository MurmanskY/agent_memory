"""BaseMemory adapter for session-level ProMem extraction and retrieval."""

from __future__ import annotations

import pickle
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ...core.base import BaseMemory, MemoryItem
from ...core.embeddings import Embedder, cosine_topk, get_embedder
from ...core.llm import get_llm
from .models import EmbedderAdapter, LLMAdapter, NLI
from .pipeline import ProMem


class ProMemMemory(BaseMemory):
    name = "promem"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm=None,
        embedder: Embedder | None = None,
        nli=None,
    ):
        super().__init__(config)
        self.llm = llm or get_llm(
            self.config.get("llm", "hf"), **self.config.get("llm_kwargs", {})
        )
        self.embedder = embedder or get_embedder(
            self.config.get("embedder", "sentence_transformers"),
            **self.config.get("embedder_kwargs", {}),
        )
        self.nli = nli or NLI(
            self.config.get("nli_model", "models/roberta-large-mnli"),
            device=self.config.get("nli_device"),
        )
        self.pipeline = ProMem(
            LLMAdapter(self.llm),
            EmbedderAdapter(self.embedder),
            self.nli,
            tau_comp=float(self.config.get("tau_comp", 0.6)),
        )
        self._items: list[MemoryItem] = []
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._pending: dict[str, list[dict]] = {}
        self._pending_meta: dict[str, dict] = {}

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        session_id = str(metadata.get("session_id", "default"))
        timestamp = str(metadata.get("timestamp", ""))
        match = re.match(r"^\[(.*?)\]([^:]+):\s*(.*)$", content, flags=re.DOTALL)
        if match:
            timestamp = match.group(1) or timestamp
            role, text = match.group(2).strip(), match.group(3)
        elif ": " in content:
            role, text = content.split(": ", 1)
        else:
            role, text = "user", content
        turn = {
            "role": metadata.get("role", role),
            "content": metadata.get("text", text),
            "timestamp": timestamp,
        }
        self._pending.setdefault(session_id, []).append(turn)
        meta = self._pending_meta.setdefault(session_id, {})
        meta.update(metadata)
        meta["user_name"] = metadata.get("user_name", meta.get("user_name", "User"))
        return f"pending-{uuid.uuid4().hex[:12]}"

    def add_session(
        self,
        dialogue: list[dict],
        user_name: str = "User",
        session_id: str = "default",
        timestamp: str = "",
    ) -> list[MemoryItem]:
        session_id = str(session_id)
        pending = self._pending.pop(session_id, [])
        self._pending_meta.pop(session_id, None)
        turns = pending + dialogue
        graph = self.pipeline.process_session({"dialogue": turns}, user_name)
        return self._store_graph(graph, session_id, timestamp)

    def session_memories(self, session_id: str) -> list[MemoryItem]:
        session_id = str(session_id)
        if session_id in self._pending:
            meta = self._pending_meta.get(session_id, {})
            self.add_session(
                [],
                user_name=meta.get("user_name", "User"),
                session_id=session_id,
                timestamp=str(meta.get("timestamp", "")),
            )
        return [item for item in self._items if item.metadata.get("session_id") == session_id]

    def _store_graph(self, graph, session_id: str, timestamp: str) -> list[MemoryItem]:
        added = []
        texts = []
        for i, node in enumerate(graph.nodes):
            details = graph.attrs.get(i, [])
            content = node
            if details:
                content += " (Details: " + "; ".join(details) + ")"
            texts.append(content)
            added.append(MemoryItem(
                content=content,
                metadata={
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "node": node,
                    "details": details,
                    "edges": graph.edges,
                    "promem_stats": graph.stats,
                },
            ))
        if texts:
            vecs = self.embedder.embed(texts)
            for item, vec in zip(added, vecs):
                item.embedding = vec.tolist()
            self._items.extend(added)
            self._matrix = np.vstack([self._matrix, vecs])
        return added

    def _flush(self) -> None:
        for session_id in list(self._pending):
            meta = self._pending_meta.get(session_id, {})
            self.add_session(
                [],
                user_name=meta.get("user_name", "User"),
                session_id=session_id,
                timestamp=str(meta.get("timestamp", "")),
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        self._flush()
        if not self._items:
            return []
        query_vec = self.embedder.embed([query])[0]
        return [self._items[i] for i, _ in cosine_topk(query_vec, self._matrix, top_k)]

    def render(self, item: MemoryItem) -> str:
        timestamp = item.metadata.get("timestamp")
        return f"{timestamp}: {item.content}" if timestamp else item.content

    def reset(self) -> None:
        self._items = []
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._pending = {}
        self._pending_meta = {}

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as file:
            pickle.dump({
                "items": self._items,
                "matrix": self._matrix,
                "pending": self._pending,
                "pending_meta": self._pending_meta,
            }, file)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as file:
            state = pickle.load(file)
        self._items = state["items"]
        self._matrix = state["matrix"]
        self._pending = state["pending"]
        self._pending_meta = state["pending_meta"]

    def __len__(self) -> int:
        return len(self._items)
