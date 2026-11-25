"""A-MEM: Agentic Memory for LLM Agents (reproduction).

Faithful port of the official implementation
(https://github.com/WujiangXu/AgenticMemory, memory_layer.py) to this
framework's BaseMemory interface. Reference: papers/AMem.pdf.

Pipeline (paper Sec. 3):
  1. Note construction (Ps1): LLM generates {keywords, context, tags} for each
     new memory; the embedding is computed over content+context+keywords+tags.
  2. Link generation + memory evolution (Ps2/Ps3, one combined LLM call):
     the k nearest historical notes are shown to the LLM, which decides whether
     to (a) "strengthen" — link the new note to neighbors and update its tags,
     and (b) "update_neighbor" — rewrite neighbors' context/tags.
  3. Retrieval: cosine top-k over note embeddings, then expand through links.

Official defaults kept: evo_threshold=100 (rebuild the retrieval index every
100 evolutions), k_neighbors=5 for evolution, all-MiniLM-L6-v2 embeddings
(paper uses sentence-transformers; any Embedder works here).
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ..core.base import BaseMemory, MemoryItem
from ..core.embeddings import Embedder, cosine_topk, get_embedder
from ..core.llm import LLMClient, get_llm

logger = logging.getLogger(__name__)

# LLM failures absorbed with a fallback instead of raised, by kind. Mirrors the
# official FAILURES counter so silent degradation shows up in experiment logs.
FAILURES: Counter = Counter()

# Prompt Ps1 (paper Appendix B.1), verbatim from the official repo.
NOTE_ANALYSIS_PROMPT = """Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object:
{
    "keywords": [
        // several specific, distinct keywords that capture key concepts and terminology
        // Order from most to least important
        // Don't include keywords that are the name of the speaker or time
        // At least three keywords, but don't be too redundant.
    ],
    "context":
        // one sentence summarizing:
        // - Main topic/domain
        // - Key arguments/points
        // - Intended audience/purpose
    ,
    "tags": [
        // several broad categories/themes for classification
        // Include domain, format, and type tags
        // At least three tags, but don't be too redundant.
    ]
}

Content for analysis:
"""

# Prompt Ps2/Ps3 (paper Appendix B.2/B.3), verbatim from the official repo.
EVOLUTION_PROMPT = """You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.
Return your decision in JSON format with the following structure:
{{
    "should_evolve": True or False,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_ids"],
    "tags_to_update": ["tag_1",..."tag_n"],
    "new_context_neighborhood": ["new context",...,"new context"],
    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
}}
"""


@dataclass
class MemoryNote:
    """Atomic memory note mi = {ci, ti, Ki, Gi, Xi, ei, Li} (paper Eq. 1)."""

    content: str
    id: str
    timestamp: str
    keywords: list[str] = field(default_factory=list)
    context: str = "General"
    tags: list[str] = field(default_factory=list)
    links: list[int] = field(default_factory=list)  # indices in insertion order
    evolution_history: list[dict] = field(default_factory=list)


def note_document(note: MemoryNote) -> str:
    """The text embedded for retrieval (paper Eq. 3)."""
    return (
        "content:" + note.content
        + " context:" + note.context
        + " keywords: " + ", ".join(note.keywords)
        + " tags: " + ", ".join(note.tags)
    )


def _parse_json_object(raw: str) -> dict:
    text = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    if not text.startswith("{"):
        text = text[text.find("{"): text.rfind("}") + 1]
    return json.loads(text)


def _as_index(x: Any) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return None


class AMemMemory(BaseMemory):
    """Agentic memory system (A-MEM)."""

    name = "amem"

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
        self.embedder = embedder or get_embedder(
            self.config.get("embedder", "openai"),
            **self.config.get("embedder_kwargs", {}),
        )
        # Evolution / linking hyperparameters (paper Sec. 4.2 and App. A.5).
        self.evo_threshold = int(self.config.get("evo_threshold", 100))
        self.k_neighbors = int(self.config.get("k_neighbors", 5))
        # False = paper behavior: evolved neighbors keep stale embeddings until
        # the index is rebuilt every evo_threshold evolutions.
        self.refresh_index = bool(self.config.get("refresh_index", False))

        self.memories: dict[str, MemoryNote] = {}
        self._docs: list[str] = []  # note_document per note, insertion order
        self._matrix: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._evo_cnt = 0

    # ------------------------------------------------------------------ write

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        timestamp = str(
            metadata.get("timestamp") or datetime.now().strftime("%Y%m%d%H%M")
        )
        note = MemoryNote(
            content=content,
            id=MemoryItem(content=content).id,
            timestamp=timestamp,
        )
        note.keywords, note.context, note.tags = self._analyze_content(content)
        self._evolve(note)
        self.memories[note.id] = note
        self._docs.append(note_document(note))
        self._matrix = np.vstack([self._matrix, self.embedder.embed([self._docs[-1]])])
        return note.id

    def _analyze_content(self, content: str) -> tuple[list[str], str, list[str]]:
        try:
            analysis = _parse_json_object(
                self.llm.chat(
                    "You must respond with a JSON object.",
                    NOTE_ANALYSIS_PROMPT + content,
                    json_mode=True,
                    max_tokens=1000,
                )
            )
            keywords = [str(k) for k in analysis.get("keywords") or []]
            context = analysis.get("context") or "General"
            if isinstance(context, list):
                context = " ".join(map(str, context))
            tags = [str(t) for t in analysis.get("tags") or []]
            return keywords, str(context), tags
        except Exception as e:  # store the note unanalyzed rather than crash
            logger.warning("note analysis failed: %s", e)
            FAILURES["analyze"] += 1
            return [], "General", []

    def _evolve(self, note: MemoryNote) -> None:
        if not self.memories:
            return
        try:
            neighbors, indices = self._find_related(note.content, self.k_neighbors)
            prompt = EVOLUTION_PROMPT.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors,
                neighbor_number=len(indices),
            )
            decision = _parse_json_object(
                self.llm.chat(
                    "You must respond with a JSON object.",
                    prompt,
                    json_mode=True,
                    max_tokens=1000,
                )
            )
            if not isinstance(decision, dict) or "should_evolve" not in decision:
                raise ValueError(f"evolution decision without should_evolve: {decision!r:.200}")
        except Exception as e:
            logger.warning("evolution failed: %s", e)
            FAILURES["evolve"] += 1
            return

        if not decision["should_evolve"]:
            return
        self._evo_cnt += 1
        notes = list(self.memories.values())
        note_ids = list(self.memories.keys())
        for action in decision.get("actions") or []:
            if action == "strengthen":
                shown = set(indices)
                for c in map(_as_index, decision.get("suggested_connections") or []):
                    if c in shown:
                        note.links.append(c)
                    else:
                        FAILURES["invalid_link"] += 1
                new_tags = decision.get("tags_to_update")
                if new_tags:  # an empty list must not wipe the note's tags
                    note.tags = [str(t) for t in new_tags]
            elif action == "update_neighbor":
                new_contexts = decision.get("new_context_neighborhood") or []
                new_tags_nb = decision.get("new_tags_neighborhood") or []
                for i in range(min(len(indices), len(new_tags_nb))):
                    neighbor = notes[indices[i]]
                    if new_tags_nb[i]:
                        neighbor.tags = [str(t) for t in new_tags_nb[i]]
                    if i < len(new_contexts) and new_contexts[i]:
                        neighbor.context = str(new_contexts[i])
                    neighbor.evolution_history.append({"from": note.id})
                    self.memories[note_ids[indices[i]]] = neighbor
                    self._docs[indices[i]] = note_document(neighbor)
                    if self.refresh_index:
                        self._matrix[indices[i]] = self.embedder.embed(
                            [self._docs[indices[i]]]
                        )[0]
        if self._evo_cnt % self.evo_threshold == 0:
            self._consolidate()

    def _consolidate(self) -> None:
        """Re-embed all notes so the index reflects evolved context/tags."""
        if not self._docs:
            return
        self._matrix = self.embedder.embed(self._docs)

    # ------------------------------------------------------------------ read

    def _search(self, query: str, k: int) -> list[int]:
        q = self.embedder.embed([query])[0]
        return [i for i, _ in cosine_topk(q, self._matrix, k)]

    def _find_related(self, query: str, k: int) -> tuple[str, list[int]]:
        """Neighbor summary string for the evolution prompt (official format)."""
        indices = self._search(query, k)
        notes = list(self.memories.values())
        out = ""
        for i in indices:
            m = notes[i]
            out += (
                f"memory index:{i}\t talk start time:{m.timestamp}\t"
                f"memory content: {m.content}\t memory context: {m.context}\t"
                f"memory keywords: {m.keywords}\t memory tags: {m.tags}\n"
            )
        return out, indices

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryItem]:
        """Top-k embedding retrieval, then expand through note links."""
        if not self.memories:
            return []
        notes = list(self.memories.values())
        seen: set[int] = set()
        out: list[MemoryItem] = []

        def emit(i: int) -> None:
            if i in seen or not (0 <= i < len(notes)):
                return
            seen.add(i)
            m = notes[i]
            out.append(
                MemoryItem(
                    content=m.content,
                    id=m.id,
                    metadata={
                        "timestamp": m.timestamp,
                        "context": m.context,
                        "keywords": m.keywords,
                        "tags": m.tags,
                        "links": m.links,
                    },
                )
            )

        for i in self._search(query, top_k):
            emit(i)
            for j, neighbor in enumerate(notes[i].links):
                if j >= top_k:  # official cap: at most k links per hit
                    break
                emit(neighbor)
        return out

    def render(self, item: MemoryItem) -> str:
        md = item.metadata
        return (
            f"talk start time:{md.get('timestamp')}\t"
            f"memory content: {item.content}\t"
            f"memory context: {md.get('context')}\t"
            f"memory keywords: {md.get('keywords')}\t"
            f"memory tags: {md.get('tags')}"
        )

    # --------------------------------------------------------------- maintain

    def save(self, path) -> None:
        """Persist the full memory state (notes + index) for reuse."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "memories": self.memories,
                    "docs": self._docs,
                    "matrix": self._matrix,
                    "evo_cnt": self._evo_cnt,
                },
                f,
            )

    def load(self, path) -> None:
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        self.memories = state["memories"]
        self._docs = state["docs"]
        self._matrix = state["matrix"]
        self._evo_cnt = state["evo_cnt"]

    def reset(self) -> None:
        self.memories = {}
        self._docs = []
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._evo_cnt = 0

    def __len__(self) -> int:
        return len(self.memories)
