"""Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.

Reproduction of the paper's LoCoMo evaluation pipeline (papers/Mem0.pdf),
ported from the official evaluation code (mem0ai/mem0@v0.1.100,
evaluation/src/memzero/{add,search}.py + evaluation/prompts.py).

Mem0 shares its memory write path with Memory-R1 (fact extraction →
ADD/UPDATE/DELETE/NOOP memory manager, per-speaker banks), which is
implemented in `baselines/memoryr1.py` — this class only changes the
answering side to the paper's protocol:

  - retrieval: top-10 memories per speaker (official default top_k=10),
  - answer prompt: the paper's "Prompt Template for Results Generation
    (Mem0)" (Appendix A), which — unlike the Memory-R1 variants — has no
    "**Answer:**" output instruction,
  - the raw completion is the answer (official code uses the response
    as-is; no "**Answer:**" extraction).

The graph variant Mem0g is not included (needs a graph store).
"""
from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.embeddings import Embedder
from ..core.llm import LLMClient
from .memoryr1 import ANSWER_PROMPTS, MemoryR1Memory

# Verbatim from mem0ai/mem0@v0.1.100 evaluation/prompts.py (ANSWER_PROMPT),
# jinja {{var}} converted to str.format {var}. Note: no "**Answer:**"
# instruction — that was added later by Memory-R1.
MEM0_ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {speaker_1_user_id}:

    {speaker_1_memories}

    Memories for user {speaker_2_user_id}:

    {speaker_2_memories}

    Question: {question}

    Answer:
    """


class Mem0Memory(MemoryR1Memory):
    """Mem0 (non-graph). See module docstring for the protocol differences."""

    name = "mem0"
    answer_prompts = {**ANSWER_PROMPTS, "mem0": MEM0_ANSWER_PROMPT}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
    ):
        cfg = {"answer_prompt": "mem0"}  # mem0's own prompt, no distillation
        cfg.update(config or {})
        super().__init__(cfg, llm=llm, embedder=embedder)

    def answer(self, question: str, llm: LLMClient, top_k: int = 10) -> tuple[str, list[str]]:
        """Official protocol: top-10 per speaker; raw completion is the answer."""
        prompt, ids = self._build_answer_prompt(question, top_k)
        raw = llm.chat(prompt, "", max_tokens=self.answer_max_tokens)
        return raw.strip(), ids

