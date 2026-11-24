from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class Session:
    """One dialogue session inside a (possibly multi-session) conversation."""

    session_id: str
    turns: list[Turn]
    timestamp: str | None = None


@dataclass
class Conversation:
    conv_id: str
    sessions: list[Session]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QASample:
    question_id: str
    conv_id: str
    question: str
    answer: str
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    """A benchmark = a set of conversations to ingest + QA to evaluate."""

    name: str = "base"

    @abstractmethod
    def conversations(self) -> Iterator[Conversation]: ...

    @abstractmethod
    def qa_samples(self, conv_id: str) -> list[QASample]: ...

    def score_sample(self, sample: QASample, pred: str, llm=None) -> dict | None:
        """Benchmark-specific scoring. Return None to use default F1/EM.

        The returned dict is merged into the result record; numeric values are
        averaged in the run summary. `llm` is provided for LLM-as-judge setups.
        """
        return None

    def run(self, memory, llm, **kwargs) -> dict | None:
        """Optional custom protocol overriding the generic runner (e.g. GateMem's
        incremental ingest-with-checkpoints). Return None to use the generic runner."""
        return None


def format_turn(turn: Turn) -> str:
    return f"{turn.speaker}: {turn.text}"
