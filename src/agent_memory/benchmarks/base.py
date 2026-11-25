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

    def format_turn(self, turn: Turn) -> str:
        """How a dialogue turn is written into memory. Override per benchmark."""
        return f"{turn.speaker}: {turn.text}"

    def retrieve_query(self, sample: QASample, llm=None) -> str:
        """The query sent to memory.retrieve for this QA sample."""
        return sample.question

    def qa_prompt(self, sample: QASample, context: str) -> str | None:
        """Benchmark-specific QA prompt over retrieved `context`.

        Return None to use the generic system prompt. When a prompt is
        returned, the runner asks for a JSON object and reads its "answer"
        field (falling back to the raw response).
        """
        return None

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
