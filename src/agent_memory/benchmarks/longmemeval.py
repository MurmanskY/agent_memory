from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..evaluation.metrics import longmemeval_judge_prompt
from .base import Benchmark, Conversation, QASample, Session, Turn


class LongMemEvalBenchmark(Benchmark):
    """LongMemEval: personalized long-dialogue memory benchmark (500 questions).

    Expects `longmemeval_s.json` (or `_m`/`_oracle`). Each item has:
      - "question_id", "question", "answer", "question_type", "question_date"
      - "haystack_sessions": [[{"role", "content"}, ...], ...]
      - "haystack_session_ids", "haystack_dates"
    Note: each item is its own conversation (conv_id == question_id), since the
    haystack differs per question.
    """

    name = "longmemeval"

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        with open(self.data_path) as f:
            self._data = json.load(f)

    def conversations(self) -> Iterator[Conversation]:
        for item in self._data:
            sessions = []
            for sid, (raw, date) in enumerate(
                zip(item["haystack_sessions"], item.get("haystack_dates", []))
            ):
                sessions.append(
                    Session(
                        session_id=item.get("haystack_session_ids", [str(sid)])[sid]
                        if item.get("haystack_session_ids")
                        else str(sid),
                        timestamp=date,
                        turns=[
                            Turn(speaker=t.get("role", "?"), text=t.get("content", ""))
                            for t in raw
                        ],
                    )
                )
            yield Conversation(
                conv_id=item["question_id"],
                sessions=sessions,
                metadata={"question_date": item.get("question_date")},
            )

    def qa_samples(self, conv_id: str) -> list[QASample]:
        item = next(it for it in self._data if it["question_id"] == conv_id)
        return [
            QASample(
                question_id=item["question_id"],
                conv_id=conv_id,
                question=item["question"],
                answer=str(item["answer"]),
                category=item.get("question_type"),
            )
        ]

    def score_sample(self, sample: QASample, pred: str, llm=None) -> dict | None:
        """Official LongMemEval LLM-judge (evaluate_qa.py in the upstream repo).

        Abstention items (question_id endswith '_abs') are judged with the
        unanswerable-question template. Requires an LLM; without one we fall
        back to the default F1/EM metrics.
        """
        if llm is None:
            return None
        abstention = "_abs" in sample.question_id
        prompt = longmemeval_judge_prompt(
            sample.category or "", sample.question, sample.answer, pred, abstention
        )
        verdict = llm.chat("", prompt).strip().lower()
        return {"judge": float(verdict.startswith("yes"))}
