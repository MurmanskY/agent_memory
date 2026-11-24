from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import Benchmark, Conversation, QASample, Session, Turn


class LoCoMoBenchmark(Benchmark):
    """LoCoMo: long conversational memory benchmark (used in the A-MEM paper).

    Expects `locomo10.json` (list of conversations). Each item has:
      - "conversation": {"session_1": [{"speaker", "text", ...}, ...],
                         "session_1_date_time": ..., "session_2": ..., ...}
      - "qa": [{"question", "answer"/"adversarial_answer", "category", "evidence"}, ...]
    """

    name = "locomoco"

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        with open(self.data_path) as f:
            self._data = json.load(f)

    def conversations(self) -> Iterator[Conversation]:
        for i, item in enumerate(self._data):
            conv = item["conversation"]
            sessions: list[Session] = []
            n = 1
            while f"session_{n}" in conv:
                raw_turns = conv[f"session_{n}"]
                sessions.append(
                    Session(
                        session_id=f"session_{n}",
                        timestamp=conv.get(f"session_{n}_date_time"),
                        turns=[
                            Turn(speaker=t.get("speaker", "?"), text=t.get("text", ""))
                            for t in raw_turns
                        ],
                    )
                )
                n += 1
            yield Conversation(
                conv_id=str(item.get("sample_id", i)),
                sessions=sessions,
                metadata={
                    "speaker_a": item.get("speaker_a"),
                    "speaker_b": item.get("speaker_b"),
                },
            )

    def qa_samples(self, conv_id: str) -> list[QASample]:
        idx = int(conv_id) if str(conv_id).isdigit() else None
        item = self._data[idx] if idx is not None and idx < len(self._data) else None
        if item is None:
            item = next(
                it for it in self._data if str(it.get("sample_id")) == str(conv_id)
            )
        samples = []
        for j, qa in enumerate(item.get("qa", [])):
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            samples.append(
                QASample(
                    question_id=f"{conv_id}-{j}",
                    conv_id=str(conv_id),
                    question=qa["question"],
                    answer=str(answer),
                    category=str(qa.get("category", "")),
                    metadata={"evidence": qa.get("evidence", [])},
                )
            )
        return samples
