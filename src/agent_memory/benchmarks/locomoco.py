"""LoCoMo benchmark with the official A-MEM evaluation protocol.

Aligned with WujiangXu/AgenticMemory (test_advanced.py):
  - ingestion: each turn stored as "Speaker {speaker}says : {text}" with the
    session date_time as timestamp;
  - retrieval query: LLM-generated keywords from the question;
  - QA prompts are category-specific:
      cat 2 (temporal)  -> answer with an approximate date from the DATE of
                           CONVERSATION, shortest possible answer;
      cat 5 (adversarial) -> forced choice between "Not mentioned in the
                           conversation" and the adversarial (wrong) answer,
                           option order seeded per question;
      others            -> short phrase, exact words from context;
  - scoring: set-F1 / BLEU-1 vs the reference; for cat 5 the reference is
    "Not mentioned in the conversation" (f1_legacy vs the adversarial answer
    is also reported, matching the paper's original numbers).
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Iterator

from ..evaluation.metrics import bleu1, f1_set
from .base import Benchmark, Conversation, QASample, Session, Turn

logger = logging.getLogger(__name__)

NOT_MENTIONED = "Not mentioned in the conversation"

# Verbatim from upstream test_advanced.py
KEYWORD_QUERY_PROMPT = """Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text.

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""

QA_PROMPT_TEMPORAL = """
Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:
"""

QA_PROMPT_SHORT = """Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:
"""

QA_PROMPT_ADVERSARIAL = """
Based on the context: {context}, answer the following question. {question}

Select the correct answer: {option_a} or {option_b}  Short answer:
"""


def _parse_answer(raw: str) -> str:
    text = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    if not text.startswith("{"):
        text = text[text.find("{"): text.rfind("}") + 1]
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"])
    except (json.JSONDecodeError, ValueError):
        pass
    return raw.strip()


def _is_not_mentioned(prediction: str, trap: str | None = None) -> float:
    """1 if the prediction declines AND does not give the trap answer (upstream utils)."""
    text = str(prediction).lower()
    declines = "not mentioned" in text or "no information available" in text
    trap = str(trap or "").lower().strip(" .")
    gives_trap = bool(trap) and not trap.startswith("not mentioned") and trap in text
    return float(declines and not gives_trap)


class LoCoMoBenchmark(Benchmark):
    """LoCoMo: long conversational memory benchmark (used in the A-MEM paper).

    Expects `locomo10.json` (list of conversations). Each item has:
      - "conversation": {"session_1": [{"speaker", "text", ...}, ...],
                         "session_1_date_time": ..., "session_2": ..., ...}
      - "qa": [{"question", "answer"/"adversarial_answer", "category", "evidence"}, ...]
    """

    name = "locomoco"

    def __init__(self, data_path: str | Path, seed: int = 42):
        self.data_path = Path(data_path)
        self.seed = seed
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
            samples.append(
                QASample(
                    question_id=f"{conv_id}-{j}",
                    conv_id=str(conv_id),
                    question=qa["question"],
                    answer=str(qa.get("answer", "")),
                    category=str(qa.get("category", "")),
                    metadata={
                        "evidence": qa.get("evidence", []),
                        "adversarial_answer": qa.get("adversarial_answer"),
                    },
                )
            )
        return samples

    # ------------------------------------------------ official protocol hooks

    def format_turn(self, turn: Turn) -> str:
        return f"Speaker {turn.speaker}says : {turn.text}"

    def retrieve_query(self, sample: QASample, llm=None) -> str:
        if llm is None:
            return sample.question
        try:
            raw = llm.chat(
                "You must respond with a JSON object.",
                KEYWORD_QUERY_PROMPT.format(question=sample.question),
                json_mode=True,
            )
            text = raw.strip()
            if not text.startswith("{"):
                text = text[text.find("{"): text.rfind("}") + 1]
            return json.loads(text)["keywords"]
        except Exception:
            logger.warning("keyword generation failed, using raw question")
            return sample.question

    def qa_prompt(self, sample: QASample, context: str) -> str | None:
        cat = sample.category
        if cat == "5":  # adversarial: forced choice, seeded option order
            rng = random.Random(f"{self.seed}:{sample.question_id}")
            options = [NOT_MENTIONED, sample.metadata.get("adversarial_answer") or ""]
            if rng.random() >= 0.5:
                options.reverse()
            return QA_PROMPT_ADVERSARIAL.format(
                context=context, question=sample.question,
                option_a=options[0], option_b=options[1],
            )
        if cat == "2":  # temporal
            return QA_PROMPT_TEMPORAL.format(
                context=context, question=sample.question
            )
        return QA_PROMPT_SHORT.format(context=context, question=sample.question)

    def score_sample(self, sample: QASample, pred: str, llm=None) -> dict | None:
        pred = _parse_answer(pred)
        if sample.category == "5":
            adv = sample.metadata.get("adversarial_answer") or ""
            return {
                "f1": f1_set(pred, NOT_MENTIONED),
                "bleu1": bleu1(pred, NOT_MENTIONED),
                "f1_legacy": f1_set(pred, adv),
                "adversarial_correct": _is_not_mentioned(pred, adv),
            }
        return {
            "f1": f1_set(pred, sample.answer),
            "bleu1": bleu1(pred, sample.answer),
        }
