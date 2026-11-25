"""HotpotQA (distractor) benchmark for the AgeMem agent.

Upstream protocol (y1y5/AgeMem, eval_hotpotQA.py): per question, the agent
first gets a casual-chat pass over the context paragraphs (Stage 1, stores
facts into LTM), then answers the question (Stage 3) using its 6 memory tools.
Judged by LLM-as-a-judge (official prompt below); we also report standard
HotpotQA EM/F1. The judge here is the local Qwen3-4B (official: qwen-max via
DashScope) — treat J as an approximation.

Data: HF hotpotqa/hotpot_qa, config "distractor", split "validation".
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

from ..evaluation.metrics import exact_match_norm, f1_score
from .base import Benchmark, Conversation, QASample

logger = logging.getLogger(__name__)

# Verbatim from official workflow_metrics.get_answer_llm_judge_score
JUDGE_PROMPT = """You are an expert judge evaluating the correctness of answers to questions.

Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {predicted_answer}

Please evaluate the generated answer on a scale of 0.0 to 1.0:
- 1.0: Perfect match or equivalent correct answer
- 0.8-0.9: Mostly correct with minor differences
- 0.6-0.7: Partially correct or close approximation
- 0.4-0.5: Some correct elements but significant errors
- 0.2-0.3: Mostly incorrect with few correct elements
- 0.0-0.1: Completely incorrect or irrelevant

Respond with only a number between 0.0 and 1.0 (e.g., "0.85")."""


class HotpotQABenchmark(Benchmark):
    """HotpotQA distractor validation. Each example = one agent task."""

    name = "hotpotqa"

    def __init__(self, data_path: str | Path, max_samples: int | None = None):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "HotpotQA data is parquet; install the 'bench' extra: "
                "pip install -e '.[bench]'"
            ) from e
        rows = pq.read_table(data_path).to_pylist()
        self._rows = rows[:max_samples] if max_samples else rows

    # The agent loop is driven by run(); the generic interface is unused.

    def conversations(self) -> Iterator[Conversation]:
        raise NotImplementedError("hotpotqa uses its own run() protocol")

    def qa_samples(self, conv_id: str) -> list[QASample]:
        raise NotImplementedError("hotpotqa uses its own run() protocol")

    def run(
        self,
        agent,
        llm,
        top_k: int = 0,  # unused; the agent decides its own retrieval
        max_episodes: int | None = None,  # reused as max_questions
        output_path: str | Path | None = None,
        **_,
    ) -> dict:
        records: list[dict[str, Any]] = []
        out_f = open(output_path, "w") if output_path else None

        rows = self._rows[:max_episodes] if max_episodes else self._rows
        for row in rows:
            paragraphs = list(zip(row["context"]["title"], row["context"]["sentences"]))
            result = agent.run_task(row["question"], paragraphs)
            pred = result["answer"]

            record: dict[str, Any] = {
                "question_id": row["id"],
                "question": row["question"],
                "gold": row["answer"],
                "pred": pred,
                "type": row.get("type"),
                "level": row.get("level"),
                "em": exact_match_norm(pred, row["answer"]),
                "f1": f1_score(pred, row["answer"]),
                "judge": self._judge(llm, row["question"], row["answer"], pred),
                "n_memories": result["n_memories"],
                "tool_usage": result["tool_usage"],
            }
            records.append(record)
            if out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("[%s] EM=%.2f F1=%.2f J=%.2f", row["id"],
                        record["em"], record["f1"], record["judge"])

        if out_f:
            out_f.close()

        n = max(len(records), 1)
        return {
            "benchmark": self.name,
            "method": getattr(agent, "name", "?"),
            "n_questions": len(records),
            "em": sum(r["em"] for r in records) / n,
            "f1": sum(r["f1"] for r in records) / n,
            "judge": sum(r["judge"] for r in records) / n,
            "avg_memories": sum(r["n_memories"] for r in records) / n,
        }

    @staticmethod
    def _judge(llm, question: str, gold: str, pred: str) -> float:
        try:
            text = llm.chat(
                "",
                JUDGE_PROMPT.format(
                    question=question, expected_answer=gold, predicted_answer=pred
                ),
                max_tokens=16,
            ).strip()
            m = re.search(r"(\d+\.?\d*)", text)
            return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
        except Exception as e:
            logger.warning("judge failed: %s", e)
            return 0.0
