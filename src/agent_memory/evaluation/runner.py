from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..benchmarks.base import Benchmark, format_turn
from ..core.base import BaseMemory
from ..core.llm import LLMClient
from .metrics import exact_match, f1_score, llm_judge

QA_SYSTEM_PROMPT = (
    "You are answering questions about a past conversation. Use ONLY the "
    "retrieved memories below. If the memories do not contain the answer, "
    'say "I don\'t know". Answer as concisely as possible.'
)


def answer_question(
    memory: BaseMemory, llm: LLMClient, question: str, top_k: int
) -> tuple[str, list[str]]:
    retrieved = memory.retrieve(question, top_k=top_k)
    context = "\n".join(f"- {memory.render(m)}" for m in retrieved) or "(no memories)"
    pred = llm.chat(
        QA_SYSTEM_PROMPT, f"Memories:\n{context}\n\nQuestion: {question}"
    )
    return pred, [m.id for m in retrieved]


def run_benchmark(
    benchmark: Benchmark,
    memory: BaseMemory,
    llm: LLMClient,
    top_k: int = 10,
    max_conversations: int | None = None,
    max_questions_per_conv: int | None = None,
    use_llm_judge: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ingest each conversation into `memory`, then QA-evaluate it.

    Writes one JSON object per line: {question_id, conv_id, question, gold,
    pred, category, retrieved_ids, f1, em, [judge]}.
    """
    records: list[dict[str, Any]] = []
    out_f = open(output_path, "w") if output_path else None

    conversations = list(benchmark.conversations())
    if max_conversations is not None:
        conversations = conversations[:max_conversations]

    for conv in tqdm(conversations, desc=f"[{benchmark.name}] {memory.name}"):
        memory.reset()
        for session in conv.sessions:
            for turn in session.turns:
                memory.add(
                    format_turn(turn),
                    metadata={"session_id": session.session_id,
                              "timestamp": session.timestamp},
                )

        samples = benchmark.qa_samples(conv.conv_id)
        if max_questions_per_conv is not None:
            samples = samples[:max_questions_per_conv]

        for sample in samples:
            pred, retrieved_ids = answer_question(memory, llm, sample.question, top_k)
            record: dict[str, Any] = {
                "question_id": sample.question_id,
                "conv_id": sample.conv_id,
                "question": sample.question,
                "gold": sample.answer,
                "pred": pred,
                "category": sample.category,
                "retrieved_ids": retrieved_ids,
            }
            scores = benchmark.score_sample(sample, pred, llm=llm)
            if scores is None:
                scores = {"f1": f1_score(pred, sample.answer),
                          "em": exact_match(pred, sample.answer)}
                if use_llm_judge:
                    scores["judge"] = llm_judge(
                        llm, sample.question, sample.answer, pred
                    )
            record.update(scores)
            records.append(record)
            if out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if out_f:
        out_f.close()

    numeric: dict[str, list[float]] = {}
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric.setdefault(k, []).append(float(v))
    summary = {
        "benchmark": benchmark.name,
        "method": memory.name,
        "n_questions": len(records),
        **{k: sum(vs) / len(vs) for k, vs in sorted(numeric.items())},
    }
    return summary
