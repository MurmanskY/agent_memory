from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..benchmarks.base import Benchmark
from ..core.base import BaseMemory
from ..core.llm import LLMClient
from .metrics import exact_match, f1_score, llm_judge

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = (
    "You are answering questions about a past conversation. Use ONLY the "
    "retrieved memories below. If the memories do not contain the answer, "
    'say "I don\'t know". Answer as concisely as possible.'
)


def answer_question(
    memory: BaseMemory, llm: LLMClient, benchmark: Benchmark, sample, top_k: int
) -> tuple[str, list[str]]:
    query = benchmark.retrieve_query(sample, llm=llm)
    retrieved = memory.retrieve(query, top_k=top_k)
    context = "\n".join(f"- {memory.render(m)}" for m in retrieved) or "(no memories)"

    prompt = benchmark.qa_prompt(sample, context)
    if prompt is None:
        pred = llm.chat(
            QA_SYSTEM_PROMPT, f"Memories:\n{context}\n\nQuestion: {sample.question}"
        )
    else:
        pred = llm.chat("You must respond with a JSON object.", prompt, json_mode=True)
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
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Ingest each conversation into `memory`, then QA-evaluate it.

    Writes one JSON object per line: {question_id, conv_id, question, gold,
    pred, category, retrieved_ids, ...scores}.

    `cache_dir`: if set and the memory implements save()/load(), the memory
    state after ingesting each conversation is cached, so later runs (e.g.
    with a different QA protocol) skip the expensive ingestion phase.
    """
    records: list[dict[str, Any]] = []
    out_f = open(output_path, "w") if output_path else None

    conversations = list(benchmark.conversations())
    if max_conversations is not None:
        conversations = conversations[:max_conversations]

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for conv in tqdm(conversations, desc=f"[{benchmark.name}] {memory.name}"):
        cache_path = None
        if cache_dir and hasattr(memory, "save") and hasattr(memory, "load"):
            cache_path = (
                Path(cache_dir) / f"{benchmark.name}_{memory.name}_{conv.conv_id}.pkl"
            )

        if cache_path and cache_path.exists():
            memory.load(cache_path)
            logger.info("loaded memory cache %s", cache_path)
        else:
            memory.reset()
            for session in conv.sessions:
                for turn in session.turns:
                    memory.add(
                        benchmark.format_turn(turn),
                        metadata={"session_id": session.session_id,
                                  "timestamp": session.timestamp},
                    )
            if cache_path:
                memory.save(cache_path)
                logger.info("saved memory cache %s", cache_path)

        samples = benchmark.qa_samples(conv.conv_id)
        if max_questions_per_conv is not None:
            samples = samples[:max_questions_per_conv]

        for sample in samples:
            pred, retrieved_ids = answer_question(memory, llm, benchmark, sample, top_k)
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
