#!/usr/bin/env python3
"""Unified experiment entry point.

Example:
    python scripts/run_experiment.py \
        --benchmark locomoco --data data/locomoco/locomo10.json \
        --method rag --config configs/baselines/rag.yaml \
        --output results/rag_locomo.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_memory.benchmarks import get_benchmark
from agent_memory.core.embeddings import get_embedder
from agent_memory.core.llm import get_llm
from agent_memory.evaluation.runner import run_benchmark
from agent_memory.methods import get_method


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True,
                   help="locomoco | longmemeval | memoryagentbench | gatemem")
    p.add_argument("--data", required=True,
                   help="dataset file, or a domain directory for gatemem")
    p.add_argument("--method", required=True, help="full_context | rag | amem | ours | ...")
    p.add_argument("--config", default=None, help="YAML config for the method")
    p.add_argument("--output", default=None, help="JSONL output path")
    p.add_argument("--llm", default=None, help="override llm backend (openai | dummy)")
    p.add_argument("--embedder", default=None,
                   help="override embedder (openai | hash | sentence_transformers)")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-conversations", type=int, default=None)
    p.add_argument("--max-questions", type=int, default=None,
                   help="max questions per conversation")
    p.add_argument("--llm-judge", action="store_true",
                   help="add generic LLM-as-judge score (QA benchmarks only)")
    p.add_argument("--cache-dir", default=None,
                   help="cache per-conversation memory state (amem supports save/load); "
                        "later runs skip ingestion")
    p.add_argument("--memory-answer", action="store_true",
                   help="use the memory system's own answer() protocol "
                        "(mem0 / memoryr1) instead of the benchmark QA prompts")
    p.add_argument("--source", default=None,
                   help="memoryagentbench only: keep one metadata.source subset")
    p.add_argument("--chunk-size", type=int, default=4096,
                   help="memoryagentbench only: context chunk size (~tokens)")
    args = p.parse_args()

    config = yaml.safe_load(open(args.config)) if args.config else {}
    if args.top_k is not None:
        config["top_k"] = args.top_k

    llm = get_llm(args.llm or config.get("llm", "openai"),
                  **config.get("llm_kwargs", {}))
    embedder = get_embedder(args.embedder or config.get("embedder", "openai"),
                            **config.get("embedder_kwargs", {}))
    memory = get_method(args.method, config, llm=llm, embedder=embedder)

    bench_kwargs = {}
    if args.benchmark == "memoryagentbench":
        bench_kwargs = {"source": args.source, "chunk_size": args.chunk_size}
    benchmark = get_benchmark(args.benchmark, args.data, **bench_kwargs)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    custom = benchmark.run(
        memory, llm,
        top_k=int(config.get("top_k", 10)),
        max_episodes=args.max_conversations,
        output_path=args.output,
    )
    if custom is not None:
        print(json.dumps(custom, indent=2, ensure_ascii=False))
        return

    summary = run_benchmark(
        benchmark,
        memory,
        llm,
        top_k=int(config.get("top_k", 10)),
        max_conversations=args.max_conversations,
        max_questions_per_conv=args.max_questions,
        use_llm_judge=args.llm_judge,
        output_path=args.output,
        cache_dir=args.cache_dir,
        use_memory_answer=args.memory_answer,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
