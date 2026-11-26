"""TrustMem reproduction: evaluation.

Subcommands:
  driver  - run rollout + judge + QA scoring, print transition error rates
  rollout - memory construction trajectories for the eval split (policy engine)
  judge   - transition-level error judging (omission/corruption/hallucination,
            Appendix D prompt) + task scoring with the frozen instruct model

Example:
  python scripts/eval_trustmem.py driver --ckpt models/Qwen3-4B --n 20 --tag base
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
WORK_DIR = ROOT / "results" / "trustmem"

BASE_MODEL = os.environ.get("TRUSTMEM_BASE_MODEL", str(ROOT / "models" / "Qwen3-4B"))
INSTRUCT_MODEL = os.environ.get(
    "TRUSTMEM_INSTRUCT_MODEL", str(ROOT / "models" / "Qwen3-4B-Instruct-2507")
)
ROLLOUT_GPU = int(os.environ.get("TRUSTMEM_ROLLOUT_GPU", "1"))
SERVICE_GPU = int(os.environ.get("TRUSTMEM_SERVICE_GPU", "0"))

from train_trustmem import load_instances


def phase_rollout(args):
    from transformers import AutoTokenizer
    from agent_memory.security.trustmem.rollout import MemoryRollout
    from agent_memory.security.trustmem.vllm_utils import batch_generate, create_engine

    instances = load_instances(args.split, args.n, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    llm = create_engine(args.ckpt, gpu=ROLLOUT_GPU, gpu_memory_utilization=0.55,
                        max_model_len=8704)

    def policy_generate(prompts, max_tokens, temperature, logprobs):
        return batch_generate(llm, prompts, max_tokens=max_tokens,
                              temperature=temperature, logprobs=logprobs)

    runner = MemoryRollout(tokenizer, policy_generate,
                           max_response_length=args.max_response_length,
                           temperature=args.temperature)
    records = runner.run(instances, num_rollouts=1, max_chunks_per_instance=args.max_chunks)
    with open(args.rollout_file, "wb") as f:
        pickle.dump({"instances": instances, "records": records}, f)
    print(f"[eval-rollout] saved {len(records)} trajectories", flush=True)


def phase_judge(args):
    from transformers import AutoTokenizer
    from agent_memory.security.trustmem.prompts import ERROR_JUDGE_PROMPT
    from agent_memory.security.trustmem.rewards import _extract_json, compute_task_score
    from agent_memory.security.trustmem.rollout import answer_questions
    from agent_memory.security.trustmem.vllm_utils import batch_chat_generate, create_engine

    with open(args.rollout_file, "rb") as f:
        data = pickle.load(f)
    instances, records = data["instances"], data["records"]

    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)
    llm = create_engine(INSTRUCT_MODEL, gpu=SERVICE_GPU, gpu_memory_utilization=0.45,
                        max_model_len=16384, enforce_eager=True)

    def gen(prompts, max_tokens=1024, temperature=0.0):
        return batch_chat_generate(llm, tokenizer, ([{"role": "user", "content": p}] for p in prompts),
                                   max_tokens=max_tokens, temperature=temperature)

    # ---- transition error judging ----
    judge_prompts, index = [], []
    for rec in records:
        inst = instances[rec.instance_idx]
        for t in rec.transitions:
            ops = json.dumps([{"name": a["name"], "arguments": a["arguments"], "result": a["result"][:200]}
                              for a in t.actions], ensure_ascii=False, default=str) or "[]"
            judge_prompts.append(ERROR_JUDGE_PROMPT.format(
                dataset_group=rec.data_source, sample_id=inst["instance_id"],
                chunk_idx=t.chunk_idx, prior_memory=t.pre_memory[:4000],
                input_chunk=t.chunk_text[:8000], memory_operations=ops[:4000]))
            index.append(t)
    outs = gen(judge_prompts, max_tokens=768, temperature=0.0)
    stats = defaultdict(lambda: [0, 0])  # error_type -> [major_count, total]
    per_transition = []
    for t, out in zip(index, outs):
        parsed = _extract_json(out) or {}
        severity = parsed.get("severity", "none")
        major = severity == "major"
        rec_out = {"judge": parsed}
        for et in ("omission", "corruption", "hallucination"):
            stats[et][1] += 1
            if major and parsed.get(et, False):
                stats[et][0] += 1
        per_transition.append(rec_out)

    # ---- task utility (R_task) ----
    answer_questions(records, instances, gen, tokenizer, max_questions=args.max_questions)
    task_scores, per_source = [], defaultdict(list)
    for rec in records:
        scores = [compute_task_score(rec.data_source, pred, ans, q)
                  for q, pred, ans in getattr(rec, "qa_results", [])]
        rec.r_task = sum(scores) / len(scores) if scores else 0.0
        task_scores.append(rec.r_task)
        per_source[rec.data_source].append(rec.r_task)

    result = {
        "ckpt": args.ckpt,
        "n_instances": len(instances),
        "n_transitions": len(index),
        "omission_rate": stats["omission"][0] / max(stats["omission"][1], 1),
        "corruption_rate": stats["corruption"][0] / max(stats["corruption"][1], 1),
        "hallucination_rate": stats["hallucination"][0] / max(stats["hallucination"][1], 1),
        "mean_task_score": sum(task_scores) / max(len(task_scores), 1),
        "per_source_task": {k: sum(v) / len(v) for k, v in per_source.items()},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


def phase_driver(args):
    os.makedirs(args.work_dir, exist_ok=True)
    rollout_file = os.path.join(args.work_dir, f"eval_{args.tag}_rollout.pkl")
    out_file = os.path.join(args.work_dir, f"eval_{args.tag}.json")
    _run(["rollout", "--split", args.split, "--n", str(args.n), "--seed", str(args.seed),
          "--ckpt", args.ckpt, "--max-chunks", str(args.max_chunks),
          "--max-response-length", str(args.max_response_length),
          "--temperature", str(args.temperature), "--rollout-file", rollout_file])
    _run(["judge", "--rollout-file", rollout_file, "--ckpt", args.ckpt,
          "--max-questions", str(args.max_questions), "--out", out_file])


def _run(argv):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_trustmem.py")] + argv
    print(f"[eval-driver] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("rollout")
    sp.add_argument("--split", default="test")
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--max-chunks", type=int, default=5)
    sp.add_argument("--max-response-length", type=int, default=2048)
    sp.add_argument("--temperature", type=float, default=0.7)
    sp.add_argument("--rollout-file", required=True)

    sp = sub.add_parser("judge")
    sp.add_argument("--rollout-file", required=True)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--max-questions", type=int, default=5)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("driver")
    sp.add_argument("--split", default="test")
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--tag", default="eval")
    sp.add_argument("--max-chunks", type=int, default=5)
    sp.add_argument("--max-questions", type=int, default=5)
    sp.add_argument("--max-response-length", type=int, default=2048)
    sp.add_argument("--temperature", type=float, default=0.7)
    sp.add_argument("--work-dir", default=str(WORK_DIR / "smoke"))

    args = p.parse_args()
    {"rollout": phase_rollout, "judge": phase_judge, "driver": phase_driver}[args.cmd](args)


if __name__ == "__main__":
    main()
