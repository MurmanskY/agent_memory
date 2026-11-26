"""TrustMem reproduction: TR-GRPO training driver.

Subcommands (each phase runs in its own process so GPU memory is reliably freed):
  driver  - orchestrate the full training loop
  rollout - policy rollouts (memory construction) on the rollout GPU
  reward  - QA + verifier + content-type analysis + reference logprobs on the service GPU
  train   - TR-GRPO policy update on the training GPU

Example (smoke):
  python scripts/train_trustmem.py driver --train-instances 16 --steps 3 --rollouts 8
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
DATA_DIR = Path(os.environ.get("TRUSTMEM_DATA_DIR", str(ROOT / "data" / "memalpha")))
WORK_DIR = ROOT / "results" / "trustmem"

BASE_MODEL = os.environ.get("TRUSTMEM_BASE_MODEL", str(ROOT / "models" / "Qwen3-4B"))
INSTRUCT_MODEL = os.environ.get(
    "TRUSTMEM_INSTRUCT_MODEL", str(ROOT / "models" / "Qwen3-4B-Instruct-2507")
)

ROLLOUT_GPU = int(os.environ.get("TRUSTMEM_ROLLOUT_GPU", "1"))
SERVICE_GPU = int(os.environ.get("TRUSTMEM_SERVICE_GPU", "0"))
REF_GPU = int(os.environ.get("TRUSTMEM_REF_GPU", "1"))
TRAIN_GPU = int(os.environ.get("TRUSTMEM_TRAIN_GPU", "1"))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_instances(split: str, n: int, seed: int = 0, balanced: bool = True):
    import pandas as pd
    df = pd.read_parquet(DATA_DIR / f"{split}.parquet")
    rows = []
    for _, r in df.iterrows():
        chunks = r["chunks"]
        qa = r["questions_and_answers"]
        if isinstance(chunks, str):
            chunks = json.loads(chunks)
        if isinstance(qa, str):
            qa = json.loads(qa)
        rows.append({"instance_id": int(r["instance_id"]), "chunks": chunks,
                     "questions_and_answers": qa, "data_source": r["data_source"]})
    if n and n < len(rows):
        if balanced:
            # round-robin over data sources for a balanced subset (paper: balanced 562)
            by_src = {}
            for row in rows:
                by_src.setdefault(row["data_source"], []).append(row)
            picked, idx = [], {k: 0 for k in by_src}
            import random
            rng = random.Random(seed)
            for v in by_src.values():
                rng.shuffle(v)
            while len(picked) < n and any(idx[k] < len(v) for k, v in by_src.items()):
                for k in sorted(by_src):
                    if idx[k] < len(by_src[k]) and len(picked) < n:
                        picked.append(by_src[k][idx[k]])
                        idx[k] += 1
            rows = picked
        else:
            rows = rows[:n]
    return rows


# ---------------------------------------------------------------------------
# phase: rollout
# ---------------------------------------------------------------------------

def phase_rollout(args):
    from transformers import AutoTokenizer
    from agent_memory.security.trustmem.rollout import MemoryRollout
    from agent_memory.security.trustmem.vllm_utils import batch_generate, create_engine

    instances = load_instances(args.split, args.n, seed=args.seed)
    with open(args.batch_file, "rb") as f:
        batch = pickle.load(f)  # list of indices into instances
    instances = [instances[i] for i in batch]

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    llm = create_engine(args.ckpt, gpu=ROLLOUT_GPU, gpu_memory_utilization=0.55,
                        max_model_len=8704)

    def policy_generate(prompts, max_tokens, temperature, logprobs):
        return batch_generate(llm, prompts, max_tokens=max_tokens,
                              temperature=temperature, logprobs=logprobs)

    runner = MemoryRollout(tokenizer, policy_generate,
                           max_response_length=args.max_response_length,
                           temperature=args.temperature)
    records = runner.run(instances, num_rollouts=args.rollouts,
                         max_chunks_per_instance=args.max_chunks)
    out = {"instances": instances, "records": records}
    with open(args.rollout_file, "wb") as f:
        pickle.dump(out, f)
    print(f"[rollout] saved {len(records)} rollouts to {args.rollout_file}", flush=True)


# ---------------------------------------------------------------------------
# phase: reward
# ---------------------------------------------------------------------------

def phase_reward(args):
    from transformers import AutoTokenizer
    from agent_memory.security.trustmem.rewards import Analyzer
    from agent_memory.security.trustmem.rollout import answer_questions
    from agent_memory.security.trustmem.tr_grpo import assign_rewards, compute_advantages, build_preference_pairs
    from agent_memory.security.trustmem.vllm_utils import batch_chat_generate, batch_ref_logprobs, create_engine

    with open(args.rollout_file, "rb") as f:
        data = pickle.load(f)
    instances, records = data["instances"], data["records"]

    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)
    instruct = create_engine(INSTRUCT_MODEL, gpu=SERVICE_GPU, gpu_memory_utilization=0.40,
                             max_model_len=16384, enforce_eager=True)
    ref = create_engine(BASE_MODEL, gpu=SERVICE_GPU, gpu_memory_utilization=0.35,
                        max_model_len=8192, enforce_eager=True)

    def instruct_generate(prompts, max_tokens=512, temperature=0.0):
        return batch_chat_generate(instruct, tokenizer,
                                   ([{"role": "user", "content": p}] for p in prompts),
                                   max_tokens=max_tokens, temperature=temperature)

    # ---- QA with frozen answer model ----
    t0 = time.time()
    answer_questions(records, instances,
                     lambda prompts, max_tokens, temperature: batch_chat_generate(
                         instruct, tokenizer, ([{"role": "user", "content": p}] for p in prompts),
                         max_tokens=max_tokens, temperature=temperature),
                     tokenizer, max_questions=args.max_questions)
    print(f"[reward] QA done in {time.time()-t0:.1f}s", flush=True)

    # ---- R_type: content-type analysis of written/updated contents ----
    analyzer = Analyzer(instruct_generate)
    items, tindex = [], []
    for rec in records:
        for t in rec.transitions:
            for a in t.actions:
                if not a["success"]:
                    continue
                name, ar = a["name"], a.get("arguments") or {}
                if isinstance(ar, str):
                    try:
                        ar = json.loads(ar)
                    except Exception:
                        continue
                mtype = (ar.get("memory_type") or "").replace("_memory", "")
                content = ar.get("content") or ar.get("new_content") or ""
                if name in ("new_memory_insert", "memory_update") and mtype and content:
                    items.append((mtype, str(content)))
                    tindex.append(t)
    scores = analyzer.analyze_contents(items) if items else []
    per_t = {}
    for t, s in zip(tindex, scores):
        per_t.setdefault(id(t), []).append(s)
    n_typed = 0
    for rec in records:
        for t in rec.transitions:
            if id(t) in per_t:
                t.r_type = sum(per_t[id(t)]) / len(per_t[id(t)])
                n_typed += 1
    print(f"[reward] content-type analysis: {len(items)} contents over {n_typed} transitions", flush=True)

    # ---- R_ver: Memory Transition Verifier ----
    transitions = [t for rec in records for t in rec.transitions]
    vinputs = [{"chunk_text": t.chunk_text, "pre_memory": t.pre_memory,
                "post_memory": t.post_memory,
                "actions": json.dumps([{"name": a["name"], "arguments": a["arguments"],
                                        "result": a["result"][:200]} for a in t.actions],
                                      ensure_ascii=False, default=str)}
               for t in transitions]
    t0 = time.time()
    vscores, _raw = analyzer.verify_transitions(vinputs)
    for t, s in zip(transitions, vscores):
        t.r_ver = s
    print(f"[reward] verifier done in {time.time()-t0:.1f}s, mean R_ver={sum(vscores)/max(len(vscores),1):.3f}", flush=True)

    # ---- rewards, advantages, preference pairs ----
    assign_rewards(records, llm_judge=lambda p: instruct_generate([p])[0])
    compute_advantages(records)
    pairs = build_preference_pairs(records)
    print(f"[reward] {len(pairs)} preference pairs", flush=True)

    # ---- reference logprobs (stored on the transition objects) ----
    seqs, seq_refs = [], []
    for rec in records:
        for t in rec.transitions:
            prompt_ids, resp_ids = t.prompt_ids, t.response_ids
            if len(prompt_ids) + len(resp_ids) > args.max_seq_len:
                prompt_ids = prompt_ids[-(args.max_seq_len - len(resp_ids)):]
            seqs.append({"prompt_ids": prompt_ids, "response_ids": resp_ids})
            seq_refs.append(t)
    t0 = time.time()
    ref_lps = batch_ref_logprobs(ref, seqs)
    for t, lp in zip(seq_refs, ref_lps):
        t.ref_logprobs = lp
    print(f"[reward] ref logprobs done in {time.time()-t0:.1f}s", flush=True)

    out = {"instances": instances, "records": records}
    with open(args.reward_file, "wb") as f:
        pickle.dump(out, f)
    print(f"[reward] saved to {args.reward_file}", flush=True)


# ---------------------------------------------------------------------------
# phase: train
# ---------------------------------------------------------------------------

def phase_train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(TRAIN_GPU)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agent_memory.security.trustmem.tr_grpo import CPUAdamW, train_step

    with open(args.reward_file, "rb") as f:
        data = pickle.load(f)
    records = data["records"]

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.cuda()
    model.config.use_cache = False

    optimizer = CPUAdamW(model.parameters(), lr=args.lr)

    stats = train_step(model, optimizer, records, max_seq_len=args.max_seq_len)

    os.makedirs(args.out_ckpt, exist_ok=True)
    model.save_pretrained(args.out_ckpt)
    AutoTokenizer.from_pretrained(args.ckpt).save_pretrained(args.out_ckpt)
    print(f"[train] stats: {stats}", flush=True)
    with open(args.stats_file, "w") as f:
        f.write(json.dumps(stats.__dict__))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def phase_driver(args):
    os.makedirs(args.work_dir, exist_ok=True)
    ckpt = BASE_MODEL
    instances = load_instances("train", args.train_instances, seed=args.seed)
    all_batch = list(range(len(instances)))
    batch_file = os.path.join(args.work_dir, "batch.pkl")
    with open(batch_file, "wb") as f:
        pickle.dump(all_batch, f)

    history = []
    for step in range(args.steps):
        print(f"===== STEP {step} (ckpt={ckpt}) =====", flush=True)
        rollout_file = os.path.join(args.work_dir, f"step{step}_rollout.pkl")
        reward_file = os.path.join(args.work_dir, f"step{step}_reward.pkl")
        stats_file = os.path.join(args.work_dir, f"step{step}_stats.json")
        out_ckpt = os.path.join(args.work_dir, f"ckpt_step{step+1}")

        if not (os.path.exists(rollout_file) and args.resume):
            _run(["rollout", "--split", "train", "--n", str(args.train_instances), "--seed", str(args.seed),
                  "--batch-file", batch_file, "--ckpt", ckpt, "--rollouts", str(args.rollouts),
                  "--max-chunks", str(args.max_chunks), "--max-response-length", str(args.max_response_length),
                  "--temperature", str(args.temperature), "--rollout-file", rollout_file])
        else:
            print(f"[driver] reusing existing {rollout_file}", flush=True)
        if not (os.path.exists(reward_file) and args.resume):
            _run(["reward", "--rollout-file", rollout_file, "--reward-file", reward_file,
                  "--max-questions", str(args.max_questions), "--max-seq-len", str(args.max_seq_len)])
        else:
            print(f"[driver] reusing existing {reward_file}", flush=True)
        if not (os.path.exists(stats_file) and os.path.isdir(out_ckpt) and args.resume):
            _run(["train", "--reward-file", reward_file, "--ckpt", ckpt, "--out-ckpt", out_ckpt,
                  "--lr", str(args.lr), "--max-seq-len", str(args.max_seq_len), "--stats-file", stats_file])
        else:
            print(f"[driver] reusing existing {out_ckpt}", flush=True)

        with open(stats_file) as f:
            stats = json.load(f)
        history.append({"step": step, **stats})
        print(f"[driver] step {step}: {stats}", flush=True)
        ckpt = out_ckpt

    with open(os.path.join(args.work_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[driver] done. final ckpt: {ckpt}", flush=True)


def _run(argv):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(ROOT / "scripts" / "train_trustmem.py")] + argv
    print(f"[driver] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--split", default="train")
        sp.add_argument("--n", type=int, default=16)
        sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("rollout")
    common(sp)
    sp.add_argument("--batch-file", required=True)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--rollouts", type=int, default=8)
    sp.add_argument("--max-chunks", type=int, default=5)
    sp.add_argument("--max-response-length", type=int, default=2048)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--rollout-file", required=True)

    sp = sub.add_parser("reward")
    sp.add_argument("--rollout-file", required=True)
    sp.add_argument("--reward-file", required=True)
    sp.add_argument("--max-questions", type=int, default=5)
    sp.add_argument("--max-seq-len", type=int, default=6144)

    sp = sub.add_parser("train")
    sp.add_argument("--reward-file", required=True)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--out-ckpt", required=True)
    sp.add_argument("--lr", type=float, default=1e-6)
    sp.add_argument("--max-seq-len", type=int, default=6144)
    sp.add_argument("--stats-file", required=True)

    sp = sub.add_parser("driver")
    sp.add_argument("--train-instances", type=int, default=16)
    sp.add_argument("--steps", type=int, default=3)
    sp.add_argument("--rollouts", type=int, default=8)
    sp.add_argument("--max-chunks", type=int, default=5)
    sp.add_argument("--max-questions", type=int, default=5)
    sp.add_argument("--max-response-length", type=int, default=2048)
    sp.add_argument("--max-seq-len", type=int, default=6144)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--lr", type=float, default=1e-6)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--resume", action="store_true",
                    help="reuse existing per-step rollout/reward artifacts when possible")
    sp.add_argument("--work-dir", default=str(WORK_DIR / "smoke"))

    args = p.parse_args()
    if args.cmd == "driver":
        phase_driver(args)
    elif args.cmd == "rollout":
        phase_rollout(args)
    elif args.cmd == "reward":
        phase_reward(args)
    elif args.cmd == "train":
        phase_train(args)


if __name__ == "__main__":
    main()
