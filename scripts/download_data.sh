#!/usr/bin/env bash
# Download agent-memory benchmarks into data/.
# Set https_proxy/http_proxy first if you need a proxy.
set -euo pipefail
mkdir -p data/locomoco data/longmemeval data/memoryagentbench data/halumem data/memalpha

# --- LoCoMo (main benchmark of the A-MEM paper) ---
# Source: https://github.com/snap-research/locomo
if [ ! -f data/locomoco/locomo10.json ]; then
  echo "Downloading LoCoMo..."
  curl -L -o data/locomoco/locomo10.json \
    https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
fi

# --- LongMemEval ---
# Source: https://github.com/xiaowu0162/LongMemEval (data on HuggingFace;
# files have no .json extension upstream). longmemeval_m is 2.7GB; oracle and s
# are the commonly used ones.
for split in longmemeval_oracle longmemeval_s; do
  if [ ! -f "data/longmemeval/${split}.json" ]; then
    echo "Downloading LongMemEval ${split}..."
    curl -L -o "data/longmemeval/${split}.json" \
      "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/${split}"
  fi
done

# --- MemoryAgentBench ---
# Source: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
for split in Accurate_Retrieval Conflict_Resolution Long_Range_Understanding Test_Time_Learning; do
  f="data/memoryagentbench/${split}.parquet"
  if [ ! -f "$f" ]; then
    echo "Downloading MemoryAgentBench ${split}..."
    curl -L -o "$f" \
      "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/${split}-00000-of-00001.parquet"
  fi
done
if [ ! -f data/memoryagentbench/entity2id.json ]; then
  curl -L -o data/memoryagentbench/entity2id.json \
    "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/entity2id.json"
fi

# --- GateMem ---
# Source: https://github.com/rzhub/GateMem (data lives inside the repo)
if [ ! -d data/gatemem/medical ]; then
  echo "Downloading GateMem data (sparse checkout of bench/data)..."
  tmp=$(mktemp -d)
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/rzhub/GateMem "$tmp/gatemem"
  (cd "$tmp/gatemem" && git sparse-checkout set bench/data)
  mkdir -p data/gatemem
  cp -r "$tmp/gatemem/bench/data/." data/gatemem/
  rm -rf "$tmp"
fi

# --- HaluMem (operation-level memory hallucination) ---
# Source: https://huggingface.co/datasets/IAAR-Shanghai/HaluMem
for split in HaluMem-Medium HaluMem-Long; do
  if [ ! -f "data/halumem/${split}.jsonl" ]; then
    curl -L -o "data/halumem/${split}.jsonl" \
      "https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/${split}.jsonl"
  fi
done

# --- Mem-alpha train/test pool used by TrustMem ---
# Source: https://huggingface.co/datasets/YuWangX/Memalpha
for split in train test; do
  if [ ! -f "data/memalpha/${split}.parquet" ]; then
    curl -L -o "data/memalpha/${split}.parquet" \
      "https://huggingface.co/datasets/YuWangX/Memalpha/resolve/main/${split}.parquet"
  fi
done

# --- HotpotQA (AgeMem 评测) ---
# Source: HF hotpotqa/hotpot_qa (distractor validation, 7405 题)
if [ ! -f data/hotpotqa/distractor_validation.parquet ]; then
  mkdir -p data/hotpotqa
  curl -L -o data/hotpotqa/distractor_validation.parquet \
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
fi

echo "Done. Files:"
ls -lhR data/ | head -40
