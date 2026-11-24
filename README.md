# Agent Memory Lab

面向 LLM Agent 长期记忆（long-term memory）的研究代码库。目标是在统一的接口与评测
流程下，对比主流 memory baseline（RAG、A-MEM 等）与我们自己的方法（`methods/ours`）。

参考论文：[`papers/AMem.pdf`](papers/AMem.pdf) — *A-Mem: Agentic Memory for LLM Agents*
（Zettelkasten 风格的动态记忆组织），已在 `baselines/amem.py` 复现。

## 目录结构

```
├── papers/                  # 相关论文
├── models/                  # 本地模型（不入库）：Qwen3-4B-Instruct-2507、all-MiniLM-L6-v2
├── configs/                 # 实验配置 (YAML)：baselines/ 与 ours/
├── data/                    # 数据集（不入库，见 scripts/download_data.sh）
│   ├── locomoco/            # LoCoMo
│   ├── longmemeval/         # LongMemEval
│   ├── memoryagentbench/    # MemoryAgentBench (parquet)
│   └── gatemem/             # GateMem 四个 domain
├── results/                 # 实验输出（不入库）
├── scripts/
│   ├── run_experiment.py    # 统一实验入口
│   └── download_data.sh     # 数据集下载
├── src/agent_memory/
│   ├── core/                # 记忆基类、数据结构、LLM/Embedding 客户端
│   ├── baselines/           # full_context / rag / amem / memgpt / langmem
│   ├── methods/ours/        # 我们的方法（框架占位）
│   ├── benchmarks/          # LoCoMo、LongMemEval、MemoryAgentBench、GateMem
│   └── evaluation/          # 指标 (F1/EM/LLM-judge/官方模板) 与 runner
└── tests/
```

## 快速开始

```bash
pip install -e ".[dev,bench,st]"     # bench: pyarrow; st: sentence-transformers
bash scripts/download_data.sh        # 下载全部数据集到 data/（需要代理时先 export https_proxy=...）

# 本地 base model（默认配置即此组合，模型在 models/ 下，GPU 运行）：
#   LLM: Qwen3-4B-Instruct-2507（llm: hf，TransformersLLM 后端）
#   Embedding: all-MiniLM-L6-v2（论文同款）
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
    --benchmark locomoco \
    --data data/locomoco/locomo10.json \
    --method amem \
    --config configs/baselines/amem.yaml \
    --output results/amem_locomo.jsonl
```

也可以改用 OpenAI 兼容端点：`--llm openai` + 环境变量 `OPENAI_API_KEY` /
`OPENAI_BASE_URL` / `OPENAI_MODEL`（如本地 vLLM 起的其他模型）。
无 API key / 无 GPU 时可用 `--llm dummy --embedder hash` 跑通全流程（仅冒烟测试）。

## Baselines

| 名称 | 状态 | 说明 |
|---|---|---|
| `full_context` | ✅ | 不做记忆管理，全部历史塞进上下文（上限截断） |
| `rag` | ✅ | Naive RAG：embedding 余弦相似度 top-k 检索 |
| `amem` | ✅ | A-MEM 复现（`baselines/amem.py`）：note 构建（LLM 生成 keywords/context/tags）→ 邻居链接 + 记忆演化（单次 LLM 决策 strengthen / update_neighbor）→ top-k 检索 + 链接扩展。prompt 与官方 repo 逐字一致，默认 evo_threshold=100、k_neighbors=5 |
| `memgpt` | 🚧 骨架 | MemGPT（分页式记忆层级），待实现 |
| `langmem` | 🚧 骨架 | LangGraph LangMem，待封装 |

新增 baseline：继承 `agent_memory.core.base.BaseMemory`，实现 `add/retrieve/reset`
（可选 `render` 自定义检索结果在 QA prompt 中的呈现），并在
`agent_memory/baselines/__init__.py` 的 `BASELINES` 注册。

## Benchmarks

| 名称 | `--benchmark` | 数据 | 评测协议 |
|---|---|---|---|
| LoCoMo | `locomoco` | `data/locomoco/locomo10.json`（已下载） | 通用 runner：ingest → QA → F1/EM（可选 --llm-judge） |
| LongMemEval | `longmemeval` | `data/longmemeval/longmemeval_oracle.json`（已下载；`_s` 见下载脚本） | 官方 LLM-judge 模板（按 question_type 分模板，`_abs` 走 abstention 模板） |
| MemoryAgentBench | `memoryagentbench` | `data/memoryagentbench/<Split>.parquet`（TTL 已下载） | context 按 chunk_size 分块注入 → 多问题作答 → exact_match / substring_exact_match（按 metadata.source 映射，与官方一致；recsys/infbench 子集暂跳过） |
| GateMem | `gatemem` | `data/gatemem/<domain>/`（medical/office/education/household，已下载） | 自定义增量协议：逐 turn 注入、在 checkpoint 边界提问，要求 JSON {action, answer}；官方规则打分（action 匹配 / utility include 覆盖 / privacy & deletion 泄漏检测），汇总 U、A、F、OR、MGS=U·(1−A)·(1−F) |

MemoryAgentBench 示例：

```bash
python scripts/run_experiment.py --benchmark memoryagentbench \
    --data data/memoryagentbench/Test_Time_Learning.parquet \
    --source icl_banking77_5900shot_balance --chunk-size 4096 \
    --method amem --config configs/baselines/amem.yaml
```

GateMem 示例（`--data` 指向 domain 目录）：

```bash
python scripts/run_experiment.py --benchmark gatemem \
    --data data/gatemem/education --method amem \
    --config configs/baselines/amem.yaml --output results/amem_gatemem_edu.jsonl
```

注意：GateMem 的 LLM-judge（论文主指标）暂未接入，当前为官方规则辅助打分。

## Roadmap

1. 用 GPT-4o-mini + all-MiniLM-L6-v2 跑 LoCoMo，对齐 A-MEM 论文 Table 1 的数字；
2. 实现 `methods/ours`（接口已就位，可直接对比）；
3. GateMem 接入 LLM-judge；MemoryAgentBench 补 recsys (Recall@5) 与 infbench (LLM-F1) 子集；
4. 成本/延迟统计（token usage、检索耗时）纳入统一报告。
