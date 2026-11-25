# Agent Memory Lab

面向 LLM Agent 长期记忆（long-term memory）的研究代码库。目标是在统一的接口与评测
流程下，对比主流 memory baseline（RAG、A-MEM 等）与我们自己的方法（`methods/ours`）。

参考论文：[`papers/AMem.pdf`](papers/AMem.pdf) — *A-Mem: Agentic Memory for LLM Agents*
（Zettelkasten 风格的动态记忆组织），已在 `baselines/amem.py` 复现；
[`papers/MemoryR1.pdf`](papers/MemoryR1.pdf) — *Memory-R1: RL-enhanced memory
management*（Memory Manager + Answer Agent），已在 `baselines/memoryr1.py` 复现推理流程；
[`papers/Mem0.pdf`](papers/Mem0.pdf) — *Mem0: Scalable Long-Term Memory*，
已在 `baselines/mem0.py` 复现（写入侧与 Memory-R1 共用，作答侧为官方 top-10 + 基础 prompt）。

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
| `mem0` | ✅ | Mem0 复现（`baselines/mem0.py`，口径对齐官方 eval @v0.1.100）：写入侧与 Memory-R1 共用（mem0 式 fact 抽取 → ADD/UPDATE/DELETE/NONE Manager，按 speaker 双记忆库）；作答侧为论文口径：每 speaker top-10 检索 + Appendix A 基础 answer prompt（无 distillation、无 `**Answer:**` 指令），原始输出即答案。Mem0g（graph 变体）未实现 |
| `memoryr1` | ✅ | Memory-R1 推理 pipeline 复现（`baselines/memoryr1.py`，prompt 与官方 repo `yansikuan/memory-r1` 逐字一致）：按 speaker 双记忆库；每个 turn 经 mem0 式 fact 抽取 → Memory Manager 决策 ADD/UPDATE/DELETE/NONE（search_k=5）；Answer Agent 每 speaker 检索 top-30（共 60 条）+ Memory Distillation（先输出选中的记忆，再在 `**Answer:**` 后给 ≤5-6 词答案）。注意：论文核心是 RL 微调（PPO/GRPO，VERL，4×H100），官方未发布 checkpoint，本复现等价于论文 Figure 5 的 "BASE"（未微调）配置；训练脚本不在本仓库范围内 |
| `memgpt` | 🚧 骨架 | MemGPT（分页式记忆层级），待实现 |
| `langmem` | 🚧 骨架 | LangGraph LangMem，待封装 |

新增 baseline：继承 `agent_memory.core.base.BaseMemory`，实现 `add/retrieve/reset`
（可选 `render` 自定义检索结果在 QA prompt 中的呈现；可选覆写 `answer` 自定义整个
作答流程，如 Memory-R1 的 Answer Agent），并在
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

## 实验结果

### A-MEM 复现（LoCoMo 全量 1986 题，Qwen3-4B-Instruct-2507 + all-MiniLM-L6-v2，~19h）

评测协议与官方代码对齐（关键词检索、分类别 prompt、对抗题二选一、set-F1/BLEU-1）：

| 类别 | n | F1 | BLEU-1 | 论文 Qwen2.5-3B (F1) | 论文 GPT-4o-mini (F1) |
|---|---|---|---|---|---|
| Multi-hop | 282 | 0.253 | 0.119 | 0.126 | 0.270 |
| Temporal | 321 | 0.331 | 0.203 | 0.276 | 0.459 |
| Open-domain | 96 | 0.087 | 0.033 | 0.071 | 0.121 |
| Single-hop | 841 | 0.324 | 0.218 | 0.172 | 0.447 |
| Adversarial | 446 | 0.747* | 0.745 | 0.279 | 0.500 |
| **总体** | 1986 | **0.398** | **0.311** | — | — |

\* 对抗题按修正口径以 "Not mentioned in the conversation" 为 gold；按论文 legacy 口径
f1_legacy = 0.281（与论文 Qwen2.5-3B 的 0.279 一致），正确拒答率 adversarial_correct = 0.738。

产物：`results/amem_locomo_qwen3_v2.jsonl`（逐题结果）、`results/memory_cache/`
（10 条对话的 A-MEM 记忆缓存，调评测参数可分钟级复跑）。

### SeCom 复现（LoCoMo 全量 1986 题，同一评测协议，~53min）

Qwen3-4B 分割 + LLMLingua-2(0.75) 去噪 + BM25 检索 + 4k token 预算
（`baselines/secom.py`，clean-room 复现）：

| 类别 | F1 | BLEU-1 | A-MEM (F1) | A-MEM (BLEU-1) |
|---|---|---|---|---|
| Multi-hop | 0.165 | 0.061 | **0.253** | **0.119** |
| Temporal | 0.221 | 0.097 | **0.331** | **0.203** |
| Open-domain | 0.087 | 0.043 | 0.087 | 0.033 |
| Single-hop | 0.227 | 0.134 | **0.324** | **0.218** |
| Adversarial | 0.728 | 0.718 | 0.747 | 0.745 |
| **总体** | 0.323 | 0.244 | **0.398** | **0.311** |

对抗题正确拒答率：SeCom 0.720 vs A-MEM 0.738。
结论：A-MEM 在需要跨记忆推理的类别（Multi-hop/Temporal/Single-hop）上明显占优，
与 A-MEM 论文的核心结论一致；SeCom 构建成本则低得多（53min vs 19h），
与论文 Table 5 的效率结论一致。

### Mem0 复现 + 三方法子集对比（conv-26，199 题/方法，快速验证协议）

Mem0（`baselines/mem0.py` + `memoryr1.py` 共享写入路径）：事实抽取 →
ADD/UPDATE/DELETE/NONE 记忆管理器 → 双 speaker bank，prompt 与官方
mem0ai/mem0@v0.1.100 逐字一致；Mem0g（图变体）未包含。

**快速验证协议**：`--max-conversations 1`（conv-26，199 题五类齐全）。
三个方法在该子集的记忆缓存均已建好（`results/memory_cache/`），
迭代重跑分钟级；新方法首轮 ingest：SeCom ~4min、Mem0 ~1.5h、A-MEM ~2h。

| 类别 | A-MEM F1 | SeCom F1 | Mem0 F1 |
|---|---|---|---|
| Multi-hop | **0.223** | 0.137 | 0.101 |
| Temporal | **0.355** | 0.282 | 0.206 |
| Open-domain | **0.130** | 0.119 | 0.078 |
| Single-hop | **0.250** | 0.229 | 0.118 |
| Adversarial | 0.623 | 0.676 | **0.808** |
| **总体** | **0.345** | 0.323 | 0.292 |

对抗题正确拒答率：Mem0 0.787 > SeCom 0.660 > A-MEM 0.617。
A-MEM 强在事实回忆/跨记忆推理，Mem0 强在抗诱导（事实级精炼记忆噪声少）。

### Memory-R1 复现（BASE 配置，conv-26 子集，~16min）

写入路径与 Mem0 共享并复用其缓存（零 ingestion 成本）；答案侧为
Memory-R1 的 Answer Agent：每 speaker top-30（共 60 候选）→ Memory
Distillation → `**Answer:**` 提取（`--memory-answer` 开关）。
论文的 RL 微调（PPO/GRPO）未发布 checkpoint，此处为论文 Figure 5 的
BASE（未微调）配置，backbone Qwen3-4B。

| 类别 | A-MEM F1 | SeCom F1 | Mem0 F1 | MemR1 F1 |
|---|---|---|---|---|
| Multi-hop | **0.223** | 0.137 | 0.101 | 0.185 |
| Temporal | 0.355 | 0.282 | 0.206 | **0.424** |
| Open-domain | 0.130 | 0.119 | 0.078 | 0.107 |
| Single-hop | 0.250 | 0.229 | 0.118 | **0.290** |
| **总体(不含对抗，MemR1 论文口径)** | 0.259 | 0.213 | 0.132 | **0.285** |
| Adversarial（参考） | 0.623 | 0.676 | **0.808** | 0.101 |

结论：(1) 同一块记忆 bank，Mem0 共享协议作答 0.132 → MemR1 Answer
Agent 0.285，蒸馏作答翻倍以上——印证论文 Answer Agent 的消融收益；
(2) MemR1 对抗题拒答率仅 0.128（其 prompt 无拒答机制，论文也将对抗题
排除在评测外），Mem0 抗诱导最强；(3) A-MEM 仍是多跳最强。

### AgeMem 复现（no-RL 变体，HotpotQA distractor 100 题子集，~30min）

AgeMem 是 agent 式方法（记忆操作作为工具调用），评测在 HotpotQA 等任务上。
复现官方推理框架（6 个记忆工具 + ReAct 循环，prompt/schema 与 y1y5/AgeMem
逐字一致）；RL 训练（三阶段 step-wise GRPO）未复现（无发布 checkpoint）。
backbone 与论文一致为 Qwen3-4B 量级（Qwen3-4B-Instruct-2507）。

| 指标 | 我们 (noRL, Qwen3-4B) | 论文 AgeMem-noRL (Qwen3-4B) | 论文 AgeMem+RL |
|---|---|---|---|
| HotpotQA J | 0.770 | 0.545* | 0.555* |
| 工具调用/题（Retrieve/Add/Update） | 3.3 / 1.7 / 0.28 | 4.6 / 2.5 / 0.13 | 4.4 / 2.6 / 0.34 |

\* 论文为 fullwiki 设定 + qwen-max 评审；我们为 distractor 设定（10 段落）+
本地 Qwen3-4B 评审，J 的绝对值不可直接比，仅供量级参考。

工具使用模式与论文 noRL 一致：Retrieve 最频繁，Summary/Filter 几乎不用，
RL 后这些才显著上升——印证论文"RL 教会主动记忆管理"的核心结论。

## Roadmap

1. ~~用本地开源模型跑 LoCoMo，对齐 A-MEM 论文结果~~（已完成，见上）
2. 实现 `methods/ours`（接口已就位，可直接对比）
3. GateMem 接入 LLM-judge；MemoryAgentBench 补 recsys (Recall@5) 与 infbench (LLM-F1) 子集
4. 成本/延迟统计（token usage、检索耗时）纳入统一报告
