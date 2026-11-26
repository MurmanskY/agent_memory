# Agent Memory Lab

面向 LLM Agent 长期记忆（long-term memory）的研究代码库，后续重点关注**记忆系统安全与可信性**：
记忆幻觉、遗漏/污染、错误更新、隐私越权与主动遗忘。目标是在统一接口与评测流程下，
对比主流 memory baseline（RAG、A-MEM 等）、安全记忆方法（TrustMem、ProMem）与我们自己的方法（`methods/ours`）。

参考论文：[`papers/AMem.pdf`](papers/AMem.pdf) — *A-Mem: Agentic Memory for LLM Agents*
（Zettelkasten 风格的动态记忆组织），已在 `baselines/amem.py` 复现；
[`papers/MemoryR1.pdf`](papers/MemoryR1.pdf) — *Memory-R1: RL-enhanced memory
management*（Memory Manager + Answer Agent），已在 `baselines/memoryr1.py` 复现推理流程；
[`papers/Mem0.pdf`](papers/Mem0.pdf) — *Mem0: Scalable Long-Term Memory*，
已在 `baselines/mem0.py` 复现（写入侧与 Memory-R1 共用，作答侧为官方 top-10 + 基础 prompt）；
[`papers/TrustMem.pdf`](papers/TrustMem.pdf) — 可信记忆转移验证与 Transition-Ranked GRPO；
[`papers/ProMem.pdf`](papers/ProMem.pdf) — 主动多粒度记忆抽取与 NLI 幻觉纯化；
[`papers/HaluMem.pdf`](papers/HaluMem.pdf) — 操作级记忆幻觉评测。

## 目录结构

```
├── papers/                  # 相关论文
├── models/                  # 本地模型（不入库）：Qwen3-4B-Instruct-2507、all-MiniLM-L6-v2
├── configs/                 # 实验配置 (YAML)：baselines/ 与 ours/
├── data/                    # 数据集（不入库，见 scripts/download_data.sh）
│   ├── locomoco/            # LoCoMo
│   ├── longmemeval/         # LongMemEval
│   ├── memoryagentbench/    # MemoryAgentBench (parquet)
│   ├── gatemem/             # GateMem 四个 domain（隐私/删除安全）
│   ├── halumem/             # HaluMem Medium/Long（幻觉/遗漏/更新安全）
│   └── memalpha/            # TrustMem 训练与验证池
├── results/                 # 实验输出（不入库）
├── scripts/
│   ├── run_experiment.py    # 统一实验入口（含 HaluMem / ProMem）
│   ├── train_trustmem.py    # TrustMem TR-GRPO 分阶段训练
│   ├── eval_trustmem.py     # 记忆转移错误率 + 任务效用评测
│   └── download_data.sh     # 数据集下载
├── src/agent_memory/
│   ├── core/                # 记忆基类、数据结构、LLM/Embedding 客户端
│   ├── baselines/           # full_context / rag / amem / mem0 / promem 等
│   ├── security/trustmem/   # TrustMem 执行器、Verifier、奖励与 TR-GRPO
│   ├── methods/ours/        # 我们的方法（框架占位）
│   ├── benchmarks/          # 通用 QA + HaluMem / GateMem 安全评测
│   └── evaluation/          # 指标与官方 judge 模板
└── tests/
```

## 快速开始

```bash
pip install -e ".[dev,bench,st]"     # bench: pyarrow; st: sentence-transformers
pip install -e ".[security]"         # ProMem：Transformers + scikit-learn
# TrustMem 训练另需：pip install -e ".[trustmem]"（vLLM）
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
| `promem` | ✅ | ProMem 复现（`baselines/promem/`）：细节/事件/关系多粒度抽取 → 完整性补全 → NLI + LLM 幻觉纯化 → 关系验证与细节锚定；通过 `add_session` 支持 HaluMem 会话级操作评测，也可经统一 `BaseMemory` 用于其他 benchmark |
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
| HaluMem | `halumem` | `data/halumem/HaluMem-{Medium,Long}.jsonl`（已下载） | 官方 judge 模板：抽取 Recall/Weighted-R/Target-P/Accuracy/FMR/F1，更新与 QA 的 Correct/Hallucination/Omission；ProMem 支持全部操作，普通 `BaseMemory` 只报可计算的更新与 QA 指标 |
| HotpotQA | `hotpotqa` | `data/hotpotqa/distractor_validation.parquet` | AgeMem agent 协议 + 本地 LLM judge |

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

HaluMem + ProMem 快速验证（1 用户、1 会话、每会话最多 3 题）：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py \
    --benchmark halumem --data data/halumem/HaluMem-Medium.jsonl \
    --method promem --config configs/baselines/promem.yaml \
    --max-conversations 1 --max-sessions 1 --max-questions 3 \
    --output results/promem_halumem_smoke.jsonl
# 仅 QA（跳过抽取/更新裁判）：--eval-mode qa
```

TrustMem 使用独立的训练/评测入口（策略 rollout 需要 token logprob，不能伪装成普通
`BaseMemory`）。默认是缩小训练：16 instance × 8 rollout × 3 step；产物在 `results/trustmem/`：

```bash
python scripts/train_trustmem.py driver --train-instances 16 --steps 3 --rollouts 8 \
    --max-chunks 5 --max-questions 5 --work-dir results/trustmem/smoke
python scripts/eval_trustmem.py driver --ckpt models/Qwen3-4B --tag base --n 20
python scripts/eval_trustmem.py driver --ckpt models/Qwen3-4B-TrustMem-smoke --tag step3 --n 20
```

TrustMem 的 Verifier 从 coverage / preservation / faithfulness 三个维度检查
`z_t=(c_t,M_{t-1},a_t,M_t)`；TR-GRPO 奖励为
`R_task + R_act + 0.1 R_ver + 0.05 R_eff + 0.1 R_type`，并用验证器分数构造转移级偏好对。
本地 Verifier/judge 为 Qwen3-4B-Instruct-2507，替代论文冻结大模型/GPT-4o-mini；
Mem-α 数据来自 `wangyu-ustc/Mem-alpha`，官方 TrustMem 代码未开源，本实现为论文复现。

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

### TrustMem 缩小复现（迁移自原 TrustMem 项目）

16 个训练实例、8 rollout、3 步；验证为 20 个均衡实例、100 次记忆转移。
基座与训练后模型均为 Qwen3-4B，裁判为本地 Qwen3-4B-Instruct-2507。

| 指标 | Base | TR-GRPO step 3 |
|---|---:|---:|
| Omission rate ↓ | 0.600 | 0.600 |
| Corruption rate ↓ | 0.000 | 0.020 |
| Hallucination rate ↓ | 0.060 | 0.070 |
| Mean task score ↑ | 0.215 | 0.225 |

训练过程 `mean_ver` 从 0.708 升至 0.819、`mean_reward` 从 1.233 升至 1.369。
但独立验证上的幻觉率没有下降，因此这组缩小实验只验证训练与评测链路，**不能作为
TrustMem 提升安全性的结论**；论文完整规模为 32×H100 × 2 天/85 步，此处不可直接比较。
结果文件：`results/trustmem/{history,eval_base,eval_step3}.json`，最终缩小 checkpoint 在
`models/Qwen3-4B-TrustMem-smoke/`（均不入库）。

### ProMem 与 Mem0 的 HaluMem 对照（迁移结果）

原项目已完成的 HaluMem-Medium 同 3 用户、521 道 QA 对照；两方法使用同一本地
Qwen3-4B-Instruct 裁判。数值单位为 %，C/H/O 分别为正确/幻觉/遗漏。

| 指标 | ProMem | Mem0-local |
|---|---:|---:|
| Recall / Weighted R ↑ | 38.86 / 52.95 | **81.30 / 85.19** |
| Target Precision ↑ | **99.70** | 98.14 |
| Accuracy / FMR ↑ | **67.72** / 59.22 | 66.55 / **70.91** |
| Extraction F1 ↑ | 55.93 | **88.93** |
| Updating C / H / O | 31.62 / **0.23** / 68.15 | **78.69** / 2.58 / **18.74** |
| QA C / H / O | **49.90** / **32.05** / 18.04 | 47.79 / 37.24 / **14.97** |

ProMem 的 Target Precision 与 QA 正确率略高，但 NLI 过剪导致召回明显偏低；
该结果支持“更精确”的方向，不支持“全面优于 Mem0”。使用 Qwen3-4B 替代论文的
GPT-4o-mini/Llama3.1-8B、Qwen3-Embedding-0.6B 替代 8B，绝对值不与论文排行榜直接比较。
迁移汇总：`results/promem_vs_mem0_halumem_medium_migrated.json`。

### 整合后 ProMem 全链路验证（HaluMem-Medium 1 用户/1 会话）

统一入口真实运行完成：15 个目标记忆点、11 条系统抽取记忆、3 道 QA，裁判解析失败数为 0。
汇总实现另在原 3 用户评测 artifact 上复核，12/12 项核心指标与官方聚合结果完全一致。

| Recall | Weighted R | Target P | Extraction F1 | QA C / H / O |
|---:|---:|---:|---:|---:|
| 0.600 | 0.667 | 0.955 | 0.737 | 0.667 / 0.333 / 0.000 |

其中 2/3 QA 判为 Correct，另 1 题因回答没有完整纠正“与伴侣在 2025 年已建立项目”
而判为 Hallucination。该会话没有更新或 interference 样本，因此不报告更新与 FMR 指标；
此结果仅用于验证整合后的端到端正确性，不与上面的 3 用户结果直接比较。
产物：`results/promem_halumem_medium_1user_1session.jsonl`、对应 `.log` 与
`results/promem_halumem_medium_1user_1session_summary.json`。

## Roadmap

1. ~~用本地开源模型跑 LoCoMo，对齐 A-MEM 论文结果~~（已完成，见上）
2. 面向记忆幻觉/污染/隐私泄露设计 `methods/ours`（接口已就位，算法暂不实现）
3. 扩大 HaluMem 多用户对照，统一 ProMem/Mem0 摄入协议；补 TrustMem 更大规模训练验证
4. GateMem 接入 LLM-judge；MemoryAgentBench 补 recsys (Recall@5) 与 infbench (LLM-F1) 子集
5. 成本/延迟统计（token usage、检索耗时）纳入统一报告
