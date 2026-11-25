"""AgeMem: Agentic Memory with unified LTM/STM tool management (reproduction).

Paper: papers/AgeMen.pdf — Yu et al., "Agentic Memory: Learning Unified
Long-Term and Short-Term Memory Management for Large Language Model Agents"
(ACL 2026). Official code: https://github.com/y1y5/AgeMem (Trinity-RFT training
pipeline + AgeMem_code_agentscope standalone agent).

Reproduction scope:
  - The inference framework: a ReAct-style agent with 6 memory tools
    (Summary_context / Clear_context / Retrieve_memory / Add_memory /
    Update_memory / Delete_memory), prompts and tool schemas verbatim from the
    official repo (trinity/common/workflows/memory_context/).
  - Evaluated on HotpotQA (distractor): Stage-1 casual chat over the context
    (agent stores facts into LTM), then Stage-3 formal QA.
  - The three-stage progressive RL training (step-wise GRPO) is NOT included:
    it needs the Trinity-RFT stack + released checkpoints do not exist. This
    implementation is the no-RL variant (cf. the paper's w/o-RL ablations).
  - Local models only: Qwen3-4B policy + local embedder for the memory store
    (official uses DashScope text-embedding-v4).
"""
from .agent import AgeMemAgent

__all__ = ["AgeMemAgent"]
