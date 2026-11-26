"""Transition-Ranked GRPO (TR-GRPO) trainer.

Implements TrustMem Sec. 3.3-3.4:
- Multi-granularity reward  R = R_task + R_act + 0.1*R_ver + 0.05*R_eff + 0.1*R_type
- GRPO with group-normalized advantages per (instance, chunk-step)
- Verifier-guided preference pairs (top-1/bottom-1, margin delta) + DPO-style
  length-normalized ranking loss
- KL regularization against a frozen reference policy (actor-side auxiliary loss)

Hardware-friendly design for 1 training GPU:
- Reference/old logprobs are produced by vLLM engines (prompt_logprobs) so the
  training phase only holds the policy on GPU.
- AdamW states + fp32 master weights live on CPU.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .rewards import compute_task_score
from .rollout import RolloutRecord, TransitionRecord

# Reward weights (paper Sec. 4.1: R = R_task + R_act + 0.1 R_ver + 0.05 R_eff + 0.1 R_type)
W_ACT = 1.0
W_VER = 0.1
W_EFF = 0.05
W_TYPE = 0.1

# TR-GRPO hyper-parameters (paper Sec. 4.1)
KL_COEF = 0.001
LR = 1e-6
CLIP_EPS = 0.2
ADV_EPS = 1e-4
RANK_MARGIN = 0.15
RANK_LAMBDA = 0.05
RANK_BETA = 0.1
MAX_PAIRS_PER_BATCH = 256


# ---------------------------------------------------------------------------
# Reward assignment
# ---------------------------------------------------------------------------

def assign_rewards(records: List[RolloutRecord], llm_judge: Optional[Callable] = None):
    """Compute R_task per rollout and the combined per-transition reward (Eq. 1)."""
    for rec in records:
        scores = []
        for question, pred, answer in getattr(rec, "qa_results", []):
            scores.append(compute_task_score(rec.data_source, pred, answer, question, llm_judge))
        rec.r_task = sum(scores) / len(scores) if scores else 0.0
        for t in rec.transitions:
            t.reward = (rec.r_task + W_ACT * t.r_act + W_VER * t.r_ver
                        + W_EFF * rec.r_eff + W_TYPE * t.r_type)


def compute_advantages(records: List[RolloutRecord]):
    """Group-normalized advantage per (instance, chunk-step) across the G rollouts."""
    by_it: Dict[Tuple[int, int], List[TransitionRecord]] = {}
    for rec in records:
        for t in rec.transitions:
            by_it.setdefault((rec.instance_idx, t.chunk_idx), []).append(t)
    for group in by_it.values():
        rs = [t.reward for t in group]
        mu = sum(rs) / len(rs)
        sigma = (sum((r - mu) ** 2 for r in rs) / len(rs)) ** 0.5
        for t in group:
            t.advantage = (t.reward - mu) / (sigma + ADV_EPS)


def build_preference_pairs(records: List[RolloutRecord], margin: float = RANK_MARGIN,
                           max_pairs: int = MAX_PAIRS_PER_BATCH) -> List[Tuple[TransitionRecord, TransitionRecord]]:
    """Top-1/bottom-1 verifier-scored transitions per (instance, chunk-step)."""
    by_it: Dict[Tuple[int, int], List[TransitionRecord]] = {}
    for rec in records:
        for t in rec.transitions:
            by_it.setdefault((rec.instance_idx, t.chunk_idx), []).append(t)
    pairs = []
    for group in by_it.values():
        best = max(group, key=lambda t: t.r_ver)
        worst = min(group, key=lambda t: t.r_ver)
        if best.r_ver - worst.r_ver >= margin and best is not worst:
            pairs.append((best, worst))
        if len(pairs) >= max_pairs:
            break
    return pairs


# ---------------------------------------------------------------------------
# CPU-offloaded AdamW
# ---------------------------------------------------------------------------

class CPUAdamW:
    def __init__(self, params: List[torch.nn.Parameter], lr: float = LR,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.masters = [p.detach().to("cpu", torch.float32) for p in self.params]
        self.m = [torch.zeros_like(mp) for mp in self.masters]
        self.v = [torch.zeros_like(mp) for mp in self.masters]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p, master, m, v in zip(self.params, self.masters, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad.detach().to("cpu", torch.float32)
            p.grad = None
            if self.weight_decay:
                master.mul_(1 - self.lr * self.weight_decay)
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            master.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)
            p.data.copy_(master.to(p.device, p.dtype, non_blocking=True))
        if torch.cuda.is_available() and any(p.is_cuda for p in self.params):
            torch.cuda.synchronize()

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# Policy log-prob helpers
# ---------------------------------------------------------------------------

def response_logprobs(model, input_ids: torch.Tensor, prompt_len: int,
                      tokenizer_pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (per-token logprobs of response tokens, response mask).
    input_ids: [1, L]; response occupies positions prompt_len:."""
    out = model(input_ids=input_ids, logits_to_keep=input_ids.shape[1] - prompt_len + 1)
    logits = out.logits[0].float()  # [resp_len+1, vocab]
    logps = F.log_softmax(logits, dim=-1)
    targets = input_ids[0, prompt_len:]
    lp = logps[:-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = (targets != tokenizer_pad_id).float()
    return lp, mask


def k3_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Low-variance (k3) KL estimator per token: exp(r) - r - 1 with r = ref - pi."""
    r = ref_logp - logp
    return torch.expm1(r) - r


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

@dataclass
class TrainStats:
    loss: float = 0.0
    grpo_loss: float = 0.0
    kl: float = 0.0
    rank_loss: float = 0.0
    mean_reward: float = 0.0
    mean_task: float = 0.0
    mean_ver: float = 0.0
    n_pairs: int = 0
    n_seqs: int = 0


def train_step(model, optimizer: CPUAdamW, records: List[RolloutRecord],
               max_seq_len: int = 6144, accum_clip: float = 1.0) -> TrainStats:
    """One TR-GRPO update over all transitions of the batch.

    Expects each TransitionRecord to carry `ref_logprobs` (frozen reference policy).
    """
    device = next(model.parameters()).device
    stats = TrainStats()
    transitions = [t for rec in records for t in rec.transitions if len(t.response_ids) > 1]
    pairs = build_preference_pairs(records)
    stats.n_pairs = len(pairs)
    stats.n_seqs = len(transitions)
    stats.mean_reward = sum(t.reward for t in transitions) / max(len(transitions), 1)
    stats.mean_task = sum(r.r_task for r in records) / max(len(records), 1)
    stats.mean_ver = sum(t.r_ver for t in transitions) / max(len(transitions), 1)

    model.gradient_checkpointing_enable()
    model.train()
    optimizer.zero_grad()

    # ---- GRPO + KL over all transitions ----
    n = max(len(transitions), 1)
    for t in transitions:
        prompt_ids, resp_ids = t.prompt_ids, t.response_ids
        total = len(prompt_ids) + len(resp_ids)
        if total > max_seq_len:  # left-truncate prompt
            keep = max_seq_len - len(resp_ids)
            prompt_ids = prompt_ids[-keep:]
        input_ids = torch.tensor([prompt_ids + resp_ids], device=device)
        lp, mask = response_logprobs(model, input_ids, len(prompt_ids), model.config.eos_token_id)

        ref_lp = torch.tensor(t.ref_logprobs, device=device, dtype=torch.float32)
        old_lp = torch.tensor(t.old_logprobs[: len(resp_ids)], device=device, dtype=torch.float32)
        L = min(len(lp), len(ref_lp), len(old_lp))
        lp, ref_lp, old_lp, mask = lp[:L], ref_lp[:L], old_lp[:L], mask[:L]

        ratio = torch.exp(lp - old_lp)
        adv = t.advantage
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
        grpo = -torch.sum(torch.min(surr1, surr2) * mask) / torch.sum(mask).clamp(min=1.0)
        kl = torch.sum(k3_kl(lp, ref_lp) * mask) / torch.sum(mask).clamp(min=1.0)
        loss = (grpo + KL_COEF * kl) / n
        loss.backward()
        stats.grpo_loss += grpo.item() / n
        stats.kl += kl.item() / n

    # ---- Transition-ranking loss over preference pairs ----
    npairs = max(len(pairs), 1)
    for t_pos, t_neg in pairs:
        seqs = []
        for t in (t_pos, t_neg):
            prompt_ids, resp_ids = t.prompt_ids, t.response_ids
            if len(prompt_ids) + len(resp_ids) > max_seq_len:
                prompt_ids = prompt_ids[-(max_seq_len - len(resp_ids)):]
            seqs.append((prompt_ids, resp_ids))
        ells, ref_ells = [], []
        for t, (prompt_ids, resp_ids) in zip((t_pos, t_neg), seqs):
            input_ids = torch.tensor([prompt_ids + resp_ids], device=device)
            lp, mask = response_logprobs(model, input_ids, len(prompt_ids), model.config.eos_token_id)
            ell = torch.sum(lp * mask) / torch.sum(mask).clamp(min=1.0)  # length-normalized
            ells.append(ell)
            ref_lp = t.ref_logprobs
            Lr = min(len(ref_lp), int(mask.sum().item()))
            ref_ells.append(sum(ref_lp[:Lr]) / max(Lr, 1))
        delta_theta = ells[0] - ells[1]
        delta_ref = ref_ells[0] - ref_ells[1]
        rank = -F.logsigmoid(RANK_BETA * (delta_theta - delta_ref))
        (RANK_LAMBDA * rank / npairs).backward()
        stats.rank_loss += rank.item() / npairs

    # gradient clipping
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().float().norm().item() ** 2
    total_norm = math.sqrt(total_norm)
    if total_norm > accum_clip:
        scale = accum_clip / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    optimizer.step()
    optimizer.zero_grad()
    stats.loss = stats.grpo_loss + KL_COEF * stats.kl + RANK_LAMBDA * stats.rank_loss
    return stats
