"""Chunk-by-chunk memory-construction rollout with vLLM, plus QA over terminal memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .memory_core import Memory, count_tokens, execute_function_call, get_memory_tool_schemas
from .prompts import UNIFIED_CHUNK_PROMPT, get_query_prompt


@dataclass(eq=False)
class TransitionRecord:
    instance_idx: int
    rollout_idx: int
    chunk_idx: int
    prompt_ids: List[int]
    response_ids: List[int]
    old_logprobs: List[float]
    response_text: str
    actions: List[Dict[str, Any]]
    pre_memory: str
    post_memory: str
    chunk_text: str
    r_act: float = 0.0
    r_type: float = 0.0
    r_ver: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    ref_logprobs: List[float] = field(default_factory=list)


@dataclass(eq=False)
class RolloutRecord:
    instance_idx: int
    rollout_idx: int
    data_source: str
    transitions: List[TransitionRecord] = field(default_factory=list)
    r_task: float = 0.0
    r_eff: float = 0.0
    final_memory: Optional[Memory] = None


# ---------------------------------------------------------------------------
# Tool-call parsing (Qwen3 native <tool_call> format + robust fallbacks)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FN_RE = re.compile(r"✿FUNCTION✿:\s*([^\n]+)\s*✿ARGS✿:\s*(.+?)(?=✿|$)", re.DOTALL)


def _extract_json_obj(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(response):
        try:
            obj = json.loads(m.group(1))
            calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
        except json.JSONDecodeError:
            js = _extract_json_obj(m.group(1))
            if js:
                try:
                    obj = json.loads(js)
                    calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
                except json.JSONDecodeError:
                    pass
    if not calls:
        # fallback: qwen raw-token format
        for m in _FN_RE.finditer(response):
            calls.append({"name": m.group(1).strip(), "arguments": m.group(2).strip()})
    return calls


def cut_response(response: str) -> str:
    """Keep text up to the last completed tool call (mirrors Mem-alpha post-processing)."""
    idx = response.rfind("</tool_call>")
    if idx >= 0:
        return response[: idx + len("</tool_call>")]
    return response


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

class MemoryRollout:
    """Runs the memory-construction loop for a batch of instances with G rollouts each."""

    def __init__(self, tokenizer, policy_generate: Callable, max_response_length: int = 2048,
                 max_prompt_length: int = 6144, temperature: float = 1.0):
        self.tokenizer = tokenizer
        self.policy_generate = policy_generate  # (prompts: list[str], max_tokens, temperature, logprobs) -> list[dict]
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.temperature = temperature

    def build_chunk_prompt(self, chunk_text: str, memory: Memory) -> str:
        query = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
        messages = memory.render_system_prompt(status="memorie", query=query,
                                               max_num_of_recent_chunks=Memory.MAX_MEMORY_ITEMS)
        user = UNIFIED_CHUNK_PROMPT.format(context=chunk_text,
                                           max_new_tokens=int(self.max_response_length * 0.8))
        messages.append({"role": "user", "content": user})
        tools = get_memory_tool_schemas(memory)
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        return text

    def run(self, instances: List[Dict[str, Any]], num_rollouts: int,
            max_chunks_per_instance: Optional[int] = None) -> List[RolloutRecord]:
        """instances: list of dicts with keys chunks, data_source. Returns B*G RolloutRecords."""
        B, G = len(instances), num_rollouts
        memories = [Memory(including_core=True) for _ in range(B * G)]
        records = [RolloutRecord(instance_idx=i, rollout_idx=g,
                                 data_source=instances[i]["data_source"])
                   for i in range(B) for g in range(G)]
        chunk_lists = [inst["chunks"] for inst in instances for _ in range(G)]
        max_chunks = max(len(c) for c in chunk_lists)
        if max_chunks_per := max_chunks_per_instance:
            max_chunks = min(max_chunks, max_chunks_per)

        for chunk_idx in range(max_chunks):
            active = [k for k in range(B * G) if chunk_idx < len(chunk_lists[k])]
            if not active:
                break
            prompts, pre_snapshots = [], []
            for k in active:
                mem = memories[k]
                pre_snapshots.append(mem.snapshot())
                p = self.build_chunk_prompt(chunk_lists[k][chunk_idx], mem)
                # left-truncate overlong prompts (keep the tail with the chunk)
                ids = self.tokenizer(p, add_special_tokens=False)["input_ids"]
                if len(ids) > self.max_prompt_length:
                    ids = ids[-self.max_prompt_length:]
                    p = self.tokenizer.decode(ids)
                prompts.append(p)

            outs = self.policy_generate(prompts, max_tokens=self.max_response_length,
                                        temperature=self.temperature, logprobs=True)
            for k, prompt, out, pre_snap in zip(active, prompts, outs, pre_snapshots):
                mem = memories[k]
                response_text = cut_response(out["text"])
                calls = parse_tool_calls(response_text)
                actions, n_success = [], 0
                for call in calls:
                    result = execute_function_call(mem, call["name"], call.get("arguments"))
                    success = "executed successfully" in result and "'status': 'skipped'" not in result
                    n_success += success
                    actions.append({"name": call["name"], "arguments": call.get("arguments"),
                                    "result": result[:500], "success": success})
                rec = records[k]
                t = TransitionRecord(
                    instance_idx=rec.instance_idx, rollout_idx=rec.rollout_idx, chunk_idx=chunk_idx,
                    prompt_ids=self.tokenizer(prompt, add_special_tokens=False)["input_ids"],
                    response_ids=out["token_ids"], old_logprobs=out["logprobs"],
                    response_text=response_text, actions=actions,
                    pre_memory=pre_snap, post_memory=mem.snapshot(),
                    chunk_text=chunk_lists[k][chunk_idx],
                    r_act=(n_success / len(calls)) if calls else 0.0,
                )
                rec.transitions.append(t)

        for k, rec in enumerate(records):
            rec.final_memory = memories[k]
            total_chunk_len = sum(count_tokens(c) for c in chunk_lists[k][:max_chunks])
            rec.r_eff = 1.0 - memories[k].total_length() / max(total_chunk_len, 1)
        return records


# ---------------------------------------------------------------------------
# QA over terminal memory with a frozen answer model
# ---------------------------------------------------------------------------

def answer_questions(records: List[RolloutRecord], instances: List[Dict[str, Any]],
                     answer_generate: Callable, tokenizer, max_questions: int = 10,
                     top_k: int = 20) -> None:
    """Fill r_task in-place. answer_generate(prompts, max_tokens, temperature) -> list[str]"""
    prompts, index = [], []
    for rec in records:
        inst = instances[rec.instance_idx]
        qa_list = inst["questions_and_answers"][:max_questions]
        qp = get_query_prompt(rec.data_source)
        for j, qa in enumerate(qa_list):
            q = qa["question"]
            if qp:
                q = qp + "\n\n" + q
            mem = rec.final_memory
            retrieved_sem = mem.memory_search("semantic", qa["question"], top_k=top_k)
            retrieved_epi = mem.memory_search("episodic", qa["question"], top_k=top_k)
            mem._retrieved_semantic = retrieved_sem
            mem._retrieved_episodic = retrieved_epi
            messages = mem.render_system_prompt(status="chat", query=qa["question"])
            messages.append({"role": "user", "content": q})
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
            prompts.append(text)
            index.append((rec, qa))
    outs = answer_generate(prompts, max_tokens=1024, temperature=0.0)
    preds = {}
    for (rec, qa), out in zip(index, outs):
        preds.setdefault(rec, []).append((qa["question"], out, qa["answer"]))
    for rec, items in preds.items():
        rec.qa_results = items  # for later scoring
