"""vLLM engine helpers: batch generation with logprobs, and reference prompt-logprobs."""

from __future__ import annotations

import os
from typing import Dict, List

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def create_engine(model_path: str, gpu: int = 0, gpu_memory_utilization: float = 0.25,
                  max_model_len: int = 8192, max_num_seqs: int = 256,
                  enforce_eager: bool = False):
    from vllm import LLM
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,
    )
    return llm


def batch_generate(llm, prompts: List[str], max_tokens: int = 2048, temperature: float = 1.0,
                   logprobs: bool = False) -> List[Dict]:
    """Returns list of dicts: {text, token_ids, logprobs(list[float] | None)}"""
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature > 0 else 1.0,
        max_tokens=max_tokens,
        logprobs=0 if logprobs else None,
    )
    outs = llm.generate(prompts, sp)
    results = []
    for out in outs:
        o = out.outputs[0]
        lp = None
        if logprobs and o.logprobs is not None:
            lp = []
            for token_id, lp_dict in zip(o.token_ids, o.logprobs):
                entry = lp_dict.get(token_id)
                lp.append(entry.logprob if entry is not None else -20.0)
        results.append({"text": o.text, "token_ids": list(o.token_ids), "logprobs": lp})
    return results


def batch_chat_generate(llm, tokenizer, messages_list: List[List[Dict]], max_tokens: int = 1024,
                        temperature: float = 0.0) -> List[str]:
    """For the frozen instruct model: plain chat generation, returns texts."""
    from vllm import SamplingParams
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        for m in messages_list
    ]
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outs]


def batch_ref_logprobs(llm, sequences: List[Dict[str, List[int]]], batch_size: int = 64) -> List[List[float]]:
    """Per-token logprobs of response tokens given the prompt, via prompt_logprobs.

    sequences: [{"prompt_ids": [...], "response_ids": [...]}]
    Returns list of per-response-token logprob lists.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    results: List[List[float]] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        prompts = [TokensPrompt(prompt_token_ids=s["prompt_ids"] + s["response_ids"]) for s in chunk]
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        outs = llm.generate(prompts, sp)
        for s, out in zip(chunk, outs):
            plp = out.prompt_logprobs
            p_len = len(s["prompt_ids"])
            lps = []
            for pos in range(p_len, p_len + len(s["response_ids"])):
                tok = s["response_ids"][pos - p_len]
                entry = plp[pos] if pos < len(plp) else None
                if entry is None or tok not in entry:
                    lps.append(-20.0)
                else:
                    lps.append(entry[tok].logprob)
            results.append(lps)
    return results
