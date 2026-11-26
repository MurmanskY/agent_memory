"""Reward components for TrustMem reproduction.

R_task : downstream QA accuracy, per data source (ported from Mem-alpha reward manager)
R_eff  : 1 - |M_T| / sum|c_t|          (memory compactness)
R_act  : fraction of successfully executed tool calls per transition
R_type : fraction of written/updated contents judged valid by the analyzer LLM
R_ver  : Memory Transition Verifier score in [0, 1]
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter

from .prompts import ACCURACY_PROMPT, LME_JUDGE_PROMPT, TYPE_ANALYZER_PROMPTS, VERIFIER_PROMPT


# ---------------------------------------------------------------------------
# Text normalization / F1 (DensePhrases-style, as in Mem-alpha)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    import string as _s
    text = text.lower()
    text = "".join(c for c in text if c not in _s.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str):
    np_, ng = normalize_answer(prediction), normalize_answer(ground_truth)
    special = {"yes", "no", "noanswer"}
    if (np_ in special or ng in special) and np_ != ng:
        return 0.0
    pt, gt = np_.split(), ng.split()
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(pt), n / len(gt)
    return 2 * p * r / (p + r)


def substring_exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def _extract_answer_from_response(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else text.strip()


def strip_think(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[1].strip()
    if "<think>" in text:
        return "Empty"
    return text


# ---------------------------------------------------------------------------
# Task reward per data source (ported from Mem-alpha verl reward manager)
# ---------------------------------------------------------------------------

def compute_task_score(data_source: str, predicted_answer: str, gold_answer: str,
                       question: str = "", llm_judge=None) -> float:
    """llm_judge: callable(prompt:str)->str, used for lme_train and squad/hotpotqa generative reward."""
    pred = strip_think(predicted_answer)

    if data_source == "booksum":
        keywords = [x.strip() for x in str(gold_answer).split(",")]
        hit = sum(1 for k in keywords if k.lower() in pred.lower())
        return hit / max(len(keywords), 1)

    if data_source == "pubmed-rct" or "ttl" in data_source or "icl" in data_source:
        extracted = _extract_answer_from_response(pred).strip('"\'').strip()
        if not re.match(r"^\d+$", extracted):
            return 0.0
        return 1.0 if extracted == str(gold_answer).strip('"\'').strip() else 0.0

    if data_source == "lme_train":
        if llm_judge is None:
            return float(substring_exact_match(pred, str(gold_answer)))
        resp = llm_judge(LME_JUDGE_PROMPT.format(question, gold_answer, pred)).strip().lower()
        return 1.0 if ("yes" in resp and "no" not in resp) else 0.0

    if data_source == "perltqa":
        gold = str(gold_answer)
        if ";" in gold:
            parts = gold.split(";")
            return sum(1 for a in parts if a.lower().strip() in pred.lower()) / len(parts)
        return 1.0 if gold.lower() in pred.lower() else 0.0

    if data_source in ("squad", "hotpotqa"):
        answer_text = str(gold_answer)
        return 1.0 if answer_text.lower() in pred.lower() else 0.0

    # default: substring match on normalized text
    return float(substring_exact_match(pred, str(gold_answer)))


# ---------------------------------------------------------------------------
# Content-type analyzer (R_type) and Transition Verifier (R_ver)
# ---------------------------------------------------------------------------

def _extract_json(text: str):
    """Extract the first balanced JSON object from text."""
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
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


class Analyzer:
    """Wraps an LLM generate() callable: Analyzer.ask(list_of_prompts) -> list_of_str."""

    def __init__(self, generate_fn):
        self.generate = generate_fn

    def analyze_contents(self, items):
        """items: list of (memory_type, content). Returns list of 0/1 validity scores."""
        prompts = []
        index = []
        for i, (mtype, content) in enumerate(items):
            if mtype not in TYPE_ANALYZER_PROMPTS or not content:
                continue
            prompts.append(TYPE_ANALYZER_PROMPTS[mtype]
                           + f"\n\nAnalyze this {mtype} memory content:\n\n{content}")
            index.append(i)
        scores = [1.0] * len(items)
        if prompts:
            outs = self.generate(prompts, max_tokens=512, temperature=0.1)
            for i, out in zip(index, outs):
                parsed = _extract_json(out)
                scores[i] = 1.0 if (parsed and parsed.get("VALID")) else 0.0
        return scores

    def verify_transitions(self, transitions):
        """transitions: list of dicts with chunk_text, pre_memory, post_memory, actions.
        Returns list of floats in [0, 1]."""
        prompts = [VERIFIER_PROMPT.format(
            chunk_text=t["chunk_text"][:8000],
            pre_memory=t["pre_memory"][:4000],
            post_memory=t["post_memory"][:8000],
            actions=t["actions"][:4000],
        ) for t in transitions]
        outs = self.generate(prompts, max_tokens=512, temperature=0.0)
        scores = []
        for out in outs:
            parsed = _extract_json(out)
            if parsed is None or "score" not in parsed:
                scores.append(0.5)  # unparseable -> neutral
                continue
            try:
                s = float(parsed["score"])
            except (TypeError, ValueError):
                s = 0.5
            scores.append(min(max(s, 0.0), 1.0))
        return scores, outs
