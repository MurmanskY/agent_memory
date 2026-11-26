"""Adapters between ProMem and the project's local model interfaces."""

from __future__ import annotations

import json
import re

import numpy as np


class LLMAdapter:
    def __init__(self, llm):
        self.llm = llm
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def ask(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        self.n_calls += 1
        return self.llm.chat(
            "You are a careful assistant that follows the requested output format.",
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def ask_many(self, prompts, max_tokens: int = 1024, temperature: float = 0.0):
        return [self.ask(p, max_tokens=max_tokens, temperature=temperature) for p in prompts]


class EmbedderAdapter:
    def __init__(self, embedder):
        self.embedder = embedder

    def encode(self, texts) -> np.ndarray:
        vecs = np.asarray(self.embedder.embed(list(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)


Embedder = EmbedderAdapter
LLMClient = LLMAdapter


class NLI:
    """roberta-large-mnli wrapper; labels are contradiction/neutral/entailment."""

    def __init__(self, path: str = "models/roberta-large-mnli", device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.torch = torch

    def check_batch(self, pairs):
        if not pairs:
            return []
        labels = ["contradiction", "neutral", "entailment"]
        results = []
        with self.torch.no_grad():
            for i in range(0, len(pairs), 32):
                batch = pairs[i:i + 32]
                enc = self.tokenizer(
                    [p[:1400] for p, _ in batch],
                    [h for _, h in batch],
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                results.extend(labels[row] for row in logits.argmax(dim=-1).tolist())
        return results


def extract_json_list(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def extract_json_obj(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def split_lines(text: str):
    items = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if line and not line.startswith(("[", "]", "{", "}")):
            items.append(line)
    return items
