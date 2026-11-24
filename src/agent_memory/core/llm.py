from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    """Minimal chat-completion interface used across the codebase."""

    def chat(self, system: str, user: str, **kwargs) -> str: ...


class OpenAIClient:
    """Works with OpenAI and any OpenAI-compatible endpoint.

    Configure via env vars: OPENAI_API_KEY, OPENAI_BASE_URL (optional),
    OPENAI_MODEL (default: gpt-4o-mini).
    """

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI()

    def chat(self, system: str, user: str, **kwargs) -> str:
        request = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=kwargs.get("temperature", 0.0),
        )
        if kwargs.get("json_mode"):
            request["response_format"] = {"type": "json_object"}
        if kwargs.get("max_tokens"):
            request["max_tokens"] = kwargs["max_tokens"]
        resp = self._client.chat.completions.create(**request)
        return resp.choices[0].message.content or ""


class DummyLLM:
    """Deterministic stand-in for smoke tests without API access."""

    def chat(self, system: str, user: str, **kwargs) -> str:
        return "dummy-answer"


class TransformersLLM:
    """Local HuggingFace causal LM (e.g. Qwen3), loaded once and reused.

    Config kwargs: model (local path or repo id), max_new_tokens, device_map.
    Generation is greedy (do_sample=False) for reproducible JSON output;
    `json_mode` relies on prompt + robust downstream parsing.
    """

    def __init__(self, model: str, max_new_tokens: int = 512,
                 device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, dtype="auto", device_map=device_map
        )
        self._model.eval()
        self.max_new_tokens = max_new_tokens

    def chat(self, system: str, user: str, **kwargs) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=int(kwargs.get("max_tokens", self.max_new_tokens)),
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True)


def get_llm(name: str = "openai", **kwargs) -> LLMClient:
    if name == "openai":
        return OpenAIClient(**kwargs)
    if name == "dummy":
        return DummyLLM()
    if name in ("hf", "transformers"):
        return TransformersLLM(**kwargs)
    raise ValueError(f"unknown llm: {name}")
