"""AgeMem agent: ReAct loop + 6 memory tools (port of the official workflow).

Prompts verbatim from y1y5/AgeMem:
  trinity/common/workflows/memory_context/workflow_prompt.py
  trinity/common/workflows/memory_context/utils.py (TOOL_SCHEMA)
Tool semantics follow eval_hotpotQA.py (_apply_tools).
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

import numpy as np

from ...core.embeddings import Embedder, cosine_topk
from ...core.llm import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts (verbatim from the official repo)
# ---------------------------------------------------------------------------

TOOL_CALL_SYS_PROMPT = """
You are an intelligent assistant that solves complex problems by managing context and memory with tools when needed.

## Available Tools:
{tools}

## Problem-Solving Workflow

You must follow a structured reasoning and action process for every task:

1. **Think & Plan**
   Always start with a <think>...</think> block.
   Inside it, explain your reasoning, plan your next step, and decide whether you need to call a tool or provide a final answer.

2. **Tool Calls**
   If you decide to use one or more tools, follow your <think> block with a <tool_call>...</tool_call> block.
   - You may call **one or multiple tools** in a single step.
   - List multiple tool calls as elements of a JSON array.
   - Each tool call must include `"name"` and `"arguments"`.
   - Example:
     <tool_call>[{{"name": "Retrieve_memory", "arguments": {{"query": "math problem solving strategies", "top_k": 3}}}}, {{"name": "Add_memory", "arguments": {{"content": "Strategy summary for reuse", "memory_type": "problem_solving"}}}}]</tool_call>

3. **Final Answer**
   When you no longer need tools and are ready to present your final output, follow your last <think> block with an <answer>...</answer> block containing the full response.

4. **Mutual Exclusivity Rule**
   After **each <think> block**, you must choose exactly **one** of the following:
   - a `<tool_call>` block (if you need tools), **or**
   - an `<answer>` block (if you are ready to respond).
   You must **never** include both `<tool_call>` and `<answer>` immediately after the same `<think>` block.

5. **Iterative Solving**
   You may repeat this sequence as needed:
   `<think>` → `<tool_call>` → `<think>` → `<tool_call>` … → `<think>` → `<answer>`
   until the problem is completely solved.

## Response Format (Strict)
Your full output must follow these rules:
- Every reasoning step must appear inside <think> tags.
- Every tool usage must appear inside one <tool_call> tag (even if it includes multiple tool invocations).
- The final solution must be wrapped in <answer> tags.
- No text should appear outside these tags.

## Example

<think>I need to solve a math problem. Let’s first recall strategies for similar problems, then I’ll store my chosen approach for future reference.</think>
<tool_call>[{{"name": "Retrieve_memory", "arguments": {{"query": "math problem solving strategies", "top_k": 3}}}}, {{"name": "Add_memory", "arguments": {{"content": "Solution approach for this type of problem", "memory_type": "problem_solving"}}}}]</tool_call>

<think>Now I’ve reviewed past strategies and applied one successfully. I can present the final result.</think>
<answer>The answer is 42.</answer>

## Guidelines
- Always start with reasoning (<think>).
- After each reasoning step, decide: call tool(s) or answer.
- You can call multiple tools within one <tool_call> JSON array.
- Be concise, logical, and explicit in reasoning.
- Manage memory actively: retrieve, add, update, summarize, filter, or delete as needed.
- Use <answer> only once when the final solution is ready.
"""

SUMMARY_CONTEXT_SYS_PROMPT = """
You are a conversation summarization assistant.
Your goal is to compress the given conversation span into a concise summary that preserves all important information, intentions, decisions, and unresolved questions.
The summary will later be used to replace the original conversation in the context, so make sure nothing essential is lost.

Instructions:
1. Read the provided conversation rounds carefully.
2. Identify the main topics, actions, results, and open issues.
3. Write a clear, factual summary in natural language.
4. Do NOT include greetings, filler text, or redundant phrasing.

Input:
- Conversation content:
###
{conversation_text}
###

Output:
- A concise yet comprehensive summary of the above conversation span.

Let's start the conversation summarization.
"""

TEXT_SIMILARITY_SYS_PROMPT = """
You are a text similarity assistant.
Your goal is to calculate the similarity between two texts.

Instructions:
1. Read the two texts carefully.
2. Calculate the similarity between the two texts.
3. Output the similarity score (a number between 0 and 1).

Input:
- Text 1:
###
{text1}
###
- Text 2:
###
{text2}
###

Output:
- The similarity score (a number between 0 and 1) between the two texts.
- ONLY output the similarity score, no other text or explanation.
"""

TOOL_SCHEMA: list[dict] = [
    {
        "name": "Summary_context",
        "description": "Summarizes previous conversation rounds in the context to reduce token usage while preserving important information. Should be called when context length approaches token limits or when summarizing specific conversation spans.",
        "parameters": {
            "type": "dict",
            "properties": {
                "span": {
                    "description": "The range of conversation rounds to summarize. Can be 'all' for entire context, or a number (e.g., '5') for the last N rounds. A system, user, assistant and 'tool' message are considered as one round.\n\nExamples:\n- \"all\": summarize the whole context\n- \"5\": summarize the last 5 rounds",
                    "type": "string",
                }
            },
            "required": ["span"],
        },
    },
    {
        "name": "Clear_context",
        "description": "Removes irrelevant or outdated content from the conversation context to improve task-solving efficiency. Helps maintain focus on current task by filtering out noise.",
        "parameters": {
            "type": "dict",
            "properties": {
                "criteria": {
                    "description": "The criteria for content removal. Can be keywords, phrases, or descriptions of content types to remove (e.g., 'the birthday of John', 'the age of Mary').",
                    "type": "string",
                }
            },
            "required": ["criteria"],
        },
    },
    {
        "name": "Retrieve_memory",
        "description": "Retrieves relevant memories from the external memory store and adds them to the current context. Uses semantic similarity search to find memories that match the query.",
        "parameters": {
            "type": "dict",
            "properties": {
                "query": {
                    "description": "The search query to find relevant memories. Should describe what kind of information or context is needed.",
                    "type": "string",
                },
                "top_k": {
                    "description": "The maximum number of memories to retrieve. Defaults to 3.",
                    "type": "integer",
                },
                "metadata_filter": {
                    "description": "Optional metadata filters to narrow down memory search (e.g., {'type': 'user_info', 'domain': 'math'}).",
                    "type": "object",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "Add_memory",
        "description": "Adds new information to the external memory store. Can store chat information, user preferences, task-solving pipelines, or any other useful knowledge for future reference.",
        "parameters": {
            "type": "dict",
            "properties": {
                "content": {
                    "description": "The content to store in memory. Should be clear, concise, and self-contained for future retrieval.",
                    "type": "string",
                },
                "metadata": {
                    "description": "Optional metadata tags to categorize and filter the memory (e.g., {'type': 'user_preference', 'domain': 'math', 'importance': 'high'}).",
                    "type": "object",
                },
                "memory_type": {
                    "description": "The type of memory being stored. Common types: 'chat_info', 'user_info', 'task_pipeline', 'knowledge', 'preference'.",
                    "type": "string",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "Update_memory",
        "description": "Updates an existing memory in the external memory store. Requires first retrieving relevant memories to identify which one to update. The memory_id should be obtained from retrieved memories.",
        "parameters": {
            "type": "dict",
            "properties": {
                "memory_id": {
                    "description": "The unique identifier of the memory to update. Must be obtained from a previous memory retrieval operation.",
                    "type": "string",
                },
                "content": {
                    "description": "The new content to replace the existing memory content. If not provided, only metadata will be updated.",
                    "type": "string",
                },
                "metadata": {
                    "description": "Updated metadata for the memory. Will be merged with existing metadata.",
                    "type": "object",
                },
            },
            "required": ["memory_id", "content"],
        },
    },
    {
        "name": "Delete_memory",
        "description": "Removes a memory from the external memory store. Requires first retrieving relevant memories to identify which one to delete. The memory_id should be obtained from retrieved memories.",
        "parameters": {
            "type": "dict",
            "properties": {
                "memory_id": {
                    "description": "The unique identifier of the memory to delete. Must be obtained from a previous memory retrieval operation.",
                    "type": "string",
                },
                "confirmation": {
                    "description": "Confirmation that this memory should be permanently deleted. Should be set to true to confirm deletion.",
                    "type": "boolean",
                },
            },
            "required": ["memory_id", "confirmation"],
        },
    },
]

# ---------------------------------------------------------------------------
# Parsing (tolerant, following official utils.py)
# ---------------------------------------------------------------------------


def parse_tool_calls(text: str) -> list[dict]:
    if not text or not text.strip():
        return []
    calls: list[dict] = []
    for m in re.finditer(r"<tool_call>\s*(\[.*?\])\s*</tool_call>", text, re.DOTALL):
        try:
            for c in json.loads(m.group(1)):
                if isinstance(c, dict) and "name" in c:
                    calls.append(c)
        except json.JSONDecodeError:
            continue
    if not calls:  # tolerate unclosed tag
        m = re.search(r"<tool_call>\s*(\[.*)", text, re.DOTALL)
        if m:
            for candidate in (m.group(1), m.group(1) + "]"):
                try:
                    for c in json.loads(candidate):
                        if isinstance(c, dict) and "name" in c:
                            calls.append(c)
                    break
                except json.JSONDecodeError:
                    continue
    # single-object form
    if not calls:
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if m:
            try:
                c = json.loads(m.group(1))
                if "name" in c:
                    calls.append(c)
            except json.JSONDecodeError:
                pass
    seen, out = set(), []
    for c in calls:
        key = json.dumps(c, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def parse_answer(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_score(text: str, default: float = 0.0) -> float:
    if not text:
        return default
    m = re.search(r"(\d+\.?\d*)", text)
    return float(m.group(1)) if m else default


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------


class InMemoryVectorStore:
    """id -> {content, metadata} with an embedding matrix (official equivalent:
    InMemoryVectorStore + text-embedding-v4; here any local Embedder)."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.ids: list[str] = []
        self.contents: list[str] = []
        self.metadatas: list[dict] = []
        self.matrix: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)

    def add(self, memory_id: str, content: str, metadata: dict) -> None:
        vec = self.embedder.embed([content])[0]
        self.ids.append(memory_id)
        self.contents.append(content)
        self.metadatas.append(metadata)
        self.matrix = np.vstack([self.matrix, vec[None, :]])

    def update(self, memory_id: str, content: str | None, metadata: dict) -> bool:
        if memory_id not in self.ids:
            return False
        i = self.ids.index(memory_id)
        if content is not None:
            self.contents[i] = content
            self.matrix[i] = self.embedder.embed([content])[0]
        if metadata:
            self.metadatas[i].update(metadata)
        return True

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self.ids:
            return False
        i = self.ids.index(memory_id)
        del self.ids[i], self.contents[i], self.metadatas[i]
        self.matrix = np.delete(self.matrix, i, axis=0)
        return True

    def retrieve(self, query: str, top_k: int, metadata_filter: dict) -> list[tuple[str, str]]:
        if not self.ids:
            return []
        q = self.embedder.embed([query])[0]
        out = []
        for i, _ in cosine_topk(q, self.matrix, top_k):
            if metadata_filter and not all(
                self.metadatas[i].get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            out.append((self.ids[i], self.contents[i]))
        return out

    def __len__(self) -> int:
        return len(self.ids)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgeMemAgent:
    """ReAct agent with the 6 AgeMem memory/context tools (no-RL variant)."""

    name = "agemem"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
    ):
        from ...core.embeddings import get_embedder
        from ...core.llm import get_llm

        self.config = config or {}
        self.llm = llm or get_llm(
            self.config.get("llm", "openai"), **self.config.get("llm_kwargs", {})
        )
        self.embedder = embedder or get_embedder(
            self.config.get("embedder", "sentence_transformers"),
            **self.config.get("embedder_kwargs", {}),
        )
        self.stage1_max_rounds = int(self.config.get("stage1_max_rounds", 5))
        self.stage3_max_rounds = int(self.config.get("stage3_max_rounds", 5))
        self.max_context_tokens = int(self.config.get("max_context_tokens", 8192))
        self.use_context_tools = bool(self.config.get("use_context_tools", True))
        self.clear_threshold = float(self.config.get("clear_threshold", 0.6))

        tools = TOOL_SCHEMA if self.use_context_tools else [
            t for t in TOOL_SCHEMA
            if t["name"] not in {"Summary_context", "Clear_context", "Retrieve_memory"}
        ]
        self._sys_prompt = TOOL_CALL_SYS_PROMPT.replace(
            "{tools}", json.dumps(tools, indent=2, ensure_ascii=False)
        )
        self.store = InMemoryVectorStore(self.embedder)
        self.context: list[dict] = []
        self.tool_counter: dict[str, int] = {}

    # ------------------------------------------------------------- tool impls

    def _apply_tools(self, tool_calls: list[dict]) -> None:
        for call in tool_calls:
            name = call.get("name")
            args = call.get("arguments", {}) or {}
            self.tool_counter[name] = self.tool_counter.get(name, 0) + 1
            try:
                note = self._apply_one(name, args)
            except Exception as e:
                logger.warning("tool %s failed: %s", name, e)
                note = f"{name}_failed:{e}"
            self.context.append({"role": "tool", "content": note})

    def _apply_one(self, name: str, args: dict) -> str:
        if name == "Add_memory":
            content = str(args.get("content", ""))
            if not content:
                return "[memory tool result]\nmemory_add_failed:empty_content"
            metadata = dict(args.get("metadata") or {})
            memory_type = args.get("memory_type")
            if memory_type:
                metadata["type"] = memory_type
            mem_id = uuid.uuid4().hex[:12]
            self.store.add(mem_id, content, metadata)
            return f"[memory tool result]\nmemory_added:{mem_id}"

        if name == "Update_memory":
            ok = self.store.update(
                str(args.get("memory_id", "")),
                args.get("content"),
                args.get("metadata") if isinstance(args.get("metadata"), dict) else {},
            )
            return f"[memory tool result]\nmemory_updated:{ok}"

        if name == "Delete_memory":
            if not args.get("confirmation"):
                return "[memory tool result]\nmemory_deletion_cancelled:confirmation_required"
            ok = self.store.delete(str(args.get("memory_id", "")))
            return f"[memory tool result]\nmemory_deleted:{ok}"

        if name == "Retrieve_memory":
            items = self.store.retrieve(
                str(args.get("query", "")),
                int(args.get("top_k", 3) or 3),
                args.get("metadata_filter") or {},
            )
            if items:
                block = "\n".join(f"- {c} (Memory ID: {i})" for i, c in items)
                return f"[retrieved memories]\n{block}"
            return "[no related memories found]"

        if name == "Summary_context":
            return self._summary_context(str(args.get("span", "all")))

        if name == "Clear_context":
            return self._clear_context(str(args.get("criteria", "")))

        return f"[tool result]\nunknown_tool:{name}"

    def _summary_target(self, span: str) -> list[int]:
        idx = [i for i, m in enumerate(self.context) if m["role"] != "system"]
        if span == "all":
            return idx
        try:
            return idx[-int(span):]
        except (TypeError, ValueError):
            return idx

    def _summary_context(self, span: str) -> str:
        idx = self._summary_target(span)
        if not idx:
            return "[context tool result]\nsummary_context_applied:no_messages_to_summarize"
        conversation_text = "\n".join(
            f"{self.context[i]['role']}: {self.context[i]['content']}" for i in idx
        )
        summary = self.llm.chat(
            "",
            SUMMARY_CONTEXT_SYS_PROMPT.format(conversation_text=conversation_text),
            max_tokens=512,
        )
        remove = set(idx)
        self.context = [
            m for i, m in enumerate(self.context) if i not in remove
        ]
        self.context.insert(min(idx), {
            "role": "tool",
            "content": f"[summary of {len(idx)} messages]\n{summary}",
        })
        return "[context tool result]\nsummary_context_applied:success"

    def _clear_context(self, criteria: str) -> str:
        kept, removed = [], 0
        for m in self.context:
            if m["role"] == "system" or not criteria:
                kept.append(m)
                continue
            score_text = self.llm.chat(
                "",
                TEXT_SIMILARITY_SYS_PROMPT.format(text1=criteria, text2=m["content"]),
                max_tokens=8,
            )
            if extract_score(score_text) >= self.clear_threshold:
                removed += 1
                continue
            kept.append(m)
        self.context = kept
        return f"[context tool result]\nclear_context_applied:success:removed_{removed}_messages"

    # -------------------------------------------------------------- main loop

    def _run_loop(self, max_rounds: int) -> str | None:
        for _ in range(max_rounds):
            messages = [{"role": "system", "content": self._sys_prompt}] + self.context
            text = self._chat_messages(messages)
            self.context.append({"role": "assistant", "content": text})
            tool_calls = parse_tool_calls(text)
            if tool_calls:
                self._apply_tools(tool_calls)
            answer = parse_answer(text)
            if answer:
                return answer
        return None

    def _chat_messages(self, messages: list[dict]) -> str:
        # Our LLMClient protocol is (system, user); fold the conversation into
        # a single user message, keeping roles explicit.
        system = ""
        lines = []
        for m in messages:
            if m["role"] == "system" and not lines:
                system = m["content"]
            else:
                lines.append(f"{m['role']}: {m['content']}")
        return self.llm.chat(system, "\n\n".join(lines), max_tokens=2048)

    def run_task(self, question: str, context_paragraphs: list[tuple[str, list[str]]]) -> dict:
        """Official eval protocol: Stage-1 casual chat over context (store to
        LTM), then Stage-3 formal QA. Returns {'answer', 'tool_usage', ...}."""
        self.context = []
        self.store = InMemoryVectorStore(self.embedder)
        self.tool_counter = {}

        # ---- Stage 1: casual chat with the context (agent stores facts)
        merged = "\n".join(
            f"{title}: {' '.join(sents[:10])}"
            for title, sents in context_paragraphs
        )
        self.context.append({
            "role": "user",
            "content": "Just chatting about several topics together. "
                       "Here are the related contents grouped by title:\n" + merged,
        })
        self._run_loop(self.stage1_max_rounds)

        # ---- Stage 3: formal QA
        self.context.append({"role": "user", "content": question})
        answer = self._run_loop(self.stage3_max_rounds)
        if answer is None:
            answer = self._run_loop(1) or ""
        return {
            "answer": answer,
            "tool_usage": dict(self.tool_counter),
            "n_memories": len(self.store),
            "n_context_messages": len(self.context),
        }
