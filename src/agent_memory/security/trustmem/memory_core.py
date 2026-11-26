"""Memory store and tool executor, ported from Mem-alpha (memory.py / functions.py).

Simplifications vs. the original:
- BM25-only retrieval (no OpenAI embeddings); embedding bookkeeping removed.
- Same three memory components (core / semantic / episodic) and the same three
  editing tools: new_memory_insert (WRITE), memory_update (REVISE),
  memory_delete (PRUNE) — matching TrustMem's unified action space.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any, Dict, List, Optional

import tiktoken

_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def count_tokens(text) -> int:
    if not isinstance(text, str):
        text = str(text)
    return len(_encoding.encode(text))


class Memory:
    """Holds core, semantic, episodic memories entirely in RAM."""

    MAX_MEMORY_ITEMS = 5  # same as MemoryAgent.MAX_MEMORY_ITEMS in Mem-alpha
    TOPK = 20

    def __init__(self, including_core: bool = False) -> None:
        self.including_core = including_core
        self.core: Optional[str] = "" if including_core else None
        self.semantic: List[Dict[str, str]] = []
        self.episodic: List[Dict[str, str]] = []

    # ---------- rendering ----------
    def _block(self, title: str = "", lines: List[Dict[str, str]] = [], content: str = None) -> str:
        if content is not None:
            return f"<{title}>\n{content}\n</{title}>" if title else content
        if not lines:
            return f"<{title}>\nEmpty.\n</{title}>" if title else "Empty."
        formatted = [f"[{mem_id}] {content}" for mem in lines for mem_id, content in mem.items()]
        body = "\n".join(formatted)
        return f"<{title}>\n{body}\n</{title}>" if title else body

    def render_system_prompt(self, status: str = "memorie", query: str = "",
                             max_num_of_recent_chunks: int = None) -> List[Dict[str, str]]:
        max_n = max_num_of_recent_chunks if max_num_of_recent_chunks is not None else self.MAX_MEMORY_ITEMS
        semantic_items = self.semantic[-max_n:] if max_n < len(self.semantic) else self.semantic
        episodic_items = self.episodic[-max_n:] if max_n < len(self.episodic) else self.episodic

        core_section = f"<core_memory>\n{self.core}\n</core_memory>" if (self.including_core and self.core is not None) else ""

        if status == "memorie":
            memory_type_instructions = []
            if self.including_core:
                memory_type_instructions.append("* core_memory: Information stored so far (stored as a compact paragraph)")
            memory_type_instructions.append("* semantic_memory: General knowledge, factual or conceptual information")
            memory_type_instructions.append("* episodic_memory: Specific personal experiences or events with timestamp (mandatory), place, or context")

            memory_state_sections = []
            if core_section:
                memory_state_sections.append(core_section)
            total_semantic = len(self.semantic)
            memory_state_sections.append(
                f"<semantic_memory> (Only show the most recent {len(semantic_items)} out of {total_semantic} memories)\n"
                f"{self._block(lines=semantic_items)}\n</semantic_memory>")
            total_episodic = len(self.episodic)
            memory_state_sections.append(
                f"<episodic_memory> (Only show the most recent {len(episodic_items)} out of {total_episodic} memories)\n"
                f"{self._block(lines=episodic_items)}\n</episodic_memory>")

            memory_state_text = "\n\n".join(memory_state_sections)
            instructions_text = "\n".join(memory_type_instructions)

            system_prompt = f'''You are a personal assistant with a sophisticated memory system. Your primary task is to carefully analyze, understand, and memorize the information provided by the user.

MEMORIZING MODE INSTRUCTIONS:
- Read and understand all information shared by the user
- Identify key facts, concepts, and relationships
- Store important information using the appropriate memory type:
{instructions_text}
- Use these cues to decide memory type based on content

CURRENT MEMORY STATE:
{memory_state_text}

Focus on understanding and memorizing. Use memory tools actively to store new information.
Since this is the memorization process, if you think all the information has been memorized, you can respond with 'Done'. This information will not be seen by the user.
Meanwhile, you will be queried only once, so make sure to call all the memory insertion functions in one turn.'''
            return [{"role": "system", "content": system_prompt}]

        else:  # status == "chat" — used by the answer model
            semantic_items = getattr(self, "_retrieved_semantic", self.semantic)
            episodic_items = getattr(self, "_retrieved_episodic", self.episodic)
            semantic_section = (
                f"<semantic_memory> (memories retrieved for the query)\n{self._block(lines=semantic_items)}\n</semantic_memory>")
            episodic_section = (
                f"<episodic_memory> (memories retrieved for the query)\n{self._block(lines=episodic_items)}\n</episodic_memory>")
            combined = "\n\n".join(s for s in [core_section, semantic_section, episodic_section] if s)
            system_prompt = (
                "You are a reasoning assistant. For each incoming query or task—whether it's a question, command, or summary request—"
                "use the structured memory below to retrieve and synthesize relevant information to produce your response.\n\n"
                f"Based on the query {query}, the following are the retrieved memories:\n{combined}\n\n"
                "You can directly use the information to answer the query.")
            return [{"role": "system", "content": system_prompt}]

    # ---------- BM25 search ----------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _bm25_search(self, mem_list: List[Dict[str, str]], query: str, top_k: int) -> List[Dict[str, str]]:
        if not mem_list:
            return []
        docs = [self._tokenize(content) for mem in mem_list for _, content in mem.items()]
        q_tokens = self._tokenize(query)
        N = len(docs)
        avgdl = sum(len(d) for d in docs) / max(N, 1)
        # document frequency
        df = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        k1, b = 1.5, 0.75
        scores = []
        for d in docs:
            score = 0.0
            tf = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / max(avgdl, 1e-9)))
            scores.append(score)
        ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)
        return [mem_list[i] for i in ranked[:top_k]]

    def memory_search(self, memory_type: str, query: str, top_k: int = None) -> List[Dict[str, str]]:
        top_k = top_k or self.TOPK
        return self._bm25_search(getattr(self, memory_type), query, top_k)

    # ---------- memory operations ----------
    def _generate_memory_id(self) -> str:
        return str(uuid.uuid4())[:4]

    def _content_exists(self, memory_type: str, content: str) -> bool:
        return any(content in mem.values() for mem in getattr(self, memory_type))

    def new_memory_insert(self, memory_type: str, content: str):
        assert memory_type in ["semantic", "episodic"]
        if self._content_exists(memory_type, content):
            return None
        memory_id = self._generate_memory_id()
        getattr(self, memory_type).append({memory_id: content})
        return {memory_id: content}

    def memory_update(self, memory_type: str, new_content: str, memory_id: str = None):
        assert memory_type in ["core", "semantic", "episodic"]
        if memory_type == "core":
            if not self.including_core:
                raise ValueError("Core memory is not available.")
            if count_tokens(new_content) > 512:
                words = new_content.split()
                while count_tokens(" ".join(words)) > 500 and words:
                    words.pop()
                new_content = " ".join(words) + " [content exceeds 512 tokens, truncated]"
            self.core = new_content
            return self.core
        mem_list = getattr(self, memory_type)
        for i, mem in enumerate(mem_list):
            if memory_id in mem:
                mem_list[i] = {memory_id: new_content}
                return {memory_id: new_content}
        raise ValueError(f"Memory ID {memory_id} not found in {memory_type} memory.")

    def memory_delete(self, memory_type: str, memory_id: str = None):
        assert memory_type in ["core", "semantic", "episodic"]
        if memory_type == "core":
            self.core = ""
            return
        mem_list = getattr(self, memory_type)
        for i, mem in enumerate(mem_list):
            if memory_id in mem:
                mem_list.pop(i)
                return
        raise ValueError(f"Memory ID {memory_id} not found in {memory_type} memory.")

    # ---------- helpers ----------
    def total_length(self) -> int:
        total = count_tokens(self.core) if (self.including_core and self.core) else 0
        for mem_list in (self.semantic, self.episodic):
            for mem in mem_list:
                for content in mem.values():
                    total += count_tokens(content)
        return total

    def snapshot(self) -> str:
        """Compact text view of the full memory state (for the verifier)."""
        parts = []
        if self.including_core and self.core:
            parts.append(f"<core_memory>\n{self.core}\n</core_memory>")
        if self.semantic:
            parts.append(self._block("semantic_memory", self.semantic))
        if self.episodic:
            parts.append(self._block("episodic_memory", self.episodic))
        return "\n\n".join(parts) if parts else "(empty)"

    def clone(self) -> "Memory":
        m = Memory(including_core=self.including_core)
        m.core = self.core
        m.semantic = [dict(x) for x in self.semantic]
        m.episodic = [dict(x) for x in self.episodic]
        return m


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI format; also used by tokenizer.apply_chat_template(tools=...))
# ---------------------------------------------------------------------------

def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def get_memory_tool_schemas(memory: Memory) -> List[Dict[str, Any]]:
    mem_types = ["semantic_memory", "episodic_memory"]
    schemas = [
        _tool(
            "new_memory_insert",
            "Infer a new memory and append it to a memory store. Creates a new memory item with a unique ID. Note: Core memory cannot be inserted, only updated.",
            {
                "memory_type": {"type": "string", "enum": mem_types,
                                "description": "Type of memory to insert: 'semantic' (general knowledge) or 'episodic' (specific experiences)."},
                "content": {"type": "string", "description": "Content of the memory to insert."},
            },
            ["memory_type", "content"],
        ),
        _tool(
            "memory_delete",
            "Delete a memory. For core memory, clears the entire paragraph content. For semantic/episodic memories, deletes the specific memory item by ID.",
            {
                "memory_type": {"type": "string", "enum": mem_types,
                                "description": "Type of memory to delete."},
                "memory_id": {"type": "string", "description": "ID of the memory to delete."},
            },
            ["memory_type", "memory_id"],
        ),
    ]
    update_enum = (["core_memory"] if memory.including_core else []) + mem_types
    schemas.insert(1, _tool(
        "memory_update",
        "Update an existing memory. For core memory, replaces the entire paragraph content. If core memory is empty, then directly write into the core memory. For semantic/episodic memories, updates the specific memory item by ID.",
        {
            "memory_type": {"type": "string", "enum": update_enum,
                            "description": "Type of memory to update."},
            "new_content": {"type": "string", "description": "New **combined** content for the memory."},
            "memory_id": {"type": "string", "description": "ID of the memory to update. Required for semantic/episodic memories, ignored for core memory."},
        },
        ["memory_type", "new_content"],
    ))
    return schemas


def execute_function_call(memory: Memory, name: str, arguments) -> str:
    """Execute one parsed tool call; returns a result string (same convention as Mem-alpha)."""
    try:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        if "memory_type" in arguments and isinstance(arguments["memory_type"], str):
            arguments["memory_type"] = arguments["memory_type"].replace("_memory", "")
        if name == "new_memory_insert":
            result = memory.new_memory_insert(arguments["memory_type"], str(arguments["content"]))
            if result is None:
                return f"[tool {name} executed successfully] → {{'status': 'skipped', 'message': 'Memory content already exists in the memory pool, insertion skipped.'}}"
            return f"[tool {name} executed successfully] → {{'status': 'ok', 'new_memory': {result}}}"
        elif name == "memory_update":
            result = memory.memory_update(arguments["memory_type"], str(arguments["new_content"]), arguments.get("memory_id"))
            return f"[tool {name} executed successfully] → {{'status': 'ok', 'updated_memory': {result}}}"
        elif name == "memory_delete":
            memory.memory_delete(arguments["memory_type"], arguments.get("memory_id"))
            return f"[tool {name} executed successfully] → {{'status': 'ok'}}"
        else:
            return f"[tool {name} error] Unknown function: {name}"
    except Exception as e:
        return f"[tool {name} error] {type(e).__name__}: {e}"
