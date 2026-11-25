"""SECOM: segment-level memory with compression-based denoising (reproduction).

Paper: papers/SeCom.pdf — Pan et al., "On Memory Construction and Retrieval for
Personalized Conversational Agents" (ICLR 2025). No official code release; this
is a clean-room implementation from the paper.

Pipeline (paper Sec. 2):
  1. Conversation segmentation: an LLM partitions each session into topically
     coherent segments (zero-shot prompt from Appendix A.1, Fig. 6). The paper
     uses GPT-4; their Mistral-7B variant shows lightweight models work — we
     use the local Qwen3-4B by default.
  2. Compression-based denoising: LLMLingua-2 (xlm-roberta-large, compression
     rate 75%) compresses each memory unit; the retrieval index is built over
     the *compressed* text (Eq. 2).
  3. Retrieval: BM25 or MPNet (multi-qa-mpnet-base-dot-v1) over denoised
     segments, selecting by score up to a token budget (4k tokens on LOCOMO,
     ~10 segments); selected segments are reordered chronologically.
  4. Response generation: concatenated segments as context (handled by the
     evaluation runner).

Adaptation note: LoCoMo turns are single utterances (two speakers), not
user-bot exchanges, so the segmentation prompt works over turns.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.base import BaseMemory, MemoryItem
from ..core.embeddings import Embedder, cosine_topk, get_embedder
from ..core.llm import LLMClient, get_llm

logger = logging.getLogger(__name__)

FAILURES: Counter = Counter()

# Adapted from paper Appendix A.1 Fig. 6 (exchange -> turn for LoCoMo data).
SEGMENTATION_PROMPT = """# Instruction
## Context
- **Goal**: Your task is to segment a multi-turn conversation into topically coherent units based on semantics. Successive turns with the same topic should be grouped into the same segmentation unit, and new segmentation units should be created when topic shifts.
- **Data**: The input data is a series of turns separated by "\\n\\n". Each turn is started with "[Turn (Turn Number)]: ".
### Output Format
- Output the segmentation results in **JSONL (JSON Lines)** format. Each dictionary represents a segment, consisting of one or more turns on the same topic.
Each dictionary should include the following keys:
- **segment_id**: The index of this segment, starting from 0.
- **start_turn_number**: The number of the **first** turn in this segment.
- **end_turn_number**: The number of the **last** turn in this segment.
- **num_turns**: An integer indicating the number of turns in this segment, calculated as: **end_turn_number** - **start_turn_number** + 1.
Here is an example of the expected output:
```
<segmentation>
{"segment_id": 0, "start_turn_number": 0, "end_turn_number": 5, "num_turns": 6}
{"segment_id": 1, "start_turn_number": 6, "end_turn_number": 8, "num_turns": 3}
...
</segmentation>
```
# Data
{text_to_be_segmented}
# Question
## Please generate the segmentation result from the input data that meets the following requirements:
- **No Missing Turns**: Ensure that the turn numbers cover all turns in the given conversation without omission.
- **No Overlapping Turns**: Ensure that successive segments have no overlap in turns.
- **Accurate Counting**: The sum of **num_turns** across all segments should equal the total number of turns.
- Provide your segmentation result between the tags: <segmentation></segmentation>.
# Output
Now, provide the segmentation result based on the instructions above.
"""


@dataclass
class Segment:
    """One memory unit: a topically coherent run of dialogue turns."""

    text: str
    compressed: str
    id: str
    timestamp: str | None = None
    session_id: str | None = None


class Lingua2Compressor:
    """LLMLingua-2 denoiser (paper: xlm-roberta-large, rate 0.75)."""

    def __init__(self, model_path: str, rate: float = 0.75, device_map: str = "cuda"):
        from llmlingua import PromptCompressor

        self._compressor = PromptCompressor(
            model_name=model_path, use_llmlingua2=True, device_map=device_map
        )
        self.rate = rate

    def compress(self, text: str) -> str:
        # xlm-roberta-large has a 512-token context; compress in word windows
        # to stay safely below it and concatenate the compressed pieces.
        words = text.split()
        if len(words) <= 380:
            return self._compressor.compress_prompt(text, rate=self.rate)[
                "compressed_prompt"
            ]
        out = []
        for i in range(0, len(words), 380):
            out.append(
                self._compressor.compress_prompt(
                    " ".join(words[i: i + 380]), rate=self.rate
                )["compressed_prompt"]
            )
        return " ".join(out)


class BM25Index:
    """Minimal Okapi BM25 over whitespace tokens."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def add(self, text: str) -> None:
        toks = text.lower().split()
        self.docs.append(toks)
        self.df.update(set(toks))
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)

    def scores(self, query: str) -> np.ndarray:
        n_docs = len(self.docs)
        out = np.zeros(n_docs, dtype=np.float32)
        for tok in query.lower().split():
            df = self.df.get(tok, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for i, doc in enumerate(self.docs):
                tf = doc.count(tok)
                if tf:
                    out[i] += (
                        idf
                        * tf
                        * (self.k1 + 1)
                        / (tf + self.k1 * (1 - self.b + self.b * len(doc) / (self.avgdl + 1e-8)))
                    )
        return out


def _parse_segmentation(raw: str, n_turns: int) -> list[tuple[int, int]] | None:
    """Parse <segmentation> JSONL into [(start, end)] turn ranges.

    Strictly validated first (contiguous, full coverage). If that fails, a
    repair pass salvages whatever ranges were parsed: overlaps are clipped,
    gaps are filled, and coverage is extended to all turns — small models
    (Qwen3-4B vs the paper's GPT-4 segmenter) occasionally misalign indices.
    Returns None only when nothing usable could be parsed.
    """
    m = re.search(r"<segmentation>(.*?)</segmentation>", raw, flags=re.DOTALL)
    body = m.group(1) if m else raw
    parsed: list[tuple[int, int]] = []
    for line in body.strip().splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        try:
            obj = json.loads(line)
            parsed.append(
                (int(obj["start_turn_number"]), int(obj["end_turn_number"]))
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # salvage individual (start, end) pairs from malformed lines
            nums = re.findall(
                r'"start_turn_number"\s*:\s*(\d+).*?"end_turn_number"\s*:\s*(\d+)',
                line,
            )
            if nums:
                parsed.append((int(nums[0][0]), int(nums[0][1])))

    def valid(ranges: list[tuple[int, int]]) -> bool:
        expected = 0
        for start, end in ranges:
            if start != expected or end < start:
                return False
            expected = end + 1
        return expected == n_turns

    if valid(parsed):
        return parsed

    repaired: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(parsed):
        start = max(start, cursor)  # clip overlaps / fill gaps
        end = min(max(end, start), n_turns - 1)
        if start > cursor:
            start = cursor
        if start >= n_turns:
            break
        repaired.append((start, end))
        cursor = end + 1
    if repaired and repaired[-1][1] < n_turns - 1:
        repaired[-1] = (repaired[-1][0], n_turns - 1)
    if repaired and repaired[0][0] == 0 and repaired[-1][1] == n_turns - 1:
        logger.info("segmentation output repaired (%d segments)", len(repaired))
        FAILURES["segment_repaired"] += 1
        return repaired
    return None


class SeComMemory(BaseMemory):
    """Segment-level memory with LLMLingua-2 denoising."""

    name = "secom"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
    ):
        super().__init__(config)
        self.llm = llm or get_llm(
            self.config.get("llm", "openai"), **self.config.get("llm_kwargs", {})
        )
        self.retriever = self.config.get("retriever", "bm25")  # bm25 | mpnet
        self.embedder = embedder
        if self.retriever == "mpnet" or self.embedder is None:
            self.embedder = self.embedder or get_embedder(
                self.config.get("embedder", "sentence_transformers"),
                **self.config.get("embedder_kwargs", {}),
            )
        self.budget_tokens = int(self.config.get("budget_tokens", 4000))
        self._counter = 0

        compressor_path = self.config.get("compressor")
        self.compressor = (
            Lingua2Compressor(
                compressor_path,
                rate=float(self.config.get("compression_rate", 0.75)),
            )
            if compressor_path
            else None
        )

        self._segments: list[Segment] = []
        self._matrix: np.ndarray | None = None  # mpnet embeddings of compressed text
        self._bm25 = BM25Index()
        self._pending: list[tuple[str, dict]] = []  # turns of the current session
        self._cur_session: str | None = None

    # -------------------------------------------------------------- ingestion

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        sid = metadata.get("session_id")
        if self._cur_session is None:
            self._cur_session = sid
        elif sid != self._cur_session:
            self._flush()  # session boundary: segment the finished session
            self._cur_session = sid
        self._pending.append((content, metadata))
        return f"pending-{len(self._pending)}"

    def _flush(self) -> None:
        """Segment the buffered session turns and index the new segments."""
        if not self._pending:
            return
        turns = [c for c, _ in self._pending]
        timestamp = self._pending[0][1].get("timestamp")
        sid = self._cur_session
        self._pending = []

        ranges = self._segment_session(turns)
        for start, end in ranges:
            text = "\n".join(turns[start: end + 1])
            compressed = self._denoise(text)
            self._counter += 1
            seg = Segment(
                text=text,
                compressed=compressed,
                id=f"seg-{self._counter}",
                timestamp=timestamp,
                session_id=sid,
            )
            self._segments.append(seg)
            self._index(seg)

    def _segment_session(self, turns: list[str]) -> list[tuple[int, int]]:
        data = "\n\n".join(f"[Turn {j}]: {t}" for j, t in enumerate(turns))
        try:
            raw = self.llm.chat(
                "You are a helpful assistant.",
                SEGMENTATION_PROMPT.replace("{text_to_be_segmented}", data),
                max_tokens=2000,
            )
            ranges = _parse_segmentation(raw, len(turns))
            if ranges is None:
                raise ValueError("invalid segmentation output")
            return ranges
        except Exception as e:
            logger.warning("segmentation failed (%s); whole session as one segment", e)
            FAILURES["segment"] += 1
            return [(0, len(turns) - 1)]

    def _denoise(self, text: str) -> str:
        if self.compressor is None:
            return text
        try:
            return self.compressor.compress(text)
        except Exception as e:
            logger.warning("compression failed: %s", e)
            FAILURES["compress"] += 1
            return text

    def _index(self, seg: Segment) -> None:
        if self.retriever == "bm25":
            self._bm25.add(seg.compressed)
        else:
            vec = self.embedder.embed([seg.compressed])
            self._matrix = (
                vec if self._matrix is None else np.vstack([self._matrix, vec])
            )

    # -------------------------------------------------------------- retrieval

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryItem]:
        self._flush()
        if not self._segments:
            return []
        if self.retriever == "bm25":
            order = np.argsort(-self._bm25.scores(query))
        else:
            q = self.embedder.embed([query])[0]
            order = np.array(
                [i for i, _ in cosine_topk(q, self._matrix, len(self._segments))]
            )

        selected: list[int] = []
        tokens = 0
        for i in order:
            if len(selected) >= top_k:
                break
            seg_tokens = int(len(self._segments[i].text.split()) * 1.3)
            if selected and tokens + seg_tokens > self.budget_tokens:
                continue
            selected.append(int(i))
            tokens += seg_tokens
        selected.sort()  # chronological order for generation (paper Sec. 2.1)
        return [
            MemoryItem(
                content=self._segments[i].text,
                id=self._segments[i].id,
                metadata={
                    "timestamp": self._segments[i].timestamp,
                    "session_id": self._segments[i].session_id,
                    "compressed": self._segments[i].compressed,
                },
            )
            for i in selected
        ]

    def render(self, item: MemoryItem) -> str:
        ts = item.metadata.get("timestamp")
        return f"[{ts}]\n{item.content}" if ts else item.content

    # --------------------------------------------------------------- maintain

    def save(self, path) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "segments": self._segments,
                    "matrix": self._matrix,
                    "bm25": self._bm25,
                    "counter": self._counter,
                },
                f,
            )

    def load(self, path) -> None:
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        self._segments = state["segments"]
        self._matrix = state["matrix"]
        self._bm25 = state["bm25"]
        self._counter = state["counter"]
        self._pending = []
        self._cur_session = None

    def reset(self) -> None:
        self._segments = []
        self._matrix = None
        self._bm25 = BM25Index()
        self._pending = []
        self._cur_session = None
        self._counter = 0

    def __len__(self) -> int:
        return len(self._segments)
