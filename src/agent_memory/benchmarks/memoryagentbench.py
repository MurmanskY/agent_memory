"""MemoryAgentBench (ICLR 2026): memory agents via incremental multi-turn interaction.

Upstream: https://github.com/HUST-AI-HYZ/MemoryAgentBench
Data:     https://huggingface.co/datasets/ai-hyz/MemoryAgentBench (4 parquet splits)

Each example is one long `context` plus multiple QA pairs ("inject once, query
multiple times"). Following the upstream protocol, the context is split into
chunks (sentence-boundary aware, `chunk_size` tokens) that are fed to the
memory system one by one; afterwards all questions are asked.

Metric mapping (upstream README):
  - Accurate_Retrieval (event_qa, ruler_qa*) : substring_exact_match
  - Conflict_Resolution (factconsolidation_*): substring_exact_match
  - Long_Range_Understanding (detectiveQA)   : exact_match
  - Test_Time_Learning (ICL_*)               : exact_match
  - recsys / infbench summarization subsets need special metrics (Recall@5 /
    LLM-judge F1) and are skipped here with a warning.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from ..evaluation.metrics import (
    exact_match_norm,
    metric_max_over_golds,
    substring_exact_match,
)
from .base import Benchmark, Conversation, QASample, Session, Turn

logger = logging.getLogger(__name__)

EXACT_MATCH_SOURCES = ("icl", "detective")
SKIP_SOURCES = ("recsys", "infbench", "movie")

_SPLITS = {
    "accurate_retrieval": "Accurate_Retrieval",
    "conflict_resolution": "Conflict_Resolution",
    "long_range_understanding": "Long_Range_Understanding",
    "test_time_learning": "Test_Time_Learning",
}


def chunk_text(text: str, chunk_size: int = 4096) -> list[str]:
    """Sentence-boundary-aware chunking, ~chunk_size tokens (words as proxy)."""
    sentences = [s for s in re.split(r"(?<=[.!?。！？\n])\s+", text) if s.strip()]
    chunks, buf, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if buf and n + w > chunk_size:
            chunks.append(" ".join(buf))
            buf, n = [], 0
        buf.append(s)
        n += w
    if buf:
        chunks.append(" ".join(buf))
    return chunks


class MemoryAgentBenchBenchmark(Benchmark):
    """Load one split parquet, e.g. data/memoryagentbench/Accurate_Retrieval.parquet.

    `source` optionally keeps only examples whose metadata.source matches
    (e.g. "event_qa", "ruler_qa1", "factconsolidation_mh_32k", "icl_banking").
    """

    name = "memoryagentbench"

    def __init__(
        self,
        data_path: str | Path,
        source: str | None = None,
        chunk_size: int = 4096,
        max_samples: int | None = None,
    ):
        self.data_path = Path(data_path)
        self.chunk_size = int(chunk_size)
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "MemoryAgentBench data is parquet; install the 'bench' extra: "
                "pip install -e '.[bench]'"
            ) from e
        rows = pq.read_table(self.data_path).to_pylist()
        if source:
            rows = [r for r in rows if (r.get("metadata") or {}).get("source") == source]
        if max_samples:
            rows = rows[:max_samples]
        self._rows = rows

    def conversations(self) -> Iterator[Conversation]:
        for i, row in enumerate(self._rows):
            meta = row.get("metadata") or {}
            sessions = [
                Session(
                    session_id=f"chunk_{j}",
                    turns=[Turn(speaker="document", text=chunk)],
                )
                for j, chunk in enumerate(chunk_text(row["context"], self.chunk_size))
            ]
            yield Conversation(
                conv_id=str(i),
                sessions=sessions,
                metadata={"source": meta.get("source")},
            )

    def qa_samples(self, conv_id: str) -> list[QASample]:
        row = self._rows[int(conv_id)]
        meta = row.get("metadata") or {}
        qa_ids = meta.get("qa_pair_ids") or []
        qtypes = meta.get("question_types") or []
        samples = []
        for j, (q, ans) in enumerate(zip(row["questions"], row["answers"])):
            samples.append(
                QASample(
                    question_id=str(qa_ids[j]) if j < len(qa_ids) else f"{conv_id}-{j}",
                    conv_id=str(conv_id),
                    question=q,
                    answer=" ||| ".join(map(str, ans)) if isinstance(ans, list) else str(ans),
                    category=str(qtypes[j]) if j < len(qtypes) else None,
                    metadata={"answers": ans, "source": meta.get("source")},
                )
            )
        return samples

    def score_sample(self, sample: QASample, pred: str, llm=None) -> dict | None:
        source = str(sample.metadata.get("source") or "").lower()
        golds = sample.metadata.get("answers") or [sample.answer]
        if any(s in source for s in SKIP_SOURCES):
            logger.warning("source %s needs a special metric; scored as skipped", source)
            return {}
        if any(s in source for s in EXACT_MATCH_SOURCES):
            return {"accuracy": metric_max_over_golds(exact_match_norm, pred, golds)}
        return {"accuracy": metric_max_over_golds(substring_exact_match, pred, golds)}
