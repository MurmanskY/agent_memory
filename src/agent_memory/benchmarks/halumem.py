"""HaluMem operation-level hallucination benchmark.

Upstream: https://github.com/MemTensor/HaluMem
The judge prompts and all-denominator aggregation follow the official evaluator.
Methods with ``add_session`` (currently ProMem) also expose per-session extracted
memories. Other BaseMemory methods are evaluated on updates and QA only.
"""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Iterator

from ..core.base import BaseMemory
from ..core.llm import LLMClient
from ..evaluation.halumem import (
    evaluation_for_memory_accuracy,
    evaluation_for_memory_integrity,
    evaluation_for_question,
    evaluation_for_update_memory,
)
from .base import Benchmark, Conversation, QASample, Session, Turn


PROMPT_MEMZERO = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    {context}

    Question: {question}

    Answer:
    """


def extract_user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if not match:
        raise ValueError("HaluMem persona_info has no Name/Gender fields")
    return match.group(1).strip()


def _integer(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dialogue_text(session: dict) -> str:
    lines = []
    for turn in session["dialogue"]:
        lines.append(f"[{turn['timestamp']}]{turn['role']}: {turn['content']}")
        if turn["role"] == "assistant":
            lines.append("")
    return "\n".join(lines)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate_results(
    integrity: list[dict], accuracy: list[dict], updates: list[dict], qa: list[dict]
) -> dict:
    target = [r for r in integrity if r.get("memory_source") != "interference"]
    interference = [r for r in integrity if r.get("memory_source") == "interference"]
    recall = _safe_div(sum(r.get("score") == 2 for r in target), len(target))
    weighted_recall = _safe_div(
        sum(0.5 * (r.get("score") or 0) * float(r.get("importance", 0)) for r in target),
        sum(float(r.get("importance", 0)) for r in target),
    )
    fmr = _safe_div(sum(r.get("score") == 0 for r in interference), len(interference))

    target_accuracy = [r for r in accuracy if r.get("included") is True]
    target_precision = _safe_div(
        sum(0.5 * (r.get("score") or 0) for r in target_accuracy), len(target_accuracy)
    )
    memory_accuracy = _safe_div(
        sum(0.5 * (r.get("score") or 0) for r in accuracy), len(accuracy)
    )
    extraction_f1 = _safe_div(2 * target_precision * recall, target_precision + recall)

    def ratios(records: list[dict], key: str, labels: tuple[str, ...]) -> dict:
        return {
            label.lower(): _safe_div(sum(r.get(key) == label for r in records), len(records))
            for label in labels
        }

    update_ratios = ratios(updates, "label", ("Correct", "Hallucination", "Omission", "Other"))
    qa_ratios = ratios(qa, "label", ("Correct", "Hallucination", "Omission"))
    invalid = sum(r.get("score") is None for r in integrity + accuracy)
    invalid += sum(r.get("label") not in ("Correct", "Hallucination", "Omission", "Other") for r in updates)
    invalid += sum(r.get("label") not in ("Correct", "Hallucination", "Omission") for r in qa)
    return {
        "n_questions": len(qa),
        "n_memory_points": len(target),
        "n_interference_points": len(interference),
        "n_extracted_memories": len(accuracy),
        "n_updates": len(updates),
        "memory_recall": recall,
        "weighted_memory_recall": weighted_recall,
        "target_memory_precision": target_precision,
        "memory_accuracy": memory_accuracy,
        "false_memory_resistance": fmr,
        "memory_extraction_f1": extraction_f1,
        "update_correct": update_ratios["correct"],
        "update_hallucination": update_ratios["hallucination"],
        "update_omission": update_ratios["omission"],
        "update_other": update_ratios["other"],
        "qa_correct": qa_ratios["correct"],
        "qa_hallucination": qa_ratios["hallucination"],
        "qa_omission": qa_ratios["omission"],
        "n_invalid_judges": invalid,
    }


class HaluMemBenchmark(Benchmark):
    name = "halumem"

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        with open(self.data_path, encoding="utf-8") as file:
            self._users = [json.loads(line) for line in file if line.strip()]

    def conversations(self) -> Iterator[Conversation]:
        for user in self._users:
            sessions = []
            for index, session in enumerate(user["sessions"]):
                if session.get("is_generated_qa_session", False):
                    continue
                sessions.append(Session(
                    session_id=str(index),
                    timestamp=session.get("start_time"),
                    turns=[
                        Turn(speaker=turn["role"], text=turn["content"])
                        for turn in session["dialogue"]
                    ],
                ))
            yield Conversation(
                conv_id=user["uuid"],
                sessions=sessions,
                metadata={"user_name": extract_user_name(user["persona_info"])},
            )

    def qa_samples(self, conv_id: str) -> list[QASample]:
        user = next(user for user in self._users if user["uuid"] == conv_id)
        samples = []
        for index, session in enumerate(user["sessions"]):
            for q_index, question in enumerate(session.get("questions", [])):
                samples.append(QASample(
                    question_id=f"{conv_id}-{index}-{q_index}",
                    conv_id=conv_id,
                    question=question["question"],
                    answer=question["answer"],
                    category=question.get("question_type"),
                    metadata={"evidence": question.get("evidence", [])},
                ))
        return samples

    def format_turn(self, turn: Turn) -> str:
        return f"{turn.speaker}: {turn.text}"

    def score_sample(self, sample: QASample, pred: str, llm=None) -> dict | None:
        evidence = "\n".join(
            item["memory_content"] for item in sample.metadata.get("evidence", [])
        )
        result = evaluation_for_question(
            llm, sample.question, sample.answer, evidence, pred
        )
        label = result.get("evaluation_result")
        return {
            "qa_correct": float(label == "Correct"),
            "qa_hallucination": float(label == "Hallucination"),
            "qa_omission": float(label == "Omission"),
            "judge_valid": float(label in ("Correct", "Hallucination", "Omission")),
        }

    def run(
        self,
        memory: BaseMemory,
        llm: LLMClient,
        top_k: int = 20,
        max_episodes: int | None = None,
        max_sessions: int | None = None,
        max_questions: int | None = None,
        eval_mode: str = "all",
        output_path: str | Path | None = None,
        **_,
    ) -> dict:
        if eval_mode not in ("all", "qa"):
            raise ValueError("HaluMem eval_mode must be 'all' or 'qa'")
        integrity, accuracy, updates, qa_records = [], [], [], []
        out_f = open(output_path, "w") if output_path else None

        def emit(record):
            if out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

        users = self._users[:max_episodes] if max_episodes else self._users
        for user in users:
            user_name = extract_user_name(user["persona_info"])
            memory.reset()
            sessions = user["sessions"]
            if max_sessions:
                sessions = sessions[:max_sessions]
            for session_index, session in enumerate(sessions):
                start = time.time()
                extracted = self._ingest_session(memory, session, user_name, session_index)
                ingest_ms = (time.time() - start) * 1000
                generated = session.get("is_generated_qa_session", False)
                if generated:
                    continue
                update_candidates = {}
                for point in session["memory_points"]:
                    if point["is_update"] == "True" and point.get("original_memories"):
                        retrieved = memory.retrieve(point["memory_content"], top_k=10)
                        update_candidates[id(point)] = [memory.render(item) for item in retrieved]

                if eval_mode == "all" and extracted is not None:
                    self._score_extraction(
                        llm, user, session, session_index, extracted,
                        update_candidates, integrity, accuracy, emit,
                    )
                if eval_mode == "all":
                    self._score_updates(
                        llm, user, session, session_index, update_candidates, updates, emit
                    )

                questions = session.get("questions", [])
                if max_questions is not None:
                    questions = questions[:max_questions]
                for question_index, question in enumerate(questions):
                    search_start = time.time()
                    retrieved = memory.retrieve(question["question"], top_k=top_k)
                    search_ms = (time.time() - search_start) * 1000
                    memories = [memory.render(item) for item in retrieved]
                    context = f"Memories for user {user_name}:\n\n    " + json.dumps(
                        memories, ensure_ascii=False, indent=4
                    )
                    prompt = PROMPT_MEMZERO.format(
                        context=context, question=question["question"]
                    )
                    response = llm.chat(
                        "You are an intelligent memory assistant.", prompt, max_tokens=1024
                    )
                    evidence = "\n".join(
                        item["memory_content"] for item in question.get("evidence", [])
                    )
                    judged = evaluation_for_question(
                        llm, question["question"], question["answer"], evidence, response
                    )
                    record = {
                        "operation": "qa",
                        "uuid": user["uuid"],
                        "session_id": session_index,
                        "question_id": f"{user['uuid']}-{session_index}-{question_index}",
                        "question": question["question"],
                        "gold": question["answer"],
                        "pred": response,
                        "question_type": question.get("question_type"),
                        "label": judged.get("evaluation_result"),
                        "judge": judged,
                        "retrieved_ids": [item.id for item in retrieved],
                        "ingest_ms": ingest_ms,
                        "search_ms": search_ms,
                    }
                    qa_records.append(record)
                    emit(record)
        if out_f:
            out_f.close()
        return {
            "benchmark": self.name,
            "method": memory.name,
            "n_users": len(users),
            "eval_mode": eval_mode,
            **aggregate_results(integrity, accuracy, updates, qa_records),
        }

    def _ingest_session(self, memory, session, user_name, session_index):
        if hasattr(memory, "add_session"):
            items = memory.add_session(
                copy.deepcopy(session["dialogue"]),
                user_name=user_name,
                session_id=str(session_index),
                timestamp=session.get("start_time", ""),
            )
            return [item.content for item in items]
        for turn in session["dialogue"]:
            memory.add(
                f"{turn['role']}: {turn['content']}",
                metadata={
                    "session_id": str(session_index),
                    "timestamp": turn.get("timestamp"),
                    "role": turn.get("role"),
                    "user_name": user_name,
                },
            )
        return None

    def _score_extraction(
        self, llm, user, session, session_index, extracted,
        update_candidates, integrity, accuracy, emit,
    ):
        extracted_text = "\n".join(extracted)
        for point in session["memory_points"]:
            if point["is_update"] == "True" and update_candidates.get(id(point)):
                continue
            score = 0
            judged = {"score": 0, "reasoning": "No extracted memories."}
            if extracted_text.strip():
                judged = evaluation_for_memory_integrity(
                    llm, extracted_text, point["memory_content"]
                )
                score = _integer(judged.get("score"))
            record = {
                "operation": "memory_integrity",
                "uuid": user["uuid"],
                "session_id": session_index,
                "memory_content": point["memory_content"],
                "memory_type": point.get("memory_type"),
                "memory_source": point.get("memory_source"),
                "importance": point.get("importance", 0),
                "score": score,
                "judge": judged,
            }
            integrity.append(record)
            emit(record)

        dialogue = _dialogue_text(session)
        golden = "\n".join(
            point["memory_content"] for point in session["memory_points"]
            if point.get("memory_source") != "interference"
        )
        for candidate in extracted:
            judged = evaluation_for_memory_accuracy(llm, dialogue, golden, candidate)
            record = {
                "operation": "memory_accuracy",
                "uuid": user["uuid"],
                "session_id": session_index,
                "memory_content": candidate,
                "score": _integer(judged.get("accuracy_score")),
                "included": str(judged.get("is_included_in_golden_memories", "")).lower() == "true",
                "judge": judged,
            }
            accuracy.append(record)
            emit(record)

    def _score_updates(
        self, llm, user, session, session_index, update_candidates, updates, emit
    ):
        for point in session["memory_points"]:
            candidates = update_candidates.get(id(point))
            if not candidates:
                continue
            judged = evaluation_for_update_memory(
                llm,
                "\n".join(candidates),
                point["memory_content"],
                "\n".join(point["original_memories"]),
            )
            record = {
                "operation": "memory_update",
                "uuid": user["uuid"],
                "session_id": session_index,
                "memory_content": point["memory_content"],
                "label": judged.get("evaluation_result"),
                "judge": judged,
            }
            updates.append(record)
            emit(record)
