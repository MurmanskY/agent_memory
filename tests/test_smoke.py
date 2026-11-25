"""Smoke tests: interfaces and end-to-end pipeline without any API access."""

import json

import pytest

from agent_memory.baselines import get_baseline
from agent_memory.baselines.amem import AMemMemory
from agent_memory.core.embeddings import get_embedder
from agent_memory.core.llm import get_llm
from agent_memory.evaluation.metrics import (
    exact_match,
    f1_score,
    longmemeval_judge_prompt,
    metric_max_over_golds,
    substring_exact_match,
)
from agent_memory.evaluation.runner import run_benchmark
from agent_memory.benchmarks.locomoco import LoCoMoBenchmark


class FakeJSONLLM:
    """Returns canned JSON for A-MEM note analysis / evolution prompts."""

    def __init__(self, evolve: bool = True):
        self.evolve = evolve

    def chat(self, system: str, user: str, **kwargs) -> str:
        if "structured analysis" in user:
            return json.dumps({
                "keywords": ["cat", "mochi", "pet"],
                "context": "The speaker talks about their pet cat.",
                "tags": ["pet", "animal", "daily-life"],
            })
        return json.dumps({
            "should_evolve": self.evolve,
            "actions": ["strengthen"],
            "suggested_connections": [0],
            "tags_to_update": ["pet", "cat"],
            "new_context_neighborhood": [],
            "new_tags_neighborhood": [],
        })


def test_rag_memory_add_retrieve_reset():
    mem = get_baseline("rag", {}, embedder=get_embedder("hash"))
    mem.add("I love hiking in the mountains.")
    mem.add("My favorite food is sushi.")
    mem.add("I went hiking last weekend with Alice.")
    hits = mem.retrieve("hiking", top_k=2)
    assert len(hits) == 2
    assert any("hiking" in h.content for h in hits)
    assert len(mem) == 3
    mem.reset()
    assert len(mem) == 0
    assert mem.retrieve("hiking") == []


def test_full_context_returns_all():
    mem = get_baseline("full_context")
    for i in range(5):
        mem.add(f"turn {i}")
    assert len(mem.retrieve("anything", top_k=2)) == 5


def test_metrics():
    assert exact_match("Paris", "paris") == 1.0
    assert f1_score("the capital is Paris", "Paris") == 0.4
    assert f1_score("London", "Paris") == 0.0
    assert substring_exact_match("The answer is Paris today.", "Paris") == 1.0
    assert metric_max_over_golds(substring_exact_match, "I pick B", ["A", "B"]) == 1.0


def test_amem_note_construction_and_linking():
    mem = get_baseline("amem", {}, llm=FakeJSONLLM(), embedder=get_embedder("hash"))
    mem.add("Alice: I adopted a cat named Mochi.", metadata={"timestamp": "2024-01-01"})
    mem.add("Bob: Cute! How old is Mochi?", metadata={"timestamp": "2024-01-01"})

    assert len(mem) == 2
    note = next(iter(mem.memories.values()))
    assert note.keywords == ["cat", "mochi", "pet"]
    assert note.context.startswith("The speaker")
    # second note linked to the first (strengthen -> suggested_connections [0])
    notes = list(mem.memories.values())
    assert notes[1].links == [0]
    assert notes[1].tags == ["pet", "cat"]

    hits = mem.retrieve("Mochi", top_k=2)
    assert hits
    rendered = mem.render(hits[0])
    assert "memory content:" in rendered and "memory context:" in rendered


def test_amem_graceful_on_bad_llm():
    mem = get_baseline("amem", {}, llm=get_llm("dummy"), embedder=get_embedder("hash"))
    mem.add("Alice: hello")  # dummy LLM returns non-JSON; note stored with defaults
    assert len(mem) == 1
    note = next(iter(mem.memories.values()))
    assert note.keywords == [] and note.context == "General"
    assert mem.retrieve("hello")


class FakeMemoryR1LLM:
    """Scripted LLM: rule-based fact extraction + queued manager ops."""

    def __init__(self, manager_ops: list[dict]):
        self.manager_ops = manager_ops
        self.manager_calls = 0

    def chat(self, system: str, user: str, **kwargs) -> str:
        if user.startswith("Input:\n"):  # fact extraction
            if "adopted" in user and "Luna" in user:
                return json.dumps({"facts": ["Alice adopted another cat named Luna"]})
            if "adopted" in user and "Mochi" in user:
                return json.dumps({"facts": ["Alice adopted a cat named Mochi"]})
            return json.dumps({"facts": []})
        assert "smart memory manager" in user  # memory manager prompt
        op = self.manager_ops[self.manager_calls]
        self.manager_calls += 1
        return json.dumps({"memory": [op]})


def test_memoryr1_manager_add_update():
    llm = FakeMemoryR1LLM([
        # 1st/2nd manager calls: ADD (Alice bank, then backfill into Bob bank)
        {"id": "0", "text": "Alice adopted a cat named Mochi", "event": "ADD"},
        {"id": "0", "text": "Alice adopted a cat named Mochi", "event": "ADD"},
        # 3rd/4th calls: UPDATE consolidates instead of fragmenting
        {"id": "0", "text": "Alice adopted two cats, Mochi and Luna",
         "event": "UPDATE", "old_memory": "Alice adopted a cat named Mochi"},
        {"id": "0", "text": "Alice adopted two cats, Mochi and Luna",
         "event": "UPDATE", "old_memory": "Alice adopted a cat named Mochi"},
    ])
    mem = get_baseline("memoryr1", {}, llm=llm, embedder=get_embedder("hash"))
    mem.add("Alice: I adopted a cat named Mochi.", metadata={"timestamp": "t1"})
    assert len(mem) == 1
    mem.add("Bob: Cute! How old is Mochi?", metadata={"timestamp": "t1"})
    assert len(mem) == 2  # one entry per speaker bank
    mem.add("Alice: I adopted another cat named Luna.", metadata={"timestamp": "t2"})
    assert len(mem) == 2  # UPDATE keeps one entry per bank
    hits = mem.retrieve("cats", top_k=5)
    assert len(hits) == 2
    assert all("Mochi and Luna" in h.content for h in hits)
    assert all(h.metadata["timestamp"] == "t2" for h in hits)
    assert "t2: Alice adopted two cats" in mem.render(hits[0])


def test_memoryr1_delete_and_noop():
    llm = FakeMemoryR1LLM([
        {"id": "0", "text": "Alice adopted a cat named Mochi", "event": "ADD"},
        {"id": "0", "text": "Alice adopted a cat named Mochi", "event": "DELETE"},
        {"id": "0", "text": "Alice adopted a cat named Mochi", "event": "NONE"},
    ])
    mem = get_baseline("memoryr1", {}, llm=llm, embedder=get_embedder("hash"))
    mem.add("Alice: I adopted a cat named Mochi.")
    assert len(mem) == 1
    mem.add("Alice: I adopted a cat named Mochi.")
    assert len(mem) == 0  # DELETE removed the entry
    mem.add("Alice: I adopted a cat named Mochi.")
    assert len(mem) == 0  # NONE changes nothing


def test_memoryr1_answer_agent_distillation():
    class FakeAnswerLLM:
        def chat(self, system, user, **kwargs):
            assert "Memories for user Alice" in system
            assert "Memories for user Bob" in system
            # memory distillation instructions from the official QWEN template
            assert "Select memories you found that are useful" in system
            return "Some reasoning.\n**Answer:** Two"

    mem = get_baseline("memoryr1", {}, llm=FakeMemoryR1LLM([]),
                       embedder=get_embedder("hash"))
    from agent_memory.baselines.memoryr1 import _Bank
    for owner in ("Alice", "Bob"):
        bank = _Bank(mem.embedder.dim)
        vec = mem.embedder.embed([f"{owner} memory about cats"])[0]
        bank.add_entry(f"id-{owner}", f"{owner} memory about cats", "t1", vec)
        mem._banks[owner] = bank
    pred, ids = mem.answer("How many cats does Alice have?", FakeAnswerLLM(), top_k=30)
    assert pred == "Two"
    assert set(ids) == {"id-Alice", "id-Bob"}


def test_memoryr1_graceful_on_bad_llm():
    mem = get_baseline("memoryr1", {}, llm=get_llm("dummy"),
                       embedder=get_embedder("hash"))
    mem.add("Alice: hello")  # non-JSON extraction -> no facts, no crash
    assert len(mem) == 0
    assert mem.retrieve("hello") == []
    pred, ids = mem.answer("hi?", get_llm("dummy"), top_k=30)
    assert pred == "dummy-answer"  # extraction falls back to the raw text


def test_mem0_base_prompt_and_raw_answer():
    class FakeAnswerLLM:
        def chat(self, system, user, **kwargs):
            assert "Memories for user Alice" in system
            # mem0's base prompt has NO memory-distillation instructions
            assert "Select memories you found that are useful" not in system
            assert "**Answer:**" not in system
            return "  Two cats  "  # official: the raw completion is the answer

    mem = get_baseline("mem0", {}, llm=FakeMemoryR1LLM([]),
                       embedder=get_embedder("hash"))
    assert mem.name == "mem0"
    from agent_memory.baselines.memoryr1 import _Bank
    for owner in ("Alice", "Bob"):
        bank = _Bank(mem.embedder.dim)
        vec = mem.embedder.embed([f"{owner} memory about cats"])[0]
        bank.add_entry(f"id-{owner}", f"{owner} memory about cats", "t1", vec)
        mem._banks[owner] = bank
    pred, ids = mem.answer("How many cats does Alice have?", FakeAnswerLLM())
    assert pred == "Two cats"  # no **Answer:** extraction
    assert set(ids) == {"id-Alice", "id-Bob"}


def test_longmemeval_judge_templates():
    p = longmemeval_judge_prompt("temporal-reasoning", "q", "a", "r")
    assert "off-by-one" in p
    p = longmemeval_judge_prompt("anything", "q", "explanation", "r", abstention=True)
    assert "unanswerable" in p


def test_end_to_end_with_dummy_llm(tmp_path):
    data = [
        {
            "sample_id": 0,
            "conversation": {
                "session_1_date_time": "2024-01-01",
                "session_1": [
                    {"speaker": "Alice", "text": "I adopted a cat named Mochi."},
                    {"speaker": "Bob", "text": "Cute! How old is Mochi?"},
                ],
            },
            "qa": [
                {"question": "What is the cat's name?", "answer": "Mochi",
                 "category": "1", "evidence": []}
            ],
        }
    ]
    data_file = tmp_path / "locomo10.json"
    data_file.write_text(json.dumps(data))

    bench = LoCoMoBenchmark(data_file)
    mem = get_baseline("rag", {}, embedder=get_embedder("hash"))
    summary = run_benchmark(bench, mem, get_llm("dummy"), top_k=2)
    assert summary["n_questions"] == 1
    assert 0.0 <= summary["f1"] <= 1.0


def _make_gatemem_fixture(tmp_path):
    ep = {
        "episode_id": "ep1",
        "domain": "education",
        "entities": {
            "principals": [
                {"principal_id": "prof_a", "role": "professor", "display_name": "Prof A"},
                {"principal_id": "stu_b", "role": "student", "display_name": "Stu B"},
            ],
            "relationships": [
                {"type": "research_project_member", "project_id": "p1",
                 "principal_id": "prof_a"},
            ],
        },
        "turns": [
            {"turn_id": "t001", "timestamp": "2026-04-08T09:00",
             "speaker": {"principal_id": "prof_a", "role": "professor"},
             "turn_kind": "dialogue",
             "text": "The dry-run date is May 12, 2026."},
            {"turn_id": "t002", "timestamp": "2026-04-08T09:03",
             "speaker": {"principal_id": "stu_b", "role": "student"},
             "turn_kind": "dialogue", "text": "Got it."},
        ],
    }
    ckpt = {
        "checkpoint_id": "ck1", "episode_id": "ep1", "as_of_turn_id": "t002",
        "asker": {"principal_id": "prof_a", "role": "professor"},
        "query_type": "utility",
        "query_text": "What is the current dry-run date?",
        "expected_action": "answer",
        "judge_spec": {"expected_action": "answer", "include": ["May 12, 2026"]},
        "leak_targets": [],
    }
    d = tmp_path / "education"
    d.mkdir()
    (d / "episodes.jsonl").write_text(json.dumps(ep) + "\n")
    (d / "checkpoints.jsonl").write_text(json.dumps(ckpt) + "\n")
    return d


def test_gatemem_incremental_protocol(tmp_path):
    from agent_memory.benchmarks.gatemem import GateMemBenchmark

    class GateLLM:
        def chat(self, system, user, **kwargs):
            assert "GLOBAL ACCESS POLICY" in user
            assert "prof_a" in user
            return json.dumps({"action": "answer",
                               "answer": "The dry-run date is May 12, 2026.",
                               "answer_structured": {}, "used_record_ids": []})

    bench = GateMemBenchmark(_make_gatemem_fixture(tmp_path))
    mem = get_baseline("full_context")
    summary = bench.run(mem, GateLLM(), top_k=5)
    assert summary["n_checkpoints"] == 1
    assert summary["utility_accuracy"] == 1.0
    assert summary["mgs"] == 1.0


def test_memoryagentbench_loader(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    from agent_memory.benchmarks.memoryagentbench import MemoryAgentBenchBenchmark

    table = pa.table({
        "context": ["The cat Mochi is cute. " * 50],
        "questions": [["What is the cat's name?"]],
        "answers": [[["Mochi"]]],
        "metadata": [{"source": "event_qa", "qa_pair_ids": ["q1"],
                      "question_types": ["single-hop"]}],
    })
    f = tmp_path / "Accurate_Retrieval.parquet"
    pq.write_table(table, f)

    bench = MemoryAgentBenchBenchmark(f, chunk_size=20)
    convs = list(bench.conversations())
    assert len(convs) == 1
    assert len(convs[0].sessions) > 1  # chunked
    samples = bench.qa_samples("0")
    assert samples[0].question == "What is the cat's name?"
    assert bench.score_sample(samples[0], "It is Mochi.") == {"accuracy": 1.0}
    assert bench.score_sample(samples[0], "It is Luna.") == {"accuracy": 0.0}


class FakeSegLLM:
    """Splits every session into 2-turn segments."""

    def chat(self, system: str, user: str, **kwargs) -> str:
        import re as _re
        n = len(_re.findall(r"\[Turn \d+\]:", user))
        lines = []
        for s in range(0, n, 2):
            e = min(s + 1, n - 1)
            lines.append({"segment_id": len(lines), "start_turn_number": s,
                          "end_turn_number": e, "num_turns": e - s + 1})
        return ("<segmentation>\n"
                + "\n".join(json.dumps(l) for l in lines)
                + "\n</segmentation>")


def test_secom_segments_and_retrieves():
    mem = get_baseline("secom", {"retriever": "bm25", "budget_tokens": 12},
                       llm=FakeSegLLM(), embedder=get_embedder("hash"))
    for i, text in enumerate([
            "Alice: I adopted a cat named Mochi.",
            "Bob: Cute! How old is Mochi?",
            "Alice: I went hiking in the Alps.",
            "Bob: Was the scenery good?"]):
        mem.add(text, metadata={"session_id": "s1", "timestamp": "t0"})
    mem.retrieve("hiking Alps", top_k=2)  # triggers flush of the pending session
    assert len(mem) == 2  # 4 turns -> two 2-turn segments
    hits = mem.retrieve("hiking Alps", top_k=2)
    assert len(hits) == 1  # budget only fits relevant ranking first
    assert "hiking" in hits[0].content
    assert "timestamp" in hits[0].metadata

    # new session boundary triggers flush of the previous one
    mem.add("Alice: new topic here.", metadata={"session_id": "s2", "timestamp": "t1"})
    mem.retrieve("new topic")
    assert len(mem) == 3

    mem.reset()
    assert len(mem) == 0
