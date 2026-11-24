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
