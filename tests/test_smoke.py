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


class FakeMem0LLM:
    """Fact extraction -> one fact; manager -> ADD it."""

    def chat(self, system: str, user: str, **kwargs) -> str:
        if "Personal Information Organizer" in system:
            return json.dumps({"facts": ["User adopted a cat named Mochi"]})
        if "smart memory manager" in user:
            return json.dumps({"memory": [
                {"id": "0", "text": "User adopted a cat named Mochi",
                 "event": "ADD"}]})
        return "Mochi"


def test_mem0_write_path_and_retrieve():
    mem = get_baseline("mem0", {}, llm=FakeMem0LLM(), embedder=get_embedder("hash"))
    mem.add("Speaker Alicesays : I adopted a cat named Mochi.",
            metadata={"timestamp": "1:56 pm on 8 May, 2023"})
    mem.add("Speaker Bobsays : Cute! How old is Mochi?",
            metadata={"timestamp": "1:56 pm on 8 May, 2023"})

    # one bank per speaker; the fact is written into both banks
    assert sorted(mem._banks) == ["Alice", "Bob"]
    for bank in mem._banks.values():
        assert len(bank) >= 1
        assert any("Mochi" in t for t in bank.texts)

    hits = mem.retrieve("cat name", top_k=2)
    assert hits and "Mochi" in hits[0].content
    assert hits[0].metadata["speaker"] in ("Alice", "Bob")
    assert mem.render(hits[0]).startswith("1:56 pm on 8 May, 2023: ")

    # cache round-trip
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pkl")
        mem.save(p)
        mem.reset()
        assert len(mem) == 0
        mem.load(p)
        assert len(mem) >= 2


def test_memoryr1_answer_agent_protocol():
    class FakeAnswerLLM(FakeMem0LLM):
        def chat(self, system: str, user: str, **kwargs) -> str:
            if "intelligent memory assistant" in system:
                return "Some distilled reasoning. **Answer:** Mochi"
            return super().chat(system, user, **kwargs)

    llm = FakeAnswerLLM()
    mem = get_baseline("memoryr1", {}, llm=llm, embedder=get_embedder("hash"))
    mem.add("Speaker Alicesays : I adopted a cat named Mochi.",
            metadata={"timestamp": "t0"})
    pred, ids = mem.answer("What is the cat's name?", llm, top_k=5)
    assert pred == "Mochi"  # extracted after "**Answer:**"
    assert ids

    # the runner honors use_memory_answer
    data = [{"sample_id": 0,
             "conversation": {"session_1_date_time": "t0", "session_1": [
                 {"speaker": "Alice", "text": "I adopted a cat named Mochi."}]},
             "qa": [{"question": "What is the cat's name?", "answer": "Mochi",
                     "category": "4"}]}]
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "locomo.json")
        open(f, "w").write(json.dumps(data))
        bench = LoCoMoBenchmark(f)
        summary = run_benchmark(bench, mem, llm, use_memory_answer=True)
        assert summary["n_questions"] == 1
        assert summary["f1"] == 1.0


def test_agemem_agent_tool_loop():
    from agent_memory.baselines.agemem import AgeMemAgent
    from agent_memory.baselines.agemem.agent import parse_tool_calls, parse_answer

    # parsing
    calls = parse_tool_calls(
        '<think>x</think><tool_call>[{"name": "Add_memory", "arguments": {"content": "cats are cute"}}]</tool_call>')
    assert calls == [{"name": "Add_memory", "arguments": {"content": "cats are cute"}}]
    assert parse_answer("<answer>42</answer>") == "42"
    assert parse_answer("no tags") is None

    class FakeAgentLLM:
        def __init__(self):
            self.n_qa = 0

        def chat(self, system, user, **kwargs):
            # stage 3: retrieve once, then answer (check before stage-1 marker:
            # the stage-3 prompt still contains the stage-1 conversation)
            if "Which city is the Eiffel Tower in?" in user:
                self.n_qa += 1
                if self.n_qa == 1:
                    return ('<think>Check memory.</think><tool_call>'
                            '[{"name": "Retrieve_memory", "arguments": {"query": "Eiffel Tower", "top_k": 3}}]'
                            '</tool_call>')
                return "<think>Memory says Paris.</think><answer>Paris</answer>"
            if "Text 1" in user:  # similarity prompt
                return "0.0"
            if "summarization" in user.lower():
                return "summary text"
            if "Just chatting about" in user:
                return ('<think>Store facts.</think><tool_call>'
                        '[{"name": "Add_memory", "arguments": {"content": "The Eiffel Tower is in Paris."}}]'
                        '</tool_call>')
            return "<answer>unknown</answer>"

    agent = AgeMemAgent({}, llm=FakeAgentLLM(), embedder=get_embedder("hash"))
    result = agent.run_task(
        "Which city is the Eiffel Tower in?",
        [("Eiffel Tower", ["The Eiffel Tower is a wrought-iron tower in Paris."])],
    )
    assert result["answer"] == "Paris"
    assert result["n_memories"] >= 1  # stage-1 rounds each store the fact
    assert result["tool_usage"].get("Retrieve_memory", 0) >= 1
    assert result["tool_usage"].get("Add_memory", 0) >= 1
    assert "Paris" in agent.store.contents[0]


def test_trustmem_memory_tools_and_transition_rewards():
    from agent_memory.security.trustmem.memory_core import Memory, execute_function_call
    from agent_memory.security.trustmem.rollout import RolloutRecord, TransitionRecord
    from agent_memory.security.trustmem.tr_grpo import (
        assign_rewards, build_preference_pairs, compute_advantages,
    )

    memory = Memory()
    inserted = execute_function_call(memory, "new_memory_insert", {
        "memory_type": "semantic_memory", "content": "Alice adopted Mochi",
    })
    memory_id = next(iter(memory.semantic[0]))
    assert "executed successfully" in inserted
    updated = execute_function_call(memory, "memory_update", {
        "memory_type": "semantic", "memory_id": memory_id,
        "new_content": "Alice adopted Mochi and Luna",
    })
    assert "executed successfully" in updated
    assert "Luna" in memory.snapshot()

    records = []
    for rollout_idx, verifier_score in enumerate((0.9, 0.2)):
        record = RolloutRecord(0, rollout_idx, "hotpotqa")
        record.qa_results = [("Who did Alice adopt?", "Mochi and Luna", "Mochi and Luna")]
        transition = TransitionRecord(
            instance_idx=0, rollout_idx=rollout_idx, chunk_idx=0,
            prompt_ids=[1], response_ids=[2], old_logprobs=[-0.1],
            response_text="", actions=[], pre_memory="", post_memory="",
            chunk_text="Alice adopted Mochi and Luna", r_act=1.0,
            r_type=1.0, r_ver=verifier_score,
        )
        record.transitions.append(transition)
        records.append(record)
    assign_rewards(records)
    compute_advantages(records)
    pairs = build_preference_pairs(records)
    assert len(pairs) == 1
    assert pairs[0][0].r_ver == 0.9
    assert records[0].transitions[0].advantage > 0
    assert records[1].transitions[0].advantage < 0


def test_promem_session_memory_offline():
    from agent_memory.baselines.promem import ProMemMemory

    class FakeProMemLLM:
        def chat(self, system, user, **kwargs):
            if "micro-detail extractor" in user:
                return '["Mochi", "cat"]'
            if "structural logic analyzer" in user:
                return "[]"
            return "Alice adopted a cat named Mochi."

    class EntailmentNLI:
        def check_batch(self, pairs):
            return ["entailment"] * len(pairs)

    mem = ProMemMemory({}, llm=FakeProMemLLM(), embedder=get_embedder("hash"),
                       nli=EntailmentNLI())
    turns = [{"role": "user", "content": "I adopted a cat named Mochi.",
              "timestamp": "2026-01-01"}]
    extracted = mem.add_session(turns, user_name="Alice", session_id="s1",
                                timestamp="2026-01-01")
    assert extracted
    assert all("Alice" in item.content for item in extracted)
    assert mem.session_memories("s1") == extracted
    assert mem.retrieve("Mochi", top_k=2)
    mem.reset()
    assert len(mem) == 0


def _make_halumem_fixture(tmp_path):
    user = {
        "uuid": "u1",
        "persona_info": "Name: Alice Example; Gender: Female;",
        "sessions": [{
            "start_time": "Jan 01, 2026, 10:00:00",
            "memory_points": [{
                "memory_content": "Alice's birth date is 1996-08-02",
                "memory_type": "Persona Memory", "is_update": "False",
                "original_memories": [], "importance": 0.75,
                "memory_source": "system",
            }],
            "dialogue": [
                {"role": "user", "content": "I was born on 1996-08-02.",
                 "timestamp": "Jan 01, 2026, 10:00:00"},
                {"role": "assistant", "content": "Thanks.",
                 "timestamp": "Jan 01, 2026, 10:00:01"},
            ],
            "questions": [{
                "question": "When was Alice born?", "answer": "1996-08-02",
                "evidence": [{"memory_content": "Alice's birth date is 1996-08-02"}],
                "question_type": "Basic Fact Recall",
            }],
        }],
    }
    path = tmp_path / "HaluMem-Medium.jsonl"
    path.write_text(json.dumps(user) + "\n")
    return path


def test_halumem_protocol_and_aggregation(tmp_path):
    from agent_memory.benchmarks.halumem import HaluMemBenchmark

    class FakeHaluMemLLM:
        def chat(self, system, user, **kwargs):
            if "Memory Integrity" in user:
                return json.dumps({"score": "2", "reasoning": "covered"})
            if "Dialogue Memory Accuracy Evaluator" in user:
                return json.dumps({"accuracy_score": "2",
                                   "is_included_in_golden_memories": "true"})
            if "evaluation expert for AI memory system question answering" in user:
                return json.dumps({"evaluation_result": "Correct", "reasoning": "match"})
            return "1996-08-02"

    bench = HaluMemBenchmark(_make_halumem_fixture(tmp_path))
    assert len(list(bench.conversations())) == 1
    memory = get_baseline("full_context")
    # Full-context has no session extraction API, so supply one for this protocol test.
    extracted = []
    def add_session(dialogue, user_name, session_id, timestamp):
        for turn in dialogue:
            memory.add(turn["content"], {"session_id": session_id,
                                        "timestamp": turn["timestamp"]})
        from agent_memory.core.base import MemoryItem
        item = MemoryItem("Alice's birth date is 1996-08-02",
                          metadata={"session_id": session_id})
        extracted.append(item)
        return [item]
    memory.add_session = add_session
    summary = bench.run(memory, FakeHaluMemLLM(), eval_mode="all")
    assert summary["n_questions"] == 1
    assert summary["memory_recall"] == 1.0
    assert summary["target_memory_precision"] == 1.0
    assert summary["memory_extraction_f1"] == 1.0
    assert summary["qa_correct"] == 1.0
    assert summary["n_invalid_judges"] == 0
