"""The four-stage ProMem pipeline (paper Sec. 3.2-3.5).

Stage 1: multi-grained disentanglement (details / topic events / relations)
Stage 2: explicit completeness check (embedding cross-reference, tau=0.6)
Stage 3: hallucination verification (turn grounding + NLI + atomic decomposition
         + LLM adjudication)
Stage 4: relation verification (probe questions) + detail anchoring (turn IDs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .models import (Embedder, LLMClient, NLI, extract_json_list, extract_json_obj, split_lines)
from . import prompts


@dataclass
class MemoryEvent:
    text: str
    turn_ids: List[int] = field(default_factory=list)


@dataclass
class MemoryGraph:
    nodes: List[str] = field(default_factory=list)          # purified memory entries
    edges: List[tuple] = field(default_factory=list)        # (event_A, relation, event_B)
    attrs: Dict[int, List[str]] = field(default_factory=dict)  # node idx -> anchored details
    stats: Dict = field(default_factory=dict)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T  # embeddings are L2-normalized by the embedder


class ProMem:
    def __init__(self, llm: LLMClient, embedder: Embedder, nli: NLI, tau_comp: float = 0.6):
        self.llm = llm
        self.embedder = embedder
        self.nli = nli
        self.tau = tau_comp

    # ---------------- stage 1 ----------------

    def extract_details(self, turns: List[Dict]) -> Dict[int, List[str]]:
        prompts_list = [prompts.P_DETAIL.format(current_turn=self._turn_text(t)) for t in turns]
        outs = self.llm.ask_many(prompts_list, max_tokens=256)
        details = {}
        for i, out in enumerate(outs):
            items = extract_json_list(out)
            if items is None:
                items = [x.strip() for x in re.split(r",|\n", out) if x.strip()]
            items = [str(x).strip().strip('"').strip("'") for x in items if str(x).strip()]
            details[i] = items[:20]
        return details

    def cluster_turns(self, turns: List[Dict], distance_threshold: float = 0.55) -> List[List[int]]:
        """Embedding agglomerative clustering of dialogue turns (LightMem-style topic clustering)."""
        n = len(turns)
        if n <= 2:
            return [list(range(n))]
        from sklearn.cluster import AgglomerativeClustering
        vecs = self.embedder.encode([self._turn_text(t)[:800] for t in turns])
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clustering.fit_predict(vecs)
        clusters = {}
        for idx, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(idx)
        return list(clusters.values())

    def abstract_events(self, turns: List[Dict], clusters: List[List[int]], user_name: str) -> List[MemoryEvent]:
        prompts_list = []
        for cluster in clusters:
            clustered = "\n".join(self._turn_text(turns[i]) for i in cluster)[:8000]
            prompts_list.append(prompts.P_EVENT.format(clustered_turns=clustered, user_name=user_name))
        outs = self.llm.ask_many(prompts_list, max_tokens=512)
        events = []
        for cluster, out in zip(clusters, outs):
            for text in split_lines(out):
                events.append(MemoryEvent(text=text, turn_ids=list(cluster)))
        return events

    def infer_relations(self, events: List[MemoryEvent]) -> List[tuple]:
        if len(events) < 2:
            return []
        event_list = "\n".join(f"E{i+1}: {e.text}" for i, e in enumerate(events))
        out = self.llm.ask(prompts.P_RELATION.format(event_list=event_list), max_tokens=768)
        edges = []
        raw = extract_json_list(out) or []
        for item in raw:
            if not (isinstance(item, list) and len(item) == 3):
                continue
            a, rel, b = item
            ia, ib = self._event_index(a), self._event_index(b)
            if ia is None or ib is None or ia == ib:
                continue
            edges.append((events[ia].text, str(rel), events[ib].text))
        return edges

    @staticmethod
    def _event_index(tag) -> Optional[int]:
        m = re.search(r"(\d+)", str(tag))
        return int(m.group(1)) - 1 if m else None

    # ---------------- stage 2 ----------------

    def initial_draft(self, turns: List[Dict], user_name: str) -> List[str]:
        dialogue = "\n".join(self._turn_text(t) for t in turns)[:30000]
        out = self.llm.ask(prompts.P_DRAFT.format(user_name=user_name, dialogue=dialogue),
                           max_tokens=2048)
        return split_lines(out)

    def completeness_check(self, draft: List[str], events: List[MemoryEvent]) -> List[MemoryEvent]:
        """Return missed events to be merged into the draft (Eq. 2-4)."""
        if not events:
            return []
        if not draft:
            return list(events)
        ev_vecs = self.embedder.encode([e.text for e in events])
        draft_vecs = self.embedder.encode(draft)
        sims = _cosine_matrix(ev_vecs, draft_vecs)
        missed = [e for e, row in zip(events, sims) if row.max() <= self.tau]
        return missed

    # ---------------- stage 3 ----------------

    def ground_to_turns(self, entries: List[str], turns: List[Dict]) -> List[int]:
        if not entries:
            return []
        entry_vecs = self.embedder.encode(entries)
        turn_vecs = self.embedder.encode([self._turn_text(t)[:800] for t in turns])
        sims = _cosine_matrix(entry_vecs, turn_vecs)
        return sims.argmax(axis=1).tolist()

    def purify(self, entries: List[str], turns: List[Dict], user_name: str,
               turn_ids_per_entry: Optional[List[List[int]]] = None) -> tuple:
        """Returns (purified_entries, grounded_turn_ids)."""
        if not entries:
            return [], []
        turn_ids = self.ground_to_turns(entries, turns)
        pairs = [(self._turn_text(turns[t]), e) for e, t in zip(entries, turn_ids)]
        verdicts = self.nli.check_batch(pairs)

        kept, kept_turns = [], []
        for entry, tid, verdict in zip(entries, turn_ids, verdicts):
            if verdict == "entailment":
                kept.append(entry)
                kept_turns.append([tid])
                continue
            # atomic decomposition + per-fact NLI + LLM adjudication
            decomp = self.llm.ask(prompts.P_DECOMP.format(memory_entry=entry, user_name=user_name),
                                  max_tokens=768)
            facts = split_lines(decomp) or [entry]
            fact_verdicts = self.nli.check_batch([(self._turn_text(turns[tid]), f) for f in facts])
            good_facts = []
            for fact, fv in zip(facts, fact_verdicts):
                if fv == "entailment":
                    good_facts.append(fact)
                    continue
                adj = self.llm.ask(prompts.P_ADJUDICATE.format(
                    source_turn=self._turn_text(turns[tid]), atomic_fact=fact), max_tokens=512)
                obj = extract_json_obj(adj) or {}
                action = str(obj.get("action", "DISCARD")).upper()
                if action == "REVISE" and obj.get("revised_fact"):
                    good_facts.append(str(obj["revised_fact"]))
            if good_facts:
                kept.append("; ".join(good_facts))
                kept_turns.append([tid])
        return kept, kept_turns

    # ---------------- stage 4 ----------------

    def verify_relations(self, edges: List[tuple], purified: List[str], turns: List[Dict]) -> List[tuple]:
        if not edges or not purified:
            return []
        flat = "\n".join(f"- {m}" for m in purified)
        valid = []
        for event_a, rel, event_b in edges:
            probe = self.llm.ask(prompts.P_PROBE.format(event_A=event_a, event_B=event_b,
                                                        relation_type=rel), max_tokens=256).strip()
            if not probe:
                continue
            ans = self.llm.ask(prompts.P_FLAT_ANSWER.format(memories=flat, question=probe),
                               max_tokens=256).strip()
            if "UNANSWERABLE" not in ans.upper():
                continue  # relation already implicit in flat memory; discard edge
            dialogue = "\n".join(self._turn_text(t) for t in turns)[:15000]
            verdict = self.llm.ask(prompts.P_REL_VERIFY.format(
                dialogue=dialogue, event_A=event_a, event_B=event_b, relation_type=rel),
                max_tokens=16).strip().lower()
            if verdict.startswith("yes"):
                valid.append((event_a, rel, event_b))
        return valid

    # ---------------- full session ----------------

    def process_session(self, session: Dict, user_name: str) -> MemoryGraph:
        turns = session["dialogue"]
        graph = MemoryGraph()

        # stage 1 (detail extraction and event abstraction in parallel roles)
        details = self.extract_details(turns)
        clusters = self.cluster_turns(turns)
        events = self.abstract_events(turns, clusters, user_name)
        edges = self.infer_relations(events)

        # stage 2
        draft = self.initial_draft(turns, user_name)
        missed = self.completeness_check(draft, events)
        entries = draft + [e.text for e in missed]

        # stage 3
        purified, grounded = self.purify(entries, turns, user_name)

        # stage 4
        valid_edges = self.verify_relations(edges, purified, turns)

        # detail anchoring via turn IDs
        graph.nodes = purified
        graph.edges = valid_edges
        for idx, tids in enumerate(grounded):
            anchored = []
            for tid in tids:
                anchored.extend(details.get(tid, []))
            if anchored:
                graph.attrs[idx] = anchored
        graph.stats = {
            "n_turns": len(turns), "n_clusters": len(clusters), "n_events": len(events),
            "n_draft": len(draft), "n_recovered": len(missed), "n_purified": len(purified),
            "n_edges_inferred": len(edges), "n_edges_valid": len(valid_edges),
            "llm_calls": self.llm.n_calls,
            "prompt_tokens": self.llm.prompt_tokens,
            "completion_tokens": self.llm.completion_tokens,
        }
        return graph

    @staticmethod
    def _turn_text(turn: Dict) -> str:
        return f"[{turn.get('timestamp', '')}]{turn['role']}: {turn['content']}"
