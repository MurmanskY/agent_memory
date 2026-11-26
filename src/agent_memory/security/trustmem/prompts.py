"""Prompt templates: chunk processing, transition verifier (paper Fig. 5),
content-type analyzer (Mem-alpha memory_server), and transition error judge (paper App. D)."""

# Same unified chunk prompt as Mem-alpha training (config/prompts_wrt_datasource.yaml)
UNIFIED_CHUNK_PROMPT = """Remember the following content chunk by completing these steps:

1. **Core Memory Update**: Maintain an understanding of the user, or a summary of what the user is reading, or a set of classification rules summarized from the classification examples (label 1: meaning; label 2: meaning, etc.). Keep updates brief (a few sentences maximum).

2. **Memory Storage**:
   - **Episodic Memory**: Record user actions, user's friends' actions and assistant actions with timestamps (format: "At timestamp t, user did X")
   - **Semantic Memory**: Record key facts and information (format: "John is User's 18-year-old friend", "Harry Potter author: J.K. Rowling", "Sample: xxx; Label: xxx")

<new_chunk>
{context}
</new_chunk>

**Important**: Response limit is {max_new_tokens} tokens. Be concise and brief in all memory updates."""

# Query prompts per data source (from Mem-alpha config/prompts_wrt_datasource.yaml)
TTL_QUERY_PROMPT = """Your core memory is used to store the classification rules of the articles. Identify the meaning of each class and how the article is classified. Your semantic memory likely includes example labels. Fulfill the user's request with the core memory.

When answering questions, you should put the final prediction in the format of \\boxed{label}. Note that the label is a single number."""

BOOKSUM_QUERY_PROMPT = "Your core memory contains a comprehensive book summary. Use this information to fulfill the user's request."


def get_query_prompt(data_source: str):
    if "icl" in data_source or "ttl" in data_source or data_source == "pubmed-rct":
        return TTL_QUERY_PROMPT
    if "booksum" in data_source or "infbench" in data_source or "long_range" in data_source:
        return BOOKSUM_QUERY_PROMPT
    return None


def including_core(data_source: str) -> bool:
    # All Mem-alpha training/eval sources use core memory
    return True


# ---------------------------------------------------------------------------
# Memory Transition Verifier (TrustMem paper, Figure 5)
# ---------------------------------------------------------------------------

VERIFIER_PROMPT = """Role: You are an expert judge for chunk-level memory consolidation.

Goal: Given a compact view of one memory transition from Mi to Mi+1 after processing a chunk ci, evaluate whether the memory update is trustworthy according to:
1. Coverage: whether Mi+1 preserves the write-worthy key information from ci;
2. Preservation: whether important information already present in Mi is not deleted or overwritten without justification;
3. Faithfulness: whether newly added or modified memory content is supported by the chunk evidence or the previous memory state.

Inputs:
- Chunk Evidence: {chunk_text}
- Pre-transition Memory State: {pre_memory}
- Post-transition Memory State: {post_memory}
- Touched Memory Actions: {actions}

Required Output: Return only a JSON object:
{{
  "score": 0.0,
  "coverage_ok": true,
  "preservation_ok": true,
  "faithfulness_ok": true,
  "issues": ["brief issue labels"],
  "explanation": "brief explanation"
}}

Rules:
- Judge only from the provided chunk evidence, pre-transition memory, post-transition memory, and touched memory actions.
- A skipped action can still be valid if the relevant information was already correctly stored and preserved.
- Legitimate merging, rewriting, or compression is allowed as long as important facts are preserved.
- Penalize critical omissions, unsupported additions, corrupted facts, and unjustified overwrites or deletions.
- Use a continuous score in [0, 1], where 1 means excellent consolidation and 0 means clearly invalid consolidation.
- Do not output extra text outside the required JSON object."""

# ---------------------------------------------------------------------------
# Content-type analyzer (Mem-alpha memory_server analyze_function, R_type)
# ---------------------------------------------------------------------------

TYPE_ANALYZER_PROMPTS = {
    "core": """You are an expert memory analyst. Analyze the quality of core memory content.

The core memory is invalid if any of the following meets:
(1) The literal content "core memory" appears in the memory such as "This is core memory ...", "The core memory has been updated ...".
(2) The core memory is apparently a placeholder such as "Here we save the summary" while not stating what the "summary" is, "Here are some rules" and not stating what the "rules" are.

Otherwise, the core memory is valid.

Respond ONLY with a JSON code block in this exact format:
```json
{{
  "VALID": true/false,
  "ISSUES": [list any problems found],
  "EXPLANATION": "brief explanation of the assessment"
}}
```""",
    "semantic": """You are an expert memory analyst. Analyze the quality of semantic memory content.

Semantic memory should contain:
- Information or Knowledge about somebody or something
- Definitions, theories, principles, or explanations
- How-to knowledge or procedural information
- Research findings or established facts

Two other memories are Core memory (User Personalities) and Episodic memory (User Experiences). The information not suitable for these two memories should be considered as semantic memory.

Respond ONLY with a JSON code block in this exact format:
```json
{{
  "VALID": true/false,
  "ISSUES": [list any problems found],
  "EXPLANATION": "brief explanation of the assessment"
}}
```""",
    "episodic": """You are an expert memory analyst. Analyze the quality of episodic memory content.

Episodic memory should contain:
- Experiences or events
- Clear temporal information (when it happened)
- Contextual details (what happened)

Respond ONLY with a JSON code block in this exact format:
```json
{{
  "VALID": true/false,
  "ISSUES": [list any problems found],
  "EXPLANATION": "brief explanation of the assessment"
}}
```""",
}

# ---------------------------------------------------------------------------
# LLM judge for lme_train task reward (Mem-alpha reward manager)
# ---------------------------------------------------------------------------

LME_JUDGE_PROMPT = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

# Generative accuracy judge for squad/hotpotqa (Mem-alpha generative reward)
ACCURACY_PROMPT = """You are an expert at evaluating whether a generated answer correctly answers a question based on a gold answer.

Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {generated_answer}

Does the generated answer correctly answer the question? Respond with exactly one label:
<label>CORRECT</label> if the generated answer is correct (it may contain extra correct information)
<label>WRONG</label> if the generated answer is incorrect or contradicts the gold answer"""

# ---------------------------------------------------------------------------
# Memory Transition Error Judge (TrustMem paper, Appendix D)
# ---------------------------------------------------------------------------

ERROR_JUDGE_PROMPT = """Role: You are a strict but fair evaluator of memory-transition quality.

Goal: Given one chunk-level memory transition, judge whether the transition introduces a substantive memory error. The judge evaluates three semantic failure modes:
1. Omission: the updated memory misses key information from the input chunk.
2. Corruption: the memory operation changes, contradicts, misattributes, or mixes a fact supported by the input chunk or prior memory.
3. Hallucination: the memory operation introduces a substantive new fact unsupported by the input chunk or prior memory.

Inputs:
- Dataset Group: {dataset_group}
- Sample ID: {sample_id}
- Chunk Index: {chunk_idx}
- Prior Memory: {prior_memory}
- Input Chunk: {input_chunk}
- Memory Operations: {memory_operations}

Definitions:
- Key information includes facts, labels, constraints, preferences, entities, relations, dates, numbers, or outcomes needed for later question answering, classification, personalization, or faithful recall.
- A concise summary is allowed; skipping nonessential details alone is not counted as omission.
- Facts already supported by prior memory are not counted as hallucinations when they are simply preserved by an update.

Required Output: Return only a JSON object:
{{
  "omission": true,
  "corruption": false,
  "hallucination": false,
  "primary_error": "omission",
  "severity": "major",
  "confidence": 85,
  "rationale": "brief explanation",
  "evidence": ["supporting evidence strings"]
}}

Rules:
- Judge only from the provided prior memory, input chunk, and memory operations.
- Do not penalize harmless meta text, timestamps, assistant acknowledgements, or memory bookkeeping unless it changes a substantive fact.
- For classification/example chunks, labels and representative examples are important.
- For long document chunks, central entities, events, relations, and numeric/date facts are important; minor wording differences are not errors.
- For omission, ask whether a future assistant would likely answer incorrectly or fail the benchmark because the key information was not stored.
- If there are multiple major error types, set primary_error to "mixed".
- Use "major" only when the error would plausibly harm later recall or QA; use "minor" for small issues that probably would not matter.
- Use "none" when no substantive error is present.
- Do not output extra text outside the required JSON object."""
