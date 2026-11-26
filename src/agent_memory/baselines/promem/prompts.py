"""Prompt templates from the ProMem paper (Appendix F, Figures 9-15), verbatim."""

# Figure 9 — fine-grained detail extraction (P_detail)
P_DETAIL = """You are a micro-detail extractor.
Dialogue Turn: {current_turn}
Your task: Extract all fine-grained, atomic details from this specific dialogue turn. Focus heavily on specific entities, times, locations, and explicit object names.
Rules:
- Do NOT summarize the overarching event. Only extract the specific attribute fragments.
- Examples of good details: "10:00 AM", "at MIT", "prefers PyTorch", "AI student".
- Output as a simple comma-separated list of short phrases.
Output format: ["detail 1", "detail 2", ...]"""

# Figure 10 — topic-based event abstraction
P_EVENT = """You are an overarching event summarizer.
We have grouped several dialogue turns that share the same topic.
Topic-clustered Dialogue: {clustered_turns}
Your task: Abstract the core "Memory Events" from this clustered dialogue. A memory event should represent a macro-level action, status, or narrative of the user ("{user_name}").
Rules:
- Ignore fragmented times or locations unless they define the event. Focus on the "Big Picture".
- Write each event as a clear, standalone sentence in the third person.
Output format: Just a list of macro events, one per line."""

# Figure 11 — relation edge inference
P_RELATION = """You are a structural logic analyzer.
Below is a list of macro memory events extracted from a user's dialogue history.
Extracted Events: {event_list}
Your task: Analyze the logical relationships between these events. Identify any explicit or highly probable "Causal" (A leads to B) or "Temporal" (A happens before B) connections.
Rules:
- Only infer relationships that are strongly supported by common sense or the event context.
- If two events are unrelated, do not force a connection.
Output format: Output strictly as a JSON list of arrays:
[ ["Event_A_ID", "relation_type", "Event_B_ID"], ... ]"""

# Figure 12 — initial memory draft
P_DRAFT = """You are a Personal Fact Extraction Engine.
Your task is to analyze conversation logs to identify and catalog every piece of factual information regarding the user ("{user_name}"). The goal is to convert dialogue fragments into explicit, self-contained statements to form an initial memory draft.
Important instructions:
1. Process the user messages and decide whether they contain factual information.
- Extract ANY assertion of fact, action, state, or preference.
- If no useful information is provided (e.g., pure greeting or filler), skip it.
2. Perform light contextual completion so that each fact is a clear standalone statement.
Example:
"User: Went to the gym." -> "User went to gym."
3. CRITICAL RULE: Always use the name "{user_name}" as the subject instead of "User", "He", "She", or "I".
Reminder: Be exhaustive. Output the extracted facts as a clear list.

Conversation logs:
{dialogue}"""

# Figure 13 — atomic decomposition (P_decomp)
P_DECOMP = """You are an expert linguist and data parser.
Current Memory Entry: {memory_entry}
Your task: Decompose the complex memory entry above into a list of independent, fine-grained "atomic facts".
Rules:
- Each atomic fact MUST contain only ONE single piece of information (e.g., one action, one time, or one location).
- Each atomic fact must be a complete sentence with a clear subject ("{user_name}").
- Do not add any external information or guess meanings. Just split the existing sentence.
Example:
Entry: "Sarah has a meeting on Sep. 25 at the Oracle office."
Output:
1. Sarah has a meeting.
2. The meeting is on Sep. 25.
3. The meeting is at the Oracle office.
Output format: Just a list of atomic facts, one per line. No other text."""

# Figure 14 — LLM intervention and correction
P_ADJUDICATE = """You are a strict factual adjudicator.
An extracted atomic fact has been flagged as potentially hallucinated or inaccurate by an NLI system.
Source Dialogue Turn: {source_turn}
Flagged Atomic Fact: {atomic_fact}
Your task: Read the source dialogue turn carefully and evaluate the flagged fact. Choose one of two actions:
1. REVISE: If the fact contains a minor error (e.g., wrong date or slight entity mismatch) but the core intent exists in the dialogue, revise it to perfectly match the dialogue.
2. DISCARD: If the fact is a pure hallucination or unsupported inference not mentioned in the dialogue, discard it entirely.
Output format: Output strictly as a JSON object:
{{"action": "REVISE" or "DISCARD", "revised_fact": "your revised text or null"}}"""

# Figure 15 — relation probe questions (P_probe)
P_PROBE = """You are an analytical investigator building a memory graph.
We have identified two memory events and a potential logical relation between them.
Event A: {event_A}
Event B: {event_B}
Inferred Relation: {relation_type} (from Event A to Event B)
Your task: Generate a specific probing question to verify if this relation truly exists in the user's history.
Rules:
- The question must focus on the LINK between the two events.
- Example: If Event A is "User likes Python" and Event B is "User learns PyTorch", with relation "Causal", the question should be: "Why did the user choose to learn PyTorch?" or "Is the user learning PyTorch because they like Python?"
Output format: Just output the generated question. No other text."""

# Flat-memory answerability test (paper Sec. 3.5: if the agent can answer with
# flat entries, the relation is already implicit and the edge is discarded)
P_FLAT_ANSWER = """You are a helpful assistant. Below is a flat list of memory entries about a user.
Memory entries:
{memories}

Question: {question}

Answer the question using ONLY the memory entries above. If the entries do not contain enough information to answer, respond exactly with "UNANSWERABLE". Keep the answer short."""

# Relation support verification against the original dialogue
P_REL_VERIFY = """You are a strict fact verifier.
Dialogue excerpts:
{dialogue}

Claim: The following relation holds between two events in the user's history:
Event A: {event_A}
Event B: {event_B}
Relation: {relation_type} (from Event A to Event B)

Is this relation supported by the dialogue excerpts? Answer only "yes" or "no"."""
