"""Content module: retrieval-backed response planning."""
from .llm import query_planner_llm, query_classifier_llm
from .memory_store import MemoryStore
from .web_search import search as web_search


def classify_message(user_message: str) -> str:
    prompt = f"""Classify the following user message into exactly one of these three categories:

    CONVERSATIONAL — use this when the message:
        - Is a greeting, introduction, small talk, or social exchange
        - Is a transitional signal that announces a coming question WITHOUT stating the question itself
          Examples: "I have some questions", "I'd like to ask you about X", "I want to know more about Y",
                    "I have one more question", "I have one more question before I say goodbye",
                    "Thanks, that helps, but I have one more question", "Before I go, I have something to ask"
        - KEY TEST: Does the message contain an actual answerable question? If NOT, it is CONVERSATIONAL.
          A message that only says "I have a question" or "I'd like to ask something" has NOT asked anything yet.

    FACTUAL — use this when the message:
        - Contains a direct, specific, answerable question requiring knowledge
        - The question itself is fully stated in the message (not just announced)
        - Examples: "What is your stance on the IMF loan?", "Where did you go to school?"

    HYBRID — use this when the message:
        - Contains BOTH a social/conversational element AND a fully stated factual question
        - Both must be present in the same message — the question must be actually asked, not merely signaled

    CRITICAL RULE: If the user says they have a question but has not yet asked it, that is CONVERSATIONAL.
    The presence of phrases like "I have one more question", "I want to ask you something", or
    "before I go I have a question" does NOT make a message FACTUAL — the question must actually appear.

    User message: {user_message}

    Reply with a single word only — CONVERSATIONAL, FACTUAL, or HYBRID:"""

    result = query_classifier_llm(prompt).strip().upper()

    # Guard against unexpected output
    if result not in {"CONVERSATIONAL", "FACTUAL", "HYBRID"}:
        return "FACTUAL"  # safe default — better to over-retrieve than under-retrieve

    return result
    

def check_sufficiency(user_message: str, retrieved: list[dict]) -> str | None:
    """Decide if retrieved chunks are sufficient, or return a search query.

    Returns:
        None — if evidence is sufficient (no web search needed).
        str  — an optimized search query to send to Tavily.
    """
    evidence_str = _format_evidence(retrieved)

    prompt = f"""You are a retrieval quality judge for an AI agent playing William Ruto (President of Kenya).

    Your job is to decide whether to use memory or trigger a web search. Apply this logic in order:

    ── STEP 1: Check for recency signals ────────────────────────────────────────────────────────
    If the user's question contains ANY of the following words or phrases:
      "latest", "recent", "recently", "current", "currently", "now", "right now",
      "today", "this week", "this month", "this year", "new", "update", "just"
    → ALWAYS return a web search query, regardless of what is in the retrieved evidence.
    Reason: the memory store is static and cannot answer what "latest" or "current" means.
    Even if the memory has related content, it may be outdated — the user explicitly asked for fresh info.

    ── STEP 2: If no recency signals — check if memory covers the topic ────────────────────────
    Answer SUFFICIENT if ANY of the following are true:
    - The evidence contains facts, names, dates, context, or events that substantially address the question
    - The question is about Ruto's biography, career history, political rise, or past actions
      and the evidence covers any part of it (partial coverage is still coverage)
    - The evidence allows a factually grounded answer, even if not exhaustive

    Return a search query only if:
    - The retrieved evidence is entirely off-topic or contains no or very little relevant information at all

    ── STEP 3: Format your answer ───────────────────────────────────────────────────────────────
    If SUFFICIENT: reply with exactly the word: SUFFICIENT
    If a search is needed: reply with a short web search query (5-10 words) [mentioning "William Ruto" if relevant]

    User question: {user_message}

    Retrieved evidence:
    {evidence_str}

    Your answer (one line only):"""

    result = query_planner_llm(prompt).strip()

    # If the model says SUFFICIENT (with any casing), no search needed
    if result.upper() == "SUFFICIENT":
        return None

    # Guard: some models prefix their response with "INSUFFICIENT:" before the search query.
    # Strip that prefix so we get only the clean search query for Tavily.
    if result.upper().startswith("INSUFFICIENT"):
        colon_idx = result.find(":")
        if colon_idx != -1:
            result = result[colon_idx + 1:].strip().strip('"')
        else:
            result = result[len("INSUFFICIENT"):].strip().strip('"')

    # If nothing useful remains after stripping, return None to skip the search
    return result if result else None


def make_conversational_plan(user_message: str) -> str:
    prompt = f"""You are a content planner for a conversational AI agent playing William Ruto, President of Kenya.

    The user has sent a social or conversational message that requires no factual knowledge to answer.
    Your job is to write a brief content plan (1-3 bullets) that tells the agent exactly what to say and how.

    GENERAL PRINCIPLES — apply these to any conversational message:
    - Match the energy and register of the message: a casual "Hey" gets a brief warm reply, not a speech
    - Never over-explain or volunteer unsolicited policy content
    - Stay in character: composed, warm, presidential — not stiff or robotic
    - Keep it human: respond the way a thoughtful, confident person would in a real conversation
    - Do NOT retrieve facts, cite events, or introduce policy unless the user explicitly asked for it
    - Maximum 1-3 bullets — concise plans only

    SPECIFIC RULES by message type (use whichever fits, these are not exhaustive):
    - Greeting ("Hey", "Hello", "Hi"): return a warm, brief acknowledgment and invite them to begin
    - Introduction request ("Introduce yourself", "Who are you?", "Tell me about yourself"):
        acknowledge the request, introduce by name and title (President of Kenya), add one brief line
        about being open to conversation — nothing more
    - Small talk / pleasantry ("How are you?", "Nice to meet you", "Good to have you here"):
        respond naturally and briefly in kind; keep it human and grounded
    - Acknowledgment with no new question ("Okay", "I see", "That makes sense", "Thanks"):
        give a brief, warm acknowledgment and invite them to continue or ask their next question
    - Transitional signal — user announces a coming question WITHOUT having asked it yet
      ("I have one more question", "I'd like to ask you something", "Before I go, I have a question"):
        do NOT introduce yourself or give any content; simply acknowledge and invite them to ask —
        e.g. "Of course, please go ahead" or "I'm listening, what would you like to know?"

    User message: {user_message}

    Content plan:"""

    return query_planner_llm(prompt, temperature=0.0)


def _format_evidence(retrieved: list[dict]) -> str:
    if not retrieved:
        return "No relevant information was found in memory."

    evidence_str = ""
    for i, chunk in enumerate(retrieved, start=1):
        source_label = chunk["topic"]
        if source_label == "web_search":
            title = chunk.get("metadata", {}).get("title", "Web")
            source_label = f"web_search — {title}"
        evidence_str += f"[Evidence {i}] (Source: {source_label})\n"
        evidence_str += f"  {chunk['text']}\n\n"
    return evidence_str


def make_factual_plan(user_message: str, retrieved: list[dict]) -> str:
    evidence_str = _format_evidence(retrieved)

    prompt = f"""You are a content planner for a conversational AI agent playing William Ruto.

    The user has asked a factual question. Build a grounded content plan using ONLY the retrieved evidence.

    Rules:
    1. Use ONLY information explicitly present in the evidence — no additions, no assumptions
    2. 3-5 bullets maximum, each tied to a specific evidence source
    3. If evidence is insufficient, write exactly one bullet: "Acknowledge the gap honestly in one sentence — do not speculate"
    4. Do not include social pleasantries — this plan is purely factual substance

    User message: {user_message}

    Retrieved evidence:
    {evidence_str}

    Content plan:"""

    return query_planner_llm(prompt, temperature=0.0)

def make_hybrid_plan(user_message: str, retrieved: list[dict]) -> str:
    evidence_str = _format_evidence(retrieved)

    prompt = f"""You are a content planner for a conversational AI agent playing William Ruto.

    The user's message contains both a social element and a real question. The plan must handle both.

    Rules:
    1. First bullet must ALWAYS be: briefly acknowledge the social part in one sentence — warm but composed
    2. Remaining 2-4 bullets cover the factual question using ONLY the retrieved evidence
    3. Each factual bullet must reference its evidence source
    4. If evidence for the factual part is insufficient, note it honestly — do not speculate
    5. Total plan: 3-5 bullets maximum

    User message: {user_message}

    Retrieved evidence:
    {evidence_str}

    Content plan:"""

    return query_planner_llm(prompt, temperature=0.0)



def _augment_with_web_search(user_message: str, retrieved: list[dict]) -> list[dict]:
    """Check if retrieved chunks are sufficient; if not, fetch from the web.

    Returns the (possibly augmented) list of evidence chunks.
    """
    search_query = check_sufficiency(user_message, retrieved)

    if search_query is None:
        # Evidence is sufficient — no web search needed
        print(f"[Web Search] Memory sufficient. No need to search.")
        return retrieved

    # Evidence is insufficient — search the web
    print(f"[Web Search] Memory insufficient. Searching: \"{search_query}\"")
    web_results = web_search(query=search_query, max_results=3)

    if web_results:
        print(f"[Web Search] Found {len(web_results)} web results. Merging with memory.")
    else:
        print("[Web Search] No web results found. Proceeding with memory only.")

    # Merge: memory chunks first, then web results appended at the end
    return retrieved + web_results


def content_step(memory: MemoryStore, user_message: str) -> dict:
    """Classify, retrieve if needed, and produce a typed content plan."""

    message_type = classify_message(user_message)
    retrieved = []

    if message_type == "CONVERSATIONAL":
        plan = make_conversational_plan(user_message)

    elif message_type == "FACTUAL":
        retrieved = memory.retrieve(user_message)
        retrieved = _augment_with_web_search(user_message, retrieved)
        plan = make_factual_plan(user_message, retrieved)

    else:  # HYBRID
        retrieved = memory.retrieve(user_message)
        retrieved = _augment_with_web_search(user_message, retrieved)
        plan = make_hybrid_plan(user_message, retrieved)

    return {"content_plan": plan, "retrieved_chunks": retrieved}