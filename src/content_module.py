"""Content module: retrieval-backed response planning.

Memory injection strategy (deliberate design decisions):

CLASSIFIER  — receives only the last 2 turns from recent history (not the full
              summary).  Reason: the classifier uses the cheapest/fastest model
              (open-mistral-nemo) and only needs enough context to disambiguate
              pronouns or understand that "them" refers to a recent topic.  The
              full summary is not needed here and would increase latency/cost on
              the hot path for no benefit.

PLANNER     — receives the full memory block (summary + all recent exact turns)
              as a dedicated reference-resolution context block.  Reason: the
              planner has to decide WHAT to say and must correctly resolve "the
              floods", "what you mentioned", or "them" to specific topics.  The
              full context is essential here.  The memory block is clearly
              labelled as for reference resolution only — factual content must
              still come exclusively from retrieved evidence.

SUFFICIENCY — receives the full memory block.  Reason: without context, an
              ambiguous query like "What are you doing to mitigate them?" may
              retrieve wrong or insufficient chunks.  The sufficiency judge
              needs to know what "them" refers to in order to evaluate whether
              the retrieved evidence covers that topic.

STYLIZER    — receives NO memory.  Reason: by the time stylize() runs, the
              content plan already has fully resolved references.  Adding memory
              to the style layer would add noise without improving the output,
              and the style layer is intentionally grounding-free (tone only).
"""

from __future__ import annotations

from .conversation_memory import ConvMemory
from .llm import query_planner_llm, query_classifier_llm
from .memory_store import MemoryStore
from .web_search import search as web_search


# ── Memory injection helpers ────────────────────────────────────────────────────

def _memory_block(conv_memory: ConvMemory, full: bool = True) -> str:
    """Return a formatted memory block for prompt injection, or empty string."""
    if conv_memory.is_empty():
        return ""
    if full:
        return conv_memory.format_for_prompt()
    else:
        return conv_memory.format_last_n_for_prompt(n=2)


def _wrap_memory_for_classifier(conv_memory: ConvMemory) -> str:
    """Lightweight context block for the classifier (last 2 turns only)."""
    block = _memory_block(conv_memory, full=False)
    if not block:
        return ""
    return (
        "\nCONVERSATION CONTEXT (use only to resolve references in the current message):\n"
        f"{block}\n"
    )


def _wrap_memory_for_planner(conv_memory: ConvMemory) -> str:
    """Full memory block for planners — labelled to prevent evidence contamination."""
    block = _memory_block(conv_memory, full=True)
    if not block:
        return ""
    return (
        "\nCONVERSATION MEMORY (for reference resolution and topic continuity ONLY):\n"
        f"{block}\n"
        "IMPORTANT: The memory above is provided so you can resolve pronouns and "
        "elliptical references (\"them\", \"it\", \"those events\", \"what you mentioned\") "
        "back to specific entities or topics. It is NOT a source of factual content. "
        "All factual claims in the plan must come exclusively from the retrieved evidence.\n"
    )


def _wrap_memory_for_sufficiency(conv_memory: ConvMemory) -> str:
    """Full memory block for the sufficiency judge."""
    block = _memory_block(conv_memory, full=True)
    if not block:
        return ""
    return (
        "\nCONVERSATION MEMORY (use to resolve ambiguous references in the user question):\n"
        f"{block}\n"
    )


# ── Core pipeline functions ─────────────────────────────────────────────────────

def classify_message(user_message: str, conv_memory: ConvMemory) -> str:
    """Classify the user message as CONVERSATIONAL, FACTUAL, or HYBRID.

    Receives the last 2 turns for lightweight reference disambiguation.
    Example: "What are you doing to mitigate them?" is correctly classified as
    FACTUAL once the classifier can see that "them" refers to the Nairobi floods.
    """
    memory_ctx = _wrap_memory_for_classifier(conv_memory)

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
{memory_ctx}
    User message: {user_message}

    Reply with a single word only — CONVERSATIONAL, FACTUAL, or HYBRID:"""

    result = query_classifier_llm(prompt).strip().upper()

    if result not in {"CONVERSATIONAL", "FACTUAL", "HYBRID"}:
        return "FACTUAL"  # safe default — better to over-retrieve than under-retrieve

    return result


def check_sufficiency(
    user_message: str,
    retrieved: list[dict],
    conv_memory: ConvMemory,
) -> str | None:
    """Decide if retrieved chunks are sufficient, or return a search query.

    Receives full memory so the judge can resolve ambiguous references in the
    user message when evaluating whether the retrieved evidence is on-topic.

    Returns:
        None — if evidence is sufficient (no web search needed).
        str  — an optimised search query to send to Tavily.
    """
    evidence_str = _format_evidence(retrieved)
    memory_ctx = _wrap_memory_for_sufficiency(conv_memory)

    prompt = f"""You are a retrieval quality judge for an AI agent playing William Ruto (President of Kenya).

    Your job is to decide whether to use memory or trigger a web search. Apply this logic in order:
{memory_ctx}
    ── STEP 1: Check for recency signals ────────────────────────────────────────────────────────
    If the user's question contains ANY of the following words or phrases:
      "latest", "recent", "recently", "current", "currently", "now", "right now",
      "today", "this week", "this month", "this year", "new", "update", "just"
    → ALWAYS return a web search query, regardless of what is in the retrieved evidence.
    Reason: the memory store is static and cannot answer what "latest" or "current" means.
    Even if the memory has related content, it may be outdated — the user explicitly asked for fresh info.

    ── STEP 2: If no recency signals — check if memory covers the topic ────────────────────────
    Use the conversation memory above (if present) to resolve any ambiguous references in the
    user's question before evaluating the retrieved evidence.
    For example, if the user asks "What are you doing to mitigate them?" and the memory shows
    "them" refers to the Nairobi floods, evaluate whether the evidence covers flood mitigation.

    Answer SUFFICIENT if ANY of the following are true:
    - The evidence contains facts, names, dates, context, or events that substantially address the question
    - The question is about Ruto's biography, career history, political rise, or past actions
      and the evidence covers any part of it (partial coverage is still coverage)
    - The evidence allows a factually grounded answer, even if not exhaustive

    Return a search query only if:
    - The retrieved evidence is entirely off-topic or contains no relevant information at all

    ── STEP 3: Format your answer ───────────────────────────────────────────────────────────────
    If SUFFICIENT: reply with exactly the word: SUFFICIENT
    If a search is needed: reply with a short web search query (5-10 words) mentioning "William Ruto"
    where relevant, and with any ambiguous pronouns replaced by their resolved antecedents.

    User question: {user_message}

    Retrieved evidence:
    {evidence_str}

    Your answer (one line only):"""

    result = query_planner_llm(prompt).strip()

    if result.upper() == "SUFFICIENT":
        return None

    # Guard: some models prefix the response with "INSUFFICIENT:" before the query.
    if result.upper().startswith("INSUFFICIENT"):
        colon_idx = result.find(":")
        if colon_idx != -1:
            result = result[colon_idx + 1:].strip().strip('"')
        else:
            result = result[len("INSUFFICIENT"):].strip().strip('"')

    return result if result else None


def make_conversational_plan(user_message: str, conv_memory: ConvMemory) -> str:
    memory_ctx = _wrap_memory_for_planner(conv_memory)

    prompt = f"""You are a content planner for a conversational AI agent playing William Ruto, President of Kenya.

    The user has sent a social or conversational message that requires no factual knowledge to answer.
    Your job is to write a brief content plan (1-3 bullets) that tells the agent exactly what to say and how.
{memory_ctx}
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


def make_factual_plan(
    user_message: str,
    retrieved: list[dict],
    conv_memory: ConvMemory,
) -> str:
    evidence_str = _format_evidence(retrieved)
    memory_ctx = _wrap_memory_for_planner(conv_memory)

    prompt = f"""You are a content planner for a conversational AI agent playing William Ruto.

    The user has asked a factual question. Build a grounded content plan using ONLY the retrieved evidence.
{memory_ctx}
    Rules:
    1. Use the conversation memory (if present) ONLY to resolve ambiguous references in the user's question.
       For example, if the user asks "What are you doing to mitigate them?", use the memory to determine
       what "them" refers to, then answer that specific topic using the evidence.
    2. Use ONLY information explicitly present in the retrieved evidence — no additions, no assumptions,
       and no facts drawn from the conversation memory.
    3. 3-5 bullets maximum, each tied to a specific evidence source.
    4. If evidence is insufficient, write exactly one bullet:
       "Acknowledge the gap honestly in one sentence — do not speculate"
    5. Do not include social pleasantries — this plan is purely factual substance.

    User message: {user_message}

    Retrieved evidence:
    {evidence_str}

    Content plan:"""

    return query_planner_llm(prompt, temperature=0.0)


def make_hybrid_plan(
    user_message: str,
    retrieved: list[dict],
    conv_memory: ConvMemory,
) -> str:
    evidence_str = _format_evidence(retrieved)
    memory_ctx = _wrap_memory_for_planner(conv_memory)

    prompt = f"""You are a content planner for a conversational AI agent playing William Ruto.

    The user's message contains both a social element and a real question. The plan must handle both.
{memory_ctx}
    Rules:
    1. First bullet must ALWAYS be: briefly acknowledge the social part in one sentence — warm but composed.
    2. Use the conversation memory (if present) ONLY to resolve ambiguous references in the factual part.
       Do not use memory as a source of facts.
    3. Remaining 2-4 bullets cover the factual question using ONLY the retrieved evidence.
    4. Each factual bullet must reference its evidence source.
    5. If evidence for the factual part is insufficient, note it honestly — do not speculate.
    6. Total plan: 3-5 bullets maximum.

    User message: {user_message}

    Retrieved evidence:
    {evidence_str}

    Content plan:"""

    return query_planner_llm(prompt, temperature=0.0)


# ── Internal helpers ────────────────────────────────────────────────────────────

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


def _augment_with_web_search(
    user_message: str,
    retrieved: list[dict],
    pipeline_steps: list[str],
    conv_memory: ConvMemory,
) -> list[dict]:
    """Check sufficiency and conditionally run a web search.

    conv_memory is forwarded to check_sufficiency so the judge can resolve
    ambiguous references (e.g. "them" → Nairobi floods) before evaluating
    whether the retrieved evidence is on-topic.
    """
    pipeline_steps.append("CHECKING SUFFICIENCY")
    search_query = check_sufficiency(user_message, retrieved, conv_memory)

    if search_query is None:
        print("[Web Search] Memory sufficient. No search needed.")
        return retrieved

    pipeline_steps.append("WEB SEARCH")
    print(f'[Web Search] Memory insufficient. Searching: "{search_query}"')
    web_results = web_search(query=search_query, max_results=3)

    if web_results:
        print(f"[Web Search] Found {len(web_results)} result(s). Merging with memory.")
    else:
        print("[Web Search] No results found. Proceeding with memory only.")

    return retrieved + web_results


def content_step(
    memory: MemoryStore,
    user_message: str,
    conv_memory: ConvMemory,
) -> dict:
    """Classify, retrieve if needed, and produce a typed content plan.

    Returns:
        {
            "content_plan":     str,
            "retrieved_chunks": list[dict],
            "pipeline_steps":   list[str],
            "message_type":     str,         # CONVERSATIONAL | FACTUAL | HYBRID
        }
    """
    pipeline_steps: list[str] = []
    retrieved: list[dict] = []

    # ── Step 1: Classify (with lightweight memory context) ───────────────────
    pipeline_steps.append("CLASSIFYING MESSAGE")
    message_type = classify_message(user_message, conv_memory)

    if message_type == "CONVERSATIONAL":
        plan = make_conversational_plan(user_message, conv_memory)

    elif message_type == "FACTUAL":
        # ── Step 2: Retrieve ─────────────────────────────────────────────────
        pipeline_steps.append("SEARCHING MEMORY")
        retrieved = memory.retrieve(user_message)

        # ── Steps 3 (+4): Sufficiency check / optional web search ───────────
        retrieved = _augment_with_web_search(
            user_message, retrieved, pipeline_steps, conv_memory
        )

        # ── Step 5: Build factual plan ───────────────────────────────────────
        pipeline_steps.append("BUILDING CONTENT PLAN")
        plan = make_factual_plan(user_message, retrieved, conv_memory)

    else:  # HYBRID
        # ── Step 2: Retrieve ─────────────────────────────────────────────────
        pipeline_steps.append("SEARCHING MEMORY")
        retrieved = memory.retrieve(user_message)

        # ── Steps 3 (+4): Sufficiency check / optional web search ───────────
        retrieved = _augment_with_web_search(
            user_message, retrieved, pipeline_steps, conv_memory
        )

        # ── Step 5: Build hybrid plan ────────────────────────────────────────
        pipeline_steps.append("BUILDING CONTENT PLAN")
        plan = make_hybrid_plan(user_message, retrieved, conv_memory)

    # Note: "STYLING RESPONSE" is appended by agent.py after stylize() completes.
    return {
        "content_plan":     plan,
        "retrieved_chunks": retrieved,
        "pipeline_steps":   pipeline_steps,
        "message_type":     message_type,
    }
