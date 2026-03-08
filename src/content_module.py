"""Content module: retrieval-backed response planning."""
from .llm import query_planner_llm, query_classifier_llm
from .memory_store import MemoryStore


def classify_message(user_message: str) -> str:
    prompt = f"""Classify the following user message into exactly one of these three categories:
    CONVERSATIONAL — a greeting, introduction, small talk, or social exchange with no factual question,
                     OR a transitional statement announcing upcoming questions
                     (e.g. "I have some questions", "I'd like to ask you about X", "I want to know more about Y")
                     These are setups, not actual questions — treat them as CONVERSATIONAL.
    FACTUAL — a direct, specific question or request that requires knowledge to answer
    HYBRID — contains both a social element AND a real direct question in the same message

    User message: {user_message}

    Reply with a single word only — CONVERSATIONAL, FACTUAL, or HYBRID:"""

    result = query_classifier_llm(prompt).strip().upper()
    
    # Guard against unexpected output
    if result not in {"CONVERSATIONAL", "FACTUAL", "HYBRID"}:
        return "FACTUAL"  # safe default — better to over-retrieve than under-retrieve
    
    return result
    

def make_conversational_plan(user_message: str) -> str:
    prompt = f"""You are a content planner for a conversational AI agent playing William Ruto.

    The user has sent a greeting, introduction, or social message that requires no factual knowledge to answer.

    User message: {user_message}

    Write a content plan of 1-2 bullets that instructs the agent to:
    - Acknowledge the user naturally and warmly but with presidential composure
    - Introduce himself briefly if asked, using only: his name, title (President of Kenya), and one line about his purpose in this conversation
    - Close with a natural, open invitation for their questions — something along the lines of
      "What questions do you have?" or "Feel free to ask whatever is on your mind" —
      keep it simple and human, not transactional
    - Keep it brief — this is a social exchange, not a policy opportunity

    Content plan:"""

    return query_planner_llm(prompt, temperature=0.0)


def _format_evidence(retrieved: list[dict]) -> str:
    if not retrieved:
        return "No relevant information was found in memory."
    
    evidence_str = ""
    for i, chunk in enumerate(retrieved, start=1):
        evidence_str += f"[Evidence {i}] (Topic: {chunk['topic']})\n"
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



def content_step(memory: MemoryStore, user_message: str) -> dict:
    """Classify, retrieve if needed, and produce a typed content plan."""
    
    message_type = classify_message(user_message)
    retrieved = []

    if message_type == "CONVERSATIONAL":
        plan = make_conversational_plan(user_message)

    elif message_type == "FACTUAL":
        retrieved = memory.retrieve(user_message)
        plan = make_factual_plan(user_message, retrieved)

    else:  # HYBRID
        retrieved = memory.retrieve(user_message)
        plan = make_hybrid_plan(user_message, retrieved)

    return {"content_plan": plan, "retrieved_chunks": retrieved}