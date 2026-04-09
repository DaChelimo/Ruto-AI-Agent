"""Stateless agent turn logic — shared by both the CLI runner and the FastAPI server.

This module has no I/O and no side effects beyond mutating the conv_memory object
it receives.  It can be called identically from a terminal loop or an HTTP handler.
"""

from .content_module import content_step
from .conversation_memory import ConvMemory
from .memory_store import MemoryStore
from .run_agent import should_end, add_signoff
from .style_module import stylize


def process_turn(
    memory: MemoryStore,
    user_message: str,
    last_agent_response: str,
    conv_memory: ConvMemory,
) -> dict:
    """Process one conversation turn and return the agent's reply.

    Mutates conv_memory in place by appending the completed turn after a
    successful response.  The caller (api.py / run_agent.py) owns the
    conv_memory object and holds it across turns.

    Args:
        memory:              Pre-loaded MemoryStore (shared across all sessions).
        user_message:        Raw text the user just sent.
        last_agent_response: Agent's previous reply — used by should_end().
        conv_memory:         Per-session conversational memory (mutated here).

    Returns a dict:
        {
            "response":       str,        # the agent's final reply
            "ended":          bool,        # True if this turn ends the conversation
            "pipeline_steps": list[str],   # labels of steps that ran
            "message_type":   str,         # CONVERSATIONAL | FACTUAL | HYBRID | ""
        }
    """
    # ── 1. Check for conversation end ────────────────────────────────────────────
    # Run before the content pipeline so we don't burn API calls on a response
    # that will never be shown.
    if should_end(user_message, last_agent_response):
        signoff = add_signoff(last_agent_response)
        # Record the closing exchange in memory before the session is discarded
        # (harmless — api.py cleans up the session dict right after this returns)
        conv_memory.add_turn(user_message, signoff)
        return {
            "response":       signoff,
            "ended":          True,
            "pipeline_steps": [],
            "message_type":   "",
        }

    # ── 2. Run the full RAG pipeline ─────────────────────────────────────────────
    # classify → (retrieve) → (sufficiency check) → (web search) → plan → stylize
    result = content_step(
        memory=memory,
        user_message=user_message,
        conv_memory=conv_memory,
    )
    pipeline_steps: list[str] = result["pipeline_steps"]

    response = stylize(
        user_message=user_message,
        content_plan=result["content_plan"],
        response_mode=result.get("response_mode", "standard"),
    )

    # ── 3. Mark the final step ───────────────────────────────────────────────────
    pipeline_steps.append("STYLING RESPONSE")

    # ── 4. Record this turn in conversational memory ─────────────────────────────
    # add_turn() appends the completed turn and triggers compression automatically
    # if the exact buffer exceeds MAX_EXACT_TURNS.
    conv_memory.add_turn(user_message, response)

    return {
        "response":       response,
        "ended":          False,
        "pipeline_steps": pipeline_steps,
        "message_type":   result["message_type"],
    }
