"""Stateless agent turn logic — shared by both the CLI runner and the FastAPI server.

This module has no I/O and no side effects. It takes inputs and returns outputs,
which means it can be called identically from a terminal loop or an HTTP handler.
"""

from .content_module import content_step
from .style_module import stylize
from .run_agent import should_end, add_signoff
from .memory_store import MemoryStore


def process_turn(
    memory: MemoryStore,
    user_message: str,
    last_agent_response: str,
) -> dict:
    """Process one conversation turn and return the agent's reply.

    Args:
        memory:              The pre-loaded MemoryStore (shared across all turns).
        user_message:        The raw text the user just sent.
        last_agent_response: The agent's previous reply, used by should_end() to
                             detect closing signals in context.

    Returns a dict:
        {
            "response": str,   # the agent's reply text
            "ended":    bool,  # True if this turn triggered a conversation sign-off
        }
    """
    # ── 1. Check for conversation end ────────────────────────────────────────────
    # Do this before the content pipeline to avoid burning API calls on a
    # response that will never be shown.
    if should_end(user_message, last_agent_response):
        signoff = add_signoff(last_agent_response)
        return {"response": signoff, "ended": True}

    # ── 2. Run the full RAG pipeline ─────────────────────────────────────────────
    # classify → (retrieve) → (sufficiency check) → (web search) → plan → stylize
    result = content_step(memory=memory, user_message=user_message)
    response = stylize(
        user_message=user_message,
        content_plan=result["content_plan"],
    )

    return {"response": response, "ended": False}
