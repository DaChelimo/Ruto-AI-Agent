"""Conversational memory: exact recent buffer + incremental structured summary.

Design philosophy:
- Keep the most recent MAX_EXACT_TURNS turns verbatim (exact recall).
- Older turns are compressed into a running structured summary optimised for
  reference resolution: pronoun anchors, entity tracking, topic continuity.
- Summary updates are INCREMENTAL — new overflow turns are merged into the
  existing summary rather than regenerating everything from scratch.
  This prevents summary drift over a long interview.
- The summary format is explicitly structured (not prose) so a future LLM
  can resolve "them", "it", "those events", "what we discussed earlier" to
  concrete antecedents.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import query_planner_llm

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_EXACT_TURNS = 4   # number of verbatim turns to keep before compressing


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    user: str
    agent: str


# ── Core class ─────────────────────────────────────────────────────────────────

class ConvMemory:
    """Rolling exact-turn buffer paired with an incremental structured summary.

    Usage
    -----
    mem = ConvMemory()

    # after each completed agent turn:
    mem.add_turn(user_message, agent_response)

    # embed in any prompt:
    context_block = mem.format_for_prompt()
    """

    def __init__(self) -> None:
        self.recent_turns: list[Turn] = []
        self.summary: str = ""          # structured summary of compressed turns

    # ── Public interface ────────────────────────────────────────────────────────

    def add_turn(self, user_message: str, agent_response: str) -> None:
        """Record one completed turn.  Triggers compression if buffer is full."""
        self.recent_turns.append(Turn(user=user_message, agent=agent_response))
        if len(self.recent_turns) > MAX_EXACT_TURNS:
            self._compress()

    def format_for_prompt(self, include_summary: bool = True) -> str:
        """Render the full memory block for embedding in planner / checker prompts."""
        parts: list[str] = []

        if include_summary and self.summary:
            parts.append("── EARLIER CONVERSATION SUMMARY ──")
            parts.append(self.summary)

        if self.recent_turns:
            parts.append("── RECENT EXACT TURNS ──")
            for i, turn in enumerate(self.recent_turns, start=1):
                parts.append(f"[Turn {i}]")
                parts.append(f"User:  {turn.user}")
                parts.append(f"Agent: {turn.agent}")

        return "\n".join(parts) if parts else ""

    def format_last_n_for_prompt(self, n: int = 2) -> str:
        """Render only the last N turns — lightweight injection for the classifier."""
        turns = self.recent_turns[-n:] if self.recent_turns else []
        if not turns:
            return ""
        lines = ["── RECENT TURNS (for reference resolution) ──"]
        for turn in turns:
            lines.append(f"User:  {turn.user}")
            lines.append(f"Agent: {turn.agent}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return not self.recent_turns and not self.summary

    # ── Private ────────────────────────────────────────────────────────────────

    def _compress(self) -> None:
        """Compress all buffered turns except the just-added one into the summary.

        This fires only when the buffer exceeds MAX_EXACT_TURNS.  All older turns
        are moved into the summary at once, and the buffer resets to contain only
        the current (most recently added) turn.  This means compression fires once
        every MAX_EXACT_TURNS turns — not after every single turn.

        Compression schedule example (MAX_EXACT_TURNS = 4):
          After T4: buffer = [T1, T2, T3, T4]           — no compress
          After T5: compress [T1,T2,T3,T4] → summary,   buffer = [T5]
          After T8: buffer = [T5,T6,T7,T8]              — no compress
          After T9: compress [T5,T6,T7,T8] → summary,   buffer = [T9]
        """
        overflow: list[Turn] = self.recent_turns[:-1]   # everything except current turn
        self.recent_turns = self.recent_turns[-1:]       # keep only the current turn

        overflow_text = "\n".join(
            f"User:  {t.user}\nAgent: {t.agent}" for t in overflow
        )

        if self.summary:
            prompt = _build_update_prompt(self.summary, overflow_text)
        else:
            prompt = _build_initial_prompt(overflow_text)

        self.summary = query_planner_llm(prompt, temperature=0.0).strip()
        print(f"[ConvMemory] Compressed {len(overflow)} turn(s) into running summary.")
        print("\n" + "═" * 60)
        print("  RUNNING SUMMARY (after compression)")
        print("═" * 60)
        print(self.summary)
        print("═" * 60 + "\n")


# ── Summary prompt builders ─────────────────────────────────────────────────────

def _build_initial_prompt(turns_text: str) -> str:
    """Prompt for building the first summary when the buffer overflows for the first time."""
    return f"""You are building a structured conversation memory for a reference resolution system.

A future language model will use this memory to correctly resolve:
- Pronouns: "them", "it", "those", "they", "that"
- Elliptical references: "the floods", "what you mentioned", "that policy", "those events"
- Follow-up questions about previously introduced subjects

The conversation is between a user and an AI agent playing William Ruto (President of Kenya).

Conversation turns being compressed:
{turns_text}

Produce a structured summary using EXACTLY this format.
Fill every section — do not skip or omit any section even if it is short.

## ENTITIES
List every named entity introduced: people, events, places, organisations, policies, crises, topics.
Format per entry:  - [name]: [brief description] — introduced in [first/second/... turn]
Example:           - Nairobi floods: severe flooding displacing thousands — introduced in first turn.

## ACTIVE TOPICS
List the main subjects actively discussed.
Format per entry:  - [topic]: [brief summary of positions taken, questions asked, claims made]

## PRONOUN ANCHORS
List every ambiguous pronoun or elliptical reference that appeared, with its resolved antecedent.
Format: - "[pronoun or phrase]" (user|agent, turn N): refers to [antecedent]
If none appeared: write  None

## UNRESOLVED THREADS
List questions or topics raised but not fully answered or resolved.
If none: write  None

## CHRONOLOGY
One line per 1-2 turns, in order, summarising what happened.
Example: Turns 1-2: User asked about Nairobi floods. Agent acknowledged severity and described relief operations.

Summary:"""


def _build_update_prompt(existing_summary: str, new_turns_text: str) -> str:
    """Prompt for incrementally updating an existing summary with new overflow turns.

    Incremental design is intentional: instead of regenerating the whole summary
    from scratch (which accumulates drift), we merge only the new overflow turns
    into the existing structure.  The existing summary is the stable anchor.
    """
    return f"""You are maintaining a structured conversation memory for a reference resolution system.

You have an EXISTING SUMMARY of earlier conversation turns.
You also have NEW TURNS that have just been compressed out of the recent exact buffer.

Your task: UPDATE the existing summary to incorporate the new turns.

STRICT RULES:
1. Do NOT remove or contradict anything already in the existing summary.
2. ADD new entities, active topics, pronoun anchors, and unresolved threads introduced in the new turns.
3. If a previously unresolved thread is now resolved by the new turns, mark it resolved.
4. If an ambiguous pronoun now has a clearer antecedent confirmed by the new turns, refine its entry.
5. EXTEND the Chronology with new entries for the new turns in order.
6. Keep entries concise — do not pad, repeat, or balloon the summary.

ENTITY PRESERVATION RULES (critical — violations cause reference resolution failures):
7. Every named entity already listed in ## ENTITIES MUST appear in the updated ## ENTITIES section
   with its original name unchanged. Never rename, merge, abbreviate, or abstract existing entities.
   Example: if "Gikomba markets" is listed, it must remain "Gikomba markets" — not "local markets"
   or "informal markets" or folded into a broader category.
8. Only ADD new entities from the new turns. Never remove or replace existing ones.

CHRONOLOGY PRESERVATION RULES:
9. The ## CHRONOLOGY section must only GROW — never shrink. Copy every existing chronology line
   exactly as written, then append new lines for the new turns at the bottom.
   Never rewrite, consolidate, or summarise existing chronology entries.

EXISTING SUMMARY:
{existing_summary}

NEW TURNS TO INCORPORATE:
{new_turns_text}

Write the COMPLETE updated summary (all sections, not just the changed parts) using EXACTLY this structure:

## ENTITIES
## ACTIVE TOPICS
## PRONOUN ANCHORS
## UNRESOLVED THREADS
## CHRONOLOGY

Updated summary:"""
