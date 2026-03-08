"""Agent runner: wire memory, content, style, and chat loop."""

from .content_module import content_step
from .memory_store import load_memory_store
from .style_module import stylize
from .llm import query_planner_llm, query_classifier_llm


def should_end(user_message: str, last_agent_response: str) -> bool:
    """Determine if the conversation should end."""

    prompt = f"""A conversation is happening between a user and an AI agent.

    Last AI response: {last_agent_response}
    Latest user message: {user_message}

    Decide whether the user is ending the conversation. Answer YES if the message is:
    - A farewell or goodbye of any kind ("bye", "take care", "goodbye", etc.)
    - A thank you that signals they are done ("thanks", "thank you for your time", etc.)
    - A closing question asking for parting words ("any parting sentiments?", "any final thoughts?", "last words?")
    - A satisfaction signal with nothing left to ask ("that's all", "I'm good", "got what I needed")

    Answer NO if the message is:
    - A genuine question seeking information (even if it contains "?")
    - A follow-up on something previously discussed
    - A statement that implies more questions are coming

    Strictly answer with a single word: YES or NO."""

    result = query_classifier_llm(prompt).strip().upper()

    if result not in {"YES", "NO"}:
        return False  # safe default — keep conversation going if classifier is uncertain

    return result == "YES"


def add_signoff(last_response: str) -> str:
    """Generate a standalone in-character sign-off message."""

    signoff_prompt = f"""You are William Ruto, rendered through the perspective of a critic.
The conversation is ending. Write a single, complete parting message — not an addition to anything else.

The sign-off must:
- Thank the person warmly for the conversation in one sentence
- Briefly reassure Kenyans that I am working relentlessly to transform Kenya into the next Singa-pool, oh sorry, I meant Singapore [ include the previous part as is (the Singa-pool and Singapore sarcasm is intentional)]
- Sound presidential, composed, and forward-looking
- Feel like a natural human goodbye, not a closing statement at a press conference

CRITICAL RULES:
1. Output exactly ONE version — no alternatives, no "or if you prefer", no options
2. Plain text only — no asterisks, no bold, no italics, no markdown of any kind
3. Do NOT mention your title or introduce yourself
4. Do NOT repeat anything from the previous response
5. Maximum 2 sentences total

Previous response for context (do not repeat):
{last_response}

Sign-off:"""

    return query_planner_llm(signoff_prompt, temperature=0.5)


def run() -> None:
    """Run a basic CLI chat loop."""
    memory_store = load_memory_store()
    print(f"Loaded memory store from data/store.json.")

    last_agent_response = ""

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue

        # Check for conversation end BEFORE running the content pipeline
        # so we don't waste API calls generating a response that won't be used
        if should_end(user_message, last_agent_response):
            signoff = add_signoff(last_agent_response)
            print(f"Agent: {signoff}")
            break

        result = content_step(memory=memory_store, user_message=user_message)
        content_plan = result["content_plan"]
        response = stylize(content_plan=content_plan)

        print(f"Agent: {response}")
        last_agent_response = response


if __name__ == "__main__":
    run()
