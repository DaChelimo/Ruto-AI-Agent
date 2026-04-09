"""FastAPI server — HTTP interface for the Ruto agent.

Start locally:
    uvicorn src.api:app --reload --port 8000

Test interactively:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .conversation_memory import ConvMemory
from .memory_store import load_memory_store
from .agent import process_turn


# ── Global state ──────────────────────────────────────────────────────────────
# memory_store is loaded once at startup and shared (read-only) across all requests.
# sessions maps session_id → per-session mutable state.
memory_store = None
sessions: dict[str, dict] = {}


# ── Lifespan: load memory store before the first request arrives ──────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_store
    print("Loading memory store...")
    memory_store = load_memory_store()
    print("Memory store loaded. Server ready.")
    yield
    print("Server shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ruto Agent API",
    description="Conversational AI agent embodying William Ruto, rendered through a critical lens.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow any origin so the React dev server / Android app can reach the backend.
# Restrict allow_origins to your production domain before going live.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models (request / response contracts) ────────────────────────────

class StartSessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    user_message: str


class ChatResponse(BaseModel):
    response: str
    ended: bool
    session_id: str
    pipeline_steps: list[str] = []
    message_type: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/session/start", response_model=StartSessionResponse)
def start_session():
    """Create a new conversation session.

    Call this once before sending any messages.  The returned session_id must
    be included in every subsequent /chat request.

    Returns:
        { "session_id": "uuid4", "message": "Session started." }
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "last_response": "",
        # Each session gets its own ConvMemory instance.
        # It is mutated in place by process_turn() after every completed turn.
        "conv_memory": ConvMemory(),
    }
    return StartSessionResponse(
        session_id=session_id,
        message="Session started. You may now send messages.",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send one message and get the agent's reply.

    The server runs the full pipeline for each turn:
        classify → (retrieve) → (sufficiency check) → (web search) → plan → stylize

    Conversational memory (recent exact turns + rolling structured summary) is
    maintained per session and injected into the classifier and planner prompts
    to enable correct pronoun and reference resolution across a long interview.

    If the user's message signals a conversation end, a sign-off is returned
    and the session is cleaned up automatically.

    Request body:
        { "session_id": "uuid4", "user_message": "What are you doing to mitigate them?" }

    Returns:
        { "response": "...", "ended": false, "session_id": "uuid4",
          "pipeline_steps": [...], "message_type": "FACTUAL" }
    """
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{request.session_id}' not found. "
                "Call POST /session/start first."
            ),
        )

    session = sessions[request.session_id]
    last_response: str = session["last_response"]
    conv_memory: ConvMemory = session["conv_memory"]

    result = process_turn(
        memory=memory_store,
        user_message=request.user_message,
        last_agent_response=last_response,
        conv_memory=conv_memory,          # mutated in place by process_turn()
    )

    # Persist the agent's reply so the next turn has context for should_end()
    session["last_response"] = result["response"]

    # Free memory once the conversation is over
    if result["ended"]:
        del sessions[request.session_id]

    return ChatResponse(
        response=result["response"],
        ended=result["ended"],
        session_id=request.session_id,
        pipeline_steps=result.get("pipeline_steps", []),
        message_type=result.get("message_type", ""),
    )


@app.get("/health")
def health():
    """Health check endpoint — used by Render to confirm the server is alive.

    Returns 200 OK as long as the server is running and the memory store is loaded.
    """
    return {
        "status": "ok",
        "memory_loaded": memory_store is not None,
        "sessions_active": len(sessions),
    }
