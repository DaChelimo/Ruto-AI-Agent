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

from .memory_store import load_memory_store
from .agent import process_turn


# ── Global state ──────────────────────────────────────────────────────────────
# memory_store is loaded once at startup and shared across all requests.
# sessions maps session_id → {"last_response": str} for conversation continuity.
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
# Allow any origin so the Android app (or Postman) can reach the server.
# Restrict allow_origins to your app's domain before going to production.
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/session/start", response_model=StartSessionResponse)
def start_session():
    """Create a new conversation session.

    Call this once before sending any messages. The returned session_id must be
    included in every subsequent /chat request.

    Returns:
        { "session_id": "uuid4", "message": "Session started." }
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"last_response": ""}
    return StartSessionResponse(
        session_id=session_id,
        message="Session started. You may now send messages.",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send one message and get the agent's reply.

    The server runs the full pipeline for each turn:
        classify → (retrieve) → (sufficiency check) → (web search) → plan → stylize

    If the user's message signals a conversation end, a sign-off is returned and
    the session is cleaned up automatically.

    Request body:
        { "session_id": "uuid4", "user_message": "What do you think about the IMF loan?" }

    Returns:
        { "response": "...", "ended": false, "session_id": "uuid4" }
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
    last_response = session["last_response"]

    result = process_turn(
        memory=memory_store,
        user_message=request.user_message,
        last_agent_response=last_response,
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
