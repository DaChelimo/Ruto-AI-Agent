# CIS 1990 — William Ruto Agent: Full Project Guide & FastAPI Migration

---

## Part 1: What This Project Is

You built a **conversational AI agent** that embodies William Ruto — President of Kenya — rendered through a critical lens. The agent is the deliverable for CIS 1990 at UPenn, but the engineering underneath it is production-grade, not homework-grade.

The agent is not just an LLM with a persona prompt. It is a full **Retrieval-Augmented Generation (RAG)** pipeline: it has a knowledge base derived from real articles about Ruto, a semantic search engine to retrieve relevant facts before answering, an intent classifier to route different question types differently, and a two-stage content → style pipeline to prevent hallucination while maintaining a consistent character voice.

The ultimate goal beyond the assignment is to **host the agent online** via a FastAPI backend and build an Android frontend that connects to it over HTTP — so that anyone can open an app on their phone and have a conversation with the agent, exactly as if it were deployed in production.

---

## Part 2: What We Have Built So Far

### 2.1 The File Structure

```
src/
├── llm.py              # Mistral API client, model constants, query functions, embeddings
├── memory_chunk.py     # MemoryChunk dataclass + ChunkSchema Pydantic validation model
├── memory_store.py     # MemoryStore, retrieval logic, build pipeline, load from disk
├── content_module.py   # Intent classifier, sufficiency checker, plan makers, content_step()
├── style_module.py     # Ruto voice style guide + stylize()
├── run_agent.py        # CLI loop, should_end(), add_signoff(), run()
└── web_search.py       # Tavily web search fallback

data/
├── sources.json        # Raw article text + metadata (your knowledge base)
└── index.json          # Pre-built chunks with embeddings (your search index)
```

---

### 2.2 `llm.py` — The Mistral Client Layer

This file owns every interaction with the Mistral API. It uses the **Singleton pattern** to ensure the Mistral client object is only initialized once across the entire application lifetime, no matter how many times it is imported.

**Key concepts here:**
- `__client` is name-mangled (double underscore) — Python renames it to `_AppClient__client` internally, which means no other class or file can accidentally touch it. This is encapsulation enforced at the language level.
- The `@property` decorator on `client` makes it a computed attribute. Instead of calling `appClient.client()` like a function, you call `appClient.client` like a variable — but Python secretly runs the getter logic each time, checking if the client is None and loading it if needed. This is called **lazy initialization**: the expensive network-touching object doesn't get created until the first time it's actually needed.
- Different tasks use different models, intentionally. `open-mistral-nemo` handles classification because it is cheap and fast. `mistral-small-latest` handles planning and styling because those tasks need stronger reasoning. This is **multi-model architecture**: matching model capability to task complexity to control cost.

**Model constants:**
```python
PLANNER_MODEL    = "mistral-small-latest"   # content planning: needs reasoning
STYLE_MODEL      = "mistral-small-latest"   # style: needs quality output
CHUNKER_MODEL    = "mistral-small-latest"   # chunking: needs reliable JSON
CLASSIFIER_MODEL = "open-mistral-nemo"      # classification: cheap + fast
EMBED_MODEL      = "mistral-embed"          # embeddings: purpose-built model
```

**What the functions do:**
- `query_planner_llm()`, `query_style_llm()`, etc. are thin wrappers that call `query_text_llm()` with the correct model name pre-filled. This means callers never hardcode model names — they just call a semantically named function.
- `embed(text)` turns a single string into a list of floats (a vector). This vector represents the *meaning* of the text in a 1024-dimensional space.
- `embed_batch(texts)` sends a whole list of strings to the embeddings API in one call instead of one call per text. This is far cheaper and faster at build time.

---

### 2.3 `memory_chunk.py` — The Data Shape

Two classes live here and they serve completely different purposes.

**`MemoryChunk` (dataclass):** The runtime container for a single chunk of knowledge. A dataclass is Python's shorthand for a class whose only job is to hold data — it auto-generates `__init__`, `__repr__`, and `__eq__` for you. The `field(default_factory=dict)` on `metadata` means each chunk gets its own fresh empty dict instead of all sharing the same dict object (a classic Python mutation bug that `default_factory` prevents).

**`ChunkSchema` (Pydantic BaseModel):** The validation gatekeeper used *only during the build step*. When the LLM returns a chunk candidate as JSON, we pass it through `ChunkSchema` before trusting it. If the text is too short, or the topic label is not one of the six allowed values, Pydantic raises a `ValidationError` and we skip that chunk rather than storing garbage. The `Literal["career", "personal_life", ...]` type annotation is enforced at runtime — not just as a hint.

The key insight: `MemoryChunk` is used at runtime everywhere. `ChunkSchema` is only used inside `convert_source_to_chunks()` during the build. They are intentionally separate because their purposes are different.

---

### 2.4 `memory_store.py` — The Knowledge Engine

This is the most complex file. It does three things: build the index, load the index, and retrieve from the index.

**Build phase (`build_memory_store`):**
1. Read `data/sources.json` — a list of raw articles you manually curated.
2. For each article, call `convert_source_to_chunks()`: send the article text to the LLM with a prompt asking it to divide the article into 3-5 sentence chunks, each with a topic label. The LLM returns a JSON array.
3. Strip markdown fences if the LLM added them anyway (it sometimes does despite instructions).
4. Run each chunk through `ChunkSchema` Pydantic validation. Skip malformed ones.
5. After validating all chunks for a source, call `embed_batch()` on all their texts at once — one API call for the whole batch rather than one per chunk.
6. Store each embedding inside the chunk's `metadata` dict under the key `"embedding"`.
7. After all sources are processed, serialize the whole `MemoryStore` to `data/index.json` using `asdict()`.

This whole process runs **once** at build time. You never re-run it during conversations. The embeddings are pre-computed and saved to disk.

**Load phase (`load_memory_store`):**
Simply reads `data/index.json` and reconstructs `MemoryChunk` objects. The embeddings come back as plain lists of floats, which is what we need.

**Retrieve phase (`retrieve`):**
1. Embed the user's query with `embed()` to get a query vector.
2. Compute **cosine similarity** between the query vector and every chunk's stored vector. Cosine similarity measures the angle between two vectors — two vectors pointing in similar directions (similar meaning) produce a score near 1.0; orthogonal vectors (unrelated meaning) produce a score near 0.
3. Use `np.argsort()[::-1]` to sort indices from highest to lowest similarity, then take the top 7.
4. Return the top 7 chunks as dicts including their similarity score.

This replaced the original TF-IDF approach because TF-IDF matches words literally — it would fail to retrieve a "CNN interview" chunk when the user asked "What did you say on television?" because the words don't overlap. Semantic embeddings match meaning, so "television interview" and "CNN interview" produce similar vectors and the right chunk gets retrieved.

---

### 2.5 `content_module.py` — The Brain

This is where the agent decides what to say before it says it. The flow has four stages:

**Stage 1 — Classify:** `classify_message()` sends the user's message to the cheap classifier model and gets back `CONVERSATIONAL`, `FACTUAL`, or `HYBRID`. This prevents the agent from treating "Hello, I'd like to ask you some questions" as a policy question that deserves a full retrieval pipeline.

**Stage 2 — Retrieve (conditional):** If the message is FACTUAL or HYBRID, run `memory.retrieve()` to get the top 7 relevant chunks. If it's CONVERSATIONAL, skip retrieval entirely.

**Stage 3 — Sufficiency check + web search fallback:** After retrieval, `_augment_with_web_search()` runs `check_sufficiency()` — a single LLM call that sees both the question and the retrieved evidence and returns either `SUFFICIENT` or an optimized web search query. If insufficient, Tavily is called and the web results are appended to the retrieved chunks list. The plan makers downstream never know or care whether evidence came from memory or from the web — they just see a list of evidence dicts.

**Stage 4 — Plan:** One of three plan makers generates a structured content plan (3-5 bullets) grounded strictly in the evidence. No invention allowed. The plan says *what* to say — not *how* to say it.

The separation between content planning and style rendering is deliberate. If you merged them into one prompt, the LLM would hallucinate freely because it's generating character voice *and* facts simultaneously. Splitting them into two calls means the first LLM call is constrained to evidence only, and the second LLM call is only allowed to rephrase what the first one produced. This is how you prevent hallucination without losing character voice.

---

### 2.6 `style_module.py` — The Voice

`stylize()` takes the content plan and rewrites it in Ruto's voice through a critic's lens. The style prompt contains 14 critical rules and a detailed character guide describing the specific texture of his speech: soft-spoken but calculating, uses "we" for policy questions but "I" for personal ones, never starts with filler words, answers the direct question first before deflecting, keeps responses under 70 words.

The character is not a parody. It is a technically constrained stylized rendering — the content plan guarantees factual grounding, and the style layer guarantees the voice stays consistent and critical without fabricating anything.

---

### 2.7 `run_agent.py` — The CLI Loop

This is the entry point for running the agent locally in a terminal. It:
1. Loads the memory store from disk.
2. Enters a `while True` loop accepting user input.
3. Before running the content pipeline, checks `should_end()` — an LLM call that determines if the user is signaling goodbye.
4. If ending: generates a sign-off via `add_signoff()` and breaks out of the loop.
5. If continuing: runs `content_step()` → `stylize()` → prints the response.

`should_end()` is pure LLM — no keyword matching. This was deliberately chosen after keyword matching caused "any parting sentiments?" (which contains `?`) to fail the heuristic check and trigger another content pipeline run instead of a graceful exit.

---

### 2.8 `web_search.py` — The Web Fallback

A thin Tavily wrapper. `search(query)` hits the Tavily API and returns results formatted as chunk-like dicts with the same shape as memory store chunks (`text`, `topic`, `metadata`, `similarity`). Because the shape matches, the entire downstream pipeline — `_format_evidence()`, the plan makers, the style layer — treats web results exactly like memory results. No special casing. Clean interface.

---

## Part 3: What We Are Trying to Accomplish Next

Right now the agent runs **locally** — you type in a terminal, it prints back. To host it online and connect an Android app to it, you need to:

1. Wrap the agent in a **FastAPI web server** that accepts HTTP requests and returns JSON responses.
2. Deploy that server to a cloud host (e.g., Render, Railway, or a VPS).
3. Build an Android app that sends HTTP `POST` requests to the server and displays the responses.

The terminal `input()` / `print()` loop in `run_agent.py` gets completely replaced by HTTP endpoints. Instead of the agent living in your terminal, it lives on a server waiting for JSON payloads from any client — Android, iOS, web browser, Postman, anything.

---

## Part 4: How FastAPI Works — The Mental Model

### What is FastAPI?

FastAPI is a Python web framework that lets you define functions and say "when someone sends an HTTP request to this URL, run this function and return the result as JSON."

HTTP is just text sent over a network in a specific format:
- A **request** has: a method (GET, POST, DELETE), a path (e.g., `/chat`), optional headers, and an optional body (JSON).
- A **response** has: a status code (200 = OK, 422 = bad input, 500 = server error), headers, and a body (JSON).

FastAPI automatically:
- Validates incoming request JSON against a Pydantic model (rejects malformed requests with 422 before your code ever runs).
- Serializes your return value to JSON.
- Generates a `/docs` page (Swagger UI) where you can test every endpoint in a browser.

### Why Not Flask?

Flask is the other major Python web framework. The difference: FastAPI uses Python's `async/await` and type annotations natively, which means it validates inputs automatically and handles concurrent requests more efficiently. For an AI agent that makes multiple LLM API calls per request, these two properties matter.

### What is a Session?

In a terminal loop, the `while True` loop is what keeps the conversation alive. On a web server, there is no loop — each HTTP request is independent. You need to track conversation state (specifically: the last agent response, used for `should_end()`) across multiple requests from the same user.

The solution is a **session**: when a user starts a conversation, the server creates a unique ID (a UUID like `"a3f8-4d21-..."`), stores the conversation state in memory mapped to that ID, and gives the ID to the client. The Android app saves this ID and sends it with every subsequent request. The server looks up the state, processes the message, updates the state, and sends back the response.

---

## Part 5: Step-by-Step FastAPI Migration

### Step 1 — Install Dependencies

```bash
pip install fastapi uvicorn
```

FastAPI is the framework. Uvicorn is the ASGI server that actually runs the FastAPI app (like how Apache runs PHP). You always run `uvicorn`, not `python api.py` directly.

---

### Step 2 — Create the New File Structure

You will add two new files and leave all existing files untouched:

```
src/
├── api.py          # NEW — FastAPI app, endpoints, session management
├── agent.py        # NEW — stateless agent logic (extracted from run_agent.py)
├── llm.py          # unchanged
├── memory_chunk.py # unchanged
├── memory_store.py # unchanged
├── content_module.py # unchanged
├── style_module.py   # unchanged
├── run_agent.py    # unchanged — still works for local CLI testing
└── web_search.py   # unchanged
```

The key idea: `run_agent.py` stays fully functional for local testing. `api.py` is purely the HTTP layer — it imports the logic from other modules. Nothing in the core pipeline changes.

---

### Step 3 — Create `agent.py`

The `run()` function in `run_agent.py` mixes two concerns: the conversation logic and the CLI loop (`input()` / `print()`). For the API, you need the logic without the loop.

Extract the per-turn logic into a stateless function:

```python
# src/agent.py
"""Stateless agent turn logic — usable by both CLI and API."""

from .content_module import content_step
from .style_module import stylize
from .run_agent import should_end, add_signoff
from .memory_store import MemoryStore


def process_turn(
    memory: MemoryStore,
    user_message: str,
    last_agent_response: str,
) -> dict:
    """
    Process one conversation turn. Returns a dict with:
        {
            "response": str,      # the agent's reply
            "ended": bool,        # True if this turn triggered a sign-off
        }

    This function has no side effects and no I/O — it takes inputs
    and returns outputs. The caller (CLI or API) decides what to do
    with the response.
    """

    if should_end(user_message, last_agent_response):
        signoff = add_signoff(last_agent_response)
        return {"response": signoff, "ended": True}

    result = content_step(memory=memory, user_message=user_message)
    response = stylize(content_plan=result["content_plan"])
    return {"response": response, "ended": False}
```

**Why this separation matters:** `process_turn()` knows nothing about HTTP or terminals. It is a pure function: same inputs, same outputs, no side effects. This makes it testable in isolation and reusable by any caller.

---

### Step 4 — Create `api.py`

This is the FastAPI app. Walk through it section by section.

```python
# src/api.py
"""FastAPI web server — HTTP interface for the Ruto agent."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from .memory_store import load_memory_store
from .agent import process_turn
```

**What each import does:**
- `asynccontextmanager` — lets you write a function that runs startup and shutdown code. This is how you load the memory store once when the server boots, not on every request.
- `FastAPI` — the app itself.
- `HTTPException` — how you return error responses (e.g., 404 if session not found).
- `CORSMiddleware` — Cross-Origin Resource Sharing. Without this, Android apps on a different domain/IP than your server will have their requests blocked by HTTP security rules.
- `BaseModel` — Pydantic models for request/response validation.
- `uuid` — generates unique session IDs.

```python
# ── Global state ──────────────────────────────────────────────────────────────

memory_store = None       # loaded once at startup, shared across all requests
sessions: dict[str, dict] = {}  # { session_id: { "last_response": str } }
```

`memory_store` is `None` at module import time. It gets filled during startup. `sessions` is a plain Python dict mapping session IDs to their state. This is **in-memory session storage** — it is simple, fast, and sufficient for a project. The trade-off: if the server restarts, all sessions are lost and users need to start new conversations. (Production systems use Redis for persistent session storage.)

```python
# ── Lifespan: startup + shutdown ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs at server startup and shutdown."""
    global memory_store
    print("Loading memory store...")
    memory_store = load_memory_store()
    print(f"Memory store loaded. Server ready.")
    yield                      # server is running — handle requests
    print("Server shutting down.")


app = FastAPI(
    title="Ruto Agent API",
    description="Conversational AI agent embodying William Ruto.",
    version="1.0.0",
    lifespan=lifespan,
)
```

The `lifespan` pattern is FastAPI's recommended way to run code once at startup. Everything before `yield` runs when the server boots. Everything after `yield` runs when the server shuts down. This ensures `memory_store` is populated before the first request ever arrives.

```python
# ── CORS middleware ───────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # during dev, allow all origins
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`allow_origins=["*"]` means any client (your Android app, Postman, a browser) can call this server. In production you would restrict this to your app's domain. For development, `"*"` is fine.

```python
# ── Pydantic request/response models ─────────────────────────────────────────

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
```

These are the contracts between your server and any client. FastAPI reads the `ChatRequest` model and automatically validates that every incoming POST body has a `session_id` (string) and `user_message` (string). If either is missing or the wrong type, FastAPI returns a 422 error before your code ever runs. No manual `if "session_id" not in body:` checks needed.

```python
# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/session/start", response_model=StartSessionResponse)
def start_session():
    """
    Create a new conversation session.

    The client calls this once to get a session_id.
    All subsequent /chat requests must include this session_id.

    Returns:
        { "session_id": "a3f8-...", "message": "Session started." }
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"last_response": ""}
    return StartSessionResponse(
        session_id=session_id,
        message="Session started. You may now send messages."
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send one message and get the agent's response.

    The client sends:
        { "session_id": "a3f8-...", "user_message": "What is your education background?" }

    The server:
        1. Looks up the session state (specifically: last_response for should_end())
        2. Runs the full agent pipeline (classify → retrieve → check → plan → stylize)
        3. Updates the session state with the new response
        4. Returns the response and whether the conversation has ended

    Returns:
        { "response": "...", "ended": false, "session_id": "a3f8-..." }
    """
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Call /session/start first."
        )

    session = sessions[request.session_id]
    last_response = session["last_response"]

    result = process_turn(
        memory=memory_store,
        user_message=request.user_message,
        last_agent_response=last_response,
    )

    # Update session state so the next turn has access to this response
    session["last_response"] = result["response"]

    # If the conversation ended, clean up the session to free memory
    if result["ended"]:
        del sessions[request.session_id]

    return ChatResponse(
        response=result["response"],
        ended=result["ended"],
        session_id=request.session_id,
    )


@app.get("/health")
def health():
    """
    Simple health check endpoint.

    Used by hosting platforms (Render, Railway) to verify the server is alive.
    Returns 200 OK with a JSON body.
    """
    return {"status": "ok", "sessions_active": len(sessions)}
```

**Why three endpoints?**
- `/session/start` — creates the session. Analogous to `memory_store = load_memory_store(); last_agent_response = ""` in the CLI setup.
- `/chat` — one request = one conversation turn. Analogous to one iteration of the `while True` loop.
- `/health` — required by cloud platforms. They periodically ping this to confirm the server is alive. If it returns anything other than 200, the platform restarts the server.

---

### Step 5 — Run the Server Locally

From your project root (the directory that contains `src/` and `data/`):

```bash
uvicorn src.api:app --reload --port 8000
```

Breaking this down:
- `src.api` — the Python module path to your `api.py` file. Because your files are inside a `src/` package, you use dot notation.
- `app` — the variable name of the FastAPI instance inside `api.py`.
- `--reload` — automatically restarts the server whenever you save a Python file. Only use this during development.
- `--port 8000` — which port to listen on. `localhost:8000` is the default dev URL.

When the server boots you should see:
```
Loading memory store...
Loaded memory store with N chunks.
Memory store loaded. Server ready.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### Step 6 — Test with Swagger UI

Open `http://localhost:8000/docs` in a browser. FastAPI auto-generates an interactive API explorer. You can:
1. Click `POST /session/start` → Execute → copy the `session_id` from the response.
2. Click `POST /chat` → paste your `session_id` and a `user_message` → Execute → see the agent's response.

This lets you fully test the server without writing any Android code yet.

---

### Step 7 — Test with curl (Alternative)

If you prefer the terminal:

```bash
# Step 1: Start a session
curl -X POST http://localhost:8000/session/start

# Returns something like:
# {"session_id": "a3f8b21c-4d21-4e3a-9f12-abc123", "message": "Session started."}

# Step 2: Send a message (replace the session_id with what you got above)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "a3f8b21c-4d21-4e3a-9f12-abc123", "user_message": "Hello, who are you?"}'
```

---

### Step 8 — Set Environment Variables for the Server

Your server needs `MISTRAL_API_KEY` and `TAVILY_API_KEY`. In local development, your `.env` file handles this because `load_dotenv()` is called inside `llm.py` and `web_search.py`. This works when running `uvicorn` from the same directory as your `.env` file.

On a cloud server, `.env` files are not uploaded. Instead, you set **environment variables** through the hosting platform's dashboard. Every platform (Render, Railway, Fly.io) has a "Environment Variables" or "Secrets" section where you paste your keys. The server process sees them via `os.getenv()` the same way.

---

### Step 9 — Deploy to a Cloud Host (Render)

Render is the recommended starting point because it has a free tier and a simple Git-based deploy workflow.

**Prerequisites:**
- Push your project to a GitHub repository.
- Make sure `data/index.json` is committed (your pre-built index). This is important — Render will not re-run your build script unless you configure it to.

**On Render:**
1. Go to render.com → New → Web Service.
2. Connect your GitHub repo.
3. Set:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
4. Under "Environment Variables," add `MISTRAL_API_KEY` and `TAVILY_API_KEY`.
5. Click Deploy.

The `--host 0.0.0.0` flag is critical on a cloud server. `localhost` (127.0.0.1) only accepts connections from the same machine. `0.0.0.0` tells the server to accept connections from anywhere — which is required for external clients to reach it.

The `$PORT` variable is set by Render automatically. You should never hardcode port 8000 on a cloud host.

After deployment, Render gives you a public URL like `https://ruto-agent.onrender.com`. This is what your Android app will point to.

**Your `requirements.txt` should include:**
```
fastapi
uvicorn
mistralai
tavily-python
python-dotenv
pydantic
numpy
scikit-learn
```

---

### Step 10 — Android Integration (Overview)

Your Android app replaces the terminal entirely. Instead of `input()`, the user types in an Android `EditText`. Instead of `print()`, the response appears in a `TextView` or `RecyclerView`.

The Android lifecycle mirrors the server's session model exactly:

| CLI (local) | Android (online) |
|---|---|
| `memory_store = load_memory_store()` | `POST /session/start` → save session_id |
| `while True: user_message = input(...)` | User types and taps Send |
| `result = process_turn(...)` | `POST /chat` with session_id + message |
| `print(f"Agent: {response}")` | Display response in chat UI |
| `break` on ended | Show goodbye message, clear session_id |

The HTTP library you use on Android is **Retrofit** (recommended) or **OkHttp**. Both let you define API interfaces and make async HTTP calls without blocking the UI thread.

**Example Retrofit interface:**
```kotlin
interface AgentApi {
    @POST("session/start")
    suspend fun startSession(): StartSessionResponse

    @POST("chat")
    suspend fun chat(@Body request: ChatRequest): ChatResponse
}
```

**Data classes to match the API:**
```kotlin
data class StartSessionResponse(val session_id: String, val message: String)
data class ChatRequest(val session_id: String, val user_message: String)
data class ChatResponse(val response: String, val ended: Boolean, val session_id: String)
```

The `suspend` keyword means these functions run on a coroutine (Kotlin's version of async/await), so they don't block the main thread while waiting for the network response.

---

## Part 6: The Full Pipeline — End to End

Here is the complete data flow for one conversation turn once the API is deployed:

```
User types message on Android
         │
         ▼
Android sends POST /chat
{ session_id: "...", user_message: "What scandals have you been in?" }
         │
         ▼
FastAPI receives request
→ Validates JSON against ChatRequest model
→ Looks up session in sessions dict
→ Gets last_response from session state
         │
         ▼
process_turn() is called
         │
         ├─ should_end()? (LLM call to open-mistral-nemo)
         │       └─ NO → continue
         │
         ├─ classify_message() (LLM call to open-mistral-nemo)
         │       └─ returns FACTUAL
         │
         ├─ memory.retrieve() (embed query → cosine similarity over 160 chunks)
         │       └─ returns top 7 chunks
         │
         ├─ _augment_with_web_search()
         │       ├─ check_sufficiency() (LLM call to open-mistral-nemo)
         │       │       └─ returns "William Ruto corruption scandal 2025" (insufficient)
         │       └─ web_search() (Tavily API call)
         │               └─ returns 3 web result chunks
         │
         ├─ make_factual_plan() (LLM call to mistral-small-latest)
         │       └─ returns 4-bullet content plan grounded in evidence
         │
         └─ stylize() (LLM call to mistral-small-latest)
                 └─ returns Ruto-voiced response string
         │
         ▼
FastAPI updates session["last_response"]
FastAPI returns ChatResponse JSON
{ response: "...", ended: false, session_id: "..." }
         │
         ▼
Android receives response
→ Displays in chat UI
→ Stores session_id for next turn
```

Total LLM calls per turn (worst case, with web search): **5**
- 1× `should_end` (Nemo)
- 1× `classify_message` (Nemo)
- 1× `check_sufficiency` (Nemo)
- 1× `make_factual_plan` (mistral-small)
- 1× `stylize` (mistral-small)

Total LLM calls per turn (best case, conversational): **2**
- 1× `should_end` (Nemo)
- 1× `classify_message` → CONVERSATIONAL → `make_conversational_plan` (mistral-small)
- No retrieval, no sufficiency check, no web search

---

## Part 7: What Remains To Do

| Task | Status | Notes |
|---|---|---|
| `llm.py` | Done | Singleton, multi-model, embed, embed_batch |
| `memory_chunk.py` | Done | Dataclass + Pydantic validation |
| `memory_store.py` | Done | Build pipeline, semantic retrieval |
| `content_module.py` | Done | Classifier, sufficiency check, web fallback, plan makers |
| `style_module.py` | Done | 14-rule critical Ruto voice |
| `run_agent.py` | Done | CLI loop with LLM-based end detection |
| `web_search.py` | Done | Tavily fallback, chunk-shaped output |
| `agent.py` | **To do** | Extract `process_turn()` from run_agent |
| `api.py` | **To do** | FastAPI app with 3 endpoints |
| `requirements.txt` | **To do** | Needed for cloud deployment |
| Cloud deployment | **To do** | Render recommended |
| Android app | **To do** | Your Android skills, Retrofit/OkHttp |

---

*This document reflects the project state as of CIS 1990, Spring 2026.*
