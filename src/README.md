# The Ruto Agent

A production-grade conversational AI agent built on a custom Retrieval-Augmented Generation (RAG) pipeline, embodying the persona of William Ruto — President of Kenya — as rendered through the lens of a political critic. The agent retrieves grounded evidence from a curated knowledge base, separates factual planning from stylistic expression, and holds natural multi-turn conversations that end gracefully.

> Built as a homework assignment for CIS 1990 at the University of Pennsylvania.

---

## What It Does

A user can converse with the agent as if interviewing President Ruto. The agent:

- Retrieves relevant information from a pre-built semantic knowledge base
- Generates factually grounded, evidence-backed responses
- Expresses those responses in Ruto's distinctive voice — polished, evasive, and politically calculated
- Detects natural conversation endings and signs off in character

The agent handles greetings, policy questions, hybrid messages, and graceful goodbyes — each routed through a purpose-built pipeline.

---

## Technical Highlights

### Semantic RAG with Mistral Embeddings

Moved beyond keyword-based TF-IDF retrieval to dense vector embeddings using Mistral's `mistral-embed` model. Each knowledge chunk is embedded at build time and stored alongside its text. At runtime, the user query is embedded and compared to all chunks via cosine similarity using vector dot products — enabling context-aware retrieval that understands meaning, not just word overlap.

### Two-Stage LLM Pipeline

Separates factual grounding from stylistic expression into two distinct LLM calls:

1. **Content Module** — produces a structured, evidence-bound content plan from retrieved chunks. No stylization, no embellishment.
2. **Style Module** — rewrites the plan in Ruto's voice. Constrained by strict prompt rules to prevent hallucination.

This separation ensures the agent never fabricates facts while still producing a natural, character-consistent response.

### Intent Classification with LLM Routing

Every user message is classified into one of three categories — `CONVERSATIONAL`, `FACTUAL`, or `HYBRID` — using a dedicated classifier LLM call. Each category routes to a purpose-built planning prompt, preventing a policy speech from being triggered by "Hey." The classifier also handles edge cases like "I have a few questions for you" (a setup, not a question) and closing messages that contain question marks.

### Multi-Model Architecture for Cost Efficiency

Each task runs on the smallest model capable of doing it well:

| Task                  | Model                  |
| --------------------- | ---------------------- |
| Content planning      | `mistral-small-latest` |
| Style generation      | `mistral-small-latest` |
| Text chunking         | `mistral-small-latest` |
| Intent classification | `open-mistral-nemo`    |
| Sign-off detection    | `open-mistral-nemo`    |
| Embeddings            | `mistral-embed`        |

Classification and sign-off detection — the simplest tasks — use the cheapest model, reducing cost per conversation turn without sacrificing accuracy where it matters.

### Pydantic Validation with Structured LLM Output

LLM responses for the chunking step are validated against a `ChunkSchema` Pydantic model with `Literal` type constraints on topic labels. Field validators autocorrect common variations (e.g., `"personal life"` → `"personal_life"`) and reject structurally invalid chunks, preventing bad data from entering the knowledge base.

### Singleton API Client

The Mistral client is implemented as a lazy-loading Singleton using Python's `@property` and name-mangled class variables (`__client`). The client is instantiated only on the first API call and reused for all subsequent calls — eliminating repeated authentication overhead across the pipeline.

### Batch Embedding at Build Time

Rather than making one API call per chunk during the memory store build, all chunk texts are embedded in batches using Mistral's batch embedding endpoint — reducing build time and API call count significantly.

### Temperature-Tuned by Task

- `0.0` for deterministic tasks: chunking, content planning, classification, sign-off detection
- `0.5` for the sign-off message (natural variation without losing coherence)
- `0.7` for stylized responses (personality and character feel)

---

## Architecture

```
User Input
    │
    ▼
should_end() ──YES──► add_signoff() ──► print ──► exit
    │
    NO
    │
    ▼
classify_message()
    │
    ├── CONVERSATIONAL ──► make_conversational_plan()
    ├── FACTUAL ────────► retrieve() ──► make_factual_plan()
    └── HYBRID ─────────► retrieve() ──► make_hybrid_plan()
                                │
                                ▼
                           stylize()
                                │
                                ▼
                           Agent Response
```

---

## Installation

### Prerequisites

- Python 3.9+
- A [Mistral AI](https://console.mistral.ai) account with an active API key

### 1. Clone the repository

```bash
git clone <repo-url>
cd celebrity_agent
```

### 2. Install dependencies

```bash
pip install mistralai scikit-learn pydantic python-dotenv numpy
```

### 3. Set your Mistral API key

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_api_key_here
```

### 4. Add your sources

Populate `data/sources.json` with your celebrity's source material. Follow the format in `data/sample_sources.json` — each source needs `source`, `title`, `url`, and `text` fields.

### 5. Build the memory store

This runs once to chunk, embed, and index your sources:

```bash
python -m src.memory_store
```

### 6. Run the agent

```bash
python -m src.run_agent
```

---

## Sample Interaction

```
You:    Hello. Welcome to today's interview. Thank you for joining me today.

Agent:  It's a pleasure to be here. I'm William Ruto, President of Kenya,
        and I'm ready to engage. What would you like to discuss?

You:    Do you believe you played a part in the 2007 post-election violence?

Agent:  The ICC matter was a legal process, and as a law-abiding citizen, I
        respect the judicial system. The case was ultimately resolved, and
        Kenya has made tremendous strides in unity and reconciliation since
        then. My focus remains on delivering for all Kenyans.

You:    Thank you. Any parting words?

Agent:  Thank you for this meaningful exchange. Kenyans, I promise you,
        we're building a future where Kenya stands tall — even if some still
        call it Singa-pool.
```

---

## Project Structure

```
celebrity_agent/
├── data/
│   ├── sources.json          # Your raw celebrity source material
│   ├── index.json            # Auto-generated chunk index with embeddings
│   └── sample_sources.json   # Format reference
└── src/
    ├── llm.py                # Mistral client singleton + model-specific query functions
    ├── memory_chunk.py       # MemoryChunk dataclass + Pydantic ChunkSchema
    ├── memory_store.py       # Build, load, and semantic retrieval pipeline
    ├── content_module.py     # Classifier + three content planning functions
    ├── style_module.py       # Stylization prompt (celebrity voice)
    └── run_agent.py          # Chat loop, sign-off detection, conversation logic
```

---

## Requirements

```
mistralai
scikit-learn
pydantic
python-dotenv
numpy
```
