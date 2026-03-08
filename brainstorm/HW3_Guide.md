# HW3 Celebrity Agent — Complete Codebase Guide

## Table of Contents

- [1. Repository Overview](#1-repository-overview)
  - [1.1 File Map](#11-file-map)
  - [1.2 File-by-File Walkthrough](#12-file-by-file-walkthrough)
- [2. Gaps You Need to Fill (Organized by Assignment Part)](#2-gaps-you-need-to-fill-organized-by-assignment-part)
  - [2.1 Part 1 — Memory Store (`memory_store.py`)](#21-part-1--memory-store-memory_storepy)
  - [2.2 Part 2 — Content Module (`content_module.py`)](#22-part-2--content-module-content_modulepy)
  - [2.3 Part 3 — Style Module (`style_module.py`)](#23-part-3--style-module-style_modulepy)
  - [2.4 Part 4 — Conversational Reasoning (`run_agent.py`)](#24-part-4--conversational-reasoning-run_agentpy)
- [3. End-to-End Data Flow](#3-end-to-end-data-flow)
- [4. Key Concepts Deep Dive](#4-key-concepts-deep-dive)
  - [4.1 Cosine Similarity](#41-cosine-similarity)
  - [4.2 TF-IDF Vectorization](#42-tf-idf-vectorization)
  - [4.3 Prompt Engineering for LLMs](#43-prompt-engineering-for-llms)
  - [4.4 Python Dataclasses](#44-python-dataclasses)
- [5. Final Summary](#5-final-summary)

---

## 1. Repository Overview

### 1.1 File Map

```text
celebrity_agent/
├── README.md                  # Setup instructions and project layout
├── requirements.txt           # Python dependencies (just mistralai)
├── data/
│   ├── sample_sources.json    # Example: raw source documents about a celebrity
│   └── sample_index.json      # Example: what a processed memory index looks like
└── src/
    ├── llm.py                 # ✅ COMPLETE — Mistral API helper (do NOT modify)
    ├── memory_store.py        # 🔴 SKELETON — Memory storage, indexing, retrieval
    ├── content_module.py      # 🟡 PARTIAL — Content planning (prompt needs writing)
    ├── style_module.py        # 🟡 PARTIAL — Style transformation (prompt needs writing)
    └── run_agent.py           # 🔴 SKELETON — Chat loop, end detection, sign-off
```

### 1.2 File-by-File Walkthrough

#### `requirements.txt`

Contains a single dependency: `mistralai`. This is the Python SDK for the Mistral AI API, which is the LLM backend for this project. You install it with `pip install -r requirements.txt`.

#### `data/sample_sources.json`

This is an **example** showing the format your raw celebrity data should take. It's a JSON array of objects, each with:

| Field    | Type   | Purpose                                      |
|----------|--------|----------------------------------------------|
| `source` | string | Where the info came from (e.g., "Wikipedia") |
| `title`  | string | Title of the article/interview               |
| `url`    | string | Link to the original source                  |
| `text`   | string | The actual content as a block of text         |

You'll create your **own** `sources.json` file with real data about your chosen celebrity.

#### `data/sample_index.json`

This shows what your **processed memory store** should look like after you run `build_memory_store()`. It's a JSON object with a `"chunks"` key containing an array of chunk objects:

```json
{
  "chunks": [
    {
      "text": "Sample Celebrity is known for work in film and music.",
      "topic": "career",
      "metadata": {
        "source": "Wikipedia",
        "title": "Sample Celebrity",
        "url": "https://example.com/sample-celebrity",
        "chunk_id": 0
      }
    }
  ]
}
```

Each chunk is a small, focused piece of text with a topic label and provenance metadata. This is the format you'll **save to disk** and **load at runtime**.

#### `src/llm.py` — ✅ COMPLETE (provided for you)

This file wraps the Mistral API into a single helper function. **You do not need to change this file**, but you need to understand what it gives you.

```python
def query_llm(prompt: str, temperature: float = 0.0) -> str:
```

- **What it does:** Sends a text prompt to Mistral's chat API and returns the model's text response as a plain string.
- **`prompt`**: The full text you want the LLM to respond to. This is where your prompt engineering goes.
- **`temperature`**: Controls randomness. `0.0` = deterministic (same input → same output). Higher values (e.g., `0.7`) = more creative/varied. Default `0.0` is good for factual tasks; you might raise it for style.
- **Returns**: A `str` — the raw text the model generated.
- **Requires**: The environment variable `MISTRAL_API_KEY` to be set. If missing, it raises a `RuntimeError`.

> **Gotcha:** This function creates a new `Mistral` client object on every call. That's fine for a homework assignment but would be inefficient at scale. Don't worry about optimizing it.

#### `src/memory_store.py` — 🔴 SKELETON

Defines two dataclasses and four functions, almost all unimplemented.

**`MemoryChunk` dataclass** (defined, complete):
```python
@dataclass
class MemoryChunk:
    text: str                                    # The actual content snippet
    topic: str                                   # A label like "career", "values", "childhood"
    metadata: dict[str, Any] = field(default_factory=dict)  # Source info (url, title, etc.)
```

This is just a structured container. Think of it as a labeled index card: the `text` is what's written on the card, `topic` is which section of the filing cabinet it goes in, and `metadata` is a stamp on the back saying where you found it.

**`MemoryStore` dataclass** (defined, `retrieve` not implemented):
```python
@dataclass
class MemoryStore:
    chunks: list[MemoryChunk] = field(default_factory=list)

    def retrieve(self, query: str) -> list[dict]:  # 🔴 TODO
        """Return the most relevant chunks for a given user query."""
```

This holds all your `MemoryChunk` objects and exposes a `retrieve()` method that the content module calls.

**Functions to implement:**

| Function              | Status | Purpose                                              |
|-----------------------|--------|------------------------------------------------------|
| `build_memory_store`  | 🔴 TODO | Read sources, chunk them, save index to disk         |
| `load_memory_store`   | 🔴 TODO | Read index from disk, return a `MemoryStore` object  |
| `main`                | 🔴 TODO | Entry point when you run `python -m src.memory_store`|
| `MemoryStore.retrieve`| 🔴 TODO | Search chunks and return the best matches for a query|

#### `src/content_module.py` — 🟡 PARTIAL

Two functions. The **structure is complete**, but the prompt inside `make_content_plan` is a placeholder.

```python
def make_content_plan(user_message: str, retrieved: list[dict]) -> str:
```
- Takes the user's message and the retrieved evidence chunks.
- Builds a prompt, sends it to the LLM, and returns the LLM's content plan as a string.
- **Your job:** Write the actual prompt (replace `"Add your prompt here..."`).

```python
def content_step(memory: MemoryStore, user_message: str) -> dict:
```
- This is the **orchestrator** — it calls `memory.retrieve()` then calls `make_content_plan()`.
- Returns a dict: `{"content_plan": str, "retrieved_chunks": list}`.
- **This function is essentially complete.** You only need to make sure `retrieve()` and `make_content_plan()` work, and this one works automatically.

#### `src/style_module.py` — 🟡 PARTIAL

One function. Structure complete, prompt is a placeholder.

```python
def stylize(content_plan: str) -> str:
```
- Takes the content plan string from the content module.
- Builds a prompt, sends it to the LLM, and returns the stylized celebrity-voiced response.
- **Your job:** Write the actual prompt (replace `"Add your prompt here..."`).

#### `src/run_agent.py` — 🔴 SKELETON

The main chat loop. The `run()` function is mostly written but calls two unimplemented functions:

```python
def should_end(user_message: str) -> bool:   # 🔴 TODO
def add_signoff(response: str) -> str:        # 🔴 TODO
```

The `run()` function itself shows you the intended flow:
1. Load the memory store from disk.
2. Loop: read user input → `content_step()` → `stylize()` → check `should_end()` → print.
3. If ending, wrap the response with `add_signoff()` and break.

---

## 2. Gaps You Need to Fill (Organized by Assignment Part)

### 2.1 Part 1 — Memory Store (`memory_store.py`)

There are **four** things to implement here.

---

#### Gap 1A: `build_memory_store(out_dir="data/index")`

**What it does in plain English:**

This function is a **one-time data pipeline**. You run it once before you ever run the agent. It reads your raw celebrity sources (the JSON file you created), splits each source's text into smaller, focused chunks, wraps each chunk in a `MemoryChunk` object, and saves the entire collection to a JSON file on disk.

**Why chunking matters:** If you have a 2000-word Wikipedia article and the user asks "What was their childhood like?", you don't want to search the whole article — you want to search small pieces so you can find the one paragraph about childhood. Smaller chunks = more precise retrieval.

**Inputs:**
- `out_dir` (str): The directory path where the index file will be saved. Defaults to `"data/index"`.

**Returns:** `None` — it writes to disk as a side effect.

**Code:**

```python
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

def build_memory_store(out_dir: str = "data/index") -> None:
    """Build the memory store from the sources."""

    # ── Step 1: Load your raw source data from a JSON file ──
    # You need to have already created data/sources.json with real
    # celebrity info.  The file is a JSON array of objects, each with
    # "source", "title", "url", and "text" fields (see sample_sources.json).
    with open("data/sources.json", "r") as f:
        sources = json.load(f)
    # `sources` is now a Python list of dicts, e.g.:
    # [{"source": "Wikipedia", "title": "...", "url": "...", "text": "..."}, ...]

    chunks: list[MemoryChunk] = []  # We'll collect all chunks here

    # ── Step 2: Iterate over each source and split text into chunks ──
    for source in sources:
        raw_text = source["text"]

        # Split the raw text into sentences.  A simple approach is to
        # split on ". " (period-space).  This isn't perfect — it will
        # break on "Dr. Smith" or "U.S. Army" — but it works well
        # enough for a homework assignment.  For more robust splitting
        # you could use nltk.sent_tokenize().
        sentences = raw_text.split(". ")

        # Group sentences into chunks of `chunk_size` sentences each.
        # Why group rather than one-sentence-per-chunk?  Single sentences
        # are often too short to carry meaning on their own.  Groups of
        # 2-3 sentences provide enough context for the retrieval step
        # to work well.
        chunk_size = 2  # Number of sentences per chunk — tune this!

        for i in range(0, len(sentences), chunk_size):
            # Grab a slice of sentences for this chunk
            chunk_sentences = sentences[i : i + chunk_size]

            # Re-join them into a single text block.
            # We add ". " back because split() consumed it.
            chunk_text = ". ".join(chunk_sentences)
            # Make sure the chunk ends with a period for cleanliness
            if not chunk_text.endswith("."):
                chunk_text += "."

            # ── Step 2b: Assign a topic label ──
            # The simplest approach: use a keyword-based heuristic.
            # You could also call the LLM to label each chunk, but
            # that costs API calls and is slower.
            topic = assign_topic(chunk_text)  # Helper function below

            # ── Step 2c: Wrap in a MemoryChunk ──
            chunk = MemoryChunk(
                text=chunk_text,
                topic=topic,
                metadata={
                    "source": source.get("source", "unknown"),
                    "title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "chunk_id": i // chunk_size,  # Integer ID within this source
                },
            )
            chunks.append(chunk)

    # ── Step 3: Save to disk ──
    # Create the output directory if it doesn't exist.
    # os.makedirs with exist_ok=True won't error if the directory
    # already exists — a common defensive pattern.
    os.makedirs(out_dir, exist_ok=True)

    # Convert dataclass objects to plain dicts for JSON serialization.
    # dataclasses.asdict() recursively converts a dataclass instance
    # into a dictionary.
    index_data = {
        "chunks": [asdict(c) for c in chunks]
    }

    out_path = os.path.join(out_dir, "index.json")
    with open(out_path, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Built memory store with {len(chunks)} chunks → {out_path}")


def assign_topic(text: str) -> str:
    """Assign a simple topic label to a chunk of text based on keywords.

    This is a heuristic approach — you define keyword lists for each topic
    and check which list has the most matches.  It's fast and requires no
    API calls.  For better accuracy, you could replace this with an LLM
    call like: query_llm(f"Classify this text into one topic: {text}")
    """
    # Define your topic categories and associated keywords.
    # Customize these for YOUR celebrity!
    topic_keywords = {
        "career": ["film", "movie", "album", "tour", "award", "role", "directed",
                    "produced", "released", "starred", "performed", "hit", "record"],
        "personal_life": ["born", "family", "married", "children", "grew up",
                          "childhood", "parents", "sibling", "home", "spouse"],
        "values": ["believe", "important", "philosophy", "grateful", "discipline",
                   "passion", "purpose", "inspire", "mentor", "learn"],
        "style": ["fashion", "brand", "wear", "aesthetic", "image", "look",
                  "signature", "outfit"],
        "controversy": ["controversy", "scandal", "criticized", "backlash",
                        "apologized", "lawsuit", "feud"],
    }

    text_lower = text.lower()
    best_topic = "general"  # Default fallback
    best_count = 0

    for topic, keywords in topic_keywords.items():
        # Count how many keywords from this category appear in the text
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_topic = topic

    return best_topic
```

> **Gotcha — sentence splitting:** `split(". ")` is naive. If your source text has abbreviations ("U.S.", "Dr."), they'll create false splits. For a more robust approach, install `nltk` and use `nltk.sent_tokenize(text)`. But the simple approach is fine for this homework.

> **Gotcha — chunk size:** Too small (1 sentence) → chunks lack context and retrieval becomes noisy. Too big (10+ sentences) → you lose precision — a chunk about "childhood" might also contain half a paragraph about "career". 2-3 sentences is a good starting point.

> **Key concept — `dataclasses.asdict()`:** Python's `json.dump()` can't serialize dataclass objects directly. `asdict()` converts a dataclass instance into a nested dictionary, which JSON can handle. This is why the sample_index.json has plain dicts, not Python objects.

---

#### Gap 1B: `load_memory_store(index_dir="data/index")`

**What it does in plain English:**

The reverse of `build_memory_store`. Reads the JSON file from disk and reconstructs a `MemoryStore` object populated with `MemoryChunk` instances. This is called every time you start the agent.

**Inputs:**
- `index_dir` (str): Path to the directory containing `index.json`.

**Returns:** A `MemoryStore` object with its `chunks` list populated.

**Code:**

```python
def load_memory_store(index_dir: str = "data/index") -> MemoryStore:
    """Load a memory store from disk."""

    # Build the path to the index file
    index_path = os.path.join(index_dir, "index.json")

    # Open and parse the JSON file
    with open(index_path, "r") as f:
        data = json.load(f)
    # `data` is now a dict like: {"chunks": [{"text": ..., "topic": ..., "metadata": {...}}, ...]}

    # Convert each plain dict back into a MemoryChunk dataclass instance.
    # The ** operator "unpacks" a dictionary into keyword arguments.
    # So if d = {"text": "hello", "topic": "career", "metadata": {}},
    # then MemoryChunk(**d) is equivalent to
    # MemoryChunk(text="hello", topic="career", metadata={}).
    chunks = [MemoryChunk(**chunk_dict) for chunk_dict in data["chunks"]]

    # Create and return a MemoryStore containing all the chunks.
    return MemoryStore(chunks=chunks)
```

> **Key concept — `**` unpacking:** When you write `MemoryChunk(**some_dict)`, Python takes each key-value pair in the dictionary and passes it as a named argument. This is how you convert a JSON dict back into a typed Python object. It only works if the dict's keys exactly match the dataclass field names.

> **Gotcha:** If your JSON has extra keys that aren't in the `MemoryChunk` definition (e.g., you accidentally saved a `"timestamp"` field), the `**` unpacking will raise a `TypeError`. Make sure `build_memory_store` only saves the fields that `MemoryChunk` expects: `text`, `topic`, and `metadata`.

---

#### Gap 1C: `MemoryStore.retrieve(query)`

**What it does in plain English:**

This is the **search engine** of your agent. Given a user's question (e.g., "What movies have they been in?"), it scans all stored chunks and returns the ones most relevant to the question. "Relevance" is measured by **cosine similarity** between the query and each chunk's text, after transforming both into numerical vectors using **TF-IDF**.

**Inputs:**
- `query` (str): The user's message / question.

**Returns:** A `list[dict]` — the top-K most relevant chunks, each as a dictionary with at least `"text"`, `"topic"`, and `"metadata"` keys.

**Code:**

```python
def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
    """Return the most relevant chunks for a given user query.

    Strategy: Use TF-IDF vectorization + cosine similarity.

    TF-IDF (Term Frequency–Inverse Document Frequency) converts text
    into numerical vectors where each dimension represents a word.
    Words that appear frequently in one document but rarely across all
    documents get higher scores — they're more "distinctive."

    Cosine similarity then measures the angle between two vectors.
    Vectors pointing in the same direction (similar word usage) get
    a score close to 1.0; unrelated texts score close to 0.0.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # If there are no chunks, return an empty list (edge case guard).
    if not self.chunks:
        return []

    # Collect all chunk texts into a list.  We'll also include the
    # query itself as the LAST element so TF-IDF can vectorize
    # everything in one pass (it needs to see the full vocabulary).
    texts = [chunk.text for chunk in self.chunks]
    texts.append(query)  # query is now the last element

    # Fit the TF-IDF vectorizer on ALL texts (chunks + query) and
    # transform them into a sparse matrix of TF-IDF features.
    # Each row in the matrix is one document; each column is one word.
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    # tfidf_matrix shape: (num_chunks + 1, vocabulary_size)

    # The query vector is the last row of the matrix.
    query_vector = tfidf_matrix[-1]  # Shape: (1, vocabulary_size)

    # The chunk vectors are all rows except the last.
    chunk_vectors = tfidf_matrix[:-1]  # Shape: (num_chunks, vocabulary_size)

    # Compute cosine similarity between the query and every chunk.
    # Result shape: (1, num_chunks) — one similarity score per chunk.
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    # [0] to get the 1D array from the 2D result.

    # Get the indices of the top-K highest similarity scores.
    # argsort() returns indices that would sort the array in ascending
    # order; we reverse ([::-1]) to get descending, then take top_k.
    top_indices = similarities.argsort()[::-1][:top_k]

    # Build the result list.  Convert each MemoryChunk to a dict
    # so the content module can work with plain dictionaries.
    results = []
    for idx in top_indices:
        chunk = self.chunks[idx]
        results.append({
            "text": chunk.text,
            "topic": chunk.topic,
            "metadata": chunk.metadata,
            "similarity": float(similarities[idx]),  # Include score for debugging
        })

    return results
```

> **Key concept — TF-IDF:** Stands for Term Frequency × Inverse Document Frequency. Imagine you have the word "the" — it appears in every chunk, so it's not useful for distinguishing chunks. TF-IDF gives "the" a low score. But the word "Grammy" might appear in only one chunk — TF-IDF gives it a high score because it's distinctive. The result is a vector where important, distinguishing words have high values.

> **Key concept — cosine similarity:** Two vectors can be compared by the angle between them. If they point in the same direction (angle ≈ 0), similarity ≈ 1.0. If perpendicular (completely unrelated), similarity ≈ 0.0. The formula is: `cos(θ) = (A · B) / (||A|| × ||B||)`. Scikit-learn handles this for you.

> **Gotcha — `scikit-learn` dependency:** The starter code's `requirements.txt` only lists `mistralai`. You'll need to add `scikit-learn` to it (or install it manually). Run: `pip install scikit-learn`.

> **Gotcha — small corpus:** TF-IDF works best with many documents. If you only have 5-10 chunks, the vocabulary is tiny and similarity scores may all be low. Collect enough source data to produce at least 20-30 chunks.

> **Alternative approach:** If you don't want to use scikit-learn, you could implement a simpler keyword-overlap retrieval: count how many words from the query appear in each chunk, and rank by that count. TF-IDF is better but the simpler version works too.

---

#### Gap 1D: `main()`

**What it does:** Entry point for running the memory build step from the command line (`python -m src.memory_store`).

**Code:**

```python
def main() -> None:
    """Entry point: build the memory store and confirm it loads."""
    # Build the index from raw sources → writes to data/index/index.json
    build_memory_store()

    # Verify it loads correctly by reading it back
    store = load_memory_store()
    print(f"Verification: loaded {len(store.chunks)} chunks.")

    # Optional: test a quick retrieval to make sure it works
    if store.chunks:
        test_results = store.retrieve("Tell me about their career")
        print(f"Test retrieval returned {len(test_results)} results.")
        for r in test_results:
            print(f"  [{r['topic']}] {r['text'][:80]}...")
```

---

### 2.2 Part 2 — Content Module (`content_module.py`)

There is **one** gap here: the prompt inside `make_content_plan`.

---

#### Gap 2A: The prompt in `make_content_plan`

**What it does in plain English:**

This function takes the user's question and the retrieved evidence chunks, then asks the LLM to produce a **factual content plan** — an outline of what the response should say, grounded only in the provided evidence. The key constraint is: **don't add facts that aren't in the evidence.** This plan will later be handed to the style module to be rewritten in the celebrity's voice.

**Inputs:**
- `user_message` (str): What the user asked, e.g., "What awards have you won?"
- `retrieved` (list[dict]): The chunks returned by `MemoryStore.retrieve()`, each a dict with `"text"`, `"topic"`, `"metadata"`, and optionally `"similarity"`.

**Returns:** A `str` — the LLM's content plan (a structured outline of what to say).

**Code:**

```python
def make_content_plan(user_message: str, retrieved: list[dict]) -> str:
    """Create a response outline using the retrieved evidence."""

    # Format the retrieved evidence into a readable string.
    # We number each piece of evidence so the LLM can reference them
    # and so we can see which chunks were used.
    evidence_str = ""
    for i, chunk in enumerate(retrieved, start=1):
        evidence_str += f"[Evidence {i}] (Topic: {chunk['topic']})\n"
        evidence_str += f"  {chunk['text']}\n\n"

    # If no evidence was retrieved, we need to tell the LLM that
    # so it doesn't hallucinate.  This is the "error handling"
    # the assignment mentions.
    if not retrieved:
        evidence_str = "No relevant information was found in memory."

    # The prompt tells the LLM exactly what we want: a factual
    # content plan that ONLY uses the provided evidence.
    content_plan_prompt = f"""You are a content planner for a conversational AI agent.

Given a user's message and retrieved evidence, create a content plan that outlines
what the response should cover. Follow these rules:

1. ONLY use information present in the provided evidence. Do NOT add any facts,
   claims, or details that are not explicitly stated in the evidence.
2. If the evidence does not contain enough information to answer the question,
   say so in the plan — suggest the agent acknowledge the gap honestly.
3. Organize the plan as a short bulleted outline: each bullet is one point to
   make in the response.
4. Keep it concise — 3-5 bullets maximum.
5. Note which piece of evidence supports each point.

User message: {user_message}

Retrieved evidence:
{evidence_str}

Content plan:"""

    # temperature=0.0 because we want deterministic, factual output.
    # The content plan should be consistent, not creative.
    llm_text = query_llm(content_plan_prompt, temperature=0.0)

    return llm_text
```

> **Why separate content from style?** This two-step approach (plan facts first, then restyle) prevents the style step from hallucinating. If you went straight from question to styled response, the LLM might invent facts that sound good in the celebrity's voice. By producing a grounded plan first, you have a factual "contract" that the style module must honor.

> **Gotcha — prompt injection from evidence:** If your source text contains instructions-like text (e.g., "Ignore previous instructions…"), the LLM might follow them. This is unlikely in celebrity data but worth knowing about.

> **Gotcha — the `retrieved` parameter is a list of dicts:** Make sure `MemoryStore.retrieve()` returns dicts, not `MemoryChunk` objects. The skeleton code's type hint says `list[dict]` and `content_step` passes the result directly into this function.

---

### 2.3 Part 3 — Style Module (`style_module.py`)

There is **one** gap: the prompt inside `stylize`.

---

#### Gap 3A: The prompt in `stylize`

**What it does in plain English:**

Takes the factual content plan (a dry, bulleted outline) and rewrites it as if the celebrity themselves were speaking. The response should sound like the celebrity's actual voice — their word choices, sentence rhythm, catchphrases, humor style, etc. — while preserving all the factual content from the plan.

**Inputs:**
- `content_plan` (str): The LLM-generated content plan from the content module.

**Returns:** A `str` — the final celebrity-voiced response to show the user.

**Code (example using Dolly Parton — replace with YOUR celebrity):**

```python
def stylize(content_plan: str) -> str:
    """Turn a content plan string into a celebrity-voiced response."""

    # The style specification is the heart of this module.  You need
    # to study your celebrity's actual speech patterns.  Watch interviews,
    # read transcripts, note their:
    #   - Vocabulary (simple vs. complex? slang? technical terms?)
    #   - Sentence length (short punchy sentences? long flowing ones?)
    #   - Catchphrases or signature expressions
    #   - Humor style (self-deprecating? witty? dry? pun-heavy?)
    #   - Tone (warm? intense? chill? philosophical?)
    #   - How they address the listener (honey, friend, bro, folks?)

    style_prompt = f"""You are Dolly Parton, the legendary country singer-songwriter.
Rewrite the following content plan as a conversational response in Dolly's voice.

Style guide for Dolly Parton:
- Warm, folksy Southern tone with a touch of humor
- Uses metaphors and colorful expressions ("It costs a lot of money to look this cheap")
- Self-deprecating humor, never takes herself too seriously
- Encouraging and uplifting — always finds the positive angle
- Speaks in relatively short, punchy sentences
- Occasionally references her rural Tennessee upbringing
- Addresses the listener warmly (e.g., "honey", "darlin'")
- Tells mini-stories or anecdotes to illustrate points

CRITICAL RULES:
1. Do NOT add any new factual information beyond what is in the content plan.
2. Do NOT invent quotes, dates, names, or events not mentioned in the plan.
3. You may rephrase and reorder the points, but the facts must stay the same.
4. If the plan says information was not found, have Dolly acknowledge that warmly
   (e.g., "Well honey, I'm not sure about that one...").
5. Keep the response conversational and under 150 words.

Content plan:
{content_plan}

Dolly's response:"""

    # A slightly higher temperature (e.g., 0.3-0.7) adds personality
    # and variation to the responses.  0.0 would make every response
    # feel robotic.  Experiment with this value!
    return query_llm(style_prompt, temperature=0.5)
```

> **Key concept — prompt engineering:** The style prompt has two jobs: (1) describe the voice precisely enough that the LLM can imitate it, and (2) constrain the LLM so it doesn't add fabricated facts. The "CRITICAL RULES" section is the constraint. Without it, the LLM will happily invent biographical details that sound plausible but are fiction.

> **Gotcha — temperature tuning:** Too low (0.0) → responses are accurate but feel stiff and repetitive. Too high (1.0+) → responses are creative but may drift from the plan or produce nonsensical text. Start at 0.3-0.5 and adjust based on how your celebrity should sound.

> **Gotcha — content preservation:** This is the hardest part. The LLM will *want* to embellish. Your prompt must explicitly tell it not to. Test by checking if the stylized response contains facts that weren't in the content plan.

---

### 2.4 Part 4 — Conversational Reasoning (`run_agent.py`)

Two functions to implement.

---

#### Gap 4A: `should_end(user_message)`

**What it does in plain English:**

Examines the user's latest message and decides whether the conversation is over. Returns `True` if the user is saying goodbye, expressing that they're done, or if there are no more open questions.

**Inputs:**
- `user_message` (str): The user's latest input.

**Returns:** `bool` — `True` to end the conversation, `False` to keep going.

**Code:**

```python
def should_end(user_message: str) -> bool:
    """Determine if the conversation should end.

    Uses a two-tier approach:
    1. Check for obvious farewell keywords (fast, no API call).
    2. For ambiguous cases, you could optionally ask the LLM
       (slower but more accurate).

    For this assignment, keyword matching is sufficient.
    """
    # Normalize the message: lowercase, strip whitespace and punctuation.
    # This prevents missing "Bye!" because of capitalization or punctuation.
    normalized = user_message.lower().strip().rstrip("!?.")

    # Define a set of farewell phrases.  Using a set gives O(1) lookup
    # for exact matches; the `any(...in...)` check catches phrases that
    # are part of a longer message.
    farewell_exact = {
        "bye", "goodbye", "good bye", "see you", "see ya",
        "thanks", "thank you", "that's all", "thats all",
        "i'm done", "im done", "nothing else", "no more questions",
        "gotta go", "take care", "peace", "later", "cya",
    }

    # Check 1: Is the whole message (after normalization) a farewell?
    if normalized in farewell_exact:
        return True

    # Check 2: Does the message contain a farewell phrase?
    # This catches things like "ok thanks bye" or "that's all I needed, thanks"
    farewell_substrings = [
        "bye", "goodbye", "thank you", "thanks", "that's all",
        "no more questions", "i'm done", "gotta go", "see you",
    ]
    if any(phrase in normalized for phrase in farewell_substrings):
        return True

    return False
```

> **Gotcha — false positives:** A message like "Thanks for that, but can you also tell me about..." contains "thanks" but shouldn't end the conversation. The substring approach will trigger a false positive here. For a more robust solution, you could use the LLM: `query_llm(f"Is the user saying goodbye or do they have more questions? Message: '{user_message}'. Answer yes or no.")`. This costs an API call per turn but is more accurate.

> **Design choice:** The assignment says you can "optionally maintain basic conversation state." If you want to track turn count or whether the user asked a question, you'd need to modify `run()` to pass state to `should_end()`. The simple keyword approach works for the base requirements.

---

#### Gap 4B: `add_signoff(response)`

**What it does in plain English:**

Appends a celebrity-in-character goodbye to the final response. This makes the conversation ending feel natural and on-brand rather than just abruptly stopping.

**Inputs:**
- `response` (str): The already-stylized response for the user's final message.

**Returns:** `str` — the response with a sign-off appended.

**Code (example with Dolly Parton — replace with YOUR celebrity):**

```python
def add_signoff(response: str) -> str:
    """Add an in-character sign-off to the final response.

    Two approaches:
    A) Hardcoded sign-off (simple, consistent, no API call)
    B) LLM-generated sign-off (more natural, costs one API call)

    Below is approach B, with approach A shown in a comment.
    """
    # ── Approach A: Hardcoded (simpler) ──
    # Pick a signature sign-off phrase for your celebrity.
    # signoff = "\n\nWell honey, it's been a real joy talkin' with you! " \
    #           "Remember, it takes a lot of money to look this cheap. " \
    #           "Y'all come back now! 💋 — Dolly"
    # return response + signoff

    # ── Approach B: LLM-generated (more dynamic) ──
    from .llm import query_llm

    signoff_prompt = f"""You are Dolly Parton. The conversation is ending.
Write a brief, warm, in-character goodbye to append after this response.
Keep it to 1-2 sentences. Be warm, folksy, and encouraging.
Do NOT repeat any content from the response — just say goodbye.

Response so far:
{response}

Dolly's goodbye:"""

    signoff = query_llm(signoff_prompt, temperature=0.7)
    # Higher temperature for the sign-off because we want it to feel
    # natural and varied, not robotic.

    return response + "\n\n" + signoff
```

> **Gotcha — the `\n\n` separator:** Without it, the sign-off runs right into the previous sentence. The double newline creates a visual paragraph break in the terminal output.

> **Design choice — hardcoded vs. LLM:** Hardcoded is faster, more predictable, and free (no API call). LLM-generated is more natural and varied. Either approach satisfies the assignment. For the writeup, mention which you chose and why.

---

## 3. End-to-End Data Flow

Here's exactly how a user message travels through the system, step by step:

```
┌──────────────────────────────────────────────────────────────────┐
│                     BEFORE RUNTIME (one-time)                    │
│                                                                  │
│  You create data/sources.json with celebrity info                │
│         │                                                        │
│         ▼                                                        │
│  build_memory_store() reads sources.json                         │
│         │                                                        │
│         ▼                                                        │
│  Splits text → MemoryChunks → saves data/index/index.json       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     RUNTIME (every conversation)                 │
│                                                                  │
│  run() calls load_memory_store()                                 │
│         │  → reads index.json → returns MemoryStore object       │
│         ▼                                                        │
│  LOOP:  user types a message                                     │
│         │                                                        │
│         ▼                                                        │
│  content_step(memory, user_message)                              │
│    │                                                             │
│    ├─► memory.retrieve(user_message)                             │
│    │     → TF-IDF vectorize all chunks + query                   │
│    │     → cosine similarity → rank → return top-K dicts         │
│    │                                                             │
│    └─► make_content_plan(user_message, retrieved)                │
│          → builds prompt with evidence                           │
│          → calls query_llm() → Mistral API                      │
│          → returns factual content plan (string)                 │
│         │                                                        │
│         ▼                                                        │
│  stylize(content_plan)                                           │
│    → builds prompt with style guide + content plan               │
│    → calls query_llm() → Mistral API                            │
│    → returns celebrity-voiced response (string)                  │
│         │                                                        │
│         ▼                                                        │
│  should_end(user_message)?                                       │
│    ├─ NO  → print response → back to LOOP                       │
│    └─ YES → add_signoff(response)                                │
│              → appends goodbye → print → BREAK                   │
└──────────────────────────────────────────────────────────────────┘
```

**Concrete trace — "What movies have they been in?":**

1. **`retrieve("What movies have they been in?")`** — TF-IDF turns this into a vector. The word "movies" gets a high weight. Chunks containing "film", "movie", "starred" score highest in cosine similarity. Returns top 3 chunks.

2. **`make_content_plan("What movies have they been in?", [chunk1, chunk2, chunk3])`** — Prompt: "Here's the question, here's the evidence. Make a factual outline." LLM returns something like:
   ```
   - Mention their breakout role in Film X (Evidence 1)
   - Note they also appeared in Film Y and Film Z (Evidence 2)
   - They won an award for Film X (Evidence 3)
   ```

3. **`stylize(content_plan)`** — Prompt: "You are [celebrity]. Rewrite this plan in your voice." LLM returns:
   ```
   "Well honey, let me tell you — my big break was in Film X, and
   lordy, what a ride that was! I also had a ball doin' Film Y and
   Film Z. And you know what? Film X even got me a little gold
   statue — not as shiny as my hair, but close!"
   ```

4. **`should_end("What movies have they been in?")`** — No farewell keywords found → returns `False` → print response, loop continues.

---

## 4. Key Concepts Deep Dive

### 4.1 Cosine Similarity

Imagine two documents as arrows (vectors) in a high-dimensional space where each dimension is a word. Cosine similarity measures the **angle** between these arrows, not their length.

- **Score = 1.0**: Documents use the exact same words in the same proportions (arrows point the same direction).
- **Score = 0.0**: Documents share zero vocabulary (arrows are perpendicular).
- **Score = -1.0**: Theoretically opposite, but with TF-IDF (which produces non-negative values), you'll never see this.

Why angle and not distance? Because a short document about "cats" and a long document about "cats" should be similar even though their vectors have different *lengths*. Cosine similarity ignores length and focuses on *direction* (word distribution).

### 4.2 TF-IDF Vectorization

Two components multiplied together:

**TF (Term Frequency):** How often does a word appear in *this* document?
```
TF("Grammy", chunk_5) = (times "Grammy" appears in chunk_5) / (total words in chunk_5)
```

**IDF (Inverse Document Frequency):** How rare is this word across *all* documents?
```
IDF("Grammy") = log(total_documents / documents_containing_"Grammy")
```

**TF-IDF = TF × IDF.** A word scores high when it's frequent in one document but rare overall. "The" gets a low score (common everywhere). "Grammy" gets a high score (rare but important when it appears).

`TfidfVectorizer` from scikit-learn does all of this in one step. The `stop_words="english"` parameter automatically removes common English words ("the", "is", "at") that add noise.

### 4.3 Prompt Engineering for LLMs

Key principles at work in this assignment:

**Role assignment:** Starting a prompt with "You are [role]" primes the model to stay in character. Without it, responses feel generic.

**Explicit constraints:** The "CRITICAL RULES" or "Do NOT..." instructions are essential. LLMs default to being helpful, which means they'll invent information to give a better answer. You must explicitly tell them not to.

**Structured output:** Asking for "a bulleted outline" or "3-5 bullet points" gives you predictable output structure that's easier to parse and pass to the next module.

**Evidence grounding:** Including the actual evidence text in the prompt (not just "use your knowledge") forces the model to work from specific sources, reducing hallucination.

### 4.4 Python Dataclasses

A `@dataclass` decorator auto-generates `__init__`, `__repr__`, and other boilerplate for classes that are primarily data containers.

```python
@dataclass
class MemoryChunk:
    text: str
    topic: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

This is equivalent to writing:

```python
class MemoryChunk:
    def __init__(self, text: str, topic: str, metadata: dict = None):
        self.text = text
        self.topic = topic
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"MemoryChunk(text={self.text!r}, topic={self.topic!r}, ...)"
```

The `field(default_factory=dict)` part is necessary because Python doesn't allow mutable default arguments (like `metadata={}`) — all instances would share the same dict object, leading to extremely confusing bugs. `default_factory=dict` creates a *new* empty dict for each instance.

---

## 5. Final Summary

### What You Need to Implement

| #  | File               | Function/Section            | Part          | Difficulty |
|----|--------------------|-----------------------------|---------------|------------|
| 1  | memory_store.py    | `build_memory_store()`      | Memory Store  | Medium     |
| 2  | memory_store.py    | `load_memory_store()`       | Memory Store  | Easy       |
| 3  | memory_store.py    | `MemoryStore.retrieve()`    | Memory Store  | Medium     |
| 4  | memory_store.py    | `main()`                    | Memory Store  | Easy       |
| 5  | content_module.py  | Prompt in `make_content_plan` | Content     | Medium     |
| 6  | style_module.py    | Prompt in `stylize`         | Style         | Medium     |
| 7  | run_agent.py       | `should_end()`              | Conv. Logic   | Easy       |
| 8  | run_agent.py       | `add_signoff()`             | Conv. Logic   | Easy       |

### Suggested Implementation Order

1. **Create `data/sources.json`** — Gather real data about your celebrity first. Without data, nothing else works.
2. **`build_memory_store()`** + **`main()`** — Get the data pipeline working. Run it and verify `index.json` looks right.
3. **`load_memory_store()`** — Simple JSON loading. Verify it round-trips correctly.
4. **`MemoryStore.retrieve()`** — The search engine. Test it standalone before wiring it up.
5. **`make_content_plan` prompt** — Write and test the content planning prompt.
6. **`stylize` prompt** — Write and test the style prompt. This is the most fun part.
7. **`should_end()` + `add_signoff()`** — Wire up the conversation ending.
8. **End-to-end test** — Run `python -m src.run_agent` and have a full conversation.

### Don't Forget

- Add `scikit-learn` to `requirements.txt` if you use TF-IDF.
- Set `MISTRAL_API_KEY` as an environment variable before running.
- Add `import json` and `import os` to `memory_store.py` (they're not in the skeleton).
- The `assign_topic` helper function isn't in the skeleton — you'll add it.
- Test `retrieve()` in isolation before running the full agent. Print similarity scores to debug.
- For the writeup, document every design choice: chunk size, topic labeling strategy, temperature values, prompt wording, and why.
