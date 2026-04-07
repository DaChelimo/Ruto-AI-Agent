"""Web search fallback: Tavily-powered search when memory store is insufficient."""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def _get_client() -> TavilyClient:
    """Lazy-load the Tavily client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing TAVILY_API_KEY environment variable. "
            "Set it before using web search fallback."
        )
    return TavilyClient(api_key=api_key)


def search(query: str, max_results: int = 3) -> list[dict]:
    """Search the web via Tavily and return results as chunk-like dicts.

    Each result dict has the same shape as a retrieved memory chunk:
        {"text": ..., "topic": "web_search", "metadata": {...}}

    This means the content planner can treat web results identically
    to memory store results — no special handling needed downstream.
    """
    client = _get_client()

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",           # "basic" is fast + cheap; "advanced" for deeper digs
        include_answer=False,           # we don't need Tavily's built-in LLM answer
        include_raw_content=False,      # snippets are enough — no full page HTML
    )

    results = response.get("results", [])
    chunks = []

    for result in results:
        content = result.get("content", "").strip()
        if not content:
            continue

        chunks.append({
            "text": content,
            "topic": "web_search",
            "metadata": {
                "source": "tavily_web_search",
                "title": result.get("title", "Untitled"),
                "url": result.get("url", ""),
            },
            "similarity": None,  # web results don't have cosine similarity
        })

    return chunks
