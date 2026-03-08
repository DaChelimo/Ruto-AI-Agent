"""Memory store skeleton: ingest sources -> index -> retrieve."""

from pydantic import BaseModel, field_validator, ValidationError
from dataclasses import dataclass, field
from typing import Any
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any
from .llm import query_chunker_llm, embed, embed_batch
from .memory_chunk import MemoryChunk
from .memory_chunk import ChunkSchema
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@dataclass
class MemoryStore:
    """Container for memory chunks and retrieval interface."""

    chunks: list[MemoryChunk] = field(default_factory=list)

    RETRIEVED_CHUNK_SIZE = 7



    def retrieve(self, query: str) -> list[dict]:
        """Return the most relevant chunks for a given user query."""

        query_vector = np.array(embed(query))
    
        similarities = []
        for chunk in self.chunks:
            chunk_vector = np.array(chunk.metadata["embedding"])
            score = np.dot(query_vector, chunk_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector)
            )
            similarities.append(score)
        
        # texts = [chunk.text for chunk in self.chunks]
        # texts.append(query)

        # vectorizer = TfidfVectorizer()
        # tfidf_matrix = vectorizer.fit_transform(texts)

        # chunk_vectors = tfidf_matrix[:-1]
        # query_vector = tfidf_matrix[-1]

        # cos_similarity = cosine_similarity(query_vector, chunk_vectors)[0]
        sorted_indices = np.array(similarities).argsort()[::-1][:self.RETRIEVED_CHUNK_SIZE]

        retrieved_chunks = []
        for index in sorted_indices:
            chunk = self.chunks[index]

            retrieved_chunks.append({
                "text": chunk.text,
                "topic": chunk.topic,
                "metadata": chunk.metadata,
                "similarity": float(similarities[index])
            })
        
        return retrieved_chunks




def build_memory_store(out_dir: str = "data/index") -> None:
    """Build the memory store from the sources."""
    # Step 1: populate data/sources.json with sources you find
    # Step 2: parse through sources and create MemoryChunks
    # Step 3: Index the MemoryChunks
    # Step 4: Save the memory store to data/index.json
    with open("data/sources.json", "r") as f:
        sources = json.load(f)

    chunks = [] 
    lastChunkCount = 0
    
    for source in sources:
        newChunks = convert_source_to_chunks(source, lastChunkCount) or []
        lastChunkCount += len(newChunks)
        chunks.extend(newChunks)

    with open("data/index.json", "w") as f:
        json.dump(asdict(MemoryStore(chunks=chunks)), f, indent=2)

    print(f"Built memory store with {len(chunks)}")


def convert_source_to_chunks(source: dict[str, Any], lastCount: int) -> list[MemoryChunk]:
    chunks = []

    schema_str = json.dumps(ChunkSchema.model_json_schema(), indent=2)

    prompt = f"""
    Role: You are a text chunker skilled in understanding articles, breaking them down, and extracting their category.
    Task: Given an article, generate focused chunks of 3 to 5 sentences each and label each chunk's topic.

    Each chunk must follow this exact JSON schema:
    {schema_str}

    Return ONLY a valid JSON array of objects matching the schema above.
    No markdown fences, no explanation: pure JSON only.

    Article title: {source['title']}
    Article text:
    {source['text']}

    VERY IMPORTANT: Your entire output must be pure JSON with no markdown fences or explanations.
    """

    raw = query_chunker_llm(prompt)
    cleaned = raw.strip()

    chunks = []

    # In case it starts with the ``` json, remove it
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").lstrip("json").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```").strip()
    
    items = []
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        print("JSONDecodeError occurred")

    
    for index, item in enumerate(items):
        try:
            validated = ChunkSchema(**item)

            chunk = MemoryChunk(
                text=validated.text,
                topic=validated.topic,
                metadata={
                    "source": source["source"],
                    "title":  source["title"],
                    "url":    source["url"],
                    "chunk_id": lastCount + index,
                }
            )

            chunks.append(chunk)

        except ValidationError:
            print("ValidationError occurred")
            continue
    
    
    embeddings = embed_batch([chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.metadata["embedding"] = embedding
    
    return chunks




def load_memory_store(index_dir: str = "data/index") -> MemoryStore:
    """Load a memory store from disk."""
    with open("data/index.json", "r") as f:
        data = json.load(f)
    chunks = [MemoryChunk(**chunk) for chunk in data["chunks"]]

    print(f"Loaded memory store with {len(chunks)} chunks.")
    return MemoryStore(chunks=chunks)


def main() -> None:
    """Entry point: build the memory store and confirm it loads."""
    build_memory_store()
    store = load_memory_store()
    print(f"Verification: loaded {len(store.chunks)} chunks.")

    # Optional: test a quick retrieval to make sure it works
    if store.chunks:
        test_results = store.retrieve("Tell me about their career")
        print(f"Test retrieval returned {len(test_results)} results.")
        for r in test_results:
            print(f"  [{r['topic']}] {r['text'][:80]}...")


if __name__ == "__main__":
    main()

