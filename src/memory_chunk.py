from pydantic import BaseModel, Field, field_validator # type: ignore
from typing import Literal
import json
from dataclasses import dataclass, field

@dataclass
class MemoryChunk:
    """Single memory chunk stored in the index."""

    text: str
    topic: str
    metadata: dict[str, any] = field(default_factory=dict)


class ChunkSchema(BaseModel):
    text: str = Field(
        description="The chunk text, 3 to 5 sentences from the article."
    )
    topic: Literal[
        "career", "personal_life", "values", "style", "controversy", "general"
    ] = Field(
        description=(
            "Topic label: career=work/films/music/awards, "
            "personal_life=family/upbringing/relationships, "
            "values=beliefs/philosophy/advice, "
            "style=fashion/aesthetic/brand, "
            "controversy=scandals/criticism/feuds, "
            "general=anything else"
        )
    )

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, v):
        return v.lower().strip().replace(" ", "_")

    @field_validator("text")
    @classmethod
    def text_must_be_meaningful(cls, v):
        if len(v.strip()) < 20:
            raise ValueError("chunk text too short")
        return v.strip()