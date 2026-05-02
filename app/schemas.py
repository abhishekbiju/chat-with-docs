from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=32000)


class SourceCitation(BaseModel):
    """One retrieved span shown to the user (and optionally the model)."""

    excerpt: str = Field(..., description="Short text preview of the chunk.")
    source_file: str | None = None
    page: int | None = None
    relevance_score: float | None = Field(
        default=None,
        description="Chroma vector distance for this hit (lower = more similar). Omitted for MMR-only retrieval.",
    )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=50,
        description="Prior turns for conversational follow-ups.",
    )
    k_retrieval: int = Field(default=4, ge=1, le=20)
    use_mmr: bool = Field(
        default=False,
        description="Use Maximal Marginal Relevance for more diverse context.",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


class StreamDonePayload(BaseModel):
    answer: str
    sources: list[SourceCitation]
