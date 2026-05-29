"""Pydantic models for the HTTP contract (§3).

Request models are permissive on input (resilience: we never want a slightly
odd-but-harmless payload to 422) but strict enough to reject malformed bodies.
Response models pin the exact contract shapes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Requests ─────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str = ""
    name: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, v: Any) -> str:
        # Tolerate null content and non-string content (e.g. structured tool output).
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        return str(v)


class TurnRequest(BaseModel):
    session_id: str
    user_id: str | None = None
    messages: list[Message] = Field(default_factory=list)
    timestamp: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, v: Any) -> dict[str, Any]:
        return v if isinstance(v, dict) else {}


class RecallRequest(BaseModel):
    query: str = ""
    session_id: str | None = None
    user_id: str | None = None
    max_tokens: int = 1024

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _clamp_tokens(cls, v: Any) -> int:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 1024
        return max(32, min(n, 32_000))


class SearchRequest(BaseModel):
    query: str = ""
    session_id: str | None = None
    user_id: str | None = None
    limit: int = 10

    @field_validator("limit", mode="before")
    @classmethod
    def _clamp_limit(cls, v: Any) -> int:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 10
        return max(1, min(n, 100))


# ── Responses ────────────────────────────────────────────────────────────────
class TurnResponse(BaseModel):
    id: str


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: list[Citation] = Field(default_factory=list)


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)


class MemoryOut(BaseModel):
    id: str
    type: str
    key: str
    value: str
    confidence: float
    cardinality: str
    subject: str
    source_session: str | None = None
    source_turn: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    active: bool


class MemoriesResponse(BaseModel):
    memories: list[MemoryOut] = Field(default_factory=list)
