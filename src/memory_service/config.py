"""Runtime configuration, loaded from environment variables (and an optional
.env file). Every value has a safe default so the service boots with no config."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Storage ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql://memory:memory@localhost:5432/memory"

    # ── Auth ─────────────────────────────────────────────────────────────────
    # If set, all endpoints except /health require `Authorization: Bearer <token>`.
    memory_auth_token: str | None = None

    # ── Extraction LLM ───────────────────────────────────────────────────────
    extraction_provider: str = "auto"  # auto | anthropic | openai | rules
    extraction_model: str | None = None  # overrides the per-provider default below
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"  # used when provider resolves to openai
    extraction_timeout_s: float = 45.0

    # ── Service ──────────────────────────────────────────────────────────────
    port: int = 8080

    # ── Embeddings & reranking (local, fastembed) ────────────────────────────
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    embedding_dim: int = 384

    # ── Recall tuning ────────────────────────────────────────────────────────
    # Relevance is gated by EITHER signal clearing its bar (logical OR):
    #   * reranker score — calibrated to this cross-encoder; great at ordering,
    #     low absolute magnitude for short facts, so the floor is small.
    #   * bge cosine — better-calibrated absolute scale (relevant ~0.6+, noise
    #     ~0.4), the primary noise-resistance signal.
    # Inclusion bar — which individual facts/turns to SHOW once we've decided to respond.
    recall_relevance_threshold: float = 0.008  # reranker sigmoid floor
    recall_cosine_threshold: float = 0.61  # bge cosine floor
    # Response gate — whether to respond at all (noise resistance). Driven mainly by
    # TURN relevance: turns carry full context and separate on-topic (~0.67+) from
    # noise (~0.56) with a wide margin, where short facts are knife-edge. A single
    # borderline fact collision must NOT trigger a profile dump, so the memory gate
    # is set high.
    recall_gate_turn_cosine: float = 0.62
    recall_gate_memory_cosine: float = 0.66
    recall_query_decomposition: bool = True
    recall_candidate_k: int = 40  # per source, per retrieval arm
    recent_turns_in_context: int = 4

    # ── Resilience ───────────────────────────────────────────────────────────
    max_turn_bytes: int = 200_000  # reject request bodies larger than this (~200 KB)
    max_messages_per_turn: int = 200
    max_message_chars: int = 100_000

    # ── Misc ─────────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    @property
    def resolved_provider(self) -> str:
        """Resolve 'auto' to a concrete provider based on which keys exist."""
        p = (self.extraction_provider or "auto").strip().lower()
        if p != "auto":
            return p
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "rules"

    @property
    def resolved_model(self) -> str | None:
        if self.extraction_model:
            return self.extraction_model
        prov = self.resolved_provider
        if prov == "anthropic":
            return "claude-3-5-haiku-latest"
        if prov == "openai":
            return self.openai_model
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
