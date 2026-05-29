"""Retrieval: hybrid (vector + lexical) candidate generation, RRF fusion, and
cross-encoder reranking.

Why hybrid + rerank rather than cosine-top-k:
  * Dense vectors (bge-small) capture paraphrase/semantic matches
    ("where do they live?" -> "Lives in Berlin").
  * Postgres full-text (BM25-like ts_rank_cd) nails keyword/entity matches
    ("dog's name?" -> "Biscuit") that embeddings of tiny strings often miss.
  * Reciprocal Rank Fusion merges the two rankings without score calibration.
  * A cross-encoder then rescensions the shortlist jointly over (query, doc),
    which is far more precise than either first-stage retriever and gives a
    calibrated 0..1 relevance used for the noise-resistance threshold.

Scoping: memories are user-scoped (shared across a user's sessions — see README);
turn text is searched across the user's sessions too, but "recent conversation"
is restricted to the current session. With no user_id, scope collapses to the
session.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import psycopg

from .config import Settings
from .embeddings import Embedder
from .llm import LLMError, complete_json, provider_available
from .taxonomy import memory_search_text

log = logging.getLogger("memory.retrieval")

RRF_K = 60


def _rrf(rankings: list[list[str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, ident in enumerate(ranking):
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


# ── First-stage arms ─────────────────────────────────────────────────────────
def _rows(cur: psycopg.Cursor) -> list[dict]:
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


def _vector_memories(conn, qvec, user_id, k) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, type, key, value, confidence, cardinality, attrs,
                   source_session, source_turn, created_at, updated_at, embedding
            FROM memories
            WHERE active = true AND embedding IS NOT NULL
              AND user_id IS NOT DISTINCT FROM %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (user_id, qvec, k),
        )
        return _rows(cur)


def _fts_memories(conn, query, user_id, k) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, type, key, value, confidence, cardinality, attrs,
                   source_session, source_turn, created_at, updated_at, embedding
            FROM memories
            WHERE active = true
              AND user_id IS NOT DISTINCT FROM %s
              AND tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) DESC
            LIMIT %s
            """,
            (user_id, query, query, k),
        )
        return _rows(cur)


def _turn_scope_clause(user_id: str | None, session_id: str | None) -> tuple[str, list]:
    # Prefer user scope (share across a user's sessions); else session scope.
    if user_id is not None:
        return "user_id = %s", [user_id]
    if session_id is not None:
        return "session_id = %s", [session_id]
    return "true", []


def _vector_turns(conn, qvec, user_id, session_id, k) -> list[dict]:
    clause, params = _turn_scope_clause(user_id, session_id)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, session_id, text_repr, ts, metadata, embedding
            FROM turns
            WHERE embedding IS NOT NULL AND {clause}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (*params, qvec, k),
        )
        return _rows(cur)


def _fts_turns(conn, query, user_id, session_id, k) -> list[dict]:
    clause, params = _turn_scope_clause(user_id, session_id)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, session_id, text_repr, ts, metadata, embedding
            FROM turns
            WHERE {clause} AND tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) DESC
            LIMIT %s
            """,
            (*params, query, query, k),
        )
        return _rows(cur)


def recent_turns(conn, session_id: str | None, limit: int) -> list[dict]:
    if not session_id:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, session_id, text_repr, ts, metadata, embedding
            FROM turns WHERE session_id = %s
            ORDER BY ts DESC, created_at DESC LIMIT %s
            """,
            (session_id, limit),
        )
        return _rows(cur)


def all_active_memories(conn, user_id: str | None, session_id: str | None) -> list[dict]:
    """Every active memory in scope — the basis for the recall 'facts' section."""
    if user_id is not None:
        where, params = "user_id = %s", [user_id]
    elif session_id is not None:
        where, params = "source_session = %s", [session_id]
    else:
        return []
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT m.id, m.type, m.key, m.value, m.confidence, m.cardinality, m.attrs,
                   m.source_session, m.source_turn, m.created_at, m.updated_at,
                   m.supersedes, m.embedding,
                   COALESCE(m.observed_at, m.updated_at) AS observed_at,
                   p.value AS prev_value
            FROM memories m
            LEFT JOIN memories p ON p.id = m.supersedes
            WHERE m.active = true AND m.{where}
            ORDER BY m.updated_at DESC
            LIMIT 300
            """,
            params,
        )
        return _rows(cur)


# ── Fusion + rerank ──────────────────────────────────────────────────────────
def _fuse(arms: list[list[dict]]) -> list[dict]:
    by_id: dict[str, dict] = {}
    rankings: list[list[str]] = []
    for arm in arms:
        order = []
        for row in arm:
            ident = str(row["id"])
            by_id.setdefault(ident, row)
            order.append(ident)
        rankings.append(order)
    fused = _rrf(rankings)
    out = []
    for ident, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        row = dict(by_id[ident])
        row["rrf"] = score
        out.append(row)
    return out


def _rerank(embedder: Embedder, query: str, rows: list[dict], text_of, limit: int) -> list[dict]:
    if not rows:
        return []
    shortlist = rows[:limit]
    docs = [text_of(r) for r in shortlist]
    scores = embedder.rerank(query, docs)
    for r, s in zip(shortlist, scores, strict=False):
        r["score"] = float(s)
    shortlist.sort(key=lambda r: r["score"], reverse=True)
    return shortlist


def _mem_text(r: dict) -> str:
    return memory_search_text(r.get("key") or "", r.get("value") or "")


def _turn_text(r: dict) -> str:
    return r.get("text_repr") or ""


def _cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _attach_cosine(rows: list[dict], qvec) -> None:
    """Add a query→passage cosine to each row (the calibrated noise-gate signal),
    then drop the bulky embedding so it doesn't travel further."""
    for r in rows:
        r["cosine"] = _cos(qvec, r.get("embedding"))
        r.pop("embedding", None)


# ── Multi-hop query decomposition (optional, LLM) ────────────────────────────
_DECOMP_SYSTEM = """\
You rewrite a user's recall question into 1-3 short, self-contained search \
queries that together retrieve everything needed to answer it. For multi-hop \
questions, split the hops. Output ONLY JSON: {"queries": ["...", "..."]}. \
Keep the original phrasing as the first query."""


def decompose_query(settings: Settings, query: str) -> list[str]:
    if not (settings.recall_query_decomposition and provider_available(settings)):
        return [query]
    try:
        obj = complete_json(settings, _DECOMP_SYSTEM, query)
        qs = obj.get("queries", [])
        out = [query] + [str(q).strip() for q in qs if isinstance(q, str) and q.strip()]
        # dedup, cap
        seen, uniq = set(), []
        for q in out:
            if q.lower() not in seen:
                seen.add(q.lower())
                uniq.append(q)
        return uniq[:4]
    except (LLMError, Exception) as exc:  # noqa: BLE001
        log.warning("query decomposition failed: %s", exc)
        return [query]


# ── Orchestration ────────────────────────────────────────────────────────────
def gather_for_recall(
    conn, embedder: Embedder, settings: Settings, *, query: str, user_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    """Produce reranked, scored material for context assembly."""
    queries = decompose_query(settings, query) if query.strip() else [query]
    k = settings.recall_candidate_k
    qvec_main = embedder.embed_query(query) if query.strip() else None

    # Turn candidates: union hybrid arms across all sub-queries, fuse, rerank.
    turn_arms: list[list[dict]] = []
    for q in queries:
        qvec = embedder.embed_query(q)
        turn_arms.append(_vector_turns(conn, qvec, user_id, session_id, k))
        turn_arms.append(_fts_turns(conn, q, user_id, session_id, k))
    fused_turns = _fuse(turn_arms) if any(turn_arms) else []
    reranked_turns = _rerank(embedder, query or " ", fused_turns, _turn_text, limit=30)

    # Active memories: rerank ALL active memories against the query so the facts
    # section can rank by genuine relevance (and we get the noise-resistance gate).
    active = all_active_memories(conn, user_id, session_id)
    reranked_mems = _rerank(embedder, query or " ", active, _mem_text, limit=len(active)) if active else []

    # Attach the calibrated cosine signal used by the relevance gate.
    _attach_cosine(reranked_mems, qvec_main)
    _attach_cosine(reranked_turns, qvec_main)

    recents = recent_turns(conn, session_id, settings.recent_turns_in_context)

    return {
        "queries": queries,
        "memories": reranked_mems,   # all active, scored
        "turns": reranked_turns,     # query-relevant, scored
        "recent_turns": recents,     # current-session continuity
    }


def search(
    conn, embedder: Embedder, settings: Settings, *, query: str, user_id: str | None,
    session_id: str | None, limit: int,
) -> list[dict]:
    """Structured search over memories + turns for the /search tool endpoint."""
    if not query.strip():
        return []
    k = settings.recall_candidate_k
    qvec = embedder.embed_query(query)

    mem_arms = [
        _vector_memories(conn, qvec, user_id, k) if user_id is not None else [],
        _fts_memories(conn, query, user_id, k) if user_id is not None else [],
    ]
    fused_mem = _fuse(mem_arms)
    turn_arms = [
        _vector_turns(conn, qvec, user_id, session_id, k),
        _fts_turns(conn, query, user_id, session_id, k),
    ]
    fused_turns = _fuse(turn_arms)

    reranked_mem = _rerank(embedder, query, fused_mem, _mem_text, limit=limit * 2)
    reranked_turn = _rerank(embedder, query, fused_turns, _turn_text, limit=limit * 2)
    _attach_cosine(reranked_mem, qvec)
    _attach_cosine(reranked_turn, qvec)

    def relevance(r: dict) -> float:
        # Blend so short facts (tiny reranker magnitude) still get a meaningful,
        # orderable score; reranker dominates when confident.
        return round(0.6 * r.get("score", 0.0) + 0.4 * max(0.0, r.get("cosine", 0.0)), 6)

    results: list[dict] = []
    for r in reranked_mem:
        results.append(
            {
                "content": r["value"],
                "score": relevance(r),
                "session_id": r.get("source_session"),
                "timestamp": r.get("updated_at"),
                "metadata": {"kind": "memory", "type": r["type"], "key": r["key"],
                             "confidence": float(r["confidence"])},
            }
        )
    for r in reranked_turn:
        text = r["text_repr"]
        results.append(
            {
                "content": text if len(text) <= 600 else text[:600] + "…",
                "score": relevance(r),
                "session_id": r.get("session_id"),
                "timestamp": r.get("ts"),
                "metadata": {"kind": "turn", **(r.get("metadata") or {})},
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]
