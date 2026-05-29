"""HTTP surface for the memory service (the §3 contract).

Sync route handlers (FastAPI runs them in a threadpool) keep read-after-write
trivial: each request does its work in one pooled connection and commits before
returning. A global exception handler guarantees the process never dies on bad
input — every error becomes a 4xx/5xx JSON response.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from . import db, store
from .config import get_settings
from .context import assemble
from .embeddings import get_embedder
from .extraction import extract
from .logging_conf import configure_logging, log_event
from .models import (
    MemoriesResponse,
    RecallRequest,
    RecallResponse,
    SearchRequest,
    SearchResponse,
    TurnRequest,
    TurnResponse,
)
from .retrieval import gather_for_recall, search
from .taxonomy import memory_search_text

log = logging.getLogger("memory.api")
_READY = {"db": False, "models": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log_event(log, "startup.begin", provider=settings.resolved_provider,
              model=settings.resolved_model, embed_model=settings.embed_model)
    db.init_db()
    _READY["db"] = True
    try:
        get_embedder()  # warm the ONNX models so the first request isn't slow
        _READY["models"] = True
    except Exception as exc:  # noqa: BLE001
        log.error("model warmup failed (retrieval will retry lazily): %s", exc)
    log_event(log, "startup.ready", **_READY)
    yield
    db.close_pool()


app = FastAPI(title="memory-service", version="0.7.0", lifespan=lifespan)


# ── Auth + payload guards ────────────────────────────────────────────────────
def require_auth(authorization: str | None = Header(default=None)) -> None:
    token = get_settings().memory_auth_token
    if not token:
        return  # auth disabled
    expected = f"Bearer {token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@app.middleware("http")
async def guards(request: Request, call_next):
    # Oversized payload guard. We drain the body before replying 413 so the
    # client's in-flight upload completes and receives a clean response instead
    # of a broken pipe — while still skipping the expensive parse/extract/embed
    # pipeline for a payload we're rejecting.
    settings = get_settings()
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > settings.max_turn_bytes:
        try:
            await request.body()
        except Exception:  # noqa: BLE001
            pass
        return JSONResponse(status_code=413, content={"detail": "payload too large"})
    try:
        return await call_next(request)
    except Exception:  # noqa: BLE001 - last-resort safety net; never crash a worker
        log.exception("unhandled error in request pipeline")
        return JSONResponse(status_code=500, content={"detail": "internal error"})


@app.exception_handler(Exception)
async def on_unhandled(_: Request, exc: Exception):
    log.exception("unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# ── Helpers ──────────────────────────────────────────────────────────────────
def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _normalize_ts(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return ts
    except (ValueError, AttributeError):
        return None


def _render_turn(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role = (m.get("role") or "user").strip()
        name = m.get("name")
        tag = role if not name else f"{role}/{name}"
        parts.append(f"{tag}: {content}")
    return "\n".join(parts)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    ok = db.healthcheck()
    if not ok:
        raise HTTPException(status_code=503, detail="database not ready")
    return {"status": "ok", "ready": _READY}


@app.post("/turns", status_code=201, response_model=TurnResponse, dependencies=[Depends(require_auth)])
def post_turn(req: TurnRequest) -> TurnResponse:
    settings = get_settings()
    messages = [m.model_dump() for m in req.messages[: settings.max_messages_per_turn]]
    for m in messages:  # resilience: clamp size + strip chars Postgres text rejects
        content = m.get("content")
        if isinstance(content, str):
            if "\x00" in content:
                content = content.replace("\x00", "")
            if len(content) > settings.max_message_chars:
                content = content[: settings.max_message_chars]
            m["content"] = content

    text_repr = _render_turn(messages)
    ts = _normalize_ts(req.timestamp)
    embedder = get_embedder()
    turn_vec = embedder.embed_documents([text_repr])[0] if text_repr.strip() else None

    # Extraction (may call the LLM; falls back to rules on any failure).
    candidates, method = extract(messages, ts, settings)
    cand_docs = [memory_search_text(c.key, c.value) for c in candidates]
    cand_vecs = embedder.embed_documents(cand_docs) if candidates else []

    pool = db.get_pool()
    with pool.connection() as conn:
        turn_id = store.insert_turn(
            conn, session_id=req.session_id, user_id=req.user_id, messages=messages,
            text_repr=text_repr, embedding=turn_vec, ts=ts, metadata=req.metadata,
        )
        actions = []
        for cand, vec in zip(candidates, cand_vecs, strict=False):
            actions.append(
                store.apply_candidate(
                    conn, cand, user_id=req.user_id, session_id=req.session_id,
                    source_turn=turn_id, embedding=vec, observed_at=ts,
                )
            )
        conn.commit()

    log_event(log, "turn.ingested", turn_id=turn_id, session_id=req.session_id,
              user_id=req.user_id, n_messages=len(messages), extraction=method,
              n_candidates=len(candidates), actions=[a["action"] for a in actions])
    return TurnResponse(id=turn_id)


@app.post("/recall", response_model=RecallResponse, dependencies=[Depends(require_auth)])
def post_recall(req: RecallRequest) -> RecallResponse:
    settings = get_settings()
    embedder = get_embedder()
    pool = db.get_pool()
    with pool.connection() as conn:
        gathered = gather_for_recall(
            conn, embedder, settings, query=req.query, user_id=req.user_id,
            session_id=req.session_id,
        )
    gathered["session_id"] = req.session_id
    context, citations = assemble(
        query=req.query, gathered=gathered, max_tokens=req.max_tokens,
        threshold=settings.recall_relevance_threshold,
        cosine_threshold=settings.recall_cosine_threshold,
        turn_gate=settings.recall_gate_turn_cosine,
        memory_gate=settings.recall_gate_memory_cosine,
    )
    log_event(log, "recall", session_id=req.session_id, user_id=req.user_id,
              n_citations=len(citations), context_chars=len(context))
    return RecallResponse(context=context, citations=citations)


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_auth)])
def post_search(req: SearchRequest) -> SearchResponse:
    settings = get_settings()
    embedder = get_embedder()
    pool = db.get_pool()
    with pool.connection() as conn:
        rows = search(
            conn, embedder, settings, query=req.query, user_id=req.user_id,
            session_id=req.session_id, limit=req.limit,
        )
    results = [
        {
            "content": r["content"],
            "score": round(float(r["score"]), 4),
            "session_id": r.get("session_id"),
            "timestamp": _iso(r.get("timestamp")),
            "metadata": r.get("metadata") or {},
        }
        for r in rows
    ]
    return SearchResponse(results=results)


@app.get("/users/{user_id}/memories", response_model=MemoriesResponse,
         dependencies=[Depends(require_auth)])
def get_memories(user_id: str) -> MemoriesResponse:
    pool = db.get_pool()
    with pool.connection() as conn:
        mems = store.list_user_memories(conn, user_id)
    return MemoriesResponse(memories=mems)


@app.delete("/sessions/{session_id}", status_code=204, dependencies=[Depends(require_auth)])
def delete_session(session_id: str) -> Response:
    pool = db.get_pool()
    with pool.connection() as conn:
        store.delete_session(conn, session_id)
        conn.commit()
    log_event(log, "session.deleted", session_id=session_id)
    return Response(status_code=204)


@app.delete("/users/{user_id}", status_code=204, dependencies=[Depends(require_auth)])
def delete_user(user_id: str) -> Response:
    pool = db.get_pool()
    with pool.connection() as conn:
        store.delete_user(conn, user_id)
        conn.commit()
    log_event(log, "user.deleted", user_id=user_id)
    return Response(status_code=204)
