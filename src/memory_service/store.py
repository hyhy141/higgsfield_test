"""The memory store: persistence + fact-evolution (supersession) logic.

Everything runs inside the caller's transaction so a /turns request commits the
raw turn and all derived memories atomically — once it returns, every read path
sees them (no eventual consistency).

Supersession model
------------------
* single-valued key (job, city, marital status, a stance on a topic):
  a new, different value flips the previous active row to inactive, links the two
  (`supersedes` / `superseded_by`), and inserts the new value as active.
* multi-valued key (pets, allergies, skills, children):
  values coexist; a new value is matched to an existing one by `entity`. A
  correction/update to that specific item supersedes just that row; a brand-new
  item is inserted alongside.
* retraction / negative polarity ("no longer has a dog"):
  the matching active row is deactivated, no replacement inserted.
* near-duplicate restatement:
  refresh `updated_at` / confidence in place — no new row, no history churn.

Inactive rows are never deleted, so `/users/{id}/memories` shows the full chain.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import psycopg

from .extraction import Candidate

log = logging.getLogger("memory.store")


# ── Turns ────────────────────────────────────────────────────────────────────
def insert_turn(
    conn: psycopg.Connection,
    *,
    session_id: str,
    user_id: str | None,
    messages: list[dict],
    text_repr: str,
    embedding: list[float] | None,
    ts: str | None,
    metadata: dict,
) -> str:
    from psycopg.types.json import Jsonb

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO turns (session_id, user_id, messages, text_repr, embedding, ts, metadata)
            VALUES (%s, %s, %s, %s, %s, COALESCE(%s::timestamptz, now()), %s)
            RETURNING id
            """,
            (session_id, user_id, Jsonb(messages), text_repr, embedding, ts, Jsonb(metadata)),
        )
        return str(cur.fetchone()[0])


# ── Memory write path (supersession) ─────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower()).strip(" .,:;!?")


def _near_dup(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) >= 8 and len(nb) >= 8 and (na in nb or nb in na):
        return True
    return False


def _fetch_active(
    conn: psycopg.Connection, user_id: str | None, subject: str, key: str
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, value, confidence, attrs FROM memories
            WHERE active = true AND key = %s AND subject = %s
              AND user_id IS NOT DISTINCT FROM %s
            ORDER BY updated_at DESC
            """,
            (key, subject, user_id),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def _deactivate(conn: psycopg.Connection, mem_id: str, superseded_by: str | None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET active = false, superseded_by = %s, updated_at = now() WHERE id = %s",
            (superseded_by, mem_id),
        )


def _refresh(conn: psycopg.Connection, mem_id: str, value: str, confidence: float) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE memories
            SET value = %s,
                confidence = GREATEST(confidence, %s),
                updated_at = now()
            WHERE id = %s
            """,
            (value, confidence, mem_id),
        )


def _insert_memory(
    conn: psycopg.Connection,
    *,
    cand: Candidate,
    user_id: str | None,
    session_id: str | None,
    source_turn: str | None,
    embedding: list[float] | None,
    supersedes: str | None,
    observed_at: str | None,
) -> str:
    from psycopg.types.json import Jsonb

    attrs = dict(cand.attrs or {})
    if cand.entity:
        attrs.setdefault("entity", cand.entity)
    if cand.polarity:
        attrs.setdefault("polarity", cand.polarity)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO memories
                (user_id, session_id, subject, type, key, value, confidence,
                 cardinality, attrs, source_session, source_turn, embedding,
                 active, supersedes, observed_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,true,%s,
                    COALESCE(%s::timestamptz, now()))
            RETURNING id
            """,
            (
                user_id, session_id, cand.subject, cand.type, cand.key, cand.value,
                cand.confidence, cand.cardinality, Jsonb(attrs), session_id,
                source_turn, embedding, supersedes, observed_at,
            ),
        )
        return str(cur.fetchone()[0])


def apply_candidate(
    conn: psycopg.Connection,
    cand: Candidate,
    *,
    user_id: str | None,
    session_id: str | None,
    source_turn: str | None,
    embedding: list[float] | None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Apply one extracted candidate, returning a small action record for logging."""
    active = _fetch_active(conn, user_id, cand.subject, cand.key)

    # 1) Retraction / negation -> deactivate matching active row(s), no new row.
    if cand.operation == "retract" or cand.polarity == "negative":
        target = _match_for_multi(active, cand) if cand.cardinality == "multi" else active
        for row in target:
            _deactivate(conn, str(row["id"]), None)
        return {"action": "retract", "key": cand.key, "deactivated": len(target)}

    # 2) Multi-valued: match by entity; coexist otherwise.
    if cand.cardinality == "multi":
        matches = _match_for_multi(active, cand)
        if matches:
            row = matches[0]
            if cand.is_correction or not _near_dup(cand.value, row["value"]):
                new_id = _insert_memory(
                    conn, cand=cand, user_id=user_id, session_id=session_id,
                    source_turn=source_turn, embedding=embedding, observed_at=observed_at, supersedes=str(row["id"]),
                )
                _deactivate(conn, str(row["id"]), new_id)
                return {"action": "supersede(multi)", "key": cand.key, "id": new_id}
            _refresh(conn, str(row["id"]), cand.value, cand.confidence)
            return {"action": "refresh(multi)", "key": cand.key, "id": str(row["id"])}
        new_id = _insert_memory(
            conn, cand=cand, user_id=user_id, session_id=session_id,
            source_turn=source_turn, embedding=embedding, observed_at=observed_at, supersedes=None,
        )
        return {"action": "insert(multi)", "key": cand.key, "id": new_id}

    # 3) Single-valued: new distinct value supersedes the current one.
    if active:
        current = active[0]
        if _near_dup(cand.value, current["value"]) and not cand.is_correction:
            _refresh(conn, str(current["id"]), cand.value, cand.confidence)
            return {"action": "refresh(single)", "key": cand.key, "id": str(current["id"])}
        new_id = _insert_memory(
            conn, cand=cand, user_id=user_id, session_id=session_id,
            source_turn=source_turn, embedding=embedding, observed_at=observed_at, supersedes=str(current["id"]),
        )
        for row in active:  # deactivate any/all current actives under this key
            _deactivate(conn, str(row["id"]), new_id)
        return {"action": "supersede(single)", "key": cand.key, "id": new_id, "superseded": current["value"]}

    new_id = _insert_memory(
        conn, cand=cand, user_id=user_id, session_id=session_id,
        source_turn=source_turn, embedding=embedding, observed_at=observed_at, supersedes=None,
    )
    return {"action": "insert(single)", "key": cand.key, "id": new_id}


def _match_for_multi(active: list[dict], cand: Candidate) -> list[dict]:
    """For multi-valued keys, find the existing row(s) that refer to the same item."""
    if cand.entity:
        ent = cand.entity.strip().lower()
        hit = [r for r in active if str((r.get("attrs") or {}).get("entity", "")).strip().lower() == ent]
        if hit:
            return hit
        # entity might be embedded in the value text
        hit = [r for r in active if ent and ent in _norm(r["value"])]
        if hit:
            return hit
    return [r for r in active if _near_dup(cand.value, r["value"])]


# ── Read paths ───────────────────────────────────────────────────────────────
def _iso(dt: Any) -> str | None:
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except AttributeError:
        return str(dt)


def list_user_memories(conn: psycopg.Connection, user_id: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, type, key, value, confidence, cardinality, subject,
                   source_session, source_turn, created_at, updated_at,
                   supersedes, superseded_by, active
            FROM memories
            WHERE user_id = %s
            ORDER BY active DESC, updated_at DESC
            """,
            (user_id,),
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
    out = []
    for r in rows:
        out.append(
            {
                "id": str(r["id"]),
                "type": r["type"],
                "key": r["key"],
                "value": r["value"],
                "confidence": float(r["confidence"]),
                "cardinality": r["cardinality"],
                "subject": r["subject"],
                "source_session": r["source_session"],
                "source_turn": str(r["source_turn"]) if r["source_turn"] else None,
                "created_at": _iso(r["created_at"]),
                "updated_at": _iso(r["updated_at"]),
                "supersedes": str(r["supersedes"]) if r["supersedes"] else None,
                "superseded_by": str(r["superseded_by"]) if r["superseded_by"] else None,
                "active": bool(r["active"]),
            }
        )
    return out


# ── Deletes ──────────────────────────────────────────────────────────────────
def delete_session(conn: psycopg.Connection, session_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE session_id = %s OR source_session = %s",
                    (session_id, session_id))
        cur.execute("DELETE FROM turns WHERE session_id = %s", (session_id,))


def delete_user(conn: psycopg.Connection, user_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM turns WHERE user_id = %s", (user_id,))
