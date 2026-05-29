"""Contract compliance: endpoints exist, shapes match, status codes are correct,
and a write is immediately readable (no eventual consistency)."""

from __future__ import annotations

import pytest

from . import harness

pytestmark = pytest.mark.usefixtures("require_service")


def test_health():
    status, body = harness.health()
    assert status == 200
    assert body["status"] == "ok"


def test_turn_roundtrip_shapes(uid, sid, cleanup_users):
    cleanup_users.append(uid)
    status, body = harness.post_turn(
        sid, uid,
        [{"role": "user", "content": "I live in Lisbon and I work at Figma as a designer."},
         {"role": "assistant", "content": "Lisbon is lovely!"}],
        timestamp="2025-05-01T12:00:00Z",
    )
    assert status == 201
    assert isinstance(body["id"], str) and body["id"]

    # Immediately recallable (synchronous correctness).
    status, rec = harness.recall("Where does the user live?", "other_sess", uid, 512)
    assert status == 200
    assert "context" in rec and "citations" in rec
    assert "Lisbon" in rec["context"]
    for c in rec["citations"]:
        assert set(c.keys()) >= {"turn_id", "score", "snippet"}

    # Memories are structured + typed, not raw chunks.
    status, mem = harness.get_memories(uid)
    assert status == 200
    assert len(mem["memories"]) >= 1
    m = mem["memories"][0]
    assert set(m.keys()) >= {"id", "type", "key", "value", "confidence", "active"}
    assert m["type"] in {"fact", "preference", "opinion", "event"}
    keys = {x["key"] for x in mem["memories"]}
    assert any(k.startswith("location") for k in keys)
    assert any(k.startswith("employment") for k in keys)


def test_search_shape(uid, sid, cleanup_users):
    cleanup_users.append(uid)
    harness.post_turn(sid, uid, [{"role": "user", "content": "I work at Figma."}],
                      timestamp="2025-05-01T12:00:00Z")
    status, body = harness.search("employer", user_id=uid, limit=5)
    assert status == 200
    assert "results" in body
    if body["results"]:
        r = body["results"][0]
        assert set(r.keys()) >= {"content", "score", "session_id", "timestamp", "metadata"}


def test_recall_cold_session_is_empty_not_error(uid):
    # Never error on a cold session/user with no data.
    status, body = harness.recall("anything at all", "cold_sess", uid, 256)
    assert status == 200
    assert body["context"] == ""
    assert body["citations"] == []


def test_delete_session_returns_204(uid, sid, cleanup_users):
    cleanup_users.append(uid)
    harness.post_turn(sid, uid, [{"role": "user", "content": "I have a dog named Rex."}],
                      timestamp="2025-05-01T12:00:00Z")
    status, _ = harness.delete_session(sid)
    assert status == 204


def test_delete_user_returns_204(uid, sid):
    harness.post_turn(sid, uid, [{"role": "user", "content": "I live in Oslo."}],
                      timestamp="2025-05-01T12:00:00Z")
    status, _ = harness.delete_user(uid)
    assert status == 204
    # After deletion, nothing remains.
    status, mem = harness.get_memories(uid)
    assert status == 200
    assert mem["memories"] == []
