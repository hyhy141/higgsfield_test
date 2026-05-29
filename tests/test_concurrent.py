"""Cross-user / cross-session scoping: concurrent sessions for different users
must not bleed. (Same-user cross-session sharing is intentional — see README —
and is covered by the recall-quality fixture.)"""

from __future__ import annotations

import uuid

import pytest

from . import harness

pytestmark = pytest.mark.usefixtures("require_service")


def test_two_users_do_not_bleed(cleanup_users):
    a = f"alice_{uuid.uuid4().hex[:8]}"
    b = f"bob_{uuid.uuid4().hex[:8]}"
    cleanup_users.extend([a, b])

    harness.post_turn("sa", a, [{"role": "user", "content": "I live in Paris and work at Datadog."}],
                      timestamp="2025-05-01T10:00:00Z")
    harness.post_turn("sb", b, [{"role": "user", "content": "I live in Madrid and work at Spotify."}],
                      timestamp="2025-05-01T10:00:00Z")

    _, ra = harness.recall("Where does the user live and work?", "qa", a, 512)
    _, rb = harness.recall("Where does the user live and work?", "qb", b, 512)

    assert "Paris" in ra["context"] and "Datadog" in ra["context"]
    assert "Madrid" not in ra["context"] and "Spotify" not in ra["context"]

    assert "Madrid" in rb["context"] and "Spotify" in rb["context"]
    assert "Paris" not in rb["context"] and "Datadog" not in rb["context"]


def test_memories_are_user_scoped(cleanup_users):
    a = f"u_{uuid.uuid4().hex[:8]}"
    b = f"u_{uuid.uuid4().hex[:8]}"
    cleanup_users.extend([a, b])
    harness.post_turn("s1", a, [{"role": "user", "content": "My cat is named Luna."}],
                      timestamp="2025-05-01T10:00:00Z")
    _, mb = harness.get_memories(b)
    assert mb["memories"] == []  # b has nothing; a's memory did not leak
    _, ma = harness.get_memories(a)
    assert any("Luna" in m["value"] for m in ma["memories"])
