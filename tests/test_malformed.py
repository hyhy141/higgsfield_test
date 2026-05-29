"""Resilience: malformed input, oversized payloads, and unicode oddities must
produce 4xx (or be handled), never crash the service. After every assault,
/health must still be 200."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from . import harness

pytestmark = pytest.mark.usefixtures("require_service")


def _raw_post(path: str, raw: bytes, content_type: str = "application/json"):
    req = urllib.request.Request(harness.BASE + path, data=raw, method="POST")
    req.add_header("Content-Type", content_type)
    if harness.TOKEN:
        req.add_header("Authorization", f"Bearer {harness.TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def test_bad_json_is_4xx():
    status = _raw_post("/turns", b"{not valid json at all ")
    assert 400 <= status < 500


def test_missing_required_field_is_422():
    # No session_id.
    status = _raw_post("/turns", json.dumps({"messages": []}).encode())
    assert status == 422


def test_wrong_types_are_4xx():
    # messages should be a list, not a string.
    body = {"session_id": "s", "user_id": "u", "messages": "oops"}
    status = _raw_post("/turns", json.dumps(body).encode())
    assert 400 <= status < 500


def test_unicode_and_control_chars_survive(uid, sid, cleanup_users):
    cleanup_users.append(uid)
    # CJK, emoji, accents, an embedded NUL (\x00, which Postgres text rejects),
    # an RTL-override, plus tab/newline — written as escapes so this source stays
    # pure ASCII.
    weird = (
        "I live in 東京 \U0001f5fc — café with a null\x00byte, "
        "rtl ‮override‬, tab\t and newline\n. I love 日本語."
    )
    status, body = harness.post_turn(sid, uid, [{"role": "user", "content": weird}],
                                     timestamp="2025-05-01T00:00:00Z")
    assert status == 201, "service must absorb unicode + NUL bytes, not 500"
    s2, _ = harness.health()
    assert s2 == 200
    s3, _ = harness.recall("where does the user live", "q", uid, 256)
    assert s3 == 200


def test_empty_messages_is_accepted(uid, sid, cleanup_users):
    cleanup_users.append(uid)
    status, body = harness.post_turn(sid, uid, [], timestamp="2025-05-01T00:00:00Z")
    assert status == 201
    assert isinstance(body["id"], str)


def test_oversized_payload_is_rejected(uid, sid):
    huge = "x" * 1_500_000  # ~1.5 MB, over the 1 MB guard
    body = {"session_id": sid, "user_id": uid, "messages": [{"role": "user", "content": huge}]}
    status = _raw_post("/turns", json.dumps(body).encode())
    assert status in (413, 422)
    s2, _ = harness.health()  # crucially, still alive
    assert s2 == 200


def test_bad_timestamp_does_not_crash(uid, sid, cleanup_users):
    cleanup_users.append(uid)
    status, _ = harness.post_turn(sid, uid, [{"role": "user", "content": "I live in Rome."}],
                                  timestamp="not-a-real-timestamp")
    assert status == 201  # tolerated; falls back to now()


def test_health_after_all_abuse():
    status, _ = harness.health()
    assert status == 200
