"""Self-eval harness + tiny HTTP client (stdlib only, no extra deps).

Usable two ways:
  * imported by the pytest suite (contract + recall-quality tests), and
  * run directly as the iteration loop:  `python tests/harness.py`
    which ingests fixtures/conversations.json, runs fixtures/probes.json against
    /recall, and prints a quality report.

Target service is MEMORY_BASE_URL (default http://localhost:8080); if
MEMORY_AUTH_TOKEN is set it is sent as a bearer token.
"""

from __future__ import annotations

import json
import os
import pathlib
import urllib.error
import urllib.request

BASE = os.environ.get("MEMORY_BASE_URL", "http://localhost:8080").rstrip("/")
TOKEN = os.environ.get("MEMORY_AUTH_TOKEN") or None
FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"


def _request(method: str, path: str, body: dict | None = None, timeout: float = 60.0):
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw
        return e.code, payload


# ── Thin API wrappers ────────────────────────────────────────────────────────
def health():
    return _request("GET", "/health")


def post_turn(session_id, user_id, messages, timestamp=None, metadata=None):
    return _request("POST", "/turns", {
        "session_id": session_id, "user_id": user_id, "messages": messages,
        "timestamp": timestamp, "metadata": metadata or {},
    })


def recall(query, session_id, user_id, max_tokens=1024):
    return _request("POST", "/recall", {
        "query": query, "session_id": session_id, "user_id": user_id, "max_tokens": max_tokens,
    })


def search(query, session_id=None, user_id=None, limit=10):
    return _request("POST", "/search", {
        "query": query, "session_id": session_id, "user_id": user_id, "limit": limit,
    })


def get_memories(user_id):
    return _request("GET", f"/users/{user_id}/memories")


def delete_user(user_id):
    return _request("DELETE", f"/users/{user_id}")


def delete_session(session_id):
    return _request("DELETE", f"/sessions/{session_id}")


# ── Fixtures ─────────────────────────────────────────────────────────────────
def load_conversations() -> dict:
    return json.loads((FIXTURES / "conversations.json").read_text(encoding="utf-8"))


def load_probes() -> dict:
    return json.loads((FIXTURES / "probes.json").read_text(encoding="utf-8"))


def fixture_user_ids() -> list[str]:
    convs = load_conversations()["conversations"]
    return sorted({c["user_id"] for c in convs})


def ingest_conversations() -> int:
    """Ingest every turn in the fixture. Returns the number of turns written."""
    convs = load_conversations()["conversations"]
    n = 0
    for conv in convs:
        for turn in conv["turns"]:
            status, _ = post_turn(conv["session_id"], conv["user_id"],
                                  turn["messages"], turn.get("timestamp"), turn.get("metadata"))
            assert status == 201, f"turn ingest failed: {status}"
            n += 1
    return n


def reset_fixture_users() -> None:
    for uid in fixture_user_ids():
        delete_user(uid)


# ── Probe evaluation ─────────────────────────────────────────────────────────
def evaluate_probe(probe: dict) -> dict:
    status, body = recall(probe["query"], probe.get("session_id", "probe"),
                          probe.get("user_id"), probe.get("max_tokens", 1024))
    ctx = (body or {}).get("context", "") if status == 200 else ""
    low = ctx.lower()
    result = {"id": probe["id"], "category": probe.get("category", ""), "status": status,
              "passed": True, "notes": [], "context_len": len(ctx)}

    if probe.get("expect_empty"):
        if ctx.strip():
            result["passed"] = False
            result["notes"].append(f"expected empty, got {len(ctx)} chars")
        return result

    for needed in [probe.get("expect_any")] if probe.get("expect_any") else []:
        if not any(s.lower() in low for s in needed):
            result["passed"] = False
            result["notes"].append(f"missing any of {needed}")

    for forbidden in probe.get("expect_absent", []):
        if forbidden.lower() in low:
            result["passed"] = False
            result["notes"].append(f"leaked '{forbidden}'")
    return result


def run_selfeval() -> dict:
    reset_fixture_users()
    n_turns = ingest_conversations()
    probes = load_probes()["probes"]
    results = [evaluate_probe(p) for p in probes]
    passed = sum(1 for r in results if r["passed"])
    return {"n_turns": n_turns, "n_probes": len(probes), "passed": passed, "results": results}


def main() -> int:
    status, _ = health()
    if status != 200:
        print(f"service not healthy at {BASE} (status {status}). Is it running?")
        return 2
    report = run_selfeval()
    print(f"\nSelf-eval against {BASE}  —  ingested {report['n_turns']} turns\n")
    print(f"{'PROBE':<32} {'CATEGORY':<22} {'RESULT'}")
    print("-" * 72)
    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        note = "" if r["passed"] else "  <- " + "; ".join(r["notes"])
        print(f"{r['id']:<32} {r['category']:<22} {mark}{note}")
    print("-" * 72)
    score = report["passed"] / report["n_probes"] if report["n_probes"] else 0.0
    print(f"\nRecall quality: {report['passed']}/{report['n_probes']} probes passed  ({score:.0%})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
