"""Recall-quality self-eval as a gating test.

Ingests fixtures/conversations.json, runs fixtures/probes.json against /recall,
and asserts a quality bar. Also enforces that the *critical* categories never
regress: fact evolution returns the current value, noise queries stay empty, and
users never bleed into each other.

Run verbosely to see the full report:  pytest -s tests/test_recall_quality.py
"""

from __future__ import annotations

import pytest

from . import harness

pytestmark = pytest.mark.usefixtures("require_service")


def test_recall_quality_fixture():
    report = harness.run_selfeval()
    by_id = {r["id"]: r for r in report["results"]}

    # Human-readable report (visible with -s).
    print(f"\nSelf-eval: {report['passed']}/{report['n_probes']} probes passed "
          f"after ingesting {report['n_turns']} turns")
    for r in report["results"]:
        if not r["passed"]:
            print(f"  FAIL {r['id']}: {'; '.join(r['notes'])}")

    # Overall bar: allow at most one soft miss.
    assert report["passed"] >= report["n_probes"] - 1, (
        f"recall quality regressed: {report['passed']}/{report['n_probes']}"
    )

    # Critical categories must pass exactly.
    assert by_id["work_superseded"]["passed"], "fact evolution: current value not returned"
    assert by_id["noise_pokemon"]["passed"], "noise resistance: off-topic query not empty"
    assert by_id["bob_live_isolation"]["passed"], "cross-user leak (live)"
    assert by_id["bob_work_isolation"]["passed"], "cross-user leak (work)"

    harness.reset_fixture_users()


def test_supersession_chain_inspectable():
    """After ingesting, the employment supersession chain is visible and correct."""
    harness.reset_fixture_users()
    harness.ingest_conversations()
    _, mem = harness.get_memories("u_alice")
    employment = [m for m in mem["memories"] if m["key"] == "employment.company"]
    active = [m for m in employment if m["active"]]
    inactive = [m for m in employment if not m["active"]]

    assert len(active) == 1, "exactly one current employer"
    assert "Notion" in active[0]["value"]
    assert any("Stripe" in m["value"] for m in inactive), "old employer kept as history"
    # The active row links back to what it replaced.
    assert active[0]["supersedes"] is not None
    harness.reset_fixture_users()
