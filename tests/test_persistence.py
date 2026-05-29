"""Restart persistence: data written before `docker compose down` is recallable
after `docker compose up`.

This drives real Docker, so it is opt-in: set RUN_DOCKER_PERSISTENCE=1 (and have
the compose stack as the MEMORY_BASE_URL target). Without it, the test skips.
`scripts/persistence_check.sh` does the same thing as a standalone script.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid

import pytest

from . import harness

RUN = os.environ.get("RUN_DOCKER_PERSISTENCE") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_DOCKER_PERSISTENCE=1 to run the docker restart test"
)


def _compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _wait_healthy(timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, _ = harness.health()
            if status == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise AssertionError("service did not become healthy after restart")


def test_data_survives_compose_restart():
    uid = f"persist_{uuid.uuid4().hex[:8]}"
    sid = "persist_sess"

    status, _ = harness.post_turn(
        sid, uid, [{"role": "user", "content": "I live in Reykjavik and own a horse named Skjoni."}],
        timestamp="2025-05-01T09:00:00Z",
    )
    assert status == 201

    # Full down/up cycle — named volume must preserve the data.
    _compose("down")
    _compose("up", "-d")
    _wait_healthy()

    status, rec = harness.recall("Where does the user live?", "after_restart", uid, 512)
    assert status == 200
    assert "Reykjavik" in rec["context"], "data did not survive restart"

    harness.delete_user(uid)
