"""Shared pytest fixtures.

The suite exercises the service over HTTP (the real contract surface), so it
needs a running instance at MEMORY_BASE_URL (default http://localhost:8080).
If none is reachable, every test is skipped with a clear message rather than
failing — so `pytest` is safe to run without a service up.
"""

from __future__ import annotations

import uuid

import pytest

from . import harness


@pytest.fixture(scope="session", autouse=True)
def _require_service():
    try:
        status, _ = harness.health()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no service reachable at {harness.BASE}: {exc}")
    if status != 200:
        pytest.skip(f"service at {harness.BASE} not healthy (status {status})")


@pytest.fixture
def uid() -> str:
    return f"test_user_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def sid() -> str:
    return f"test_sess_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def cleanup_users():
    created: list[str] = []
    yield created
    for u in created:
        try:
            harness.delete_user(u)
        except Exception:  # noqa: BLE001
            pass
