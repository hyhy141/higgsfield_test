"""Database access: a synchronous psycopg3 connection pool over Postgres+pgvector.

We use the *synchronous* driver on purpose. FastAPI runs sync route handlers in a
threadpool, the embedding/rerank/LLM calls are themselves blocking, and the eval
explicitly says not to invest in async orchestration. Sync code keeps the
read-after-write story trivial to reason about: each request does its work inside
one connection/transaction and commits before returning.

Startup ordering matters: every pooled connection runs `register_vector()` in its
`configure` hook, which fails if the `vector` extension does not yet exist. So we
create the extension + schema over a *direct* connection (with a readiness retry)
*before* opening the pool.
"""

from __future__ import annotations

import logging
import pathlib
import threading
import time

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import get_settings

log = logging.getLogger("memory.db")

_SCHEMA_PATH = pathlib.Path(__file__).with_name("schema.sql")

_pool: ConnectionPool | None = None
_lock = threading.Lock()


def _configure(conn: psycopg.Connection) -> None:
    """Run once per pooled connection: enable pgvector adaptation."""
    register_vector(conn)


def _create_schema(database_url: str, *, attempts: int = 30, delay: float = 1.0) -> None:
    """Create the extension + schema over a direct connection, retrying until the
    database accepts connections (handles a DB that is still booting)."""
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    last: Exception | None = None
    for i in range(attempts):
        try:
            with psycopg.connect(database_url, autocommit=True, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
            log.info("schema ready (attempt %d)", i + 1)
            return
        except Exception as exc:  # noqa: BLE001 - DB may not be up yet
            last = exc
            if i == 0 or (i + 1) % 5 == 0:
                log.info("waiting for database... (%s)", exc)
            time.sleep(delay)
    raise RuntimeError(f"database not ready after {attempts} attempts: {last}")


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                settings = get_settings()
                pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=1,
                    max_size=10,
                    max_idle=60,
                    configure=_configure,
                    kwargs={"autocommit": False},
                    open=False,
                )
                pool.open()
                pool.wait(timeout=30)
                _pool = pool
                log.info("db pool opened")
    return _pool


def init_db() -> None:
    """Idempotent: ensure the schema exists, then open the pool. Safe to call on
    every startup."""
    settings = get_settings()
    _create_schema(settings.database_url)
    get_pool()


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def healthcheck() -> bool:
    """Cheap liveness probe against the DB."""
    try:
        pool = get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() is not None
    except Exception as exc:  # noqa: BLE001 - defensive
        log.warning("healthcheck failed: %s", exc)
        return False
