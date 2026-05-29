-- Memory service schema. Idempotent: safe to run on every startup.
-- One Postgres database holds everything — raw turns, extracted memories, their
-- vector embeddings and full-text indexes — so a single transaction makes a
-- write atomically visible to every read path (no cross-store consistency gap).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ── Raw conversation turns ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS turns (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    TEXT NOT NULL,
    user_id       TEXT,
    messages      JSONB NOT NULL DEFAULT '[]'::jsonb,
    text_repr     TEXT NOT NULL DEFAULT '',
    embedding     vector(384),
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text_repr, ''))) STORED,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS turns_session_idx ON turns (session_id);
CREATE INDEX IF NOT EXISTS turns_user_idx    ON turns (user_id);
CREATE INDEX IF NOT EXISTS turns_ts_idx      ON turns (ts DESC);
CREATE INDEX IF NOT EXISTS turns_tsv_idx     ON turns USING GIN (tsv);
CREATE INDEX IF NOT EXISTS turns_embedding_idx
    ON turns USING hnsw (embedding vector_cosine_ops);

-- ── Extracted, structured memories ───────────────────────────────────────────
-- A memory is a typed (key, value) assertion about a subject (usually the user).
-- Fact evolution is modelled as a supersession chain: a contradicting/updated
-- value flips the old row's `active` to false and links the two rows, so the
-- current truth is queryable while full history is preserved.
CREATE TABLE IF NOT EXISTS memories (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       TEXT,
    session_id    TEXT,
    subject       TEXT NOT NULL DEFAULT 'user',
    type          TEXT NOT NULL DEFAULT 'fact',     -- fact | preference | opinion | event
    key           TEXT NOT NULL,                    -- canonical dotted key
    value         TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 0.6,
    cardinality   TEXT NOT NULL DEFAULT 'single',   -- single | multi
    attrs         JSONB NOT NULL DEFAULT '{}'::jsonb, -- entity, polarity, temporal, ...
    source_session TEXT,
    source_turn   UUID,
    embedding     vector(384),
    tsv           tsvector GENERATED ALWAYS AS (
                      to_tsvector('english', coalesce(key, '') || ' ' || coalesce(value, ''))
                  ) STORED,
    active        BOOLEAN NOT NULL DEFAULT true,
    supersedes    UUID,    -- the memory this row replaced
    superseded_by UUID,    -- set on the old row when it is replaced
    observed_at   TIMESTAMPTZ,  -- when the user stated the fact (turn timestamp)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),  -- when the row was written
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memories_user_active_idx  ON memories (user_id, active);
CREATE INDEX IF NOT EXISTS memories_user_key_idx     ON memories (user_id, key, subject, active);
CREATE INDEX IF NOT EXISTS memories_session_idx      ON memories (session_id);
CREATE INDEX IF NOT EXISTS memories_source_turn_idx  ON memories (source_turn);
CREATE INDEX IF NOT EXISTS memories_tsv_idx          ON memories USING GIN (tsv);
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING hnsw (embedding vector_cosine_ops);
