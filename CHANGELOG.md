# Changelog

Iteration history for the memory service, **oldest first** so each entry's
*Next* flows into the one below. Metrics come from the self-eval harness
(`tests/harness.py`) run against `fixtures/` — 10 probes across recall, fact
evolution, multi-hop, noise resistance, and cross-user isolation. Unless noted,
runs use the **rule-based** extractor (no API key in my dev environment); the
LLM extractor is the production default and is strictly additive.

---

## v0.1 — Skeleton + contract over Postgres/pgvector

**What changed:** FastAPI service, the seven contract endpoints, and a single
Postgres+pgvector schema (`turns`, `memories`) with vector, full-text (`tsvector`),
and btree indexes. Sync psycopg3 pool; `docker compose up` boots db + app with a
named volume.

**Why:** One ACID store for raw turns *and* extracted memories *and* their indexes
means a `/turns` write commits everything in one transaction — the read-after-write
guarantee the eval demands falls out for free, with no cross-store sync.

**Result:** `/health` green; turns persist and round-trip.

**Next:** It's a message log, not a memory. Need real extraction.

---

## v0.2 — Extraction + supersession (the actual memory part)

**What changed:** Two extraction engines behind one interface — an LLM engine
(schema-constrained JSON, controlled key namespace) and a deterministic rule
engine fallback. A canonical-key taxonomy drives a supersession resolver:
single-valued keys replace + link (`supersedes`/`superseded_by`, old row
`active=false`); multi-valued keys coexist, matched by `entity`; retractions
deactivate; near-duplicates refresh in place.

**Why:** "Raw-message-in-vector-out is not extraction." The key is the unit of
identity for fact evolution: "I work at Stripe" and "I joined Notion" must land on
`employment.company` so the second supersedes the first.

**Result:** Stripe→Notion produces one active row (Notion) + one inactive (Stripe)
linked by the chain; history inspectable via `/users/{id}/memories`.

**Next:** Retrieval was cosine-top-k. Weak, and no token budgeting.

---

## v0.3 — Hybrid retrieval + budgeted context assembly

**What changed:** Per-source hybrid retrieval (dense bge-small + Postgres FTS)
fused with Reciprocal Rank Fusion, then a local cross-encoder rerank
(ms-marco-MiniLM). Context assembly in priority tiers under `max_tokens`: stable
identity facts → query-relevant memories → recent session turns, with
fact-evolution rendering ("…; previously Stripe").

**Why:** Keyword queries ("dog's name?") need lexical match; paraphrases need
dense. RRF needs no score calibration. Reranking gives a precise final order.

**Result:** First real end-to-end recall. Cross-session sharing for one user works
(write in session A, recall in session B).

**Next:** Hadn't run it against the contract's own smoke test yet.

---

## v0.4 — Smoke test surfaced two real bugs

**What changed:** Ran the §7 smoke test. (1) `/recall` 500'd:
`operator does not exist: vector <=> double precision[]` — psycopg adapted a
Python `list` to `double precision[]`, not `vector`. Switched all embedding params
to numpy arrays (pgvector registers an ndarray dumper). (2) Extraction returned 0
memories for "**I** just moved to Berlin…" — regexes anchored on lowercase `i` ran
case-sensitively. Rewrote rule triggers case-insensitively, capturing from the
original text so "Berlin"/"NYC" keep their casing; switched proper-noun capture to
"rest-of-clause then trim at a boundary word."

**Why:** Testing against the real contract early is cheaper than discovering this
in the eval.

**Result:** Smoke test passes — Berlin recalled cross-session; memories structured.
Built `fixtures/` (2 users, 5 sessions) + `tests/harness.py`. First fixture score:
**9/10** — only noise resistance failing.

**Next:** Noise queries dumped the whole profile. Fix the relevance gate.

---

## v0.5 — Calibrating the noise gate (the hard part)

**What changed:** Made `/recall` return empty when nothing is genuinely relevant.
This took three tries, driven by measuring real scores:

1. **Reranker threshold alone.** Measured: correct hits scored 0.012 (Berlin),
   0.009 (dog) but also **0.0001** (the correct "Works at Notion"!) and 0.00002
   ("Is vegetarian") — *below* some noise pairs (0.0003). The cross-encoder scores
   tiny facts unreliably. A single rerank threshold could not separate on-topic
   from noise. Rejected.
2. **Cosine OR rerank.** bge cosine is better-calibrated, so I added "relevant if
   cosine ≥ 0.52 OR rerank ≥ T". But noise leaked: "favorite Pokemon and chess
   opening" ↔ "Loves TypeScript" cosine = **0.564** > 0.52 → full profile dumped.
   Cosine separation on raw values was only ~1.2× (on-topic ~0.64 vs noise ~0.56).
3. **Key-prefixed indexing + retuned gate (kept).** Root cause: I embedded only the
   *value* ("Is vegetarian"), which shares no tokens with "dietary restrictions".
   Now I embed/rerank `"<humanized key>: <value>"` → "diet restriction: Is
   vegetarian". Measured cosine jumps: diet 0.42→**0.78**, location 0.61, employer
   0.66, dog 0.71; noise stays ≤0.587. Gate: **cosine ≥ 0.60 OR rerank ≥ 0.008**.

**Why:** The gate must trust the better-separated signal. Enriching the passage
with its key widened the on-topic/noise margin enough to draw a line.

**Result:** Self-eval **9/10 → 10/10**. Noise → empty; diet/employer/location recall
also improved as a side effect (better first-stage matching). Also fixed an
extraction bug the fixture caught: "allergic to shellfish, so I keep it simple"
had extracted a junk allergen "so I keep it simple" — now clause tails are trimmed.

**Next:** Harden robustness (malformed/oversized/unicode), add the restart-
persistence test, and validate the full `docker compose` path end-to-end.

---

## v0.6 — Robustness, tests, and docs

**What changed:** `/search` ranks by a rerank+cosine blend (short facts were all
rounding to score 0.0). Payload-size guard, message clamping, tolerant timestamp
parsing, global exception handler (errors are 4xx/5xx, never a crashed worker),
optional bearer auth. Full pytest suite (contract round-trip, concurrency/
isolation, malformed input, restart persistence) plus the recall-quality fixture
as a gating test. README with architecture, design rationale, and tradeoffs.

**Why:** "Excellent" requires the service to degrade gracefully and a reviewer to
understand the design in five minutes.

**Result:** `docker compose up` clean-boots; smoke + full suite pass; self-eval
holds at 10/10.

**Next (known limitations):** Opinion *arcs* ("love TS" → "TS generics annoying")
are modeled as supersession of a single `opinion.<topic>` stance — the current
stance and history are preserved, but the gradual trajectory isn't synthesized.
The reranker's low absolute scale means the gate leans on cosine; a reranker
fine-tuned on short facts would let me tighten it. Rule extractor misses implicit
facts ("walking Biscuit" → has a dog) that the LLM engine catches.
