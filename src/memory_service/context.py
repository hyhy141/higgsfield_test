"""Context assembly under a token budget.

Turns reranked material into the prose `context` string injected into the agent's
prompt, plus citations. The ordering encodes the priority logic the eval asks us
to defend:

  1. Known facts about the user   — stable identity facts + query-relevant
     preferences. Cheap, dense, useful for almost any follow-up, and the place
     fact-evolution shows up ("...; previously Stripe").
  2. Relevant context from earlier — opinions/events and query-relevant turns
     from the user's *other* sessions.
  3. Recent conversation          — the current session's latest turns, for
     continuity.

Rationale: an agent that forgets *who the user is* fails worse than one that
forgets a passing detail, so durable identity wins the first tokens. Within a
tier we sort by reranker relevance, then confidence, then recency. When the
budget is tight, lower tiers are truncated or dropped entirely — never the facts.

Noise resistance: if nothing (memory or turn) clears the relevance threshold for
the query, we return empty context instead of dumping an irrelevant profile.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .taxonomy import is_identity_key
from .tokens import estimate_tokens


def _fmt_date(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    s = str(value)
    return s[:10]


def _fact_line(m: dict) -> str:
    value = m["value"].rstrip(".")
    prev = m.get("prev_value")
    if prev and prev.strip().lower() != m["value"].strip().lower():
        date = _fmt_date(m.get("observed_at") or m.get("updated_at"))
        date_part = f"updated {date}; " if date else ""
        line = f"- {value} ({date_part}previously {prev.rstrip('.')})"
    else:
        line = f"- {value}"
    return line


def _turn_snippet(text: str, max_chars: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def _priority(m: dict) -> tuple:
    ident = 1 if is_identity_key(m["key"]) else 0
    return (ident, round(m.get("score", 0.0), 4), float(m.get("confidence", 0.0)),
            _fmt_date(m.get("observed_at") or m.get("updated_at")))


def assemble(
    *, query: str, gathered: dict, max_tokens: int, threshold: float,
    cosine_threshold: float = 0.52,
) -> tuple[str, list[dict]]:
    memories: list[dict] = gathered.get("memories", [])
    turns: list[dict] = gathered.get("turns", [])
    recents: list[dict] = gathered.get("recent_turns", [])

    def is_relevant(x: dict) -> bool:
        # Either calibrated signal clearing its bar marks an item relevant.
        return x.get("score", 0.0) >= threshold or x.get("cosine", 0.0) >= cosine_threshold

    has_query = bool(query.strip())
    rel_mems = [m for m in memories if is_relevant(m)]
    rel_turns = [t for t in turns if is_relevant(t)]

    # Noise-resistance gate: nothing relevant -> empty context.
    if has_query and not rel_mems and not rel_turns:
        return "", []

    # ── Tier 1: known facts ──────────────────────────────────────────────────
    fact_pool = [m for m in memories if m["type"] in ("fact", "preference")]
    chosen: list[dict] = []
    seen_vals: set[str] = set()
    for m in fact_pool:
        relevant = is_relevant(m)
        identity = is_identity_key(m["key"])
        # Include a fact if it's query-relevant, or (once we're past the gate)
        # it's stable identity grounding. With no query, only identity facts.
        if has_query:
            include = relevant or identity
        else:
            include = identity
        if not include:
            continue
        v = m["value"].strip().lower()
        if v in seen_vals:
            continue
        seen_vals.add(v)
        chosen.append(m)
    chosen.sort(key=_priority, reverse=True)

    # ── Tier 2: opinions/events + other-session relevant turns ───────────────
    earlier_mems = [m for m in rel_mems if m["type"] in ("opinion", "event")]
    earlier_mems.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    other_turns = [t for t in rel_turns]
    other_turns.sort(key=lambda t: t.get("score", 0.0), reverse=True)

    # ── Tier 3: recent current-session turns ─────────────────────────────────
    recent_ids = {str(t["id"]) for t in recents}

    # ── Emit under budget ────────────────────────────────────────────────────
    out_lines: list[str] = []
    citations: list[dict] = []
    used = 0
    cited_turns: set[str] = set()

    def remaining() -> int:
        return max_tokens - used

    def add_block(title: str, items: list[tuple[str, dict | None]]) -> None:
        nonlocal used
        if not items or remaining() <= 0:
            return
        header = f"## {title}"
        header_cost = estimate_tokens(header) + 1
        if remaining() <= header_cost + 2:
            return
        staged: list[str] = []
        staged_citations: list[dict] = []
        cost = header_cost
        for line, cite in items:
            lc = estimate_tokens(line) + 1
            if cost + lc > remaining():
                break
            staged.append(line)
            cost += lc
            if cite:
                staged_citations.append(cite)
        if not staged:
            return
        if out_lines:
            out_lines.append("")
        out_lines.append(header)
        out_lines.extend(staged)
        citations.extend(staged_citations)
        used += cost

    # Tier 1
    fact_items: list[tuple[str, dict | None]] = []
    for m in chosen:
        cite = None
        if m.get("source_turn"):
            cite = {"turn_id": str(m["source_turn"]), "score": round(float(m.get("score", 0.0)), 4),
                    "snippet": m["value"]}
        fact_items.append((_fact_line(m), cite))
    add_block("Known facts about this user", fact_items)

    # Tier 2
    earlier_items: list[tuple[str, dict | None]] = []
    for m in earlier_mems:
        date = _fmt_date(m.get("observed_at") or m.get("updated_at"))
        prefix = f"[{date}] " if date else ""
        cite = None
        if m.get("source_turn"):
            cite = {"turn_id": str(m["source_turn"]), "score": round(float(m.get("score", 0.0)), 4),
                    "snippet": m["value"]}
        earlier_items.append((f"- {prefix}{m['value'].rstrip('.')}", cite))
    for t in other_turns:
        tid = str(t["id"])
        if tid in recent_ids:
            continue  # shown under recent conversation instead
        date = _fmt_date(t.get("ts"))
        prefix = f"[{date}] " if date else ""
        snip = _turn_snippet(t["text_repr"])
        cited_turns.add(tid)
        earlier_items.append(
            (f"- {prefix}{snip}",
             {"turn_id": tid, "score": round(float(t.get("score", 0.0)), 4), "snippet": snip})
        )
    add_block("Relevant context from earlier conversations", earlier_items)

    # Tier 3
    recent_items: list[tuple[str, dict | None]] = []
    for t in recents:
        tid = str(t["id"])
        if tid in cited_turns:
            continue
        date = _fmt_date(t.get("ts"))
        prefix = f"[{date}] " if date else ""
        snip = _turn_snippet(t["text_repr"])
        recent_items.append(
            (f"- {prefix}{snip}",
             {"turn_id": tid, "score": 0.0, "snippet": snip})
        )
    add_block("Recent conversation", recent_items)

    return "\n".join(out_lines).strip(), citations
