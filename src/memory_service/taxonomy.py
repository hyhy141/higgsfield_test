"""Canonical key taxonomy.

The `key` on a memory is the linchpin of fact evolution: two statements about the
same real-world attribute must land on the SAME key so the store can detect that
a new value supersedes an old one. We give the extractor a controlled namespace
(dot-separated, snake_case) and use prefix rules here to decide:

  * cardinality  — is this attribute single-valued (one current truth, new
                   replaces old) or a set (values coexist)?
  * identity     — is this a stable "who is this user" fact that should be
                   surfaced in recall as grounding regardless of the exact query?

Keeping these rules in code (not just the prompt) means the deterministic
rule-based extractor and the supersession logic share one source of truth, and
the LLM's `cardinality` hint is a fallback, not the only signal.
"""

from __future__ import annotations

# Key prefixes whose values are inherently a SET — new values coexist with old
# ones rather than replacing them (a user can have several pets, allergies,
# skills, children, ...). Everything else defaults to single-valued.
MULTI_VALUED_PREFIXES: tuple[str, ...] = (
    "pet",
    "family.child",
    "family.children",
    "family.sibling",
    "diet.restriction",
    "diet.allergy",
    "allergy",
    "skill",
    "language",
    "hobby",
    "interest",
    "like",
    "dislike",
    "favorite",
    "goal",
    "project",
    "tool",
    "device",
    "subscription",
    "event",
)

# Stable "identity" facts that ground almost any personal query. These get a
# priority boost in context assembly and are eligible for inclusion as profile
# grounding when a recall is on-topic.
IDENTITY_PREFIXES: tuple[str, ...] = (
    "name",
    "age",
    "employment",
    "job",
    "role",
    "company",
    "occupation",
    "location",
    "residence",
    "address",
    "nationality",
    "family",
    "pet",
    "diet",
    "allergy",
    "relationship",
    "marital",
    "language",
)

VALID_TYPES: frozenset[str] = frozenset({"fact", "preference", "opinion", "event"})


def _matches(key: str, prefixes: tuple[str, ...]) -> bool:
    key = (key or "").strip().lower()
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def default_cardinality(key: str) -> str:
    """'multi' if the key names a set-valued attribute, else 'single'."""
    return "multi" if _matches(key, MULTI_VALUED_PREFIXES) else "single"


def is_identity_key(key: str) -> bool:
    return _matches(key, IDENTITY_PREFIXES)


def humanize_key(key: str) -> str:
    """'diet.restriction' -> 'diet restriction' (for embedding/search text)."""
    return (key or "").replace(".", " ").replace("_", " ").strip()


def memory_search_text(key: str, value: str) -> str:
    """The text we embed, full-text index, and rerank a memory by.

    Prefixing the humanized key turns a terse value ("Is vegetarian") into a
    self-describing passage ("diet restriction: Is vegetarian"). This sharply
    improves asymmetric matching for category-style questions ("what dietary
    restrictions...?") where the value alone shares no tokens with the query.
    """
    hk = humanize_key(key)
    return f"{hk}: {value}" if hk else value


def normalize_type(t: str | None) -> str:
    t = (t or "").strip().lower()
    return t if t in VALID_TYPES else "fact"


def normalize_key(key: str | None) -> str:
    """Coerce a free-form key into the canonical shape: lowercase, snake_case,
    dot-separated, no stray punctuation."""
    if not key:
        return "misc.note"
    key = key.strip().lower()
    out = []
    for ch in key:
        if ch.isalnum() or ch in "._":
            out.append(ch)
        elif ch in " -/":
            out.append("_")
        # drop everything else
    cleaned = "".join(out).strip("._")
    # collapse repeats
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    return cleaned or "misc.note"
