"""Extraction: turn raw conversation messages into structured memory candidates.

Two engines behind one interface:

  * LLM engine (primary)  — a single, schema-constrained call asks the model to
    emit typed (key, value) memories using a controlled key namespace, including
    implicit facts and corrections. This is what runs in the eval (an API key is
    provided).
  * Rule engine (fallback) — deterministic regex/heuristics over user messages.
    It runs when no LLM key is configured or the LLM call fails, so /turns always
    produces *structured* memories (never raw chunks) and never crashes.

Both engines emit the same `Candidate` shape; `store.py` applies supersession.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import Settings
from .llm import LLMError, complete_json, provider_available
from .taxonomy import default_cardinality, normalize_key, normalize_type

log = logging.getLogger("memory.extraction")


@dataclass
class Candidate:
    type: str
    key: str
    value: str
    confidence: float = 0.6
    cardinality: str = "single"
    subject: str = "user"
    entity: str | None = None
    polarity: str = "positive"  # positive | negative
    operation: str = "assert"  # assert | retract
    is_correction: bool = False
    temporal: str = "current"
    attrs: dict = field(default_factory=dict)

    def normalized(self) -> Candidate:
        self.type = normalize_type(self.type)
        self.key = normalize_key(self.key)
        if self.cardinality not in ("single", "multi"):
            self.cardinality = default_cardinality(self.key)
        self.value = (self.value or "").strip()
        self.subject = (self.subject or "user").strip().lower() or "user"
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.6
        return self


# ── Public API ───────────────────────────────────────────────────────────────
def extract(messages: list[dict], timestamp: str | None, settings: Settings) -> tuple[list[Candidate], str]:
    """Return (candidates, method). Never raises on extraction problems."""
    method = "rules"
    candidates: list[Candidate] = []
    if settings.resolved_provider in ("anthropic", "openai") and provider_available(settings):
        try:
            candidates = _llm_extract(messages, timestamp, settings)
            method = "llm"
        except LLMError as exc:
            log.warning("LLM extraction failed, falling back to rules: %s", exc)
            candidates = []
            method = "rules(llm-failed)"
        except Exception as exc:  # noqa: BLE001 - never let extraction crash /turns
            log.warning("LLM extraction error, falling back to rules: %s", exc)
            candidates = []
            method = "rules(llm-error)"
    if not candidates:
        candidates = _rule_extract(messages)
        if method == "rules":
            method = "rules"
        else:
            method = method + "+rules"
    # Normalize + drop empties.
    out = []
    for c in candidates:
        c.normalized()
        if c.value:
            out.append(c)
    out = _resolve_conflicts(out)
    return out, method


def _resolve_conflicts(cands: list[Candidate]) -> list[Candidate]:
    """Within a single turn, a retract for a single-valued key is redundant (and
    order-dependent harmful) when the same key is also asserted — the assert
    already supersedes the old value. Drop those retracts."""
    asserted_single = {
        c.key for c in cands if c.operation == "assert" and c.cardinality == "single"
    }
    return [c for c in cands if not (c.operation == "retract" and c.key in asserted_single)]


# ── LLM engine ───────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the extraction engine of a long-term memory service for a personal AI \
assistant. Read one conversation turn (one or more messages) and extract durable, \
structured knowledge about the USER and the entities they mention. You output ONLY \
a JSON object — no prose.

Return: {"memories": [ <memory>, ... ]}  (empty list if nothing durable).

Each <memory> has:
  type        one of: fact | preference | opinion | event
  key         a canonical, dot-separated, snake_case topic id. The SAME real-world
              attribute MUST always get the SAME key, so later updates can be
              detected. Use this namespace (extend it consistently when needed):
                identity/profile: name.full, name.first, age, gender, nationality
                work:   employment.company, employment.role, employment.status,
                        employment.industry, occupation
                place:  location.city, location.country, location.region,
                        location.hometown, residence.type
                people: family.spouse, family.partner, family.child,
                        family.sibling, family.parent, relationship.status
                pets:   pet.dog, pet.cat, pet.<species>
                diet:   diet.restriction, diet.allergy
                prefs:  preference.<topic>            (e.g. preference.communication_style)
                views:  opinion.<topic>               (e.g. opinion.typescript)
                skills: skill.<tech>, language.spoken
                misc:   goal.<topic>, project.<name>, event.<slug>
  value       a concise, self-contained statement a human can read, e.g.
              "Works at Notion as a Product Manager", "Lives in Berlin",
              "Has a dog named Biscuit", "Allergic to shellfish",
              "Prefers concise, direct answers".
  subject     "user" for the user; otherwise the person/entity name.
  entity      for set-valued keys (pets, allergies, skills, children, opinions),
              the distinguishing item (e.g. "Biscuit", "shellfish", "typescript");
              else null.
  cardinality "single" if there is one current value (new replaces old: job,
              city, marital status, a stance on a topic) or "multi" if values
              coexist (pets, allergies, skills, children, hobbies).
  polarity    "positive" normally; "negative" for negations/removals
              ("no longer has a dog", "not allergic to nuts anymore").
  operation   "assert" to state/update; "retract" to remove a prior fact.
  is_correction  true if this fixes something stated earlier ("actually...",
              "sorry, not X — Y", "I misspoke").
  temporal    "current" | "past" | "future" | "ongoing".
  confidence  0..1 — how sure you are this is a real, durable user fact.

Rules:
  - Extract IMPLICIT facts: "walking Biscuit this morning" -> pet.dog,
    value "Has a dog named Biscuit", entity "Biscuit".
  - A move like "moved to Berlin from NYC" yields BOTH location.city = Berlin
    (cardinality single) AND an event.move "Moved from NYC to Berlin".
  - Capture corrections and updates as new memories with is_correction set and
    the corrected value; do not try to delete anything yourself.
  - Do NOT extract the assistant's statements, generic chit-chat, or transient
    task state. Only durable knowledge about the user and their world.
  - Prefer fewer, higher-quality memories over many noisy ones.
"""


def _render_messages(messages: list[dict], timestamp: str | None) -> str:
    lines = []
    if timestamp:
        lines.append(f"[turn timestamp: {timestamp}]")
    for m in messages:
        role = (m.get("role") or "user").strip()
        name = m.get("name")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        tag = role if not name else f"{role}:{name}"
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)


def _llm_extract(messages: list[dict], timestamp: str | None, settings: Settings) -> list[Candidate]:
    rendered = _render_messages(messages, timestamp)
    if not rendered.strip():
        return []
    obj = complete_json(settings, _SYSTEM_PROMPT, rendered)
    raw = obj.get("memories", [])
    if not isinstance(raw, list):
        raise LLMError("'memories' was not a list")
    out: list[Candidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", "")).strip()
        key = str(item.get("key", "")).strip()
        if not value or not key:
            continue
        out.append(
            Candidate(
                type=str(item.get("type", "fact")),
                key=key,
                value=value,
                confidence=item.get("confidence", 0.7),
                cardinality=str(item.get("cardinality", "") or default_cardinality(key)),
                subject=str(item.get("subject", "user") or "user"),
                entity=(str(item["entity"]).strip() if item.get("entity") else None),
                polarity=str(item.get("polarity", "positive") or "positive"),
                operation=str(item.get("operation", "assert") or "assert"),
                is_correction=bool(item.get("is_correction", False)),
                temporal=str(item.get("temporal", "current") or "current"),
                attrs={k: item.get(k) for k in ("temporal", "entity", "polarity") if item.get(k)},
            )
        )
    return out


# ── Rule engine ──────────────────────────────────────────────────────────────
# Conservative, high-precision patterns over the user's own messages. Tuned
# against fixtures/ ; the LLM engine handles the long tail.

# Words that terminate a proper-noun phrase: when we capture "the rest of the
# clause" after a trigger ("...moved to <REST>"), we keep words up to the first
# of these. "of"/"the" are intentionally NOT here so "Bank of America" survives.
_BOUNDARY = {
    "as", "and", "or", "but", "since", "now", "last", "this", "because", "where",
    "which", "who", "after", "before", "currently", "recently", "though",
    "however", "while", "so", "today", "yesterday", "tomorrow", "ago", "when",
    "in", "on", "at", "for", "from", "to", "with", "by", "that", "right", "back",
    "again", "still", "also", "really", "very", "just", "anymore", "any", "no",
}


# Common "places" so "I'm at home/the gym" is not misread as an employer.
_NON_EMPLOYERS = {
    "home", "work", "the office", "office", "the gym", "gym", "school", "lunch",
    "dinner", "the airport", "airport", "a conference", "the conference", "a meeting",
    "the hospital", "the beach", "the park", "the moment", "it", "least", "peace",
}


def _clean_proper(s: str) -> str:
    """Keep the leading proper-noun phrase: strip surrounding punctuation, then
    take words up to the first boundary word (max 5)."""
    s = (s or "").strip().strip(".,!?;:\"'()")
    kept: list[str] = []
    for w in re.split(r"\s+", s):
        wl = w.lower().strip(".,!?;:\"'()")
        if not wl or wl in _BOUNDARY:
            break
        kept.append(w.strip(".,!?;:\"'()"))
        if len(kept) >= 5:
            break
    return " ".join(kept).strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _user_text(messages: list[dict]) -> str:
    return "\n".join(
        (m.get("content") or "") for m in messages if (m.get("role") or "") == "user"
    )


def _rule_extract(messages: list[dict]) -> list[Candidate]:
    text = _user_text(messages)
    if not text.strip():
        return []
    out: list[Candidate] = []
    for sent in _sentences(text):
        out.extend(_rules_for_sentence(sent))
    return _dedup(out)


_I = re.I


def _after(sent: str, pattern: str) -> str | None:
    """Return the cleaned proper-noun phrase captured by `pattern` (whose single
    group grabs the rest of the clause), or None."""
    m = re.search(pattern, sent, _I)
    if not m:
        return None
    return _clean_proper(m.group(1)) or None


def _rules_for_sentence(sent: str) -> list[Candidate]:
    out: list[Candidate] = []
    correction = bool(re.search(r"\b(actually|sorry,? not|i meant|to clarify|i misspoke|correction)\b", sent, _I))

    # ── Employment ──
    m = re.search(r"\bi(?:['’]m| am) (?:a|an) ([\w ]+?) at (.+)", sent, _I)
    if m:
        role = _clean_proper(m.group(1))
        company = _clean_proper(m.group(2))
        if company:
            out.append(Candidate("fact", "employment.company", f"Works at {company}", 0.85))
        if role:
            out.append(Candidate("fact", "employment.role", f"Works as a {role}", 0.8))
    else:
        company = _after(
            sent,
            r"\bi (?:just |recently |now |currently )?(?:work|started working|started|got a job|joined|"
            r"am working|['’]m working|began)(?: a job)?(?: at| for| with)\s+(.+)",
        )
        if not company:
            # Compound sentence ("...and work at Notion") — no "I" before "work".
            c2 = _after(sent, r"\bwork(?:ing)? (?:at|for) (.+)")
            if c2 and c2.lower() not in _NON_EMPLOYERS:
                company = c2
        if company:
            out.append(Candidate("fact", "employment.company", f"Works at {company}", 0.82))
        else:
            # "I'm (now) at <Company>" — guarded against places (home, gym, ...).
            m_at = re.search(r"\bi(?:['’]m| am)(?: now| currently)? (?:working |back )?at (.+)", sent, _I)
            if m_at:
                cand = _clean_proper(m_at.group(1))
                if cand and cand.lower() not in _NON_EMPLOYERS:
                    out.append(Candidate("fact", "employment.company", f"Works at {cand}", 0.7))
        role = _after(sent, r"\bwork(?:ing)? as (?:a |an )?(.+)")
        if not role and company:  # "...work at X as a <role>"
            role = _after(sent, r"\bas (?:a|an) (.+)")
        if role:
            out.append(Candidate("fact", "employment.role", f"Works as a {role}", 0.78))

    # Retraction: "I don't work at X anymore" / "I no longer work at X"
    m = re.search(r"\bi (?:don['’]?t|do not|no longer) work (?:at|for) (.+)", sent, _I)
    if m:
        out.append(Candidate("fact", "employment.company", f"No longer at {_clean_proper(m.group(1))}",
                             0.7, operation="retract", polarity="negative"))

    # ── Location: move (split off the origin in Python) ──
    m = re.search(r"\bi (?:just |recently )?(?:moved|relocated) to (.+)", sent, _I)
    if m:
        rest = m.group(1)
        prev = None
        low_rest = rest.lower()
        if " from " in low_rest:
            idx = low_rest.index(" from ")
            city = _clean_proper(rest[:idx])
            prev = _clean_proper(rest[idx + 6:])
        else:
            city = _clean_proper(rest)
        if city:
            attrs = {"previous": prev} if prev else {}
            out.append(Candidate("fact", "location.city", f"Lives in {city}", 0.85, attrs=attrs))
            if prev:
                out.append(Candidate("event", "event.move", f"Moved from {prev} to {city}", 0.8,
                                     cardinality="multi", entity=city))
    else:
        city = _after(sent, r"\bi (?:live|am living|['’]m living|reside|am based|['’]m based) (?:in|at) (.+)")
        if city:
            out.append(Candidate("fact", "location.city", f"Lives in {city}", 0.82))
        home = _after(sent, r"\bi(?:['’]m| am) from (.+)")
        if home:
            out.append(Candidate("fact", "location.hometown", f"Is from {home}", 0.75))

    # ── Pets ──
    for m in re.finditer(r"\bmy (dog|cat|puppy|kitten|hamster|parrot|bird|rabbit|horse|pet)\b"
                         r"(?:[^.\n]*?\b(?:named|called)\s+(\w+)|\s+(\w+))?", sent, _I):
        species = m.group(1).lower()
        name = m.group(2) or m.group(3)
        if name and name.lower() in ("is", "and", "the", "was", "who", "that", "loves", "likes"):
            name = None
        species = {"puppy": "dog", "kitten": "cat"}.get(species, species)
        if name:
            out.append(Candidate("fact", f"pet.{species}", f"Has a {species} named {name}", 0.85,
                                  cardinality="multi", entity=name))
        else:
            out.append(Candidate("fact", f"pet.{species}", f"Has a {species}", 0.7,
                                  cardinality="multi", entity=species))
    m = re.search(r"\bi (?:have|have got|got|adopted) (?:a|an|another|two|three) (dog|cat|puppy|kitten)"
                  r"(?:[^.\n]*?\b(?:named|called)\s+(\w+))?", sent, _I)
    if m:
        species = {"puppy": "dog", "kitten": "cat"}.get(m.group(1).lower(), m.group(1).lower())
        name = m.group(2)
        val = f"Has a {species} named {name}" if name else f"Has a {species}"
        out.append(Candidate("fact", f"pet.{species}", val, 0.82, cardinality="multi",
                             entity=name or species))

    # ── Diet ──
    m = re.search(r"\bi(?:['’]m| am)?\s*(?:a\s+)?(vegetarian|vegan|pescatarian|pescetarian)\b", sent, _I)
    if m:
        word = m.group(1).lower()
        out.append(Candidate("preference", "diet.restriction", f"Is {word}", 0.85,
                             cardinality="multi", entity=word))
    m = re.search(r"\ballergic to ([\w ,]+?)(?:[.!?]|$)", sent, _I)
    if m:
        for raw in re.split(r",|\band\b", m.group(1), flags=_I):
            allergen = _clean_proper(raw)  # cut clause tails like "so I keep it simple"
            if allergen and len(allergen.split()) <= 3:
                out.append(Candidate("fact", "diet.allergy", f"Allergic to {allergen}", 0.85,
                                     cardinality="multi", entity=allergen.lower()))

    # ── Family ──
    m = re.search(r"\bmy (wife|husband|partner|spouse|girlfriend|boyfriend)\b"
                  r"(?:[^.\n]*?\b(?:named|called|is)\s+([A-Z][a-z]+))?", sent, _I)
    if m:
        rel = m.group(1).lower()
        name = m.group(2)
        val = f"Has a {rel}" + (f" named {name}" if name else "")
        key = "family.partner" if rel in ("partner", "girlfriend", "boyfriend") else "family.spouse"
        out.append(Candidate("fact", key, val, 0.8, entity=name))
    for m in re.finditer(r"\bmy (son|daughter)\b(?:[^.\n]*?\b(?:named|called)\s+([A-Z][a-z]+))?", sent, _I):
        name = m.group(2)
        val = f"Has a {m.group(1).lower()}" + (f" named {name}" if name else "")
        out.append(Candidate("fact", "family.child", val, 0.78, cardinality="multi", entity=name))

    # ── Communication style ──
    m = re.search(r"\bi (?:prefer|like|want|appreciate) (?:my )?(?:answers|responses|replies|"
                  r"explanations|communication)\s*(?:to be\s+)?([\w, ]+?)(?:[.!?]|$)", sent, _I)
    if not m:
        m = re.search(r"\bi (?:prefer|like) ([\w, ]+?) (?:answers|responses|replies|explanations)", sent, _I)
    if m:
        style = m.group(1).strip().strip(".,")
        if style and len(style) <= 50:
            out.append(Candidate("preference", "preference.communication_style",
                                 f"Prefers {style} answers", 0.75))

    # ── Preferences / sentiment (multiple per sentence) ──
    for m in re.finditer(r"\bi (love|really like|like|enjoy|adore|prefer|hate|dislike|can['’]?t stand|loathe) "
                         r"([\w +#.\-]*?)(?:[.,!?]|$| over | because | but | for )", sent, _I):
        verb = m.group(1).lower().replace("really ", "")
        obj = m.group(2).strip().strip(".,!?")
        if verb == "prefer" and re.search(r"answers|responses|replies|explanations|communication", sent, _I):
            continue  # already captured as communication_style
        if obj and len(obj) <= 40 and obj.lower() not in ("it", "that", "this", "them", "you"):
            slug = normalize_key(obj)
            neg = verb.replace("’", "'") in ("hate", "dislike", "can't stand", "cant stand", "loathe")
            verb_disp = {"can't stand": "Dislikes", "cant stand": "Dislikes", "loathe": "Hates",
                         "hate": "Hates", "dislike": "Dislikes", "prefer": "Prefers",
                         "love": "Loves", "adore": "Loves", "like": "Likes", "enjoy": "Enjoys"}.get(verb, "Likes")
            out.append(Candidate("preference", f"preference.{slug}", f"{verb_disp} {obj}", 0.7,
                                 polarity="negative" if neg else "positive"))

    # ── Name ──
    m = re.search(r"\bmy name is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", sent, _I)
    if m:
        out.append(Candidate("fact", "name.full", f"Name is {m.group(1)}", 0.9))

    if correction:
        for c in out:
            c.is_correction = True
    return out


def _dedup(cands: list[Candidate]) -> list[Candidate]:
    seen: dict[tuple, Candidate] = {}
    for c in cands:
        sig = (c.subject, c.key, (c.entity or "").lower(), c.value.lower())
        if sig not in seen:
            seen[sig] = c
    return list(seen.values())
