"""In-process unit tests for the extraction logic — no running service needed.

Crucially this is where the LLM extraction *path* (used by the eval) is covered:
we mock the model call and assert the JSON is parsed into the right Candidates,
and that any LLM failure falls back to the rule engine.
"""

from __future__ import annotations

from memory_service import extraction
from memory_service.config import Settings
from memory_service.extraction import Candidate, _resolve_conflicts


def test_rule_extraction_core_facts():
    msgs = [{"role": "user", "content": (
        "I moved to Berlin from NYC. I work at Stripe as an engineer. "
        "My dog Biscuit is great. I'm vegetarian and allergic to shellfish."
    )}]
    cands, method = extraction.extract(msgs, "2025-01-01T00:00:00Z",
                                       Settings(extraction_provider="rules"))
    keys = {c.key for c in cands}
    assert method == "rules"
    assert "location.city" in keys
    assert "employment.company" in keys
    assert any(k.startswith("pet") for k in keys)
    assert "diet.restriction" in keys
    assert "diet.allergy" in keys
    by_key = {c.key: c.value for c in cands}
    assert "Berlin" in by_key["location.city"]
    assert "Stripe" in by_key["employment.company"]


def test_conflict_resolution_drops_retract_when_asserted():
    cands = [
        Candidate("fact", "employment.company", "Works at Google", cardinality="single",
                  operation="assert"),
        Candidate("fact", "employment.company", "No longer at Stripe", cardinality="single",
                  operation="retract"),
    ]
    out = _resolve_conflicts(cands)
    assert len(out) == 1 and out[0].operation == "assert"


def test_multi_valued_retract_is_kept():
    # A retraction on a multi-valued key (e.g. removing one pet) is NOT dropped.
    cands = [
        Candidate("fact", "pet.dog", "Has a dog named Rex", cardinality="multi", operation="assert"),
        Candidate("fact", "pet.cat", "No longer has a cat", cardinality="multi", operation="retract"),
    ]
    out = _resolve_conflicts(cands)
    assert len(out) == 2


def test_llm_extraction_parsing(monkeypatch):
    settings = Settings(extraction_provider="anthropic", anthropic_api_key="test-key")
    canned = {"memories": [
        {"type": "fact", "key": "location.city", "value": "Lives in Berlin", "cardinality": "single"},
        {"type": "fact", "key": "pet.dog", "value": "Has a dog named Biscuit",
         "cardinality": "multi", "entity": "Biscuit"},
        {"type": "preference", "key": "preference.communication_style",
         "value": "Prefers concise answers"},
        {"type": "fact", "key": "", "value": "dropped because no key"},
    ]}
    monkeypatch.setattr(extraction, "complete_json", lambda *a, **k: canned)
    cands, method = extraction.extract([{"role": "user", "content": "hi"}], None, settings)
    assert method == "llm"
    keys = {c.key for c in cands}
    assert keys == {"location.city", "pet.dog", "preference.communication_style"}
    dog = next(c for c in cands if c.key == "pet.dog")
    assert dog.entity == "Biscuit" and dog.cardinality == "multi"


def test_llm_failure_falls_back_to_rules(monkeypatch):
    from memory_service.llm import LLMError

    settings = Settings(extraction_provider="anthropic", anthropic_api_key="test-key")

    def boom(*a, **k):
        raise LLMError("simulated outage")

    monkeypatch.setattr(extraction, "complete_json", boom)
    cands, method = extraction.extract([{"role": "user", "content": "I live in Oslo."}], None, settings)
    assert "rules" in method
    assert any(c.key == "location.city" and "Oslo" in c.value for c in cands)
