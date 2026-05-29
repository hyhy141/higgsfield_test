"""In-process unit tests for taxonomy, token budgeting, and request-model coercion."""

from __future__ import annotations

from memory_service.models import Message, RecallRequest, SearchRequest
from memory_service.taxonomy import (
    default_cardinality,
    humanize_key,
    is_identity_key,
    memory_search_text,
    normalize_key,
)
from memory_service.tokens import estimate_tokens


def test_cardinality_rules():
    assert default_cardinality("pet.dog") == "multi"
    assert default_cardinality("diet.allergy") == "multi"
    assert default_cardinality("skill.python") == "multi"
    assert default_cardinality("employment.company") == "single"
    assert default_cardinality("location.city") == "single"
    assert default_cardinality("opinion.typescript") == "single"


def test_identity_keys():
    assert is_identity_key("location.city")
    assert is_identity_key("employment.company")
    assert is_identity_key("pet.dog")
    assert not is_identity_key("opinion.typescript")
    assert not is_identity_key("project.foo")


def test_normalize_key():
    assert normalize_key("Location.City") == "location.city"
    assert normalize_key("employment / company") == "employment_company"
    assert normalize_key("  Pet..Dog  ") == "pet.dog"
    assert normalize_key("") == "misc.note"


def test_memory_search_text():
    assert memory_search_text("diet.restriction", "Is vegetarian") == "diet restriction: Is vegetarian"
    assert humanize_key("preference.communication_style") == "preference communication style"


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    # A short line should land in a sane range, not wildly off.
    assert 2 <= estimate_tokens("the user lives in Berlin") <= 12


def test_message_coerces_bad_content():
    assert Message(role="tool", content=None).content == ""
    assert Message(role="user", content=123).content == "123"


def test_recall_request_clamps_tokens():
    assert RecallRequest(query="x", max_tokens=-5).max_tokens == 32
    assert RecallRequest(query="x", max_tokens=10_000_000).max_tokens == 32_000
    assert RecallRequest(query="x", max_tokens="not a number").max_tokens == 1024


def test_search_request_clamps_limit():
    assert SearchRequest(query="x", limit=-1).limit == 1
    assert SearchRequest(query="x", limit=99999).limit == 100
