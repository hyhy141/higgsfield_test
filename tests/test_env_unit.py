"""Environment-driven behavior: provider selection from keys, and the optional
bearer-auth dependency. In-process, no running service required."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from memory_service import main
from memory_service.config import Settings


# ── Provider / OpenAI-optional behavior ──────────────────────────────────────
def test_no_openai_key_uses_rules():
    s = Settings(extraction_provider="auto", openai_api_key=None, anthropic_api_key=None)
    assert s.resolved_provider == "rules"
    assert s.resolved_model is None  # nothing calls OpenAI


def test_openai_key_enables_openai_with_default_model():
    s = Settings(extraction_provider="auto", openai_api_key="sk-test")
    assert s.resolved_provider == "openai"
    assert s.resolved_model == "gpt-4o-mini"


def test_openai_model_override():
    s = Settings(extraction_provider="auto", openai_api_key="sk-test", openai_model="gpt-4.1")
    assert s.resolved_model == "gpt-4.1"


def test_anthropic_preferred_when_present():
    s = Settings(extraction_provider="auto", anthropic_api_key="key", openai_api_key="sk-test")
    assert s.resolved_provider == "anthropic"


# ── Optional bearer auth ──────────────────────────────────────────────────────
def _patch_token(monkeypatch, token):
    monkeypatch.setattr(main, "get_settings", lambda: Settings(memory_auth_token=token))


def test_auth_disabled_when_token_unset(monkeypatch):
    _patch_token(monkeypatch, None)
    assert main.require_auth(authorization=None) is None  # no auth required


def test_auth_required_rejects_missing_token(monkeypatch):
    _patch_token(monkeypatch, "secret")
    with pytest.raises(HTTPException) as ei:
        main.require_auth(authorization=None)
    assert ei.value.status_code == 401


def test_auth_required_rejects_wrong_token(monkeypatch):
    _patch_token(monkeypatch, "secret")
    with pytest.raises(HTTPException) as ei:
        main.require_auth(authorization="Bearer nope")
    assert ei.value.status_code == 401


def test_auth_accepts_correct_token(monkeypatch):
    _patch_token(monkeypatch, "secret")
    assert main.require_auth(authorization="Bearer secret") is None


def test_auth_error_does_not_leak_token(monkeypatch):
    _patch_token(monkeypatch, "super-secret-value")
    with pytest.raises(HTTPException) as ei:
        main.require_auth(authorization="Bearer wrong")
    assert "super-secret-value" not in str(ei.value.detail)
