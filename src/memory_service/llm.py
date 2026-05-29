"""Provider-agnostic LLM client used by the extraction pipeline.

Supports Anthropic and OpenAI behind one `complete_json` call that always returns
a parsed dict. Any failure (missing key, bad model id, timeout, non-JSON output)
raises LLMError; the extractor catches it and falls back to the rule-based engine,
so a flaky or unconfigured LLM never fails a /turns request.
"""

from __future__ import annotations

import json
import logging
import re

from .config import Settings

log = logging.getLogger("memory.llm")


class LLMError(RuntimeError):
    pass


def provider_available(settings: Settings) -> bool:
    prov = settings.resolved_provider
    if prov == "anthropic":
        return bool(settings.anthropic_api_key)
    if prov == "openai":
        return bool(settings.openai_api_key)
    return False


def complete_json(settings: Settings, system: str, user: str) -> dict:
    """Call the configured provider and parse its reply as a JSON object."""
    prov = settings.resolved_provider
    model = settings.resolved_model or ""
    timeout = settings.extraction_timeout_s
    if prov == "anthropic":
        text = _anthropic(settings.anthropic_api_key or "", model, system, user, timeout)
    elif prov == "openai":
        text = _openai(settings.openai_api_key or "", model, system, user, timeout)
    else:
        raise LLMError(f"no LLM provider available (provider={prov})")
    return _parse_json_object(text)


def _anthropic(api_key: str, model: str, system: str, user: str, timeout: float) -> str:
    if not api_key:
        raise LLMError("anthropic api key missing")
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"anthropic call failed: {exc}") from exc


def _openai(api_key: str, model: str, system: str, user: str, timeout: float) -> str:
    if not api_key:
        raise LLMError("openai api key missing")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"openai call failed: {exc}") from exc


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(text: str) -> dict:
    """Tolerant JSON-object extraction: handles fenced code blocks and prose
    wrapped around the JSON."""
    if not text:
        raise LLMError("empty LLM response")
    text = text.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(text)
        if not m:
            raise LLMError("no JSON object in LLM response") from None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"unparseable JSON from LLM: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMError("LLM JSON was not an object")
    return obj
