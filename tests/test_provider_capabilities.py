"""
Tests for capability interfaces, response normalisation and provider
diagnostics (Section 5).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.provider_capabilities import (
    Capability,
    NormalizedUsage,
    StopReason,
    normalize_stop_reason,
    normalize_tool_calls,
    normalize_usage,
    provider_capabilities,
    select_for_capability,
    supports,
    validate_capability,
)
from orion_core.provider_diagnostics import (
    DiagnosticCategory,
    classify_provider_error,
    diagnose_profile_config,
    persist_diagnosis,
    recent_diagnoses,
)


@dataclass
class _Profile:
    name: str
    kind: str = "openai_compatible"
    model: str = "gpt-4o-mini"
    api_key: str = "k"
    base_url: str = "https://api.example.com/v1"
    enabled: bool = True

    @property
    def is_local(self) -> bool:
        return self.base_url.startswith(("http://127.0.0.1", "http://localhost"))


# ── capabilities ──────────────────────────────────────────────────────────────

def test_chat_model_capabilities():
    caps = provider_capabilities(_Profile("openai"))
    assert Capability.TEXT_GENERATION in caps
    assert Capability.TOOL_CALLING in caps
    assert Capability.STREAMING_TEXT in caps


def test_gemini_live_realtime_audio():
    caps = provider_capabilities(_Profile("gemini", kind="gemini_live", model="native-audio"))
    assert Capability.REALTIME_AUDIO in caps
    assert Capability.VISION in caps


def test_embedding_model_only_embeds():
    caps = provider_capabilities(_Profile("embed", model="text-embedding-3-large"))
    assert caps == {Capability.EMBEDDINGS}


def test_disabled_or_unconfigured_has_no_capabilities():
    assert provider_capabilities(_Profile("x", enabled=False)) == set()
    assert provider_capabilities(_Profile("x", api_key="", base_url="https://api.example.com/v1")) == set()


def test_local_needs_no_key():
    caps = provider_capabilities(_Profile("ollama", api_key="", base_url="http://127.0.0.1:11434/v1", model="llama3.1"))
    assert Capability.TEXT_GENERATION in caps


def test_validate_and_select_capability():
    text = _Profile("openai")
    embed = _Profile("embed", model="text-embedding-3-small")
    ok, _ = validate_capability(text, Capability.TEXT_GENERATION)
    assert ok
    bad, reason = validate_capability(embed, Capability.TEXT_GENERATION)
    assert not bad and "does not support" in reason
    chosen = select_for_capability([text, embed], Capability.TEXT_GENERATION)
    assert chosen == [text]


# ── normalisation ────────────────────────────────────────────────────────────

def test_normalize_openai_usage_with_cached_and_reasoning():
    raw = {
        "prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 800},
        "completion_tokens_details": {"reasoning_tokens": 120},
    }
    u = normalize_usage(raw, "openai_compatible")
    assert u.input_tokens == 1200 and u.output_tokens == 300
    assert u.cached_input_tokens == 800 and u.reasoning_tokens == 120
    assert u.total_tokens == 1500


def test_normalize_gemini_usage():
    raw = {
        "prompt_token_count": 900, "candidates_token_count": 210,
        "cached_content_token_count": 100, "thoughts_token_count": 40,
        "total_token_count": 1150,
    }
    u = normalize_usage(raw, "gemini_live")
    assert u.input_tokens == 900 and u.output_tokens == 210
    assert u.cached_input_tokens == 100 and u.reasoning_tokens == 40
    assert u.total_tokens == 1150


def test_normalize_usage_missing_is_none_not_zero():
    u = normalize_usage(None)
    assert u == NormalizedUsage()
    assert u.input_tokens is None  # never fabricated


def test_normalize_stop_reason():
    assert normalize_stop_reason("stop") is StopReason.STOP
    assert normalize_stop_reason("length") is StopReason.LENGTH
    assert normalize_stop_reason("tool_calls") is StopReason.TOOL_CALLS
    assert normalize_stop_reason("SAFETY") is StopReason.CONTENT_FILTER
    assert normalize_stop_reason(None) is StopReason.UNKNOWN


def test_normalize_tool_calls_openai_and_gemini():
    openai = [{"id": "c1", "function": {"name": "globe", "arguments": '{"place":"Tokyo"}'}}]
    calls = normalize_tool_calls(openai)
    assert calls[0].name == "globe" and calls[0].arguments == {"place": "Tokyo"}
    gemini = [{"name": "max_zoom_in_globe", "args": {"max_seconds": 5}}]
    calls2 = normalize_tool_calls(gemini)
    assert calls2[0].name == "max_zoom_in_globe" and calls2[0].arguments == {"max_seconds": 5}


# ── diagnostics ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("no text provider is available", DiagnosticCategory.MISSING_CONFIG),
    ("API key not valid. Please pass a valid API key.", DiagnosticCategory.INVALID_CREDENTIALS),
    ("Cannot connect to host 127.0.0.1:11434 ssl:default [refused]", DiagnosticCategory.LOCAL_MODEL_DOWN),
    ("Connection refused to api.example.com", DiagnosticCategory.UNREACHABLE_ENDPOINT),
    ("getaddrinfo failed", DiagnosticCategory.DNS_FAILURE),
    ("Request timed out after 30s", DiagnosticCategory.TIMEOUT),
    ("Rate limit exceeded (429)", DiagnosticCategory.RATE_LIMIT),
    ("You exceeded your current quota, insufficient_quota", DiagnosticCategory.QUOTA_EXHAUSTED),
    ("The model `gpt-9` does not exist", DiagnosticCategory.UNSUPPORTED_MODEL),
    ("This model does not support tool calling", DiagnosticCategory.UNSUPPORTED_CAPABILITY),
    ("maximum context length is 8192 tokens", DiagnosticCategory.CONTEXT_OVERFLOW),
])
def test_error_classification(text, expected):
    d = classify_provider_error(text, provider="p")
    assert d.category is expected
    assert d.remediation                # actionable
    assert "key" not in d.detail.lower() or "api key" in d.detail.lower()  # no raw secret


def test_config_diagnosis_detects_missing_key():
    bad = _Profile("openai", api_key="", base_url="https://api.example.com/v1")
    d = diagnose_profile_config(bad)
    assert d.category is DiagnosticCategory.MISSING_CONFIG
    good = _Profile("openai")
    assert diagnose_profile_config(good).category is DiagnosticCategory.OK


def test_diagnosis_persistence_roundtrip(tmp_path):
    journal = tmp_path / "diag.jsonl"
    d = classify_provider_error("rate limit exceeded", provider="openai")
    persist_diagnosis(d, journal=journal)
    persist_diagnosis(d, journal=journal)
    rows = recent_diagnoses(limit=10, journal=journal)
    assert len(rows) == 2
    assert rows[0]["category"] == "rate_limit"
