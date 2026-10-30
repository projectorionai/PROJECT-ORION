"""Routine replies must pass the supported latency settings to their API."""

from __future__ import annotations

import pytest

from orion_core.provider_transport import ChatRequest, OpenAIChatTransport
from orion_core.providers import AIProviderProfile, ProviderRouter


@pytest.mark.parametrize("base_url,model,expected", [
    ("https://api.deepseek.com", "deepseek-flash", {"thinking": {"type": "disabled"}}),
    ("https://api.deepseek.com/v1", "deepseek-v4-pro", {"thinking": {"type": "disabled"}}),
    ("https://api.cerebras.ai/v1", "gpt-oss-120b", {"reasoning_effort": "low"}),
    ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", {"reasoning_effort": "low"}),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash", {"reasoning_effort": "none"}),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash-lite", {"reasoning_effort": "none"}),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.1-flash-lite", {"reasoning_effort": "minimal"}),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.5-flash-lite", {"reasoning_effort": "minimal"}),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3-flash-preview", {"reasoning_effort": "low"}),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.5-flash", {"reasoning_effort": "low"}),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.6-flash", {"reasoning_effort": "low"}),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.7-flash", {"reasoning_effort": "low"}),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash", {"reasoning_effort": "low"}),
])
@pytest.mark.parametrize("stream", [False, True])
def test_supported_low_thinking_controls_reach_the_serialised_request(
    monkeypatch, base_url, model, expected, stream,
):
    monkeypatch.delenv("ORION_MODEL_THINKING", raising=False)
    router = ProviderRouter.__new__(ProviderRouter)
    profile = AIProviderProfile(name="test", kind="openai_compatible", base_url=base_url, model=model)
    quirks = {"model": model}
    router._apply_model_quirks(profile, quirks)
    quirks.pop("model")
    request = ChatRequest(model=model, system="system", prompt="Hello", max_tokens=777,
                          stream=stream, extra=quirks)
    payload = OpenAIChatTransport(base_url).payload(request)
    for field, value in expected.items():
        assert payload[field] == value
    assert payload["max_tokens"] == 777
    assert payload["stream"] is stream


@pytest.mark.parametrize("base_url,model", [
    ("https://api.deepseek.com", "deepseek-flash"),
    ("https://api.cerebras.ai/v1", "gpt-oss-120b"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.5-flash-lite"),
])
def test_explicit_thinking_choice_restores_provider_defaults(monkeypatch, base_url, model):
    monkeypatch.setenv("ORION_MODEL_THINKING", "1")
    profile = AIProviderProfile(name="test", kind="openai_compatible", base_url=base_url, model=model)
    payload = {"model": model}
    ProviderRouter.__new__(ProviderRouter)._apply_model_quirks(profile, payload)
    assert payload == {"model": model}


@pytest.mark.parametrize("base_url,model", [
    ("https://api.deepseek.com.attacker.invalid", "deepseek-flash"),
    ("https://proxy.invalid/api.deepseek.com", "deepseek-flash"),
    ("https://api.deepseek.com", "custom-model"),
    ("https://api.cerebras.ai.attacker.invalid/v1", "gpt-oss-120b"),
    ("https://api.cerebras.ai/v1", "custom-model"),
    ("https://proxy.invalid/v1", "gpt-oss-120b"),
    ("https://generativelanguage.googleapis.com.attacker.invalid/v1", "gemini-2.5-flash"),
    ("https://proxy.invalid/generativelanguage.googleapis.com", "gemini-3.8-flash"),
    ("https://api.groq.com.attacker.invalid/v1", "gpt-oss-120b"),
    ("https://proxy.invalid/api.groq.com/v1", "gpt-oss-120b"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash-image"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash-tts"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash-tts"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.1-flash-lite-image"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.1-pro-preview"),
])
def test_api_specific_controls_do_not_leak_to_other_hosts_or_models(monkeypatch, base_url, model):
    monkeypatch.delenv("ORION_MODEL_THINKING", raising=False)
    profile = AIProviderProfile(name="test", kind="openai_compatible", base_url=base_url, model=model)
    payload = {"model": model}
    ProviderRouter.__new__(ProviderRouter)._apply_model_quirks(profile, payload)
    assert payload == {"model": model}
