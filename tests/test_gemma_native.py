"""Hosted Gemma 4 requests use native Gemini JSON and never execute tools."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from orion_core.gemma_native import GemmaNativeTransport, gemma_payload
from orion_core.provider_capabilities import Capability, provider_capabilities
from orion_core.provider_transport import ChatRequest
from orion_core.providers import (
    ProviderRouter, ProviderText, _default_provider_payload,
    _merge_provider_payload, _settings_from_payload,
)


class _Content:
    def __init__(self, raw: bytes):
        self.raw = raw

    async def read(self, limit: int):
        return self.raw[:limit]

    async def iter_chunked(self, size: int):
        for at in range(0, len(self.raw), max(1, size // 4)):
            yield self.raw[at:at + max(1, size // 4)]


class _Response:
    def __init__(self, raw: bytes, *, content_type: str = "application/json",
                 status: int = 200):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.content = _Content(raw)
        self.closed = False

    def close(self):
        self.closed = True

    def release(self):
        pass


class _Session:
    def __init__(self, response):
        self.response = response
        self.sent = None

    async def request(self, method, url, **kwargs):
        self.sent = (method, url, kwargs)
        return self.response


def _request(**overrides):
    raw = dict(model="gemma-4-26b-a4b-it", system="Use British English.",
               prompt="Summarise this.", max_tokens=200, api_key="unit-test-key")
    raw.update(overrides)
    return ChatRequest(**raw)


def test_native_payload_and_endpoint_keep_key_out_of_body(monkeypatch):
    monkeypatch.delenv("ORION_MODEL_THINKING", raising=False)
    req = _request(extra={"function_declarations": [{
        "name": "suggest_route", "description": "Suggest a route",
        "parameters": {"type": "object", "properties": {
            "destination": {"type": "string"}}, "required": ["destination"]},
    }]})
    payload = gemma_payload(req)
    assert payload["contents"][0]["parts"][0] == {"text": "Summarise this."}
    assert payload["systemInstruction"]["parts"][0]["text"] == "Use British English."
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "suggest_route"
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "unit-test-key" not in json.dumps(payload)
    assert GemmaNativeTransport().endpoint(req.model, False).endswith(
        "/models/gemma-4-26b-a4b-it:generateContent")
    with pytest.raises(ValueError):
        gemma_payload(_request(model="gemini-3.8-flash"))


def test_gemma_thinking_is_minimal_for_both_models_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("ORION_MODEL_THINKING", raising=False)
    for model in ("gemma-4-26b-a4b-it", "gemma-4-31b-it"):
        assert gemma_payload(_request(model=model))["generationConfig"][
            "thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert gemma_payload(_request(extra={"thinking_level": "high"}))["generationConfig"][
        "thinkingConfig"] == {"thinkingLevel": "high"}
    monkeypatch.setenv("ORION_MODEL_THINKING", "1")
    assert gemma_payload(_request())["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "high"}
    with pytest.raises(ValueError, match="thinking level"):
        gemma_payload(_request(extra={"thinking_level": "invalid"}))


def test_native_text_and_function_proposal_are_separate_results():
    data = {"candidates": [{"content": {"parts": [
        {"text": "Private reasoning", "thought": True},
        {"functionCall": {"name": "suggest_route", "args": {"destination": "Bath"}}},
    ]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3}}
    session = _Session(_Response(json.dumps(data).encode()))
    req = _request(extra={"function_declarations": [{"name": "suggest_route"}]})
    result = asyncio.run(GemmaNativeTransport().chat(
        session, req, timeout=object(), trace={}))
    assert result.text == ""
    assert result.tool_calls == [{"name": "suggest_route",
                                  "args": {"destination": "Bath"}, "id": ""}]
    assert result.usage["prompt_tokens"] == 5
    assert session.sent[2]["headers"]["x-goog-api-key"] == "unit-test-key"
    assert "unit-test-key" not in session.sent[2]["json"].__str__()


def test_undeclared_function_cannot_escape_as_claimed_completion():
    data = {"candidates": [{"content": {"parts": [{"text": "Done.",
        }, {"functionCall": {"name": "send_email", "args": {}}}]},
        "finishReason": "STOP"}]}
    session = _Session(_Response(json.dumps(data).encode()))
    with pytest.raises(RuntimeError, match="undeclared function"):
        asyncio.run(GemmaNativeTransport().chat(
            session, _request(), timeout=object(), trace={}))


def test_native_stream_requires_completion_and_forwards_only_answer_text():
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "internal", "thought": True},
                                                 {"text": "Hello "}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "there."}]},
                         "finishReason": "STOP"}]},
    ]
    raw = b"".join(("data: " + json.dumps(chunk) + "\n\n").encode() for chunk in chunks)
    seen = []
    session = _Session(_Response(raw, content_type="text/event-stream"))
    result = asyncio.run(GemmaNativeTransport().chat(
        session, _request(stream=True, on_delta=seen.append), timeout=object(), trace={}))
    assert result.text == "Hello there."
    assert seen == ["Hello ", "there."]
    interrupted = _Session(_Response(b'data: {"candidates": []}\n\n',
                                     content_type="text/event-stream"))
    with pytest.raises(RuntimeError, match="did not finish"):
        asyncio.run(GemmaNativeTransport().chat(
            interrupted, _request(stream=True), timeout=object(), trace={}))


@pytest.mark.parametrize("stream", [False, True])
def test_empty_max_tokens_is_a_failure_eligible_for_provider_fallback(stream):
    data = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"totalTokenCount": 27}}
    if stream:
        raw = ("data: " + json.dumps(data) + "\n\n").encode()
        response = _Response(raw, content_type="text/event-stream")
    else:
        response = _Response(json.dumps(data).encode())
    with pytest.raises(RuntimeError, match="output budget before answering"):
        asyncio.run(GemmaNativeTransport().chat(
            _Session(response), _request(stream=stream, max_tokens=20),
            timeout=object(), trace={}))


def test_max_tokens_with_partial_text_can_be_continued():
    data = {"candidates": [{"content": {"parts": [{"text": "First part"}]},
                            "finishReason": "MAX_TOKENS"}]}
    result = asyncio.run(GemmaNativeTransport().chat(
        _Session(_Response(json.dumps(data).encode())), _request(max_tokens=20),
        timeout=object(), trace={}))
    assert result.text == "First part"
    assert result.done_reason == "MAX_TOKENS"


def test_gemma_profiles_use_native_api_and_preserve_custom_text_choices():
    defaults = _default_provider_payload("unit-test-key")
    assert defaults["providers"]["gemma_4_31b"]["kind"] == "gemma_native"
    assert defaults["providers"]["gemma_4_26b"]["model"] == "gemma-4-26b-a4b-it"
    profile = _settings_from_payload(defaults).providers["gemma_4_31b"]
    assert Capability.TOOL_CALLING in provider_capabilities(profile)
    router = ProviderRouter(_settings_from_payload(defaults), SimpleNamespace(),
                            memory=object())
    transport = asyncio.run(router._transport_for(profile))
    assert isinstance(transport, GemmaNativeTransport)
    assert "gemma_4_26b" in [p.name for p in router.select_text_profiles("Hello")]
    assert "gemma_4_31b" in [p.name for p in router.select_text_profiles(
        "Design a reliable network architecture")]
    customised = _merge_provider_payload(defaults, {"providers": {
        "gemini_text": {"model": "gemini-2.5-pro", "models": ["gemini-2.5-pro"]}}})
    assert not customised["providers"]["gemma_4_31b"]["enabled"]
    assert not customised["providers"]["gemma_4_26b"]["enabled"]


def test_proposal_route_returns_calls_without_running_them(monkeypatch):
    monkeypatch.setenv("ORION_MODE", "cloud")
    settings = _settings_from_payload(_default_provider_payload("unit-test-key"))
    router = ProviderRouter(settings, SimpleNamespace(), memory=object())
    observed = []

    async def fake_attempt(profile, prompt, **kwargs):
        observed.append((profile.name, kwargs["function_declarations"]))
        result = ProviderText("")
        result.tool_calls = [{"name": "reviewed_action", "args": {"item": "notes"}}]
        return result

    monkeypatch.setattr(router, "_chat_attempt", fake_attempt)
    profile, calls, answer = asyncio.run(router.propose_gemma_functions(
        "Please propose this action", [{"name": "reviewed_action"}]))
    assert profile.kind == "gemma_native" and answer == ""
    assert calls == [{"name": "reviewed_action", "args": {"item": "notes"}}]
    assert observed[0][0] == profile.name
