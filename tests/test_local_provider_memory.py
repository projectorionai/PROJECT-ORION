"""
Tests for local-model memory-footprint controls — a bounded context length
and immediate unload for local Ollama/LM Studio fallback calls, so the
local safety net never balloons past what a modest machine (e.g. 16GB RAM,
8GB VRAM) can actually hold. Cloud profiles must be untouched by either.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.providers import AIProviderProfile, OrionProviderSettings, ProviderRouter


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _FakePostResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body


class _FakePostCtx:
    def __init__(self, response: _FakePostResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, body: str) -> None:
        self.calls: list[dict] = []
        self._body = body

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakePostCtx(_FakePostResponse(200, self._body))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _router() -> ProviderRouter:
    settings = OrionProviderSettings(active_provider="none", provider_order=[], providers={})
    return ProviderRouter(settings, _StubBus(), memory=object())


_OK_BODY = json.dumps({
    "choices": [{"message": {"content": "hello, sir"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
})

_LOCAL_PROFILE = dict(
    name="local_ollama", kind="openai_compatible", model="llama3.1:latest",
    base_url="http://127.0.0.1:11434/v1", api_key="local", enabled=True,
)
_CLOUD_PROFILE = dict(
    name="cloud_x", kind="openai_compatible", model="gpt-x",
    base_url="https://api.example.com/v1", api_key="sk-x", enabled=True,
)


def test_local_profile_bounds_context_and_unloads_after_the_call(monkeypatch):
    import orion_core.providers as providers_module

    router = _router()
    profile = AIProviderProfile(**_LOCAL_PROFILE)
    session = _FakeSession(_OK_BODY)
    monkeypatch.setattr(providers_module, "ClientSession", lambda *a, **k: session)

    result = asyncio.run(router._openai_compatible_chat(profile, "hi", instruction="SYS"))

    assert result == "hello, sir"
    sent = session.calls[0]["json"]
    assert sent["keep_alive"] == 0
    assert sent["options"] == {"num_ctx": ProviderRouter.LOCAL_NUM_CTX}


def test_cloud_profile_sends_neither_keep_alive_nor_num_ctx(monkeypatch):
    import orion_core.providers as providers_module

    router = _router()
    profile = AIProviderProfile(**_CLOUD_PROFILE)
    session = _FakeSession(_OK_BODY)
    monkeypatch.setattr(providers_module, "ClientSession", lambda *a, **k: session)

    asyncio.run(router._openai_compatible_chat(profile, "hi", instruction="SYS"))

    sent = session.calls[0]["json"]
    assert "keep_alive" not in sent
    assert "options" not in sent


def test_local_profile_streaming_variant_also_bounds_context(monkeypatch):
    import orion_core.providers as providers_module

    router = _router()
    profile = AIProviderProfile(**_LOCAL_PROFILE)

    sse_body = (
        'data: {"choices": [{"delta": {"content": "hi"}}]}\n'
        'data: [DONE]\n'
    )

    class _StreamResponse:
        status = 200

        def __init__(self, body: str) -> None:
            self._lines = [line.encode("utf-8") for line in body.splitlines(keepends=True)]

        @property
        def content(self):
            return self._aiter()

        async def _aiter(self):
            for line in self._lines:
                yield line

    class _StreamSession:
        def __init__(self, body: str) -> None:
            self.calls: list[dict] = []
            self._body = body

        def post(self, url, headers=None, json=None):
            self.calls.append({"url": url, "headers": headers, "json": json})
            return _FakePostCtx(_StreamResponse(self._body))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    session = _StreamSession(sse_body)
    monkeypatch.setattr(providers_module, "ClientSession", lambda *a, **k: session)

    result = asyncio.run(
        router._openai_compatible_chat_stream(profile, "hi", instruction="SYS"))

    assert result == "hi"
    sent = session.calls[0]["json"]
    assert sent["keep_alive"] == 0
    assert sent["options"] == {"num_ctx": ProviderRouter.LOCAL_NUM_CTX}
