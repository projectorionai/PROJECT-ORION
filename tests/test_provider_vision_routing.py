"""Which backend a camera scan actually reaches (ProviderRouter.generate_vision).

The electronics workbench is only as reliable as this routing, and it had a
quiet flaw: ``generate_vision`` tries at most two profiles, and several named
profiles routinely resolve to ONE endpoint. On a real installation here, the
general profile and the dedicated 'thoughts' profile both pointed at the same
OpenRouter model with the same key, so both attempts were byte-for-byte the
same HTTP request — no fallback at all — while a configured, healthy Gemini
vision endpoint sat third in the list and was never reached.

These tests pin the routing rather than the transport: ``_openai_compatible_chat``
is stubbed, so nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.providers import (  # noqa: E402
    AIProviderProfile,
    OrionProviderSettings,
    ProviderRouter,
)

JPEG = b"\xff\xd8\xff" + b"\x00" * 64


class _Signal:
    def __init__(self, sink, name):
        self._sink, self._name = sink, name

    def emit(self, *payload):
        self._sink.append((self._name, payload[0] if len(payload) == 1 else payload))

    def connect(self, *_a, **_k):
        pass


class _StubBus:
    def __init__(self):
        self.events = []
        self.log = _Signal(self.events, "log")

    def __getattr__(self, name):
        sig = _Signal(self.events, name)
        object.__setattr__(self, name, sig)
        return sig


def _profile(name, **overrides):
    base = dict(
        name=name, kind="openai_compatible", model="gpt-4o-mini",
        api_key="key-1", base_url="https://openrouter.ai/api/v1", enabled=True,
    )
    base.update(overrides)
    return AIProviderProfile(**base)


def _router(*profiles):
    settings = OrionProviderSettings(
        active_provider=profiles[0].name,
        provider_order=[p.name for p in profiles],
        providers={p.name: p for p in profiles},
    )
    return ProviderRouter(settings, _StubBus(), memory=object())


def _record_attempts(router, *, failing=()):
    """Stub the transport; return the list that records each attempt."""
    attempts: list[tuple[str, str, str]] = []

    async def chat(profile, prompt, *_a, **kwargs):
        attempts.append((profile.name, str(profile.base_url), str(profile.model)))
        if profile.name in failing:
            raise RuntimeError(f"{profile.name} is unavailable")
        return "{}"

    router._openai_compatible_chat = chat
    return attempts


def _run(router):
    return asyncio.run(router.generate_vision(
        JPEG, "inspect this board", instruction="be precise",
        task="electronics_inspection"))


# ── the defect ───────────────────────────────────────────────────────────────

def test_identical_endpoints_do_not_consume_both_attempts():
    """Two names, one endpoint: the duplicate must not crowd out a real
    alternative."""
    router = _router(
        _profile("openrouter"),
        _profile("thoughts"),                     # same url + model + key
        _profile("gemini_vision", model="gemini-2.5-flash",
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                 api_key="key-2"),
    )
    attempts = _record_attempts(router, failing=("openrouter", "thoughts"))
    _profile_used, answer = _run(router)
    names = [name for name, _url, _model in attempts]
    assert "gemini_vision" in names, (
        "the duplicate endpoint was tried twice and the genuine fallback was "
        f"never reached: {names}")
    assert names.count("thoughts") == 0
    assert answer == "{}", "the scan failed although a healthy provider existed"


def test_profiles_differing_only_by_key_are_both_kept():
    """Same endpoint with a DIFFERENT key is real redundancy, not a duplicate."""
    router = _router(
        _profile("primary", api_key="key-1"),
        _profile("spare", api_key="key-2"),
    )
    attempts = _record_attempts(router, failing=("primary",))
    _run(router)
    assert [name for name, _u, _m in attempts] == ["primary", "spare"]


# ── ordinary routing still holds ─────────────────────────────────────────────

def test_the_first_healthy_vision_provider_answers():
    router = _router(_profile("openrouter"))
    attempts = _record_attempts(router)
    _profile_used, answer = _run(router)
    assert answer == "{}"
    assert len(attempts) == 1


def test_stalled_provider_leaves_time_for_fallback(monkeypatch):
    router = _router(_profile("primary"), _profile("spare", api_key="other-key"))
    attempts = _record_attempts(router)
    deadlines = []

    async def bounded(coro, timeout):
        deadlines.append(timeout)
        if len(deadlines) == 1:
            coro.close()
            raise asyncio.TimeoutError
        return await coro

    monkeypatch.setattr(asyncio, "wait_for", bounded)
    profile, answer = _run(router)
    assert profile.name == "spare" and answer == "{}"
    assert deadlines == [15.0, 15.0]


def test_a_live_audio_profile_is_offered_as_a_vision_endpoint():
    """A native-audio model id cannot service chat completions, so the live
    profile is remapped onto Google's OpenAI-compatible vision endpoint rather
    than being discarded."""
    router = _router(_profile(
        "gemini", kind="gemini_live",
        model="models/gemini-2.5-flash-native-audio-preview-12-2025",
        base_url=""))
    attempts = _record_attempts(router)
    _run(router)
    assert attempts, "a live profile offered no vision endpoint at all"
    _name, url, model = attempts[0]
    assert "generativelanguage.googleapis.com" in url
    assert "native-audio" not in model


def test_no_vision_capable_provider_is_reported_not_guessed():
    """A text-only model must never be asked to pretend it saw pixels."""
    router = _router(_profile("tiny", model="llama-3.1-8b-instant"))
    _record_attempts(router)
    with pytest.raises(Exception):
        _run(router)


@pytest.mark.parametrize("kind", ["empty", "none", "not-bytes", "oversized"])
def test_an_unusable_image_is_rejected_before_any_provider_is_called(kind):
    # Built here rather than parametrised: an 8 MB value in a test id overflows
    # the environment variable pytest passes it through on Windows.
    image = {"empty": b"", "none": None, "not-bytes": "not bytes",
             "oversized": b"x" * 8_000_001}[kind]
    router = _router(_profile("openrouter"))
    attempts = _record_attempts(router)
    with pytest.raises(ValueError):
        asyncio.run(router.generate_vision(
            image, "inspect", instruction="i", task="electronics_inspection"))
    assert attempts == []
