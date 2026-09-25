"""
Tests for provider availability and degraded mode (Section 2).

* No usable text provider -> a typed NoTextProviderError (not a bare
  RuntimeError) plus a single degraded-mode broadcast with alternatives.
* Recovery -> a single 'recovered' broadcast when a provider succeeds again.
* The self-improvement heartbeat skips gracefully (no fault) while degraded.
* Deterministic operation never depends on a provider.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.providers import (
    AIProviderProfile,
    NoTextProviderError,
    OrionProviderSettings,
    ProviderRouter,
)


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
        self.provider_degraded = _Signal(self.events, "provider_degraded")

    def __getattr__(self, name):
        sig = _Signal(self.events, name)
        object.__setattr__(self, name, sig)
        return sig


def _router(providers=None, order=None):
    settings = OrionProviderSettings(
        active_provider="none",
        provider_order=order or [],
        providers=providers or {},
    )
    bus = _StubBus()
    return ProviderRouter(settings, bus, memory=object()), bus


def _degraded_events(bus):
    return [p for (n, p) in bus.events if n == "provider_degraded"]


def test_no_provider_raises_typed_error_and_signals_degraded():
    router, bus = _router()
    try:
        asyncio.run(router.generate_text("hello"))
        raised = None
    except NoTextProviderError as exc:
        raised = exc
    assert isinstance(raised, NoTextProviderError)
    assert raised.remediation                      # actionable, no secrets
    events = _degraded_events(bus)
    assert events and events[0]["degraded"] is True


def test_degraded_broadcast_is_emitted_once():
    router, bus = _router()
    for _ in range(3):
        try:
            asyncio.run(router.generate_text("hi"))
        except NoTextProviderError:
            pass
    # Only the first failure announces degraded mode — no per-request flooding.
    assert len(_degraded_events(bus)) == 1


def test_recovery_signals_once_when_provider_succeeds():
    profile = AIProviderProfile(
        name="local_test", kind="openai_compatible", model="m",
        base_url="http://127.0.0.1:9/v1", api_key="local", enabled=True,
    )
    router, bus = _router({"local_test": profile}, ["local_test"])

    async def scenario():
        # Force degraded, then a successful call must recover exactly once.
        router._enter_degraded("forced for test")

        async def fake_chat(_profile, _prompt, _extra="", **_kw):
            return "all good, sir"

        router._openai_compatible_chat = fake_chat  # type: ignore[assignment]
        prof, text = await router.generate_text("status?")
        assert text == "all good, sir"

    asyncio.run(scenario())
    recovered = [p for p in _degraded_events(bus) if p["degraded"] is False]
    assert len(recovered) == 1


def test_status_snapshot_is_redaction_safe():
    profile = AIProviderProfile(
        name="cloud_x", kind="openai_compatible", model="m",
        base_url="https://api.example.com/v1", api_key="super-secret-key", enabled=True,
    )
    router, _ = _router({"cloud_x": profile}, ["cloud_x"])
    snap = router.status()
    assert set(snap) >= {"degraded", "text_available", "mode", "online"}
    assert "super-secret-key" not in repr(snap)


def test_text_available_reflects_config():
    empty, _ = _router()
    assert empty.text_available() is False
    profile = AIProviderProfile(
        name="local_test", kind="openai_compatible", model="m",
        base_url="http://127.0.0.1:9/v1", api_key="local", enabled=True,
    )
    populated, _ = _router({"local_test": profile}, ["local_test"])
    assert populated.text_available() is True


def test_heartbeat_pauses_without_provider_and_never_faults():
    from orion_core.forge import ImprovementHeartbeat

    class _Router:
        def __init__(self, available):
            self._available = available

        def text_available(self):
            return self._available

    class _HbBus:
        def __init__(self):
            self.lines = []
            self.log = _Signal([], "log")
            self.log.emit = self.lines.append  # capture lines

        def __getattr__(self, name):
            return _Signal([], name)

    hb = ImprovementHeartbeat.__new__(ImprovementHeartbeat)
    hb.bus = _HbBus()
    hb.router = _Router(available=False)
    hb._provider_paused = False
    assert hb._text_generation_available() is False
    hb.router._available = True
    assert hb._text_generation_available() is True
