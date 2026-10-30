"""
The provider router's circuit breaker.

* Transient faults (HTTP 5xx, timeouts, refused connections) escalate: each
  fault in a row doubles the cooldown from 45 s to a 10 minute ceiling, with
  downward jitter. A flat 45 s retried a dead endpoint for the whole session.
* A success closes the breaker and resets every escalation, including the
  out-of-GPU-memory one, which previously never reset.
* After a cooldown the provider is half-open: its next call is a probe, and
  while that probe is in flight concurrent turns try other providers first —
  but the recovering provider is still offered, never removed outright.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.providers import (
    AIProviderProfile,
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

    def __getattr__(self, name):
        sig = _Signal(self.events, name)
        object.__setattr__(self, name, sig)
        return sig


def _profile(name="p", **overrides):
    base = dict(
        name=name, kind="openai_compatible", model="model-a",
        api_key="key-1", base_url="https://api.example.com/v1", enabled=True,
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


def _remaining(router, name):
    return router._cooldowns.get(name, 0.0) - time.monotonic()


def _expire(router, name):
    router._cooldowns[name] = 0.0


def test_transient_faults_escalate_to_a_ceiling():
    profile = _profile()
    router = _router(profile)
    expected = [45.0, 90.0, 180.0, 360.0, 600.0, 600.0]
    for ceiling in expected:
        router.mark_failure(profile, RuntimeError("HTTP 503: upstream unavailable"))
        remaining = _remaining(router, "p")
        # Jitter only shortens the wait (never past the ceiling).
        assert 0.8 * ceiling - 1.0 < remaining <= ceiling
        _expire(router, "p")


@pytest.mark.parametrize("kind,status", [("openai_compatible", 503), ("gemma_native", 500)])
def test_google_server_burst_recovers_before_opening_breaker(kind, status):
    profile = _profile(kind=kind, base_url="https://generativelanguage.googleapis.com/v1beta")
    router = _router(profile)
    calls = []

    async def call():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError(f"HTTP {status}: briefly overloaded")
        return "recovered"

    assert asyncio.run(router._call_with_heal(profile, call)) == "recovered"
    assert len(calls) == 2
    assert router.breaker_state(profile) == "closed"
    assert profile.name not in router._fault_strikes


def test_google_retry_failure_opens_breaker_once_and_preserves_final_fault():
    profile = _profile(base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    router = _router(profile)
    calls = []

    async def call():
        calls.append(True)
        raise RuntimeError(f"HTTP 503: failed attempt {len(calls)}")

    with pytest.raises(RuntimeError, match="failed attempt 2"):
        asyncio.run(router._call_with_heal(profile, call))
    assert len(calls) == 2
    assert router._fault_strikes[profile.name] == 1
    assert router.breaker_state(profile) == "open"


def test_google_overload_uses_configured_alternate_without_persisting_choice(monkeypatch):
    profile = _profile(base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                       models=["model-a", "model-b"])
    router = _router(profile)
    models = []
    persisted = []
    monkeypatch.setattr(router, "_persist_model", lambda p, **_: persisted.append(p.name))

    async def call():
        models.append(router.active_model(profile))
        if len(models) == 1:
            raise RuntimeError("HTTP 503: primary is overloaded")
        return "alternate answered"

    assert asyncio.run(router._call_with_heal(profile, call)) == "alternate answered"
    assert models == ["model-a", "model-b"]
    assert profile.model == "model-a"
    assert persisted == []


@pytest.mark.parametrize("error,base,already_shown", [
    ("HTTP 503: busy", "https://generativelanguage.googleapis.com/v1beta", True),
    ("HTTP 429: rate limit", "https://generativelanguage.googleapis.com/v1beta", False),
    ("HTTP 401: invalid key", "https://generativelanguage.googleapis.com/v1beta", False),
    ("HTTP 503: busy", "https://generativelanguage.googleapis.com.example.com/v1", False),
])
def test_google_retry_does_not_replay_output_or_retry_auth_quota_and_other_hosts(
        error, base, already_shown):
    profile = _profile(base_url=base)
    router = _router(profile)
    calls = []

    async def call():
        calls.append(True)
        raise RuntimeError(error)

    with pytest.raises(RuntimeError):
        asyncio.run(router._call_with_heal(profile, call, may_retry=lambda: not already_shown))
    assert len(calls) == 1


def test_google_retry_respects_whole_call_deadline():
    from orion_core.providers import _CALL, _CallScope

    profile = _profile(base_url="https://generativelanguage.googleapis.com/v1beta")
    router = _router(profile)
    calls = []

    async def call():
        calls.append(True)
        raise RuntimeError("HTTP 503: busy")

    async def flow():
        token = _CALL.set(_CallScope("interactive", time.monotonic() + 0.1))
        try:
            with pytest.raises(RuntimeError, match="HTTP 503"):
                await router._call_with_heal(profile, call)
        finally:
            _CALL.reset(token)

    asyncio.run(flow())
    assert len(calls) == 1


def test_google_array_error_preserves_reason_and_retry_delay():
    import json
    from orion_core.provider_transport import _error_text

    raw = json.dumps([{"error": {"code": 503, "message": "Model is overloaded",
                     "details": [{"retryDelay": "12s"}]}}], indent=2)
    result = _error_text(raw)
    assert result.startswith("Model is overloaded")
    assert ProviderRouter.RETRY_RE.search(result).group(1) == "12"


def test_connection_refused_is_transient_and_escalates():
    profile = _profile()
    router = _router(profile)
    router.mark_failure(profile, ConnectionRefusedError("Cannot connect to host"))
    first = _remaining(router, "p")
    _expire(router, "p")
    router.mark_failure(profile, ConnectionRefusedError("Cannot connect to host"))
    assert _remaining(router, "p") > first
    assert any("fault 2 in a row" in str(payload) for _n, payload in router.bus.events)


def test_success_resets_every_escalation():
    profile = _profile()
    router = _router(profile)
    for _ in range(3):
        router.mark_failure(profile, RuntimeError("HTTP 502: bad gateway"))
        _expire(router, "p")
    router.mark_failure(profile, RuntimeError("CUDA error: out of memory"))
    _expire(router, "p")
    router._record_success(profile)
    assert router._fault_strikes.get("p") is None
    assert router._oom_strikes.get("p") is None
    router.mark_failure(profile, RuntimeError("HTTP 502: bad gateway"))
    assert _remaining(router, "p") <= 45.0


def test_classified_failures_keep_their_own_cooldowns():
    """Quota backoff stays deterministic and does not count as a transient
    streak — only unclassified faults take the escalating path."""
    profile = _profile()
    router = _router(profile)
    router.mark_failure(profile, RuntimeError("HTTP 429: rate limit exceeded"))
    assert 4.0 < _remaining(router, "p") <= 5.0
    assert router._fault_strikes.get("p") is None


def test_breaker_state_cycles_open_half_open_closed():
    profile = _profile()
    router = _router(profile)
    assert router.breaker_state(profile) == "closed"
    router.mark_failure(profile, RuntimeError("HTTP 500: internal error"))
    assert router.breaker_state(profile) == "open"
    assert router.is_available(profile) is False
    _expire(router, "p")
    assert router.breaker_state(profile) == "half_open"
    assert router.is_available(profile) is True
    router._record_success(profile)
    assert router.breaker_state(profile) == "closed"


def test_an_in_flight_probe_moves_its_provider_to_the_back():
    a, b = _profile("a"), _profile("b")
    router = _router(a, b)
    router.set_connectivity(type("C", (), {"online": True, "is_online": lambda self: True})())
    router.mark_failure(a, RuntimeError("HTTP 500: internal error"))
    _expire(router, "a")
    order_before = [p.name for p in router.select_text_profiles("hello")]
    assert set(order_before) == {"a", "b"}
    with router._admit(a):
        # A concurrent turn tries the healthy provider first, but the probing
        # one is still offered rather than dropped.
        assert [p.name for p in router.select_text_profiles("hello")] == ["b", "a"]
        # Only one probe at a time: a nested admission is not a second probe.
        with router._admit(a):
            assert "a" in router._probing
        assert "a" in router._probing
    assert "a" not in router._probing


def test_a_healthy_provider_is_never_marked_as_probing():
    profile = _profile()
    router = _router(profile)
    with router._admit(profile):
        assert router._probing == set()


def test_breaker_state_is_reported_in_diagnostics_and_snapshot():
    profile = _profile()
    router = _router(profile)
    router.mark_failure(profile, RuntimeError("HTTP 500: internal error"))
    diag = router.diagnostics()
    assert diag["providers"][0]["breaker"] == "open"
    assert router.provider_snapshot()["breakers"] == {"p": "open"}


def test_a_real_transport_failure_then_recovery_closes_the_breaker():
    """End to end through the HTTP transport: a 503 opens the breaker, the
    recovery call runs as the half-open probe, and its success closes it."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    calls = {"n": 0}
    seen_probing = []

    async def handler(request):
        calls["n"] += 1
        seen_probing.append("p" in router._probing)
        if calls["n"] == 1:
            return web.Response(status=503, text="overloaded")
        return web.json_response({"choices": [{"message": {"content": "hello there"}}]})

    async def flow():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        server = TestServer(app)
        await server.start_server()
        try:
            profile.base_url = str(server.make_url("/v1"))
            try:
                await router._openai_compatible_chat(profile, "hi", instruction="be brief")
            except RuntimeError as exc:
                router.mark_failure(profile, exc)
            assert router.breaker_state(profile) == "open"
            _expire(router, "p")
            assert router.breaker_state(profile) == "half_open"
            text = await router._openai_compatible_chat(profile, "hi", instruction="be brief")
            assert text == "hello there"
        finally:
            await server.close()

    profile = _profile()
    router = _router(profile)
    asyncio.run(flow())
    # The first call was not a probe (the breaker was closed); the second was.
    assert seen_probing == [False, True]
    assert router.breaker_state(profile) == "closed"
    assert router._probing == set()


def test_routing_and_breaker_events_reach_the_dashboard_feed():
    """The Command Deck shows the provider answering, its latency, and a
    breaker opening, from structured events rather than log text."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def handler(request):
        return web.json_response({"choices": [{"message": {"content": "fine"}}]})

    async def flow():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        server = TestServer(app)
        await server.start_server()
        try:
            profile.base_url = str(server.make_url("/v1"))
            router.set_connectivity(type("C", (), {"is_online": lambda self: True})())
            await router.generate_text("hello", instruction="be brief")
        finally:
            await server.close()

    profile = _profile()
    router = _router(profile)
    asyncio.run(flow())
    routes = [p for n, p in router.bus.events if n == "dashboard_event" and p[0] == "provider_route"]
    assert routes and routes[-1][1]["provider"] == "p" and routes[-1][1]["fallbacks"] == 0
    assert routes[-1][1]["latency_s"] >= 0.0
    router.mark_failure(profile, RuntimeError("HTTP 500: internal error"))
    breakers = [p for n, p in router.bus.events if n == "dashboard_event" and p[0] == "provider_breaker"]
    assert breakers[-1][1]["provider"] == "p" and breakers[-1][1]["state"] == "open"


# ── background work waits out a short cooldown instead of failing ────────────

def test_wait_for_text_returns_when_a_cooling_provider_comes_back():
    router = _router(_profile("groq"))
    router._cooldowns["groq"] = time.monotonic() + 0.3
    started = time.monotonic()
    assert asyncio.run(router.wait_for_text(5.0)) is True
    assert 0.25 <= time.monotonic() - started < 2.0


def test_wait_for_text_gives_up_at_its_bound():
    router = _router(_profile("groq"))
    router._cooldowns["groq"] = time.monotonic() + 60.0
    started = time.monotonic()
    assert asyncio.run(router.wait_for_text(0.3)) is False
    assert time.monotonic() - started < 1.5


def test_wait_for_text_does_not_wait_when_nothing_can_come_back():
    router = _router(_profile("groq", api_key=""))
    started = time.monotonic()
    assert asyncio.run(router.wait_for_text(30.0)) is False
    assert time.monotonic() - started < 0.5


def test_the_alternate_that_answered_is_tried_first_next_session(monkeypatch):
    """Mark XXXX.4: a 5xx primary cost every session its timeout first; the
    alternate that answered now leads for a few hours (never the config)."""
    from orion_core import providers as mod

    profile = _profile(base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                       models=["model-a", "model-b"])
    router = _router(profile)
    monkeypatch.setattr(router, "_persist_model", lambda p, **_: None)
    calls = []

    async def call():
        calls.append(router.active_model(profile))
        if len(calls) == 1:
            raise RuntimeError("HTTP 503: primary is overloaded")
        return "ok"

    asyncio.run(router._call_with_heal(profile, call))
    assert _router(profile).active_model(profile) == "model-b"     # a new session
    assert profile.model == "model-a"
    monkeypatch.setattr(mod, "STICKY_MODEL_S", -1)
    router._remember_model(profile)
    assert _router(profile).active_model(profile) == "model-a"
