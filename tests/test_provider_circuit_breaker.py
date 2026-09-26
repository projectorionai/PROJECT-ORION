"""
The provider router's circuit breaker.

* Transient faults (HTTP 5xx, timeouts, refused connections) escalate: each
  fault in a row triples the cooldown from 5 s to a 10 minute ceiling, with
  downward jitter. A flat 45 s retried a dead endpoint for the whole session.
* The first transient fault is retried once after a short pause (not a
  timeout), so one overload blip does not fail the turn.
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
    expected = [5.0, 15.0, 45.0, 135.0, 405.0, 600.0, 600.0]
    for ceiling in expected:
        router.mark_failure(profile, RuntimeError("HTTP 503: upstream unavailable"))
        remaining = _remaining(router, "p")
        # Jitter only shortens the wait (never past the ceiling).
        assert 0.8 * ceiling - 1.0 < remaining <= ceiling
        _expire(router, "p")


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
    assert _remaining(router, "p") <= 5.0


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


# ── one quick retry after a first blip ─────────────────────────────────────

def _counting_call(outcomes):
    calls = []

    async def call():
        calls.append(1)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    return call, calls


def test_a_first_blip_is_retried_once_and_the_turn_succeeds():
    profile = _profile()
    router = _router(profile)
    call, calls = _counting_call([RuntimeError("HTTP 503: overloaded"), "answer"])
    assert asyncio.run(router._call_with_heal(profile, call)) == "answer"
    assert len(calls) == 2
    router._record_success(profile)        # the transport records it on success
    assert router.breaker_state(profile) == "closed"
    assert router.is_available(profile), "a provider that just answered is still cooling"


def test_a_timeout_is_not_retried():
    """Retrying a timeout would double an already long wait."""
    profile = _profile()
    router = _router(profile)
    call, calls = _counting_call([asyncio.TimeoutError(), "never"])
    try:
        asyncio.run(router._call_with_heal(profile, call))
    except asyncio.TimeoutError:
        pass
    assert len(calls) == 1


def test_a_provider_already_failing_is_not_retried_again():
    profile = _profile()
    router = _router(profile)
    router.mark_failure(profile, RuntimeError("HTTP 500: internal error"))
    _expire(router, "p")
    call, calls = _counting_call([RuntimeError("HTTP 500: internal error"), "never"])
    try:
        asyncio.run(router._call_with_heal(profile, call))
    except RuntimeError:
        pass
    assert len(calls) == 1


def test_a_quota_failure_is_not_given_the_blip_retry():
    """429s have their own rotation and cooldown; the blip retry is for
    transient faults only."""
    profile = _profile()
    router = _router(profile)
    call, calls = _counting_call([RuntimeError("HTTP 429: rate limit exceeded"), "never"])
    try:
        asyncio.run(router._call_with_heal(profile, call))
    except RuntimeError:
        pass
    assert len(calls) == 1
