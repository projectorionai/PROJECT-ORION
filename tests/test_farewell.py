"""
Tests for the Mark XXI farewell work — GenAILiveWorker.compose_farewell()
(time-varying, honorific-aware) and its wiring into the voice-initiated
self-shutdown path (_handle_power_command), which sets _farewell_spoken so
app.py's teardown (separately, not unit-tested here — see its own inline
reasoning) doesn't speak a second goodbye.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.live_worker as lw
from orion_core.live_worker import GenAILiveWorker


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubSpeech:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak_text(self, text: str) -> None:
        self.spoken.append(text)


class _StubIdentity:
    def __init__(self, honorific: str = "sir", frequency: str = "occasional") -> None:
        self.preferences = {"honorific": honorific, "honorific_frequency": frequency}


def _worker(identity=None) -> types.SimpleNamespace:
    w = types.SimpleNamespace()
    w.dispatcher = types.SimpleNamespace(identity=identity)
    w.bus = _StubBus()
    w.speech = _StubSpeech()
    w._say = lambda text: w.speech.speak_text(text)
    w.compose_farewell = lambda: GenAILiveWorker.compose_farewell(w)
    # _handle_power_command now resolves quiet-mode phrases first (so "stand
    # by" cannot be mistaken for a shutdown). Wire the real implementation in
    # rather than stubbing it out, so these tests would still catch a
    # quiet-mode pattern that wrongly swallowed a genuine power command.
    w.quiet_mode = False
    w._QUIET_ON_RE = GenAILiveWorker._QUIET_ON_RE
    w._QUIET_OFF_RE = GenAILiveWorker._QUIET_OFF_RE
    w._handle_quiet_command = lambda lowered: GenAILiveWorker._handle_quiet_command(
        w, lowered)
    return w


class _FixedDatetime(datetime):
    _fixed_hour = 12

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, cls._fixed_hour, 0, 0)


def _at_hour(monkeypatch, hour: int) -> None:
    fixed = type("_F", (_FixedDatetime,), {"_fixed_hour": hour})
    monkeypatch.setattr(lw, "datetime", fixed)


# ── compose_farewell() ───────────────────────────────────────────────────────
#
# These assertions moved from ONE exact string to the property that string was
# standing in for. "he should variate his goodbyes" means there is no longer a
# single correct sentence — a fixed sign-off is the most machine-like thing an
# assistant can do, since it is the last thing you hear every single time. What
# still has to hold is that a night farewell is nightly, a daytime one is not,
# and the honorific rules are obeyed in every option.

import orion_core.live_worker as _lw


def _is_night(text: str) -> bool:
    return "night" in text.lower() or "sleep" in text.lower()

def test_daytime_farewell_is_goodbye_or_see_you_later(monkeypatch):
    _at_hour(monkeypatch, 14)
    w = _worker(_StubIdentity())
    text = GenAILiveWorker.compose_farewell(w)
    assert text in [line.format(h=", sir") for line in _lw._FAREWELLS_DAY], text


def test_late_night_farewell_is_a_night_farewell(monkeypatch):
    _at_hour(monkeypatch, 23)
    w = _worker(_StubIdentity())
    for _ in range(len(_lw._FAREWELLS_LATE) + 2):
        text = GenAILiveWorker.compose_farewell(w)
        assert _is_night(text), text
        assert "sir" in text


def test_early_hours_farewell_is_also_a_night_farewell(monkeypatch):
    _at_hour(monkeypatch, 3)
    w = _worker(_StubIdentity())
    assert _is_night(GenAILiveWorker.compose_farewell(w))


def test_boundary_hour_5am_is_daytime_not_night(monkeypatch):
    _at_hour(monkeypatch, 5)
    w = _worker(_StubIdentity())
    text = GenAILiveWorker.compose_farewell(w)
    assert not _is_night(text), text
    assert text in [line.format(h=", sir") for line in _lw._FAREWELLS_DAY], text


def test_boundary_hour_22_is_night(monkeypatch):
    _at_hour(monkeypatch, 22)
    w = _worker(_StubIdentity())
    assert _is_night(GenAILiveWorker.compose_farewell(w))


def test_honorific_frequency_never_drops_the_honorific(monkeypatch):
    _at_hour(monkeypatch, 12)
    w = _worker(_StubIdentity(frequency="never"))
    text = GenAILiveWorker.compose_farewell(w)
    assert text in [line.format(h="") for line in _lw._FAREWELLS_DAY], text
    # Check for the honorific itself. A comma-based proxy fails on farewells
    # that legitimately contain one ("Okay, I'll see you soon.").
    assert "sir" not in text.lower(), f"honorific leaked in: {text}"
    assert "sir" not in text


def test_honorific_frequency_always_still_includes_it(monkeypatch):
    _at_hour(monkeypatch, 22)
    w = _worker(_StubIdentity(frequency="always"))
    assert ", sir" in GenAILiveWorker.compose_farewell(w)


def test_custom_honorific_is_used(monkeypatch):
    _at_hour(monkeypatch, 22)
    w = _worker(_StubIdentity(honorific="boss"))
    assert ", boss" in GenAILiveWorker.compose_farewell(w)


def test_no_identity_attached_degrades_to_no_honorific(monkeypatch):
    _at_hour(monkeypatch, 22)
    w = _worker(identity=None)
    text = GenAILiveWorker.compose_farewell(w)
    assert _is_night(text), text
    assert "sir" not in text.lower(), f"honorific leaked in: {text}"


def test_broken_identity_preferences_never_raises(monkeypatch):
    _at_hour(monkeypatch, 22)

    class _Broken:
        preferences = None   # .get() on this will raise AttributeError

    w = _worker(_Broken())
    text = GenAILiveWorker.compose_farewell(w)
    assert _is_night(text), text
    assert text in [line.format(h="") for line in _lw._FAREWELLS_LATE], text


# ── wired into the voice self-shutdown path ─────────────────────────────────
#
# A power command now ASKS before acting:
#
#   "When ORION shuts down I want him to confirm the shut down with me and see
#    if I meant it and then if I confirm, before shutting down he should say
#    [goodbye] then proceed."
#
# So the farewell no longer lands on the power command itself — it lands on the
# CONFIRMATION. These use a real instance rather than a SimpleNamespace,
# because the flow now spans several of the class's own methods.


def _real_worker(identity):
    """A GenAILiveWorker with only the collaborators this flow touches."""
    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    spoken: list[str] = []
    worker._say = spoken.append
    worker.speech = types.SimpleNamespace(spoken=spoken, is_busy=lambda: False)
    worker.bus = types.SimpleNamespace(
        log=types.SimpleNamespace(emit=lambda m: None),
        request_shutdown=types.SimpleNamespace(emit=lambda: None),
        request_restart=types.SimpleNamespace(emit=lambda: None))
    worker.dispatcher = types.SimpleNamespace(identity=identity, system_guard=None)
    worker._pending_shutdown = None
    worker._last_farewell = ""
    worker._farewell_spoken = False
    return worker, spoken


def test_a_shutdown_command_asks_before_saying_goodbye(monkeypatch):
    _at_hour(monkeypatch, 22)
    worker, spoken = _real_worker(_StubIdentity())

    async def _scenario():
        assert await GenAILiveWorker._handle_power_command(worker, "stop orion") is True

    asyncio.run(_scenario())
    assert getattr(worker, "_farewell_spoken", False) is False, (
        "said goodbye before checking the user meant it")
    assert "?" in spoken[-1], spoken


def test_confirming_then_speaks_the_farewell(monkeypatch):
    _at_hour(monkeypatch, 22)
    worker, spoken = _real_worker(_StubIdentity())

    async def _scenario():
        await GenAILiveWorker._handle_power_command(worker, "stop orion")

    asyncio.run(_scenario())
    worker._resolve_shutdown_confirmation("yes")
    assert worker._farewell_spoken is True
    assert _is_night(spoken[-1]), spoken[-1]
    assert "sir" in spoken[-1]


def test_restart_does_not_set_farewell_spoken(monkeypatch):
    _at_hour(monkeypatch, 12)
    worker, spoken = _real_worker(_StubIdentity())

    async def _scenario():
        assert await GenAILiveWorker._handle_power_command(
            worker, "restart orion") is True

    asyncio.run(_scenario())
    worker._resolve_shutdown_confirmation("yes")
    assert getattr(worker, "_farewell_spoken", False) is False
    assert "Restarting now" in spoken[-1]
