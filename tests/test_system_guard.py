"""
Tests for the destructive-action safety layer (Section 10).

Covers intent separation, ambiguity, token-bound confirmation, expiry, replay,
unrelated ("yes") replies, cancellation, the safe-by-default power primitives,
and the dispatcher gating that never executes an OS shutdown on model text.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.system_guard import (
    ActionIntent,
    DecisionKind,
    SystemActionGuard,
    classify_intent,
)
from orion_core.peripherals import PeripheralController


# ── intent classification ────────────────────────────────────────────────────

def test_bare_shutdown_is_ambiguous_not_os_shutdown():
    assert classify_intent("shut down") is ActionIntent.AMBIGUOUS
    assert classify_intent("shutdown") is ActionIntent.AMBIGUOUS


def test_explicit_os_shutdown():
    assert classify_intent("shut down the computer") is ActionIntent.OS_SHUTDOWN
    assert classify_intent("power off the pc") is ActionIntent.OS_SHUTDOWN
    assert classify_intent("turn off the computer") is ActionIntent.OS_SHUTDOWN
    assert classify_intent("power down the pc") is ActionIntent.OS_SHUTDOWN


def test_self_directed_shutdown_synonyms_close_orion():
    # A bare/self-addressed power command means ORION, never the PC.
    for phrase in ("power down", "turn off now please", "shut it down"):
        assert classify_intent(phrase) is ActionIntent.AMBIGUOUS, phrase
    for phrase in ("turn yourself off", "go offline", "power yourself down",
                   "shut yourself down"):
        assert classify_intent(phrase) in (
            ActionIntent.CLOSE_APP, ActionIntent.STOP_ORION), phrase


def test_device_commands_are_not_power_commands():
    # "turn off the lights" carries an object → a device request for the model,
    # NOT a command to shut ORION (or the PC) down.
    for phrase in ("turn off the lights", "turn off the music", "turn off the tv",
                   "shut down the download", "turn off notifications",
                   "restart the download"):
        assert classify_intent(phrase) is ActionIntent.UNKNOWN, phrase


def test_app_and_soft_stops_are_not_destructive():
    assert classify_intent("close orion") is ActionIntent.CLOSE_APP
    assert classify_intent("stop orion") is ActionIntent.STOP_ORION
    assert classify_intent("stop speaking") is ActionIntent.STOP_SPEAKING
    assert classify_intent("stop the task") is ActionIntent.STOP_TASK
    assert classify_intent("restart orion") is ActionIntent.STOP_ORION


def test_other_os_power_intents():
    assert classify_intent("restart the computer") is ActionIntent.OS_RESTART
    assert classify_intent("reboot the machine") is ActionIntent.OS_RESTART
    assert classify_intent("log out") is ActionIntent.OS_LOGOUT
    assert classify_intent("hibernate the computer") is ActionIntent.OS_HIBERNATE
    assert classify_intent("put the computer to sleep") is ActionIntent.OS_SLEEP


# ── guard flow ────────────────────────────────────────────────────────────────

def _guard():
    ticks = {"t": 0.0}
    guard = SystemActionGuard(machine="TEST-PC", ttl_s=30.0,
                              clock=lambda: ticks["t"])
    return guard, ticks


def test_ambiguous_command_asks_for_clarification():
    guard, _ = _guard()
    d = guard.evaluate("shut down")
    assert d.kind is DecisionKind.CLARIFY
    assert d.token is None


def test_destructive_requires_confirmation_and_token_release():
    guard, _ = _guard()
    d = guard.evaluate("shut down the computer")
    assert d.kind is DecisionKind.CONFIRM
    assert d.token
    assert "TEST-PC" in d.message
    ok = guard.confirm(d.token)
    assert ok.kind is DecisionKind.EXECUTE
    assert ok.intent is ActionIntent.OS_SHUTDOWN


def test_non_destructive_executes_without_token():
    guard, _ = _guard()
    d = guard.evaluate("stop speaking")
    assert d.kind is DecisionKind.EXECUTE
    assert d.token is None


def test_expired_confirmation_is_rejected():
    guard, ticks = _guard()
    d = guard.evaluate("restart the computer")
    ticks["t"] = 31.0  # past the 30s TTL
    rejected = guard.confirm(d.token)
    assert rejected.kind is DecisionKind.REJECT


def test_replay_is_rejected_token_is_single_use():
    guard, _ = _guard()
    d = guard.evaluate("shut down the computer")
    first = guard.confirm(d.token)
    assert first.kind is DecisionKind.EXECUTE
    replay = guard.confirm(d.token)
    assert replay.kind is DecisionKind.REJECT


def test_bare_yes_does_not_confirm_when_ambiguous_pending_count():
    guard, _ = _guard()
    # No pending action → a stray "yes" confirms nothing.
    assert guard.confirm_phrase("yes").kind is DecisionKind.REJECT
    # Two pending actions → "yes" cannot be bound safely.
    guard.evaluate("shut down the computer")
    guard.evaluate("restart the computer")
    assert guard.confirm_phrase("yes").kind is DecisionKind.REJECT


def test_yes_confirms_only_single_bound_pending():
    guard, _ = _guard()
    guard.evaluate("shut down the computer")
    d = guard.confirm_phrase("yes, go ahead")
    assert d.kind is DecisionKind.EXECUTE
    assert d.intent is ActionIntent.OS_SHUTDOWN


def test_cancellation_clears_pending():
    guard, _ = _guard()
    d = guard.evaluate("shut down the computer")
    guard.cancel(d.token)
    assert guard.pending_count() == 0
    assert guard.confirm(d.token).kind is DecisionKind.REJECT


def test_unknown_token_rejected():
    guard, _ = _guard()
    assert guard.confirm("not-a-real-token").kind is DecisionKind.REJECT


# ── safe-by-default primitives ────────────────────────────────────────────────

class _Log:
    def emit(self, *_a, **_k):
        pass


class _Bus:
    log = _Log()


def test_peripheral_power_primitives_are_safe_by_default():
    # These must return a ToolResult WITHOUT executing any OS command when not
    # explicitly confirmed — including the pre-existing test that calls
    # peripheral.shutdown() with no args.
    p = PeripheralController(_Bus())
    for res in (p.shutdown(), p.restart(), p.logout(), p.sleep(), p.hibernate()):
        assert res.ok is False
        assert "confirmation" in res.text.lower()


# ── dispatcher gating (no OS shutdown from model text) ────────────────────────

class _Signal:
    def __init__(self, sink, name):
        self._sink, self._name = sink, name

    def emit(self, *payload):
        self._sink.append((self._name, payload[0] if len(payload) == 1 else payload))


class _StubBus2:
    def __init__(self):
        self.emitted = []
        self.log = _Signal(self.emitted, "log")
        self.confirm_action = _Signal(self.emitted, "confirm_action")


class _RecordingPeripherals:
    def __init__(self):
        self.executed = []

    def shutdown(self, confirm=False):
        self.executed.append(("shutdown", confirm))
        from orion_core.data import ToolResult
        return ToolResult("host shutting down", ok=True)


def _dispatcher_with_guard(bus, peripherals):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.system_guard import SystemActionGuard
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = bus
    d.peripherals = peripherals
    d.system_guard = SystemActionGuard(machine="TEST", logger=lambda *_: None)
    return d


def test_dispatcher_shutdown_arms_but_does_not_execute():
    bus = _StubBus2()
    per = _RecordingPeripherals()
    disp = _dispatcher_with_guard(bus, per)
    res = disp.peripherals_tool({"action": "shutdown"})
    assert res.ok is False                      # not executed
    assert per.executed == []                   # OS command never ran
    # A confirmation prompt was surfaced to the UI carrying a token.
    prompts = [p for (n, p) in bus.emitted if n == "confirm_action"]
    assert prompts and prompts[0]["token"]
    # Releasing with the bound token executes exactly once, with confirm=True.
    token = prompts[0]["token"]
    ok = disp.confirm_system_action(token)
    assert ok.ok is True
    assert per.executed == [("shutdown", True)]
    # Replay of the same token does nothing.
    per.executed.clear()
    disp.confirm_system_action(token)
    assert per.executed == []
