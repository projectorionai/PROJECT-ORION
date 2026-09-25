"""
Regression tests for self-repair health recovery.

The reported fault: a Gemini Live WebSocket 1011 ("Deadline expired") left the
self-repair subsystem pinned to DEGRADED forever.  These tests lock in the
model that DEGRADED means "an unresolved *code* fault exists" — transient
live-channel drops are recorded but never degrade, and they clear on reconnect.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.selfrepair import SelfRepairAgent


class _Signal:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Health:
    def __init__(self):
        self.status: dict[str, tuple[str, str]] = {}

    def register(self, name, **k):
        pass

    def beat(self, name, status="OK", detail="", ttl=0.0):
        self.status[name] = (status, detail)


class _Metrics:
    def incr(self, *a, **k):
        pass


class _Log:
    def recent(self, limit=25):
        return []


class _Telemetry:
    def __init__(self):
        self.health = _Health()
        self.metrics = _Metrics()
        self.log = _Log()


def _agent(monkeypatch):
    monkeypatch.setenv("ORION_AUTOREPAIR", "0")
    return SelfRepairAgent(_StubBus(), telemetry=_Telemetry(), router=None)


def _capture(agent, exc):
    try:
        raise exc
    except type(exc) as e:
        return agent.capture(type(e), e, e.__traceback__)


def test_transient_1011_is_not_degraded(monkeypatch):
    agent = _agent(monkeypatch)
    inc = _capture(agent, ConnectionError(
        "received 1011 (internal error) Deadline expired before operation could complete"))
    assert inc.transient is True
    assert agent.telemetry.health.status["self_repair"][0] == "OK"


def test_real_code_fault_degrades(monkeypatch):
    agent = _agent(monkeypatch)
    _capture(agent, ValueError("a genuine defect in ORION's own logic"))
    assert agent.telemetry.health.status["self_repair"][0] == "DEGRADED"


def test_transient_clears_to_ok_on_reconnect(monkeypatch):
    agent = _agent(monkeypatch)
    _capture(agent, ConnectionError("1011 deadline expired"))
    assert agent.telemetry.health.status["self_repair"][0] == "OK"
    # The live-channel comes back → the transient incident is resolved.
    agent._on_connection_state({"state": "connected"})
    inc = agent.latest()
    assert inc.resolved is True
    assert agent.telemetry.health.status["self_repair"] == ("OK", "healthy — no open incidents")


def test_reconnect_does_not_hide_a_real_fault(monkeypatch):
    agent = _agent(monkeypatch)
    _capture(agent, ValueError("real fault"))
    _capture(agent, ConnectionError("1011 deadline expired"))
    assert agent.telemetry.health.status["self_repair"][0] == "DEGRADED"
    # Reconnect clears the transient, but the real fault must keep it DEGRADED.
    agent._on_connection_state({"state": "connected"})
    assert agent.telemetry.health.status["self_repair"][0] == "DEGRADED"


def test_informational_record_incident_does_not_degrade(monkeypatch):
    agent = _agent(monkeypatch)
    agent.record_incident("ForgeFailure", "could not synthesise tool")
    assert agent.telemetry.health.status["self_repair"][0] == "OK"
