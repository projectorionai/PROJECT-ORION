"""
Tests for ImprovementHeartbeat seeing the whole system: diagnostics FAIL/WARN
checks and the cognitive loop's awareness digest (both already broadcast on
bus.dashboard_event, previously read by nobody in forge.py) plus telemetry
tool-failure counters (accepted in the constructor, previously never read).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.forge import ImprovementHeartbeat
from orion_core.telemetry import Telemetry


class _Signal:
    def __init__(self):
        self._slots = []

    def emit(self, *payload):
        for slot in self._slots:
            slot(*payload)

    def connect(self, slot):
        self._slots.append(slot)


class _StubBus:
    def __init__(self):
        self.log = _Signal()
        self.dashboard_event = _Signal()

    def __getattr__(self, name):
        return _Signal()


class _StubForge:
    sessions: dict = {}


def _heartbeat(telemetry=None) -> ImprovementHeartbeat:
    bus = _StubBus()
    return ImprovementHeartbeat(bus, router=None, forge=_StubForge(), telemetry=telemetry)


# ── passive dashboard_event tap ─────────────────────────────────────────────

def test_diagnostics_event_is_cached(tmp_path, monkeypatch):
    from orion_core import forge as forge_module
    monkeypatch.setattr(forge_module, "SELF_IMPROVE_DIR", tmp_path)
    hb = _heartbeat()
    payload = {"at": "12:00:00", "fails": 1, "warns": 2,
               "checks": [{"name": "imports", "status": "FAIL", "detail": "broken module"}]}
    hb.bus.dashboard_event.emit("diagnostics", payload)
    assert hb._last_diagnostics == payload


def test_awareness_event_is_cached(tmp_path, monkeypatch):
    from orion_core import forge as forge_module
    monkeypatch.setattr(forge_module, "SELF_IMPROVE_DIR", tmp_path)
    hb = _heartbeat()
    payload = {"at": "12:00:00", "focus": "shipping", "active_project": "orion",
               "deadlines": ["report due Friday"]}
    hb.bus.dashboard_event.emit("awareness", payload)
    assert hb._last_awareness == payload


def test_unrelated_channel_is_ignored(tmp_path, monkeypatch):
    from orion_core import forge as forge_module
    monkeypatch.setattr(forge_module, "SELF_IMPROVE_DIR", tmp_path)
    hb = _heartbeat()
    hb.bus.dashboard_event.emit("briefing", "some briefing text")
    assert hb._last_diagnostics is None
    assert hb._last_awareness is None


def test_later_event_overwrites_the_cached_snapshot(tmp_path, monkeypatch):
    from orion_core import forge as forge_module
    monkeypatch.setattr(forge_module, "SELF_IMPROVE_DIR", tmp_path)
    hb = _heartbeat()
    hb.bus.dashboard_event.emit("diagnostics", {"fails": 1, "warns": 0, "checks": []})
    hb.bus.dashboard_event.emit("diagnostics", {"fails": 3, "warns": 1, "checks": []})
    assert hb._last_diagnostics["fails"] == 3


# ── _compose_context ─────────────────────────────────────────────────────────

def test_compose_context_omits_sections_with_no_evidence(tmp_path, monkeypatch):
    from orion_core import forge as forge_module
    monkeypatch.setattr(forge_module, "SELF_IMPROVE_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "IMPROVE_JOURNAL", tmp_path / "improve.jsonl")
    hb = _heartbeat()
    context = hb._compose_context()
    assert "LATEST DIAGNOSTICS" not in context
    assert "AWARENESS SNAPSHOT" not in context
    assert "TELEMETRY" not in context


def test_compose_context_includes_diagnostics_fails_and_warns(tmp_path, monkeypatch):
    from orion_core import forge as forge_module
    monkeypatch.setattr(forge_module, "SELF_IMPROVE_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "IMPROVE_JOURNAL", tmp_path / "improve.jsonl")
    hb = _heartbeat()
    hb.bus.dashboard_event.emit("diagnostics", {
        "at": "12:00:00", "fails": 1, "warns": 1,
        "checks": [
            {"name": "imports", "status": "FAIL", "detail": "orion_core.foo broken"},
            {"name": "resources", "status": "WARN", "detail": "disk 90% full"},
            {"name": "compile", "status": "PASS", "detail": "all modules compile"},
        ],
    })
    context = hb._compose_context()
    assert "LATEST DIAGNOSTICS" in context
    assert "orion_core.foo broken" in context
    assert "disk 90% full" in context
    assert "all modules compile" not in context   # PASS checks are noise here


def test_compose_context_includes_awareness_focus_and_deadlines(tmp_path, monkeypatch):
    from orion_core import forge as forge_module
    monkeypatch.setattr(forge_module, "SELF_IMPROVE_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "IMPROVE_JOURNAL", tmp_path / "improve.jsonl")
    hb = _heartbeat()
    hb.bus.dashboard_event.emit("awareness", {
        "at": "12:00:00", "focus": "shipping the chess feature",
        "active_project": "orion", "deadlines": ["ship chess by Friday"],
    })
    context = hb._compose_context()
    assert "AWARENESS SNAPSHOT" in context
    assert "shipping the chess feature" in context
    assert "ship chess by Friday" in context


def test_compose_context_includes_telemetry_tool_failure_counts(tmp_path, monkeypatch):
    from orion_core import forge as forge_module
    monkeypatch.setattr(forge_module, "SELF_IMPROVE_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "IMPROVE_JOURNAL", tmp_path / "improve.jsonl")
    telemetry = Telemetry(_StubBus())
    for _ in range(5):
        telemetry.metrics.incr("tool.calls")
        telemetry.metrics.incr("tool.security_recon.calls")
    for _ in range(3):
        telemetry.metrics.incr("tool.failures")
        telemetry.metrics.incr("tool.security_recon.failures")
    hb = _heartbeat(telemetry=telemetry)
    context = hb._compose_context()
    assert "TELEMETRY" in context
    assert "security_recon" in context
    assert "3" in context


def test_compose_context_telemetry_section_never_faults_on_a_bad_telemetry_object(tmp_path, monkeypatch):
    from orion_core import forge as forge_module
    monkeypatch.setattr(forge_module, "SELF_IMPROVE_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(forge_module, "IMPROVE_JOURNAL", tmp_path / "improve.jsonl")

    class _BrokenTelemetry:
        @property
        def metrics(self):
            raise RuntimeError("boom")

    hb = _heartbeat(telemetry=_BrokenTelemetry())
    context = hb._compose_context()   # must not raise
    assert "NO EVIDENCE" in context or context
