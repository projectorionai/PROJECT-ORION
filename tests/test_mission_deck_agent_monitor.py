"""
Tests for the Mission Deck AGENT MONITOR fix (Mark XX design-spec, Critical
item 1): the dock titled AGENT MONITOR used to render
telemetry.health.snapshot() (component heartbeats) because its own
self.agents constructor argument was assigned to self.agents and never read
again anywhere in the file — a label promising agent data while actually
showing something else. Fixed by splitting into two honestly-named docks:
AGENT MONITOR (the real specialist roster, via AgentManager.describe()) and
COMPONENT HEALTH (the telemetry data that used to hide under the wrong
name).

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.gui.mission_deck import MissionDeckView


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubAgents:
    def describe(self):
        return [
            {"name": "coding", "title": "Coding Agent", "focus": "software engineering",
             "calls": 4, "last_active": "12:03:00"},
            {"name": "fashion", "title": "Fashion Agent", "focus": "styling",
             "calls": 0, "last_active": ""},
        ]


class _StubHealth:
    def snapshot(self):
        return [{"name": "cognition", "status": "OK"}, {"name": "forge", "status": "DEGRADED"}]


class _StubTelemetry:
    health = _StubHealth()


def test_agent_monitor_and_component_health_are_separate_panels(_app):
    view = MissionDeckView(_StubBus(), agents=_StubAgents(), telemetry=_StubTelemetry())
    assert "AGENT MONITOR" in view._panels
    assert "COMPONENT HEALTH" in view._panels
    assert view._panels["AGENT MONITOR"] is not view._panels["COMPONENT HEALTH"]


def test_agent_monitor_renders_the_real_agent_roster(_app):
    view = MissionDeckView(_StubBus(), agents=_StubAgents())
    text = view._render_agent_roster()
    assert "coding" in text
    assert "software engineering" in text
    assert "4 call(s)" in text
    assert "not yet consulted" in text   # the fashion agent, zero calls


def test_agent_monitor_degrades_cleanly_with_no_agent_manager(_app):
    view = MissionDeckView(_StubBus())
    assert view._render_agent_roster() == "Agent manager offline."


def test_agent_monitor_reports_empty_roster_cleanly(_app):
    class _EmptyAgents:
        def describe(self):
            return []

    view = MissionDeckView(_StubBus(), agents=_EmptyAgents())
    assert view._render_agent_roster() == "No specialist agents registered."


def test_component_health_still_renders_telemetry_data(_app):
    view = MissionDeckView(_StubBus(), telemetry=_StubTelemetry())
    text = view._render_component_health()
    assert "cognition" in text
    assert "OK" in text
    assert "forge" in text
    assert "DEGRADED" in text


def test_component_health_degrades_cleanly_with_no_telemetry(_app):
    view = MissionDeckView(_StubBus())
    assert view._render_component_health() == "Telemetry offline."


def test_refresh_updates_both_panels_independently(_app):
    view = MissionDeckView(_StubBus(), agents=_StubAgents(), telemetry=_StubTelemetry())
    view.show()
    view._refresh()
    assert "coding" in view._panels["AGENT MONITOR"].toPlainText()
    assert "cognition" in view._panels["COMPONENT HEALTH"].toPlainText()
    view.hide()


# ── Awareness strip (Critical item 2) ────────────────────────────────────────

class _StubCognitiveLoop:
    def __init__(self, digest):
        self._digest = digest

    def last_digest(self):
        return self._digest


def test_awareness_strip_shows_a_placeholder_before_any_digest(_app):
    view = MissionDeckView(_StubBus())
    assert "building situational context" in view._awareness_label.text()


def test_awareness_strip_seeds_from_cognitive_loop_last_digest_at_construction(_app):
    digest = {"focus": "orion_core/app.py", "active_project": "ORION Main", "deadlines": ["x"]}
    view = MissionDeckView(_StubBus(), cognitive_loop=_StubCognitiveLoop(digest))
    text = view._awareness_label.text()
    assert "orion_core/app.py" in text
    assert "ORION Main" in text
    assert "1 deadline" in text


def test_awareness_strip_updates_live_on_the_awareness_channel(_app):
    view = MissionDeckView(_StubBus())
    view._on_event("awareness", {"focus": "chess_view.py", "active_project": "", "deadlines": []})
    assert "chess_view.py" in view._awareness_label.text()


def test_awareness_channel_does_not_also_flood_the_activity_feed(_app):
    view = MissionDeckView(_StubBus())
    before = view._panels["ACTIVITY FEED"].toPlainText()
    view._on_event("awareness", {"focus": "x", "active_project": "", "deadlines": []})
    assert view._panels["ACTIVITY FEED"].toPlainText() == before


def test_awareness_strip_handles_an_empty_digest_gracefully(_app):
    view = MissionDeckView(_StubBus())
    view._on_event("awareness", {})
    assert "no active focus recorded" in view._awareness_label.text()


def test_cognitive_loop_seed_failure_never_crashes_construction(_app):
    class _BrokenLoop:
        def last_digest(self):
            raise RuntimeError("boom")

    view = MissionDeckView(_StubBus(), cognitive_loop=_BrokenLoop())   # must not raise
    assert "AWARENESS" in view._awareness_label.text()
