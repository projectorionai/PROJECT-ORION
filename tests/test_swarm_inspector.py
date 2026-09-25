"""
Tests for orion_core.swarm.inspect and the deck's persistent view state
(Mark XXII, Phase 4).

The rule these mostly defend: an ABSENT measurement must never render as 0.
"0 ms latency" and "this node has never reported latency" are different facts,
and a monitoring surface that conflates them lies to the person reading it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.inspect import (  # noqa: E402
    ABSENT,
    build_report,
    report_from_dict,
)
from orion_core.swarm.model import (  # noqa: E402
    Activity,
    EdgeKind,
    Health,
    NodeKind,
    NodeTelemetry,
    SwarmEdge,
    SwarmNode,
    SwarmSnapshot,
)

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.gui.swarm_view import SwarmDeckView  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _Signal:
    def __init__(self): self._slots = []
    def connect(self, slot): self._slots.append(slot)
    def emit(self, *a, **k):
        for s in self._slots:
            s(*a, **k)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def _node(**kw) -> SwarmNode:
    base = dict(id="module:memory", kind=NodeKind.SUBSYSTEM, label="memory",
                cluster=cl.MEMORY)
    base.update(kw)
    return SwarmNode(**base)


def _rows(report) -> dict[str, str]:
    return {row.label: row.value for row in report.rows}


# ── absent vs zero ───────────────────────────────────────────────────────────

def test_a_node_with_no_latency_data_shows_a_dash_not_zero():
    rows = _rows(build_report(_node()))
    assert rows["Latency p50"] == ABSENT
    assert rows["Latency p95"] == ABSENT


def test_a_never_called_node_shows_a_dash_for_failures():
    """'Called 0 times' and 'called and never failed' are different facts."""
    assert _rows(build_report(_node()))["Failures"] == ABSENT


def test_a_called_node_shows_a_real_failure_count():
    node = _node(telemetry=NodeTelemetry(calls=10, failures=2))
    assert "2" in _rows(build_report(node))["Failures"]


def test_absent_queue_and_tokens_show_a_dash():
    rows = _rows(build_report(_node()))
    assert rows["Queue"] == ABSENT
    assert rows["Tokens"] == ABSENT


def test_a_zero_queue_is_reported_as_zero_not_absent():
    node = _node(telemetry=NodeTelemetry(queue_depth=0))
    assert _rows(build_report(node))["Queue"] == "0"


# ── formatting ───────────────────────────────────────────────────────────────

def test_latency_is_formatted_in_milliseconds():
    node = _node(telemetry=NodeTelemetry(calls=1, latency_p50_ms=12.34))
    assert _rows(build_report(node))["Latency p50"] == "12.3 ms"


def test_a_very_slow_call_is_formatted_in_seconds():
    node = _node(telemetry=NodeTelemetry(calls=1, latency_max_ms=4200.0))
    assert _rows(build_report(node))["Latency max"] == "4.20 s"


def test_large_request_counts_are_thousands_separated():
    node = _node(telemetry=NodeTelemetry(calls=1234567))
    assert _rows(build_report(node))["Requests"] == "1,234,567"


def test_heartbeat_age_is_humanised():
    fresh = _node(telemetry=NodeTelemetry(heartbeat_age_s=12.0))
    old = _node(telemetry=NodeTelemetry(heartbeat_age_s=7200.0))
    assert _rows(build_report(fresh))["Last heartbeat"] == "12s ago"
    assert "h ago" in _rows(build_report(old))["Last heartbeat"]


def test_a_never_beating_component_shows_a_dash():
    node = _node(telemetry=NodeTelemetry(heartbeat_age_s=-1.0))
    assert _rows(build_report(node))["Last heartbeat"] == ABSENT


def test_the_error_rate_is_shown_as_a_percentage():
    node = _node(telemetry=NodeTelemetry(calls=10, failures=3))
    assert "30%" in _rows(build_report(node))["Failures"]


# ── severity ─────────────────────────────────────────────────────────────────

def test_a_down_node_marks_its_health_row_as_an_alarm():
    report = build_report(_node(health=Health.DOWN))
    health = next(r for r in report.rows if r.label == "Health")
    assert health.severity == "alarm"


def test_a_healthy_node_has_no_alarm_rows():
    report = build_report(_node(health=Health.OK, activity=Activity.IDLE))
    assert all(r.severity == "normal" for r in report.rows)


def test_a_failing_activity_marks_an_alarm():
    report = build_report(_node(activity=Activity.ERROR))
    activity = next(r for r in report.rows if r.label == "Activity")
    assert activity.severity == "alarm"


def test_a_high_error_rate_marks_the_failures_row():
    node = _node(telemetry=NodeTelemetry(calls=20, failures=15))
    failures = next(r for r in build_report(node).rows if r.label == "Failures")
    assert failures.severity == "alarm"


def test_a_low_error_rate_is_not_an_alarm():
    node = _node(telemetry=NodeTelemetry(calls=100, failures=1))
    failures = next(r for r in build_report(node).rows if r.label == "Failures")
    assert failures.severity == "normal"


# ── context ──────────────────────────────────────────────────────────────────

def test_dependencies_and_capabilities_are_listed():
    node = _node(dependencies=("memory", "bus"),
                 capabilities=("read", "write"))
    rows = _rows(build_report(node))
    assert "memory" in rows["Depends on"]
    assert "read" in rows["Capabilities"]


def test_a_long_capability_list_is_truncated_with_a_count():
    node = _node(capabilities=tuple(f"tool{i}" for i in range(20)))
    assert "+12 more" in _rows(build_report(node))["Capabilities"]


def test_neighbours_are_collected_from_both_edge_directions():
    node = _node(id="module:memory")
    snapshot = SwarmSnapshot(
        nodes=(node,),
        edges=(SwarmEdge("cluster:MEMORY", "module:memory"),
               SwarmEdge("module:memory", "module:bus", EdgeKind.DEPENDENCY)))
    report = build_report(node, snapshot)
    assert set(report.neighbours) == {"cluster:MEMORY", "module:bus"}


def test_neighbours_are_deduplicated():
    node = _node(id="a")
    snapshot = SwarmSnapshot(
        nodes=(node,),
        edges=(SwarmEdge("a", "b", EdgeKind.MEMBERSHIP),
               SwarmEdge("a", "b", EdgeKind.DEPENDENCY)))
    assert build_report(node, snapshot).neighbours == ("b",)


# ── actions ──────────────────────────────────────────────────────────────────

def test_an_agent_offers_opening_its_workspace():
    node = _node(id="agent:coding", kind=NodeKind.AGENT, label="Coding Agent")
    report = build_report(node, can_open_agent=True)
    assert report.action_label == "Open workspace"
    assert report.action_target == "coding"


def test_an_agent_without_a_handler_offers_no_action():
    node = _node(id="agent:coding", kind=NodeKind.AGENT)
    assert build_report(node, can_open_agent=False).action_target is None


def test_a_plain_subsystem_offers_no_action():
    assert build_report(_node()).action_target is None


def test_the_report_renders_as_text():
    node = _node(label="memory", telemetry=NodeTelemetry(calls=5))
    text = build_report(node).as_text()
    assert "memory" in text
    assert "Requests" in text


# ── the legacy dict path uses the same formatter ─────────────────────────────

def test_a_dict_node_produces_the_same_shaped_report():
    report = report_from_dict({"id": "agent:coding", "kind": "agent",
                               "label": "Coding Agent", "cluster": "DEVELOPMENT",
                               "health": "OK"}, can_open_agent=True)
    assert report.title == "Coding Agent"
    assert report.action_target == "coding"


def test_a_malformed_dict_never_raises():
    report = report_from_dict({"id": "x", "kind": "nonsense",
                               "health": "???", "activity": "???"})
    assert report.title == "x"


def test_a_dict_carrying_telemetry_renders_it():
    report = report_from_dict({"id": "m", "kind": "subsystem", "label": "m",
                               "telemetry": {"calls": 9, "latency_p95_ms": 3.5}})
    rows = _rows(report)
    assert rows["Requests"] == "9"
    assert rows["Latency p95"] == "3.5 ms"


# ── the widget renders it, and keeps state ───────────────────────────────────

def test_the_inspector_shows_telemetry_rows(_app):
    view = SwarmDeckView(_StubBus())
    view._on_swarm_event({"id": "module:memory", "kind": "subsystem",
                          "label": "memory", "cluster": "MEMORY",
                          "health": "OK",
                          "telemetry": {"calls": 42, "latency_p50_ms": 7.0}})
    text = view.inspector_detail.text()
    assert "42" in text
    assert "7.0 ms" in text


def test_the_inspector_flags_an_alarm_row(_app):
    view = SwarmDeckView(_StubBus())
    view._on_swarm_event({"id": "module:vision", "kind": "subsystem",
                          "label": "vision", "health": "DOWN"})
    assert "!" in view.inspector_detail.text()


def test_view_state_round_trips_the_selection(_app):
    view = SwarmDeckView(_StubBus())
    view._on_swarm_event({"id": "agent:coding", "kind": "agent",
                          "label": "Coding Agent"})
    assert view.view_state()["selected"] == "agent:coding"


def test_restoring_a_malformed_state_is_safe(_app):
    view = SwarmDeckView(_StubBus())
    view.restore_view_state({"camera": "not a dict", "selected": 12345})
    view.restore_view_state(None)          # must not raise


def test_the_deck_keeps_pages_alive_across_a_tab_switch(_app):
    """'Changing tabs should never destroy objects.' UnifiedDashboard uses a
    QStackedWidget, so a hidden page keeps its widget, scene and state — this
    pins that rather than assuming it."""
    from PyQt6.QtWidgets import QLabel

    from orion_core.gui.unified_dashboard import UnifiedDashboard

    swarm = SwarmDeckView(_StubBus())
    deck = UnifiedDashboard(_StubBus(), [("SWARM", swarm), ("OTHER", QLabel("x"))])
    swarm._on_swarm_event({"id": "agent:coding", "kind": "agent", "label": "C"})
    deck.show_page_named("OTHER")
    deck.show_page_named("SWARM")
    assert swarm.parent() is not None                  # never destroyed
    assert swarm.view_state()["selected"] == "agent:coding"   # state survived
