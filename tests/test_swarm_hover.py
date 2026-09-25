"""
Hover diagnostics (Mark XXIII).

The behaviours worth pinning are the ones that make a hover readout usable
rather than annoying: it must not open while the pointer is merely passing
over things, it must open fast once the user has clearly asked for detail,
and it must never print a measurement ORION cannot actually make.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm.hover import (  # noqa: E402
    HOVER_DELAY_S,
    HoverController,
    diagnostic_rows,
)
from orion_core.swarm.model import (  # noqa: E402
    Activity,
    Health,
    NodeKind,
    NodeTelemetry,
    SwarmNode,
)


class _Clock:
    def __init__(self) -> None: self.t = 100.0
    def __call__(self) -> float: return self.t
    def advance(self, dt: float) -> None: self.t += dt


class _Load:
    cpu_percent = 41.0
    gpu_percent = 12.0
    mem_percent = 63.0
    self_cpu_percent = 8.0


def _node(node_id="module:vision", kind=NodeKind.SUBSYSTEM, **kw):
    fields = dict(
        id=node_id, kind=kind, label=kw.pop("label", node_id.split(":")[-1]),
        cluster=kw.pop("cluster", "SYSTEM"),
        health=kw.pop("health", Health.OK),
        activity=kw.pop("activity", Activity.IDLE),
    )
    fields.update(kw)
    return SwarmNode(**fields)


def _values(rows):
    return {r.label: r.value for r in rows}


# ── dwell: passing over must not open anything ───────────────────────────────

def test_nothing_opens_the_instant_the_pointer_arrives():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("module:vision")
    assert hover.unfold() == 0.0
    assert hover.open_node() is None


def test_it_opens_once_the_pointer_rests():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("module:vision")
    clock.advance(HOVER_DELAY_S + 0.5)
    assert hover.unfold() == 1.0
    assert hover.open_node() == "module:vision"


def test_sweeping_across_many_nodes_opens_none_of_them():
    """The reason the dwell gate exists — a drag across a dense field crosses
    dozens of nodes and must not strobe a panel for each."""
    clock = _Clock()
    hover = HoverController(clock=clock)
    for i in range(30):
        hover.point_at(f"module:{i}")
        clock.advance(HOVER_DELAY_S / 4.0)
        assert hover.unfold() == 0.0


def test_pointing_at_the_same_node_again_is_not_a_change():
    hover = HoverController(clock=_Clock())
    assert hover.point_at("core") is True
    assert hover.point_at("core") is False


# ── warm swap: comparing two nodes must be fluid ─────────────────────────────

def test_the_second_node_opens_without_the_delay():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("a")
    clock.advance(HOVER_DELAY_S + 0.5)
    assert hover.open_node() == "a"

    hover.point_at("b")
    clock.advance(0.001)
    assert hover.open_node() == "b"


def test_the_delay_returns_once_the_pointer_has_been_away_a_while():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("a")
    clock.advance(HOVER_DELAY_S + 0.5)
    hover.unfold()
    hover.clear()
    clock.advance(5.0)                      # gone cold
    hover.point_at("b")
    assert hover.open_node() is None


# ── folding: a panel collapses, it does not vanish ───────────────────────────

def test_leaving_a_node_collapses_its_panel_rather_than_cutting_it():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("a")
    clock.advance(HOVER_DELAY_S + 0.5)
    hover.unfold()
    hover.clear()
    clock.advance(0.05)
    partial = hover.unfold()
    assert 0.0 < partial < 1.0
    clock.advance(1.0)
    assert hover.unfold() == 0.0


def test_rows_arrive_in_sequence_rather_than_all_at_once():
    """A single alpha on the panel is a fade; staggered rows are an unfold."""
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("core")
    clock.advance(HOVER_DELAY_S)
    clock.advance(0.02)
    panel = hover.panel(_node("core", NodeKind.CORE, label="ORION"), _Load())
    assert panel is not None
    reveals = [r.reveal for r in panel.rows]
    assert reveals == sorted(reveals, reverse=True)
    assert reveals[0] > reveals[-1]


def test_the_panel_reports_only_the_rows_that_have_arrived():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("core")
    clock.advance(HOVER_DELAY_S + 0.01)
    panel = hover.panel(_node("core", NodeKind.CORE, label="ORION"), _Load())
    assert panel is not None
    assert len(panel.visible_rows) < len(panel.rows)


def test_no_panel_when_nothing_is_hovered():
    hover = HoverController(clock=_Clock())
    assert hover.panel(_node()) is None


# ── honest attribution ───────────────────────────────────────────────────────

def test_host_load_is_shown_on_orion_himself():
    rows = _values(diagnostic_rows(_node("core", NodeKind.CORE, label="ORION"), _Load()))
    assert rows["CPU"] == "41%"
    assert rows["GPU"] == "12%"
    assert rows["ORION process"] == "8%"


def test_host_load_is_never_attributed_to_an_individual_subsystem():
    """ORION cannot measure per-module CPU. Printing the machine's figure next
    to one module would invent an attribution that does not exist."""
    rows = _values(diagnostic_rows(_node("module:vision"), _Load()))
    assert "CPU" not in rows
    assert "GPU" not in rows


def test_a_cluster_anchor_does_carry_host_load():
    node = _node("cluster:SYSTEM", NodeKind.SUBSYSTEM, label="System",
                 cluster="SYSTEM")
    assert "CPU" in _values(diagnostic_rows(node, _Load()))


def test_missing_host_readings_render_as_absent_not_zero():
    class _Partial:
        cpu_percent = 55.0
        gpu_percent = None          # no discrete GPU on this machine
        mem_percent = 40.0
        self_cpu_percent = None
    rows = _values(diagnostic_rows(_node("core", NodeKind.CORE), _Partial()))
    assert rows["GPU"] == "—"
    assert "ORION process" not in rows


def test_no_load_source_at_all_is_safe():
    rows = _values(diagnostic_rows(_node("core", NodeKind.CORE), None))
    assert "CPU" not in rows


def test_never_called_is_distinguished_from_called_and_never_failed():
    quiet = _values(diagnostic_rows(_node()))
    assert quiet["Failures"] == "—"
    used = _values(diagnostic_rows(
        _node(telemetry=NodeTelemetry(calls=40, failures=0))))
    assert used["Failures"].startswith("0")


# ── per-kind content ─────────────────────────────────────────────────────────

def test_a_memory_tier_shows_recall_distance_and_no_latency():
    node = _node("memory:archived", NodeKind.MEMORY_TIER, label="archived",
                 cluster="MEMORY", telemetry=NodeTelemetry(queue_depth=5000))
    rows = _values(diagnostic_rows(node))
    assert rows["Stored"] == "5,000"
    assert rows["Recall"] == "cold storage"
    assert "Latency p95" not in rows


def test_working_memory_reads_as_immediate():
    node = _node("memory:working", NodeKind.MEMORY_TIER, label="working",
                 cluster="MEMORY", telemetry=NodeTelemetry(queue_depth=9))
    assert _values(diagnostic_rows(node))["Recall"] == "immediate"


def test_an_mcp_server_shows_how_many_tools_it_offers():
    node = _node("mcp:github", NodeKind.MCP_SERVER, label="github",
                 cluster="COMMUNICATION", capabilities=("a", "b", "c"))
    assert _values(diagnostic_rows(node))["Tools"] == "3"


def test_a_deck_page_says_how_to_open_it():
    node = _node("page:LIBRARY", NodeKind.PAGE, label="LIBRARY",
                 cluster="KNOWLEDGE")
    rows = _values(diagnostic_rows(node))
    assert rows["Zone"] == "Knowledge"
    assert "double-click" in rows["Open"]


def test_an_agent_shows_when_it_last_ran():
    node = _node("agent:research", NodeKind.AGENT, label="Research",
                 cluster="INTELLIGENCE",
                 telemetry=NodeTelemetry(calls=3, heartbeat_age_s=90.0))
    assert _values(diagnostic_rows(node))["Last active"] == "2m ago"


def test_latency_is_reported_at_p95_not_as_an_average():
    """The tail is what the user feels as ORION going quiet; a mean hides it."""
    node = _node(telemetry=NodeTelemetry(calls=10, latency_avg_ms=20.0,
                                         latency_p95_ms=900.0))
    assert _values(diagnostic_rows(node))["Latency p95"] == "900.0 ms"


# ── alarm ────────────────────────────────────────────────────────────────────

def test_a_failing_node_marks_its_panel_as_an_alarm():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("module:forge")
    clock.advance(HOVER_DELAY_S + 0.5)
    node = _node("module:forge", health=Health.DOWN, activity=Activity.ERROR)
    panel = hover.panel(node)
    assert panel is not None and panel.alarm is True


def test_a_healthy_node_is_not_an_alarm():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("module:vision")
    clock.advance(HOVER_DELAY_S + 0.5)
    panel = hover.panel(_node(telemetry=NodeTelemetry(calls=10, failures=0)))
    assert panel is not None and panel.alarm is False


def test_a_high_error_rate_alone_raises_the_alarm():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("module:x")
    clock.advance(HOVER_DELAY_S + 0.5)
    node = _node("module:x", telemetry=NodeTelemetry(calls=10, failures=9))
    panel = hover.panel(node)
    assert panel is not None and panel.alarm is True


def test_the_readout_stays_short_enough_to_glance_at():
    """If it grows to the inspector's length it IS the inspector, arriving
    uninvited."""
    for kind in NodeKind:
        node = _node("x:y", kind, telemetry=NodeTelemetry(calls=5, failures=1))
        assert len(diagnostic_rows(node, _Load())) <= 7


def test_reset_closes_everything():
    clock = _Clock()
    hover = HoverController(clock=clock)
    hover.point_at("a")
    clock.advance(HOVER_DELAY_S + 0.5)
    hover.reset()
    assert hover.open_node() is None
    assert hover.unfold() == 0.0
