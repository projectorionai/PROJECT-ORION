"""
Tests for orion_core.swarm.traffic (Mark XXII, Phase 3).

The clock is injected throughout, so decay is tested by advancing a fake
monotonic counter rather than sleeping — deterministic, instant, and it makes
the half-life assertions exact instead of approximate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm.model import EdgeKind, SwarmEdge  # noqa: E402
from orion_core.swarm.traffic import (  # noqa: E402
    DEFAULT_BOOST,
    TrafficLedger,
    edge_for_tool_call,
)


class _Clock:
    def __init__(self): self.now = 1000.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def _ledger(half_life=2.0, **kw):
    clock = _Clock()
    return TrafficLedger(half_life_s=half_life, clock=clock, **kw), clock


# ── recording ────────────────────────────────────────────────────────────────

def test_an_unrecorded_edge_has_no_traffic():
    ledger, _ = _ledger()
    assert ledger.intensity("core", "agent:coding") == 0.0


def test_recording_raises_the_level():
    ledger, _ = _ledger()
    ledger.record("core", "agent:coding")
    assert ledger.intensity("core", "agent:coding") == DEFAULT_BOOST


def test_repeated_events_accumulate():
    ledger, _ = _ledger()
    for _ in range(2):
        ledger.record("core", "agent:coding")
    assert ledger.intensity("core", "agent:coding") > DEFAULT_BOOST


def test_intensity_is_capped_at_one():
    ledger, _ = _ledger()
    for _ in range(50):
        ledger.record("core", "agent:coding")
    assert ledger.intensity("core", "agent:coding") == 1.0


def test_edges_of_different_kinds_are_tracked_separately():
    """A dependency and a live MCP request between the same pair are
    different facts and must not share a level."""
    ledger, _ = _ledger()
    ledger.record("a", "b", EdgeKind.MCP_REQUEST)
    assert ledger.intensity("a", "b", EdgeKind.MCP_REQUEST) > 0.0
    assert ledger.intensity("a", "b", EdgeKind.DEPENDENCY) == 0.0


def test_direction_matters():
    ledger, _ = _ledger()
    ledger.record("a", "b")
    assert ledger.intensity("b", "a") == 0.0


def test_a_blank_endpoint_is_ignored():
    ledger, _ = _ledger()
    ledger.record("", "b")
    ledger.record("a", "")
    assert ledger.stats()["tracked_edges"] == 0


def test_weight_scales_the_contribution():
    ledger, _ = _ledger()
    ledger.record("a", "b", weight=2.0)
    assert ledger.intensity("a", "b") > DEFAULT_BOOST


# ── decay ────────────────────────────────────────────────────────────────────

def test_intensity_halves_over_one_half_life():
    ledger, clock = _ledger(half_life=2.0)
    ledger.record("core", "agent:coding")
    before = ledger.intensity("core", "agent:coding")
    clock.advance(2.0)
    assert abs(ledger.intensity("core", "agent:coding") - before / 2) < 1e-9


def test_intensity_keeps_decaying_over_time():
    ledger, clock = _ledger(half_life=1.0)
    ledger.record("a", "b")
    clock.advance(10.0)
    assert ledger.intensity("a", "b") < 0.01


def test_decay_is_lazy_and_costs_nothing_while_idle():
    """No timer, no sweep — an untouched ledger does no work at all."""
    ledger, clock = _ledger()
    ledger.record("a", "b")
    clock.advance(100.0)
    assert ledger.intensity("a", "b") < 0.001      # correct without any tick


def test_a_new_event_flares_a_decayed_edge_again():
    ledger, clock = _ledger(half_life=1.0)
    ledger.record("a", "b")
    clock.advance(5.0)
    faded = ledger.intensity("a", "b")
    ledger.record("a", "b")
    assert ledger.intensity("a", "b") > faded


def test_a_sustained_trickle_stays_lit():
    """A steady stream of calls should read as continuously busy."""
    ledger, clock = _ledger(half_life=2.5)
    for _ in range(10):
        ledger.record("a", "b")
        clock.advance(0.5)
    assert ledger.intensity("a", "b") > 0.5


# ── applying to edges ────────────────────────────────────────────────────────

def test_apply_sets_traffic_from_measured_activity():
    ledger, _ = _ledger()
    ledger.record("core", "agent:coding", EdgeKind.DELEGATION)
    edges = (SwarmEdge("core", "agent:coding", EdgeKind.DELEGATION),)
    assert ledger.apply(edges)[0].traffic > 0.0


def test_a_structural_edge_with_no_traffic_survives_at_rest():
    """Dropping quiet edges would make the graph appear to dissolve whenever
    ORION goes idle."""
    ledger, _ = _ledger()
    edges = (SwarmEdge("core", "cluster:MEMORY"),)
    applied = ledger.apply(edges)
    assert len(applied) == 1
    assert applied[0].traffic == 0.0


def test_apply_returns_the_same_object_when_nothing_changed():
    """Identity is what lets the diff engine skip the upload entirely."""
    ledger, _ = _ledger()
    edge = SwarmEdge("core", "cluster:MEMORY")
    assert ledger.apply((edge,))[0] is edge


def test_apply_preserves_edge_identity_fields():
    ledger, _ = _ledger()
    ledger.record("a", "b", EdgeKind.MEMORY_WRITE)
    applied = ledger.apply((SwarmEdge("a", "b", EdgeKind.MEMORY_WRITE),))[0]
    assert applied.source == "a" and applied.target == "b"
    assert applied.kind is EdgeKind.MEMORY_WRITE


def test_apply_handles_an_empty_edge_list():
    ledger, _ = _ledger()
    assert ledger.apply(()) == ()


# ── housekeeping ─────────────────────────────────────────────────────────────

def test_prune_drops_fully_decayed_edges():
    ledger, clock = _ledger(half_life=1.0)
    ledger.record("a", "b")
    clock.advance(20.0)
    assert ledger.prune() == 1
    assert ledger.stats()["tracked_edges"] == 0


def test_prune_keeps_still_active_edges():
    ledger, _ = _ledger()
    ledger.record("a", "b")
    assert ledger.prune() == 0


def test_the_ledger_is_bounded_under_an_event_storm():
    ledger, _ = _ledger(max_edges=64)
    for i in range(500):
        ledger.record(f"n{i}", f"m{i}")
    assert ledger.stats()["tracked_edges"] <= 64


def test_busiest_ranks_the_most_active_edges():
    ledger, _ = _ledger()
    ledger.record("a", "b")
    for _ in range(3):
        ledger.record("c", "d")
    assert ledger.busiest(1)[0][0][:2] == ("c", "d")


def test_reset_clears_everything():
    ledger, _ = _ledger()
    ledger.record("a", "b")
    ledger.reset()
    assert ledger.intensity("a", "b") == 0.0


# ── routing agrees with the dispatcher's own pulse target ────────────────────

def test_an_mcp_call_routes_from_server_to_tool():
    source, target, kind = edge_for_tool_call("mcp__github__list_repos")
    assert (source, target) == ("mcp:github", "mcp_tool:github:list_repos")
    assert kind is EdgeKind.MCP_REQUEST


def test_agent_dispatch_routes_from_the_agents_cluster():
    """Agents hang off their CLUSTER, not off core."""
    source, target, kind = edge_for_tool_call("agent_dispatch", {"agent": "coding"})
    assert (source, target) == ("cluster:DEVELOPMENT", "agent:coding")
    assert kind is EdgeKind.DELEGATION


def test_a_workflow_run_routes_from_the_automation_cluster():
    source, target, _ = edge_for_tool_call("workflow", {"name": "morning"})
    assert (source, target) == ("cluster:AUTOMATION", "workflow:morning")


def test_a_memory_tool_routes_to_the_memory_cluster():
    source, target, _ = edge_for_tool_call("remember", {})
    assert (source, target) == ("core", "cluster:MEMORY")


def test_an_unrecognised_tool_falls_back_to_the_system_cluster():
    source, target, _ = edge_for_tool_call("some_new_tool", {})
    assert (source, target) == ("core", "cluster:SYSTEM")


# ── the routing must name edges that REALLY EXIST ────────────────────────────

def _real_graph():
    from orion_core.swarm.sources import compose

    class _Agents:
        def describe(self):
            return [{"name": n, "title": n, "focus": "", "calls": 0}
                    for n in ("coding", "research", "marketing")]

    class _Modules:
        def all_described(self, t):
            return {"memory": {"role": "r", "health": "OK", "dependencies": []}}

    class _Reg:
        modules = _Modules()

    class _MCP:
        def catalogue(self):
            return {"github": {"tools": [{"name": "list_repos"}]}}
        def health_snapshot(self): return {"github": "OK"}

    class _WF:
        definitions = {"morning": {"steps": []}}

    class _Mem:
        def tiers_snapshot(self): return {"short_term": 1}

    return compose(agents=_Agents(), registries=_Reg(), mcp_host=_MCP(),
                   workflow_engine=_WF(), memory=_Mem())


def test_every_routed_edge_exists_in_the_real_graph():
    """The regression that made traffic silently invisible: routing named
    core -> agent:X while the graph actually builds cluster:Y -> agent:X, so
    every recorded event was discarded and no edge ever lit up."""
    snapshot = _real_graph()
    real = {(e.source, e.target, e.kind.value) for e in snapshot.edges}
    for tool, args in (
        ("agent_dispatch", {"agent": "coding"}),
        ("reason", {"agent": "research"}),
        ("agent_dispatch", {"agent": "marketing"}),
        ("workflow", {"name": "morning"}),
        ("mcp__github__list_repos", {}),
        ("remember", {}),
        ("some_unknown_tool", {}),
    ):
        source, target, kind = edge_for_tool_call(tool, args)
        assert (source, target, kind.value) in real, f"{tool} -> {source}->{target}"


def test_recorded_traffic_actually_lights_a_real_edge():
    """End-to-end: dispatch event -> ledger -> a lit edge in the snapshot."""
    ledger, _ = _ledger()
    snapshot = _real_graph()
    source, target, kind = edge_for_tool_call("agent_dispatch", {"agent": "coding"})
    ledger.record(source, target, kind)
    applied = ledger.apply(snapshot.edges)
    lit = [e for e in applied if e.traffic > 0.0]
    assert len(lit) == 1
    assert (lit[0].source, lit[0].target) == ("cluster:DEVELOPMENT", "agent:coding")


def test_routing_targets_the_same_node_the_dispatcher_pulses():
    """The edge that lights and the node that flashes must agree."""
    from orion_core.dispatcher import OrionDispatcher
    for tool, args in (("agent_dispatch", {"agent": "coding"}),
                       ("workflow", {"name": "morning"})):
        _, edge_target, _ = edge_for_tool_call(tool, args)
        assert edge_target == OrionDispatcher._swarm_pulse_target(tool, args)
