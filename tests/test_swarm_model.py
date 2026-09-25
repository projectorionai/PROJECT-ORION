"""
Tests for orion_core.swarm's typed model and cluster taxonomy
(Mark XXII, Phase 0).

Pure Python — no Qt, no display, nothing lazy to trip over. The model
deliberately imports no GUI, and these tests assert that stays true.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.model import (  # noqa: E402
    EMPTY_TELEMETRY,
    Activity,
    EdgeKind,
    Health,
    NodeKind,
    NodeTelemetry,
    SwarmEdge,
    SwarmNode,
    SwarmSnapshot,
)


def _node(nid="module:memory", **kw) -> SwarmNode:
    base = dict(id=nid, kind=NodeKind.SUBSYSTEM, label="memory",
                cluster=cl.MEMORY)
    base.update(kw)
    return SwarmNode(**base)


# ── the model imports no GUI ─────────────────────────────────────────────────

def test_the_model_layer_never_pulls_in_qt():
    """The whole point of the swarm/ package split: this model must be usable
    from a worker thread, a headless diagnostic and the remote surface."""
    import orion_core.swarm.model as m
    import orion_core.swarm.diff as d
    for module in (m, d, cl):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "PyQt6" not in source, f"{module.__name__} must not import Qt"


# ── NodeTelemetry ────────────────────────────────────────────────────────────

def test_error_rate_is_zero_when_nothing_was_called():
    assert NodeTelemetry().error_rate == 0.0


def test_error_rate_is_derived_from_the_counters():
    assert NodeTelemetry(calls=10, failures=2).error_rate == 0.2


def test_error_rate_can_never_exceed_one():
    # Defensive: counters are incremented independently, so a bad caller
    # could in principle report more failures than calls.
    assert NodeTelemetry(calls=2, failures=9).error_rate == 1.0


def test_is_quiet_only_when_never_called_and_doing_nothing():
    assert NodeTelemetry().is_quiet is True
    assert NodeTelemetry(calls=1).is_quiet is False
    assert NodeTelemetry(current_action="indexing").is_quiet is False


def test_absent_measurements_stay_none_rather_than_becoming_zero():
    # "no latency data" and "0 ms latency" must never look the same.
    tel = NodeTelemetry(calls=3)
    assert tel.latency_p95_ms is None
    assert tel.as_dict()["latency_p95_ms"] is None


def test_telemetry_as_dict_includes_the_derived_error_rate():
    assert NodeTelemetry(calls=4, failures=1).as_dict()["error_rate"] == 0.25


# ── SwarmNode ────────────────────────────────────────────────────────────────

def test_visual_key_ignores_telemetry_entirely():
    """The single most important property in the model: telemetry churn must
    not read as a visual change, or the renderer uploads every tick."""
    quiet = _node()
    busy = _node(telemetry=NodeTelemetry(calls=900, latency_p95_ms=42.0))
    assert quiet.visual_key == busy.visual_key


def test_visual_key_changes_when_health_changes():
    assert _node(health=Health.OK).visual_key != _node(health=Health.DOWN).visual_key


def test_visual_key_changes_when_activity_changes():
    a = _node(activity=Activity.IDLE)
    b = _node(activity=Activity.BUSY)
    assert a.visual_key != b.visual_key


def test_visual_key_changes_when_cluster_changes():
    assert _node(cluster=cl.MEMORY).visual_key != _node(cluster=cl.SYSTEM).visual_key


def test_needs_attention_for_degraded_and_down_and_error():
    assert _node(health=Health.DEGRADED).needs_attention is True
    assert _node(health=Health.DOWN).needs_attention is True
    assert _node(activity=Activity.ERROR).needs_attention is True


def test_a_healthy_idle_node_does_not_need_attention():
    assert _node(health=Health.OK, activity=Activity.IDLE).needs_attention is False


def test_with_telemetry_returns_a_new_node_and_leaves_the_original_alone():
    original = _node()
    updated = original.with_telemetry(NodeTelemetry(calls=7))
    assert updated is not original
    assert original.telemetry is EMPTY_TELEMETRY
    assert updated.telemetry.calls == 7


def test_nodes_are_immutable():
    import dataclasses
    node = _node()
    try:
        node.label = "hacked"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SwarmNode must be frozen")


def test_node_as_dict_serialises_enums_to_plain_strings():
    payload = _node(health=Health.OK, activity=Activity.BUSY).as_dict()
    assert payload["kind"] == "subsystem"
    assert payload["health"] == "OK"
    assert payload["activity"] == "busy"
    assert isinstance(payload["telemetry"], dict)


# ── SwarmEdge ────────────────────────────────────────────────────────────────

def test_edge_key_identifies_source_target_and_kind():
    edge = SwarmEdge("a", "b", EdgeKind.DEPENDENCY)
    assert edge.key == ("a", "b", EdgeKind.DEPENDENCY)


def test_two_edges_between_the_same_pair_of_different_kinds_are_distinct():
    # A module can both depend on another AND stream data to it; collapsing
    # those into one edge would lose real information.
    dep = SwarmEdge("a", "b", EdgeKind.DEPENDENCY)
    stream = SwarmEdge("a", "b", EdgeKind.DATA_STREAM)
    assert dep.key != stream.key


def test_edges_rest_at_zero_traffic():
    assert SwarmEdge("a", "b").traffic == 0.0


# ── SwarmSnapshot queries ────────────────────────────────────────────────────

def _snapshot() -> SwarmSnapshot:
    return SwarmSnapshot(
        nodes=(
            _node("core", kind=NodeKind.CORE, cluster=cl.INTELLIGENCE, health=Health.OK),
            _node("module:memory", cluster=cl.MEMORY, health=Health.OK),
            _node("module:vision", cluster=cl.SYSTEM, health=Health.DEGRADED),
            _node("agent:coding", kind=NodeKind.AGENT, cluster=cl.DEVELOPMENT),
        ),
        edges=(SwarmEdge("core", "module:memory"),),
        core_state="LISTENING",
    )


def test_snapshot_node_ids_and_lookup():
    snap = _snapshot()
    assert "module:vision" in snap.node_ids()
    assert snap.by_id()["module:vision"].health is Health.DEGRADED


def test_snapshot_filters_by_cluster_and_kind():
    snap = _snapshot()
    assert len(snap.in_cluster(cl.MEMORY)) == 1
    assert len(snap.of_kind(NodeKind.AGENT)) == 1


def test_snapshot_surfaces_only_the_nodes_needing_attention():
    ids = [n.id for n in _snapshot().needing_attention()]
    assert ids == ["module:vision"]


def test_snapshot_serialises_whole_graph():
    payload = _snapshot().as_dict()
    assert len(payload["nodes"]) == 4
    assert payload["core_state"] == "LISTENING"
    assert payload["degraded_sources"] == []


def test_an_empty_snapshot_is_valid_and_answers_queries():
    empty = SwarmSnapshot()
    assert empty.node_ids() == frozenset()
    assert empty.needing_attention() == ()


# ── clusters ─────────────────────────────────────────────────────────────────

def test_cluster_order_matches_orions_twelve_real_deck_zones():
    """These must stay in lockstep with UnifiedDashboard.ZONE_ORDER, or a
    swarm cluster and the deck tab of the same name drift apart."""
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    assert set(cl.CLUSTER_ORDER) == set(UnifiedDashboard.ZONE_ORDER)
    assert len(cl.CLUSTER_ORDER) == 12


def test_every_registered_module_name_maps_to_a_real_cluster():
    for name in ("dispatcher", "memory", "telemetry", "forge", "reasoning",
                 "workflow_engine", "security_recon", "outlook", "avatar"):
        assert cl.is_known_cluster(cl.cluster_for_module(name))


def test_an_unmapped_module_falls_back_to_system_rather_than_vanishing():
    assert cl.cluster_for_module("some_brand_new_subsystem") == cl.SYSTEM


def test_module_lookup_is_case_insensitive():
    assert cl.cluster_for_module("MEMORY") == cl.MEMORY


def test_blank_module_name_is_safe():
    assert cl.cluster_for_module("") == cl.FALLBACK_CLUSTER


def test_the_six_built_in_agents_land_in_their_deck_page_zones():
    assert cl.cluster_of_agent("coding") == cl.DEVELOPMENT
    assert cl.cluster_of_agent("marketing") == cl.BUSINESS
    assert cl.cluster_of_agent("research") == cl.RESEARCH
    assert cl.cluster_of_agent("design") == cl.CREATIVE
    assert cl.cluster_of_agent("fashion") == cl.CREATIVE
    assert cl.cluster_of_agent("entertainment") == cl.CREATIVE


def test_a_declarative_agent_is_classified_by_keyword():
    # config/agents/*.json can add an agent with no code change; it must still
    # cluster somewhere meaningful rather than beside the event bus.
    assert cl.cluster_of_agent("legal_advisor") == cl.BUSINESS
    assert cl.cluster_of_agent("threat_hunter") == cl.SECURITY
    assert cl.cluster_of_agent("devops_engineer") == cl.DEVELOPMENT


def test_an_unclassifiable_agent_still_gets_a_cluster():
    assert cl.is_known_cluster(cl.cluster_of_agent("zzz_unknown"))


def test_normalise_cluster_coerces_case_and_rejects_nonsense():
    assert cl.normalise_cluster("memory") == cl.MEMORY
    assert cl.normalise_cluster("not a zone") == cl.FALLBACK_CLUSTER


def test_cluster_index_is_stable_and_ordered():
    assert cl.cluster_index(cl.INTELLIGENCE) == 0
    assert cl.cluster_index(cl.SYSTEM) == 11
    # deterministic placement is what stops the layout reshuffling on restart
    assert cl.cluster_index("nonsense") == cl.cluster_index(cl.FALLBACK_CLUSTER)
