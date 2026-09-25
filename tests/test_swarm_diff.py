"""
Tests for orion_core.swarm.diff (Mark XXII, Phase 0).

The diff engine is the fix for the audit's worst finding — the old view
resent the whole graph every 2 s (7 KB today, 196 KB at 1,000 agents) and the
JS side disposed and rebuilt every geometry. These tests pin the behaviour
that makes that impossible: telemetry churn must never be classified as a
visual change, and an unchanged graph must produce an empty delta.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm.clusters import MEMORY, SYSTEM  # noqa: E402
from orion_core.swarm.diff import diff_snapshots  # noqa: E402
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


def _n(nid, **kw) -> SwarmNode:
    base = dict(id=nid, kind=NodeKind.SUBSYSTEM, label=nid, cluster=SYSTEM)
    base.update(kw)
    return SwarmNode(**base)


def _snap(nodes=(), edges=(), core_state="STANDBY", degraded=()) -> SwarmSnapshot:
    return SwarmSnapshot(nodes=tuple(nodes), edges=tuple(edges),
                         core_state=core_state, degraded_sources=tuple(degraded))


# ── first frame ──────────────────────────────────────────────────────────────

def test_first_frame_reports_everything_as_added():
    current = _snap([_n("a"), _n("b")], [SwarmEdge("a", "b")])
    delta = diff_snapshots(None, current)
    assert [n.id for n in delta.added] == ["a", "b"]
    assert len(delta.edges_added) == 1
    assert delta.needs_gpu_upload is True
    assert delta.needs_layout is True


def test_first_frame_of_an_empty_graph_is_not_treated_as_a_no_op():
    # core_state still has to reach the renderer even with no nodes.
    delta = diff_snapshots(None, _snap())
    assert delta.core_state_changed is True


# ── the steady state ─────────────────────────────────────────────────────────

def test_an_identical_snapshot_produces_a_completely_empty_delta():
    snap = _snap([_n("a"), _n("b")], [SwarmEdge("a", "b")])
    same = _snap([_n("a"), _n("b")], [SwarmEdge("a", "b")])
    delta = diff_snapshots(snap, same)
    assert delta.is_empty is True
    assert delta.needs_gpu_upload is False
    assert delta.needs_layout is False


# ── the critical classification ──────────────────────────────────────────────

def test_telemetry_only_change_never_requests_a_gpu_upload():
    """The whole reason the model splits visual identity from telemetry."""
    before = _snap([_n("a", telemetry=NodeTelemetry(calls=10, latency_p95_ms=4.0))])
    after = _snap([_n("a", telemetry=NodeTelemetry(calls=11, latency_p95_ms=4.4))])
    delta = diff_snapshots(before, after)
    assert [n.id for n in delta.telemetry_changed] == ["a"]
    assert delta.visual_changed == ()
    assert delta.needs_gpu_upload is False
    assert delta.needs_layout is False
    assert delta.is_empty is False          # the inspector still needs it


def test_health_change_is_visual_and_does_request_an_upload():
    before = _snap([_n("a", health=Health.OK)])
    after = _snap([_n("a", health=Health.DOWN)])
    delta = diff_snapshots(before, after)
    assert [n.id for n in delta.visual_changed] == ["a"]
    assert delta.telemetry_changed == ()
    assert delta.needs_gpu_upload is True


def test_recolouring_a_node_never_triggers_a_relayout():
    # Health flips are frequent; re-solving the layout for one would undo the
    # whole point of moving layout to a worker thread.
    before = _snap([_n("a", health=Health.OK)])
    after = _snap([_n("a", health=Health.DEGRADED)])
    assert diff_snapshots(before, after).needs_layout is False


def test_activity_change_is_visual():
    before = _snap([_n("a", activity=Activity.IDLE)])
    after = _snap([_n("a", activity=Activity.BUSY)])
    assert [n.id for n in diff_snapshots(before, after).visual_changed] == ["a"]


def test_a_node_changing_both_appearance_and_telemetry_is_reported_only_once():
    """Reporting it in both buckets would make callers do the work twice."""
    before = _snap([_n("a", health=Health.OK, telemetry=NodeTelemetry(calls=1))])
    after = _snap([_n("a", health=Health.DOWN, telemetry=NodeTelemetry(calls=99))])
    delta = diff_snapshots(before, after)
    assert [n.id for n in delta.visual_changed] == ["a"]
    assert delta.telemetry_changed == ()


def test_a_changed_detail_string_counts_as_telemetry_not_visual():
    before = _snap([_n("a", detail="idle")])
    after = _snap([_n("a", detail="indexing 40 files")])
    delta = diff_snapshots(before, after)
    assert [n.id for n in delta.telemetry_changed] == ["a"]
    assert delta.needs_gpu_upload is False


# ── add / remove ─────────────────────────────────────────────────────────────

def test_added_and_removed_nodes_are_detected():
    before = _snap([_n("a"), _n("b")])
    after = _snap([_n("b"), _n("c")])
    delta = diff_snapshots(before, after)
    assert [n.id for n in delta.added] == ["c"]
    assert delta.removed == ("a",)
    assert delta.needs_layout is True


def test_an_added_node_is_not_also_reported_as_changed():
    before = _snap([_n("a")])
    after = _snap([_n("a"), _n("b", health=Health.DOWN)])
    delta = diff_snapshots(before, after)
    assert [n.id for n in delta.added] == ["b"]
    assert delta.visual_changed == ()


def test_results_are_deterministically_ordered():
    before = _snap([])
    after = _snap([_n("zebra"), _n("alpha"), _n("mid")])
    assert [n.id for n in diff_snapshots(before, after).added] == ["alpha", "mid", "zebra"]


# ── edges ────────────────────────────────────────────────────────────────────

def test_edge_addition_and_removal_are_detected():
    before = _snap([_n("a"), _n("b")], [SwarmEdge("a", "b")])
    after = _snap([_n("a"), _n("b")], [SwarmEdge("b", "a")])
    delta = diff_snapshots(before, after)
    assert len(delta.edges_added) == 1
    assert delta.edges_removed == (("a", "b", "membership"),)


def test_an_edge_changing_kind_is_an_add_plus_a_remove():
    before = _snap(edges=[SwarmEdge("a", "b", EdgeKind.MEMBERSHIP)])
    after = _snap(edges=[SwarmEdge("a", "b", EdgeKind.DEPENDENCY)])
    delta = diff_snapshots(before, after)
    assert len(delta.edges_added) == 1
    assert len(delta.edges_removed) == 1


def test_tiny_traffic_jitter_is_ignored():
    """Traffic is a continuously drifting float; re-uploading on every
    micro-change would reintroduce exactly the churn this module removes."""
    before = _snap(edges=[SwarmEdge("a", "b", traffic=0.50)])
    after = _snap(edges=[SwarmEdge("a", "b", traffic=0.505)])
    assert diff_snapshots(before, after).edges_traffic_changed == ()


def test_a_real_traffic_change_is_reported():
    before = _snap(edges=[SwarmEdge("a", "b", traffic=0.1)])
    after = _snap(edges=[SwarmEdge("a", "b", traffic=0.9)])
    delta = diff_snapshots(before, after)
    assert len(delta.edges_traffic_changed) == 1
    # traffic animates on the GPU without reallocating anything
    assert delta.needs_gpu_upload is False


# ── snapshot-level fields ────────────────────────────────────────────────────

def test_core_state_change_is_reported():
    delta = diff_snapshots(_snap(core_state="IDLE"), _snap(core_state="SPEAKING"))
    assert delta.core_state_changed is True
    assert delta.is_empty is False


def test_a_source_going_degraded_is_reported():
    delta = diff_snapshots(_snap(), _snap(degraded=("mcp",)))
    assert delta.degraded_sources_changed is True


# ── robustness + reporting ───────────────────────────────────────────────────

def test_a_none_current_snapshot_never_crashes_the_render_loop():
    assert diff_snapshots(_snap(), None).is_empty is True


def test_summary_reports_counts_and_the_work_flags():
    before = _snap([_n("a", health=Health.OK)])
    after = _snap([_n("a", health=Health.DOWN), _n("b")])
    summary = diff_snapshots(before, after).summary()
    assert summary["added"] == 1
    assert summary["visual_changed"] == 1
    assert summary["gpu_upload"] is True
    assert summary["relayout"] is True


def test_a_large_graph_with_one_real_change_yields_a_tiny_delta():
    """The headline behaviour: cost scales with what moved, not with size."""
    big_before = _snap([_n(f"m{i}", cluster=MEMORY) for i in range(1000)])
    nodes = [_n(f"m{i}", cluster=MEMORY) for i in range(1000)]
    nodes[500] = _n("m500", cluster=MEMORY, health=Health.DOWN)
    delta = diff_snapshots(big_before, _snap(nodes))
    assert len(delta.visual_changed) == 1
    assert delta.added == () and delta.removed == ()
    assert delta.needs_layout is False
