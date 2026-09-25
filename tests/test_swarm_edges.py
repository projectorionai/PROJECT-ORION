"""
Tests for orion_core.swarm.render.edges and the traffic path through
SwarmScene (Mark XXII, Phase 3).

The behaviour that matters most: a frame where only TRAFFIC changed must
touch the edge buffer and leave the node buffer completely alone. Traffic
moves constantly — that is the whole point of measuring it — so if it dragged
node uploads along with it, the redesign's central saving would be undone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.model import (  # noqa: E402
    EdgeKind,
    Health,
    NodeKind,
    SwarmEdge,
    SwarmNode,
    SwarmSnapshot,
)
from orion_core.swarm.render.edges import (  # noqa: E402
    EDGE_FLOATS,
    VERTS_PER_EDGE,
    EdgeBuffer,
    edge_colour,
)
from orion_core.swarm.render.scene import SwarmScene  # noqa: E402
from orion_core.swarm.traffic import TrafficLedger  # noqa: E402

KEY = ("a", "b", "membership")


def _n(nid, cluster=cl.SYSTEM, **kw) -> SwarmNode:
    base = dict(id=nid, kind=NodeKind.SUBSYSTEM, label=nid, cluster=cluster)
    base.update(kw)
    return SwarmNode(**base)


def _snap(nodes, edges=()) -> SwarmSnapshot:
    return SwarmSnapshot(nodes=tuple(nodes), edges=tuple(edges))


# ── EdgeBuffer basics ────────────────────────────────────────────────────────

def test_an_edge_occupies_two_vertices():
    buf = EdgeBuffer()
    buf.upsert(KEY, (0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert buf.live_count == 1
    assert buf.vertex_count == VERTS_PER_EDGE
    assert buf.view().shape == (2, EDGE_FLOATS)


def test_the_two_vertices_carry_the_endpoints():
    buf = EdgeBuffer()
    buf.upsert(KEY, (1.0, 0.0, 0.0), (0.0, 5.0, 0.0))
    source, target = buf.endpoints_of(KEY)
    assert source == (1.0, 0.0, 0.0)
    assert target == (0.0, 5.0, 0.0)


def test_the_t_parameter_runs_from_source_to_target():
    """The shader animates the packet purely from t, so 0 -> 1 must hold."""
    buf = EdgeBuffer()
    buf.upsert(KEY, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    view = buf.view()
    assert float(view[0][6]) == 0.0
    assert float(view[1][6]) == 1.0


def test_updating_an_edge_reuses_its_slot():
    buf = EdgeBuffer()
    first = buf.upsert(KEY, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    second = buf.upsert(KEY, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert first == second
    assert buf.live_count == 1


def test_a_removed_edge_is_collapsed_and_its_slot_pooled():
    buf = EdgeBuffer()
    buf.upsert(KEY, (0.0, 0.0, 0.0), (9.0, 9.0, 9.0))
    buf.remove(KEY)
    assert buf.free_slots == 1
    source, target = (tuple(buf.view()[0][:3]), tuple(buf.view()[1][:3]))
    assert source == target        # degenerate: draws nothing


def test_a_pooled_edge_slot_is_reused():
    buf = EdgeBuffer()
    buf.upsert(KEY, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    buf.remove(KEY)
    reused = buf.upsert(("c", "d", "dependency"), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert reused == 0
    assert buf.free_slots == 0


def test_removing_an_unknown_edge_is_safe():
    assert EdgeBuffer().remove(("x", "y", "z")) is False


def test_the_buffer_grows_by_doubling():
    buf = EdgeBuffer(capacity=2)
    for i in range(9):
        buf.upsert((f"a{i}", "b", "membership"), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert buf.capacity == 16
    assert buf.live_count == 9


def test_growing_preserves_existing_edges():
    buf = EdgeBuffer(capacity=2)
    buf.upsert(KEY, (7.0, 7.0, 7.0), (8.0, 8.0, 8.0))
    for i in range(10):
        buf.upsert((f"a{i}", "b", "membership"), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert buf.endpoints_of(KEY)[0] == (7.0, 7.0, 7.0)


# ── traffic updates are cheap ────────────────────────────────────────────────

def test_set_traffic_writes_both_vertices():
    buf = EdgeBuffer()
    buf.upsert(KEY, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    buf.set_traffic(KEY, 0.75)
    assert float(buf.view()[0][7]) == 0.75
    assert float(buf.view()[1][7]) == 0.75


def test_set_traffic_leaves_the_endpoints_untouched():
    """The common per-tick update must not rewrite positions."""
    buf = EdgeBuffer()
    buf.upsert(KEY, (3.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    buf.set_traffic(KEY, 0.9)
    assert buf.endpoints_of(KEY) == ((3.0, 0.0, 0.0), (4.0, 0.0, 0.0))


def test_set_traffic_on_an_unknown_edge_is_a_no_op():
    assert EdgeBuffer().set_traffic(KEY, 0.5) is False


def test_only_the_touched_edge_is_dirtied():
    buf = EdgeBuffer()
    for i in range(50):
        buf.upsert((f"a{i}", "b", "membership"), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    buf.clear_dirty()
    buf.set_traffic(("a20", "b", "membership"), 0.5)
    assert buf.dirty_slot_count == 1
    assert buf.dirty_vertex_range() == (40, 42)


def test_a_clean_buffer_reports_no_dirty_range():
    buf = EdgeBuffer()
    buf.upsert(KEY, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    buf.clear_dirty()
    assert buf.dirty_vertex_range() is None


def test_mark_all_dirty_covers_every_vertex():
    buf = EdgeBuffer()
    for i in range(4):
        buf.upsert((f"a{i}", "b", "membership"), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    buf.clear_dirty()
    buf.mark_all_dirty()
    assert buf.dirty_vertex_range() == (0, 8)


# ── appearance ───────────────────────────────────────────────────────────────

def test_edge_colour_varies_by_kind():
    assert edge_colour("mcp_request") != edge_colour("membership")


def test_an_unknown_edge_kind_still_gets_a_colour():
    assert edge_colour("something_new") is not None


def test_packet_phase_is_stable_across_runs():
    """crc32-derived, so packets do not re-phase on every restart."""
    a, b = EdgeBuffer(), EdgeBuffer()
    a.upsert(KEY, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    b.upsert(KEY, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert float(a.view()[0][8]) == float(b.view()[0][8])


def test_different_edges_get_different_packet_phases():
    """Otherwise every edge pulses in lockstep and reads as a screensaver."""
    buf = EdgeBuffer()
    buf.upsert(("a", "b", "membership"), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    buf.upsert(("c", "d", "membership"), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert float(buf.view()[0][8]) != float(buf.view()[2][8])


# ── through the scene ────────────────────────────────────────────────────────

def _graph(traffic=0.0):
    nodes = [_n("cluster:MEMORY", cluster=cl.MEMORY), _n("module:m1", cluster=cl.MEMORY)]
    edges = [SwarmEdge("cluster:MEMORY", "module:m1", EdgeKind.MEMBERSHIP, traffic)]
    return _snap(nodes, edges)


def test_the_scene_populates_the_edge_buffer():
    scene = SwarmScene()
    scene.apply(_graph())
    assert scene.edges.live_count == 1
    assert scene.edges.vertex_count == 2


def test_edge_endpoints_come_from_the_layout():
    scene = SwarmScene()
    scene.apply(_graph())
    source, _ = scene.edges.endpoints_of(
        ("cluster:MEMORY", "module:m1", "membership"))
    assert source == scene.buffer.position_of("cluster:MEMORY")


def test_a_traffic_only_frame_never_touches_the_node_buffer():
    """The headline Phase 3 behaviour."""
    scene = SwarmScene()
    scene.apply(_graph(0.0))
    scene.buffer.clear_dirty()
    scene.edges.clear_dirty()

    delta = scene.apply(_graph(0.85))

    assert len(delta.edges_traffic_changed) == 1
    assert scene.buffer.is_dirty is False        # zero node uploads
    assert scene.edges.is_dirty is True
    # float32 buffer, so compare with tolerance rather than exactly
    assert abs(scene.edges.traffic_of(
        ("cluster:MEMORY", "module:m1", "membership")) - 0.85) < 1e-6


def test_an_edge_to_a_missing_node_is_not_drawn():
    """Otherwise it would be drawn to the origin, spraying lines into the
    centre of the graph."""
    scene = SwarmScene()
    scene.apply(_snap([_n("a")], [SwarmEdge("a", "ghost")]))
    assert scene.edges.live_count == 0


def test_removing_an_edge_frees_its_slot():
    scene = SwarmScene()
    scene.apply(_graph())
    scene.apply(_snap([_n("cluster:MEMORY", cluster=cl.MEMORY),
                       _n("module:m1", cluster=cl.MEMORY)], []))
    assert scene.edges.live_count == 0
    assert scene.edges.free_slots == 1


def test_a_relayout_rewrites_surviving_edge_endpoints():
    """Adding a node re-solves the layout; every edge's endpoints must follow
    or they would point at stale coordinates."""
    scene = SwarmScene()
    scene.apply(_graph())
    bigger = _snap(
        [_n("cluster:MEMORY", cluster=cl.MEMORY), _n("module:m1", cluster=cl.MEMORY),
         _n("cluster:SECURITY", cluster=cl.SECURITY)],
        [SwarmEdge("cluster:MEMORY", "module:m1", EdgeKind.MEMBERSHIP),
         SwarmEdge("cluster:SECURITY", "module:m1", EdgeKind.DEPENDENCY)])
    scene.apply(bigger)
    source, target = scene.edges.endpoints_of(
        ("cluster:MEMORY", "module:m1", "membership"))
    assert source == scene.buffer.position_of("cluster:MEMORY")
    assert target == scene.buffer.position_of("module:m1")


def test_scene_reset_clears_edges_too():
    scene = SwarmScene()
    scene.apply(_graph())
    scene.reset()
    assert scene.edges.live_count == 0


def test_scene_stats_report_edges():
    scene = SwarmScene()
    scene.apply(_graph())
    stats = scene.stats()
    assert stats["edges"] == 1
    assert stats["edge_vertices"] == 2


# ── ledger -> scene, end to end ──────────────────────────────────────────────

def test_a_recorded_event_lights_the_edge_in_the_buffer():
    scene = SwarmScene()
    ledger = TrafficLedger()
    base = _graph()
    scene.apply(base)
    scene.buffer.clear_dirty()
    scene.edges.clear_dirty()

    ledger.record("cluster:MEMORY", "module:m1", EdgeKind.MEMBERSHIP)
    scene.apply(replace(base, edges=ledger.apply(base.edges)))

    assert scene.edges.traffic_of(
        ("cluster:MEMORY", "module:m1", "membership")) > 0.0
    assert scene.buffer.is_dirty is False


def test_traffic_fades_back_out_over_time():
    class _Clock:
        def __init__(self): self.now = 0.0
        def __call__(self): return self.now

    clock = _Clock()
    ledger = TrafficLedger(half_life_s=1.0, clock=clock)
    scene = SwarmScene()
    base = _graph()
    scene.apply(base)

    ledger.record("cluster:MEMORY", "module:m1", EdgeKind.MEMBERSHIP)
    scene.apply(replace(base, edges=ledger.apply(base.edges)))
    lit = scene.edges.traffic_of(("cluster:MEMORY", "module:m1", "membership"))

    clock.now = 8.0
    scene.apply(replace(base, edges=ledger.apply(base.edges)))
    faded = scene.edges.traffic_of(("cluster:MEMORY", "module:m1", "membership"))
    assert faded < lit
