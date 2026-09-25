"""
Tests for orion_core.swarm.render.scene (Mark XXII, Phase 1).

SwarmScene is where the diff engine is cashed in, so these tests assert the
redesign's central performance claim directly and mechanically: a frame in
which only telemetry moved must produce ZERO writes to the instance buffer.
If that ever regresses, the whole rebuild loses its justification, so it is
pinned here rather than left as a design intention.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.model import (  # noqa: E402
    Activity,
    Health,
    NodeKind,
    NodeTelemetry,
    SwarmEdge,
    SwarmNode,
    SwarmSnapshot,
)
from orion_core.swarm.render.scene import SwarmScene  # noqa: E402


def _n(nid, cluster=cl.SYSTEM, **kw) -> SwarmNode:
    base = dict(id=nid, kind=NodeKind.SUBSYSTEM, label=nid, cluster=cluster)
    base.update(kw)
    return SwarmNode(**base)


def _snap(nodes, edges=(), core_state="STANDBY") -> SwarmSnapshot:
    return SwarmSnapshot(nodes=tuple(nodes), edges=tuple(edges),
                         core_state=core_state)


def _graph(n=20, sick=(), calls=0):
    nodes = [SwarmNode(id="core", kind=NodeKind.CORE, label="ORION",
                       cluster=cl.INTELLIGENCE, health=Health.OK)]
    nodes.append(_n("cluster:MEMORY", cluster=cl.MEMORY))
    for i in range(n):
        nodes.append(_n(f"module:m{i}", cluster=cl.MEMORY,
                        health=Health.DOWN if i in sick else Health.OK,
                        telemetry=NodeTelemetry(calls=calls)))
    return _snap(nodes)


# ── first frame ──────────────────────────────────────────────────────────────

def test_the_first_snapshot_populates_the_buffer():
    scene = SwarmScene()
    scene.apply(_graph(5))
    assert scene.buffer.live_count == 7      # core + cluster + 5 modules
    assert scene.buffer.draw_count == 7


def test_every_node_gets_a_position_from_the_layout():
    scene = SwarmScene()
    scene.apply(_graph(5))
    for i in range(5):
        assert scene.position_of(f"module:m{i}") is not None


def test_core_lands_at_the_origin():
    scene = SwarmScene()
    scene.apply(_graph(2))
    assert scene.buffer.position_of("core") == (0.0, 0.0, 0.0)


# ── THE performance thesis ───────────────────────────────────────────────────

def test_a_telemetry_only_frame_writes_nothing_to_the_gpu():
    """The single most important behaviour in the redesign."""
    scene = SwarmScene()
    scene.apply(_graph(200, calls=0))
    scene.buffer.clear_dirty()

    delta = scene.apply(_graph(200, calls=5))       # every node's counters moved

    assert len(delta.telemetry_changed) > 0
    assert delta.visual_changed == ()
    assert scene.buffer.is_dirty is False           # zero GPU writes
    assert scene.buffer.dirty_slot_count == 0


def test_telemetry_refresh_stays_cheap_even_after_a_long_idle_period():
    """Wall-clock time must not re-layout the neural map behind the user."""
    now = [0.0]
    scene = SwarmScene(clock=lambda: now[0])
    scene.apply(_graph(40, calls=0))
    scene.buffer.clear_dirty()
    scene.edges.clear_dirty()

    now[0] = 3600.0
    delta = scene.apply(_graph(40, calls=5))

    assert delta.telemetry_changed
    assert scene.buffer.is_dirty is False
    assert scene.edges.is_dirty is False


def test_an_unchanged_frame_writes_nothing_and_is_reported_empty():
    scene = SwarmScene()
    scene.apply(_graph(50))
    scene.buffer.clear_dirty()
    delta = scene.apply(_graph(50))
    assert delta.is_empty is True
    assert scene.buffer.is_dirty is False


def test_one_failing_subsystem_dirties_exactly_one_slot():
    scene = SwarmScene()
    scene.apply(_graph(500))
    scene.buffer.clear_dirty()

    scene.apply(_graph(500, sick={137}))

    assert scene.buffer.dirty_slot_count == 1
    span = scene.buffer.dirty_range()
    assert span is not None and span[1] - span[0] == 1


def test_the_upload_ratio_reflects_how_rarely_the_gpu_is_touched():
    scene = SwarmScene()
    scene.apply(_graph(30))                       # 1 upload (first frame)
    for calls in range(1, 20):
        scene.apply(_graph(30, calls=calls))      # 19 telemetry-only frames
    stats = scene.stats()
    assert stats["uploads"] == 1
    assert stats["skipped_uploads"] == 19
    assert stats["upload_ratio"] < 0.06


# ── structural change ────────────────────────────────────────────────────────

def test_adding_a_node_allocates_a_slot_without_moving_the_others():
    """Insertion must not displace existing nodes.

    Time is FROZEN here on purpose. Clusters now ride rotating cognition
    rings, so across real time every position legitimately changes; the
    guarantee being pinned is that adding a node does not displace its
    siblings AT A GIVEN INSTANT, which is the property that stops nodes
    jumping when the graph grows."""
    scene = SwarmScene(clock=lambda: 0.0)
    scene.apply(_graph(10))
    before = {f"module:m{i}": scene.buffer.position_of(f"module:m{i}")
              for i in range(10)}

    scene.apply(_graph(11))

    assert scene.buffer.live_count == 13
    for node_id, position in before.items():
        assert scene.buffer.position_of(node_id) == position


def test_removing_a_node_frees_its_slot_for_reuse():
    scene = SwarmScene()
    scene.apply(_graph(10))
    assert scene.buffer.free_slots == 0
    scene.apply(_graph(9))
    assert scene.buffer.free_slots == 1
    scene.apply(_graph(10))
    assert scene.buffer.free_slots == 0        # the freed slot was reused


def test_health_change_rewrites_the_instance_but_never_relayouts():
    scene = SwarmScene()
    scene.apply(_graph(20))
    position = scene.buffer.position_of("module:m3")
    scene.buffer.clear_dirty()

    delta = scene.apply(_graph(20, sick={3}))

    assert delta.needs_layout is False
    assert scene.buffer.position_of("module:m3") == position   # did not move
    assert scene.buffer.dirty_slot_count == 1


def test_a_failing_node_actually_changes_colour_in_the_buffer():
    scene = SwarmScene()
    scene.apply(_graph(10))
    healthy = scene.buffer.row("module:m4")[3:6].copy()
    scene.apply(_graph(10, sick={4}))
    failing = scene.buffer.row("module:m4")[3:6]
    assert not (healthy == failing).all()


def test_a_node_moving_cluster_is_repositioned():
    scene = SwarmScene()
    scene.apply(_snap([_n("cluster:MEMORY", cluster=cl.MEMORY),
                       _n("cluster:SECURITY", cluster=cl.SECURITY),
                       _n("module:x", cluster=cl.MEMORY)]))
    before = scene.buffer.position_of("module:x")
    scene.apply(_snap([_n("cluster:MEMORY", cluster=cl.MEMORY),
                       _n("cluster:SECURITY", cluster=cl.SECURITY),
                       _n("module:x", cluster=cl.SECURITY)]))
    assert scene.buffer.position_of("module:x") != before


# ── selection ────────────────────────────────────────────────────────────────

def test_selecting_a_node_returns_it_and_flags_it_in_the_buffer():
    scene = SwarmScene()
    scene.apply(_graph(10))
    node = scene.select("module:m2")
    assert node is not None and node.id == "module:m2"
    assert float(scene.buffer.row("module:m2")[9]) == 1.0


def test_selecting_never_dirties_more_than_two_slots():
    scene = SwarmScene()
    scene.apply(_graph(400))
    scene.buffer.clear_dirty()
    scene.select("module:m10")
    scene.select("module:m300")
    assert scene.buffer.dirty_slot_count <= 3


def test_selecting_an_unknown_node_clears_the_selection_safely():
    scene = SwarmScene()
    scene.apply(_graph(5))
    scene.select("module:m1")
    assert scene.select("module:does_not_exist") is None
    assert scene.selected is None


def test_a_picking_hit_resolves_from_slot_back_to_the_node():
    scene = SwarmScene()
    scene.apply(_graph(10))
    slot = scene.buffer.slot_of("module:m6")
    assert scene.select_by_slot(slot).id == "module:m6"


def test_selection_survives_an_unrelated_refresh():
    scene = SwarmScene()
    scene.apply(_graph(10))
    scene.select("module:m5")
    scene.apply(_graph(10, calls=3))
    assert scene.selected == "module:m5"


def test_a_newly_added_node_is_not_written_as_selected():
    scene = SwarmScene()
    scene.apply(_graph(5))
    scene.select("module:m1")
    scene.apply(_graph(6))
    assert float(scene.buffer.row("module:m5")[9]) == 0.0


# ── lookups + lifecycle ──────────────────────────────────────────────────────

def test_the_scene_exposes_the_live_node_for_the_inspector():
    scene = SwarmScene()
    scene.apply(_graph(5))
    assert scene.node("module:m1").cluster == cl.MEMORY
    assert scene.node("nope") is None


def test_selected_node_tracks_the_latest_snapshot_not_a_stale_copy():
    """The inspector must show current telemetry, not what was true when the
    node was first clicked."""
    scene = SwarmScene()
    scene.apply(_graph(5, calls=0))
    scene.select("module:m2")
    scene.apply(_graph(5, calls=99))
    assert scene.selected_node().telemetry.calls == 99


def test_reset_clears_everything_for_a_context_loss():
    scene = SwarmScene()
    scene.apply(_graph(10))
    scene.select("module:m1")
    scene.reset()
    assert scene.buffer.live_count == 0
    assert scene.snapshot is None
    assert scene.selected is None


def test_the_scene_rebuilds_cleanly_after_a_reset():
    scene = SwarmScene()
    scene.apply(_graph(10))
    scene.reset()
    scene.apply(_graph(10))
    assert scene.buffer.live_count == 12


def test_stats_report_buffer_and_frame_accounting():
    scene = SwarmScene()
    scene.apply(_graph(8))
    stats = scene.stats()
    assert stats["nodes"] == 10
    assert stats["live"] == 10
    assert stats["frames"] == 1


def test_edges_do_not_consume_instance_slots():
    """Edges are drawn as lines, not instanced spheres; letting them into the
    node buffer would corrupt picking, which maps slot -> node."""
    scene = SwarmScene()
    nodes = [_n("a"), _n("b")]
    scene.apply(_snap(nodes, edges=[SwarmEdge("a", "b")]))
    assert scene.buffer.live_count == 2


def test_activity_change_alone_is_enough_to_rewrite_an_instance():
    scene = SwarmScene()
    scene.apply(_snap([_n("a", activity=Activity.IDLE)]))
    scene.buffer.clear_dirty()
    scene.apply(_snap([_n("a", activity=Activity.BUSY)]))
    assert scene.buffer.dirty_slot_count == 1
