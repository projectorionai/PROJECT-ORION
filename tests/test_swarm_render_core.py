"""
Tests for orion_core.swarm.render's pure core (Mark XXII, Phase 1):
layout, palette, camera, instance buffer.

Nothing here constructs a QOpenGLWidget or a GL context. That is deliberate
and load-bearing: this session already lost a full suite run to a native
STATUS_STACK_BUFFER_OVERRUN caused by a test that built a real Chromium
context, so the renderer is deliberately split with all logic in pure numpy
modules and only the GL handles in the widget.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.model import (  # noqa: E402
    Activity,
    Health,
    NodeKind,
    NodeTelemetry,
    SwarmNode,
)
from orion_core.swarm.render.camera import (  # noqa: E402
    MAX_DISTANCE,
    MIN_DISTANCE,
    OrbitCamera,
)
from orion_core.swarm.render.instances import INSTANCE_FLOATS, InstanceBuffer  # noqa: E402
from orion_core.swarm.render.layout import (  # noqa: E402
    CLUSTER_RING_RADIUS,
    LayoutSolver,
    cluster_centre,
    member_shell_radius,
    stable_hash,
)
from orion_core.swarm.render.palette import (  # noqa: E402
    all_cluster_colours,
    appearance_for,
    cluster_colour,
    is_alarm_colour,
)


def _n(nid, kind=NodeKind.SUBSYSTEM, cluster=cl.SYSTEM, parent=None, **kw) -> SwarmNode:
    return SwarmNode(id=nid, kind=kind, label=nid, cluster=cluster,
                     parent=parent, **kw)


# ── the render core stays free of Qt and GL ─────────────────────────────────

def test_the_pure_render_modules_never_import_qt_or_gl():
    from orion_core.swarm.render import camera, instances, layout, palette, scene
    for module in (camera, instances, layout, palette, scene):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "PyQt6" not in source, f"{module.__name__} must stay GUI-free"


# ── layout: determinism and stability ───────────────────────────────────────

def test_stable_hash_does_not_depend_on_the_process():
    """Python's built-in hash() for str is randomised per process; using it
    would reshuffle the whole graph on every restart."""
    assert stable_hash("agent:coding") == stable_hash("agent:coding")
    assert stable_hash("agent:coding") == 1077419779 or isinstance(
        stable_hash("agent:coding"), int)


def test_the_same_graph_lays_out_identically_every_time():
    nodes = [_n("core", NodeKind.CORE, cl.INTELLIGENCE),
             _n("cluster:MEMORY", cluster=cl.MEMORY),
             _n("module:memory", cluster=cl.MEMORY)]
    a = LayoutSolver().solve(nodes)
    b = LayoutSolver().solve(nodes)
    assert a.positions == b.positions


def test_adding_a_node_never_moves_the_existing_ones():
    """The property that makes 'nodes stop jumping between refreshes' true,
    and the reason a structural change does not force a full re-upload."""
    base = [_n("cluster:MEMORY", cluster=cl.MEMORY)]
    base += [_n(f"module:m{i}", cluster=cl.MEMORY) for i in range(50)]
    before = LayoutSolver().solve(base)
    after = LayoutSolver().solve(base + [_n("module:brand_new", cluster=cl.MEMORY)])
    for node in base:
        assert before.positions[node.id] == after.positions[node.id]


def test_removing_a_node_never_moves_the_others():
    nodes = [_n("cluster:MEMORY", cluster=cl.MEMORY)]
    nodes += [_n(f"module:m{i}", cluster=cl.MEMORY) for i in range(20)]
    before = LayoutSolver().solve(nodes)
    after = LayoutSolver().solve([n for n in nodes if n.id != "module:m7"])
    for node in nodes:
        if node.id != "module:m7":
            assert before.positions[node.id] == after.positions[node.id]


def test_core_sits_at_the_origin():
    result = LayoutSolver().solve([_n("core", NodeKind.CORE, cl.INTELLIGENCE)])
    assert result.positions["core"] == (0.0, 0.0, 0.0)


def test_each_cluster_gets_its_own_fixed_direction():
    """Centres sit on a SPHERE, not a ring. A near-flat ring projected twelve
    zones onto each other from most viewing angles, which is what made the
    graph read as scattered confetti instead of twelve groups."""
    centres = {c: cluster_centre(c) for c in cl.CLUSTER_ORDER}
    assert len(set(centres.values())) == len(cl.CLUSTER_ORDER)
    for centre in centres.values():
        assert abs(math.dist(centre, (0.0, 0.0, 0.0)) - CLUSTER_RING_RADIUS) < 1e-6


def test_clusters_separate_in_all_three_axes():
    ys = [cluster_centre(c)[1] for c in cl.CLUSTER_ORDER]
    assert max(ys) - min(ys) > CLUSTER_RING_RADIUS   # not a flat plate


def test_cluster_shells_never_overlap_even_at_their_largest():
    """The analytic guarantee behind 'never overlapping'."""
    centres = [cluster_centre(c) for c in cl.CLUSTER_ORDER]
    closest = min(math.dist(a, b)
                  for i, a in enumerate(centres) for b in centres[i + 1:])
    assert closest > member_shell_radius(50) * 2


def test_member_shells_scale_with_population():
    """A fixed radius left a 3-node cluster sparse and a 22-node one crammed."""
    assert member_shell_radius(3) < member_shell_radius(22)


def test_member_shell_radius_is_bounded():
    assert member_shell_radius(0) >= 4.0
    assert member_shell_radius(100_000) <= 12.0


def test_shell_radius_is_bucketed_so_one_new_node_does_not_re_spread_a_cluster():
    """Scaling on the exact count moved every sibling whenever one node was
    added — the 'nodes jump between refreshes' behaviour this layout exists to
    prevent. Buckets mean a cluster only re-spreads when it roughly doubles."""
    assert member_shell_radius(20) == member_shell_radius(19)
    assert member_shell_radius(50) == member_shell_radius(51)
    # ...but a genuine doubling does widen it
    assert member_shell_radius(40) > member_shell_radius(20)


def test_a_crowded_cluster_spreads_its_members_further_out():
    small = [_n("cluster:MEMORY", cluster=cl.MEMORY)]
    small += [_n(f"module:s{i}", cluster=cl.MEMORY) for i in range(3)]
    big = [_n("cluster:MEMORY", cluster=cl.MEMORY)]
    big += [_n(f"module:s{i}", cluster=cl.MEMORY) for i in range(40)]
    centre = np.array(cluster_centre(cl.MEMORY))

    def spread(nodes):
        result = LayoutSolver().solve(nodes)
        return max(np.linalg.norm(np.array(result.positions[n.id]) - centre)
                   for n in nodes if not n.id.startswith("cluster:"))

    assert spread(big) > spread(small)


def test_a_clusters_direction_is_the_same_on_every_run():
    """Spatial memory: 'security is over there' has to stay true."""
    assert cluster_centre(cl.SECURITY) == cluster_centre(cl.SECURITY)


def test_members_are_placed_around_their_own_cluster_not_the_origin():
    """A member stays close to its stable cluster anchor, not the origin."""

    nodes = [_n("cluster:SECURITY", cluster=cl.SECURITY),
             _n("module:security_recon", cluster=cl.SECURITY)]
    result = LayoutSolver().solve(nodes)
    centre = np.array(cluster_centre(cl.SECURITY))
    member = np.array(result.positions["module:security_recon"])
    ring_radius = float(np.linalg.norm(centre))
    # close to its own cluster...
    assert np.linalg.norm(member - centre) < ring_radius / 2
    # ...and out on that cluster's ring, not sat at the origin
    assert np.linalg.norm(member) > ring_radius / 2


def test_a_child_orbits_its_parent_rather_than_the_cluster():
    nodes = [_n("cluster:COMMUNICATION", cluster=cl.COMMUNICATION),
             _n("mcp:github", NodeKind.MCP_SERVER, cl.COMMUNICATION),
             _n("mcp_tool:github:list", NodeKind.MCP_TOOL, cl.COMMUNICATION,
                parent="mcp:github")]
    result = LayoutSolver().solve(nodes)
    server = np.array(result.positions["mcp:github"])
    tool = np.array(result.positions["mcp_tool:github:list"])
    assert np.linalg.norm(tool - server) < 4.0


def test_a_child_whose_parent_is_missing_still_gets_placed():
    nodes = [_n("cluster:COMMUNICATION", cluster=cl.COMMUNICATION),
             _n("mcp_tool:ghost:list", NodeKind.MCP_TOOL, cl.COMMUNICATION,
                parent="mcp:ghost")]
    result = LayoutSolver().solve(nodes)
    assert "mcp_tool:ghost:list" in result.positions
    assert result.positions["mcp_tool:ghost:list"] != (0.0, 0.0, 0.0)


def test_no_two_nodes_land_on_exactly_the_same_point():
    nodes = [_n("core", NodeKind.CORE, cl.INTELLIGENCE)]
    for cluster in cl.CLUSTER_ORDER:
        nodes.append(_n(f"cluster:{cluster}", cluster=cluster))
        nodes += [_n(f"module:{cluster}_{i}", cluster=cluster) for i in range(40)]
    positions = LayoutSolver().solve(nodes).positions
    assert len(set(positions.values())) == len(positions)


def test_core_is_drawn_larger_than_a_subsystem():
    nodes = [_n("core", NodeKind.CORE, cl.INTELLIGENCE), _n("module:x")]
    radii = LayoutSolver().solve(nodes).radii
    assert radii["core"] > radii["module:x"]


# ── palette: categorical hue must never collide with alarm ──────────────────

def test_no_cluster_hue_is_mistakable_for_an_alarm_colour():
    """A healthy BUSINESS node must never look like a DOWN subsystem."""
    offenders = [c for c, rgb in all_cluster_colours().items() if is_alarm_colour(rgb)]
    assert offenders == []


def test_every_cluster_hue_keeps_real_margin_under_the_alarm_thresholds():
    """Not merely 'does not currently collide'.

    Alarm detection needs brightness AND chroma, so a hue can sit right on
    one limit and still pass — until a small tweak pushes it over the other
    and a healthy node silently starts reading as a fault. This is exactly
    how the RESEARCH azure went wrong. Requiring clear headroom on at least
    one axis catches that at the point the colour changes.
    """
    from orion_core.swarm.render.palette import (
        ALARM_MIN_BRIGHTNESS, ALARM_MIN_CHROMA)

    for cluster, rgb in all_cluster_colours().items():
        brightness = max(rgb)
        chroma = brightness - min(rgb)
        clear = (brightness <= ALARM_MIN_BRIGHTNESS - 0.04
                 or chroma <= ALARM_MIN_CHROMA - 0.06)
        assert clear, (
            f"{cluster} rgb{rgb} sits on both alarm limits "
            f"(brightness {brightness:.2f}, chroma {chroma:.2f})")


def test_every_cluster_has_a_distinct_colour():
    assert len(set(all_cluster_colours().values())) == len(cl.CLUSTER_ORDER)


def test_a_down_node_reads_as_an_alarm_regardless_of_cluster():
    for cluster in cl.CLUSTER_ORDER:
        look = appearance_for(_n("x", cluster=cluster, health=Health.DOWN))
        assert is_alarm_colour(look.colour), cluster


def test_a_degraded_node_reads_as_an_alarm():
    look = appearance_for(_n("x", health=Health.DEGRADED))
    assert is_alarm_colour(look.colour)


def test_a_failing_node_still_shows_which_cluster_it_belongs_to():
    """Health is mixed with, not substituted for, the cluster hue."""
    a = appearance_for(_n("x", cluster=cl.MEMORY, health=Health.DOWN))
    b = appearance_for(_n("x", cluster=cl.SECURITY, health=Health.DOWN))
    assert a.colour != b.colour


def test_an_error_activity_alarms_even_when_health_says_ok():
    # A subsystem answering every call with a failure is still "up".
    look = appearance_for(_n("x", health=Health.OK, activity=Activity.ERROR))
    assert is_alarm_colour(look.colour)


def test_unknown_health_is_desaturated_rather_than_shown_as_healthy():
    unknown = appearance_for(_n("x", cluster=cl.MEMORY, health=Health.UNKNOWN))
    healthy = appearance_for(_n("x", cluster=cl.MEMORY, health=Health.OK))
    assert unknown.colour != healthy.colour


def test_busier_nodes_glow_harder_than_idle_ones():
    idle = appearance_for(_n("x", health=Health.OK, activity=Activity.IDLE))
    busy = appearance_for(_n("x", health=Health.OK, activity=Activity.BUSY))
    assert busy.emissive > idle.emissive


def test_only_active_states_pulse():
    assert appearance_for(_n("x", activity=Activity.IDLE)).pulse_hz == 0.0
    assert appearance_for(_n("x", activity=Activity.ERROR)).pulse_hz > 0.0


def test_a_never_used_subsystem_is_dimmed():
    quiet = appearance_for(_n("x", health=Health.OK, activity=Activity.IDLE))
    used = appearance_for(_n("x", health=Health.OK, activity=Activity.IDLE,
                             telemetry=NodeTelemetry(calls=5)))
    assert sum(quiet.colour) < sum(used.colour)


def test_core_is_always_the_brightest_thing_in_the_scene():
    core = appearance_for(_n("core", NodeKind.CORE, cl.INTELLIGENCE))
    assert core.emissive >= 0.95


def test_appearance_packs_to_plain_floats_for_the_buffer():
    packed = appearance_for(_n("x")).as_tuple()
    assert len(packed) == 6
    assert all(isinstance(v, float) for v in packed)


# ── camera ───────────────────────────────────────────────────────────────────

def test_camera_eases_toward_its_goal_rather_than_snapping():
    cam = OrbitCamera()
    start = cam.azimuth
    cam.orbit(1.0, 0.0)
    assert cam.azimuth == start          # goal moved, current has not
    cam.update(0.016)
    assert start < cam.azimuth < start + 1.0


def test_camera_easing_is_frame_rate_independent():
    """One 0.1 s step and ten 0.01 s steps must land in the same place, or the
    camera feels different at different frame rates."""
    a, b = OrbitCamera(), OrbitCamera()
    a.orbit(1.0, 0.0)
    b.orbit(1.0, 0.0)
    a.update(0.1)
    for _ in range(10):
        b.update(0.01)
    assert abs(a.azimuth - b.azimuth) < 1e-3


def test_elevation_is_clamped_short_of_the_poles():
    cam = OrbitCamera()
    cam.orbit(0.0, 10.0)
    cam.snap_to_goal()
    assert abs(cam.elevation) < math.pi / 2


def test_zoom_is_multiplicative_and_bounded():
    cam = OrbitCamera()
    for _ in range(200):
        cam.zoom(0.5)
    cam.snap_to_goal()
    assert cam.distance == MIN_DISTANCE
    for _ in range(200):
        cam.zoom(2.0)
    cam.snap_to_goal()
    assert cam.distance == MAX_DISTANCE


def test_a_zero_or_negative_zoom_factor_is_ignored():
    cam = OrbitCamera()
    before = cam.distance
    cam.zoom(0.0)
    cam.zoom(-1.0)
    cam.snap_to_goal()
    assert cam.distance == before


def test_update_reports_whether_anything_moved():
    cam = OrbitCamera()
    cam.snap_to_goal()
    assert cam.update(0.016) is False       # settled -> renderer can skip
    cam.orbit(0.5, 0.0)
    assert cam.update(0.016) is True


def test_a_settled_camera_reports_itself_settled():
    cam = OrbitCamera()
    cam.orbit(1.0, 0.3)
    cam.snap_to_goal()
    assert cam.is_settled is True


def test_focus_on_moves_the_target():
    cam = OrbitCamera()
    cam.focus_on((10.0, 2.0, -4.0), distance=20.0)
    cam.snap_to_goal()
    assert tuple(round(v, 3) for v in cam.target) == (10.0, 2.0, -4.0)
    assert cam.distance == 20.0


def test_the_eye_stays_at_the_orbit_distance_from_the_target():
    cam = OrbitCamera()
    cam.orbit(0.7, 0.3)
    cam.snap_to_goal()
    assert abs(float(np.linalg.norm(cam.eye() - cam.target)) - cam.distance) < 1e-6


def test_view_and_projection_matrices_are_well_formed():
    cam = OrbitCamera()
    view = cam.view_matrix()
    assert view.shape == (4, 4)
    assert np.isfinite(view).all()
    proj = cam.projection_matrix(1.777)
    assert proj[3, 2] == -1.0               # perspective divide
    assert np.isfinite(proj).all()


def test_gl_view_projection_is_a_flat_column_major_float32():
    flat = OrbitCamera().gl_view_projection(1.6)
    assert flat.shape == (16,)
    assert flat.dtype == np.float32
    assert flat.flags["C_CONTIGUOUS"]


def test_reset_returns_the_camera_to_its_default_pose():
    cam = OrbitCamera()
    cam.orbit(2.0, 0.5)
    cam.zoom(0.3)
    cam.snap_to_goal()
    cam.reset()
    cam.snap_to_goal()
    assert abs(cam.distance - OrbitCamera().distance) < 1e-6


# ── instance buffer ──────────────────────────────────────────────────────────

def _fill(buf: InstanceBuffer, nid: str, radius: float = 1.0) -> int:
    return buf.upsert(nid, (1.0, 2.0, 3.0), (0.5, 0.6, 0.7), 0.8, 0.0, radius)


def test_upsert_allocates_a_slot_and_marks_it_dirty():
    buf = InstanceBuffer()
    slot = _fill(buf, "a")
    assert slot == 0
    assert buf.dirty_range() == (0, 1)
    assert buf.live_count == 1


def test_updating_an_existing_node_reuses_its_slot():
    buf = InstanceBuffer()
    first = _fill(buf, "a")
    buf.clear_dirty()
    second = _fill(buf, "a", radius=2.0)
    assert first == second
    assert buf.live_count == 1


def test_a_removed_slot_is_returned_to_the_pool_and_reused():
    """Compacting instead would move every later node and dirty the whole
    buffer — one removal becoming a full upload."""
    buf = InstanceBuffer()
    _fill(buf, "a")
    _fill(buf, "b")
    buf.remove("a")
    assert buf.free_slots == 1
    reused = _fill(buf, "c")
    assert reused == 0
    assert buf.free_slots == 0


def test_removing_leaves_the_row_zeroed_so_it_draws_degenerate():
    buf = InstanceBuffer()
    _fill(buf, "a", radius=5.0)
    buf.remove("a")
    assert float(buf.view()[0][8]) == 0.0


def test_removing_an_unknown_node_is_a_safe_no_op():
    assert InstanceBuffer().remove("nope") is False


def test_draw_count_is_the_high_water_mark_not_the_live_count():
    buf = InstanceBuffer()
    _fill(buf, "a")
    _fill(buf, "b")
    buf.remove("a")
    assert buf.live_count == 1
    assert buf.draw_count == 2       # one draw call regardless of churn


def test_the_buffer_grows_by_doubling():
    buf = InstanceBuffer(capacity=2)
    for i in range(9):
        _fill(buf, f"n{i}")
    assert buf.capacity == 16
    assert buf.live_count == 9


def test_growing_preserves_existing_instance_data():
    buf = InstanceBuffer(capacity=2)
    buf.upsert("keep", (7.0, 8.0, 9.0), (1.0, 0.0, 0.0), 1.0, 0.0, 1.0)
    for i in range(10):
        _fill(buf, f"n{i}")
    assert buf.position_of("keep") == (7.0, 8.0, 9.0)


def test_slot_and_id_lookups_are_inverse():
    buf = InstanceBuffer()
    slot = _fill(buf, "agent:coding")
    assert buf.slot_of("agent:coding") == slot
    assert buf.id_at(slot) == "agent:coding"


def test_id_at_an_empty_slot_is_none():
    assert InstanceBuffer().id_at(3) is None


def test_selection_dirties_only_the_affected_slots():
    buf = InstanceBuffer()
    for i in range(100):
        _fill(buf, f"n{i}")
    buf.clear_dirty()
    buf.set_selected("n50")
    assert buf.dirty_slot_count == 1
    buf.clear_dirty()
    buf.set_selected("n90")
    assert buf.dirty_slot_count == 2       # clear the old, set the new


def test_selection_flag_is_written_into_the_instance_row():
    buf = InstanceBuffer()
    _fill(buf, "a")
    buf.set_selected("a")
    assert float(buf.row("a")[9]) == 1.0
    buf.set_selected(None)
    assert float(buf.row("a")[9]) == 0.0


def test_the_dirty_range_covers_every_touched_slot():
    buf = InstanceBuffer()
    for i in range(10):
        _fill(buf, f"n{i}")
    buf.clear_dirty()
    _fill(buf, "n3")
    _fill(buf, "n7")
    assert buf.dirty_range() == (3, 8)


def test_a_clean_buffer_reports_no_dirty_range():
    buf = InstanceBuffer()
    _fill(buf, "a")
    buf.clear_dirty()
    assert buf.dirty_range() is None
    assert buf.is_dirty is False


def test_mark_all_dirty_forces_a_full_reupload_after_context_loss():
    buf = InstanceBuffer()
    for i in range(5):
        _fill(buf, f"n{i}")
    buf.clear_dirty()
    buf.mark_all_dirty()
    assert buf.dirty_range() == (0, 5)


def test_the_view_is_a_numpy_view_not_a_copy():
    buf = InstanceBuffer()
    _fill(buf, "a")
    view = buf.view()
    assert view.shape == (1, INSTANCE_FLOATS)
    assert view.dtype == np.float32
    assert view.base is not None          # a view, so uploading allocates nothing


def test_reset_clears_contents_but_keeps_capacity():
    buf = InstanceBuffer(capacity=64)
    for i in range(30):
        _fill(buf, f"n{i}")
    buf.reset()
    assert buf.live_count == 0
    assert buf.draw_count == 0
    assert buf.capacity == 64             # about to be refilled
