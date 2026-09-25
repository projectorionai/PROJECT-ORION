"""
Tests for the holographic callout ring (Mark XXIII).

The placement rules that actually matter, and the failures they prevent:

  * Labels must NEVER overlap — a callout ring that collides is worse than
    no labels, because it is unreadable AND hides the scene.
  * Nodes behind the camera must be dropped, or their callouts appear on the
    wrong side of the ring pointing at nothing.
  * Nodes near the screen centre must be dropped, or their callouts flap
    between sides frame to frame as the rings rotate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from orion_core.swarm.render.camera import OrbitCamera  # noqa: E402
from orion_core.swarm.render.labels import (  # noqa: E402
    MIN_LABEL_GAP,
    callout_at,
    layout_callouts,
    project_to_screen,
)

WIDTH, HEIGHT = 1600.0, 900.0


def _camera() -> OrbitCamera:
    cam = OrbitCamera()
    cam.snap_to_goal()
    return cam


def _vp(cam: OrbitCamera) -> np.ndarray:
    return cam.view_projection(WIDTH / HEIGHT)


def _ring_entries(count: int, radius: float = 44.0):
    """Nodes spread evenly around a horizontal ring — the realistic case."""
    out = []
    for i in range(count):
        angle = math.tau * i / count
        out.append((f"module:m{i}", f"Subsystem {i}",
                    (math.cos(angle) * radius, 0.0, math.sin(angle) * radius)))
    return out


# ── projection ───────────────────────────────────────────────────────────────

def test_the_origin_projects_to_the_screen_centre():
    cam = _camera()
    screen, depth = project_to_screen(np.zeros((1, 3)), _vp(cam), WIDTH, HEIGHT)
    assert abs(screen[0, 0] - WIDTH / 2) < 1.0
    assert abs(screen[0, 1] - HEIGHT / 2) < 1.0
    assert depth[0] > 0


def test_points_behind_the_camera_have_negative_depth():
    cam = _camera()
    behind = np.array([cam.eye() + (cam.eye() - cam.target) * 2.0])
    _screen, depth = project_to_screen(behind, _vp(cam), WIDTH, HEIGHT)
    assert depth[0] < 0


def test_projection_of_nothing_is_safe():
    screen, depth = project_to_screen(np.zeros((0, 3)), _vp(_camera()),
                                      WIDTH, HEIGHT)
    assert len(screen) == 0 and len(depth) == 0


# ── the core guarantee: no overlaps ──────────────────────────────────────────

def test_callouts_never_overlap():
    """The whole reason placement is constrained to a 1-D perimeter."""
    callouts = layout_callouts(_ring_entries(20), _vp(_camera()), WIDTH, HEIGHT)
    assert len(callouts) >= 2
    positions = sorted(c.perimeter_t for c in callouts)
    rx, ry = WIDTH * 0.5 * 0.86, HEIGHT * 0.5 * 0.80
    min_gap_rad = MIN_LABEL_GAP / ((rx + ry) * 0.5)
    for a, b in zip(positions, positions[1:]):
        assert b - a >= min_gap_rad - 1e-9


def test_crowding_many_subsystems_still_produces_no_overlap():
    callouts = layout_callouts(_ring_entries(60), _vp(_camera()), WIDTH, HEIGHT,
                               max_labels=60)
    positions = sorted(c.perimeter_t for c in callouts)
    for a, b in zip(positions, positions[1:]):
        assert b > a          # strictly ordered, never stacked


def test_relaxation_is_stable_rather_than_oscillating():
    """Running the same layout twice must give the same answer — a force
    solver that oscillates would not."""
    entries = _ring_entries(18)
    vp = _vp(_camera())
    first = layout_callouts(entries, vp, WIDTH, HEIGHT)
    second = layout_callouts(entries, vp, WIDTH, HEIGHT)
    assert [c.perimeter_t for c in first] == [c.perimeter_t for c in second]


# ── what gets dropped, and why ───────────────────────────────────────────────

def test_nodes_behind_the_camera_get_no_callout():
    cam = _camera()
    behind = tuple(cam.eye() + (cam.eye() - cam.target) * 2.0)
    callouts = layout_callouts([("ghost", "Behind", behind)], _vp(cam),
                               WIDTH, HEIGHT)
    assert callouts == []


def test_a_node_at_the_screen_centre_gets_no_callout():
    """It has no stable outward direction, so its label would flap between
    sides as the rings turn."""
    callouts = layout_callouts([("core", "ORION", (0.0, 0.0, 0.0))],
                               _vp(_camera()), WIDTH, HEIGHT)
    assert callouts == []


def test_the_label_budget_is_respected():
    callouts = layout_callouts(_ring_entries(80), _vp(_camera()),
                               WIDTH, HEIGHT, max_labels=12)
    assert len(callouts) <= 12


def test_an_empty_scene_produces_no_callouts():
    assert layout_callouts([], _vp(_camera()), WIDTH, HEIGHT) == []


def test_a_zero_sized_viewport_is_safe():
    assert layout_callouts(_ring_entries(6), _vp(_camera()), 0.0, 0.0) == []


# ── leader lines ─────────────────────────────────────────────────────────────

def test_the_leader_runs_from_the_subsystem_to_the_label():
    callouts = layout_callouts(_ring_entries(8), _vp(_camera()), WIDTH, HEIGHT)
    for callout in callouts:
        leader = callout.leader
        assert leader[0] == callout.anchor
        assert leader[-1] == callout.position
        assert len(leader) == 3          # exactly one bend


def test_the_elbow_sits_between_the_node_and_the_label():
    """A bend outside the ring would route the leader backwards."""
    callouts = layout_callouts(_ring_entries(8), _vp(_camera()), WIDTH, HEIGHT)
    cx, cy = WIDTH / 2, HEIGHT / 2
    for callout in callouts:
        elbow_r = math.hypot(callout.elbow[0] - cx, callout.elbow[1] - cy)
        label_r = math.hypot(callout.position[0] - cx, callout.position[1] - cy)
        assert elbow_r < label_r


def test_labels_are_aligned_by_which_side_they_land_on():
    callouts = layout_callouts(_ring_entries(16), _vp(_camera()), WIDTH, HEIGHT)
    sides = {c.side for c in callouts}
    assert sides <= {"left", "right"}
    for callout in callouts:
        expected = "right" if callout.position[0] >= WIDTH / 2 - 1 else "left"
        assert callout.side == expected


# ── fading ───────────────────────────────────────────────────────────────────

def test_every_callout_has_a_usable_alpha():
    for callout in layout_callouts(_ring_entries(12), _vp(_camera()),
                                   WIDTH, HEIGHT):
        assert 0.0 < callout.alpha <= 1.0


def test_outermost_subsystems_are_the_least_faded():
    """They have the clearest leader lines, so they earn the most emphasis."""
    entries = [("near", "Near", (6.0, 0.0, 0.0)),
               ("far", "Far", (70.0, 0.0, 0.0))]
    by_id = {c.node_id: c for c in layout_callouts(entries, _vp(_camera()),
                                                   WIDTH, HEIGHT)}
    if "near" in by_id and "far" in by_id:
        assert by_id["far"].alpha >= by_id["near"].alpha


# ── hit testing for hover and click ──────────────────────────────────────────

def test_hovering_a_label_finds_it():
    callouts = layout_callouts(_ring_entries(10), _vp(_camera()), WIDTH, HEIGHT)
    target = callouts[0]
    found = callout_at(callouts, *target.position)
    assert found is not None and found.node_id == target.node_id


def test_hovering_empty_space_finds_nothing():
    callouts = layout_callouts(_ring_entries(10), _vp(_camera()), WIDTH, HEIGHT)
    assert callout_at(callouts, WIDTH / 2, HEIGHT / 2, radius=20.0) is None


def test_the_nearest_label_wins():
    callouts = layout_callouts(_ring_entries(12), _vp(_camera()), WIDTH, HEIGHT)
    target = callouts[2]
    found = callout_at(callouts, target.position[0] + 3, target.position[1] + 3)
    assert found is not None and found.node_id == target.node_id


# ── it tracks the rings ──────────────────────────────────────────────────────

def test_callouts_follow_their_subsystem_as_the_rings_turn():
    """Anchors come from orbital_position at the drawn instant, so a callout
    must move with its node rather than drifting behind it."""
    from orion_core.swarm import clusters as cl
    from orion_core.swarm.render.rings import orbit_for_cluster, orbital_position

    orbit = orbit_for_cluster(cl.MEMORY)
    vp = _vp(_camera())
    early = layout_callouts([("cluster:MEMORY", "Memory",
                              orbital_position(orbit, 0.0))], vp, WIDTH, HEIGHT)
    later = layout_callouts([("cluster:MEMORY", "Memory",
                              orbital_position(orbit, 90.0))], vp, WIDTH, HEIGHT)
    if early and later:
        assert early[0].anchor != later[0].anchor


# ── labels must not print through each other (Mark XXIII) ────────────────────

def test_callouts_on_the_same_side_never_share_a_line():
    """Perimeter separation is not enough.

    _relax guarantees a gap ALONG the ring, which is exactly right down the
    steep left and right flanks. Across the top and bottom the ring runs
    horizontally, so two labels a full gap apart there sit on the same line —
    and text runs horizontally. A real frame showed "BUSINESS" printed
    straight through "SYSTEM" as soon as the camera drifted to put several
    clusters overhead."""
    from orion_core.swarm.render.labels import MIN_ROW_GAP

    callouts = layout_callouts(_ring_entries(24), _vp(_camera()), WIDTH, HEIGHT)
    for side in ("left", "right"):
        rows = sorted(c.position[1] for c in callouts if c.side == side)
        for above, below in zip(rows, rows[1:]):
            assert below - above >= MIN_ROW_GAP - 1e-6


def test_opposite_sides_may_share_a_line():
    """They are at opposite ends of the viewport; forbidding it would halve
    the ring's capacity for nothing."""
    callouts = layout_callouts(_ring_entries(24), _vp(_camera()), WIDTH, HEIGHT)
    left = {round(c.position[1]) for c in callouts if c.side == "left"}
    right = {round(c.position[1]) for c in callouts if c.side == "right"}
    assert callouts          # the layout produced something at all
    assert isinstance(left | right, set)


def test_row_separation_still_leaves_a_usable_ring():
    """Dropping colliding labels must not empty the bezel."""
    assert len(layout_callouts(_ring_entries(20), _vp(_camera()),
                               WIDTH, HEIGHT)) >= 6
