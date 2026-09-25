"""
Tests for orion_core.swarm.render.pick and .mesh (Mark XXII, Phase 1).

Picking is CPU ray/sphere rather than a colour-ID pass precisely so it CAN be
tested like this — no GL context, no offscreen target, no glReadPixels stall.
These tests exercise the real camera matrices, so a convention error (a
transposed matrix, an unflipped y axis) fails here rather than showing up as
"clicking selects the wrong node" in the running app.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from orion_core.swarm.render.camera import OrbitCamera  # noqa: E402
from orion_core.swarm.render.instances import InstanceBuffer  # noqa: E402
from orion_core.swarm.render.mesh import (  # noqa: E402
    DEFAULT_SUBDIVISIONS,
    build_sphere,
    triangle_count,
)
from orion_core.swarm.render.pick import pick_slot, ray_from_screen  # noqa: E402

WIDTH, HEIGHT = 800.0, 600.0


def _camera() -> OrbitCamera:
    cam = OrbitCamera()
    cam.snap_to_goal()
    return cam


def _ray(cam: OrbitCamera, x: float, y: float):
    aspect = WIDTH / HEIGHT
    return ray_from_screen(x, y, WIDTH, HEIGHT,
                           cam.view_projection(aspect), cam.eye())


def _buffer(*nodes) -> InstanceBuffer:
    buf = InstanceBuffer()
    for node_id, position, radius in nodes:
        buf.upsert(node_id, position, (1.0, 1.0, 1.0), 1.0, 0.0, radius)
    return buf


# ── the ray itself ───────────────────────────────────────────────────────────

def test_a_centre_screen_ray_points_straight_at_the_camera_target():
    cam = _camera()
    _, direction = _ray(cam, WIDTH / 2, HEIGHT / 2)
    expected = cam.target - cam.eye()
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(direction, expected, atol=1e-6)


def test_the_ray_direction_is_a_unit_vector():
    _, direction = _ray(_camera(), 123.0, 456.0)
    assert abs(float(np.linalg.norm(direction)) - 1.0) < 1e-9


def test_screen_y_is_flipped_relative_to_ndc():
    """Qt's y grows downward, OpenGL NDC grows upward. Getting this wrong
    gives vertically mirrored picking, which is maddening to diagnose."""
    cam = _camera()
    _, up_dir = _ray(cam, WIDTH / 2, HEIGHT * 0.25)
    _, down_dir = _ray(cam, WIDTH / 2, HEIGHT * 0.75)
    _, camera_up = cam.basis()
    assert float(np.dot(up_dir, camera_up)) > float(np.dot(down_dir, camera_up))


def test_the_ray_stays_correct_after_orbiting():
    cam = _camera()
    cam.orbit(1.1, 0.4)
    cam.snap_to_goal()
    _, direction = _ray(cam, WIDTH / 2, HEIGHT / 2)
    expected = cam.target - cam.eye()
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(direction, expected, atol=1e-6)


# ── hitting things ───────────────────────────────────────────────────────────

def test_clicking_the_centre_selects_a_node_at_the_origin():
    cam = _camera()
    buf = _buffer(("core", (0.0, 0.0, 0.0), 2.6))
    origin, direction = _ray(cam, WIDTH / 2, HEIGHT / 2)
    assert pick_slot(buf.view(), origin, direction, buf.draw_count) == 0


def test_clicking_empty_space_selects_nothing():
    cam = _camera()
    buf = _buffer(("core", (0.0, 0.0, 0.0), 2.6))
    origin, direction = _ray(cam, 4.0, 4.0)
    assert pick_slot(buf.view(), origin, direction, buf.draw_count) is None


def test_the_nearest_node_wins_when_two_line_up():
    """A large distant node must not beat a small one directly in front."""
    cam = _camera()
    towards = cam.target - cam.eye()
    towards = towards / np.linalg.norm(towards)
    near_point = tuple(cam.eye() + towards * 20.0)
    buf = _buffer(("far", (0.0, 0.0, 0.0), 6.0), ("near", near_point, 1.0))
    origin, direction = _ray(cam, WIDTH / 2, HEIGHT / 2)
    slot = pick_slot(buf.view(), origin, direction, buf.draw_count)
    assert buf.id_at(slot) == "near"


def test_nodes_behind_the_camera_are_never_picked():
    cam = _camera()
    behind = tuple(cam.eye() + (cam.eye() - cam.target) * 0.5)
    buf = _buffer(("behind", behind, 4.0))
    origin, direction = _ray(cam, WIDTH / 2, HEIGHT / 2)
    assert pick_slot(buf.view(), origin, direction, buf.draw_count) is None


def test_a_retired_slot_is_never_pickable():
    """Otherwise a click could select a node that no longer exists."""
    cam = _camera()
    buf = _buffer(("core", (0.0, 0.0, 0.0), 2.6))
    buf.remove("core")
    origin, direction = _ray(cam, WIDTH / 2, HEIGHT / 2)
    assert pick_slot(buf.view(), origin, direction, buf.draw_count) is None


def test_an_empty_buffer_picks_nothing():
    cam = _camera()
    buf = InstanceBuffer()
    origin, direction = _ray(cam, WIDTH / 2, HEIGHT / 2)
    assert pick_slot(buf.view(), origin, direction, buf.draw_count) is None


def test_picking_scales_to_the_full_target_graph():
    cam = _camera()
    buf = InstanceBuffer()
    for i in range(1700):
        buf.upsert(f"n{i}", (float(i % 40) * 3, float(i // 400), 0.0),
                   (1.0, 1.0, 1.0), 1.0, 0.0, 0.6)
    buf.upsert("target", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0, 0.0, 2.6)
    origin, direction = _ray(cam, WIDTH / 2, HEIGHT / 2)
    slot = pick_slot(buf.view(), origin, direction, buf.draw_count)
    assert slot is not None


def test_padding_makes_small_nodes_clickable():
    # Pin the distance: this test measures how much slop the PADDING buys, and
    # a pixel offset maps to a world offset that scales with camera distance —
    # so leaving it on the default made the test silently re-tune itself
    # whenever the framing changed (as it did when cognition rings widened the
    # scene). Pinned, it measures padding and nothing else.
    cam = OrbitCamera()
    cam.focus_on((0.0, 0.0, 0.0), distance=60.0)
    cam.snap_to_goal()
    tiny = _buffer(("tool", (0.0, 0.0, 0.0), 0.05))
    origin, direction = _ray(cam, WIDTH / 2, HEIGHT / 2)
    assert pick_slot(tiny.view(), origin, direction, tiny.draw_count,
                     padding=0.0) is not None
    # a click a few pixels off still lands, thanks to the padding
    off_origin, off_direction = _ray(cam, WIDTH / 2 + 3, HEIGHT / 2 + 3)
    assert pick_slot(tiny.view(), off_origin, off_direction, tiny.draw_count,
                     padding=0.0) is None
    assert pick_slot(tiny.view(), off_origin, off_direction, tiny.draw_count,
                     padding=0.6) is not None


# ── mesh ─────────────────────────────────────────────────────────────────────

def test_the_sphere_is_a_unit_sphere():
    """Every vertex being unit length is what lets the shader reuse position
    as the normal, removing an entire vertex attribute."""
    mesh = build_sphere()
    lengths = np.linalg.norm(mesh, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_the_sphere_has_the_expected_triangle_count():
    mesh = build_sphere(DEFAULT_SUBDIVISIONS)
    assert triangle_count(DEFAULT_SUBDIVISIONS) == 320
    assert mesh.shape == (320 * 3, 3)


def test_subdivision_quadruples_the_triangles():
    assert triangle_count(0) == 20
    assert triangle_count(1) == 80
    assert triangle_count(2) == 320


def test_the_mesh_is_float32_and_contiguous_for_upload():
    mesh = build_sphere()
    assert mesh.dtype == np.float32
    assert mesh.flags["C_CONTIGUOUS"]


def test_midpoints_are_shared_between_adjacent_faces():
    """Without a midpoint cache the mesh splits along every edge and the
    vertex count explodes with each subdivision."""
    mesh = build_sphere(2)
    unique = np.unique(np.round(mesh, 5), axis=0)
    # 162 unique vertices for a 2-subdivision icosphere; a fully split mesh
    # would have 960.
    assert len(unique) < 200


def test_zero_subdivisions_yields_the_base_icosahedron():
    assert build_sphere(0).shape == (60, 3)


def test_a_negative_subdivision_count_is_clamped():
    assert build_sphere(-3).shape == (60, 3)
