"""
Tests for ORION's face as native scene geometry (Mark XXIII).

This is the migration that takes the avatar out of a WebEngine overlay and
puts it INSIDE the 3D scene. The properties pinned here are the ones that
make that migration worth doing:

  * It is a point cloud in the same buffer format the particle pipeline
    already instances — no second renderer, no mesh pipeline.
  * It has a real face, not a sphere: features must survive the deformation
    with enough density to be visible.
  * Animation is driven by per-point FEATURE tags in the shader, so blinking
    and speech never rewrite positions on the CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from orion_core.swarm.render.face_geometry import (  # noqa: E402
    FACE_FLOATS,
    FEATURE_BROW,
    FEATURE_EYE_L,
    FEATURE_EYE_R,
    FEATURE_HALO,
    FEATURE_MOUTH,
    FEATURE_SKIN,
    HEAD_RADIUS,
    bounds,
    build_face,
    feature_counts,
)


# ── it is upload-ready point-cloud data ──────────────────────────────────────

def test_the_face_is_a_float32_point_cloud():
    face = build_face()
    assert face.dtype == np.float32
    assert face.flags["C_CONTIGUOUS"]
    assert face.shape[1] == FACE_FLOATS


def test_the_face_is_dense_enough_to_read_as_a_surface():
    assert len(build_face()) > 4000


def test_the_face_costs_only_one_upload():
    """A couple of megabytes, written ONCE — the same deal as the particle
    field, which is what lets the avatar live in the shared pipeline.

    The budget was half a megabyte while the head was 8,700 points. It had to
    grow: at that density the points did not tile the surface, so the sculpted
    nose and brow had nothing continuous to shade and the whole face rendered
    as a fog ball. The ceiling still exists — this is a one-time upload, not a
    licence for unbounded geometry."""
    assert build_face().nbytes < 3 * 1024 * 1024


# ── it is a FACE, not a sphere ───────────────────────────────────────────────

def test_every_feature_is_present():
    counts = feature_counts(build_face())
    for feature in ("skin", "eye_left", "eye_right", "mouth", "brow", "halo"):
        assert counts[feature] > 0, feature


def test_features_are_dense_enough_to_be_visible():
    """Tagging skin alone gave each eye ~25 points out of 5,200 — the face
    had no visible eyes at all. Features are generated in their own right."""
    counts = feature_counts(build_face())
    assert counts["eye_left"] > 150
    assert counts["eye_right"] > 150
    assert counts["mouth"] > 150


def test_the_eyes_are_symmetric():
    counts = feature_counts(build_face())
    assert abs(counts["eye_left"] - counts["eye_right"]) < counts["eye_left"] * 0.3


def test_the_eyes_sit_on_opposite_sides_of_the_face():
    face = build_face()
    left = face[face[:, 7] == FEATURE_EYE_L][:, 0].mean()
    right = face[face[:, 7] == FEATURE_EYE_R][:, 0].mean()
    assert left < 0 < right


def test_the_mouth_sits_below_the_eyes():
    face = build_face()
    eye_y = face[face[:, 7] == FEATURE_EYE_L][:, 1].mean()
    mouth_y = face[face[:, 7] == FEATURE_MOUTH][:, 1].mean()
    assert mouth_y < eye_y


def test_the_brows_sit_above_the_eyes():
    face = build_face()
    eye_y = face[face[:, 7] == FEATURE_EYE_L][:, 1].mean()
    brow_y = face[face[:, 7] == FEATURE_BROW][:, 1].mean()
    assert brow_y > eye_y


def test_the_features_face_forward():
    """A face whose eyes were on the back of the head would look correct from
    exactly one angle and wrong from every other."""
    face = build_face()
    for feature in (FEATURE_EYE_L, FEATURE_EYE_R, FEATURE_MOUTH):
        assert face[face[:, 7] == feature][:, 2].mean() > 0


def test_the_head_is_taller_than_it_is_wide():
    """A plain sphere reads as a ball however it is lit."""
    (min_x, min_y, _), (max_x, max_y, _) = bounds(build_face())
    assert (max_y - min_y) > (max_x - min_x)


def test_the_jaw_is_narrower_than_the_cranium():
    face = build_face()
    skin = face[face[:, 7] == FEATURE_SKIN]
    upper = skin[skin[:, 1] > HEAD_RADIUS * 0.35]
    lower = skin[skin[:, 1] < -HEAD_RADIUS * 0.45]
    assert np.abs(lower[:, 0]).max() < np.abs(upper[:, 0]).max()


# ── the halo ─────────────────────────────────────────────────────────────────

def test_the_halo_surrounds_the_head_rather_than_sitting_on_it():
    face = build_face()
    skin = face[face[:, 7] == FEATURE_SKIN]
    halo = face[face[:, 7] == FEATURE_HALO]
    skin_r = np.linalg.norm(skin[:, 0:3], axis=1).mean()
    halo_r = np.linalg.norm(halo[:, 0:3], axis=1).mean()
    assert halo_r > skin_r


# ── shader-side animation ────────────────────────────────────────────────────

def test_every_point_carries_a_feature_tag():
    """Blink and speech are applied per-feature in the shader; an untagged
    point could not be animated without rewriting positions on the CPU."""
    tags = build_face()[:, 7]
    known = {FEATURE_SKIN, FEATURE_EYE_L, FEATURE_EYE_R, FEATURE_MOUTH,
             FEATURE_BROW, FEATURE_HALO}
    assert set(np.unique(tags).tolist()) <= known


def test_points_carry_independent_phases():
    """Shared phase would make the whole face shimmer in lockstep."""
    phase = build_face()[:, 8]
    assert phase.max() - phase.min() > 3.0


def test_every_point_has_a_colour_and_a_size():
    face = build_face()
    assert (face[:, 3:6] > 0).any(axis=1).all()
    assert (face[:, 6] > 0).all()


def test_the_eye_has_a_bright_core_in_a_dark_rim():
    """An eye is graded, not uniformly bright.

    It used to be asserted that the eye was brighter than skin ON AVERAGE,
    and a flat white patch satisfied that. On a real render those patches
    read as two headlamps and drowned every bit of modelling around them. An
    eye needs an iris brighter than skin AND a lash edge darker than it —
    that contrast is what seats it in its socket."""
    face = build_face()
    skin = face[face[:, 7] == FEATURE_SKIN][:, 3:6].mean()
    eye = face[face[:, 7] == FEATURE_EYE_L][:, 3:6].mean(axis=1)
    assert eye.max() > skin        # the iris
    assert eye.min() < skin * 0.5  # the lash line


def test_the_eyes_are_graded_from_their_own_centre():
    """The grade is measured from the eye's position ON the head surface.

    Passing the raw feature coordinate — which is not a unit direction — put
    that centre inside the skull, so every point measured as far away, the
    whole eye graded to the lash colour, and both eyes rendered as black
    sockets with no iris at all."""
    face = build_face()
    eye = face[face[:, 7] == FEATURE_EYE_L]
    brightness = eye[:, 3:6].mean(axis=1)
    centre = eye[np.argmax(brightness), 0:3]
    distance = np.linalg.norm(eye[:, 0:3] - centre, axis=1)
    near = brightness[distance < np.median(distance)].mean()
    far = brightness[distance >= np.median(distance)].mean()
    assert near > far


# ── determinism and scaling ──────────────────────────────────────────────────

def test_the_face_is_identical_on_every_launch():
    assert np.array_equal(build_face(), build_face())


def test_resolution_scales_the_whole_face_including_features():
    """Adaptive quality thins the face without losing its eyes."""
    low = feature_counts(build_face(skin_points=1300, halo_points=400))
    high = feature_counts(build_face(skin_points=5200, halo_points=1800))
    assert low["skin"] < high["skin"]
    assert 0 < low["eye_left"] < high["eye_left"]


def test_an_empty_face_is_handled():
    face = build_face(skin_points=0, halo_points=0)
    assert face.shape[1] == FACE_FLOATS
    assert bounds(face) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


# ── it is a lit SOLID, not a fog ball (Mark XXIII rebuild) ───────────────────
#
# The first native avatar deformed a sphere and shaded it by camera distance
# alone. On a real display that produced a ball of blue fog with two bright
# specks — no nose, no brow, and the back of the skull printing straight
# through the front. Every test below pins one of the things that fixed it,
# because none of them are visible to a test that only counts points.

def test_every_point_carries_a_unit_surface_normal():
    """Without normals there is no lighting, and without lighting a point
    cloud has no form at any density."""
    from orion_core.swarm.render.face_geometry import build_face as _build
    normals = _build()[:, 9:12]
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-3)


def test_normals_follow_the_sculpt_rather_than_the_underlying_sphere():
    """A radial normal describes the ellipsoid the features were carved out
    of, so the nose and the cheek beside it would light identically."""
    from orion_core.swarm.render.face_geometry import _normals, _sculpt

    # Two directions either side of the nose ridge.
    left = np.array([[-0.09, -0.06, 0.99]])
    right = np.array([[0.09, -0.06, 0.99]])
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    n_left, n_right = _normals(left)[0], _normals(right)[0]
    # They must lean in OPPOSITE directions across the ridge...
    assert n_left[0] < -0.2 and n_right[0] > 0.2
    # ...which a radial normal would not do to anything like this degree.
    assert abs(n_left[0] - left[0, 0]) > 0.3
    del _sculpt


def test_the_nose_actually_protrudes():
    """The single most important feature. Without it the cloud reads as a
    mask rather than a head."""
    from orion_core.swarm.render.face_geometry import profile_depth
    assert profile_depth(-0.06) > profile_depth(-0.06, 0.35) + 0.15


def test_the_brow_stands_proud_of_the_eye_socket():
    from orion_core.swarm.render.face_geometry import profile_depth
    assert profile_depth(0.30, 0.31) > profile_depth(0.14, 0.31)


def test_the_face_narrows_to_a_chin():
    from orion_core.swarm.render.face_geometry import profile_depth
    assert profile_depth(-0.68) < profile_depth(-0.06)


def test_facial_relief_does_not_wrap_around_the_back_of_the_skull():
    """Without a front weighting the nose ridge becomes a fin down his neck."""
    from orion_core.swarm.render.face_geometry import _sculpt
    back = np.array([[0.0, -0.06, -1.0]])
    beside = np.array([[0.25, -0.06, -0.97]])
    beside /= np.linalg.norm(beside)
    assert abs(_sculpt(back)[0, 2] - _sculpt(beside)[0, 2] * 1.0) < 2.0


def test_the_scatter_never_swallows_the_features_it_sits_on():
    """Jitter breaks the Fibonacci regularity so the surface reads as a swarm
    holding a shape. At the original 0.045 the scatter was +-0.47 world units
    — deeper than the nose it was scattering, so the head could only ever be
    a fog ball however well it was lit."""
    from orion_core.swarm.render.face_geometry import HEAD_RADIUS, profile_depth
    import inspect as _inspect

    from orion_core.swarm.render import face_geometry as fg
    jitter = _inspect.signature(fg.build_face).parameters["jitter"].default
    nose = (profile_depth(-0.06) - profile_depth(-0.06, 0.35)) * HEAD_RADIUS
    assert jitter * HEAD_RADIUS < nose * 0.5
