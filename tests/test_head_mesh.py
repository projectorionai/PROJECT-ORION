"""
Head geometry tests — the rig, and the two defects that rendering exposed.

Most of this file exists because the mesh passed every plausible numeric check
while being visibly wrong. Vertex counts, unit normals and landmark ordering
were all correct, and the face still could not open its mouth. So the tests
that matter here assert *behaviour under deformation*, not shape statistics.

Pure numpy: no Qt, no window, no event loop. The renderer is tested separately
and in a subprocess, because standing up a QApplication inside this suite has
been observed to break later Qt tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.gui import head_mesh
from orion_core.gui.head_mesh import LANDMARKS

#: MediaPipe indices used by the assertions below.
UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10


@pytest.fixture(scope="module")
def mesh():
    return head_mesh.build_head()


# ── the asset ────────────────────────────────────────────────────────────────

def test_the_face_asset_is_present_and_the_expected_model():
    verts, faces = head_mesh.load_obj()
    assert verts.shape == (468, 3), "MediaPipe canonical face is 468 vertices"
    assert faces.shape == (898, 3), "MediaPipe canonical face is 898 triangles"


def test_the_asset_carries_its_apache_attribution():
    """The geometry is MediaPipe's under Apache-2.0; the header is the notice."""
    text = head_mesh.OBJ_PATH.read_text(encoding="utf-8")[:1200]
    assert "MediaPipe" in text
    assert "Apache" in text


# ── the head ─────────────────────────────────────────────────────────────────

def test_the_mask_is_closed_into_a_head(mesh):
    """The face model is an open mask; the cranium and neck are built onto it."""
    assert len(mesh["verts"]) > 468
    assert mesh["n_face"] == 468
    assert mesh["n_head"] > mesh["n_face"]
    assert mesh["neck_start"] >= mesh["n_head"]


def test_faces_index_real_vertices_and_are_not_degenerate(mesh):
    faces, verts = mesh["faces"], mesh["verts"]
    assert faces.min() >= 0
    assert faces.max() < len(verts)
    assert (faces[:, 0] != faces[:, 1]).all()
    assert (faces[:, 1] != faces[:, 2]).all()
    assert (faces[:, 0] != faces[:, 2]).all()


def test_normals_are_unit_length(mesh):
    lengths = np.linalg.norm(mesh["normals"], axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-6)


def test_the_coordinate_frame_is_what_the_renderer_assumes(mesh):
    """+x right, +y up, +z out of the face, eyes near y = 0."""
    verts = mesh["verts"]
    eye_y = verts[LANDMARKS["eye_l"] + LANDMARKS["eye_r"]][:, 1].mean()
    assert abs(eye_y) < 0.05, "eyes should land on y = 0"
    assert verts[NOSE_TIP, 2] > 0.2, "the nose must point out of the face (+z)"
    assert verts[CHIN, 1] < verts[FOREHEAD, 1], "chin below forehead"
    assert verts[LANDMARKS["eye_l"]][:, 0].mean() < 0 < verts[LANDMARKS["eye_r"]][:, 0].mean()


def test_the_head_is_taller_than_it_is_wide(mesh):
    """A head that is wider than tall reads as a balloon, not a person."""
    verts = mesh["verts"][:mesh["n_head"]]
    width = verts[:, 0].max() - verts[:, 0].min()
    height = verts[:, 1].max() - verts[:, 1].min()
    assert 0.55 < width / height < 0.85, f"width/height was {width / height:.2f}"


# ── the jaw rig: the defect that made the mouth useless ──────────────────────

def test_the_jaw_rig_separates_the_two_lips(mesh):
    """THE regression test for this module.

    The first rig faded jaw weight smoothly down the face, which gave the upper
    lip 0.956 and the lower lip 0.988 — so a jaw rotation carried the whole
    mouth downward as one piece and the lip GAP changed by under a pixel. Every
    static check still passed: the mesh deformed, the weights were in range, the
    rig ran. Only measuring the gap revealed it.

    The mandible boundary is a step at the mouth line, not a ramp.
    """
    jaw = mesh["jaw"]
    assert jaw[UPPER_INNER_LIP] < 0.1, "the upper lip belongs to the skull"
    assert jaw[LOWER_INNER_LIP] > 0.8, "the lower lip belongs to the mandible"
    assert jaw[LOWER_INNER_LIP] - jaw[UPPER_INNER_LIP] > 0.7


def test_the_upper_face_never_moves_with_the_jaw(mesh):
    jaw = mesh["jaw"]
    assert jaw[NOSE_TIP] < 0.05
    for key in ("eye_l", "eye_r", "brow_l", "brow_r"):
        assert jaw[LANDMARKS[key]].max() < 0.05, f"{key} must not follow the jaw"


def test_the_chin_follows_the_jaw_fully(mesh):
    assert mesh["jaw"][CHIN] > 0.9


def test_the_cranium_is_tapered_rather_than_rigid(mesh):
    """A rigid cranium against a rotating chin shears the join into slivers.

    The first rings of the sweep follow the rim vertex they grew from, so the
    weights there must be between "moves fully" and "does not move at all".
    """
    jaw = mesh["jaw"]
    cranium = jaw[mesh["n_face"]:mesh["n_head"]]
    assert cranium.max() > 0.05, "a fully rigid cranium tears at the chin"
    assert cranium.max() < 0.95, "the cranium must not swing with the jaw"


def test_weights_are_normalised(mesh):
    for key in ("jaw", "lips", "fade"):
        values = mesh[key]
        assert values.min() >= 0.0 and values.max() <= 1.0, key


def test_only_the_face_mask_has_lips(mesh):
    assert mesh["lips"][mesh["n_face"]:].max() == 0.0


# ── landmarks ────────────────────────────────────────────────────────────────

def test_landmarks_sit_where_they_claim(mesh):
    """Guarded at build time too; asserted here so a bad index fails loudly."""
    verts = mesh["verts"]
    eyes = verts[LANDMARKS["eye_l"] + LANDMARKS["eye_r"]][:, 1].mean()
    brows = verts[LANDMARKS["brow_l"] + LANDMARKS["brow_r"]][:, 1].mean()
    lips = verts[LANDMARKS["lips_outer"]][:, 1].mean()
    assert lips < eyes < brows


def test_a_wrong_landmark_index_is_rejected_at_build_time(monkeypatch):
    """A wrong index does not crash — it animates the cheek instead of the eye,
    which survives for months because it merely looks slightly off."""
    broken = dict(LANDMARKS)
    broken["eye_l"] = [400, 401, 402]        # right-hand side of the face
    monkeypatch.setattr(head_mesh, "LANDMARKS", broken)
    with pytest.raises(ValueError):
        head_mesh.build_head()


# ── caching ──────────────────────────────────────────────────────────────────

def test_the_mesh_is_cached_across_callers():
    """Every avatar shares one set of arrays; the renderer copies before it
    deforms, so this saves rebuilding ~20 ms of numpy per face."""
    first = head_mesh.get_head_mesh()
    second = head_mesh.get_head_mesh()
    assert first is second
    assert first["verts"] is second["verts"]


def test_building_the_head_stays_off_the_critical_path():
    """Cheap enough to warm in the background rather than block first paint."""
    import time

    start = time.perf_counter()
    head_mesh.build_head()
    elapsed = (time.perf_counter() - start) * 1000.0
    assert elapsed < 250.0, f"build_head took {elapsed:.0f} ms"
