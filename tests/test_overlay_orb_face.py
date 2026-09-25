"""
The compact overlay orb shows ORION (Mark XXIII).

The orb is meant to be ORION's face, always on top, while he keeps listening.
It rendered completely empty. The cause was structural rather than a typo:
when the native renderer owns the face, app.py hides this window's face
column — ORION is drawn inside the swarm, at its centre — and overlay mode
then hid the swarm and the deck too, leaving a 380x400 window containing
nothing at all.

The fix is not a second avatar. It is the same renderer, framed on the one
ORION it already draws. These tests pin that, and pin the framing itself,
because a portrait camera pointed at the wrong bearing photographs his ear
and no headless assertion about "the face is visible" would notice.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.swarm.render.camera import FACE_AZIMUTH, OrbitCamera  # noqa: E402
from orion_core.swarm.render.face_geometry import (  # noqa: E402
    FACING,
    HEAD_RADIUS,
    build_face,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


# ── the camera actually looks at his face ────────────────────────────────────

def test_the_portrait_camera_stands_where_orion_is_looking():
    """His geometry faces +Z and the eye orbits (cos az, ., sin az), so his
    front is azimuth 90 degrees. At the default 35 the render is of his ear —
    which is exactly what the first close-up produced."""
    camera = OrbitCamera()
    camera.face_on(HEAD_RADIUS * 3.0)
    camera.snap_to_goal()
    eye = camera.eye()
    forward = np.array(FACING, dtype=float)
    # The camera must be on the same side as his gaze.
    assert float(eye @ forward) > 0.0
    # ...and nearly straight in front of him, not off to one shoulder.
    bearing = eye / np.linalg.norm(eye)
    assert float(bearing @ forward) > 0.95


def test_the_face_bearing_is_named_once_rather_than_re_derived():
    assert FACE_AZIMUTH == pytest.approx(math.pi / 2.0)


def test_the_portrait_camera_fills_the_view_with_him():
    """Close enough to be a portrait, far enough that his HEAD is not cropped.

    Measured against the head, not the halo: the halo is a loose shell of
    presence that is meant to bleed off the edges of a close framing, and
    sizing the shot to contain it would push him into the distance."""
    from orion_core.swarm.render.face_geometry import FEATURE_HALO

    camera = OrbitCamera()
    camera.face_on(HEAD_RADIUS * 3.15)
    camera.snap_to_goal()
    half_height = camera.distance * math.tan(math.radians(camera.fov_degrees / 2))
    face = build_face()
    head = np.abs(face[face[:, 7] != FEATURE_HALO][:, 1]).max()
    assert head < half_height           # nothing of him is cropped
    assert head > half_height * 0.6     # and he is not a speck in the frame


def test_drift_is_suppressed_while_a_portrait_is_held():
    """A slow orbit would rotate his portrait round to the back of his head
    while the user is looking at it."""
    camera = OrbitCamera()
    camera.face_on(30.0)
    camera.drift_enabled = False
    before = camera._goal_azimuth
    for tick in range(400):
        camera.cinematic_drift(tick * 0.016)
    assert camera._goal_azimuth == before


def test_the_whole_field_holds_a_stable_frame_but_drift_still_works():
    """Drift is OFF in normal operation — camera.py states the rationale
    outright: "a neural map needs a dependable frame of reference." So the
    whole-field camera does not wander by default. Drift is a special-
    presentation mode, and the mechanism still moves the goal once enabled."""
    camera = OrbitCamera()
    before = camera._goal_azimuth
    for tick in range(400):
        camera.cinematic_drift(tick * 0.016)
    assert camera._goal_azimuth == before          # stable frame by default

    camera.drift_enabled = True
    for tick in range(400):
        camera.cinematic_drift(tick * 0.016)
    assert camera._goal_azimuth != before          # ...but drift itself works


# ── the renderer's portrait mode ─────────────────────────────────────────────

def test_portrait_mode_frames_him_and_drops_the_field_chrome(_app):
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    view.set_portrait_mode(True)
    assert view.portrait_mode is True
    assert view.camera.drift_enabled is False
    # The callout ring is a legend for the whole field; at portrait distance
    # it would be a wall of text across his face.
    assert view._window.labels_enabled is False
    assert view.camera._goal_azimuth == pytest.approx(FACE_AZIMUTH)


def test_leaving_portrait_mode_restores_the_whole_field(_app):
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    view.set_portrait_mode(True)
    view.set_portrait_mode(False)
    assert view.portrait_mode is False
    # Back to the whole field: the callout ring returns, and the frame is the
    # stable, non-drifting reference the neural map is navigated against.
    assert view.camera.drift_enabled is False
    assert view._window.labels_enabled is True


def test_resetting_the_camera_also_leaves_portrait_mode(_app):
    """Otherwise pressing R in the orb leaves it in a state nothing clears."""
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    view.set_portrait_mode(True)
    view.reset_camera()
    assert view.portrait_mode is False
    assert view._window.labels_enabled is True


# ── the swarm page's portrait mode strips its chrome ─────────────────────────

def _swarm(_app):
    from orion_core.bus import OrionBus
    from orion_core.gui.swarm_view import SwarmDeckView
    return SwarmDeckView(OrionBus())


def test_the_orb_shows_only_orion_not_a_titled_panel(_app):
    view = _swarm(_app)
    view.set_portrait_mode(True)
    assert view.portrait_mode is True
    assert view._title_label.isVisible() is False
    assert view._inspector_panel.isVisible() is False


def test_leaving_the_orb_puts_the_panel_back(_app):
    view = _swarm(_app)
    view.set_portrait_mode(True)
    view.set_portrait_mode(False)
    assert view.portrait_mode is False
    assert view._outer_layout.contentsMargins().left() > 0


# ── the window ─────────────────────────────────────────────────────────────
#
# Overlay mode no longer borrows this renderer at all: since Mark XXXI the
# compact orb is its own QPainter window (gui/orb_overlay.py) and the main
# window is only hidden. Its tests live in test_compact_orb.py and
# test_core_window_facefirst.py.


# ── the orb is his FACE, not a close-up of the whole field ───────────────────

def test_nothing_is_culled_while_the_whole_field_is_on_show(_app):
    from orion_core.swarm.render.gl_view import SwarmGLView
    assert SwarmGLView()._window.portrait_cull_distance() == 0.0


def test_the_orb_clears_whatever_orbits_between_you_and_him(_app):
    """At portrait range a cluster passing in front fills a third of the frame
    and the orb stops being a face."""
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    view.set_portrait_mode(True)
    view.camera.snap_to_goal()
    cull = view._window.portrait_cull_distance()
    assert cull > 0.0
    # It clears everything level with him or nearer, leaving only what is
    # genuinely behind his head...
    assert cull > view.camera.distance - HEAD_RADIUS
    # ...and never reaches the camera itself, which would cull the field
    # entirely and leave him floating in an empty void.
    assert cull < view.camera.distance


def test_the_cull_is_derived_from_the_framing_not_hard_coded(_app):
    """A magic number would silently stop matching if the portrait distance
    were ever retuned."""
    from orion_core.swarm.render.gl_view import PORTRAIT_DISTANCE, SwarmGLView

    view = SwarmGLView()
    view.set_portrait_mode(True)
    view.camera.snap_to_goal()
    assert view.camera.distance == pytest.approx(PORTRAIT_DISTANCE)
    settled = view._window.portrait_cull_distance()

    # Move in closer and the cull tightens with the shot.
    view.camera.face_on(PORTRAIT_DISTANCE * 0.6)
    view.camera.snap_to_goal()
    assert view._window.portrait_cull_distance() < settled


def test_flying_into_the_orb_does_not_dissolve_the_whole_graph(_app):
    """Entering eases in from the whole field's distance. A cull scaled to
    THAT would exceed the radius of the entire graph, so every node would
    vanish on the way in and swim back as the camera arrived."""
    from orion_core.swarm.render.gl_view import PORTRAIT_DISTANCE, SwarmGLView

    view = SwarmGLView()
    view.set_portrait_mode(True)          # goal set; camera still far away
    assert view.camera.distance > PORTRAIT_DISTANCE
    mid_flight = view._window.portrait_cull_distance()
    view.camera.snap_to_goal()
    assert mid_flight == pytest.approx(view._window.portrait_cull_distance())


def test_orion_himself_is_never_culled(_app):
    """The face pass must not declare the cull at all — he is the thing it
    exists to protect."""
    from orion_core.swarm.render import gl_view as g
    assert "u_portrait_cull" not in g._FACE_VERTEX_SHADER


def test_every_pass_that_can_obscure_him_honours_the_cull():
    from orion_core.swarm.render import gl_view as g
    for source in (g._VERTEX_SHADER, g._EDGE_VERTEX_SHADER,
                   g._PARTICLE_VERTEX_SHADER):
        assert "u_portrait_cull" in source


def test_hovering_is_suppressed_in_the_orb(_app):
    """Culled nodes are still pickable, so a readout would unfold for
    something the user cannot see, anchored to empty space."""
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    from orion_core.swarm.render.gl_view import SwarmGLView
    from orion_core.swarm.sources import compose

    view = SwarmGLView()
    window = view._window
    window.apply_snapshot(compose())
    window.set_portrait_mode(True)

    event = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(10.0, 10.0),
                        QPointF(10.0, 10.0), Qt.MouseButton.NoButton,
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    window.mouseMoveEvent(event)
    assert window.hover.target is None


def test_entering_the_orb_closes_a_readout_left_open_behind_it(_app):
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    window = view._window
    window.hover.point_at("cluster:MEMORY")
    window.set_portrait_mode(True)
    assert window.hover.target is None
