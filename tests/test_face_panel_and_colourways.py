"""
ORION's face on the primary monitor, and what colour he is.

Everything here comes from one report: the running app opened on a flat 2-D
voxel face, showed no 3-D avatar anywhere, drew two identical swarms when the
Command Deck was opened, and could not navigate to the chess page at all.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from orion_core.swarm.render.face_geometry import (  # noqa: E402
    DEFAULT_COLOURWAY,
    FEATURE_EYE_L,
    FEATURE_HALO,
    FEATURE_SKIN,
    build_face,
    colourway_names,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _panel():
    from orion_core.gui.native_face import NativeFacePanel
    return NativeFacePanel()


class _GL:
    def __init__(self) -> None:
        self.mode = None
        self.amplitude = None
        self.looked = None
        self.portrait = False
        self.scene = type("_S", (), {"cognition": None})()

    def set_face_mode(self, mode): self.mode = mode
    def set_speech_amplitude(self, value): self.amplitude = float(value)
    def look_at(self, x, y): self.looked = (x, y)


class _Renderer(QWidget):
    """A stand-in for SwarmDeckView. A real QWidget, because the panel puts it
    into its own layout."""

    def __init__(self) -> None:
        super().__init__()
        self.gl_view = _GL()

    def set_portrait_mode(self, enabled): self.gl_view.portrait = bool(enabled)


# ── the face panel is an adapter, not a second avatar ────────────────────────

def test_the_panel_draws_nothing_until_it_is_given_a_renderer(_app):
    """The window is built long before app.py has backends; creating a GL
    context during construction would put it on the startup path."""
    panel = _panel()
    assert panel.renderer is None
    # ...and every signal that may arrive meanwhile is survivable.
    panel.set_state("THINKING")
    panel.set_amplitude(0.4)
    panel.set_speaking(True)
    panel.set_pose(0.1, 0.2)
    panel.apply_emotion("calm", {})


def test_attaching_the_renderer_frames_him_on_his_face(_app):
    panel = _panel()
    renderer = _Renderer()
    panel.attach_renderer(renderer)
    assert panel.renderer is renderer
    assert renderer.gl_view.portrait is True


def test_state_that_arrived_before_the_renderer_is_replayed(_app):
    """Startup emits state well before the GL view exists; dropping it would
    leave him idle-faced until the next change."""
    panel = _panel()
    panel.set_state("THINKING")
    renderer = _Renderer()
    panel.attach_renderer(renderer)
    assert renderer.gl_view.mode == "thinking"


def test_attaching_twice_is_harmless(_app):
    panel = _panel()
    renderer = _Renderer()
    panel.attach_renderer(renderer)
    panel.attach_renderer(renderer)
    assert panel.renderer is renderer


def test_the_bus_signals_reach_the_renderer(_app):
    panel = _panel()
    renderer = _Renderer()
    panel.attach_renderer(renderer)

    panel.set_state("SPEAKING")
    assert renderer.gl_view.mode == "speaking"
    panel.set_amplitude(0.62)
    assert renderer.gl_view.amplitude == pytest.approx(0.62)
    panel.set_pose(0.3, -0.2)
    assert renderer.gl_view.looked == (0.3, -0.2)


def test_an_unrecognised_state_leaves_his_expression_alone(_app):
    """The state string has grown over many passes; a stray value must not
    blank his face."""
    panel = _panel()
    renderer = _Renderer()
    panel.attach_renderer(renderer)
    panel.set_state("SPEAKING")
    panel.set_state("WAT")
    assert renderer.gl_view.mode == "speaking"


def test_it_carries_the_timer_the_2d_rig_had(_app):
    """Overlay mode retunes face.timer; the attribute must exist even though
    the renderer drives its own frame loop."""
    assert _panel().timer is not None


def test_a_label_can_be_shown_under_him(_app):
    panel = _panel()
    panel.set_label("Double-click to restore")
    assert "restore" in panel.label.text()


def test_bad_numbers_never_reach_the_renderer(_app):
    panel = _panel()
    renderer = _Renderer()
    panel.attach_renderer(renderer)
    panel.set_amplitude("loud")
    panel.set_pose(None, None)
    assert renderer.gl_view.amplitude is None


# ── colourways ───────────────────────────────────────────────────────────────

def test_orion_is_crimson_by_default():
    """He was crimson before the native rebuild; blue arrived with it."""
    assert DEFAULT_COLOURWAY == "crimson"
    skin = build_face(skin_points=2000, halo_points=200)
    skin = skin[skin[:, 7] == FEATURE_SKIN][:, 3:6].mean(axis=0)
    assert skin[0] > skin[2] * 2.0        # red dominant, not blue


def test_blue_is_still_available():
    face = build_face(skin_points=2000, halo_points=200, colourway="blue")
    skin = face[face[:, 7] == FEATURE_SKIN][:, 3:6].mean(axis=0)
    assert skin[2] > skin[0] * 1.3


def test_both_colourways_are_offered():
    assert set(colourway_names()) == {"blue", "crimson"}


def test_the_eyes_take_the_colourway_too():
    """A crimson ORION with two blue eyes would look like a bug, because it
    would be one."""
    crimson = build_face(skin_points=2000, halo_points=200, colourway="crimson")
    eye = crimson[crimson[:, 7] == FEATURE_EYE_L][:, 3:6]
    brightest = eye[np.argmax(eye.mean(axis=1))]
    assert brightest[0] >= brightest[2]


def test_every_colourway_keeps_its_eye_contrast():
    """The iris must out-read the skin and the lash line must undercut it, or
    the eye stops sitting in its socket."""
    for name in colourway_names():
        face = build_face(skin_points=3000, halo_points=300, colourway=name)
        skin = face[face[:, 7] == FEATURE_SKIN][:, 3:6].mean()
        eye = face[face[:, 7] == FEATURE_EYE_L][:, 3:6].mean(axis=1)
        assert eye.max() > skin, name
        assert eye.min() < skin * 0.6, name


def test_no_channel_sits_on_the_clipping_cliff():
    """A channel that reaches exactly 1.0 came back as ~0 through this
    renderer's readback path — that is what turned the old skin's blue 1.00
    into yellow-green speckles across his temple."""
    for name in colourway_names():
        face = build_face(skin_points=2000, halo_points=200, colourway=name)
        assert face[:, 3:6].max() < 1.0, name


def test_an_unknown_colourway_still_gives_him_a_face():
    """It comes from an environment variable; a typo must not leave him
    invisible."""
    face = build_face(skin_points=1000, halo_points=100, colourway="chartreuse")
    assert len(face) > 0
    assert face[:, 3:6].max() > 0.0


def test_the_halo_follows_the_colourway(_app):
    crimson = build_face(skin_points=1500, halo_points=400, colourway="crimson")
    halo = crimson[crimson[:, 7] == FEATURE_HALO][:, 3:6].mean(axis=0)
    assert halo[0] > halo[2]


# ── never two of the same graph ──────────────────────────────────────────────

def _window():
    from orion_core.gui.core_window import OrionCoreWindow
    return OrionCoreWindow.__new__(OrionCoreWindow)


class _Rail:
    def __init__(self, visible=True): self.visible = visible
    def hide(self): self.visible = False
    def isHidden(self): return not self.visible


class _Bus:
    def __init__(self): self.messages = []
    log = property(lambda self: self)
    def emit(self, text): self.messages.append(text)


def test_opening_the_deck_on_swarm_collapses_the_rail(_app):
    """Both render the same graph from the same backends, so having both open
    put two identical swarms side by side."""
    window = _window()
    window.swarm_rail = _Rail()
    window.bus = _Bus()
    window._collapse_rail_for_deck("SWARM")
    assert window.swarm_rail.visible is False


def test_opening_the_deck_on_any_other_page_leaves_the_rail_alone(_app):
    """The rail beside CHESS is a perfectly reasonable layout."""
    window = _window()
    window.swarm_rail = _Rail()
    window.bus = _Bus()
    for page in ("CHESS", "MISSION", "COMMAND CENTRE", ""):
        window._collapse_rail_for_deck(page)
        assert window.swarm_rail.visible is True, page


def test_an_already_collapsed_rail_is_not_announced_again(_app):
    window = _window()
    window.swarm_rail = _Rail(visible=False)
    window.bus = _Bus()
    window._collapse_rail_for_deck("SWARM")
    assert window.bus.messages == []
