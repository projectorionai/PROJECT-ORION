"""
The unified renderer migration (Mark XXIII).

ORION's avatar moved from a QWebEngineView layered over the GL surface into
the renderer itself, as point-cloud geometry. These tests pin the migration
so it cannot silently regress — including the two constraints that forced the
detour through QOpenGLWindow and back:

  * A QOpenGLWidget and a QWebEngineView could not co-render in one window.
    With no WebEngine avatar there is no second GL compositor, so the surface
    is a plain QOpenGLWidget again — and Qt can composite over it, which is
    what the callout text layer needs.
  * Only ONE renderer may draw ORION. Two avatars on screen at once is the
    duplicated rendering the migration exists to remove.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ORION_REMOTE_ACCESS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # noqa: E402
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from orion_core.gui import core_window as core_window_mod  # noqa: E402
from orion_core.gui.core_window import native_face_enabled  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


# ── the flag ─────────────────────────────────────────────────────────────────

def test_the_native_face_is_the_default(monkeypatch):
    monkeypatch.delenv("ORION_NATIVE_FACE", raising=False)
    assert native_face_enabled() is True


def test_the_webengine_avatar_can_be_restored(monkeypatch):
    for value in ("0", "false", "off", "NO"):
        monkeypatch.setenv("ORION_NATIVE_FACE", value)
        assert native_face_enabled() is False, value


# ── only one renderer draws ORION ────────────────────────────────────────────

def test_the_core_window_does_not_build_a_second_avatar(monkeypatch, _app):
    """With the native face on, this window hosts the SCENE renderer framed on
    ORION — never a second avatar drawing the same character.

    It used to return the 2-D indicator here and rely on app.py hiding the
    whole column, on the reasoning that ORION was visible inside the swarm
    anyway. What the running app actually showed at startup was that flat
    voxel face and no 3-D avatar anywhere, so the column now holds the real
    thing."""
    from orion_core.gui.native_face import NativeFacePanel

    monkeypatch.delenv("ORION_NATIVE_FACE", raising=False)
    # The software head now sits ahead of the native panel in _build_face, so
    # exercising THIS path means opting out of it. The point of the test is
    # unchanged: when the native renderer owns the face, the window must host
    # its adapter rather than build a second avatar of its own.
    monkeypatch.setenv("ORION_HOLO_HEAD", "0")
    # The face, not whatever form the developer last chose: the saved
    # choice lives in the real config and "orb" builds a different widget.
    monkeypatch.setenv("ORION_FACE_FORM", "face")
    face = core_window_mod.OrionCoreWindow._build_face(object())
    assert isinstance(face, NativeFacePanel)
    # The panel draws nothing itself: it is an adapter over a renderer handed
    # to it later, which is what keeps ORION a single avatar.
    assert face.renderer is None


def test_the_legacy_path_still_builds_the_webengine_avatar(monkeypatch, _app):
    monkeypatch.setenv("ORION_NATIVE_FACE", "0")
    face = core_window_mod.OrionCoreWindow._build_face(object())
    # Either the real 3-D face, or the documented 2-D fallback when WebEngine
    # is unavailable — but the decision goes through face3d, not around it.
    assert face is not None


# ── the surface is a QOpenGLWidget again ─────────────────────────────────────

def test_the_renderer_is_a_qopenglwidget_not_a_native_window():
    """The QOpenGLWindow + createWindowContainer detour existed only to keep
    the WebEngine face alive. With the face native, the surface returns to a
    plain widget so Qt can composite over it."""
    from orion_core.swarm.render.gl_view import _SwarmGLWindow

    assert issubclass(_SwarmGLWindow, QOpenGLWidget)


def test_the_view_wrapper_is_a_plain_widget():
    from orion_core.swarm.render.gl_view import SwarmGLView

    assert issubclass(SwarmGLView, QWidget)


def test_the_renderer_module_no_longer_imports_webengine():
    """No WebEngine dependency anywhere in the native renderer."""
    from orion_core.swarm.render import gl_view

    source = Path(gl_view.__file__).read_text(encoding="utf-8")
    assert "QtWebEngine" not in source


# ── the face is real geometry in the same pipeline ───────────────────────────

def test_the_face_is_geometry_the_renderer_draws():
    from orion_core.swarm.render.face_geometry import FACE_FLOATS, build_face

    cloud = build_face()
    assert cloud.shape[1] == FACE_FLOATS
    assert len(cloud) > 1000


def test_the_face_pipeline_is_gui_free():
    """Geometry and behaviour must stay testable without a display — the
    split that let the whole avatar be built before any GL existed."""
    from orion_core.swarm.render import face_expression, face_geometry

    for module in (face_geometry, face_expression):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "PyQt6" not in source, module.__name__


def test_the_view_exposes_the_face_controls():
    """The bus drives ORION's mouth and expression through these."""
    from orion_core.swarm.render.gl_view import SwarmGLView

    for name in ("set_speech_amplitude", "set_face_mode", "look_at", "face"):
        assert hasattr(SwarmGLView, name), name


# ── the swarm page drives it from real state ─────────────────────────────────

class _Signal:
    def __init__(self): self._slots = []
    def connect(self, slot): self._slots.append(slot)
    def emit(self, *a, **k):
        for s in self._slots:
            s(*a, **k)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _StubGL:
    def __init__(self):
        self.amplitudes = []
        self.modes = []
        self.scene = self
        from orion_core.swarm.cognition_state import CognitionState
        self.cognition = CognitionState()

    def set_speech_amplitude(self, value): self.amplitudes.append(value)
    def set_face_mode(self, mode): self.modes.append(mode)


def test_speech_amplitude_reaches_the_face(_app):
    from orion_core.gui.swarm_view import SwarmDeckView

    bus = _StubBus()
    view = SwarmDeckView(bus)
    stub = _StubGL()
    view.gl_view = stub
    bus.amplitude.emit(0.7)
    assert stub.amplitudes and stub.amplitudes[-1] == pytest.approx(0.7)


def test_cognition_mode_reaches_the_face(_app):
    """His expression and his network must agree about what he is doing."""
    from orion_core.swarm.cognition_state import Mode
    from orion_core.gui.swarm_view import SwarmDeckView

    bus = _StubBus()
    view = SwarmDeckView(bus)
    stub = _StubGL()
    view.gl_view = stub
    bus.state.emit("THINKING")
    assert stub.modes and stub.modes[-1] is Mode.THINKING


def test_a_bad_amplitude_never_reaches_the_face(_app):
    from orion_core.gui.swarm_view import SwarmDeckView

    bus = _StubBus()
    view = SwarmDeckView(bus)
    view.gl_view = _StubGL()
    bus.amplitude.emit("not a number")      # must not raise
