"""Orb camera controls stay opt-in and route only to the Three.js orb."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ORION_REMOTE_ACCESS", "0")

import pytest

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication

from orion_core.bus import OrionBus
from orion_core.gui.core_window import OrionCoreWindow
from orion_core.gui.face3d import QuantumFace3D
from orion_core.memory import MemoryAgent, OrionMemoryMatrix


class _Face:
    form = "orb"

    def __init__(self):
        self.enabled = []
        self.samples = []

    def set_orb_camera_enabled(self, enabled):
        self.enabled.append(enabled)

    def set_orb_hands(self, sample):
        self.samples.append(sample)


class _Gestures:
    def __init__(self):
        self.started = self.stopped = 0

    def start_orb(self):
        self.started += 1
        return type("Result", (), {"ok": True})()

    def stop_orb(self):
        self.stopped += 1


class _Tracker:
    def __init__(self):
        self.running = False
        self.starts = self.stops = 0

    def start(self):
        self.running = True
        self.starts += 1
        return type("Result", (), {"ok": True})()

    def stop(self):
        self.running = False
        self.stops += 1


def test_orb_camera_toggle_routes_samples_and_stops_cleanly(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    bus = OrionBus()
    matrix = OrionMemoryMatrix(tmp_path / "core.db", tmp_path, bus)
    window = OrionCoreWindow(bus, MemoryAgent(matrix, bus))
    face, gestures = _Face(), _Gestures()
    window.face = face
    monkeypatch.setattr(window, "set_face_form", lambda _form: "orb")
    window.attach_orb_controls(gestures)

    assert window.orb_hands_action.isEnabled()
    assert window.set_orb_hand_control(True)
    sample = {"hands": 2, "x": 0.1, "y": 0.0, "spread": 0.8}
    bus.orb_gesture.emit(sample)
    app.processEvents()
    assert face.samples == [sample]
    assert face.enabled[-1] is True
    assert gestures.started == 1

    window.set_orb_hand_control(False)
    assert gestures.stopped == 1
    assert face.enabled[-1] is False
    assert not window.orb_hands_action.isChecked()


def test_threejs_bridge_rejects_hand_input_in_face_form(monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    face = QuantumFace3D()
    scripts = []
    monkeypatch.setattr(face, "_js", lambda script: scripts.append(script) or True)
    sample = {"hands": 2, "x": 0.1, "y": -0.2, "spread": 0.9}
    face.set_form("orb")
    face.set_orb_camera_enabled(True)
    face.set_orb_hands(sample)
    assert any("orionOrbHands" in script and '"spread": 0.9' in script
               for script in scripts)

    scripts.clear()
    face.set_form("face")
    face.set_orb_hands(sample)
    assert not any("orionOrbHands" in script for script in scripts)


def test_failed_hand_worker_unchecks_mode_and_releases_owned_camera(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    bus = OrionBus()
    matrix = OrionMemoryMatrix(tmp_path / "core.db", tmp_path, bus)
    window = OrionCoreWindow(bus, MemoryAgent(matrix, bus))
    face, gestures, tracker = _Face(), _Gestures(), _Tracker()
    window.face = face
    monkeypatch.setattr(window, "set_face_form", lambda _form: "orb")
    window.attach_orb_controls(gestures, tracker)

    assert window.set_orb_hand_control(True)
    assert tracker.starts == 1
    bus.orb_gesture.emit({"hands": 0, "stopped": True})
    app.processEvents()
    assert not window.orb_hands_action.isChecked()
    assert not window._orb_hands_enabled
    assert gestures.stopped == 1
    assert tracker.stops == 1


def test_late_failure_from_previous_camera_session_is_ignored(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    bus = OrionBus()
    matrix = OrionMemoryMatrix(tmp_path / "core.db", tmp_path, bus)
    window = OrionCoreWindow(bus, MemoryAgent(matrix, bus))
    window.face = _Face()
    gestures = _Gestures()
    gestures.orb_session = 2
    monkeypatch.setattr(window, "set_face_form", lambda _form: "orb")
    window.attach_orb_controls(gestures)

    assert window.set_orb_hand_control(True)
    bus.orb_gesture.emit({"hands": 0, "stopped": True, "session": 1})
    app.processEvents()
    assert window._orb_hands_enabled
    bus.orb_gesture.emit({"hands": 0, "stopped": True, "session": 2})
    app.processEvents()
    assert not window._orb_hands_enabled


def test_selected_form_ignores_old_morph_frames_and_behaviour_changes(monkeypatch):
    app = QApplication.instance() or QApplication([])
    face = QuantumFace3D()
    scripts = []
    monkeypatch.setattr(face, "_js", lambda script: scripts.append(script) or True)
    for form, endpoint, opposing in (("orb", "0.000", 1.0), ("face", "1.000", 0.0)):
        face.set_form(form)
        assert f"orionMorph({endpoint})" in scripts[-1]
        scripts.clear()
        for state in ("LISTENING", "IDLE", "SPEAKING", "STANDBY", "OFFLINE"):
            face.set_state(state)
            face.set_morph(opposing)
        assert not any("orionMorph(" in script for script in scripts)
        face._on_page_loaded(True)
        assert any(f"orionMorph({endpoint})" in script for script in scripts)
        scripts.clear()
    assert app is not None


def test_behaviour_frame_uses_one_js_call_and_never_sends_morph(monkeypatch):
    from orion_core.avatar import AvatarController

    app = QApplication.instance() or QApplication([])
    face = QuantumFace3D()
    scripts = []
    # No WebEngine process is needed to check the actual Python bridge.
    face.view = object()
    monkeypatch.setattr(face, "_js", lambda script: scripts.append(script) or True)
    controller = AvatarController()
    controller.attach(face)
    controller.on_state("LISTENING")
    controller.tick(0.1)
    assert len(scripts) == 1
    assert "orionAmplitude(" in scripts[0]
    assert "orionState(" in scripts[0]
    assert "orionPose(" in scripts[0]
    assert "orionMorph(" not in scripts[0]
    face.view = None
    assert app is not None
