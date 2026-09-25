"""
Tests for the camera preview surfaces (snapshot still + live tracker view).

Headless (offscreen Qt) — no display and no physical camera required. The
tracker is stubbed: these windows are strictly consumers and must never open
a camera handle of their own (a second handle on one webcam is the Windows
MSMF -1072873821 failure the whole borrow-the-frame design exists to avoid).
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from orion_core.gui.camera_preview import (
    LiveCameraWindow,
    SnapshotPreviewWindow,
    bgr_to_qimage,
)

np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _Signal:
    def __init__(self) -> None:
        self._slots = []

    def emit(self, *args) -> None:
        for slot in self._slots:
            slot(*args)

    def connect(self, slot) -> None:
        self._slots.append(slot)


class _StubBus:
    def __init__(self) -> None:
        self.camera_frame = _Signal()

    def __getattr__(self, name):
        return _Signal()


class _StubTracker:
    def __init__(self, frame=None, capturing: bool = True) -> None:
        self._frame = frame
        self._capturing = capturing

    def is_capturing(self) -> bool:
        return self._capturing

    def latest_frame(self):
        return self._frame


def _frame(width: int = 64, height: int = 48):
    """A synthetic BGR frame, the shape OpenCV hands back."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _jpeg_bytes(width: int = 32, height: int = 24) -> bytes:
    pillow = pytest.importorskip("PIL.Image")
    image = pillow.new("RGB", (width, height), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


# ── bgr_to_qimage ────────────────────────────────────────────────────────────

def test_bgr_to_qimage_returns_none_for_none():
    assert bgr_to_qimage(None) is None


def test_bgr_to_qimage_converts_a_real_frame(_app):
    pytest.importorskip("cv2")
    image = bgr_to_qimage(_frame(64, 48))
    assert image is not None
    assert image.width() == 64
    assert image.height() == 48


def test_bgr_to_qimage_survives_a_malformed_frame(_app):
    """A garbage 'frame' must degrade to None, never raise onto the GUI thread."""
    assert bgr_to_qimage("not a frame") is None


# ── SnapshotPreviewWindow ────────────────────────────────────────────────────

def test_snapshot_window_starts_hidden(_app):
    window = SnapshotPreviewWindow(_StubBus())
    assert window.isVisible() is False


def test_snapshot_window_shows_on_a_snapshot_bus_event(_app):
    bus = _StubBus()
    window = SnapshotPreviewWindow(bus)
    bus.camera_frame.emit({"kind": "snapshot", "jpeg": _jpeg_bytes(), "note": "camera 0"})
    assert window.isVisible() is True
    assert window.image_label.pixmap() is not None
    assert "camera 0" in window.title_label.text()


def test_snapshot_window_ignores_other_payload_kinds(_app):
    bus = _StubBus()
    window = SnapshotPreviewWindow(bus)
    bus.camera_frame.emit({"kind": "something_else", "jpeg": _jpeg_bytes()})
    assert window.isVisible() is False


def test_snapshot_window_ignores_a_non_dict_payload(_app):
    bus = _StubBus()
    window = SnapshotPreviewWindow(bus)
    bus.camera_frame.emit("not a dict")
    assert window.isVisible() is False


def test_snapshot_window_ignores_undecodable_bytes(_app):
    bus = _StubBus()
    window = SnapshotPreviewWindow(bus)
    bus.camera_frame.emit({"kind": "snapshot", "jpeg": b"not a jpeg at all"})
    assert window.isVisible() is False


def test_snapshot_show_snapshot_scales_and_reveals(_app):
    window = SnapshotPreviewWindow(_StubBus())
    pixmap = QPixmap(400, 300)
    window.show_snapshot(pixmap, "test")
    assert window.isVisible() is True
    shown = window.image_label.pixmap()
    assert shown is not None
    assert shown.width() <= 300 and shown.height() <= 220


# ── LiveCameraWindow ─────────────────────────────────────────────────────────

def test_live_window_starts_hidden_and_not_polling(_app):
    window = LiveCameraWindow(_StubBus(), tracker=_StubTracker())
    assert window.isVisible() is False
    assert window._timer.isActive() is False


def test_live_window_start_shows_and_polls(_app):
    window = LiveCameraWindow(_StubBus(), tracker=_StubTracker(_frame()))
    window.start()
    assert window.isVisible() is True
    assert window._timer.isActive() is True
    window.stop()


def test_live_window_stop_hides_and_halts_polling(_app):
    window = LiveCameraWindow(_StubBus(), tracker=_StubTracker(_frame()))
    window.start()
    window.stop()
    assert window.isVisible() is False
    assert window._timer.isActive() is False


def test_live_window_renders_a_tracker_frame(_app):
    pytest.importorskip("cv2")
    window = LiveCameraWindow(_StubBus(), tracker=_StubTracker(_frame()))
    window.start()
    window._refresh()
    assert window.image_label.pixmap() is not None
    window.stop()


def test_live_window_reports_when_tracking_is_off(_app):
    window = LiveCameraWindow(_StubBus(), tracker=_StubTracker(capturing=False))
    window.start()
    window._refresh()
    assert "off" in window.image_label.text().lower()
    window.stop()


def test_live_window_without_a_tracker_says_so(_app):
    window = LiveCameraWindow(_StubBus(), tracker=None)
    window.start()
    window._refresh()   # must not raise with no tracker attached
    assert "not available" in window.image_label.text().lower()
    window.stop()


def test_live_window_refresh_is_a_noop_while_hidden(_app):
    """Polling a hidden window is wasted work every 100ms — and the timer is
    stopped on hide precisely so it cannot happen."""
    window = LiveCameraWindow(_StubBus(), tracker=_StubTracker(_frame()))
    window.start()
    window.hide()
    assert window._timer.isActive() is False


def test_live_window_toggle_flips_state(_app):
    window = LiveCameraWindow(_StubBus(), tracker=_StubTracker(_frame()))
    window.toggle()
    assert window._timer.isActive() is True
    window.toggle()
    assert window._timer.isActive() is False


def test_live_window_survives_a_throwing_tracker(_app):
    class _Broken:
        def is_capturing(self):
            raise RuntimeError("camera exploded")

        def latest_frame(self):
            raise RuntimeError("camera exploded")

    window = LiveCameraWindow(_StubBus(), tracker=_Broken())
    window.start()
    window._refresh()   # a failing tracker must never propagate onto the GUI thread
    window.stop()
