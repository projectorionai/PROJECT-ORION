"""
OpenCV and mediapipe are heavy imports (~155 ms and ~790 ms respectively)
that were being paid at module scope — vision.py in particular is imported
at the top of app.py, so every single app start paid for OpenCV whether or
not a camera was ever touched that session. They now import inside the
specific worker methods that need them, matching the convention already
used by utils.open_camera_capture, ocr_engine, presence and verification.

The availability flags those module-level imports used to set are now
answered by importlib.util.find_spec, which consults the module finders
without executing the module — so `FaceTracker.available` /
`GestureEngine.available` still report correctly at zero import cost.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.face_tracking import FaceTracker, _cv2_available
from orion_core.gesture_control import GestureEngine, _mediapipe_available


# ── the availability helpers answer without importing ───────────────────────

def test_cv2_available_reports_true_when_the_spec_resolves():
    assert _cv2_available() is True   # opencv is installed in this environment


def test_cv2_available_reports_false_when_the_spec_is_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert _cv2_available() is False


def test_mediapipe_available_reports_true_when_the_spec_resolves():
    assert _mediapipe_available() is True


def test_mediapipe_available_reports_false_when_the_spec_is_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert _mediapipe_available() is False


def test_availability_helpers_do_not_import_the_heavy_module(monkeypatch):
    # find_spec must be what answers the question — if these helpers ever
    # regress to a real `import`, this fails because __import__ is banned.
    def _banned(*_a, **_k):
        raise AssertionError("availability check must not import the module")

    monkeypatch.setattr("builtins.__import__", _banned)
    _cv2_available()
    _mediapipe_available()


# ── the class-level flags still work ────────────────────────────────────────

def test_face_tracker_available_flag_is_a_real_bool():
    assert isinstance(FaceTracker.available, bool)


def test_gesture_engine_available_flag_is_a_real_bool():
    assert isinstance(GestureEngine.available, bool)


# ── the imports really are deferred ─────────────────────────────────────────

def test_importing_vision_does_not_pull_in_opencv():
    # A fresh interpreter is the only honest way to assert this — by the time
    # this test file runs, other tests may already have imported cv2.
    import subprocess
    code = (
        "import sys, orion_core.vision; "
        "print('cv2' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "importing orion_core.vision must not import cv2"


def test_importing_gesture_control_does_not_pull_in_mediapipe():
    import subprocess
    code = (
        "import sys, orion_core.gesture_control; "
        "print('mediapipe' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", \
        "importing orion_core.gesture_control must not import mediapipe"
