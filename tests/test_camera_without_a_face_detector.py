"""A missing data file must not read as "no camera".

OpenCV 5 ships ``cv2/data/`` with the Haar cascade XMLs REMOVED. Two OpenCV
behaviours then combine badly:

  * ``CascadeClassifier(path)`` does NOT raise when the path is absent — it
    returns an *empty* classifier;
  * the error surfaces later, from ``detectMultiScale``, on the first frame.

That exception propagated out of FaceTracker's capture loop, whose ``finally``
sets ``running = False`` and releases the device. So with a healthy webcam
plugged in, ORION reported "face tracking is off", the camera lab showed
"Camera paused", and nothing pointed at a missing XML file.

Capturing frames is what the camera lab, on-demand vision and gesture control
all actually need. Detection is a bonus on top, and is now allowed to be
absent.

Offline: no camera is opened.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.face_tracking import FaceTracker  # noqa: E402


def _tracker() -> FaceTracker:
    return FaceTracker(lambda _sample: None)


def test_a_missing_cascade_yields_none_rather_than_an_empty_classifier():
    """An empty classifier is a landmine: it looks fine and detonates on the
    first frame, inside the loop that owns the camera."""
    tracker = _tracker()
    cascade = tracker._load_cascade()
    if cascade is not None:
        assert not cascade.empty(), "an empty classifier was handed back"
        return
    assert tracker.detection_error, "it gave up without saying why"


def test_capture_survives_a_detector_that_cannot_run():
    """The loop must treat detection as optional. Asserted on the structure,
    because reproducing it needs a physical camera."""
    source = inspect.getsource(FaceTracker._loop)
    assert "if cascade is not None:" in source, (
        "detection is being run unconditionally again")
    detect_at = source.index("self._detect(frame, cascade)")
    try_at = source.rindex("try:", 0, detect_at)
    assert "except Exception" in source[detect_at:detect_at + 400], (
        "a detection error can still escape and kill the capture loop")
    assert try_at < detect_at


def test_frames_are_stored_before_detection_is_attempted():
    """Whoever borrows the camera gets their picture even if detection then
    fails — the frame is the part other subsystems need."""
    source = inspect.getsource(FaceTracker._loop)
    assert source.index("self._latest_frame = frame") < source.index(
        "self._detect(frame, cascade)")


def test_a_missing_detector_is_not_reported_as_a_capture_fault():
    """`error` means the CAPTURE failed. Conflating the two is what sent
    someone hunting for a camera fault that did not exist."""
    tracker = _tracker()
    tracker._load_cascade()
    assert tracker.error == "", "a missing model was recorded as a camera fault"


def test_the_status_says_which_half_is_broken():
    tracker = _tracker()
    tracker.running = True
    tracker.frames = 12
    tracker.detection_error = "no face-detection model installed"
    text = tracker.status().text.lower()
    assert "camera is live" in text
    assert "face tracking is off" not in text


def test_it_only_complains_once():
    """A per-frame log line at 15fps is not a diagnostic, it is a flood."""
    tracker = _tracker()
    seen: list[str] = []
    tracker.bus = type("_B", (), {"log": type("_L", (), {"emit": lambda _s, m: seen.append(m)})()})()
    for _ in range(5):
        tracker._log("same message")
    assert len(seen) == 1
