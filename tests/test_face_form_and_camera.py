"""ORION's form on screen, and what his camera can actually do.

Two unrelated complaints with one thing in common: the honest answer came from
measuring the hardware rather than from what an API reported.

THE FORM. The default face is a rendered human one, and a human face is not
always what you want in the room. The orb is now a first-class form, asked for
by voice or from the menu, and the choice is remembered — somebody who turned
the face off because it unsettles a housemate should not have to do it again
every restart.

THE CAMERA. It reported 1920x1080 at 60 fps and delivered 5.1. DirectShow
ignores a request to switch to MJPG and keeps sending uncompressed YUY2, which
saturates USB at 1080p. Media Foundation negotiates a compressed format by
itself and delivers 30.1 fps at the same resolution.

THE LANDMARKS. OpenCV 5 ships no Haar cascade, so face tracking had no model at
all. YuNet replaces it and returns five landmarks with every face — both eyes,
the nose tip and both mouth corners — from a 227 KB model built into cv2.

Offline: no camera is opened and no window is built.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import appearance, face_tracking, utils  # noqa: E402

WINDOW_SOURCE = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(
    encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(appearance, "APPEARANCE_PATH", tmp_path / "appearance.json")
    monkeypatch.delenv("ORION_FACE_FORM", raising=False)


# -- the form ---------------------------------------------------------------

@pytest.mark.parametrize("said,expected", [
    ("orb", "orb"),
    ("go into your orb form", "orb"),
    ("sphere", "orb"),
    ("switch to the ball", "orb"),
    ("face", "face"),
    ("show me your face", "face"),
    ("human", "face"),
    ("avatar", "face"),
])
def test_the_words_people_use_are_understood(said, expected):
    """Reached by voice as often as by menu, so "orb form" must not fail on
    the word "form"."""
    assert appearance.normalise(said) == expected


@pytest.mark.parametrize("said", ["", "   ", "banana", "loud"])
def test_an_unrecognised_form_changes_nothing(said):
    """Silently putting the face back on somebody who asked for the orb is the
    one outcome to avoid here."""
    appearance.set_face_form("orb")
    assert appearance.set_face_form(said) == ""
    assert appearance.face_form() == "orb"


def test_the_choice_outlives_the_session():
    appearance.set_face_form("orb")
    assert appearance.face_form() == "orb"
    assert appearance.APPEARANCE_PATH.is_file(), "nothing was written down"


def test_the_default_is_the_face():
    assert appearance.face_form() == "face"


def test_an_override_wins(monkeypatch):
    appearance.set_face_form("face")
    monkeypatch.setenv("ORION_FACE_FORM", "orb")
    assert appearance.face_form() == "orb"


def test_the_window_honours_the_choice_before_every_fallback():
    """The orb is a first-class form rather than a degraded mode, so it is
    chosen ahead of the automatic face fallbacks, not after them."""
    build = WINDOW_SOURCE[WINDOW_SOURCE.index("def _build_face"):]
    build = build[:build.index("\n    def ", 10)]
    assert 'face_form() == "orb"' in build
    # Consulted before any face is constructed (webengine_face_possible() is
    # only asked beforehand so the orb itself can be the Three.js one).
    assert build.index('face_form() == "orb"') < build.index("if webgl_ok:")


def test_switching_rebuilds_whichever_kind_is_showing():
    """_rebuild_face returns early unless the face is a WebEngine one, because
    it exists to recover from a native window recreation. Switching has to
    work orb -> face too, where the OLD widget is not WebEngine at all."""
    from orion_core.gui.core_window import OrionCoreWindow

    swap = inspect.getsource(OrionCoreWindow._swap_face)
    assert "_face_is_webengine" not in swap
    assert "_build_face()" in swap


def test_the_menu_route_exists():
    assert 'more_menu.addMenu("Appearance")' in WINDOW_SOURCE
    assert '"face_form", "appearance", "form"' in WINDOW_SOURCE


@pytest.mark.parametrize("phrase,want", [
    ("go into your orb form", "orb"),
    ("switch to orb", "orb"),
    ("change to your face", "face"),
    ("orb mode", "orb"),
    ("show me your orb", "orb"),
])
def test_the_voice_route_exists(phrase, want):
    from orion_core.reflex import match_reflex

    match = match_reflex(phrase)
    assert match is not None, phrase
    assert match.args == {"action": "face_form", "target": want}


# -- the camera -------------------------------------------------------------

def test_high_resolution_prefers_media_foundation():
    """DirectShow reports 60 fps at 1080p and delivers 5.1, because it stays
    on uncompressed YUY2. Media Foundation delivers 30.1."""
    source = inspect.getsource(utils.open_camera_capture)
    assert '("CAP_MSMF", "CAP_DSHOW", "CAP_ANY") if hi_res' in source


def test_ordinary_use_still_prefers_directshow():
    """MSMF's grab path throws -1072873821 on many Windows setups, which is
    why it was second. The order flips only when resolution is wanted."""
    source = inspect.getsource(utils.open_camera_capture)
    assert 'else ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY")' in source


def test_the_size_is_negotiated_before_the_validating_read():
    """Setting the resolution after a frame has been pulled makes Media
    Foundation fail to re-select the stream and hand back a malformed Mat —
    zero frames, and a camera that looks broken."""
    source = inspect.getsource(utils.open_camera_capture)
    assert source.index("CAP_PROP_FRAME_WIDTH") < source.index("cap.read()")


def test_a_backend_that_cannot_deliver_a_frame_is_rejected():
    """isOpened() is not the same as working."""
    source = inspect.getsource(utils.open_camera_capture)
    read_at = source.index("if cap.read()[0]:")
    assert "continue" in source[read_at:read_at + 500]


def test_the_tracker_asks_for_the_resolution_the_camera_is():
    assert face_tracking._CAPTURE_WIDTH == 1920
    assert face_tracking._CAPTURE_HEIGHT == 1080


def test_the_frame_rate_matches_what_the_sensor_can_give():
    """A C920 tops out at 1080p30; 60 fps exists only at 720p and below.
    Asking for more buys nothing and burns a core on repeat frames."""
    assert 24 <= face_tracking._TARGET_FPS <= 31


def test_the_tracker_does_not_renegotiate_after_opening():
    source = inspect.getsource(face_tracking.FaceTracker._loop)
    assert "size=(_CAPTURE_WIDTH, _CAPTURE_HEIGHT)" in source
    # .get() is fine — that is how the negotiated size is read back for the
    # log. It is .set() afterwards that breaks Media Foundation.
    assert "set(cv2.CAP_PROP_FRAME_WIDTH" not in source, (
        "the caller is setting the size again after the opener validated it")


# -- the landmarks ----------------------------------------------------------

def test_the_landmark_model_ships_with_orion():
    from orion_core.constants import resource_path

    model = Path(resource_path("assets", "face",
                               "face_detection_yunet_2023mar.onnx"))
    assert model.is_file(), "the face model is missing"
    assert model.stat().st_size > 100_000


def test_detection_runs_on_a_downscaled_copy():
    """98 ms on a full 1920x1080 frame against 6.7 ms at 640x360. At 30 fps
    the first is three frames of budget for a single detection."""
    assert face_tracking._DETECT_WIDTH <= 800
    source = inspect.getsource(face_tracking.FaceTracker._detect_yunet)
    assert "cv2.resize" in source


def test_yunet_is_preferred_over_the_absent_cascade():
    source = inspect.getsource(face_tracking.FaceTracker._loop)
    assert "detector = self._load_detector()" in source
    assert source.index("_load_detector") < source.index("_load_cascade")


@pytest.mark.parametrize("part", [
    "right_eye", "left_eye", "nose", "mouth_right", "mouth_left",
])
def test_every_requested_landmark_is_reported(part):
    assert part in inspect.getsource(face_tracking.FaceTracker._detect_yunet)


def test_landmarks_are_fractions_not_pixels():
    """A consumer should not have to know what resolution the camera happened
    to negotiate, and they keep working when it changes."""
    source = inspect.getsource(face_tracking.FaceTracker._detect_yunet)
    assert "/ sw" in source and "/ sh" in source


def test_a_missing_model_degrades_instead_of_killing_capture():
    """Capturing frames is what the camera lab, vision and gestures need.
    Detection is a bonus on top and is allowed to be absent."""
    assert "return None" in inspect.getsource(face_tracking.FaceTracker._load_detector)
    assert "elif cascade is not None:" in inspect.getsource(face_tracking.FaceTracker._loop)
