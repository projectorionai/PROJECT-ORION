"""Seeing the camera and being told about it are different requests.

Asking ORION to bring up the camera ran `vision_analyse action='camera'`,
which grabs one frame, describes it in words and throws the frame away. That
answers "what can you see" — a question the user had not asked. There was no
path at all to the thing they wanted: the Camera Lab, open, with the live feed
running.

Both now exist and are distinct, and the clear imperatives are matched locally
so they never depend on the model routing them correctly.

Offline: regex and source only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.reflex import match_reflex  # noqa: E402


@pytest.mark.parametrize("spoken", [
    "show me the camera",
    "bring up the camera",
    "open the camera",
    "open the camera lab",
    "camera lab",
    "let me see the camera",
    "show me the live camera feed",
    "pull up the webcam",
])
def test_asking_to_see_the_camera_opens_the_live_feed(spoken):
    match = match_reflex(spoken)
    assert match is not None, f"{spoken!r} did not match"
    assert match.tool == "interface_control"
    assert match.args == {"action": "camera"}


@pytest.mark.parametrize("spoken", [
    "what can you see",
    "what is in front of me",
    "what's on the camera",
    "describe what you can see",
    "can you see me",
])
def test_asking_what_he_can_see_is_left_to_the_model(spoken):
    """These are questions for vision_analyse. A regex answering them would be
    the same mistake in the other direction."""
    assert match_reflex(spoken) is None


@pytest.mark.parametrize("spoken,target", [
    ("open the memory page", "memory"),
    ("go to the diagnostics page", "diagnostics"),
    ("take me to the brain page", "brain"),
    ("switch to the globe tab", "globe"),
])
def test_explicit_page_requests_navigate(spoken, target):
    match = match_reflex(spoken)
    assert match is not None, f"{spoken!r} did not match"
    assert match.tool == "interface_control"
    assert match.args == {"action": "page", "target": target}


@pytest.mark.parametrize("spoken", [
    "show me the memory",          # a memory, not the page
    "open the door",
    "how are you",
    "go to bed",
])
def test_navigation_does_not_fire_on_ordinary_sentences(spoken):
    """The trailing page/deck/tab is what keeps this anchored. A reflex that
    fires on a conversational sentence is worse than one that misses, because
    falling through to the model is always correct and only slower."""
    match = match_reflex(spoken)
    assert match is None or match.args.get("action") != "page"


def test_the_live_feed_actually_starts_the_camera():
    """Opening the page without starting the feed is the toolbar button's job;
    this path is the one that shows something."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    start = source.index("def open_camera_lab_live")
    body = source[start:source.index("\n    def ", start + 10)]
    assert "start_camera()" in body
    assert "WORKBENCH" in body


def test_it_reports_whether_the_camera_actually_started():
    """`start_camera()` returns None whether it opened a device or gave up —
    it writes "CAMERA UNAVAILABLE" on its own canvas and returns. Returning
    True regardless would be claiming a success with no evidence for it, which
    is the defect this whole pass kept finding."""
    from orion_core.gui.core_window import OrionCoreWindow

    class _Log:
        def __init__(self): self.lines = []
        def emit(self, line): self.lines.append(line)

    class _Bus:
        def __init__(self): self.log = _Log()

    def _window(workbench):
        class _Deck:
            _pages = [("WORKBENCH", workbench)]

        class _Window:
            bus = _Bus()
            dashboard = _Deck()
            command_centre = None

            def _toggle_deck_page(self, page, label):
                pass

            _deck_page_widget = OrionCoreWindow._deck_page_widget

        return _Window()

    class _Works:
        _camera_enabled = False

        def start_camera(self):
            self._camera_enabled = True

    class _GivesUp:
        """No tracker attached: sets nothing, raises nothing, returns None."""

        _camera_enabled = False

        def start_camera(self):
            pass

    class _Raises:
        def start_camera(self):
            raise RuntimeError("device busy")

    assert OrionCoreWindow.open_camera_lab_live(_window(_Works())) is True

    quiet = _window(_GivesUp())
    assert OrionCoreWindow.open_camera_lab_live(quiet) is False
    assert quiet.bus.log.lines, "a camera that never started said nothing"

    noisy = _window(_Raises())
    assert OrionCoreWindow.open_camera_lab_live(noisy) is False
    assert noisy.bus.log.lines


def test_the_bus_action_is_handled():
    """A tool action nothing listens for is a command that vanishes."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    assert '"camera", "camera_lab", "webcam", "live_camera"' in source
    assert "open_camera_lab_live()" in source


def test_the_model_is_told_the_two_are_different():
    """Left to guess, it picks the vision tool — which is what it did."""
    schema = (ROOT / "orion_core" / "dispatch_schema.py").read_text(encoding="utf-8")
    assert "START THE LIVE FEED" in schema
    assert "not the same as vision_analyse" in schema.lower()
