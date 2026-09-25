"""
Every frame ORION sends the model must say where it came from.

The failure this prevents
-------------------------
An image arrives at a multimodal model as pixels and nothing else. With no
sentence saying what it IS, the model infers from content — and ORION's own
window has a large rendered human face in the middle of it. A screenshot taken
while his avatar is on screen therefore looks exactly like a photograph of the
user, and questions about "what do you see" get answered with a confident
description of his own face presented as a description of the person sitting
there.

That is not a hypothetical class of error; it is what happens by default when
a screen capture and a webcam photo are both handed over as bare image/jpeg.

The fix is a sentence rather than a mechanism, which is why these tests check
the wording reaches the tool result the model reads alongside the frame.

Offline: no capture, no model, no Qt.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import vision
from orion_core.vision import (
    FILE_FRAME_NOTE,
    SCREEN_FRAME_NOTE,
    WEBCAM_FRAME_NOTE,
)


# ── the notes themselves ─────────────────────────────────────────────────────

def test_the_screenshot_note_names_the_source():
    assert "SCREENSHOT" in SCREEN_FRAME_NOTE


def test_the_screenshot_note_disowns_orions_own_avatar():
    """THE point. Without this the model describes its own face as the user."""
    lowered = SCREEN_FRAME_NOTE.lower()
    assert "avatar" in lowered
    assert "not a photograph of the user" in lowered
    assert "person in the room" in lowered


def test_the_webcam_note_says_it_is_a_photograph():
    assert "PHOTOGRAPH" in WEBCAM_FRAME_NOTE
    assert "webcam" in WEBCAM_FRAME_NOTE.lower()


def test_a_file_is_not_presented_as_something_just_captured():
    """A picture the user pointed at is not evidence of what is happening now."""
    assert "FILE" in FILE_FRAME_NOTE
    assert "not something orion captured just now" in FILE_FRAME_NOTE.lower()


def test_the_three_notes_are_distinguishable():
    notes = {SCREEN_FRAME_NOTE, WEBCAM_FRAME_NOTE, FILE_FRAME_NOTE}
    assert len(notes) == 3
    for note in notes:
        assert note.startswith("SOURCE:"), note[:40]


# ── every site that sends an image uses one ──────────────────────────────────

def _image_sites(source: str) -> list[str]:
    """Lines that attach a JPEG to a ToolResult."""
    return [line for line in source.split("\n")
            if 'media={"data"' in line and "image/jpeg" in line]


def test_every_image_site_in_vision_is_accounted_for():
    """A new capture path that forgets its label reopens the whole problem, so
    the count of image sites is pinned against the count of labels."""
    source = inspect.getsource(vision)
    sites = _image_sites(source)
    assert sites, "no image-returning sites found — has vision.py moved?"
    labels = (source.count("SCREEN_FRAME_NOTE") - 1
              + source.count("WEBCAM_FRAME_NOTE") - 1
              + source.count("FILE_FRAME_NOTE") - 1)
    assert labels >= len(sites), (
        f"{len(sites)} image sites in vision.py but only {labels} carry a "
        f"source label"
    )


@pytest.mark.parametrize("function,expected", [
    ("_analyse_screen_sync", "SCREEN_FRAME_NOTE"),
    ("_detect_errors_sync", "SCREEN_FRAME_NOTE"),
    ("_encode_frame", "WEBCAM_FRAME_NOTE"),
    ("_inspect_image", "FILE_FRAME_NOTE"),
])
def test_each_capture_path_labels_itself_correctly(function, expected):
    """A screen grab labelled as a webcam photo would be worse than no label."""
    source = inspect.getsource(vision)
    start = source.index(f"def {function}(")
    # To the start of the next def at the same indentation.
    rest = source[start:]
    end = rest.find("\n    def ", 1)
    body = rest if end < 0 else rest[:end]
    assert expected in body, f"{function} sends a frame with no {expected}"


def test_the_volatile_screen_grab_is_labelled_too():
    """dispatch_vision has its own capture path, separate from vision.py."""
    from orion_core import dispatch_vision

    source = inspect.getsource(dispatch_vision)
    assert "SCREEN_FRAME_NOTE" in source, (
        "the volatile monitor grab sends an unlabelled frame"
    )


def test_labels_are_short_enough_to_be_read():
    """This rides on every single frame; a paragraph would crowd out the tool
    result it is attached to."""
    for note in (SCREEN_FRAME_NOTE, WEBCAM_FRAME_NOTE, FILE_FRAME_NOTE):
        assert len(note) < 320, f"{len(note)} characters is too long to repeat"
