"""
The intermittent black flicker on ORION's face.

Root cause, reproduced headlessly (it is plain Python arithmetic, no GPU):
_rebuild computed ``u = min(w, h) * 0.42`` and then ``du = cell / u``. During a
transient 0-or-tiny size — the first show, a splitter collapse, the portrait-
toggle resize — ``u`` was 0 and the division raised. paintEvent had already
filled the whole widget with C.BG ("#050608", deep void black"); Qt swallows a
paintEvent exception, so that bare void-black fill was left on screen for one
frame. That single frame, every now and then, is the flicker.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.gui.face import C, HologramFace  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_a_degenerate_size_never_crashes_the_rebuild(qapp):
    face = HologramFace()
    # These are the exact sizes paintEvent used to hand _rebuild mid-resize.
    face._rebuild(0, 200)      # must not raise (was ZeroDivisionError)
    face._rebuild(200, 0)      # must not raise
    face._rebuild(1, 1)        # must not raise
    # ...and a healthy size still bakes a real face.
    face._rebuild(300, 300)
    assert face._cells


def test_paint_skips_a_tiny_frame_instead_of_flashing_the_void(qapp, monkeypatch):
    """At a sub-face size paintEvent must return BEFORE it fills the widget with
    the deep-void-black C.BG and before it touches _rebuild — otherwise a
    degenerate resize frame flashes black. Calling paintEvent directly is only
    safe precisely because the guard returns before constructing a QPainter."""
    face = HologramFace()
    monkeypatch.setattr(face, "width", lambda: 2)
    monkeypatch.setattr(face, "height", lambda: 2)
    reached: list = []
    monkeypatch.setattr(face, "_rebuild", lambda *a: reached.append(a))

    assert face.paintEvent(None) is None
    assert reached == [], "paint proceeded past the guard at a degenerate size"
    # The background it would have flashed really is near-black — which is why
    # the bug read as a black flicker rather than a coloured one. Asserted as
    # a property: the exact value is a palette decision and has already moved
    # once (#050608 -> #060606, dropping the blue cast), while "dark enough to
    # read as a black flash" is the part this test depends on.
    void = C.BG.lstrip("#")
    assert max(int(void[i:i + 2], 16) for i in (0, 2, 4)) < 24, (
        f"C.BG is {C.BG}, too light to flash as void black")


def test_a_healthy_size_actually_paints_the_face(qapp):
    """The guard must not be so eager that it skips normal frames: a full-size
    render draws over the background, so the image is no longer uniform."""
    face = HologramFace()
    face.resize(240, 240)
    img = QImage(240, 240, QImage.Format.Format_RGB32)
    img.fill(QColor("#000000"))
    face.render(img)
    colours = {img.pixel(x, y) for x in range(0, 240, 24)
               for y in range(0, 240, 24)}
    assert len(colours) > 1, "a healthy frame should not be a flat fill"
