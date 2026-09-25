"""
Viseme lip-sync on the HologramFace (Mark XXVI, Phase 3).

The mouth-shape channels are deterministic, so they are the proof: set_viseme
posts a posture, the channels ease toward it while it is refreshed, and its
influence (gain) decays to nothing when it stops — leaving the old amplitude-driven
mouth exactly as it was.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.gui.face import HologramFace  # noqa: E402
from orion_core.viseme import VISEMES  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_set_viseme_posts_the_target_posture(qapp):
    face = HologramFace()
    face.set_viseme("ou", 0.9)
    assert face._vis_open_t == VISEMES["ou"].open
    assert face._vis_width_t == VISEMES["ou"].width
    assert face._vis_round_t == VISEMES["ou"].round
    assert face._vis_gain_t == pytest.approx(0.9)


def test_an_unknown_viseme_rests_the_mouth(qapp):
    face = HologramFace()
    face.set_viseme("not_a_viseme", 1.0)
    assert face._vis_open_t == VISEMES["sil"].open


def test_repeated_visemes_raise_the_gain_and_reach_the_posture(qapp):
    face = HologramFace()
    face.isVisible = lambda: True
    for _ in range(12):
        face.set_viseme("aa", 1.0)
        face._tick()
    assert face._vis_gain > 0.8
    assert face._vis_open == pytest.approx(VISEMES["aa"].open, abs=0.1)


def test_the_gain_decays_when_no_viseme_arrives(qapp):
    face = HologramFace()
    face.isVisible = lambda: True
    face.set_viseme("aa", 1.0)
    for _ in range(20):
        face._tick()                      # no re-set → the posture releases
    assert face._vis_gain < 0.2


def test_the_mouth_still_paints_under_a_viseme(qapp):
    from PyQt6.QtGui import QColor, QImage

    face = HologramFace()
    face.resize(300, 300)
    face.isVisible = lambda: True
    face.set_state("SPEAKING")
    for _ in range(12):
        face.set_viseme("aa", 1.0)
        face._tick()
    img = QImage(300, 300, QImage.Format.Format_RGB32)
    img.fill(QColor("#050608"))
    face.render(img)                       # must not raise
    colours = {img.pixel(x, y) for x in range(0, 300, 20) for y in range(0, 300, 20)}
    assert len(colours) > 1


# ── the bus wiring (source-checked: importing core_window crashes offscreen Qt) ─

def test_the_bus_carries_a_viseme_signal():
    import inspect

    from orion_core.bus import OrionBus
    assert "viseme" in inspect.getsource(OrionBus)


def test_core_window_forwards_visemes_to_the_face():
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "gui"
           / "core_window.py").read_text(encoding="utf-8")
    assert "self.bus.viseme.connect(self._face_set_viseme)" in src
    assert 'self._to_face("set_viseme"' in src
