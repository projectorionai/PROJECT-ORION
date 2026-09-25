"""
Face render harness (Mark XXVI, §28) — a visual regression net for the face.

Every expression must render a real, non-black frame (the black-flicker class was
a whole prior bug), the expanded §8 vocabulary must be present and complete, each
profile must map onto the parameters the face actually consumes, and brightness
must track glow (a cheap, RNG-robust proxy that the expression genuinely changed
the render). Set ORION_FACE_HARNESS_DIR to also dump PNGs for eyeball inspection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.emotion import EMOTIONS  # noqa: E402
from orion_core.gui.face import HologramFace  # noqa: E402

#: The §8 expression vocabulary this pass set out to deliver.
SPEC_EXPRESSIONS = {
    "neutral", "thinking", "concentrating", "listening", "speaking", "happy",
    "amused", "excited", "proud", "curious", "reassuring", "empathetic",
    "concerned", "confused", "uncertain", "disappointed", "sad", "frustrated",
    "alert",
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _render(name: str, size: int = 240) -> QImage:
    face = HologramFace()
    face.resize(size, size)
    face.isVisible = lambda: True
    face._blink_in = 10_000          # keep eyes open — RNG-stable frame
    face.apply_emotion(name, EMOTIONS[name].params())
    for _ in range(30):
        face._tick()
    img = QImage(size, size, QImage.Format.Format_RGB32)
    img.fill(QColor("#050608"))
    face.render(img)
    out_dir = os.getenv("ORION_FACE_HARNESS_DIR")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        img.save(str(Path(out_dir) / f"face_{name}.png"))
    return img


def _mean_luminance(img: QImage) -> float:
    total, n = 0.0, 0
    for x in range(0, img.width(), 6):
        for y in range(0, img.height(), 6):
            c = QColor(img.pixel(x, y))
            total += 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
            n += 1
    return total / max(1, n)


def _distinct_colours(img: QImage) -> int:
    return len({img.pixel(x, y) for x in range(0, img.width(), 6)
                for y in range(0, img.height(), 6)})


# ── vocabulary ────────────────────────────────────────────────────────────────

def test_the_full_spec_vocabulary_is_present():
    missing = SPEC_EXPRESSIONS - set(EMOTIONS)
    assert not missing, f"expressions missing from the library: {missing}"


def test_the_nine_new_expressions_were_added():
    for new in ("concentrating", "curious", "amused", "confused", "uncertain",
                "disappointed", "proud", "reassuring", "empathetic"):
        assert new in EMOTIONS, new


def test_every_profile_maps_to_the_face_params():
    needed = {"brow", "glow", "eye_width", "eye_height", "mouth_curve",
              "mouth_tension", "particle_direction", "palette_bright", "accent"}
    for name, profile in EMOTIONS.items():
        params = profile.params()
        assert needed <= set(params), f"{name} missing {needed - set(params)}"


# ── rendering (the visual net) ────────────────────────────────────────────────

def test_every_expression_renders_a_non_black_frame(qapp):
    for name in EMOTIONS:
        img = _render(name)
        lum = _mean_luminance(img)
        colours = _distinct_colours(img)
        assert lum > 12.0, f"{name} rendered near-black (lum={lum:.1f})"
        assert colours > 8, f"{name} rendered near-flat ({colours} colours)"


def test_brightness_tracks_glow(qapp):
    # A cheap, RNG-robust proxy that the expression really changed the render:
    # a dim profile (sad, glow 0.30) must render darker than a bright one
    # (proud, glow 0.90), with neutral (0.55) in between.
    sad = _mean_luminance(_render("sad"))
    neutral = _mean_luminance(_render("neutral"))
    proud = _mean_luminance(_render("proud"))
    assert sad < neutral < proud, f"sad={sad:.1f} neutral={neutral:.1f} proud={proud:.1f}"


def test_the_emotion_tool_advertises_the_expanded_set():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "emotion")
    desc = tool["description"]
    for name in ("curious", "proud", "empathetic", "concentrating"):
        assert name in desc, f"{name} not advertised in the emotion tool schema"


# ── render budget (the face paints on the qasync/audio thread) ───────────────

def test_the_face_stays_inside_its_frame_budget(qapp):
    """Before this pass the voxel loop issued ~2,500 setBrush/drawRect calls a
    frame (7.9 ms at 600px, a quarter of a 30fps budget) on the same thread as
    the audio callback. Voxels are now grouped by quantised colour into one
    drawRects call per group. The ceiling is loose so a slow box does not fail
    the suite, but well under a frame."""
    import time
    face = HologramFace()
    face.resize(600, 600)
    face.isVisible = lambda: True
    face._blink_in = 10_000
    face.set_state("SPEAKING")
    for _ in range(20):
        face._tick()
    img = QImage(600, 600, QImage.Format.Format_RGB32)
    face.render(img)                                   # warm

    start = time.perf_counter()
    for _ in range(15):
        img.fill(QColor("#050608"))
        face.render(img)
    per_frame = (time.perf_counter() - start) * 1000 / 15
    assert per_frame < 25.0, f"the face costs {per_frame:.1f} ms a frame"
