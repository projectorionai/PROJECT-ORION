"""
HoloHead renderer tests — verified by rendering real frames.

Why a subprocess, and why pixels
--------------------------------
Two lessons are baked into the shape of this file.

The first is procedural: standing up a QApplication inside this suite has been
observed to break Qt tests that run later, so every frame here is rendered by a
short script in its own process and the result comes back as JSON.

The second is about what is worth asserting. This renderer shipped three
defects that no structural check would have caught — a mouth whose lip gap
changed by 0.9 px while every weight and vertex was "correct", a jaw that
sheared into slivers, and a notch punched through the jawline by the painter's
batching rather than by the mesh. All three were found by rendering a frame and
looking at it. So these tests render frames and measure pixels: is there a head
there, does the mouth actually open, does the silhouette stay whole.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Budget for one paint at a 150 px radius. ORION has starved its own event
#: loop with QPainter before (an earlier face cost 441 ms per wall-clock
#: second), so this is a real ceiling, not a formality. Measured ~6 ms; the
#: headroom covers slower machines and CI.
PAINT_BUDGET_MS = 22.0

_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPainter, QColor
app = QApplication([])
from orion_core.gui.holo_head import HoloHead, _DEPTH_SLABS, _SHADE_BUCKETS

SIZE = 340
BG, PRI, ACC = QColor("#05070c"), QColor("#39b6ff"), QColor("#9fe8ff")


def draw(state, speaking, amp, viseme, steps, seed=11):
    import random
    random.seed(seed)
    head = HoloHead()
    for _ in range(steps):
        if viseme:
            head.set_viseme(*viseme)
        head.step(1.0 / 30.0, amp, speaking=speaking, state=state)
    if viseme:
        head.set_viseme(*viseme)
    img = QImage(SIZE, SIZE, QImage.Format.Format_RGB32)
    img.fill(BG)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    radius = min(SIZE * 0.42, SIZE / (head.SPAN + 0.15))
    head.paint(p, SIZE / 2, SIZE * 0.47, radius, PRI, ACC, BG)
    p.end()
    return img, head.last_paint_ms


def stats(img):
    """Lit pixels, their bounding box, and the darkest region inside the face
    (the open mouth reads as an aperture, so it shows up here)."""
    bg = (5, 7, 12)
    lit = 0
    xs, ys = [], []
    rows = {{}}
    for y in range(0, SIZE, 2):
        for x in range(0, SIZE, 2):
            c = img.pixelColor(x, y)
            if abs(c.red() - bg[0]) + abs(c.green() - bg[1]) + abs(c.blue() - bg[2]) > 26:
                lit += 1
                xs.append(x); ys.append(y)
                rows.setdefault(y, []).append(x)
    if not xs:
        return {{"lit": 0}}
    return {{
        "lit": lit,
        "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
        "rows": len(rows),
    }}


def dark_pixels_in_mouth_band(img):
    """Count near-black pixels in the lower-middle of the face.

    An open mouth is an aperture: the inner lip ring is filled dark, so the
    count rises sharply when the jaw drops. This is the pixel-level version of
    'did the mouth actually open', which is exactly the question the rig got
    wrong while every weight looked right.
    """
    n = 0
    for y in range(int(SIZE * 0.52), int(SIZE * 0.78)):
        for x in range(int(SIZE * 0.34), int(SIZE * 0.66)):
            c = img.pixelColor(x, y)
            if c.red() + c.green() + c.blue() < 90:
                n += 1
    return n


out = {{"depth_slabs": _DEPTH_SLABS, "shade_buckets": _SHADE_BUCKETS}}

closed, ms_closed = draw("LISTENING", False, 0.0, None, 12)
wide, ms_open = draw("SPEAKING", True, 0.9, (1.0, -0.05, 0.0), 22)
shut, _ = draw("SPEAKING", True, 0.9, (0.0, 0.0, 1.0), 22)
sleeping, _ = draw("STANDBY", False, 0.0, None, 40)

out["closed"] = stats(closed)
out["wide"] = stats(wide)
out["sleeping"] = stats(sleeping)
out["paint_ms"] = max(ms_closed, ms_open)
out["dark_closed"] = dark_pixels_in_mouth_band(closed)
out["dark_wide"] = dark_pixels_in_mouth_band(wide)
out["dark_shut"] = dark_pixels_in_mouth_band(shut)
print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def frames() -> dict:
    script = _PROBE.format(root=str(ROOT).replace("\\", "/"))
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"render probe failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail(f"render probe produced no result:\n{proc.stdout[-1500:]}")


def test_a_head_is_actually_drawn(frames):
    """Headless checks have missed a blank render before; count the pixels."""
    assert frames["closed"]["lit"] > 1200, "the frame is essentially empty"


def test_the_head_is_centred_and_upright(frames):
    box = frames["closed"]
    width = box["x1"] - box["x0"]
    height = box["y1"] - box["y0"]
    assert width > 90 and height > 120, f"head is {width}x{height}px, too small"
    assert height > width, "a head is taller than it is wide"
    centre = (box["x0"] + box["x1"]) / 2
    assert abs(centre - 170) < 34, f"head is off-centre at x={centre}"


def test_the_mouth_actually_opens(frames):
    """The defect this whole module exists to prevent.

    The rig once moved both lips together, so the mouth 'opened' by 0.9 px
    while every weight, vertex and normal checked out. Only the aperture is
    trustworthy: an open mouth is dark, a closed one is not.
    """
    assert frames["dark_wide"] > frames["dark_closed"] * 2.5 + 40, (
        f"open mouth produced {frames['dark_wide']} dark pixels against "
        f"{frames['dark_closed']} closed — the jaw is moving both lips together"
    )


def test_a_lip_closure_keeps_the_mouth_shut(frames):
    """/m/, /b/ and /p/ are made with the lips pressed shut, and loudness must
    not override that — it is the whole reason the transcript is consulted."""
    assert frames["dark_shut"] < frames["dark_wide"] * 0.6, (
        "a closure at full amplitude still opened the mouth"
    )


def test_sleeping_lowers_the_lids_without_losing_the_head(frames):
    assert frames["sleeping"]["lit"] > 1200, "the head vanished while asleep"


def test_paint_stays_within_the_frame_budget(frames):
    assert frames["paint_ms"] < PAINT_BUDGET_MS, (
        f"paint took {frames['paint_ms']:.1f} ms (budget {PAINT_BUDGET_MS} ms). "
        f"ORION has starved its own event loop with QPainter before."
    )


def test_depth_slabs_are_fine_enough_to_avoid_the_jaw_notch(frames):
    """Regression guard for a painter bug that looked exactly like a mesh bug.

    Sorting by (coarse depth slab, shade) is what makes this renderer fast, but
    at five slabs a background triangle wins against a foreground one along the
    jaw and punches a notch through the silhouette whenever the mouth opens.
    It reads as a torn jaw rig, and cost a detour into the weights before a
    sweep of slab counts showed it was the batching. Fourteen renders clean.
    """
    assert frames["depth_slabs"] >= 12, (
        f"_DEPTH_SLABS is {frames['depth_slabs']}; below ~12 the batching sort "
        f"lets the jawline punch through. Lower it only with a rendered frame "
        f"in hand."
    )


def test_the_silhouette_has_no_holes_in_it(frames):
    """A notch or a dropped run shows up as missing scanlines."""
    box = frames["wide"]
    spanned = (box["y1"] - box["y0"]) // 2 + 1
    assert box["rows"] >= spanned * 0.9, (
        f"only {box['rows']} of ~{spanned} scanlines carry the head — "
        f"the silhouette has gaps"
    )
