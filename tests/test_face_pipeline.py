"""
The face, rasterised off the event loop.

Under qasync the Qt thread IS the asyncio event loop, so every millisecond
QPainter spends on ORION's head is a millisecond the audio callback does not
get — measured at ~12.5 ms a frame, about 375 ms of every second at 30 Hz.

The binding constraint is that **nothing about the face may change**. Not the
mesh, not the polygon count, not the shading, not where it sits. So the first
and most important test here compares pixels: the same head painted both ways
must come out byte-identical. If that ever stops holding, this whole approach
is wrong and the flag should go back to off.

Everything else guards the threading itself: that the worker owns the head
(two threads touching the same vertex arrays is a race that shows up as a torn
face once an hour and never reproduces), that commands are coalesced rather
than replayed, and that the thread stops when asked.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_PROBE = r'''
import json, os, random, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.constants import C
from orion_core.gui.face_pipeline import FaceRenderPipeline, FrameBuffer
from orion_core.gui.holo_head import HoloHead

W, H = 640, 560
CX, CY, R = W / 2.0, H * 0.47, 200.0
PRI, ACC, BG = QColor(C.PRI), QColor(C.ACCENT), QColor(C.BG)
out = {{}}


def paint_sync(head):
    image = QImage(W, H, QImage.Format.Format_RGB32)
    image.fill(BG)
    p = QPainter(image)
    try:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        head.paint(p, CX, CY, R, PRI, ACC, BG)
    finally:
        p.end()
    return image


def differs(a, b):
    if a.size() != b.size():
        return -1
    n = 0
    for y in range(0, a.height(), 2):
        for x in range(0, a.width(), 2):
            if a.pixel(x, y) != b.pixel(x, y):
                n += 1
    return n


# ── the one that matters: identical pixels ──────────────────────────────────
random.seed(4242)
head = HoloHead()
for _ in range(40):
    head.step(1 / 30, 0.44, speaking=True, state="speaking")
reference = paint_sync(head)

pipeline = FaceRenderPipeline(head, fps=1000.0)
pipeline.head.step = lambda *a, **k: None       # compare the PAINT, not time
pipeline.set_geometry(W, H, CX, CY, R)
pipeline.set_palette(PRI, ACC, BG)
pipeline.start()
deadline = time.monotonic() + 10
while pipeline.frames.generation == 0 and time.monotonic() < deadline:
    time.sleep(0.01)
threaded, generation = pipeline.frames.take()
pipeline.stop()
out["produced_a_frame"] = threaded is not None
out["differing_samples"] = differs(reference, threaded) if threaded else -1
out["stopped"] = not pipeline.running

# The reference must not be blank, or "identical" means nothing.
lit = sum(1 for y in range(0, H, 8) for x in range(0, W, 8)
          if reference.pixel(x, y) != BG.rgb())
out["reference_is_not_blank"] = lit

# ── it keeps rendering ──────────────────────────────────────────────────────
random.seed(7)
live = FaceRenderPipeline(HoloHead(), fps=60.0)
live.set_geometry(W, H, CX, CY, R)
live.set_palette(PRI, ACC, BG)
live.start()
time.sleep(0.5)
out["frames_in_half_a_second"] = live.frames.generation
live.stop()
out["stopped_again"] = not live.running

# ── commands reach the head, coalesced ──────────────────────────────────────
class _Spy:
    SPAN = 2.0
    def __init__(self):
        self.visemes = []
        self.glances = []
        self.steps = 0
    def step(self, dt, *a, **k):
        self.steps += 1
    def paint(self, *a, **k):
        pass
    def set_viseme(self, o, w, c=0.0):
        self.visemes.append(o)
    def glance(self, dx, dy, hold=1.1):
        self.glances.append((dx, dy))

spy = _Spy()
slow = FaceRenderPipeline(spy, fps=4.0)
slow.set_geometry(W, H, CX, CY, R)
slow.set_palette(PRI, ACC, BG)
slow.start()
for i in range(40):
    slow.call("set_viseme", i / 40.0, 0.2)
slow.call("glance", 0.0, -0.6)
time.sleep(0.7)
slow.stop()
out["viseme_calls_applied"] = len(spy.visemes)
out["last_viseme"] = spy.visemes[-1] if spy.visemes else None
out["glance_applied"] = spy.glances

# ── a broken command must not kill the renderer ─────────────────────────────
class _Hostile(_Spy):
    def set_viseme(self, *a, **k):
        raise RuntimeError("no")

hostile = _Hostile()
tough = FaceRenderPipeline(hostile, fps=50.0)
tough.set_geometry(W, H, CX, CY, R)
tough.set_palette(PRI, ACC, BG)
tough.start()
tough.call("set_viseme", 1.0, 1.0)
tough.call("nonexistent_method", 1)
time.sleep(0.4)
out["survived_bad_commands"] = tough.frames.generation
tough.stop()

# ── it will not paint before it has been told where ─────────────────────────
idle = FaceRenderPipeline(_Spy(), fps=100.0)
idle.start()
time.sleep(0.25)
out["no_geometry_no_frames"] = idle.frames.generation
idle.stop()

# ── step kwargs reach the head whole ────────────────────────────────────────
class _Kw(_Spy):
    def __init__(self):
        super().__init__()
        self.seen = []
    def step(self, dt, *a, **k):
        self.seen.append(dict(k))

kw = _Kw()
pk = FaceRenderPipeline(kw, fps=60.0)
pk.set_geometry(W, H, CX, CY, R)
pk.set_palette(PRI, ACC, BG)
pk.set_step_kwargs(amplitude=0.4, speaking=True, state="SPEAKING")
pk.start()
time.sleep(0.3)
pk.stop()
out["step_kwargs"] = kw.seen[-1] if kw.seen else {{}}

# ── the frame buffer does not reallocate at a steady size ───────────────────
buf = FrameBuffer()
a = buf.back_image(100, 80)
b = buf.back_image(100, 80)
out["buffer_reused"] = a is b
c = buf.back_image(120, 80)
out["buffer_reallocated_on_resize"] = c is not a

print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def pipeline() -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(root=str(ROOT).replace("\\", "/"))],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2500:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail(f"probe produced nothing:\n{proc.stdout[-2000:]}")


# ── the face must not change ──────────────────────────────────────────────────

def test_the_threaded_render_is_pixel_identical(pipeline):
    """The binding constraint.

    Geometry, mesh, polygon count and fidelity must not change — so the same
    head painted on a worker thread must produce the same bytes as the same
    head painted on the GUI thread. If this ever fails, the approach is wrong
    and ORION_FACE_THREAD should go back to off.
    """
    assert pipeline["produced_a_frame"] is True
    assert pipeline["differing_samples"] == 0, (
        f"{pipeline['differing_samples']} samples differ — the face changed")


def test_the_reference_was_actually_a_face(pipeline):
    """Two blank images are also identical. This makes the test above mean
    something."""
    assert pipeline["reference_is_not_blank"] > 200


# ── the threading ─────────────────────────────────────────────────────────────

def test_it_renders_continuously(pipeline):
    assert pipeline["frames_in_half_a_second"] >= 3


def test_it_stops_when_asked(pipeline):
    """Joined, not abandoned — a thread holding a QImage while Qt tears down
    its graphics stack is how "destroyed but pending" becomes a crash."""
    assert pipeline["stopped"] is True
    assert pipeline["stopped_again"] is True


def test_it_paints_nothing_until_it_knows_where(pipeline):
    assert pipeline["no_geometry_no_frames"] == 0


# ── commands ──────────────────────────────────────────────────────────────────

def test_rapid_calls_are_coalesced_to_the_last(pipeline):
    """Forty amplitude updates between two frames describe thirty-nine moments
    that have already passed. Replaying them would animate the past."""
    assert pipeline["viseme_calls_applied"] <= 3, (
        f"{pipeline['viseme_calls_applied']} of 40 were replayed")
    assert pipeline["last_viseme"] == pytest.approx(39 / 40.0)


def test_different_commands_do_not_displace_each_other(pipeline):
    """Coalescing is per method name, not a single slot."""
    assert pipeline["glance_applied"] == [[0.0, -0.6]]


def test_a_failing_command_does_not_stop_the_renderer(pipeline):
    """The face going still is worse than one ignored viseme."""
    assert pipeline["survived_bad_commands"] > 0


def test_step_arguments_arrive_together(pipeline):
    """An amplitude from this frame with a speaking flag from the last one
    would make the mouth lie."""
    assert pipeline["step_kwargs"] == {
        "amplitude": 0.4, "speaking": True, "state": "SPEAKING"}


# ── the buffer ────────────────────────────────────────────────────────────────

def test_the_buffer_is_reused_between_frames(pipeline):
    """Reallocating a megapixel image thirty times a second would hand back
    everything this saves."""
    assert pipeline["buffer_reused"] is True


def test_the_buffer_follows_a_resize(pipeline):
    assert pipeline["buffer_reallocated_on_resize"] is True


# ── the panel's wiring ────────────────────────────────────────────────────────

def _head_source() -> str:
    return (ROOT / "orion_core" / "gui" / "holo_head.py").read_text(
        encoding="utf-8")


def test_the_panel_can_be_put_back_on_the_gui_thread():
    """The first thing to try if the face ever misbehaves."""
    source = _head_source()
    assert "ORION_FACE_THREAD" in source
    assert "def face_thread_enabled" in source


def test_the_panel_queues_instead_of_touching_the_head():
    """Two threads on the same vertex arrays is a race that shows up as a
    torn face once an hour and never reproduces."""
    source = _head_source()
    start = source.index("class HoloHeadPanel")
    body = source[start:]
    assert "_to_head(" in body
    for forbidden in ("self._head.set_viseme(", "self._head.glance("):
        assert forbidden not in body, (
            f"{forbidden} bypasses the queue and races the worker")


def test_the_panel_stops_its_thread():
    source = _head_source()
    assert "def stop_pipeline" in source
    assert "closeEvent" in source


def test_a_pipeline_failure_falls_back_rather_than_blanking():
    """A face that does not appear is worse than a face that costs the loop."""
    source = _head_source()
    start = source.index("def _ensure_pipeline")
    body = source[start:start + 1400]
    assert "self._threaded = False" in body
