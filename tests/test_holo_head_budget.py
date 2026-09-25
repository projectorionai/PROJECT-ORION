"""
The face must not spend frames on a picture that is not changing.

Why this matters more than it sounds
------------------------------------
Under qasync the Qt thread IS the asyncio event loop. A widget painting for
6 ms at a fixed 30 Hz is not costing 180 ms a second next to ORION's brain, it
is costing 180 ms a second *inside* it — time during which the audio callback
cannot be serviced, the Live socket cannot be read and a queued coroutine
cannot run. ORION's measured stutter was exactly this: two hand-painted
widgets at 441 ms/sec between them, spent against a real-time deadline.

animation_budget.py already solved that for the voxel face. The software head
shipped without it and ran flat out forever, which put the regression straight
back. These tests pin the fix.

The head is never perfectly still — it breathes and sways — so the saving here
is redrawing an almost-still picture LESS OFTEN, never freezing it. A frozen
face reads as a hang; a slow one reads as calm.

Rendered in a subprocess: a QApplication inside this suite has been observed to
break later Qt tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.gui.holo_head import HoloHeadPanel

panel = HoloHeadPanel()
panel.resize(360, 420)
panel.show()
panel.set_state("LISTENING")
panel.set_speaking(False)
panel.set_amplitude(0.0)

out = {{"start_ms": panel.timer.interval()}}
for _ in range(40):
    panel._tick()
out["idle_ms"] = panel.timer.interval()
out["idling"] = bool(panel._budget.idling)

panel.set_speaking(True)
panel.set_amplitude(0.85)
panel.set_viseme(0.9, 0.0, 0.0)
panel._tick()
out["speaking_ms"] = panel.timer.interval()
out["speaking_idling"] = bool(panel._budget.idling)

panel.set_speaking(False)
panel.set_amplitude(0.0)
panel.set_viseme(0.0, 0.0, 0.0)
for _ in range(60):
    panel._tick()
out["resettled_ms"] = panel.timer.interval()

# A blink is a meaningful channel: it must be able to wake the rate.
panel._head._blink = 1.0
panel._tick()
out["after_blink_ms"] = panel.timer.interval()

print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def rates() -> dict:
    proc = subprocess.run([sys.executable, "-c", _PROBE.format(
        root=str(ROOT).replace("\\", "/"))], cwd=ROOT,
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-1200:]}\n{proc.stderr[-2000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail(f"probe produced nothing:\n{proc.stdout[-1200:]}")


def test_the_face_starts_at_full_rate(rates):
    assert rates["start_ms"] <= 34, f"{rates['start_ms']} ms is slower than 30 Hz"


def test_an_idle_face_drops_its_frame_rate(rates):
    """THE regression: the head shipped at a fixed rate and never settled."""
    assert rates["idling"] is True, "the budget never engaged"
    assert rates["idle_ms"] > rates["start_ms"] * 1.5, (
        f"idle rate {rates['idle_ms']} ms is barely below active "
        f"{rates['start_ms']} ms — the saving is not real"
    )


def test_the_idle_rate_still_animates_rather_than_freezing(rates):
    """Breathing and sway continue. A frozen face reads as a hang."""
    assert rates["idle_ms"] <= 125, (
        f"{rates['idle_ms']} ms between frames would make breathing visibly "
        f"step rather than drift"
    )


def test_speech_restores_full_rate_on_the_very_next_frame(rates):
    """The mouth is the whole point of this face. A settled budget must never
    cost it a frame of lip-sync."""
    assert rates["speaking_idling"] is False
    assert rates["speaking_ms"] <= 34, (
        f"still at {rates['speaking_ms']} ms while speaking"
    )


def test_it_settles_again_afterwards(rates):
    assert rates["resettled_ms"] == rates["idle_ms"], (
        "the budget engaged once and then stopped working"
    )


def test_a_blink_wakes_the_rate(rates):
    """Blinks last ~130 ms. At the idle rate that is one or two frames, so a
    blink has to restore full speed or it reads as a glitch rather than a
    blink."""
    assert rates["after_blink_ms"] <= 34


def test_the_saving_is_worth_having(rates):
    """Stated as the number that matters: CPU inside the event loop."""
    paint_ms = 6.0                       # measured; see test_holo_head.py
    before = paint_ms * (1000.0 / rates["start_ms"])
    after = paint_ms * (1000.0 / rates["idle_ms"])
    assert before - after > 80.0, (
        f"only {before - after:.0f} ms/sec saved while idle"
    )


# ── the render cap ───────────────────────────────────────────────────────────

_CAP_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
app = QApplication([])
import orion_core.gui.holo_head as hh
from orion_core.gui.holo_head import HoloHeadPanel

out = {{"cap": hh.MAX_RENDER_RADIUS}}


def shot(size):
    panel = HoloHeadPanel()
    panel.resize(size, int(size * 1.2))
    panel.show()
    panel.set_speaking(True)
    panel.set_amplitude(0.6)
    for _ in range(4):
        panel._tick()
    img = QImage(size, int(size * 1.2), QImage.Format.Format_RGB32)
    panel.render(img)
    buf = panel._buffer
    first = None if buf is None else (buf.width(), buf.height())
    panel.render(img)
    buf2 = panel._buffer
    second = None if buf2 is None else (buf2.width(), buf2.height())
    natural = min(size * 0.42, size * 1.2 / (panel._head.SPAN + 0.15))
    return {{"buffer": first, "buffer_again": second,
            "reused": buf is buf2, "natural_radius": natural}}


out["small"] = shot(300)
out["large"] = shot(1000)
print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def cap() -> dict:
    proc = subprocess.run([sys.executable, "-c", _CAP_PROBE.format(
        root=str(ROOT).replace("\\", "/"))], cwd=ROOT,
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-1200:]}\n{proc.stderr[-2000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail("probe produced nothing")


def test_a_small_face_renders_natively(cap):
    """The cap must not cost quality where there is no problem to solve."""
    assert cap["small"]["natural_radius"] <= cap["cap"]
    assert cap["small"]["buffer"] is None, "a small head was needlessly rescaled"


def test_a_large_face_renders_at_the_capped_size(cap):
    """Paint here is fill-bound, so drawing at the widget's own size makes cost
    track AREA: measured 7.2 ms at 240 px against 20.3 ms at 1000 px, which at
    30 Hz is 608 ms of every second on the thread that IS the asyncio event
    loop — worse than the stutter this codebase already fixed once."""
    assert cap["large"]["natural_radius"] > cap["cap"]
    buffer = cap["large"]["buffer"]
    assert buffer is not None, "a 1000 px head was rasterised at full size"
    # The buffer is the widget scaled by cap/radius, so its width is bounded.
    assert buffer[0] < 1000, f"buffer is {buffer[0]} px wide — the cap did nothing"


def test_the_upscale_stays_modest(cap):
    """Beyond about 2x the softness is visible on a face, and this one is the
    thing the user actually looks at."""
    factor = cap["large"]["natural_radius"] / cap["cap"]
    assert factor < 2.0, f"{factor:.1f}x upscale would read as blurry"


def test_the_buffer_is_reused_between_frames(cap):
    """Reallocating a megapixel image 30 times a second would hand back
    everything the cap saves."""
    assert cap["large"]["reused"] is True
    assert cap["large"]["buffer"] == cap["large"]["buffer_again"]
