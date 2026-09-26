"""The face stuttered because of the thing that was meant to make it cheap.

The report was "he's still stuttering a lot, it's literally just his face UI".
Everything that usually explains a stutter had already been ruled out by
measurement on the running app: the Qt thread showed ZERO pauses over 50 ms in
60 s probed at 100 Hz (max 26.9 ms), and the GPU's 3d engine averaged ~11%.
Nothing was stalling.

The frame-rate governor was. It drops the face from 60 fps to 30 when nothing
is moving, deciding per frame from three signals -- how far the morph is from
its target, the audio amplitude, and the speech envelope. Every one of those is
an exponential lerp, so it approaches its threshold asymptotically and then
SITS on it; st.amp in particular is lerped toward live microphone amplitude, so
an ordinary quiet room parks it either side of 0.01. With a single threshold
and no dwell, the target rate flipped 60/30/60/30 from one frame to the next.

That is worse than either rate on its own. A steady 30 is an exact divisor of a
60 Hz display and looks like film; a rate that alternates makes each frame's
delta jump between 16.7 ms and 33.3 ms, and uneven deltas are exactly what the
eye reads as stutter. Simulated against a quiet-room amplitude trace, the old
governor changed rate 130 times in 600 frames -- 13 times a second.

The fix is the shape the Python-side AnimationBudget already had and the JS
lacked: separate thresholds for entering and leaving, and a dwell of calm
frames before dropping. Leaving idle stays instant, and any one signal can do
it, because being slow to wake is a worse fault than being slow to sleep.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.gui import face3d  # noqa: E402

HTML = face3d.FACE_HTML


def _const(name: str) -> float:
    """A numeric constant out of the page's governor."""
    match = re.search(rf"\b{name}\s*=\s*([0-9.]+)", HTML)
    assert match, f"{name} is not in the face page any more"
    return float(match.group(1))


# -- the property that fixes it ----------------------------------------------

@pytest.mark.parametrize("entering,leaving", [
    ("QUIET_IN", "QUIET_OUT"),
    ("SETTLE_IN", "SETTLE_OUT"),
])
def test_the_thresholds_are_hysteretic(entering, leaving):
    """One threshold per signal is what made it dither. Leaving must need
    more movement than entering needed stillness."""
    assert _const(leaving) > _const(entering), (
        f"{leaving} must be above {entering} or the governor dithers")


def test_idle_needs_a_run_of_calm_frames():
    """A dwell, so a single quiet frame cannot drop the rate. The Python
    AnimationBudget has always had this as SETTLE_FRAMES; the page did not."""
    assert _const("IDLE_AFTER") >= 30


def test_waking_is_instant_and_any_one_signal_can_do_it():
    """Being slow to wake is a worse fault than being slow to sleep, so the
    leaving test is an OR with no dwell in front of it."""
    governor = HTML[HTML.index("function targetFrameMs()"):]
    governor = governor[:governor.index("}\n\nfunction animate")]
    leaving = governor[governor.index("if(idleRate)"):governor.index("else")]
    assert "||" in leaving, "leaving idle requires every signal to agree"
    assert "IDLE_AFTER" not in leaving, "waking up is behind a dwell"


# -- what it does against a real signal --------------------------------------

def _quiet_room(frames=600, seed=7):
    """st.amp lerped toward jittery ambient microphone amplitude, which is
    what the page actually does: st.amp=lerp(st.amp,tg.amp,0.35)."""
    random.seed(seed)
    amp, trace = 0.0, []
    for _ in range(frames):
        target = max(0.0, 0.011 + random.gauss(0, 0.004))
        amp += (target - amp) * 0.35
        trace.append(amp)
    return trace


def _governor(trace, quiet_in, quiet_out, dwell):
    idle, calm, rates = False, 0, []
    for amp in trace:
        if idle:
            if amp > quiet_out:
                idle, calm = False, 0
        else:
            calm = calm + 1 if amp < quiet_in else 0
            if calm >= dwell:
                idle = True
        rates.append(30 if idle else 60)
    return rates


def _changes(rates):
    return sum(1 for a, b in zip(rates, rates[1:]) if a != b)


def test_a_single_threshold_dithers_which_is_the_bug():
    """Kept as the control. Without it the next test proves nothing."""
    trace = _quiet_room()
    single = [30 if amp < 0.010 else 60 for amp in trace]
    assert _changes(single) > 50, (
        "the trace no longer reproduces the original fault")


def test_the_shipped_constants_do_not_dither():
    trace = _quiet_room()
    rates = _governor(trace, _const("QUIET_IN"), _const("QUIET_OUT"),
                      _const("IDLE_AFTER"))
    assert _changes(rates) == 0, (
        f"the face still changes rate {_changes(rates)} times in "
        f"{len(trace)} frames")


def test_real_speech_still_gets_the_full_rate():
    """The saving must not cost responsiveness: anything moving is 60 fps."""
    trace = [0.0] * 200 + [0.35] * 100 + [0.0] * 200
    rates = _governor(trace, _const("QUIET_IN"), _const("QUIET_OUT"),
                      _const("IDLE_AFTER"))
    speaking = rates[200:300]
    assert set(speaking) == {60}, "the face dropped frames while speaking"
    # And it woke on the very first loud frame, not after a dwell.
    assert rates[200] == 60


# -- the override ------------------------------------------------------------

def test_the_idle_rate_is_thirty_by_default():
    assert face3d.idle_fps() == 30


def test_somebody_who_can_still_see_it_can_buy_it_back(monkeypatch):
    """"Even 30" and "smooth" are not the same judgement for everybody."""
    monkeypatch.setenv("ORION_FACE_IDLE_FPS", "60")
    assert face3d.idle_fps() == 60


@pytest.mark.parametrize("value,expected", [
    ("5", 10), ("240", 60), ("", 30), ("banana", 30), ("  45 ", 45),
])
def test_the_override_is_clamped_and_never_raises(monkeypatch, value, expected):
    monkeypatch.setenv("ORION_FACE_IDLE_FPS", value)
    assert face3d.idle_fps() == expected


def test_the_rate_reaches_the_page(monkeypatch):
    monkeypatch.setenv("ORION_FACE_IDLE_FPS", "48")
    page = face3d.three_sourced(face3d.FACE_HTML)
    assert "__IDLE_FPS__" not in page, "the placeholder was never substituted"
    assert "IDLE_FPS=48" in page


def test_the_page_never_ships_an_unsubstituted_placeholder():
    page = face3d.three_sourced(face3d.FACE_HTML)
    assert "__" not in page.split("<script")[0] or "__IDLE_FPS__" not in page


# -- the governor the frame governor broke -----------------------------------

def test_quality_is_judged_against_the_rate_we_asked_for():
    """A second, coupled fault. The quality governor halves point density when
    frames run long, against a fixed 26 ms -- which is 16.7*1.55, calibrated
    for a 60 fps target. Once the face settles to 30 fps ON PURPOSE, a fixed
    threshold reads the deliberate 33.3 ms as overload and thins the face for
    the whole idle, so it visibly thinned whenever ORION stopped talking and
    thickened again when he started.
    """
    line = [l for l in HTML.splitlines() if "quality=lerp" in l]
    assert line, "the quality governor is gone"
    assert "frameTargetMs" in line[0], (
        "quality is still judged against a fixed frame time")
    assert "26" not in line[0], "the 60 fps-only constant is still there"


def test_the_original_sixty_fps_calibration_is_preserved():
    """frameTargetMs*1.55 at a 16.7 ms target is 25.8 ms -- the 26 it replaced.
    The change must not quietly re-tune the case that was already right."""
    factor = float(re.search(r"frameEMA>frameTargetMs\*([0-9.]+)",
                             HTML).group(1))
    assert round((1000 / 60) * factor) == 26


def test_the_frame_target_is_computed_once_per_frame():
    """targetFrameMs() advances the dwell counter, so it is NOT idempotent.
    Calling it again for the quality governor would count the same calm frame
    twice and halve the dwell."""
    body = HTML[HTML.index("function animate()"):]
    assert body.count("targetFrameMs()") == 1, (
        "the frame target is computed more than once per frame")
    assert "let frameTargetMs" in HTML, (
        "the target is not published for step() to read")
