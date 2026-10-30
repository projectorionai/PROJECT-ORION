"""The Screen Face's richer emotions (Mark XXXVII).

The owner asked for expressions that read more clearly and a face that never
looks frozen, still cute and never creepy, still cheap and off the GUI thread.
What was added, and what these tests pin:

  * every channel eases through TWO lags in series - an S-curve, so a feeling
    neither lurches in nor stops dead;
  * a feeling lands with a short overshoot (a quick widen on surprise, a small
    sink on sadness), then settles to the same resting value as before;
  * three new channels - squint (the cheeks lift the lower lids: amused), brow
    (one brow up: curious) and sparkle (twinkling glints: happy) - all driven by
    the existing apply_emotion names;
  * a soft head bob while he thinks, a slow head-cock while he listens;
  * an idle flourish every ten seconds or so (a perk, a tilt, a flicker of a
    smile) so a quiet face is never frozen - and none under Reduced motion, none
    while he speaks, thinks, rests or boots.

The rig is not a widget, so almost all of this is plain arithmetic: no Qt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.gui import screen_face as sf  # noqa: E402

DT = 1.0 / 30.0


def rig(seed: int = 5, *, emotion: str | None = None, state: str = "IDLE",
        settle: float = 0.0) -> sf.ScreenFace:
    face = sf.ScreenFace(seed=seed)
    face.skip_boot()
    if emotion:
        face.apply_emotion(emotion, {"intensity": 1.0})
    run(face, settle, state)
    return face


def run(face: sf.ScreenFace, seconds: float, state: str = "IDLE", *, amp: float = 0.0,
        speaking: bool = False, hold_blink: bool = True) -> None:
    for _ in range(int(round(seconds / DT))):
        if hold_blink:
            face._blink_at = face._t + 10.0
            face._blink, face._blink_age = 0.0, -1.0
        face.step(DT, amp, speaking=speaking, state=state)


# ── the vocabulary ───────────────────────────────────────────────────────────

def test_the_new_channels_exist_and_every_emotion_only_uses_known_ones():
    for key in ("squint", "brow", "sparkle"):
        assert key in sf.NEUTRAL and sf.NEUTRAL[key] == 0.0
    for name, preset in sf.EMOTIONS.items():
        assert set(preset) <= set(sf.NEUTRAL), (name, set(preset) - set(sf.NEUTRAL))


def test_the_emotions_use_them_the_way_a_person_would_read_them():
    e = sf.EMOTIONS
    assert e["amused"]["squint"] > e["happy"]["squint"] > 0          # the gentle squint
    assert e["curious"]["brow"] > 0 and e["concerned"]["brow"] > 0   # a brow, a worried brow
    for glad in ("happy", "excited", "grateful", "love"):
        assert e[glad]["sparkle"] >= 0.7, glad                       # twinkling eyes
    for dour in ("sad", "angry", "disappointed", "bored", "concerned"):
        assert "sparkle" not in e[dour], dour                        # a sad face never twinkles
    # never creepy: no negative feeling squints into a glare or smiles
    for name in ("angry", "sad", "disappointed", "alarmed"):
        assert e[name].get("squint", 0.0) == 0.0 and e[name].get("smile", 0.0) == 0.0


@pytest.mark.parametrize("name", sorted(set(sf.EMOTIONS) | set(sf._EMOTION_ALIASES)))
def test_every_emotion_settles_to_valid_painted_channels(name):
    face = rig(emotion=name, settle=2.0)
    pe = face._pe
    for key in ("lid_top", "lid_in", "lid_out", "squint_l", "squint"):
        assert 0.0 <= pe[key] <= 0.8, (name, key, pe[key])
    assert -1.0 <= pe["brow"] <= 1.0 and 0.0 <= pe["sparkle"] <= 1.0
    assert 0.6 <= pe["size"] <= 1.45 and 0.3 <= pe["glow"] <= 1.4
    assert -1.0 <= pe["mouth"] <= 1.0 and abs(pe["tilt"]) < 0.25


# ── smoother transitions: two lags in series ─────────────────────────────────

def test_a_change_starts_gently_instead_of_lurching():
    """One lag has its greatest speed on the very first frame; two lags in
    series start at zero and build up (an S-curve)."""
    face = rig(settle=0.5)
    face.apply_emotion("happy", {"intensity": 1.0})
    seen = []
    last = face._p["smile"]
    for _ in range(4):
        face.step(1 / 60, 0.0, state="IDLE")
        seen.append(face._p["smile"] - last)
        last = face._p["smile"]
    assert seen[0] > 0
    assert seen[0] < seen[1] < seen[2], seen          # accelerating out of rest
    single_lag_first_step = 0.92 * sf._ease(1 / 60, sf._TAU_DEFAULT)
    assert seen[0] < single_lag_first_step * 0.4


def test_a_change_settles_gently_and_arrives_in_about_the_old_time():
    face = rig(settle=0.5)
    face.apply_emotion("happy", {"intensity": 1.0})
    steps = []
    last = face._p["smile"]
    for _ in range(90):
        face.step(1 / 60, 0.0, state="IDLE")
        steps.append(face._p["smile"] - last)
        last = face._p["smile"]
    peak = max(range(len(steps)), key=steps.__getitem__)
    tail = steps[peak:]
    assert all(a >= b - 1e-9 for a, b in zip(tail, tail[1:])), "the approach must decelerate"
    assert abs(face._p["smile"] - 0.92) < 0.01          # settled within 1.5 s
    assert peak * (1 / 60) < 0.4                        # fastest well before the end


def test_power_off_sleep_and_eyelids_keep_their_own_timing():
    assert {"open", "power", "sleep"} == set(sf._SINGLE_LAG)
    face = rig(settle=0.2)
    run(face, 0.5, "SHUTTING DOWN")
    assert face._p["power"] < 0.2


def test_the_two_lags_start_level_and_stay_level_at_rest():
    face = rig(settle=3.0)
    for key, value in face._p.items():
        assert face._p1[key] == pytest.approx(value, abs=1e-3), key
    for key in ("squint", "brow", "sparkle", "tilt"):
        assert face._pe[key] == pytest.approx(face._p[key], abs=0.12), key


# ── a feeling lands: the overshoot ───────────────────────────────────────────

def test_surprise_widens_fast_then_settles_to_the_same_size():
    face = rig(settle=0.6)
    resting = face._pe["size"]
    face.apply_emotion("surprised", {"intensity": 1.0})
    peak, peak_at, wide_at = 0.0, 0.0, None
    for i in range(1, 91):
        face.step(1 / 60, 0.0, state="IDLE")
        if face._pe["size"] > peak:
            peak, peak_at = face._pe["size"], i / 60
        if wide_at is None and face._pe["size"] > resting + 0.15:
            wide_at = i / 60
    final = face._p["size"]
    assert peak > resting + 0.15                       # widened
    assert peak > final + 0.01                         # ...past where it settles: the pop
    assert wide_at is not None and wide_at < 0.2       # a QUICK widen
    assert peak_at < 0.4
    assert abs(face._pe["size"] - final) < 0.01        # and it settles


def test_surprise_is_wider_than_the_eased_value_during_the_first_tenth_of_a_second():
    popped, plain = rig(settle=0.5), rig(settle=0.5)
    popped.apply_emotion("surprised", {})
    plain.apply_emotion("surprised", {})
    plain._onset = 0.0                                  # the same feeling without the overshoot
    for _ in range(6):                                  # 0.1 s
        popped.step(1 / 60, 0.0, state="IDLE")
        plain.step(1 / 60, 0.0, state="IDLE")
    assert popped._pe["size"] > plain._pe["size"] + 0.03


def test_sadness_arrives_with_a_small_sink():
    face = rig(settle=0.5)
    face.apply_emotion("sad", {})
    for _ in range(4):
        face.step(1 / 60, 0.0, state="IDLE")
    assert face._pe["size"] < face._p["size"] - 0.005 and face._onset_gain < 0


def test_a_repeat_of_the_same_feeling_and_a_return_to_neutral_do_not_pop():
    face = rig(emotion="happy", settle=2.0)
    face.apply_emotion("happy", {})                     # the controller re-sends it every tick
    assert face._onset == pytest.approx(0.0, abs=1e-3)
    face.apply_emotion("neutral", {})
    assert face._onset_gain == 0.0                      # quiet return
    face.apply_emotion("excited", {})
    assert face._onset == 1.0 and face._onset_gain > 0.5


def test_every_onset_kind_is_a_real_emotion_and_modest():
    for name, gain in sf._ONSET.items():
        assert name in sf.EMOTIONS and -1.0 <= gain <= 1.0, name


# ── head movement: thinking bob, listening cock ──────────────────────────────

def _tilt_range(face: sf.ScreenFace, seconds: float, state: str):
    low = high = 0.0
    for _ in range(int(seconds / DT)):
        face._blink_at = face._t + 10.0
        face.step(DT, 0.0, state=state)
        low, high = min(low, face._pe["tilt"]), max(high, face._pe["tilt"])
    return low, high


def test_a_thinking_head_bobs_softly_both_ways():
    face = rig(state="PROCESSING", settle=1.5)
    low, high = _tilt_range(face, 5.0, "PROCESSING")
    assert low < -0.03 and high > 0.03                  # a visible bob
    assert low > -0.05 and high < 0.05                  # a soft one: a couple of degrees


def test_an_idle_head_does_not_bob():
    face = rig(settle=1.5)
    low, high = _tilt_range(face, 4.0, "IDLE")          # before any flourish (>= 6 s)
    assert abs(low) < 0.005 and abs(high) < 0.005


def test_a_listening_head_cocks_a_hair():
    face = rig(state="LISTENING", settle=1.5)
    low, high = _tilt_range(face, 7.0, "LISTENING")
    assert high - low > 0.01 and max(abs(low), abs(high)) < 0.03


def test_reduced_motion_shrinks_the_bob():
    full, calm = rig(state="PROCESSING", settle=1.5), rig(state="PROCESSING", settle=1.5)
    calm.set_ambient(0.3)
    full_low, full_high = _tilt_range(full, 5.0, "PROCESSING")
    calm_low, calm_high = _tilt_range(calm, 5.0, "PROCESSING")
    assert (calm_high - calm_low) < (full_high - full_low) * 0.65


# ── idle variety ─────────────────────────────────────────────────────────────

def _flourishes(face: sf.ScreenFace, seconds: float, state: str = "IDLE", **kwargs):
    """The names of the flourishes that started, in order, and when."""
    started = []
    current = None
    for i in range(int(seconds / DT)):
        face._blink_at = face._t + 10.0
        face.step(DT, kwargs.get("amp", 0.0), speaking=kwargs.get("speaking", False), state=state)
        now = face._variety[0] if face._variety else None
        if now and now != current or (now and current is None):
            started.append((now, round(face._t, 2)))
        current = now
    return started


def test_an_idle_face_does_something_small_every_ten_seconds_or_so():
    face = rig(settle=0.5)
    events = _flourishes(face, 150.0)
    assert 7 <= len(events) <= 26, events
    assert len({name for name, _ in events}) >= 4        # variety, not one trick
    times = [t for _, t in events]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert max(gaps) < 22.0, gaps                        # never a long frozen stretch
    assert min(gaps) > 5.0, gaps                         # and not twitchy


def test_it_is_never_the_same_flourish_twice_running():
    events = [name for name, _ in _flourishes(rig(seed=11, settle=0.5), 300.0)]
    assert len(events) > 10
    assert all(a != b for a, b in zip(events, events[1:]))


def test_a_flourish_comes_and_goes_without_leaving_a_trace():
    face = rig(settle=0.5)
    _flourishes(face, 40.0)
    face._variety = None
    face._variety_at = face._t + 1000.0
    run(face, 6.0)                                      # let every channel settle
    assert face._fx_mag == pytest.approx(0.0, abs=0.01)
    for key in ("smile", "mouth", "sparkle", "brow", "squint", "tilt", "asym", "size"):
        assert face._pe[key] == pytest.approx(face._p[key], abs=0.01), key


def test_a_flourish_keeps_the_frame_rate_up_while_it_plays():
    face = rig(settle=0.5)
    face._variety_at = face._t + 0.01
    run(face, 0.3)
    assert face._variety is not None and face.anticipating
    assert face._fx_mag > 0.0 or face._variety_gain < 0.5      # the budget can see it


@pytest.mark.parametrize("state,kwargs", [
    ("SPEAKING", {"amp": 0.5, "speaking": True}), ("PROCESSING", {}), ("STANDBY", {}),
    ("PAUSED", {}), ("OFFLINE", {}), ("MUTED", {}), ("SHUTTING DOWN", {}),
])
def test_no_flourish_while_he_speaks_thinks_rests_or_is_off(state, kwargs):
    face = rig(settle=0.5)
    assert _flourishes(face, 40.0, state, **kwargs) == []


def test_no_flourish_while_booting():
    face = sf.ScreenFace(seed=5)
    face._variety_at = 0.0
    for _ in range(int(sf.BOOT_SECONDS / DT) - 1):
        face.step(DT, 0.0, state="IDLE")
    assert face._variety is None


def test_reduced_motion_means_no_flourishes_at_all():
    face = rig(settle=0.5)
    face.set_ambient(0.3)
    assert _flourishes(face, 60.0) == []
    face.set_ambient(1.0)
    assert _flourishes(face, 40.0)                       # and they come back with it


def test_a_listening_face_flourishes_less_often_and_never_squints():
    """(Measured over 100 s: an idle face dozes off after 120 s of quiet, and
    a dozing face does no flourishes.)"""
    listening, idle = [], []
    for seed in (1, 2, 3, 4):
        listening += _flourishes(rig(seed=seed, state="LISTENING", settle=0.5), 100.0, "LISTENING")
        idle += _flourishes(rig(seed=seed, settle=0.5), 100.0)
    assert listening and all(name != "squint" for name, _ in listening)
    assert len(listening) < len(idle)


def test_with_a_feeling_on_only_harmless_flourishes_are_added():
    face = rig(emotion="sad", settle=0.5)
    names = {name for name, _ in _flourishes(face, 200.0)}
    assert names and names <= set(sf._VARIETY_WITH_EMOTION)
    assert "smile" not in names                          # a smile on top of a frown is a mixed message


def test_flourishes_are_small():
    for name, (duration, peaks) in sf.VARIETY.items():
        assert 0.8 <= duration <= 3.0, name
        assert peaks.get("smile", 0) <= 0.3 and peaks.get("tilt", 0) <= 0.08
        assert peaks.get("size", 0) <= 0.08 and peaks.get("lid_top", 0) <= 0.15


# ── geometry: squint and brow ────────────────────────────────────────────────

def test_a_squint_lifts_the_lower_lid_and_leaves_the_upper_one_alone():
    face = rig(settle=0.5)
    r = 100.0
    before = [face._eye_box(side, r, 0.0, 0.0, 1.0, 1.0) for side in (-1, 1)]
    face._pe["squint"] = 0.6
    after = [face._eye_box(side, r, 0.0, 0.0, 1.0, 1.0) for side in (-1, 1)]
    for old, new in zip(before, after):
        old_top, old_bottom = old[1] - old[4] / 2, old[1] + old[4] / 2
        new_top, new_bottom = new[1] - new[4] / 2, new[1] + new[4] / 2
        assert new_top == pytest.approx(old_top, abs=1e-6)
        assert new_bottom < old_bottom - r * 0.05
        assert new[4] < old[4] and new[2] == old[2]


def test_a_raised_brow_opens_one_eye_and_lowers_the_other_lid():
    face = rig(settle=0.5)
    r = 100.0
    face._pe["brow"] = 1.0
    left, right = (face._eye_box(side, r, 0.0, 0.0, 1.0, 1.0) for side in (-1, 1))
    assert right[3] > left[3] * 1.05                    # the raised brow's eye is taller
    assert max(face._lid_drops(-1)) > max(face._lid_drops(1))       # the other lid is lower
    face._pe["brow"] = -1.0                             # mirrored
    left, right = (face._eye_box(side, r, 0.0, 0.0, 1.0, 1.0) for side in (-1, 1))
    assert left[3] > right[3] * 1.05
    assert max(face._lid_drops(1)) > max(face._lid_drops(-1))


def test_curious_is_a_brow_and_a_tilt_not_a_second_face():
    face = rig(emotion="curious", state="LISTENING", settle=3.0)
    assert face._pe["brow"] > 0.5 and face._pe["tilt"] > 0.03
    right, left = face._eye_box(1, 100.0, 0.0, 0.0, 1.0, 1.0), face._eye_box(-1, 100.0, 0.0, 0.0, 1.0, 1.0)
    assert right[3] > left[3]


# ── the picture: sparkle never leaves the eye ────────────────────────────────

@pytest.fixture(scope="module")
def painter_env():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _frame(face: sf.ScreenFace, w: int = 640, h: int = 440):
    from PyQt6.QtGui import QColor, QImage, QPainter

    from orion_core.constants import C

    bg = QColor(C.BG)
    image = QImage(w, h, QImage.Format.Format_RGB32)
    image.fill(bg)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    cx, cy, r = sf.face_geometry(w, h)
    face.paint(painter, cx, cy, r, QColor(C.PRI), QColor(C.ACCENT), bg)
    painter.end()
    return image


def _white(image, box):
    x0, y0, x1, y1 = (int(v) for v in box)
    count = 0
    for y in range(max(0, y0), min(image.height(), y1)):
        for x in range(max(0, x0), min(image.width(), x1)):
            c = image.pixelColor(x, y)
            if c.red() > 235 and c.green() > 215 and c.blue() > 215:
                count += 1
    return count


def test_happy_eyes_twinkle_and_only_on_the_eyes(painter_env):
    w, h = 640, 440
    cx, cy, r = sf.face_geometry(w, h)
    # The twinkle is the Companion design's; Sentinel (default) has none.
    twinkling = rig(emotion="happy", settle=2.0)
    flat = rig(emotion="happy", settle=2.0)
    sleek = _frame(twinkling, w, h)
    twinkling.set_design("companion")
    flat.set_design("companion")
    flat._pe["sparkle"] = 0.0
    flat._p["sparkle"] = 0.0
    a, b = _frame(twinkling, w, h), _frame(flat, w, h)
    eyes = [(cx + side * 0.52 * r - 0.34 * r, cy - 0.1 * r - 0.5 * r,
             cx + side * 0.52 * r + 0.34 * r, cy - 0.1 * r + 0.5 * r) for side in (-1, 1)]
    assert sum(_white(a, box) for box in eyes) > sum(_white(b, box) for box in eyes)
    # Nothing changed anywhere that is not an eye: the glints cannot float off the glass.
    outside = 0
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in eyes):
                continue
            if a.pixel(x, y) != b.pixel(x, y):
                outside += 1
    assert outside == 0, outside
    assert sum(_white(sleek, box) for box in eyes) == 0, "Sentinel draws no cartoon twinkle"


def test_a_sad_face_paints_no_glints(painter_env):
    face = rig(emotion="sad", settle=2.0)
    assert face._pe["sparkle"] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("name", ["happy", "amused", "curious", "concerned", "surprised", "excited",
                                  "love", "grateful", "sad", "angry", "confused", "shy"])
@pytest.mark.parametrize("size", [(120, 100), (400, 300), (1500, 900)])
def test_every_new_expression_paints_at_every_size(painter_env, name, size):
    face = rig(emotion=name, settle=1.0)
    face._variety = ("smile", face._t - 0.5, 1.8, dict(sf.VARIETY["smile"][1]))
    face._variety_gain = 1.0
    face.step(DT, 0.0, state="IDLE")
    image = _frame(face, *size)
    assert not image.isNull()


def test_a_flourish_in_progress_paints(painter_env):
    for name, (duration, peaks) in sf.VARIETY.items():
        face = rig(settle=0.5)
        face._variety = (name, face._t - duration / 2.0, duration, dict(peaks))
        face._variety_gain = 1.0
        face.step(DT, 0.0, state="IDLE")
        assert _frame(face) is not None


# ── cost ─────────────────────────────────────────────────────────────────────

def test_the_rig_step_stays_cheap():
    """Per-frame arithmetic only: far under a millisecond even on a busy
    machine, so the face is still dominated by painting (about a millisecond at
    640 px) and still lives on the pipeline's worker thread."""
    import time

    face = rig(emotion="happy", settle=1.0)
    start = time.perf_counter()
    for _ in range(600):
        face.step(DT, 0.0, state="IDLE")
    per_step_ms = (time.perf_counter() - start) * 1000.0 / 600
    assert per_step_ms < 0.6, per_step_ms


def test_the_budget_still_gets_a_channel_vector_of_one_length():
    face = rig(settle=0.5)
    lengths = set()
    for state in ("IDLE", "LISTENING", "PROCESSING", "SPEAKING"):
        run(face, 0.5, state, amp=0.3, speaking=state == "SPEAKING")
        lengths.add(len(face.channels))
    assert lengths == {13}


def test_the_face_is_still_headless_safe_and_off_the_gui_thread():
    source = (ROOT / "orion_core" / "gui" / "screen_face.py").read_text(encoding="utf-8")
    import re

    assert not re.search(r"^\s*(import|from)\s+(numpy|PyQt6\.QtWebEngine)", source, re.M)
    assert "face_pipeline" in source                     # still rendered on the worker thread
