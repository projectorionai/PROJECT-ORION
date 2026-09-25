"""
The head's nod on a stressed syllable.

What was there before subtracted a fraction of the current amplitude from the
head's pitch. That is not a nod: it ties head position to the waveform, so the
head rides every consonant and returns the instant the sound does — a jitter
locked to loudness rather than a gesture. A nod *lags* the syllable that caused
it and outlives it, and that lag is most of what makes it read as emphasis.

So the tests here are about timing and restraint, not about whether a number
changed. Four things have to hold at once: it fires on a rise, it does not fire
on silence, it does not fire on speech that is merely loud, and it never buzzes.

Measured through ``_nod_offset``, which the head records each frame precisely
so this can be checked rather than assumed.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Frames per second the head is stepped at in these measurements.
DT = 1.0 / 30.0

_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.gui.holo_head import (
    _NOD_DEPTH, _NOD_DURATION, _NOD_REFRACTORY, HoloHead)

DT = 1.0 / 30.0

def trace(amps, speaking=True):
    head = HoloHead()
    head._speaking = speaking
    out = []
    for amp in amps:
        head.step(DT, amp, speaking=speaking, state="speaking")
        out.append(head._nod_offset)
    return out

def dips(values, floor=-0.01):
    return sum(1 for i in range(1, len(values) - 1)
               if values[i] < values[i - 1]
               and values[i] <= values[i + 1]
               and values[i] < floor)

out = {{"depth": _NOD_DEPTH, "duration": _NOD_DURATION,
        "refractory": _NOD_REFRACTORY}}

# Silence.
out["silence_max"] = max(abs(v) for v in trace([0.004] * 90))

# Not speaking at all, with plenty of amplitude on the line.
out["not_speaking_max"] = max(abs(v) for v in trace([0.4] * 90, speaking=False))

# One stressed syllable after 30 frames of steady speech.
one = trace([0.05] * 30 + [0.30] * 3 + [0.05] * 40)
out["one_trace"] = one
out["one_dip"] = min(one)
out["one_dip_frame"] = one.index(min(one))
out["one_settles"] = max(abs(v) for v in one[-10:])
out["one_after_sound"] = sum(1 for v in one[33:] if abs(v) > 0.005)
out["one_before_stress"] = max(abs(v) for v in one[:30])

# Sustained loudness: no rises, so nothing is stressed.
out["sustained_dips"] = dips(trace([0.35] * 150))
out["steady_loud_min"] = min(trace([0.22] * 40 + [0.23] * 40 + [0.22] * 40))

# A quiet speaker: the same relative rise, a fifth of the absolute level.
out["quiet_dip"] = min(trace([0.04] * 30 + [0.12] * 3 + [0.04] * 40))

# Four stresses, with room for the last to finish.
rhythm = []
for _ in range(4):
    rhythm += [0.06] * 27 + [0.28] * 3
out["rhythm_dips"] = dips(trace(rhythm + [0.06] * 20))

# A long loud passage that does rise repeatedly: bounded by the refractory.
noisy = []
for _ in range(60):
    noisy += [0.06, 0.40]
out["noisy_dips"] = dips(trace(noisy))
out["noisy_seconds"] = len(noisy) * DT

print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def nod() -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(root=str(ROOT).replace("\\", "/"))],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail(f"probe produced nothing:\n{proc.stdout[-1500:]}")


# ── it fires when it should ───────────────────────────────────────────────────

def test_a_stressed_syllable_dips_the_head(nod):
    assert nod["one_dip"] < -0.01, "no nod on a plainly stressed syllable"


def test_the_nod_lags_the_syllable(nod):
    """The lag is most of what makes it read as emphasis rather than vibration.

    A head that moves in the same frame as the sound looks driven by it; one
    that moves just after looks like it meant to.
    """
    lag_ms = (nod["one_dip_frame"] - 30) * DT * 1000
    assert 30 <= lag_ms <= 220, f"nod peaked {lag_ms:.0f} ms after the stress"


def test_the_nod_outlives_the_sound(nod):
    """It plays out on its own clock. Tied to amplitude it would end with the
    syllable, which is the behaviour this replaced."""
    assert nod["one_after_sound"] >= 3


def test_the_nod_returns_to_rest(nod):
    assert nod["one_settles"] < 0.001


def test_the_dip_is_one_smooth_movement(nod):
    """Down and back up with no step at either edge — a discontinuity in head
    pitch reads as a glitch, not a gesture."""
    trace = nod["one_trace"]
    steps = [abs(trace[i] - trace[i - 1]) for i in range(1, len(trace))]
    assert max(steps) < nod["depth"] * 0.5, "the nod jumps rather than moves"


def test_a_quiet_speaker_still_gets_a_nod(nod):
    """The threshold is a rise above the speech around it, not a fixed level.

    A fixed threshold nods through every word of a loud sentence and through
    none of a quiet one.
    """
    assert nod["quiet_dip"] < -0.01


def test_each_stress_gets_its_own_nod(nod):
    assert nod["rhythm_dips"] == 4


# ── and not when it should not ────────────────────────────────────────────────

def test_silence_does_not_nod(nod):
    assert nod["silence_max"] < 0.001


def test_a_head_that_is_not_speaking_does_not_nod(nod):
    """Amplitude on the line while ORION is listening is the user's voice."""
    assert nod["not_speaking_max"] < 0.001


def test_the_first_syllable_is_not_treated_as_a_stress(nod):
    """The defect this was built with.

    The first audible frame has nothing to be louder than. Compared against a
    running level still sitting at zero it always reads as a huge rise, so
    every utterance opened with a nod on its first syllable whether or not
    that syllable was stressed.
    """
    assert nod["one_before_stress"] < 0.001


def test_speech_that_is_merely_loud_does_not_nod(nod):
    """Loudness is not emphasis. A speaker with a loud delivery would
    otherwise nod continuously."""
    assert nod["sustained_dips"] == 0
    assert nod["steady_loud_min"] > -0.001


def test_it_cannot_buzz(nod):
    """Without a refractory period a passage that rises repeatedly fires a nod
    every few frames and the head vibrates."""
    ceiling = nod["noisy_seconds"] / nod["refractory"]
    assert nod["noisy_dips"] <= ceiling + 1, (
        f"{nod['noisy_dips']} nods in {nod['noisy_seconds']:.1f}s exceeds "
        f"the refractory ceiling of {ceiling:.0f}")


# ── restraint ─────────────────────────────────────────────────────────────────

def test_the_nod_is_small(nod):
    """A nod you notice as a nod is too big — this fires several times a
    sentence."""
    assert math.degrees(nod["depth"]) < 6.0


def test_the_nod_is_quick(nod):
    """Shorter reads as a twitch; longer and it is still returning when the
    next stressed syllable lands."""
    assert 0.15 <= nod["duration"] <= 0.45
