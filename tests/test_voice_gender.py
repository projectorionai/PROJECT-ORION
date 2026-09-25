"""
Speaker gender recognition tests (#14).

Synthetic tones stand in for voices: a low fundamental reads as male, a high
one as female, silence and noise as unknown.  The estimator must never raise
and must stay honest (unknown when it cannot tell).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.voice_gender import (
    SpeakerGenderTracker,
    classify,
    estimate_f0,
    estimate_gender,
    pcm16_to_samples,
)

SR = 16_000


def _tone(freq_hz: float, seconds: float = 0.8, amplitude: int = 6000,
          harmonics: bool = True) -> np.ndarray:
    """A voice-like periodic signal at *freq_hz* (fundamental + a few harmonics)."""
    t = np.arange(int(SR * seconds)) / SR
    sig = np.sin(2 * np.pi * freq_hz * t)
    if harmonics:
        sig += 0.5 * np.sin(2 * np.pi * 2 * freq_hz * t)
        sig += 0.33 * np.sin(2 * np.pi * 3 * freq_hz * t)
    sig = sig / np.max(np.abs(sig))
    return (sig * amplitude).astype(np.int16)


def _pcm(arr: np.ndarray) -> bytes:
    return arr.astype("<i2").tobytes()


# ── F0 estimation ─────────────────────────────────────────────────────────────

def test_estimate_f0_recovers_the_fundamental():
    for freq in (110.0, 150.0, 200.0, 240.0):
        f0, periodicity = estimate_f0(_tone(freq), SR)
        assert f0 == pytest.approx(freq, rel=0.06), f"{freq} → {f0}"
        assert periodicity > 0.3


def test_silence_is_unvoiced():
    f0, periodicity = estimate_f0(np.zeros(SR // 2, dtype=np.int16), SR)
    assert f0 == 0.0 and periodicity == 0.0


def test_white_noise_is_not_confidently_periodic():
    rng = np.random.default_rng(0)
    noise = (rng.normal(0, 3000, SR // 2)).astype(np.int16)
    label, _f0, conf = estimate_gender(noise, SR)
    # Noise may occasionally show a lag peak, but it must not be confident.
    assert label == "unknown" or conf < 0.5


# ── classification ────────────────────────────────────────────────────────────

def test_low_pitch_reads_male_high_pitch_reads_female():
    male_label, _, male_conf = estimate_gender(_tone(120.0), SR)
    female_label, _, female_conf = estimate_gender(_tone(220.0), SR)
    assert male_label == "male" and male_conf > 0
    assert female_label == "female" and female_conf > 0


def test_classify_boundaries():
    assert classify(100.0, 0.9)[0] == "male"
    assert classify(230.0, 0.9)[0] == "female"
    assert classify(300.0, 0.9)[0] == "female" or classify(300.0, 0.9)[0] == "unknown"
    assert classify(0.0, 0.9)[0] == "unknown"          # out of range
    assert classify(120.0, 0.1)[0] == "unknown"        # too weak to trust


def test_estimate_never_raises_on_junk():
    for junk in (b"", b"\x01", [1, 2, 3], None):
        # Should degrade to unknown, never raise.
        try:
            label, _f0, _conf = estimate_gender(junk if junk is not None else [], SR)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"estimate_gender raised on {junk!r}: {exc}")
        assert label in {"male", "female", "unknown"}


# ── PCM decode ─────────────────────────────────────────────────────────────────

def test_pcm_roundtrip():
    arr = _tone(180.0)
    decoded = pcm16_to_samples(_pcm(arr))
    assert decoded is not None
    assert len(decoded) == len(arr)


# ── tracker ─────────────────────────────────────────────────────────────────────

class _Sig:
    def __init__(self):
        self.calls = []

    def emit(self, *payload):
        self.calls.append(payload)


class _Bus:
    def __init__(self):
        self.log = _Sig()
        self.speaker_identified = _Sig()


def test_tracker_settles_on_a_verdict():
    tracker = SpeakerGenderTracker(bus=_Bus(), sample_rate=SR, eval_interval=0.0)
    male_pcm = _pcm(_tone(120.0, seconds=0.6))
    for _ in range(4):
        tracker.feed(male_pcm, SR)
    cur = tracker.current()
    assert cur["label"] == "male"
    assert cur["confidence"] > 0
    assert "male" in tracker.describe().lower()


def test_tracker_reset_clears_state():
    tracker = SpeakerGenderTracker(bus=_Bus(), sample_rate=SR, eval_interval=0.0)
    tracker.feed(_pcm(_tone(220.0, seconds=0.6)), SR)
    assert tracker.current()["label"] in {"female", "male"}
    tracker.reset()
    assert tracker.current()["label"] == "unknown"


def test_tracker_feed_never_raises_on_bad_input():
    tracker = SpeakerGenderTracker(bus=None, sample_rate=SR, eval_interval=0.0)
    tracker.feed(b"", SR)
    tracker.feed(b"\x00\x01\x02", SR)
    assert tracker.current()["label"] == "unknown"


def test_tracker_emits_bus_signal_on_identification():
    bus = _Bus()
    tracker = SpeakerGenderTracker(bus=bus, sample_rate=SR, eval_interval=0.0)
    for _ in range(3):
        tracker.feed(_pcm(_tone(130.0, seconds=0.6)), SR)
    assert bus.speaker_identified.calls, "should emit speaker_identified"
