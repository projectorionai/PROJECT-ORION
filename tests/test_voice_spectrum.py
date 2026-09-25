"""
Tests for AudioPlaybackThread._spectral_bands() — the real-time spectral
profile that drives the avatar's mouth shape (Mark X.14), replacing the
loudness-only signal that couldn't distinguish an open 'oh' from a narrow
'ee' at the same volume.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.audio import AudioPlaybackThread
from orion_core.constants import RECEIVE_SAMPLE_RATE


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def _tone_chunk(freq_hz: float, duration_s: float = 0.085, amplitude: float = 0.6) -> bytes:
    n = int(RECEIVE_SAMPLE_RATE * duration_s)
    t = np.arange(n) / RECEIVE_SAMPLE_RATE
    samples = (amplitude * 32767 * np.sin(2 * np.pi * freq_hz * t)).astype("<i2")
    return samples.tobytes()


def _renderer() -> AudioPlaybackThread:
    return AudioPlaybackThread(_StubBus())


def test_low_frequency_tone_scores_high_in_low_band():
    renderer = _renderer()
    bands = renderer._spectral_bands(_tone_chunk(200))
    assert bands["low"] > bands["mid"]
    assert bands["low"] > bands["high"]


def test_mid_frequency_tone_scores_high_in_mid_band():
    renderer = _renderer()
    bands = renderer._spectral_bands(_tone_chunk(1000))
    assert bands["mid"] > bands["low"]
    assert bands["mid"] > bands["high"]


def test_high_frequency_tone_scores_high_in_high_band():
    renderer = _renderer()
    bands = renderer._spectral_bands(_tone_chunk(4000))
    assert bands["high"] > bands["low"]
    assert bands["high"] > bands["mid"]


def test_bands_sum_to_roughly_the_whole_spectrum():
    renderer = _renderer()
    bands = renderer._spectral_bands(_tone_chunk(1000))
    # Not exactly 1.0 (some energy falls outside 80-6000Hz, e.g. DC/Nyquist
    # edges), but the dominant band alone should carry most of the energy
    # for a pure tone comfortably inside the 500-2000Hz mid band.
    assert bands["mid"] > 0.5


def test_two_different_frequencies_at_the_same_amplitude_produce_different_shapes():
    """The whole point: loudness-only amplitude can't tell these apart, but
    the spectral profile must — this is the fix for the 'wiggling lips'
    symptom (identical loudness, different mouth shape needed)."""
    renderer = _renderer()
    low = renderer._spectral_bands(_tone_chunk(200, amplitude=0.6))
    high = renderer._spectral_bands(_tone_chunk(4000, amplitude=0.6))
    assert low != high
    assert low["low"] > high["low"]
    assert high["high"] > low["high"]


def test_silence_and_tiny_chunks_degrade_cleanly():
    renderer = _renderer()
    assert renderer._spectral_bands(b"") == {"low": 0.0, "mid": 0.0, "high": 0.0}
    assert renderer._spectral_bands(b"\x00") == {"low": 0.0, "mid": 0.0, "high": 0.0}
    silence = renderer._spectral_bands(b"\x00\x00" * 2048)
    assert silence["low"] == 0.0 and silence["mid"] == 0.0 and silence["high"] == 0.0
