"""Optional local double-clap wake detector for the existing microphone stream.

Two short, sharp sounds separated by quiet are required. This is deliberately
opt-in: rooms and microphones differ, and a single knock must never wake ORION.
"""

from __future__ import annotations

from array import array
from math import sqrt
import sys


class DoubleClapDetector:
    """Recognise two brief percussive peaks without opening another audio device."""

    def __init__(self) -> None:
        self._quiet_since: float | None = None
        self._burst_since: float | None = None
        self._first_at: float | None = None
        self._last_fire = float("-inf")

    def reset(self) -> None:
        self._quiet_since = None
        self._burst_since = None
        self._first_at = None

    def feed(self, chunk: bytes, now: float) -> bool:
        usable = len(chunk) - len(chunk) % 2
        if usable < 128:
            return False
        samples = array("h")
        samples.frombytes(chunk[:usable])
        if sys.byteorder != "little":
            samples.byteswap()
        # Striding bounds work on the gatekeeper thread to at most 256 samples.
        stride = max(1, len(samples) // 256)
        values = samples[::stride]
        peak = max(abs(value) for value in values) / 32768.0
        rms = sqrt(sum(value * value for value in values) / len(values)) / 32768.0
        quiet = rms < 0.018
        sharp = peak >= 0.20 and rms >= 0.035 and peak >= rms * 2.8
        if quiet:
            if self._quiet_since is None:
                self._quiet_since = now
            if self._burst_since is not None:
                if now - self._burst_since > 0.12:
                    self._first_at = None
                self._burst_since = None
            return False
        if not sharp:
            if self._burst_since is not None and now - self._burst_since > 0.12:
                self._burst_since = None
                self._first_at = None
            self._quiet_since = None
            return False
        if self._burst_since is not None:
            return False
        if self._quiet_since is None or now - self._quiet_since < 0.06:
            return False
        self._burst_since = now
        self._quiet_since = None
        first = self._first_at
        if first is not None and 0.16 <= now - first <= 0.85 and now - self._last_fire >= 5.0:
            self._last_fire = now
            self._first_at = None
            return True
        self._first_at = now
        return False
