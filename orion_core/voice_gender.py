"""
Speaker gender recognition (#14) — ORION distinguishing male vs female voices.

A small, offline, dependency-light estimator.  It measures the fundamental
frequency (F0, perceived pitch) of a short voiced audio frame by
autocorrelation and classifies the speaker as ``male``, ``female`` or
``unknown``.  Typical adult speech: male F0 ≈ 85–165 Hz, female ≈ 165–255 Hz;
the boundary sits near 165 Hz with a deliberate "uncertain" band around it.

Design principles
-----------------
* **Never raises.**  Silence, noise, a signal without clear periodicity, or a
  missing numpy all return ``('unknown', 0.0, 0.0)``.  This code sits next to
  the live microphone path, so it must be impossible for it to break capture.
* **Honest, not confident.**  It reports a probable gender with a confidence,
  and stays ``unknown`` unless the pitch is clearly periodic and in range — a
  wrong guess ("that was a woman" when it was the user) is worse than "I'm not
  sure".
* **Cheap.**  One windowed autocorrelation over ≤1.5 s of 16 kHz audio; the
  stateful tracker throttles evaluation so the realtime thread only ever
  appends bytes between infrequent, sub-millisecond evaluations.
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any, Sequence

# Vocal-range gate (Hz): outside this, treat as non-speech / unknown.
_F0_MIN_HZ = 70.0
_F0_MAX_HZ = 320.0
# Male / female decision boundaries with an uncertain band between them.
_MALE_MAX_HZ = 155.0
_FEMALE_MIN_HZ = 175.0
# Minimum autocorrelation-peak strength (0..1) to trust the pitch.
_MIN_PERIODICITY = 0.30
# Minimum RMS (int16 scale) to count a frame as speech rather than silence.
_MIN_RMS = 180.0


def _to_float_array(samples: Any):
    try:
        import numpy as np
    except Exception:
        return None
    try:
        arr = np.asarray(samples, dtype=np.float64).ravel()
    except Exception:
        return None
    return arr if arr.size else None


def pcm16_to_samples(pcm: bytes):
    """Decode little-endian int16 PCM bytes to a numpy float array (or None)."""
    try:
        import numpy as np
    except Exception:
        return None
    if not pcm:
        return None
    try:
        return np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    except Exception:
        return None


def estimate_f0(samples: Sequence[float] | Any, sample_rate: int) -> tuple[float, float]:
    """Return ``(f0_hz, periodicity)`` for *samples*; ``(0.0, 0.0)`` if unvoiced.

    ``periodicity`` is the normalised autocorrelation peak (0..1) — how strongly
    the frame repeats at the detected period, i.e. how voiced it is.
    """
    import_ok = True
    try:
        import numpy as np
    except Exception:
        import_ok = False
    if not import_ok:
        return 0.0, 0.0
    arr = _to_float_array(samples)
    if arr is None or arr.size < 2:
        return 0.0, 0.0
    sr = int(sample_rate or 0)
    if sr <= 0:
        return 0.0, 0.0

    # Silence gate on RMS energy.
    rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
    if rms < _MIN_RMS:
        return 0.0, 0.0

    arr = arr - float(np.mean(arr))         # remove DC offset
    window = np.hanning(arr.size)
    arr = arr * window

    # Autocorrelation via FFT (fast, O(N log N)).
    n = int(1 << (int(arr.size).bit_length() + 1))
    spectrum = np.fft.rfft(arr, n=n)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n)[: arr.size]
    if acf.size == 0 or acf[0] <= 0:
        return 0.0, 0.0

    min_lag = max(1, int(sr / _F0_MAX_HZ))
    max_lag = min(arr.size - 1, int(sr / _F0_MIN_HZ))
    if max_lag <= min_lag:
        return 0.0, 0.0

    segment = acf[min_lag:max_lag + 1]
    if segment.size == 0:
        return 0.0, 0.0
    peak_offset = int(np.argmax(segment))
    lag = min_lag + peak_offset
    periodicity = float(acf[lag] / acf[0])
    if lag <= 0:
        return 0.0, 0.0
    f0 = sr / float(lag)
    return f0, max(0.0, min(1.0, periodicity))


def classify(f0_hz: float, periodicity: float) -> tuple[str, float]:
    """Map an ``(f0, periodicity)`` pair to ``(label, confidence)``."""
    if periodicity < _MIN_PERIODICITY or not (_F0_MIN_HZ <= f0_hz <= _F0_MAX_HZ):
        return "unknown", 0.0
    if f0_hz <= _MALE_MAX_HZ:
        label = "male"
        # Confidence grows as pitch sits further below the uncertain band.
        margin = (_MALE_MAX_HZ - f0_hz) / (_MALE_MAX_HZ - _F0_MIN_HZ)
    elif f0_hz >= _FEMALE_MIN_HZ:
        label = "female"
        margin = (f0_hz - _FEMALE_MIN_HZ) / (_F0_MAX_HZ - _FEMALE_MIN_HZ)
    else:
        # Uncertain band: lean to the nearer side but with low confidence.
        mid = (_MALE_MAX_HZ + _FEMALE_MIN_HZ) / 2.0
        label = "male" if f0_hz < mid else "female"
        margin = 0.1
    confidence = max(0.0, min(1.0, 0.5 * periodicity + 0.5 * margin))
    return label, round(confidence, 3)


def estimate_gender(samples: Sequence[float] | Any, sample_rate: int) -> tuple[str, float, float]:
    """Convenience: ``(label, f0_hz, confidence)`` for one frame."""
    f0, periodicity = estimate_f0(samples, sample_rate)
    label, confidence = classify(f0, periodicity)
    return label, round(f0, 1), confidence


class SpeakerGenderTracker:
    """Stateful, thread-safe smoother fed voiced PCM from the capture thread.

    The realtime gatekeeper calls :meth:`feed` with each voiced chunk; that only
    appends bytes.  Heavy evaluation runs at most once per ``eval_interval`` and
    over a short rolling window, so the audio thread is never blocked.  The
    smoothed verdict (a majority of recent estimates, weighted by confidence) is
    read by the ``speaker_id`` tool and surfaced on the bus.
    """

    def __init__(
        self,
        bus: Any | None = None,
        sample_rate: int = 16_000,
        telemetry: Any | None = None,
        eval_interval: float = 1.1,
        window_seconds: float = 1.2,
    ) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.sample_rate = int(sample_rate)
        self._eval_interval = float(eval_interval)
        self._min_bytes = int(self.sample_rate * 0.45) * 2       # 0.45 s int16
        self._max_bytes = int(self.sample_rate * window_seconds) * 2
        self._buf = bytearray()
        self._recent: deque[tuple[str, float]] = deque(maxlen=6)
        self._last_eval = 0.0
        self._current: dict[str, Any] = {
            "label": "unknown", "f0": 0.0, "confidence": 0.0, "at": "",
        }
        self._lock = Lock()

    # ── realtime intake (audio thread) ────────────────────────────────────────

    def feed(self, pcm: bytes, sample_rate: int | None = None) -> None:
        """Append a voiced PCM chunk; evaluate opportunistically. Never raises."""
        try:
            if not pcm:
                return
            with self._lock:
                self._buf.extend(pcm)
                if len(self._buf) > self._max_bytes:
                    del self._buf[: len(self._buf) - self._max_bytes]
                now = time.monotonic()
                if len(self._buf) < self._min_bytes:
                    return
                if (now - self._last_eval) < self._eval_interval:
                    return
                self._last_eval = now
                snapshot = bytes(self._buf)
            self._evaluate(snapshot, int(sample_rate or self.sample_rate))
        except Exception:
            # This lives next to the live mic — swallow everything.
            pass

    def _evaluate(self, pcm: bytes, sample_rate: int) -> None:
        samples = pcm16_to_samples(pcm)
        if samples is None:
            return
        label, f0, confidence = estimate_gender(samples, sample_rate)
        if label == "unknown":
            return
        with self._lock:
            self._recent.append((label, confidence))
            smoothed, agg_conf = self._smoothed_locked()
            changed = smoothed != self._current.get("label")
            self._current = {
                "label": smoothed,
                "f0": f0,
                "confidence": round(agg_conf, 3),
                "at": time.strftime("%H:%M:%S"),
            }
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr(f"speaker.{smoothed}")
            except Exception:
                pass
        if changed and self.bus is not None:
            try:
                self.bus.log.emit(
                    f"SPEAKER: voice sounds {smoothed} "
                    f"(~{f0:.0f} Hz, confidence {agg_conf:.2f}).")
            except Exception:
                pass
        if self.bus is not None:
            emit = getattr(self.bus, "speaker_identified", None)
            if emit is not None:
                try:
                    emit.emit(dict(self._current))
                except Exception:
                    pass

    def _smoothed_locked(self) -> tuple[str, float]:
        """Confidence-weighted majority over recent estimates (lock held)."""
        scores: dict[str, float] = {}
        for label, conf in self._recent:
            scores[label] = scores.get(label, 0.0) + max(0.05, conf)
        if not scores:
            return "unknown", 0.0
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1.0
        return best, scores[best] / total

    # ── read side (any thread) ────────────────────────────────────────────────

    def current(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._current)

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()
            self._recent.clear()
            self._current = {"label": "unknown", "f0": 0.0, "confidence": 0.0, "at": ""}

    def describe(self) -> str:
        cur = self.current()
        label = cur.get("label", "unknown")
        if label == "unknown" or not cur.get("at"):
            return "I haven't heard a clear enough voice to tell yet."
        return (f"The last voice I heard sounded {label} "
                f"(around {cur.get('f0', 0):.0f} Hz, "
                f"confidence {cur.get('confidence', 0):.2f}).")
