"""
Telling the user's voice apart from ORION's own, coming back through the speakers.

The problem
-----------
Writing audio to a device returns when the buffer ACCEPTS the sound, not when
the room has finished with it. So for a moment after a reply "ends", ORION is
still audible. Streaming the microphone across that gap is how an assistant
hears its own last sentence, decides it was addressed, and answers itself.

ORION's existing defence is ``_ECHO_GUARD_SECONDS = 0.9`` in audio.py: after he
stops speaking, the microphone is ignored for nine hundred milliseconds. That
works, and it costs exactly what it looks like it costs — for most of a second
after every single reply, ORION cannot hear you at all. Answering the instant he
stops, which is how people actually talk, does not work.

Why a level test is not the answer either
-----------------------------------------
"Is the microphone louder than the echo" needs to know how loud the echo IS, and
that is not a constant. It depends on the speaker volume, where the microphone
sits, the room, whether headphones are plugged in, and whether someone has a
hand over the mic. A single tuned number is wrong for almost everyone: too eager
on a loud desktop speaker, too deaf on a quiet laptop.

What this does instead
----------------------
Two signals, neither of which needs tuning per machine.

1. **Content.** Echo is not merely loud, it is *the same sound we just played*.
   Both streams are reduced to a handful of band energies and then, rather than
   merely compared, as much of the recent output as fits is SUBTRACTED from the
   microphone block. Pure echo cancels to almost nothing. A second voice cannot
   be cancelled by ours, because its formants sit in bands where ours were weak,
   so it survives the subtraction. That distinction holds even when the two
   arrive at the same loudness, which is exactly the case a level test gets
   wrong.

2. **A learned gain.** Whenever the content check says "that is definitely just
   echo", the observed microphone-to-output ratio folds into a running estimate.
   The guard therefore calibrates to the actual room within a few seconds of the
   first sentence, and re-calibrates when the volume changes or a hand covers
   the microphone.

Band energies rather than raw spectra also make the comparison sample-rate
agnostic, which matters here: ORION's microphone runs at 16 kHz and his voice
comes back at 24 kHz.

Numpy only — no Qt, no device, no network — so all of it is unit-testable.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

#: Log-spaced edges across the range that carries speech. Coarse on purpose:
#: fine bins track PITCH, and pitch is exactly what differs between two people
#: saying the same word. What we want is the timbre that echo preserves.
_BAND_EDGES = (200, 400, 700, 1100, 1700, 2600, 3800, 5200, 7000)

#: How far back an echo could plausibly have been played.
_HISTORY_SECONDS = 1.5

#: Below this the microphone is carrying room noise and there is nothing to
#: decide either way.
_MIN_LEVEL = 0.06

#: Residual above which the block is treated as a real voice rather than echo.
#: Not a tuned constant so much as a placement: measured residuals for pure echo
#: sit far below this, and a distinct voice far above, with the gap wide enough
#: that the exact value does not matter.
_VOICE_RESIDUAL = 0.22

#: How fast the learned echo gain follows the room.
_GAIN_ALPHA = 0.25


def band_energies(pcm, sample_rate: int) -> np.ndarray:
    """Reduce a PCM block to normalised energy per speech band.

    Normalising makes the comparison independent of absolute level, which is
    what lets a 16 kHz microphone block be compared with a 24 kHz playback
    block at a different volume.
    """
    try:
        x = np.asarray(pcm, dtype=np.float32)
        if x.ndim > 1:
            x = x.reshape(-1)
        if x.size < 64:
            return np.zeros(len(_BAND_EDGES) - 1, dtype=np.float32)
        if float(np.abs(x).max()) > 1.5:            # raw int16
            x = x / 32768.0
        window = np.hanning(x.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(x * window))
        freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
        bands = np.empty(len(_BAND_EDGES) - 1, dtype=np.float32)
        for i in range(len(_BAND_EDGES) - 1):
            mask = (freqs >= _BAND_EDGES[i]) & (freqs < _BAND_EDGES[i + 1])
            bands[i] = float(spectrum[mask].sum()) if mask.any() else 0.0
        total = float(bands.sum())
        if total <= 1e-9:
            return np.zeros_like(bands)
        return bands / total
    except Exception:
        return np.zeros(len(_BAND_EDGES) - 1, dtype=np.float32)


def rms_level(pcm) -> float:
    """Loudness of a block, 0..1."""
    try:
        x = np.asarray(pcm, dtype=np.float32)
        if x.ndim > 1:
            x = x.reshape(-1)
        if x.size == 0:
            return 0.0
        if float(np.abs(x).max()) > 1.5:
            x = x / 32768.0
        return float(min(1.0, float(np.sqrt(float(np.mean(x ** 2)))) * 3.2))
    except Exception:
        return 0.0


class EchoGuard:
    """Decides whether a microphone block is ORION's own voice returning.

    Usage:
        guard.note_output(pcm, 24000, level)     # whenever ORION plays audio
        is_echo, residual = guard.classify(mic_pcm, 16000)
    """

    def __init__(self, *, history_seconds: float = _HISTORY_SECONDS) -> None:
        self._history: deque[tuple[float, np.ndarray, float]] = deque()
        self._history_seconds = float(history_seconds)
        self._gain = 0.0
        self._calibrated = False
        self.last_residual = 0.0

    # ── what we played ───────────────────────────────────────────────────────

    def note_output(self, pcm, sample_rate: int = 24000,
                    level: float | None = None, *,
                    at: float | None = None) -> None:
        """Record a block ORION just sent to the speakers."""
        try:
            now = time.monotonic() if at is None else float(at)
            bands = band_energies(pcm, sample_rate)
            loud = rms_level(pcm) if level is None else float(level)
            self._history.append((now, bands, loud))
            self._expire(now)
        except Exception:
            pass

    def _expire(self, now: float) -> None:
        cutoff = now - self._history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def gain(self) -> float:
        """Learned microphone-to-output ratio for this room."""
        return self._gain

    def reset(self) -> None:
        self._history.clear()
        self._gain = 0.0
        self._calibrated = False
        self.last_residual = 0.0

    # ── the decision ─────────────────────────────────────────────────────────

    def classify(self, pcm, sample_rate: int = 16000, *,
                 at: float | None = None) -> tuple[bool, float]:
        """(is_our_own_echo, residual) for a microphone block.

        The residual is how much of the block SURVIVES subtracting the recent
        output: near zero means we are hearing ourselves, high means someone
        else is talking. Never raises — a guard that throws would take the
        microphone down, which is worse than a wrong answer.
        """
        try:
            now = time.monotonic() if at is None else float(at)
            self._expire(now)
            level = rms_level(pcm)
            if level < _MIN_LEVEL:
                self.last_residual = 0.0
                return False, 0.0          # room noise: nothing to decide
            if not self._history:
                self.last_residual = 1.0
                return False, 1.0          # nothing was playing; it is not echo

            mic = band_energies(pcm, sample_rate)
            if not mic.any():
                self.last_residual = 0.0
                return False, 0.0

            # Subtract the best-matching recent output. "Best" is the one whose
            # removal leaves least behind — the echo may be delayed by anything
            # between a few and a few hundred milliseconds, so the right block
            # to cancel is found rather than assumed.
            best = 1.0
            best_loud = 0.0
            for _when, played, loud in self._history:
                if not played.any():
                    continue
                # Least-squares scale: how much of `played` fits inside `mic`.
                denominator = float(np.dot(played, played))
                if denominator <= 1e-9:
                    continue
                scale = float(np.dot(mic, played)) / denominator
                scale = max(0.0, min(1.25, scale))
                residual_bands = mic - scale * played
                residual = float(np.linalg.norm(residual_bands)
                                 / max(1e-9, float(np.linalg.norm(mic))))
                if residual < best:
                    best = residual
                    best_loud = loud

            self.last_residual = best
            is_echo = best < _VOICE_RESIDUAL
            if is_echo and best_loud > 1e-3:
                # Confidently echo: fold the observed ratio into the room gain.
                observed = level / max(1e-6, best_loud)
                self._gain = ((1.0 - _GAIN_ALPHA) * self._gain
                              + _GAIN_ALPHA * observed) if self._calibrated else observed
                self._calibrated = True
            return is_echo, best
        except Exception:
            self.last_residual = 1.0
            return False, 1.0              # on doubt, treat it as a real voice


_shared: EchoGuard | None = None


def shared() -> EchoGuard:
    """The process-wide guard.

    Playback and capture live in different objects on different threads and
    neither owns the other, so the guard is reached rather than injected: the
    playback thread calls ``note_output`` with what it just sent to the
    speakers, and the microphone thread calls ``classify`` on what it just
    heard. Both operations are short and the history is a bounded deque, so no
    lock is needed for correctness of the decision — a torn read costs at worst
    one misclassified 60 ms block, and the fallback for any doubt is "treat it
    as a real voice", which is the safe direction.
    """
    global _shared
    if _shared is None:
        _shared = EchoGuard()
    return _shared


__all__ = ["EchoGuard", "band_energies", "rms_level", "shared"]
