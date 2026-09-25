"""
Streaming silence compression — the last of the "why does he sound halting?".

Adapted from MARK XL's ``_compress_silence`` (core/tts.py), which caps a neural
voice's punctuation pauses so a long reply keeps its rhythm. Two changes were
needed to make it usable here:

  * MARK XL compresses a COMPLETE utterance array. ORION streams — chunks
    arrive from ElevenLabs and go straight into the renderer's queue, so the
    compressor has to work on a chunk at a time and carry its silence run
    ACROSS chunk boundaries. A per-chunk implementation that forgot its state
    would leave a full-length pause every time one straddled a boundary, which
    is most of them.
  * It works on int16 PCM bytes rather than float arrays, because that is what
    the renderer takes and converting twice per chunk is wasted work on the
    audio path (see the GIL-starvation note in audio.py).

What it does NOT do
-------------------
It never removes silence entirely. Speech without its pauses sounds gabbled and
robotic — the pauses carry the punctuation. It caps a run at
``max_silence_ms`` and leaves everything shorter completely untouched, so
natural phrasing survives and only the dead air goes.

It also never touches non-silent audio: a frame above the threshold resets the
run and is always passed through byte-for-byte.
"""

from __future__ import annotations

import array
from dataclasses import dataclass

#: ~10 ms at 24 kHz — small enough to find the edge of a pause accurately,
#: large enough that the RMS of a frame is meaningful.
FRAME_SAMPLES = 240

#: int16 amplitude below which a frame counts as silence. Deliberately low:
#: clipping quiet speech is far worse than leaving a pause slightly long.
DEFAULT_THRESHOLD = 260

#: Punctuation pauses longer than this are trimmed back to it.
DEFAULT_MAX_SILENCE_MS = 420


@dataclass
class SilenceCompressor:
    """Caps long silences in a stream of 16-bit PCM, one chunk at a time."""

    sample_rate: int = 24_000
    max_silence_ms: int = DEFAULT_MAX_SILENCE_MS
    threshold: int = DEFAULT_THRESHOLD

    def __post_init__(self) -> None:
        self._silent_samples = 0        # length of the run currently open
        self._carry = b""               # partial frame from the previous chunk
        self.trimmed_ms = 0.0           # how much dead air was removed

    @property
    def _max_silent_samples(self) -> int:
        return int(self.max_silence_ms * self.sample_rate / 1000)

    def reset(self) -> None:
        """Start a new utterance. Silence never carries between utterances."""
        self._silent_samples = 0
        self._carry = b""

    def feed(self, chunk: bytes) -> bytes:
        """Return *chunk* with over-long silence runs trimmed.

        Never raises and never returns more audio than it was given.
        """
        if not chunk:
            return b""
        data = self._carry + chunk
        # Work in whole frames; keep any remainder for the next call so a frame
        # is never evaluated on half its samples.
        usable = (len(data) // (FRAME_SAMPLES * 2)) * (FRAME_SAMPLES * 2)
        self._carry = data[usable:]
        if usable == 0:
            return b""

        try:
            samples = array.array("h")
            samples.frombytes(data[:usable])
        except ValueError:                       # odd byte count: pass through
            self._carry = b""
            return data

        out = array.array("h")
        limit = self._max_silent_samples
        for start in range(0, len(samples), FRAME_SAMPLES):
            frame = samples[start:start + FRAME_SAMPLES]
            if self._is_silent(frame):
                self._silent_samples += len(frame)
                if self._silent_samples <= limit:
                    out.extend(frame)
                else:
                    self.trimmed_ms += len(frame) / self.sample_rate * 1000.0
            else:
                self._silent_samples = 0
                out.extend(frame)
        return out.tobytes()

    def flush(self) -> bytes:
        """Emit any partial frame held back at the end of an utterance."""
        tail, self._carry = self._carry, b""
        return tail

    def _is_silent(self, frame) -> bool:
        if not len(frame):
            return True
        # Mean absolute amplitude rather than true RMS: same decision at this
        # threshold, and it avoids a square root per frame on the audio path.
        total = 0
        for value in frame:
            total += value if value >= 0 else -value
        return (total // len(frame)) < self.threshold

    def describe(self) -> str:
        return (f"silence compressor: pauses capped at {self.max_silence_ms} ms, "
                f"{self.trimmed_ms:.0f} ms of dead air removed")


__all__ = ["DEFAULT_MAX_SILENCE_MS", "DEFAULT_THRESHOLD", "FRAME_SAMPLES",
           "SilenceCompressor"]
