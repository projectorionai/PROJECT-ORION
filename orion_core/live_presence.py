"""
Complete utterances, assembled from the live microphone stream.

The offline transcriber gets this for free: ``transcribe_pcm`` is handed a
whole utterance that some upstream VAD already delimited, so reading who spoke
and how they sounded is one call in one obvious place. The Gemini Live path has
no such moment. It forwards **every** chunk, voice and silence alike, because
the server's own VAD is what detects end-of-turn — filtering to voiced chunks
is what once made the live channel deaf.

So on Live there was no complete utterance anywhere to read. ``voice_presence``
was only ever fed from the offline path, which meant that in a live session
``voice_tone`` answered from a stale reading or none at all, the enrolled
voiceprint was never compared against anything, and the only-my-voice gate
could not apply. Half of ORION's hearing worked only in the half of the
pipeline most people do not use.

This assembles that missing utterance: voiced chunks accumulate here while
someone is talking, and when the speech ends the whole thing is handed over in
one piece. It is deliberately a plain object with no threads, no Qt and no
audio device, so the gate thread can call it directly and a test can drive it
with bytes.

What it costs the capture thread
--------------------------------
An append to a list and an integer add, per chunk. Everything expensive — the
speaker embedding, the pitch tracking — happens elsewhere, on whatever thread
the owner of this object chooses. The realtime path is not a place to spend
13 milliseconds on an ONNX session.

Why there is a ceiling
----------------------
``MAX_SECONDS`` caps what is held. Somebody who talks for five minutes without
a pause should not accumulate five minutes of PCM in memory, and the speaker
encoder learns nothing from the fourth minute that it did not know after the
second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

#: What the Live path captures at.
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

#: Shortest utterance worth reading. Below this the speaker embedding is noisy
#: and the tone reading is guesswork, and acting on either would be worse than
#: not acting at all.
MIN_SECONDS = 1.2

#: Most audio held for one utterance. A long monologue is truncated rather
#: than accumulated: the encoder gains nothing after the first few seconds.
MAX_SECONDS = 12.0

#: How much speech to collect before asking whose voice it is, when the gate
#: is on. It has to be enough for the encoder to be worth trusting and little
#: enough that a stranger is cut off early rather than answered.
DECIDE_AFTER_SECONDS = 1.6


def _seconds(byte_count: int, sample_rate: int) -> float:
    return byte_count / float(max(1, sample_rate * BYTES_PER_SAMPLE))


@dataclass
class LiveUtterance:
    """Voiced audio accumulating between two silences.

    Not thread-safe in any formal sense, and it does not need to be: one
    capture thread feeds it and reads it, and the only field another thread
    writes is ``muted``, which is a single boolean.
    """

    sample_rate: int = SAMPLE_RATE
    max_seconds: float = MAX_SECONDS
    min_seconds: float = MIN_SECONDS
    decide_after: float = DECIDE_AFTER_SECONDS

    _chunks: list = field(default_factory=list)
    _bytes: int = 0
    _truncated: bool = False
    #: Set from the event loop when an utterance is judged to belong to
    #: somebody other than ORION's owner. Read by the capture thread, which
    #: then stops forwarding for the rest of this utterance.
    muted: bool = False
    #: True once the owner check for THIS utterance has been started, so it is
    #: asked once rather than on every chunk after the threshold.
    asked: bool = False

    # ── accumulating ─────────────────────────────────────────────────────────

    def feed(self, chunk: bytes) -> None:
        """Add a voiced chunk. Cheap enough for the realtime thread."""
        if not chunk:
            return
        if _seconds(self._bytes, self.sample_rate) >= self.max_seconds:
            self._truncated = True
            return
        self._chunks.append(chunk)
        self._bytes += len(chunk)

    @property
    def seconds(self) -> float:
        return _seconds(self._bytes, self.sample_rate)

    @property
    def truncated(self) -> bool:
        return self._truncated

    def ready_to_decide(self) -> bool:
        """Whether there is enough speech to ask whose voice this is.

        Answers True once, so the caller starts one check per utterance rather
        than one per chunk after the threshold.
        """
        if self.asked or self.seconds < self.decide_after:
            return False
        self.asked = True
        return True

    def audio(self) -> bytes:
        """Everything accumulated so far, without ending the utterance."""
        return b"".join(self._chunks)

    # ── finishing ────────────────────────────────────────────────────────────

    def end(self) -> bytes:
        """Take the finished utterance and start a fresh one.

        Returns b"" when there was not enough speech to be worth reading,
        which is the common case: a cough, a chair, one word of agreement.
        """
        audio = self.audio() if self.seconds >= self.min_seconds else b""
        self.reset()
        return audio

    def reset(self) -> None:
        self._chunks = []
        self._bytes = 0
        self._truncated = False
        self.muted = False
        self.asked = False


class LivePresenceBridge:
    """Carries finished live utterances to VoicePresence, and back again.

    The capture thread calls :meth:`observe` and :meth:`decide`; both return
    immediately, having handed the work to *schedule*. Nothing here blocks,
    and nothing here raises: an exception on the capture thread would take
    ORION's hearing with it, and a tone reading is not worth that.
    """

    def __init__(self, presence: Any = None,
                 schedule: Callable[[Callable[[], Any]], None] | None = None,
                 sample_rate: int = SAMPLE_RATE,
                 log: Callable[[str], None] | None = None) -> None:
        self.presence = presence
        self.schedule = schedule
        self.sample_rate = sample_rate
        self.log = log
        self.utterance = LiveUtterance(sample_rate=sample_rate)

    # -- the two things the capture thread asks for --------------------------

    def observe(self, pcm: bytes) -> None:
        """Read who spoke and how they sounded, off this thread."""
        if not pcm or self.presence is None or self.schedule is None:
            return
        self._hand_off(lambda: self.presence.observe_pcm(pcm, self.sample_rate))

    def decide(self, pcm: bytes, on_verdict: Callable[[Any], None]) -> None:
        """Ask whose voice this is, and report back off this thread."""
        if not pcm or self.schedule is None:
            return

        def work() -> None:
            from . import voiceprint

            on_verdict(voiceprint.should_listen(pcm, self.sample_rate))

        self._hand_off(work)

    def _hand_off(self, work: Callable[[], Any]) -> None:
        try:
            self.schedule(work)
        except Exception as exc:                # pragma: no cover - defensive
            if self.log is not None:
                try:
                    self.log(f"AUDIO: presence hand-off failed - {exc}")
                except Exception:
                    pass


__all__ = [
    "DECIDE_AFTER_SECONDS", "MAX_SECONDS", "MIN_SECONDS", "SAMPLE_RATE",
    "LivePresenceBridge", "LiveUtterance",
]
