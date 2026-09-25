"""
Who just spoke, and how they sounded.

Two requests that turn out to be the same piece of work:

    "I want ORION to be able to distinguish between my different emotions"
    "ORION must be able to recognise my mum's voice too"

Both are questions about an utterance's ACOUSTICS rather than its words, both
need the same waveform, and — critically — each needs the other to be any good.
Emotion is read against a speaker's own normal (see voice_emotion), so it needs
to know whose normal to use; and a household with more than one voice in it
needs to not attribute mum's pitch range to the user. Doing them separately
would mean measuring the same audio twice and then getting both answers wrong
whenever anyone but the user is talking.

So this is the one place that answers "who is this and how do they sound",
identifying first and reading tone against that person's baseline.

Where it runs, and why not in the real-time thread
--------------------------------------------------
Given a COMPLETE utterance, never a live chunk. ``audio.py``'s gate thread is
a hard real-time path — PortAudio callbacks must not do inference — and
voice_speaker_id.py already draws that boundary deliberately. Speaker
embedding and pitch tracking are both far too slow to sit there. This is
called from the transcription path instead, which already has the whole
utterance assembled and is already off the audio device thread.

Everything degrades to silence rather than to an error. If resemblyzer is not
installed, tone still works and the speaker reads "you". If the audio is too
short, both say so and nothing is claimed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .voice_emotion import EmotionReader, Reading

#: Below this, speaker identification is not attempted — an embedding from a
#: fragment is noise, and a confident wrong name is worse than no name.
MIN_ID_SECONDS = 1.0

#: The name used when nobody has been identified. Not "unknown": the
#: overwhelmingly common case is the user at their own machine, and calling
#: them a stranger would put their tone readings under a separate baseline.
DEFAULT_SPEAKER = "you"


@dataclass
class Presence:
    """What ORION could tell about one utterance."""

    speaker: str = DEFAULT_SPEAKER
    speaker_confidence: float = 0.0
    identified: bool = False
    emotion: Reading = field(default_factory=Reading)
    at: float = field(default_factory=time.time)

    @property
    def is_known_person(self) -> bool:
        """A named individual ORION has actually been introduced to."""
        return self.identified and self.speaker != DEFAULT_SPEAKER

    def describe(self) -> str:
        who = ("You" if not self.is_known_person
               else self.speaker[:1].upper() + self.speaker[1:])
        if self.emotion.emotion == "neutral":
            return f"{who} - level."
        return f"{who} - sounds {self.emotion.emotion} ({self.emotion.because})."

    def prompt_line(self) -> str:
        """A line for the model's context, or "" when there is nothing to say.

        Deliberately empty most of the time. Telling the model on every single
        turn that the user "sounds neutral" is noise that buys nothing and
        costs context; and prefixing every turn with a mood guess encourages it
        to comment on the user's emotional state unprompted, which gets old
        immediately.
        """
        parts = []
        if self.is_known_person:
            parts.append(f"The person speaking is {self.speaker}, not the usual user")
        if self.emotion.certain_enough_to_mention:
            parts.append(f"they sound {self.emotion.emotion} "
                         f"({self.emotion.because})")
        if not parts:
            return ""
        return ("[voice] " + "; ".join(parts) +
                ". Take this as a hint about tone only - do not announce it, "
                "and do not assume it is right if what they say suggests "
                "otherwise.")


class VoicePresence:
    """Identifies the speaker and reads their tone, from a complete utterance."""

    def __init__(self, bus: Any | None = None, speaker_id: Any = None,
                 emotion: EmotionReader | None = None) -> None:
        self.bus = bus
        self.speaker_id = speaker_id
        self.emotion = emotion or EmotionReader()
        self.last: Presence | None = None
        self._announced: dict[str, float] = {}

    # ── the one entry point ──────────────────────────────────────────────────

    def observe(self, audio: Any, sample_rate: int = 16000,
                learn: bool = True) -> Presence:
        """Read one complete utterance. Never raises."""
        try:
            return self._observe(audio, sample_rate, learn)
        except Exception as exc:                 # never break the voice path
            if self.bus is not None:
                try:
                    self.bus.log.emit(f"VOICE: presence read failed - {exc}")
                except Exception:
                    pass
            return Presence()

    def observe_pcm(self, pcm: bytes, sample_rate: int = 16000,
                    learn: bool = True) -> Presence:
        """As :meth:`observe`, for the 16-bit mono PCM the capture path carries."""
        if not pcm:
            return Presence()
        try:
            import numpy as np
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return Presence()
        return self.observe(samples, sample_rate, learn)

    def _observe(self, audio: Any, sample_rate: int, learn: bool) -> Presence:
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32).flatten()
        seconds = samples.size / float(sample_rate or 1)
        presence = Presence()

        # Identify FIRST: the tone reading needs to know whose baseline to use.
        if seconds >= MIN_ID_SECONDS and self.speaker_id is not None:
            name, confidence = self._identify(samples, sample_rate)
            if name:
                presence.speaker = name
                presence.speaker_confidence = confidence
                presence.identified = True

        presence.emotion = self.emotion.read(
            samples, sample_rate, speaker=presence.speaker, learn=learn)
        self.last = presence
        self._maybe_log(presence)
        return presence

    def _identify(self, samples: Any, sample_rate: int) -> tuple[str, float]:
        try:
            name, confidence = self.speaker_id.identify(samples, sample_rate)
        except Exception:
            return "", 0.0
        if not name or str(name).lower() in {"unknown", "", "none"}:
            return "", 0.0
        return str(name), float(confidence)

    def _maybe_log(self, presence: Presence) -> None:
        """Log a NEW person arriving, not every utterance they speak.

        A line per utterance would bury the log and, worse, would train the
        user to ignore exactly the line that matters — mum being recognised for
        the first time in a conversation.
        """
        if self.bus is None or not presence.is_known_person:
            return
        now = time.time()
        last = self._announced.get(presence.speaker, 0.0)
        if now - last < 600.0:
            return
        self._announced[presence.speaker] = now
        try:
            self.bus.log.emit(
                f"VOICE: that's {presence.speaker} speaking "
                f"({presence.speaker_confidence:.0%} match).")
        except Exception:
            pass

    # ── reporting ────────────────────────────────────────────────────────────

    def describe(self) -> str:
        known = []
        if self.speaker_id is not None:
            try:
                known = list(self.speaker_id.list_profiles())
            except Exception:
                known = []
        lines = []
        if known:
            lines.append("Voices I can recognise: " + ", ".join(known) + ".")
        else:
            lines.append("I haven't been introduced to anyone's voice yet - "
                         "ask me to remember a voice and I'll record a short "
                         "clip with their permission.")
        lines.append(self.emotion.describe())
        if self.last is not None:
            lines.append("Last utterance: " + self.last.describe())
        return " ".join(lines)


__all__ = ["DEFAULT_SPEAKER", "MIN_ID_SECONDS", "Presence", "VoicePresence"]
