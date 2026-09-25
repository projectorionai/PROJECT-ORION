"""
VoiceInterruptManager (Mark X.5) — the always-on interruption listener.

Why this exists
---------------
The Mark VIII half-duplex rule (speak-then-listen) made ORION reliable but
DEAF while speaking: the audio gate swallowed every captured chunk, so a
spoken "ORION stop" could never be heard, and the old pause path destroyed
the playback queue — resuming meant regenerating the whole response.

Mark X.5 fixes both halves:

    • LISTENING WHILE SPEAKING — the AudioGateThread feeds every chunk
      captured during ORION's own speech into ``feed()``.  Matching runs on a
      dedicated Vosk recogniser CONSTRAINED to the command-phrase grammar
      (sharing the already-loaded model, so no extra memory).  A constrained
      grammar makes false positives against ORION's own acoustic bleed
      vanishingly rare while keeping the listener extremely cheap — it is a
      command detector, not a transcriber.

    • POSITION-PRESERVING CONTROL — matched pause-family phrases route to the
      worker's hold path (playback queue and utterance position preserved);
      resume-family phrases lift the hold so speech continues from the exact
      interruption point.  No context loss, no response regeneration.

Fully offline: Vosk runs locally; when Vosk is unavailable the manager
degrades gracefully (``feed`` returns nothing, the GUI pause button and typed
commands still work) and says so once in the log.

Threading: ``feed()`` is called on the AudioGateThread — it must never touch
Qt, never block on the event loop, and never raise.  ``classify()`` is pure
and safe from any thread.
"""

from __future__ import annotations

import json
import time
from threading import RLock
from typing import Any

from .bus import OrionBus
from .constants import (
    INTERRUPT_COOLDOWN_SECONDS,
    INTERRUPT_PAUSE_PHRASES,
    INTERRUPT_RESUME_PHRASES,
)


class VoiceInterruptManager:
    """Grammar-constrained spoken-command listener that outlives speech."""

    ACTION_PAUSE = "pause"
    ACTION_RESUME = "resume"

    def __init__(
        self,
        bus: OrionBus,
        recogniser: Any,
        telemetry: Any | None = None,
    ) -> None:
        # `recogniser` is the shared LocalSpeechRecogniser; only
        # ``create_command_recogniser`` and ``available`` are used.
        self.bus = bus
        self.recogniser = recogniser
        self.telemetry = telemetry
        self._lock = RLock()
        self._command_rec: Any = None
        self._last_trigger = 0.0
        self._unavailable_logged = False
        self._phrases: tuple[str, ...] = (
            *INTERRUPT_PAUSE_PHRASES, *INTERRUPT_RESUME_PHRASES,
        )
        if self.telemetry is not None:
            self.telemetry.health.register("voice_interrupt")

    # ── classification (pure, thread-safe) ────────────────────────────────────

    @classmethod
    def classify(cls, text: str) -> str:
        """Map a transcript fragment to an action: 'pause', 'resume' or ''."""
        lowered = " " + " ".join(str(text or "").lower().split()) + " "
        for phrase in INTERRUPT_RESUME_PHRASES:
            if f" {phrase} " in lowered or lowered.strip() == phrase:
                return cls.ACTION_RESUME
        for phrase in INTERRUPT_PAUSE_PHRASES:
            if f" {phrase} " in lowered or lowered.strip() == phrase:
                return cls.ACTION_PAUSE
        return ""

    # ── streaming feed (AudioGateThread) ──────────────────────────────────────

    def feed(self, chunk: bytes) -> str:
        """
        Feed one 16 kHz PCM chunk captured while ORION is speaking.  Returns
        the matched action ('pause' / 'resume') or '' — never raises.
        """
        if not chunk:
            return ""
        rec = self._ensure_recogniser()
        if rec is None:
            return ""
        try:
            with self._lock:
                # ONLY act on a FINALISED utterance — never a partial.  A
                # grammar-constrained recogniser will fit fragments of ORION's
                # OWN voice to a command phrase, so partial matching caused him
                # to interrupt himself mid-sentence (spurious pauses + trailing
                # off).  A real spoken "ORION stop" finalises within a beat.
                if not rec.AcceptWaveform(chunk):
                    return ""
                payload = json.loads(rec.Result() or "{}")
                text = str(payload.get("text") or "")
        except Exception:
            return ""
        # Require the whole finalised phrase, not a loose substring, so an
        # incidental word in ORION's speech cannot trip it.
        action = self.classify(text)
        if not action or not self._is_exact_command(text):
            return ""
        now = time.monotonic()
        if (now - self._last_trigger) < INTERRUPT_COOLDOWN_SECONDS:
            return ""
        self._last_trigger = now
        try:
            with self._lock:
                rec.Reset()
        except Exception:
            pass
        if self.telemetry is not None:
            self.telemetry.metrics.incr(f"voice_interrupt.{action}")
        return action

    @staticmethod
    def _is_exact_command(text: str) -> bool:
        """True only when the finalised utterance IS a command phrase (a couple
        of stray words tolerated), not merely contains a word that resembles one."""
        words = str(text or "").lower().split()
        return 1 <= len(words) <= 4 and "orion" in words

    def note_trigger(self) -> None:
        """Record an externally-honoured command (transcript path) so the
        streaming listener's cooldown also covers it."""
        self._last_trigger = time.monotonic()

    # ── plumbing ──────────────────────────────────────────────────────────────

    def _ensure_recogniser(self) -> Any:
        """Lazily build the grammar recogniser once the Vosk model is loaded."""
        if self._command_rec is not None:
            return self._command_rec
        if self.recogniser is None or not getattr(self.recogniser, "available", False):
            return None
        rec = self.recogniser.create_command_recogniser(list(self._phrases))
        if rec is None:
            if not self._unavailable_logged:
                self._unavailable_logged = True
                self.bus.log.emit(
                    "VOICE: interruption listener degraded — no local recogniser; "
                    "the pause button and typed commands remain available."
                )
            return None
        self._command_rec = rec
        self.bus.log.emit(
            "VOICE: true interruption listener armed — "
            f"{len(self._phrases)} command phrase(s) live even while I speak."
        )
        if self.telemetry is not None:
            self.telemetry.health.beat("voice_interrupt", "OK", "grammar listener armed")
        return rec

    def describe(self) -> dict[str, Any]:
        return {
            "armed": self._command_rec is not None,
            "phrases": list(self._phrases),
            "cooldown_s": INTERRUPT_COOLDOWN_SECONDS,
        }
