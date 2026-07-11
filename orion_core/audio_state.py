"""
Audio state machine (Phase 1) — the single, event-driven source of truth for
"is ORION speaking?".

Mark VIII decided this by polling ``output_active()`` every 80 ms, which meant
the microphone gate and the HUD could disagree for up to 80 ms and the
turn-completion check raced a momentarily-empty queue (TOCTOU).  Mark IX
inverts control: the two things that actually *produce* audio — the native
PCM playback thread and the local TTS thread — push transitions here the
instant they start or stop.  Nothing polls; the gate and the HUD read the
same atomic flag the transitions set, so they can never desynchronise.

States
------
    IDLE         no source active; the microphone may listen.
    SPEAKING     at least one source (native playback or local TTS) active.
    INTERRUPTED  an explicit stop cut speech short; collapses to IDLE once the
                 sources confirm they have stopped.

Two independent "sources" are tracked (``native`` PCM and ``tts``).  The
machine is SPEAKING while either is active and returns to IDLE only when both
have reported stopped — including the deterministic playback tail, which the
playback thread owns, so the last syllable is never clipped by a state flip.

Thread-safety: transitions are computed under a lock; callbacks (bus signal,
worker state hook) fire *outside* the lock to avoid re-entrancy deadlocks.
"""

from __future__ import annotations

from enum import Enum
from threading import RLock
from typing import Callable, Optional

from .bus import OrionBus


class SpeechState(str, Enum):
    IDLE = "IDLE"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class AudioStateMachine:
    """Event-driven speaking-state authority shared by playback, TTS and gate."""

    def __init__(self, bus: OrionBus, telemetry: Optional[object] = None) -> None:
        self.bus = bus
        self._telemetry = telemetry
        self._lock = RLock()
        self._native_active = False
        self._tts_active = False
        self._state = SpeechState.IDLE
        # Fired (outside the lock) on every real transition.  The worker sets
        # this to marshal LISTENING/STANDBY onto its event loop.
        self.on_transition: Callable[[SpeechState, bool], None] | None = None

    # ── source events (called by the producing threads) ──────────────────────

    def native_started(self) -> None:
        self._set(native=True)

    def native_stopped(self) -> None:
        self._set(native=False)

    def tts_started(self) -> None:
        self._set(tts=True)

    def tts_stopped(self) -> None:
        self._set(tts=False)

    def interrupted(self) -> None:
        """Explicit stop — force sources down and pass through INTERRUPTED."""
        transition: tuple[SpeechState, bool] | None = None
        with self._lock:
            self._native_active = False
            self._tts_active = False
            if self._state is not SpeechState.IDLE:
                self._state = SpeechState.INTERRUPTED
                transition = (SpeechState.INTERRUPTED, False)
        if transition is not None:
            self._announce(*transition)
            # INTERRUPTED is momentary; settle straight to IDLE.
            self._set()  # recompute → IDLE, emits the idle transition

    # ── core transition ───────────────────────────────────────────────────────

    def _set(self, native: bool | None = None, tts: bool | None = None) -> None:
        transition: tuple[SpeechState, bool] | None = None
        with self._lock:
            if native is not None:
                self._native_active = native
            if tts is not None:
                self._tts_active = tts
            active = self._native_active or self._tts_active
            new_state = SpeechState.SPEAKING if active else SpeechState.IDLE
            if new_state is not self._state:
                self._state = new_state
                transition = (new_state, active)
        if transition is not None:
            self._announce(*transition)

    def _announce(self, state: SpeechState, active: bool) -> None:
        if self._telemetry is not None:
            try:
                self._telemetry.metrics.gauge("audio.speaking", 1.0 if active else 0.0)
                self._telemetry.metrics.gauge("audio.active_streams", float(self.active_streams()))
                self._telemetry.metrics.incr("audio.transitions")
            except Exception:
                pass
        try:
            self.bus.speaking.emit(active)
        except RuntimeError:
            pass  # Qt shutting down
        if self.on_transition is not None:
            try:
                self.on_transition(state, active)
            except Exception:
                pass

    # ── reads (atomic, lock-free-consistent) ──────────────────────────────────

    def is_active(self) -> bool:
        return self._native_active or self._tts_active

    @property
    def state(self) -> SpeechState:
        return self._state

    def active_streams(self) -> int:
        return int(self._native_active) + int(self._tts_active)

    def describe(self) -> dict[str, object]:
        return {
            "state": self._state.value,
            "native_active": self._native_active,
            "tts_active": self._tts_active,
            "active_streams": self.active_streams(),
        }
