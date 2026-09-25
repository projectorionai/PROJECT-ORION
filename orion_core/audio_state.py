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
    LISTENING    ORION is producing no audio AND the microphone gate has
                 detected the user's own voice (echo-guarded, so his own
                 trailing sound is never mistaken for this).  Read-only: it
                 never affects ``is_active()``/``bus.speaking`` — those stay
                 strictly "is ORION speaking?" for the half-duplex mic gate.
    INTERRUPTED  an explicit stop cut speech short; collapses to IDLE once the
                 sources confirm they have stopped.

Two independent "sources" are tracked (``native`` PCM and ``tts``).  The
machine is SPEAKING while either is active and returns to IDLE only when both
have reported stopped — including the deterministic playback tail, which the
playback thread owns, so the last syllable is never clipped by a state flip.

A third, read-only signal — ``user_speech_started``/``user_speech_stopped``,
pushed by the AudioGateThread's own VAD the instant it hears the user — drives
LISTENING.  Before this the machine only ever reflected ORION's own output, so
the Command Centre's "audio state" panel read IDLE even while the user was
actively talking to him and never seemed to "coordinate" with what was really
happening.

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
    LISTENING = "LISTENING"
    INTERRUPTED = "INTERRUPTED"
    # ── explicit lifecycle states (added for the standby repair) ──────────────
    # Before these, "in standby" and "recovering from standby" were scattered
    # booleans on the live worker (`standby_mode`, `_standby`,
    # `_microphone_before_standby`, `_drop_live_output`, `microphone_enabled`)
    # with no single place that knew whether the combination was even legal.
    # A wake that missed one of them left ORION in a state that looked alive
    # and did nothing — the reported bug.  Naming the states makes an illegal
    # combination impossible to hold and a stuck one obvious in the log.
    PROCESSING = "PROCESSING"   # a turn is in flight; not yet speaking
    STANDBY = "STANDBY"         # dormant: capture closed, output silenced
    RECOVERING = "RECOVERING"   # running the deterministic wake sequence
    ERROR = "ERROR"             # audio could not be restored; needs attention


#: Modes that OVERRIDE the source-derived state.  While one of these is set the
#: machine reports it regardless of what playback/TTS are doing, because the
#: lifecycle fact ("he is asleep", "he is coming back") outranks the momentary
#: fact ("a buffer is draining").
_OVERRIDE_MODES = (SpeechState.STANDBY, SpeechState.RECOVERING, SpeechState.ERROR)

#: Legal transitions between the override modes.  The source-derived states
#: (IDLE/SPEAKING/LISTENING/INTERRUPTED) are collapsed to NORMAL here — they
#: are driven by hardware events and are always legal among themselves.
_NORMAL = None
_LEGAL: dict[object, frozenset] = {
    _NORMAL: frozenset({SpeechState.STANDBY, SpeechState.RECOVERING, SpeechState.ERROR}),
    SpeechState.STANDBY: frozenset({SpeechState.RECOVERING, SpeechState.ERROR}),
    # RECOVERING may return to normal (woke up fine) or fail to ERROR.  It may
    # also re-enter itself: the wake sequence is idempotent by contract, so
    # calling it twice must be a no-op rather than a rejected transition.
    SpeechState.RECOVERING: frozenset({_NORMAL, SpeechState.RECOVERING,
                                       SpeechState.ERROR, SpeechState.STANDBY}),
    # ERROR is always escapable — a state you cannot leave is the bug, not the
    # protection against it.
    SpeechState.ERROR: frozenset({_NORMAL, SpeechState.RECOVERING, SpeechState.STANDBY}),
}


class AudioStateMachine:
    """Event-driven speaking-state authority shared by playback, TTS and gate."""

    def __init__(self, bus: OrionBus, telemetry: Optional[object] = None) -> None:
        self.bus = bus
        self._telemetry = telemetry
        self._lock = RLock()
        self._native_active = False
        self._tts_active = False
        self._listening_active = False
        self._processing = False
        self._mode: SpeechState | None = _NORMAL
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

    def user_speech_started(self) -> None:
        """The mic gate's VAD just heard the user (echo-guarded).  Read-only —
        never contributes to ``is_active()``/``bus.speaking``, only to the
        displayed ``state`` (LISTENING), so the half-duplex mic gate's
        contract is untouched."""
        self._set(listening=True)

    def user_speech_stopped(self) -> None:
        self._set(listening=False)

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

    # ── lifecycle modes (standby / recovery / fault) ──────────────────────────

    def _enter_mode(self, mode: SpeechState | None, reason: str = "") -> bool:
        """Move to an override mode (or back to normal).  Returns True if the
        transition was legal and taken.

        Illegal transitions are refused and LOGGED rather than silently
        applied — a machine that quietly accepts anything is not a state
        machine, and the whole point here is that a bad combination cannot be
        reached in the first place.
        """
        transition: tuple[SpeechState, bool] | None = None
        with self._lock:
            current = self._mode
            if current is mode and mode is not SpeechState.RECOVERING:
                return True                      # already there: idempotent
            if mode not in _LEGAL.get(current, frozenset()):
                try:
                    self.bus.log.emit(
                        f"AUDIO-STATE: refused {self._name(current)} → "
                        f"{self._name(mode)}"
                        f"{(' (' + reason + ')') if reason else ''}.")
                except RuntimeError:
                    pass
                return False
            self._mode = mode
            if mode is SpeechState.STANDBY:
                # Dormant means dormant: no source can be considered active.
                self._native_active = False
                self._tts_active = False
                self._listening_active = False
                self._processing = False
            new_state = self._compute_locked()
            if new_state is not self._state:
                self._state = new_state
                transition = (new_state, self._native_active or self._tts_active)
        try:
            self.bus.log.emit(
                f"AUDIO-STATE: {self._name(mode)}"
                f"{(' — ' + reason) if reason else ''}.")
        except RuntimeError:
            pass
        if transition is not None:
            self._announce(*transition)
        return True

    @staticmethod
    def _name(mode: SpeechState | None) -> str:
        return "NORMAL" if mode is _NORMAL else mode.value

    def enter_standby(self, reason: str = "") -> bool:
        """ORION goes dormant: capture closed, output silenced."""
        return self._enter_mode(SpeechState.STANDBY, reason)

    def begin_recovery(self, reason: str = "") -> bool:
        """The deterministic wake sequence has started."""
        return self._enter_mode(SpeechState.RECOVERING, reason)

    def recovery_complete(self, reason: str = "") -> bool:
        """Audio is verified working again; back to normal source-driven state."""
        return self._enter_mode(_NORMAL, reason)

    def fault(self, reason: str = "") -> bool:
        """Audio could not be restored.  Visible, not swallowed."""
        return self._enter_mode(SpeechState.ERROR, reason)

    def processing_started(self) -> None:
        """A turn is in flight — reasoning or a tool run, not yet speech."""
        self._set(processing=True)

    def processing_finished(self) -> None:
        self._set(processing=False)

    @property
    def mode(self) -> SpeechState | None:
        return self._mode

    def in_standby(self) -> bool:
        return self._mode is SpeechState.STANDBY

    def recovering(self) -> bool:
        return self._mode is SpeechState.RECOVERING

    def faulted(self) -> bool:
        return self._mode is SpeechState.ERROR

    # ── core transition ───────────────────────────────────────────────────────

    def _compute_locked(self) -> SpeechState:
        """The state implied by the current flags.  Caller holds the lock."""
        if self._mode in _OVERRIDE_MODES:
            return self._mode
        if self._native_active or self._tts_active:
            return SpeechState.SPEAKING
        if self._processing:
            return SpeechState.PROCESSING
        if self._listening_active:
            return SpeechState.LISTENING
        return SpeechState.IDLE

    def _set(
        self,
        native: bool | None = None,
        tts: bool | None = None,
        listening: bool | None = None,
        processing: bool | None = None,
    ) -> None:
        transition: tuple[SpeechState, bool] | None = None
        with self._lock:
            if native is not None:
                self._native_active = native
            if tts is not None:
                self._tts_active = tts
            if listening is not None:
                self._listening_active = listening
            if processing is not None:
                self._processing = processing
            active = self._native_active or self._tts_active
            new_state = self._compute_locked()
            if new_state is not self._state:
                self._state = new_state
                # `active` stays strictly "is ORION's own audio playing" —
                # bus.speaking and the half-duplex mic gate both key off this
                # exact boolean, so LISTENING must never flip it true.
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
            "mode": self._name(self._mode),
            "native_active": self._native_active,
            "tts_active": self._tts_active,
            "listening_active": self._listening_active,
            "processing": self._processing,
            "active_streams": self.active_streams(),
        }

    def illegal_combination(self) -> str:
        """Name an impossible flag combination, or '' when the state is sane.

        The brief asked specifically that states like "not speaking, not
        listening, microphone open, playback active" be prevented.  The flags
        that can produce those live in three different objects, so this is the
        one place that can see all of them at once — used by the health model
        and asserted in tests.
        """
        if self._mode is SpeechState.STANDBY and (self._native_active or self._tts_active):
            return "STANDBY while a source is still producing audio"
        if self._mode is SpeechState.STANDBY and self._listening_active:
            return "STANDBY while the gate reports the user speaking"
        if self._state is SpeechState.IDLE and (self._native_active or self._tts_active):
            return "IDLE while a source is active"
        if self._state is SpeechState.SPEAKING and not (self._native_active or self._tts_active):
            return "SPEAKING with no active source"
        return ""
