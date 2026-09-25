"""
AudioRecovery — the deterministic wake-from-standby sequence.

Why this exists
---------------
Waking ORION used to be five lines inside the spoken-command handler:

    self.quiet_mode = False
    self.standby_mode = False
    prior_mic = getattr(self, "_microphone_before_standby", None)
    if was_standby and prior_mic is not None:
        ... re-enable the microphone ...

Three separate defects lived in that:

1. ``_drop_live_output`` — set True on the way INTO standby so a mid-flight
   reply goes quiet — was never cleared on the way out.  It is only cleared by
   ``_mark_turn`` (text turns) and ``turn_complete``.  A *spoken* turn on the
   live channel goes through neither on its way in, so every audio chunk of the
   first reply after waking was discarded.  ORION heard you, answered, and you
   heard nothing.  That is "he cannot speak after standby".

2. ``prior_mic is not None`` — ``_microphone_before_standby`` is only set by
   the spoken standby path.  Enter standby any other way (the orb, a
   ``gui_command``) and the attribute does not exist, ``getattr`` returns None,
   and the microphone is NEVER re-enabled.  That is "he appears alive but does
   nothing".

3. Nothing verified, reopened or even mentioned the audio DEVICES.  A hold left
   set on the renderer, a dead output stream, or a microphone whose stream had
   been closed all survived the wake untouched.

The fix is one idempotent sequence that owns every flag, runs in a fixed order,
proves each device rather than assuming it, and reports honestly when it
cannot.  Running it twice is a no-op — it never stacks a second listener,
stream, worker or task.

The sequence (the brief's ladder, in order)
-------------------------------------------
    cancel standby audio state
    restore + VERIFY the output device   (XRocker preferred)
    restore + VERIFY the input device    (Fifine preferred)
    reinitialise streams if necessary
    reset microphone state
    reset playback state
    reset interruption state
    reset speaking/listening flags
    restart listening
    confirm readiness

Preferred-device reality
------------------------
XRocker and Fifine are the user's stated hardware.  Neither is guaranteed to be
attached — on the machine this was developed against, NEITHER was present.  So
"preferred device missing" is an ordinary, expected outcome: the sequence falls
back to a verified working device and says so plainly.  It never claims to have
restored hardware that is not there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from . import audio_devices
from .audio_state import SpeechState


@dataclass
class RecoveryReport:
    """What the wake sequence actually did.  Spoken, logged and asserted on."""

    ok: bool = False
    steps: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    output_name: str = ""
    input_name: str = ""
    output_preferred: bool = False   # did we get the XRocker specifically?
    input_preferred: bool = False    # did we get the Fifine specifically?
    duration_ms: float = 0.0
    skipped: bool = False            # already awake; nothing to do

    def note(self, step: str) -> None:
        self.steps.append(step)

    def problem(self, detail: str) -> None:
        self.problems.append(detail)

    def summary(self) -> str:
        if self.skipped:
            return "Already awake — wake sequence skipped."
        head = "Audio restored" if self.ok else "Audio NOT fully restored"
        parts = [f"{head}: output {self.output_name or 'none'}, "
                 f"input {self.input_name or 'none'}"]
        if self.problems:
            parts.append("Problems: " + "; ".join(self.problems))
        return ". ".join(parts) + f" ({self.duration_ms:.0f} ms)"

    def spoken(self) -> str:
        """What ORION says when he comes back.  Honest, and short."""
        if not self.ok:
            return ("I'm awake, but my audio is not fully restored. "
                    + (self.problems[0] if self.problems else "Check the log."))
        missing = []
        if not self.output_preferred:
            missing.append("the XRocker")
        if not self.input_preferred:
            missing.append("the Fifine")
        if missing:
            return f"Back with you. {' and '.join(missing)} isn't connected, so I'm using the system default."
        return "Back with you."


class AudioRecovery:
    """Owns wake-from-standby.  One instance per live worker.

    Every method is safe to call from any thread and safe to call repeatedly:
    the lock plus the ``_running`` guard make a concurrent or duplicated wake
    collapse into a single sequence rather than two racing ones.
    """

    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self._lock = RLock()
        self._running = False
        self.last_report: RecoveryReport | None = None

    # ── helpers onto the worker's parts (all optional, all guarded) ───────────

    @property
    def _bus(self) -> Any:
        return getattr(self.worker, "bus", None)

    @property
    def _speech(self) -> Any:
        return getattr(self.worker, "speech", None)

    @property
    def _mic(self) -> Any:
        return getattr(self.worker, "mic", None)

    @property
    def _machine(self) -> Any:
        speech = self._speech
        return getattr(speech, "machine", None) if speech is not None else None

    def _log(self, message: str) -> None:
        bus = self._bus
        if bus is not None:
            try:
                bus.log.emit(message)
            except RuntimeError:
                pass

    # ── entering standby ──────────────────────────────────────────────────────

    def enter_standby(self, reason: str = "user request") -> None:
        """The dormant half.  Recorded here so wake knows exactly what to undo.

        Deliberately stores the microphone's prior enablement UNCONDITIONALLY
        (never via a getattr that can yield None), which is the defect that let
        a non-spoken standby leave the mic permanently off.
        """
        with self._lock:
            worker = self.worker
            worker._microphone_before_standby = bool(
                getattr(worker, "microphone_enabled", True))
            worker.standby_mode = True
            worker.quiet_mode = True
            worker.microphone_enabled = False
            mic = self._mic
            if mic is not None:
                try:
                    mic.set_enabled(False)
                    mic._drain()
                except Exception as exc:
                    self._log(f"AUDIO: standby could not quiet the microphone - {exc}")
            speech = self._speech
            if speech is not None:
                try:
                    speech.interrupt_all()
                except Exception as exc:
                    self._log(f"AUDIO: standby could not stop playback - {exc}")
            machine = self._machine
            if machine is not None and hasattr(machine, "enter_standby"):
                machine.enter_standby(reason)
            clear_turn = getattr(worker, "_clear_turn", None)
            if callable(clear_turn):
                clear_turn()
            worker.tool_busy = False
            # Mute any reply still streaming from the live model.  Wake() is
            # now the thing that guarantees this gets cleared again.
            worker._drop_live_output = bool(getattr(worker, "connected", False))

    # ── the wake sequence ─────────────────────────────────────────────────────

    def wake(self, reason: str = "user request", *, force: bool = False) -> RecoveryReport:
        """Run the full deterministic recovery.  Idempotent.

        Returns a report rather than raising: a wake that half-worked must
        still leave ORION usable and must say what is wrong.
        """
        report = RecoveryReport()
        started = time.monotonic()
        with self._lock:
            if self._running:
                # A second wake arriving mid-sequence is not an error — it is
                # the user saying "wake up" twice.  Collapse it.
                report.skipped = True
                report.note("a wake sequence was already running")
                return report
            worker = self.worker
            was_standby = bool(getattr(worker, "standby_mode", False)
                               or getattr(worker, "_standby", False))
            if not was_standby and not force:
                report.skipped = True
                report.ok = True
                report.note("not in standby")
                return report
            self._running = True
        try:
            return self._run_sequence(report, reason, started)
        finally:
            with self._lock:
                self._running = False
            self.last_report = report

    def _run_sequence(self, report: RecoveryReport, reason: str,
                      started: float) -> RecoveryReport:
        worker = self.worker
        machine = self._machine
        speech = self._speech
        mic = self._mic

        # 1 ── cancel standby audio state ─────────────────────────────────────
        if machine is not None and hasattr(machine, "begin_recovery"):
            machine.begin_recovery(reason)
        worker.standby_mode = False
        worker.quiet_mode = False
        worker._standby = False
        report.note("standby state cancelled")

        # 2 ── restore + verify the OUTPUT device (XRocker preferred) ─────────
        out = audio_devices.verify("output")
        report.output_name = out.name
        report.output_preferred = bool(
            out.index is not None
            and out.index == audio_devices.preferred_index("output"))
        report.note(f"output: {out.detail}")
        if not out.ok:
            report.problem(f"no output device could be opened ({out.detail})")
        elif not report.output_preferred:
            report.problem(out.detail)
        self._log(f"AUDIO: wake — {out.detail}")

        # 3 ── move the live renderer onto it and reinitialise the stream ─────
        if speech is not None and out.ok:
            playback = getattr(speech, "playback", None)
            if playback is not None:
                try:
                    # request_output_device is applied ON the renderer thread
                    # between writes — never touch the stream cross-thread.
                    playback.request_output_device(out.index)
                    audio_devices.note_live_device("output", out.index)
                    report.note("renderer asked to reopen on the verified device")
                except Exception as exc:
                    report.problem(f"renderer reopen failed: {exc}")

        # 4 ── restore + verify the INPUT device (Fifine preferred) ───────────
        inp = audio_devices.verify("input")
        report.input_name = inp.name
        report.input_preferred = bool(
            inp.index is not None
            and inp.index == audio_devices.preferred_index("input"))
        report.note(f"input: {inp.detail}")
        if not inp.ok:
            report.problem(f"no input device could be opened ({inp.detail})")
        elif not report.input_preferred:
            report.problem(inp.detail)
        self._log(f"AUDIO: wake — {inp.detail}")

        # 5 ── reset PLAYBACK state (a hold left set = permanent silence) ─────
        if speech is not None:
            playback = getattr(speech, "playback", None)
            if playback is not None:
                try:
                    if playback.held():
                        playback._hold.clear()
                        report.note("cleared a playback hold left over from standby")
                    playback._held_tail = b""
                except Exception as exc:
                    report.problem(f"playback reset failed: {exc}")
            tts = getattr(speech, "tts", None)
            if tts is not None:
                try:
                    if tts.held():
                        tts.resume_speech()
                        report.note("released a held local-voice utterance")
                except Exception as exc:
                    report.problem(f"local voice reset failed: {exc}")

        # 6 ── reset INTERRUPTION / turn state ────────────────────────────────
        # THE standby bug: set True on the way in, cleared only by _mark_turn or
        # turn_complete — neither of which a spoken turn passes through on its
        # way in — so the entire first reply after waking was thrown away.
        worker._drop_live_output = False
        worker.tool_busy = False
        report.note("live-output mute cleared (first reply after wake is audible)")
        if getattr(worker, "paused", False):
            worker.paused = False
            try:
                worker.bus.paused.emit(False)
            except Exception:
                pass
            report.note("pause released")

        # 7 ── reset MICROPHONE state and restart capture ─────────────────────
        prior = getattr(worker, "_microphone_before_standby", None)
        # Default to ON.  The old code required this to be a real bool and did
        # nothing when it was None, which is exactly how the mic stayed dead.
        worker.microphone_enabled = True if prior is None else bool(prior)
        worker._microphone_before_standby = None
        if mic is not None:
            try:
                mic.set_enabled(worker.microphone_enabled)
                # Drop anything captured while dormant so the first thing ORION
                # hears after waking is the user, not a stale buffer.
                mic._drain()
                report.note(f"microphone re-enabled ({worker.microphone_enabled})")
                if inp.ok and inp.index is not None:
                    restarted = self._restart_input(mic, inp.index, report)
                    if restarted:
                        audio_devices.note_live_device("input", inp.index)
            except Exception as exc:
                report.problem(f"microphone restart failed: {exc}")
        else:
            report.note("microphone engine not started yet — capture starts with it")

        # 8 ── reset speaking/listening flags ─────────────────────────────────
        if machine is not None:
            try:
                machine.user_speech_stopped()
                machine.processing_finished()
            except Exception:
                pass

        # 9 ── settle the machine and report readiness ────────────────────────
        report.ok = out.ok and inp.ok
        report.duration_ms = (time.monotonic() - started) * 1000.0
        if machine is not None:
            if report.ok and hasattr(machine, "recovery_complete"):
                machine.recovery_complete(reason)
            elif not report.ok and hasattr(machine, "fault"):
                machine.fault("; ".join(report.problems) or "audio unavailable")
        try:
            worker.bus.state.emit("LISTENING" if report.ok else "ERROR")
            worker.bus.gui_command.emit({"action": "wake", "target": ""})
        except Exception:
            pass
        self._log(f"AUDIO: wake sequence — {report.summary()}")
        return report

    def _restart_input(self, mic: Any, index: int, report: RecoveryReport) -> bool:
        """Reopen the capture stream on *index*, without stacking a second one.

        MicrophoneEngine._restart_stream closes the existing stream before
        opening the new one, which is what makes repeated wakes safe.
        """
        try:
            # MicrophoneEngine tracks the live device through audio_devices'
            # _live table (note_live_device) and holds the stream on _stream.
            current = audio_devices.resolve_effective("input")
            if current == index and getattr(mic, "_stream", None) is not None:
                report.note("capture stream already on the verified device")
                return True
            restart = getattr(mic, "_restart_stream", None)
            if callable(restart):
                restart(index)
                report.note(f"capture stream reopened on device {index}")
                return True
            switch = getattr(mic, "switch_to_input", None)
            if callable(switch):
                switch(index, persist=False)
                report.note(f"capture switched to device {index}")
                return True
        except Exception as exc:
            report.problem(f"capture reopen failed: {exc}")
        return False

    # ── health ────────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Current audio-path health for the unified health model."""
        machine = self._machine
        state = getattr(machine, "state", None)
        report = self.last_report
        if machine is not None and machine.faulted():
            status = "OFFLINE"
        elif machine is not None and machine.recovering():
            status = "RECOVERING"
        elif machine is not None and machine.in_standby():
            status = "DEGRADED"
        elif report is not None and not report.ok and not report.skipped:
            status = "DEGRADED"
        else:
            status = "ONLINE"
        return {
            "status": status,
            "state": getattr(state, "value", str(state)),
            "output": audio_devices.device_name("output"),
            "input": audio_devices.device_name("input"),
            "xrocker_present": audio_devices.preferred_index("output") is not None,
            "fifine_present": audio_devices.preferred_index("input") is not None,
            "last_recovery": report.summary() if report is not None else "none yet",
            "illegal": (machine.illegal_combination()
                        if machine is not None and hasattr(machine, "illegal_combination")
                        else ""),
        }


__all__ = ["AudioRecovery", "RecoveryReport", "SpeechState"]
