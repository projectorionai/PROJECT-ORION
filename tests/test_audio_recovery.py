"""
Wake-from-standby — the deterministic recovery sequence.

Each test here corresponds to a way ORION was observed to come back from
standby "alive but doing nothing".  They are written against the real
AudioRecovery against fake worker parts, because the defects were all in the
ORDER and COMPLETENESS of the flag handling, not in the hardware.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion_core import audio_devices
from orion_core.audio_recovery import AudioRecovery, RecoveryReport
from orion_core.audio_state import AudioStateMachine, SpeechState


class FakeSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args if len(args) != 1 else args[0])

    def connect(self, *_a, **_k):
        pass


class FakeBus:
    def __init__(self):
        self.log = FakeSignal()
        self.state = FakeSignal()
        self.speaking = FakeSignal()
        self.gui_command = FakeSignal()
        self.paused = FakeSignal()
        self.mic_enabled = FakeSignal()

    def messages(self):
        return [str(m) for m in self.log.emitted]


class FakePlayback:
    def __init__(self):
        self.held_flag = False
        self._held_tail = b""
        self.requested_device = "unset"
        self.cleared = False

        class _Hold:
            def __init__(self, outer):
                self.outer = outer

            def clear(self):
                self.outer.held_flag = False

            def is_set(self):
                return self.outer.held_flag

        self._hold = _Hold(self)

    def held(self):
        return self.held_flag

    def request_output_device(self, index):
        self.requested_device = index

    def is_active(self):
        return False

    def clear(self):
        self.cleared = True


class FakeTts:
    def __init__(self):
        self.held_flag = False
        self.resumed = False

    def held(self):
        return self.held_flag

    def resume_speech(self):
        self.resumed = True
        self.held_flag = False
        return True

    def is_busy(self):
        return False

    def interrupt(self):
        pass


class FakeSpeech:
    def __init__(self, bus):
        self.playback = FakePlayback()
        self.tts = FakeTts()
        self.machine = AudioStateMachine(bus)
        self.interrupted = False

    def interrupt_all(self):
        self.interrupted = True
        self.machine.interrupted()
        return True

    def output_active(self):
        return self.machine.is_active()


class FakeMic:
    def __init__(self):
        self.enabled = None
        self.drained = 0
        self.restarts = []
        self._stream = object()

    def set_enabled(self, value):
        self.enabled = bool(value)

    def _drain(self):
        self.drained += 1

    def _restart_stream(self, index):
        self.restarts.append(index)


def make_worker(*, connected=True, with_mic=True):
    bus = FakeBus()
    worker = SimpleNamespace(
        bus=bus,
        speech=FakeSpeech(bus),
        mic=FakeMic() if with_mic else None,
        microphone_enabled=True,
        standby_mode=False,
        quiet_mode=False,
        _standby=False,
        connected=connected,
        tool_busy=False,
        paused=False,
        _drop_live_output=False,
        _microphone_before_standby=None,
        _clear_turn=lambda: None,
    )
    worker.recovery = AudioRecovery(worker)
    return worker


# ── the confirmed root causes ─────────────────────────────────────────────────

def test_standby_mutes_the_live_reply_and_wake_unmutes_it():
    """THE bug: _drop_live_output was set on the way in and never cleared on
    the way out, so the first reply after waking was discarded entirely."""
    worker = make_worker(connected=True)
    worker.recovery.enter_standby()
    assert worker._drop_live_output is True, "standby should mute a mid-flight reply"

    worker.recovery.wake()
    assert worker._drop_live_output is False, (
        "wake must clear the live-output mute or the first reply is thrown away")


def test_wake_re_enables_the_microphone_even_when_standby_came_from_elsewhere():
    """The orb / gui_command standby path never set _microphone_before_standby,
    so `prior_mic is not None` was False and the mic stayed dead forever."""
    worker = make_worker()
    # Simulate a standby that did NOT go through enter_standby().
    worker.standby_mode = True
    worker.microphone_enabled = False
    worker.mic.set_enabled(False)
    worker._microphone_before_standby = None

    worker.recovery.wake()
    assert worker.microphone_enabled is True
    assert worker.mic.enabled is True


def test_wake_clears_a_playback_hold_left_over_from_standby():
    """A hold left set means the renderer never writes again — permanent
    silence that looks exactly like 'he cannot speak'."""
    worker = make_worker()
    worker.recovery.enter_standby()
    worker.speech.playback.held_flag = True
    worker.speech.playback._held_tail = b"leftover"

    worker.recovery.wake()
    assert worker.speech.playback.held() is False
    assert worker.speech.playback._held_tail == b""


def test_wake_releases_a_held_local_voice_utterance():
    worker = make_worker()
    worker.recovery.enter_standby()
    worker.speech.tts.held_flag = True
    worker.recovery.wake()
    assert worker.speech.tts.resumed is True


def test_wake_clears_pause_and_tool_busy():
    worker = make_worker()
    worker.recovery.enter_standby()
    worker.paused = True
    worker.tool_busy = True
    worker.recovery.wake()
    assert worker.paused is False
    assert worker.tool_busy is False


# ── idempotence (the brief requires this explicitly) ──────────────────────────

def test_running_the_wake_sequence_twice_changes_nothing_the_second_time():
    worker = make_worker()
    worker.recovery.enter_standby()
    first = worker.recovery.wake()
    second = worker.recovery.wake()

    assert first.skipped is False
    assert second.skipped is True, "a second wake when already awake is a no-op"
    # No duplicate stream restarts, no duplicate drains beyond the first wake.
    assert len(worker.mic.restarts) <= 1


def test_repeated_wakes_never_stack_microphone_restarts():
    worker = make_worker()
    for _ in range(5):
        worker.recovery.enter_standby()
        worker.recovery.wake()
    # One restart per genuine wake at most — never a growing pile of streams.
    assert len(worker.mic.restarts) <= 5


def test_a_wake_arriving_mid_sequence_is_collapsed():
    """Two 'wake up's in quick succession must not run two sequences."""
    worker = make_worker()
    worker.recovery.enter_standby()
    worker.recovery._running = True          # pretend one is in flight
    report = worker.recovery.wake()
    assert report.skipped is True
    assert "already running" in " ".join(report.steps)


def test_waking_when_not_in_standby_is_a_no_op():
    worker = make_worker()
    report = worker.recovery.wake()
    assert report.skipped is True
    assert report.ok is True


# ── device verification, honestly reported ────────────────────────────────────

def test_wake_verifies_both_devices_and_records_their_names():
    worker = make_worker()
    worker.recovery.enter_standby()
    report = worker.recovery.wake()
    assert report.output_name, "an output device must be named"
    assert report.input_name, "an input device must be named"
    assert any("output:" in s for s in report.steps)
    assert any("input:" in s for s in report.steps)


def test_a_missing_preferred_device_is_reported_not_hidden(monkeypatch):
    """XRocker/Fifine absent is an ordinary outcome — ORION must SAY so
    rather than silently using something else and claiming success."""
    monkeypatch.setattr(audio_devices, "preferred_index", lambda kind: None)
    worker = make_worker()
    worker.recovery.enter_standby()
    report = worker.recovery.wake()

    assert report.output_preferred is False
    assert report.input_preferred is False
    assert report.problems, "an absent preferred device must be surfaced"
    spoken = report.spoken()
    assert "XRocker" in spoken or "Fifine" in spoken


def test_when_the_preferred_devices_are_present_they_are_selected(monkeypatch):
    check = audio_devices.DeviceCheck(
        "output", "xrocker", True, True, 42, "[42] XRocker", "output verified on XRocker")
    check_in = audio_devices.DeviceCheck(
        "input", "fifine", True, True, 7, "[7] fifine Microphone", "input verified on fifine")
    monkeypatch.setattr(audio_devices, "verify",
                        lambda kind, spec="", **k: check if kind == "output" else check_in)
    monkeypatch.setattr(audio_devices, "preferred_index",
                        lambda kind: 42 if kind == "output" else 7)
    monkeypatch.setattr(audio_devices, "note_live_device", lambda *a, **k: None)
    monkeypatch.setattr(audio_devices, "resolve_effective", lambda kind: None)

    worker = make_worker()
    worker.recovery.enter_standby()
    report = worker.recovery.wake()

    assert report.output_preferred is True
    assert report.input_preferred is True
    assert report.ok is True
    assert report.spoken() == "Back with you."
    assert worker.speech.playback.requested_device == 42
    assert worker.mic.restarts == [7]


def test_a_total_output_failure_is_a_fault_not_a_silent_success(monkeypatch):
    dead = audio_devices.DeviceCheck(
        "output", "xrocker", False, False, None, "none",
        "no output device could be opened at all")
    live_in = audio_devices.DeviceCheck(
        "input", "fifine", True, True, 3, "[3] mic", "input verified")
    monkeypatch.setattr(audio_devices, "verify",
                        lambda kind, spec="", **k: dead if kind == "output" else live_in)
    monkeypatch.setattr(audio_devices, "preferred_index", lambda kind: None)

    worker = make_worker()
    worker.recovery.enter_standby()
    report = worker.recovery.wake()

    assert report.ok is False
    assert worker.speech.machine.faulted() is True
    assert "not fully restored" in report.spoken()


def test_recovery_never_raises_when_the_microphone_engine_is_absent():
    worker = make_worker(with_mic=False)
    worker.recovery.enter_standby()
    report = worker.recovery.wake()
    assert isinstance(report, RecoveryReport)
    assert any("not started yet" in s for s in report.steps)


# ── state machine integration ─────────────────────────────────────────────────

def test_the_machine_passes_standby_then_recovering_then_normal():
    worker = make_worker()
    machine = worker.speech.machine

    worker.recovery.enter_standby()
    assert machine.in_standby() is True
    assert machine.state is SpeechState.STANDBY

    worker.recovery.wake()
    assert machine.in_standby() is False
    assert machine.recovering() is False
    assert machine.state in (SpeechState.IDLE, SpeechState.LISTENING)


def test_health_reports_the_devices_and_whether_the_hardware_is_present():
    worker = make_worker()
    health = worker.recovery.health()
    assert health["status"] in {"ONLINE", "DEGRADED", "RECOVERING", "OFFLINE"}
    assert "xrocker_present" in health
    assert "fifine_present" in health
    assert health["illegal"] == ""


def test_standby_leaves_no_illegal_flag_combination():
    worker = make_worker()
    worker.recovery.enter_standby()
    assert worker.speech.machine.illegal_combination() == ""


# ── the report itself ─────────────────────────────────────────────────────────

def test_report_summary_is_a_single_readable_line():
    worker = make_worker()
    worker.recovery.enter_standby()
    report = worker.recovery.wake()
    summary = report.summary()
    assert "\n" not in summary
    assert "ms)" in summary
