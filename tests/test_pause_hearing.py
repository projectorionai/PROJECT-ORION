"""
Hearing "ORION resume" while paused.

  "when I speak to ORION and tell him to pause, when I say ORION resume it
   doesn't actually resume I don't think ORION can hear me at all during this
   time - can we keep his hearing active just not his interactions and keep it
   separate?"

The command-grammar listener that catches "ORION stop" / "ORION resume" only
ran while ORION was SPEAKING. Once paused and silent it stopped, and the live
interaction channel is gated off while paused — so the one word meant to bring
him back was the one word he could not hear.

The fix runs that dedicated listener while paused as well as while speaking.
It is the exact separation asked for: HEARING (the command grammar) stays live;
INTERACTION (the general channel) stays held. Hearing a command never means
acting on general speech.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import audio, live_worker  # noqa: E402


def test_the_gate_accepts_a_command_listen_check():
    params = inspect.signature(audio.AudioGateThread.__init__).parameters
    assert "command_listen_check" in params


def test_the_command_listener_runs_when_speaking_or_command_listening():
    """The condition must be an OR — speaking (interrupt him) plus the paused
    check (bring him back)."""
    source = inspect.getsource(audio.AudioGateThread.run)
    assert "command_listening = speaking or (" in source
    assert "self.command_listen_check" in source


def test_the_command_feed_is_no_longer_only_under_if_speaking():
    """The regression guard: if the interrupt feed goes back inside `if
    speaking:`, pause goes deaf again."""
    source = inspect.getsource(audio.AudioGateThread.run)
    feed_at = source.index("self.interrupt_feed(chunk)")
    guard_at = source.index("if command_listening and self.interrupt_feed")
    assert guard_at < feed_at, "the feed is not guarded by command_listening"


def test_the_worker_keeps_the_listener_alive_while_paused():
    source = inspect.getsource(live_worker.GenAILiveWorker)
    assert "command_listen_check=lambda: self.paused" in source


def test_a_resume_command_from_the_listener_calls_resume():
    """_on_interrupt_phrase maps the RESUME action to resume() — the path the
    paused listener now feeds."""
    source = inspect.getsource(live_worker.GenAILiveWorker._on_interrupt_phrase)
    assert "ACTION_RESUME" in source
    assert "self.resume()" in source


def test_paused_still_gates_the_interaction_channel():
    """Hearing is separated from interaction — the live channel must STILL be
    gated off while paused, or pause would stop meaning anything."""
    source = inspect.getsource(live_worker.GenAILiveWorker._can_capture_microphone)
    assert "not self.paused" in source


def test_the_command_listener_fires_resume_end_to_end():
    """Drive the real gate loop for a few chunks with a stub command feed that
    returns 'resume', and confirm on_interrupt is called with it."""
    import queue
    import threading
    import types

    fired: list[str] = []

    gate = audio.AudioGateThread.__new__(audio.AudioGateThread)
    # Minimal hand-wired state — this exercises the run() decision, not Qt.
    gate.stop_event = threading.Event()
    gate.raw_q = queue.Queue()
    gate.out_q = types.SimpleNamespace(full=lambda: False, put_nowait=lambda x: None)
    gate.loop = types.SimpleNamespace(call_soon_threadsafe=lambda fn, *a: fn(*a))
    gate.bus = types.SimpleNamespace(
        amplitude=types.SimpleNamespace(emit=lambda *a: None),
        log=types.SimpleNamespace(emit=lambda *a: None))
    gate.vad = types.SimpleNamespace(confidence=lambda c: 0.0, threshold=0.5)
    gate.recogniser = None
    gate.on_transcript = None
    gate.speaking_check = lambda: False          # NOT speaking
    gate.on_barge_in = None
    gate.interrupt_feed = lambda chunk: "resume"  # the listener hears "resume"
    gate.on_interrupt = fired.append
    gate.command_listen_check = lambda: True       # i.e. paused
    gate.can_capture = lambda: False               # interaction channel gated
    gate.speaker_tracker = None
    gate.is_own_echo = None
    gate.on_input_silent = None
    gate.on_input_restored = None
    # Fields the loop touches.
    gate._was_speaking = False
    gate._voice_until = 0.0
    gate._deaf_until = 0.0
    gate._last_signal_at = 0.0
    gate._silence_fired = False
    gate.on_input_silent = None
    gate._SIGNAL_FLOOR = 1e9      # treat as silence so watchdog stays quiet
    gate._barge_in_streak_started = None
    gate._barge_in_rec = None

    # One chunk, then stop.
    gate.raw_q.put(b"\x00\x01" * 256)

    def _stop_soon():
        import time
        time.sleep(0.15)
        gate.stop_event.set()

    stopper = threading.Thread(target=_stop_soon, daemon=True)
    stopper.start()
    try:
        gate.run()
    except Exception:
        pass          # a missing optional field is fine; we only need the feed path
    stopper.join(timeout=1.0)

    assert "resume" in fired, "the paused command listener did not fire resume"
