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


def test_the_local_recogniser_hears_the_start_of_the_word():
    """The recogniser only received chunks after the voice detector fired, so a
    soft opening ("O-" of "Orion") under its threshold was never heard."""
    import queue
    import threading
    import types

    fed: list[bytes] = []
    gate = audio.AudioGateThread.__new__(audio.AudioGateThread)
    gate.stop_event = threading.Event()
    gate.raw_q = queue.Queue()
    gate.loop = types.SimpleNamespace(call_soon_threadsafe=lambda fn, *a: fn(*a))
    logged: list[str] = []
    gate.bus = types.SimpleNamespace(
        amplitude=types.SimpleNamespace(emit=lambda *a: None),
        log=types.SimpleNamespace(emit=logged.append))
    voiced = b"\x7f\x7f" * 256
    gate.vad = types.SimpleNamespace(confidence=lambda c: 0.9 if c == voiced else 0.1,
                                     threshold=0.5)
    gate.recogniser = types.SimpleNamespace(feed=lambda c: fed.append(c) or "")
    gate.on_transcript = lambda text: None
    gate.speaking_check = lambda: False
    gate.on_barge_in = None
    gate.interrupt_feed = None
    gate.on_interrupt = None
    gate.command_listen_check = None
    gate.can_capture = lambda: False
    gate.speaker_tracker = None
    gate.is_own_echo = None
    gate.on_input_silent = None
    gate.on_input_restored = None
    gate._was_speaking = False
    gate._voice_until = 0.0
    gate._deaf_until = 0.0
    gate._last_signal_at = 0.0
    gate._silence_fired = False
    gate._SIGNAL_FLOOR = 1e9
    gate._barge_in_streak_started = None
    gate._barge_in_rec = None
    gate._presence = None
    soft = [bytes([i, 0]) * 256 for i in range(12)]
    for chunk in soft + [voiced]:
        gate.raw_q.put(chunk)
    gate.raw_q.put(None)
    gate.run()
    assert not [m for m in logged if "recovered" in m], logged
    assert fed == soft[-audio.AudioGateThread.LOCAL_PREROLL_CHUNKS:] + [voiced]


def _offline_gate(monkeypatch, transcribe, capture=True):
    """Real gate and presence buffer, synthetic PCM, no microphone or model."""
    import queue
    import types

    callbacks = queue.Queue()
    clock = [100.0]
    monkeypatch.setattr(audio.time, "monotonic", lambda: clock[0])
    voiced = b"\x7f\x7f" * 512
    quiet = b"\x01\x00" * 512
    def confidence(chunk):
        clock[0] += 0.032
        return 0.9 if chunk == voiced else 0.1
    received = []
    logs = []
    bus = types.SimpleNamespace(log=types.SimpleNamespace(emit=logs.append),
                                amplitude=types.SimpleNamespace(emit=lambda *_: None))
    gate = audio.AudioGateThread(
        types.SimpleNamespace(call_soon_threadsafe=lambda fn, *args: callbacks.put((fn, args))),
        types.SimpleNamespace(full=lambda: False, put_nowait=lambda _: None),
        bus, lambda: capture, types.SimpleNamespace(confidence=confidence, threshold=0.5),
        recogniser=types.SimpleNamespace(feed=lambda _: "boundary brood"),
        on_transcript=received.append, speaking_check=lambda: False)
    gate.transcribe_utterance = transcribe
    gate.on_wake_word = lambda _: None
    gate._speaker_judged = lambda: False
    gate._SIGNAL_FLOOR = 1e9
    gate._presence.utterance.min_seconds = 0.2
    soft = [bytes([i, 0]) * 512 for i in range(12)]
    # Include the production 1.5-second VAD hangover, not just a short pause.
    for chunk in soft + [voiced] * 40 + [quiet] * 55:
        gate.raw_q.put(chunk)
    gate.raw_q.put(None)
    return gate, callbacks, received, logs, soft, voiced


def _deliver_gate_callbacks(callbacks):
    import queue
    while True:
        try:
            fn, args = callbacks.get_nowait()
        except queue.Empty:
            break
        fn(*args)


def test_offline_utterance_uses_whisper_once_with_the_existing_preroll(monkeypatch):
    import threading
    completed = threading.Event()
    captured = []
    def transcribe(pcm):
        captured.append(pcm)
        completed.set()
        return "Orion, show my reminders."
    gate, callbacks, received, logs, soft, voiced = _offline_gate(monkeypatch, transcribe)
    try:
        gate.run()
        assert completed.wait(1), "finished offline PCM never reached Whisper"
        # Worker completion enqueues delivery after the transcriber returns.
        for _ in range(100):
            _deliver_gate_callbacks(callbacks)
            if received:
                break
            threading.Event().wait(0.005)
        assert received == ["Orion, show my reminders."]
        assert len(captured) == 1
        assert captured[0].startswith(b"".join(soft[-10:]) + voiced * 40)
        assert not [m for m in logs if "recovered" in m]
    finally:
        gate.stop()
        thread = getattr(gate, "_transcription_thread", None)
        if thread is not None:
            thread.join(1)


def test_vosk_finals_reject_low_or_missing_confidence_but_keep_clean_words():
    from orion_core.listening_vocabulary import vosk_final_text
    assert vosk_final_text({"text": "orion stop", "result": [
        {"conf": 0.91}, {"conf": 0.97}]}) == "orion stop"
    for words in ([], [{"conf": 0.3}, {"conf": 0.9}], [{"conf": float("nan")}],
                  [{"conf": -1}], [{"conf": "bad"}], [{}]):
        assert vosk_final_text({"text": "orion stop", "result": words}) == ""
    assert vosk_final_text({"partial": "orion"}) == ""


def test_offline_microphone_keeps_the_wake_word_and_pause_listener(monkeypatch):
    import asyncio
    import types
    from tests.test_live_worker_construction import _construct
    import orion_core.live_worker as lw
    worker = _construct(monkeypatch)
    worker.wake_mode_enabled = True
    worker.wake_active_until = 0
    worker.standby_mode = False
    assert worker._can_capture_offline_microphone()
    assert not worker._can_capture_microphone()
    calls = []
    mic = types.SimpleNamespace(set_enabled=lambda _: None, start=lambda: None)
    monkeypatch.setattr(lw, "MicrophoneEngine", lambda *a, **kw: calls.append(kw) or mic)
    async def start():
        worker._ensure_fallback_mic()
    asyncio.run(start())
    assert calls[0]["transcribe_utterance"] == worker._transcribe_offline_utterance
    assert calls[0]["on_wake_word"] == worker._on_local_wake
    worker.paused = True
    assert calls[0]["command_listen_check"]()
    assert not worker._can_capture_offline_microphone()


def test_offline_transcript_keeps_casing_and_punctuation(monkeypatch):
    from tests.test_live_worker_construction import _construct
    import orion_core.live_worker as lw
    worker = _construct(monkeypatch)
    worker.wake_mode_enabled = False
    worker.standby_mode = False
    worker.local_brain = None
    worker._is_own_echo = lambda _: False
    worker._is_duplicate_command = lambda _: False
    worker._handle_quiet_command = lambda _: False
    worker._handle_focus_command = lambda _: False
    worker._handle_finish_speaking_command = lambda _: False
    worker._handle_device_command = lambda _: False
    monkeypatch.setattr(lw.VoiceInterruptManager, "classify", lambda _: "")
    received = []
    async def submit(text):
        received.append(text)
    worker.submit_text = submit
    monkeypatch.setattr(lw.background, "spawn", lambda coro: __import__("asyncio").run(coro))
    worker._on_local_transcript("Orion, remind Kieran about Snow Hill.")
    assert received == ["remind Kieran about Snow Hill."]


def test_whisper_does_not_block_capture_and_stopped_gates_drop_late_words(monkeypatch):
    import threading
    entered, release = threading.Event(), threading.Event()
    caller = threading.get_ident()
    workers = []
    def slow(pcm):
        workers.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        return "This old request must not execute."
    gate, callbacks, received, logs, _soft, _voiced = _offline_gate(monkeypatch, slow)
    try:
        gate.run()  # Must return while the recogniser is still waiting.
        assert entered.wait(1)
        assert workers == [workers[0]] and workers[0] != caller
        gate.stop()
        release.set()
        gate._transcription_thread.join(1)
        _deliver_gate_callbacks(callbacks)
        assert received == []
        assert not [m for m in logs if "recovered" in m]
    finally:
        release.set()
        gate.stop()


def test_closed_interaction_gate_never_transcribes_or_executes_vosk_fragments(monkeypatch):
    calls = []
    gate, callbacks, received, logs, _soft, _voiced = _offline_gate(
        monkeypatch, lambda pcm: calls.append(pcm) or "Never act.", capture=False)
    try:
        gate.run()
        _deliver_gate_callbacks(callbacks)
        assert calls == []
        assert received == []
        assert not [m for m in logs if "recovered" in m]
    finally:
        gate.stop()
