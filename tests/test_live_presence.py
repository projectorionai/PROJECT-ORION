"""ORION hears who is speaking on the LIVE path too, not only the offline one.

``voice_presence`` — the enrolled voiceprint, the tone reading and the
only-my-voice gate — was fed from exactly one place: ``speech_offline``, which
is handed a whole utterance by an upstream VAD. The Gemini Live path has no
such moment. It forwards every chunk, voice and silence alike, because the
server's VAD is what finds end-of-turn; filtering to voiced chunks is what once
made the live channel deaf.

So in a live session ``voice_tone`` answered from a stale reading or nothing at
all, the enrolled voiceprint was never compared against anything, and the gate
could not apply. Half of ORION's hearing worked only in the half of the
pipeline most conversations do not take.

These drive the assembly with bytes. No microphone, no ONNX session, no
QApplication — the point is that the seam exists and behaves, and the pieces
either side of it are already covered by their own tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import live_presence  # noqa: E402
from orion_core.live_presence import LivePresenceBridge, LiveUtterance  # noqa: E402

RATE = live_presence.SAMPLE_RATE


def chunk(seconds: float) -> bytes:
    """Silence of a given length, in the format the gate captures."""
    return b"\x01\x00" * int(RATE * seconds)


# ── assembling an utterance ──────────────────────────────────────────────────

def test_chunks_accumulate_into_one_utterance():
    utterance = LiveUtterance()
    for _ in range(4):
        utterance.feed(chunk(0.5))
    assert utterance.seconds == pytest.approx(2.0, abs=0.01)
    assert len(utterance.audio()) == len(chunk(2.0))


def test_a_finished_utterance_is_handed_over_whole_and_resets():
    utterance = LiveUtterance()
    utterance.feed(chunk(2.0))
    assert len(utterance.end()) == len(chunk(2.0))
    assert utterance.seconds == 0.0, "the next utterance inherited this one"


def test_something_too_short_to_read_is_discarded():
    """A cough, a chair, one word of agreement. Reading a speaker from a
    quarter of a second produces a noisy embedding, and acting on it is worse
    than not acting."""
    utterance = LiveUtterance()
    utterance.feed(chunk(0.3))
    assert utterance.end() == b""


def test_a_monologue_is_capped_rather_than_accumulated():
    """Five minutes without a pause should not hold five minutes of PCM. The
    encoder learns nothing in the fourth minute it did not know in the
    second."""
    utterance = LiveUtterance(max_seconds=3.0)
    for _ in range(20):
        utterance.feed(chunk(1.0))
    assert utterance.seconds <= 3.0 + 1.0
    assert utterance.truncated is True


def test_the_owner_is_asked_about_once_per_utterance():
    """Once, part-way through — not on every chunk after the threshold, which
    would start an ONNX inference thirty times a second."""
    utterance = LiveUtterance(decide_after=1.0)
    utterance.feed(chunk(0.5))
    assert utterance.ready_to_decide() is False, "asked before anyone spoke"
    utterance.feed(chunk(0.8))
    assert utterance.ready_to_decide() is True
    assert utterance.ready_to_decide() is False, "asked twice for one utterance"


def test_ending_an_utterance_clears_the_mute_and_the_question():
    """Each utterance is judged on its own. A stranger speaking once must not
    silence whoever talks next."""
    utterance = LiveUtterance(decide_after=1.0)
    utterance.feed(chunk(1.5))
    utterance.ready_to_decide()
    utterance.muted = True
    utterance.end()
    assert utterance.muted is False
    assert utterance.asked is False


# ── handing it off ───────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self.seen = []

    def observe_pcm(self, pcm, rate):
        self.seen.append((len(pcm), rate))


def test_a_finished_utterance_reaches_voice_presence():
    """The whole point: a live session updates the same reader the offline
    transcriber updates."""
    presence = Recorder()
    work = []
    bridge = LivePresenceBridge(presence=presence, schedule=work.append)
    bridge.observe(chunk(2.0))
    assert len(work) == 1, "nothing was scheduled"
    work[0]()
    assert presence.seen == [(len(chunk(2.0)), RATE)]


def test_nothing_is_run_on_the_calling_thread():
    """The capture thread has a 32 ms deadline. A speaker embedding is 13 ms
    of it, and the pitch tracking more, so both belong somewhere else."""
    presence = Recorder()
    bridge = LivePresenceBridge(presence=presence, schedule=lambda work: None)
    bridge.observe(chunk(2.0))
    assert presence.seen == [], "the work ran inline on the capture thread"


def test_no_presence_reader_is_survivable():
    bridge = LivePresenceBridge(presence=None, schedule=lambda work: None)
    bridge.observe(chunk(2.0))          # must not raise


def test_an_identified_owner_is_not_described_as_a_different_person(monkeypatch):
    from orion_core import voice_enrolment, voiceprint
    from orion_core.voice_presence import Presence

    turns = voice_enrolment.SpeakerTurns()
    monkeypatch.setattr(voice_enrolment, "_TURNS", turns)
    turns.record(voiceprint.Verdict(True, .9, "SyntheticOwner", confident=True, kind="owner"))
    assert Presence(speaker="SyntheticOwner", identified=True).prompt_line() == ""
    assert "not the usual user" in Presence(speaker="Other", identified=True).prompt_line()


def test_cleaned_offline_audio_does_not_repeat_raw_identity_but_phone_audio_still_checks():
    from types import SimpleNamespace
    from orion_core.speech_offline import OfflineTranscriber

    observed, judged = [], []
    transcriber = OfflineTranscriber(defer=True)
    transcriber._selected = True
    transcriber.engine = "synthetic"
    transcriber._run_engine = lambda _pcm: "words"
    transcriber.presence = SimpleNamespace(observe_pcm=lambda *a: observed.append(a))
    transcriber._owner_is_speaking = lambda *a: judged.append(a) or True
    pcm = chunk(2)
    assert transcriber.transcribe_pcm(pcm, identity_checked=True) == "words"
    assert not observed and not judged
    assert transcriber.transcribe_pcm(pcm) == "words"
    assert observed == [(pcm, RATE)] and judged == [(pcm, RATE)]


def test_a_scheduler_that_throws_never_reaches_the_capture_thread():
    """An exception here costs ORION his hearing, and a tone reading is not
    worth that."""
    logged = []

    def broken(work):
        raise RuntimeError("the loop is gone")

    bridge = LivePresenceBridge(presence=Recorder(), schedule=broken,
                                log=logged.append)
    bridge.observe(chunk(2.0))          # must not raise
    assert logged and "presence" in logged[0]


# ── it is actually wired in ──────────────────────────────────────────────────

AUDIO = (ROOT / "orion_core" / "audio.py").read_text(encoding="utf-8")
WORKER = (ROOT / "orion_core" / "live_worker.py").read_text(encoding="utf-8")
APP = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")


def test_the_capture_loop_assembles_the_utterance():
    assert "presence.utterance.feed(raw_chunk)" in AUDIO, (
        "the live path still has no complete utterance to read")


def test_the_capture_loop_survives_having_no_presence_reader():
    """Reading who is speaking is a nice-to-have. Hearing "ORION resume"
    while paused is not, and the first must never take out the second.

    Caught by test_pause_hearing, which builds a gate with __new__ and sets
    only the fields the capture loop touches — a reasonable test of the
    capture loop, which should not have to know this was added later.
    """
    assert "\n    _presence = None" in AUDIO, (
        "there is no class-level default, so an instance built by __new__ "
        "raises in the capture loop")
    body = AUDIO[AUDIO.index("presence = self._presence"):]
    body = body[:body.index("command_listening")]
    assert "presence is not None" in body


def test_it_is_read_on_the_same_terms_as_the_gender_tracker():
    """Voiced, not ORION's own output, past the echo guard, capture allowed.

    Feeding it ORION's own voice would enrol him as a speaker and then
    recognise him as one.
    """
    assert "own_voice = (" in AUDIO
    body = AUDIO[AUDIO.index("own_voice = ("):]
    body = body[:body.index("if own_voice and self.speaker_tracker")]
    for condition in ("voiced", "not speaking", "_past_echo_guard",
                      "can_capture()"):
        assert condition in body, f"{condition} is no longer required"


def test_a_muted_utterance_is_not_forwarded_to_the_model():
    """Otherwise the gate records a verdict and changes nothing, which is
    exactly the state this replaced."""
    guard = "self._presence is not None and self._presence.utterance.muted"
    assert guard in AUDIO
    assert AUDIO.index(guard) < AUDIO.index('media = {"data": chunk')


def test_only_a_confident_miss_mutes_anything():
    """Every uncertain case already resolves to listening inside voiceprint.
    A second place that can decide to go deaf is a second place to get it
    wrong."""
    body = AUDIO[AUDIO.index("def _on_speaker_verdict"):]
    body = body[:body.index("\n    def ", 10)]
    assert 'getattr(verdict, "is_owner", True)' in body, (
        "an unreadable verdict would silence him")


def test_the_worker_carries_it_to_both_microphone_engines():
    """There are two — the reconnect path builds its own — and a reader
    attached to one of them works until the first reconnection."""
    assert WORKER.count("voice_presence=self.voice_presence,") == 2


def test_app_hands_the_reader_to_the_worker():
    assert "worker.voice_presence = voice_presence" in APP
    assert APP.index("voice_presence = VoicePresence") < APP.index(
        "worker.voice_presence = voice_presence"), (
        "the worker is handed the reader before it exists")


def test_the_offline_path_still_has_its_own():
    """This adds a second feeder; it must not have moved the first."""
    assert "offline_stt.presence = voice_presence" in APP


def _room_recorder(tmp_path, **kwargs):
    from orion_core.conversation_recorder import ConversationRecorder

    settings = {"clock": lambda: 100.0, "transcribe": lambda *_: "Synthetic room words",
                "identify": lambda *_: None}
    settings.update(kwargs)
    return ConversationRecorder(tmp_path / "recordings", **settings)


def test_recording_tap_precedes_interaction_queue_drops_and_drain(tmp_path):
    from queue import Queue
    from orion_core.audio import AudioGateThread

    service = _room_recorder(tmp_path)
    assert service.start(consent=True).ok
    gate = AudioGateThread.__new__(AudioGateThread)
    gate._conversation_recorder = service
    gate.raw_q = Queue(maxsize=1)
    gate._enqueue_capture(b"\x00\x10" * 6400, 100.4)
    gate._enqueue_capture(b"\x00\x10" * 6400, 100.8)
    assert gate.raw_q.qsize() == 1
    gate.drain()
    result = service.stop()
    assert result.receipt["audio_seconds"] == pytest.approx(.8)


def test_room_recording_still_obeys_explicit_mute_pause_and_shutdown_check(tmp_path):
    service = _room_recorder(tmp_path)
    assert service.start(consent=True).ok
    service.set_capture_check(lambda: False)
    assert not service.feed(b"\x00\x10" * 6400, 100.4)
    service.set_capture_check(lambda: True)
    assert service.feed(b"\x00\x10" * 6400, 100.8)
    result = service.stop()
    assert result.receipt["audio_seconds"] == pytest.approx(.4)
    assert not service.feed(chunk(.4), 101.2)


@pytest.mark.parametrize("vosk,whisper,brain,provider,expected", [
    (True, False, True, False, True),
    (False, True, True, False, True),
    (False, True, False, True, True),
    (False, False, True, False, False),
    (True, False, False, False, False),
])
def test_offline_voice_readiness_includes_deferred_cached_hearing_and_local_rules(
        monkeypatch, vosk, whisper, brain, provider, expected):
    from types import SimpleNamespace
    from orion_core import live_worker, speech_offline

    transcriber = SimpleNamespace(ready=False, cached_available=whisper)
    monkeypatch.setattr(speech_offline, "shared", lambda: transcriber)
    worker = SimpleNamespace(
        mic=SimpleNamespace(_stream=object(), transcribe_utterance=lambda *_: ""),
        microphone_enabled=True, recogniser=SimpleNamespace(available=vosk),
        local_brain=object() if brain else None,
        router=SimpleNamespace(has_text_fallback=lambda: provider))
    assert live_worker.GenAILiveWorker._offline_voice_ready(worker) is expected


@pytest.mark.parametrize("muted,closed", [(True, False), (False, True)])
def test_offline_voice_readiness_requires_an_open_unmuted_microphone(monkeypatch, muted, closed):
    from types import SimpleNamespace
    from orion_core import live_worker, speech_offline

    monkeypatch.setattr(speech_offline, "shared", lambda: pytest.fail("muted/closed mic probed STT"))
    worker = SimpleNamespace(
        mic=SimpleNamespace(_stream=None if closed else object(), transcribe_utterance=lambda *_: ""),
        microphone_enabled=not muted, recogniser=SimpleNamespace(available=True),
        local_brain=object(), router=SimpleNamespace(has_text_fallback=lambda: True))
    assert not live_worker.GenAILiveWorker._offline_voice_ready(worker)


@pytest.mark.parametrize("fault", ["log", "persist", "speak"])
def test_completed_offline_action_is_not_retried_when_output_callback_fails(fault):
    import asyncio
    import threading
    from types import SimpleNamespace
    from orion_core.live_worker import GenAILiveWorker

    handled = []

    async def preflight(text):
        handled.append(text)
        return "A completed local action."

    def callback(stage):
        def complete(*_):
            if stage == fault:
                raise RuntimeError("synthetic post-action output failure")
        return complete

    worker = SimpleNamespace(
        paused=False, stop_event=threading.Event(), _fallback_epoch=0,
        bus=SimpleNamespace(log=SimpleNamespace(emit=callback("log"))),
        router=SimpleNamespace(is_online=lambda: False, has_text_fallback=lambda: pytest.fail(
            "an already completed local action reached provider retry")),
        local_brain=SimpleNamespace(try_known_response=preflight),
        _emit_state=lambda *_: None, _persist_episode=callback("persist"), _say=callback("speak"))
    with pytest.raises(RuntimeError, match="post-action output failure"):
        asyncio.run(GenAILiveWorker._run_text_fallback(worker, "Synthetic local action"))
    assert handled == ["Synthetic local action"]


def test_room_recording_discards_every_byte_captured_before_session_consent(tmp_path):
    service = _room_recorder(tmp_path)
    assert service.start(consent=True).ok
    assert not service.feed(b"\x00\x10" * 6400, 99.9)
    assert service.feed(b"\x00\x10" * 6400, 100.1)
    result = service.stop()
    assert result.receipt["audio_seconds"] == pytest.approx(.1, abs=1 / RATE)


def test_room_recording_overflow_is_nonblocking_and_reported_durably(tmp_path):
    import json
    import threading

    service = _room_recorder(tmp_path, queue_limit=1)
    assert service.start(consent=True).ok
    session = service._session
    entered, release = threading.Event(), threading.Event()
    write = session.wav.writeframes

    def delayed(block):
        entered.set()
        assert release.wait(3), "synthetic writer was not released"
        write(block)

    session.wav.writeframes = delayed
    try:
        assert service.feed(b"\x00\x10" * 6400, 100.4)
        assert entered.wait(3)
        assert service.feed(b"\x00\x10" * 6400, 100.8)
        assert not service.feed(b"\x00\x10" * 6400, 101.2)
    finally:
        release.set()
        result = service.stop()
    assert result.receipt["state"] == "partial"
    assert result.receipt["dropped_blocks"] == 1
    assert result.receipt["dropped_seconds"] == pytest.approx(.4)
    saved = json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["dropped_blocks"] == 1 and saved["frames"] == 12800


def test_room_recording_slow_transcription_cannot_block_audio_writer_or_restart(tmp_path):
    import threading

    entered, release = threading.Event(), threading.Event()

    def delayed(*_):
        entered.set()
        assert release.wait(3), "synthetic transcriber was not released"
        return "delayed words"

    service = _room_recorder(tmp_path, transcribe=delayed)
    assert service.start(consent=True).ok
    service.feed(b"\x00\x10" * 6400, 100.4)
    # Enough silence ends the first segment while capture continues.
    service.feed(b"\x00\x00" * 12000, 101.15)
    try:
        assert entered.wait(3)
        result = service.stop(timeout=.1)
        assert not service.active and result.receipt["state"] == "finalizing"
        assert result.receipt["audio_seconds"] == pytest.approx(1.15)
        assert not service.start(consent=True).ok
    finally:
        release.set()
        service.close()
    assert service.snapshot()["state"] == "stopped"


def test_room_recording_duration_limit_stops_and_never_implicitly_restarts(tmp_path):
    clock = [100.0]
    service = _room_recorder(tmp_path, max_seconds=.5, clock=lambda: clock[0])
    assert service.start(consent=True).ok
    assert service.feed(b"\x00\x10" * 6400, 100.4)
    clock[0] = 100.6
    assert not service.active
    assert not service.feed(b"\x00\x10" * 6400, 100.8)
    result = service.stop()
    assert result.receipt["stop_reason"] == "duration limit"
    assert result.receipt["audio_seconds"] <= .5


def test_record_only_mode_holds_room_transcripts_and_replies_but_allows_typed_turns(monkeypatch):
    from types import SimpleNamespace
    from orion_core import conversation_recorder, live_worker, voice_enrolment

    service = SimpleNamespace(record_only=True)
    monkeypatch.setattr(conversation_recorder, "recorder", lambda: service)
    monkeypatch.setattr(voice_enrolment, "turns", lambda: SimpleNamespace(reply_allowed=lambda: True))
    worker = live_worker.GenAILiveWorker.__new__(live_worker.GenAILiveWorker)
    assert not worker._voice_reply_allowed()
    # No other worker state is needed: overheard commands are held first.
    assert worker._on_local_transcript("shutdown Orion") is None
    worker._typed_live_turn = True
    assert worker._voice_reply_allowed()
    worker._typed_live_turn = False
    service.record_only = False
    assert worker._voice_reply_allowed()


def test_record_only_stop_tool_keeps_captured_typed_origin_after_live_turn_finishes(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from orion_core import conversation_recorder, live_worker, speaker_gate
    from orion_core.data import ToolResult

    monkeypatch.setattr(conversation_recorder, "recorder", lambda: SimpleNamespace(record_only=True))
    seen = []

    async def dispatch(name, args):
        seen.append((name, args, speaker_gate.VOICE_TURN.get()))
        return ToolResult("stopped")

    worker = live_worker.GenAILiveWorker.__new__(live_worker.GenAILiveWorker)
    worker.dispatcher = SimpleNamespace(dispatch_chain=dispatch)
    worker._typed_live_turn = False          # turn_complete already arrived
    token = live_worker._LIVE_TOOL_ORIGIN.set({"typed": True})
    try:
        payload, _ = asyncio.run(worker._dispatch_live_tool("conversation_recording", {"action": "stop"}))
    finally:
        live_worker._LIVE_TOOL_ORIGIN.reset(token)
    assert payload["ok"] and seen == [("conversation_recording", {"action": "stop"}, False)]
    worker._typed_live_turn = True           # a later typed request cannot authorise old room audio
    token = live_worker._LIVE_TOOL_ORIGIN.set({"typed": False})
    try:
        payload, _ = asyncio.run(worker._dispatch_live_tool("conversation_recording", {"action": "stop"}))
    finally:
        live_worker._LIVE_TOOL_ORIGIN.reset(token)
    assert not payload["ok"] and len(seen) == 1


@pytest.mark.parametrize("arrives_after_stop", [False, True])
def test_room_audio_captured_before_stop_stays_local_when_raw_queue_is_consumed_later(tmp_path, arrives_after_stop):
    import threading
    from queue import Queue
    from orion_core.audio import AudioGateThread

    clock = [100.0]
    service = _room_recorder(tmp_path, clock=lambda: clock[0])
    assert service.start(consent=True).ok
    gate = AudioGateThread.__new__(AudioGateThread)
    gate._conversation_recorder = service
    gate.raw_q = Queue()
    gate.stop_event = threading.Event()
    gate._check_input_silence = lambda: None
    gate._clean_capture = lambda *_: pytest.fail("room block reached interaction processing after stop")
    clock[0] = 100.5
    if arrives_after_stop:
        service.stop()
    gate._enqueue_capture(b"\x00\x10" * 6400, 100.4)
    if not arrives_after_stop:
        service.stop()
    assert not service.active
    assert gate.raw_q.queue[0][2] is True
    gate.raw_q.put_nowait(None)
    gate.run()
    assert gate.raw_q.empty()


def test_capture_block_straddling_recording_stop_is_entirely_local(tmp_path):
    clock = [100.0]
    service = _room_recorder(tmp_path, clock=lambda: clock[0])
    assert service.start(consent=True).ok
    clock[0] = 100.5
    service.stop()
    assert service.record_only_at(100.6, duration=.2)
    assert not service.record_only_at(100.8, duration=.2)


def test_record_only_suppresses_speaker_context_notes_even_when_typed_reply_is_allowed(monkeypatch):
    import asyncio
    import threading
    from types import SimpleNamespace
    from orion_core import conversation_recorder, live_worker

    monkeypatch.setattr(conversation_recorder, "recorder", lambda: SimpleNamespace(record_only=True))
    worker = live_worker.GenAILiveWorker.__new__(live_worker.GenAILiveWorker)
    worker._typed_live_turn = True
    worker.stop_event = threading.Event()
    # A suppressed note needs neither a session, speech renderer nor transport.
    worker._on_listener_changed("Synthetic speaker context")
    asyncio.run(worker._send_listener_note("Synthetic speaker context"))


def test_app_recording_capture_predicate_obeys_mute_ptt_pause_standby_and_stop(tmp_path, monkeypatch):
    import ast
    import threading
    from types import SimpleNamespace
    from orion_core import push_to_talk

    worker = SimpleNamespace(microphone_enabled=True, paused=False, standby_mode=False,
                             _standby=False, stop_event=threading.Event(),
                             wake_mode_enabled=True, wake_active_until=0.0)
    ptt = [True]
    monkeypatch.setattr(push_to_talk, "gate_allows_capture", lambda: ptt[0])
    # Execute only the real composition predicate; never boot the app or a mic.
    node = next(n for n in ast.walk(ast.parse(APP))
                if isinstance(n, ast.FunctionDef) and n.name == "_recording_capture_allowed")
    namespace = {"worker": worker, "__package__": "orion_core"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<recording capture policy>", "exec"), namespace)
    allowed = namespace["_recording_capture_allowed"]
    service = _room_recorder(tmp_path)
    service.set_capture_check(allowed)
    assert service.start(consent=True).ok
    try:
        for field, blocked in [("microphone_enabled", False), ("paused", True),
                               ("standby_mode", True), ("_standby", True)]:
            previous = getattr(worker, field)
            setattr(worker, field, blocked)
            assert not allowed() and not service.feed(chunk(.4), 100.4)
            setattr(worker, field, previous)
        worker.stop_event.set()
        assert not allowed() and not service.feed(chunk(.4), 100.4)
        worker.stop_event.clear()
        ptt[0] = False
        assert not allowed() and not service.feed(chunk(.4), 100.4)
        ptt[0] = True
        # The ordinary wake gate may be asleep; explicit recording still works.
        assert allowed() and service.feed(chunk(.4), 100.4)
    finally:
        result = service.stop()
    assert result.receipt["audio_seconds"] == pytest.approx(.4)



def test_a_quiet_room_is_not_a_dead_microphone():
    """20-29 Oct 2026: 30 s of the owner simply not talking read as a dead
    microphone - "are you still there?" and standby, 21 times."""
    import array
    from orion_core.audio import AudioGateThread
    room = array.array("h", [20 if i % 2 else -20 for i in range(512)]).tobytes()   # about -64 dBFS
    dither = array.array("h", [1 if i % 2 else -1 for i in range(512)]).tobytes()
    assert AudioGateThread._peak(room) > AudioGateThread._SIGNAL_FLOOR
    assert AudioGateThread._peak(b"\x00\x00" * 512) <= AudioGateThread._SIGNAL_FLOOR
    assert AudioGateThread._peak(dither) <= AudioGateThread._SIGNAL_FLOOR
