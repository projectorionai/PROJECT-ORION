"""Long streamed replies keep their speech and do not inherit old pauses."""

from __future__ import annotations

from array import array
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

from orion_core.audio import AudioPlaybackThread, SpeechQueueManager, SpeechSynthesiser
from orion_core.audio_state import AudioStateMachine
from orion_core.constants import CHUNK_SIZE, PLAYBACK_PREBUFFER_CHUNKS, RECEIVE_SAMPLE_RATE


class _Signal:
    def emit(self, *_args):
        pass


class _Bus:
    def __getattr__(self, name):
        signal = _Signal()
        setattr(self, name, signal)
        return signal


class _Tts:
    def attach_renderer(self, renderer):
        self.renderer = renderer

    def is_busy(self):
        return False


def _manager():
    bus = _Bus()
    playback = AudioPlaybackThread(bus)
    manager = SpeechQueueManager(bus, playback, _Tts(), AudioStateMachine(bus))
    return manager, playback


def _pcm(ms: int, sample: int) -> bytes:
    return array("h", [sample] * (RECEIVE_SAMPLE_RATE * ms // 1000)).tobytes()


def _queued_audio(playback: AudioPlaybackThread) -> bytes:
    parts = []
    while not playback.queue.empty():
        parts.append(playback.queue.get_nowait()[1])
    return b"".join(parts)


def test_native_long_pause_across_chunks_is_capped_without_losing_words():
    manager, playback = _manager()
    leading = _pcm(200, 6000)
    trailing = _pcm(200, -6000)
    response = leading + _pcm(1500, 0) + trailing

    for start in range(0, len(response), 4096):
        manager.enqueue_native_audio(response[start:start + 4096])
    manager.finish_native_audio()

    played = _queued_audio(playback)
    assert played.startswith(leading)
    assert played.endswith(trailing)
    assert len(played) < len(leading + _pcm(500, 0) + trailing)


def test_native_turn_end_flushes_partial_frame_and_resets_pause_state():
    manager, playback = _manager()
    short = _pcm(2, 6000)
    manager.enqueue_native_audio(short)
    assert playback.queue.empty()
    manager.finish_native_audio()
    assert _queued_audio(playback) == short

    manager.enqueue_native_audio(_pcm(1400, 0))
    manager.finish_native_audio()
    _queued_audio(playback)
    next_reply = _pcm(200, 0)
    manager.enqueue_native_audio(next_reply)
    manager.finish_native_audio()
    assert _queued_audio(playback) == next_reply


def test_renderer_starts_with_a_useful_network_jitter_buffer():
    _, playback = _manager()
    packet = b"\x01\x00" * CHUNK_SIZE
    for _ in range(PLAYBACK_PREBUFFER_CHUNKS - 1):
        playback.enqueue(packet)

    buffered = playback._prebuffer(packet)
    assert len(buffered) >= 16 * len(packet)
    assert buffered == packet * PLAYBACK_PREBUFFER_CHUNKS


def test_network_gap_does_not_reopen_the_microphone_mid_reply():
    manager, playback = _manager()
    manager.enqueue_native_audio(_pcm(40, 6000))
    manager.machine.native_started()
    _queued_audio(playback)
    manager.machine.native_stopped()  # The device briefly ran out of PCM.

    assert manager.output_active()
    assert not manager.output_playing()
    assert manager.machine.describe()["native_stream_active"]
    assert manager.machine.illegal_combination() == ""

    manager.enqueue_native_audio(_pcm(40, -6000))
    manager.machine.native_started()
    manager.finish_native_audio()
    assert manager.output_active()  # Final PCM still belongs to the device.
    manager.machine.native_stopped()
    assert not manager.output_active()


def test_missing_turn_completion_releases_the_voice_channel_after_a_bounded_gap(monkeypatch):
    import orion_core.audio as module

    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    manager, playback = _manager()
    manager.enqueue_native_audio(_pcm(2, 6000))
    clock[0] = 127.9
    assert not manager.expire_native_stream()
    assert manager.output_active()
    clock[0] = 128.1
    assert manager.expire_native_stream()
    assert not manager.machine.describe()["native_stream_active"]
    _queued_audio(playback)
    manager.machine.native_stopped()
    assert not manager.output_active()
    assert manager._native_stream_at is None


def test_completion_before_the_renderer_starts_keeps_queued_final_words_reserved():
    manager, playback = _manager()
    manager.enqueue_native_audio(_pcm(2, 6000))
    manager.finish_native_audio()
    assert not playback.is_active()
    assert manager.output_active()
    assert _queued_audio(playback) == _pcm(2, 6000)
    manager.machine.native_stopped()
    assert not manager.output_active()


def test_cancelled_native_reply_cannot_keep_the_microphone_reserved():
    manager, _ = _manager()
    manager.enqueue_native_audio(_pcm(40, 6000))
    assert manager.output_active()
    manager.interrupt_all()
    assert not manager.output_active()
    assert manager._native_stream_at is None


def test_late_cloud_chunks_from_before_stop_cannot_enter_the_next_reply():
    _, playback = _manager()
    old_generation = playback._generation
    playback.enqueue_for_generation(_pcm(20, 6000), old_generation)
    playback.clear()
    # A provider callback can run after clear() and before TTS notices its stop.
    playback.enqueue_for_generation(_pcm(20, -6000), old_generation)
    assert playback.queue.empty()
    playback.enqueue(_pcm(20, 4000))
    assert _queued_audio(playback) == _pcm(20, 4000)


def _speech_manager():
    bus = _Bus()
    machine = AudioStateMachine(bus)
    playback = AudioPlaybackThread(bus, machine)
    tts = SpeechSynthesiser(bus, machine)
    # No SAPI initialisation, cache warming, devices, models, or network.
    tts._initialise_engine = lambda: None
    tts._start_prewarm = lambda: None
    tts._start_kokoro_watch = lambda: None
    return SpeechQueueManager(bus, playback, tts, machine), playback, tts


def _speech_pcm(value, samples=240):
    return array("h", [value] * samples).tobytes()


def _drain_speech(playback):
    parts = []
    while not playback.queue.empty():
        parts.append(playback.queue.get_nowait()[1])
    return b"".join(parts)


def test_native_turn_waits_for_a_reserved_local_batch_including_partial_frame():
    manager, playback, _tts = _speech_manager()
    local_first, local_last = _speech_pcm(9000), _speech_pcm(-9000)
    native_first, native_last = _speech_pcm(6000), _speech_pcm(-6000, samples=17)

    assert manager._acquire_tts_output()
    playback.enqueue(local_first)
    manager.enqueue_native_audio(native_first)
    playback.enqueue(local_last)
    manager.enqueue_native_audio(native_last)
    manager.finish_native_audio()

    assert _drain_speech(playback) == local_first + local_last
    assert manager._deferred_native_audio
    manager._release_tts_output()
    assert _drain_speech(playback) == native_first + native_last
    assert manager.output_active(), "Completed native audio still awaits its device turn"
    assert not manager._deferred_native_audio
    manager.machine.native_stopped()
    assert not manager.output_active()


def test_local_provider_waits_for_native_turn_completion_and_queued_audio():
    manager, playback, tts = _speech_manager()
    started = Event()
    manager.enqueue_native_audio(_speech_pcm(6000))
    tts._speak_one = lambda _text: started.set()
    tts.speak("Local reply")
    tts.start()
    try:
        assert tts._waiting_for_output.wait(2)
        assert tts.is_busy(), "A dequeued local utterance is still pending speech"
        assert not started.is_set()
        # Emptying the device queue during a network gap does not end Live.
        _drain_speech(playback)
        assert not started.wait(0.06)
        manager.enqueue_native_audio(_speech_pcm(-6000))
        manager.finish_native_audio()
        assert not started.wait(0.06), "Final queued PCM retains the native turn"
        _drain_speech(playback)
        manager.machine.native_stopped()
        assert started.wait(2)
    finally:
        tts.stop_event.set()
        tts.queue.put_nowait(None)
        tts.join(2)
    assert not tts.is_alive()


def test_interruption_discards_native_pcm_parked_behind_local_speech():
    manager, playback, _tts = _speech_manager()
    assert manager._acquire_tts_output()
    manager.enqueue_native_audio(_speech_pcm(6000))
    manager.finish_native_audio()
    assert manager._deferred_native_audio

    manager.interrupt_all()
    manager._release_tts_output()

    assert playback.queue.empty()
    assert not manager._deferred_native_audio
    assert not manager.output_active()


def test_interruption_cancels_a_local_utterance_waiting_for_native_speech():
    manager, playback, tts = _speech_manager()
    started = Event()
    manager.enqueue_native_audio(_speech_pcm(6000))
    tts._speak_one = lambda _text: started.set()
    tts.speak("This pending reply must be cancelled")
    tts.start()
    try:
        assert tts._waiting_for_output.wait(2)
        manager.interrupt_all()
        tts.stop_event.set()
        tts.queue.put_nowait(None)
        tts.join(2)
        assert not started.is_set()
        assert playback.queue.empty()
        assert not manager._tts_output_reserved
    finally:
        tts.stop_event.set()
        tts.join(2)
    assert not tts.is_alive()


def test_native_turn_arriving_during_local_provider_is_flushed_on_release():
    manager, playback, tts = _speech_manager()
    first, last, native = _speech_pcm(9000), _speech_pcm(-9000), _speech_pcm(6000)
    released = Event()

    def speak(_text):
        playback.enqueue(first)
        manager.enqueue_native_audio(native)
        manager.finish_native_audio()
        playback.enqueue(last)
        # Renderer is faked by draining each whole local batch here.
        assert _drain_speech(playback) == first + last

    release = tts.release_output

    def release_and_stop():
        release()
        tts.stop_event.set()
        released.set()

    tts._speak_one = speak
    tts.release_output = release_and_stop
    tts.speak("A local device acknowledgement")
    tts.start()
    try:
        assert released.wait(2)
        tts.join(2)
        assert _drain_speech(playback) == native
        assert not manager._tts_output_reserved
    finally:
        tts.stop_event.set()
        tts.queue.put_nowait(None)
        tts.join(2)
    assert not tts.is_alive()


def test_held_local_waiter_keeps_its_text_until_native_audio_and_resume_finish():
    manager, playback, tts = _speech_manager()
    spoken, started = [], Event()
    manager.enqueue_native_audio(_speech_pcm(6000))

    def speak(text):
        spoken.append(text)
        started.set()
        tts.stop_event.set()

    tts._speak_one = speak
    tts.speak("Keep this reply for resume")
    tts.start()
    try:
        assert tts._waiting_for_output.wait(2)
        manager.hold_all()
        for _ in range(200):
            if tts._resume_text:
                break
            started.wait(0.01)
        assert tts._resume_text == "Keep this reply for resume"
        assert not started.is_set()
        _drain_speech(playback)
        manager.finish_native_audio()
        manager.machine.native_stopped()
        manager.resume_all()
        assert started.wait(2)
        tts.join(2)
        assert spoken == ["Keep this reply for resume"]
    finally:
        tts.stop_event.set()
        tts.queue.put_nowait(None)
        tts.join(2)
    assert not tts.is_alive()


def test_output_underflows_distinguish_idle_start_from_a_gap_during_speech():
    manager, playback, _tts = _speech_manager()
    playback._stream = SimpleNamespace(write=lambda _chunk: True)
    playback._write(_speech_pcm(6000, samples=playback._WRITE_SLICE_BYTES * 3 // 2))
    assert playback.output_underflows == 3
    assert playback.mid_utterance_underflows == 2
    playback._mark_inactive()
    playback._write(_speech_pcm(6000))
    assert playback.output_underflows == 4
    assert playback.mid_utterance_underflows == 2
    snapshot = manager.telemetry_snapshot()
    assert snapshot["output_underflows"] == 4
    assert snapshot["mid_utterance_underflows"] == 2


class _BrokenMetrics:
    def gauge(self, *_args):
        raise RuntimeError("metrics unavailable")

    observe = gauge
    incr = gauge


def test_metrics_failure_never_rejects_pcm_or_faults_a_successful_device_write():
    playback = AudioPlaybackThread(_Bus(), telemetry=SimpleNamespace(metrics=_BrokenMetrics()))
    played = []
    playback._stream = SimpleNamespace(write=lambda chunk: played.append(bytes(chunk)))
    pcm = _speech_pcm(6000)
    playback.enqueue(pcm)
    assert _drain_speech(playback) == pcm
    playback._write(pcm)
    assert played == [pcm]
    assert playback.last_audio_time > 0.0


def test_metrics_failure_cannot_leave_the_local_voice_or_microphone_reserved():
    manager, _playback, tts = _speech_manager()
    tts.telemetry = SimpleNamespace(metrics=_BrokenMetrics())
    tts._speak_one = lambda _text: tts.stop_event.set()
    tts.speak("Finish this utterance")
    tts.start()
    tts.join(2)
    assert not tts.is_alive()
    assert not tts.is_busy()
    assert not manager._tts_output_reserved
    assert not manager.output_active()


def test_renderer_opens_current_system_default_without_overriding_an_explicit_device(monkeypatch):
    from orion_core import audio as audio_module, audio_devices

    resolved, opened = [], []
    monkeypatch.setattr(audio_devices, "system_default_index",
                        lambda kind: resolved.append(kind) or 7)
    monkeypatch.setattr(audio_devices, "_live", {})
    monkeypatch.setattr(audio_devices, "_devices", lambda: [
        {"name": f"Output {index}", "max_output_channels": 1} for index in range(9)])

    class Stream:
        def __init__(self, device):
            self.device = 8 if device == 7 else device

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    def open_stream(**kwargs):
        opened.append(kwargs["device"])
        return Stream(kwargs["device"])

    # Replace the whole lazy module, so even importing a hardware backend is
    # unnecessary to prove which device the renderer asks PortAudio to open.
    monkeypatch.setattr(audio_module, "sd", SimpleNamespace(RawOutputStream=open_stream))
    bus, messages = _Bus(), []
    bus.log = SimpleNamespace(emit=messages.append)
    playback = AudioPlaybackThread(bus)
    playback._open_output_stream(None)
    assert audio_devices.resolve_effective("output") == 8
    playback._open_output_stream(3)
    assert audio_devices.resolve_effective("output") == 3
    playback.request_output_device(None)
    playback._apply_pending_device()
    assert audio_devices.resolve_effective("output") == 8
    assert any("[8] Output 8" in message for message in messages)
    assert resolved == ["output", "output"]
    assert opened == [7, 3, 7]


def test_releasing_native_pcm_during_hold_does_not_reactivate_a_paused_device():
    manager, playback, tts = _speech_manager()
    assert manager._acquire_tts_output()
    manager.machine.tts_started()
    manager.enqueue_native_audio(_speech_pcm(6000))
    manager.finish_native_audio()
    playback.hold()
    playback._mark_inactive()
    manager._release_tts_output()
    manager.machine.tts_stopped()
    assert not manager.output_active()
    assert _drain_speech(playback) == _speech_pcm(6000)
    tts._interrupted.set()


def test_local_reservation_rechecks_interruption_after_taking_native_lock():
    manager, _playback, tts = _speech_manager()

    class InterruptOnLock:
        def __enter__(self):
            tts._interrupted.set()

        def __exit__(self, *_args):
            pass

    manager._native_audio_lock = InterruptOnLock()
    assert not manager._acquire_tts_output()
    assert not manager._tts_output_reserved


def test_cancel_at_local_reservation_boundary_releases_without_starting_provider():
    manager, _playback, tts = _speech_manager()
    spoken = []

    def acquire_then_cancel():
        reserved = manager._acquire_tts_output()
        tts.interrupt()
        tts.stop_event.set()
        return reserved

    tts.acquire_output = acquire_then_cancel
    tts._speak_one = spoken.append
    tts.speak("A cancelled acknowledgement")
    tts.start()
    tts.join(2)
    assert not tts.is_alive()
    assert spoken == []
    assert not manager._tts_output_reserved
    assert not manager.output_active()


def test_hold_at_local_reservation_boundary_preserves_text_and_releases_device():
    manager, _playback, tts = _speech_manager()
    spoken, started = [], Event()

    def acquire_then_hold():
        reserved = manager._acquire_tts_output()
        tts.hold()
        return reserved

    def speak(text):
        spoken.append(text)
        started.set()
        tts.stop_event.set()

    tts.acquire_output = acquire_then_hold
    tts._speak_one = speak
    tts.speak("Resume this acknowledgement exactly once")
    tts.start()
    try:
        for _ in range(200):
            if tts._resume_text:
                break
            started.wait(0.01)
        assert tts._resume_text == "Resume this acknowledgement exactly once"
        assert not manager._tts_output_reserved
        assert spoken == []
        tts.acquire_output = manager._acquire_tts_output
        manager.resume_all()
        assert started.wait(2)
        tts.join(2)
        assert spoken == ["Resume this acknowledgement exactly once"]
    finally:
        tts.stop_event.set()
        tts.queue.put_nowait(None)
        tts.join(2)
    assert not tts.is_alive()


def test_local_batch_yields_to_native_before_taking_another_queued_utterance():
    manager, playback, tts = _speech_manager()
    spoken, released = [], Event()
    native = _speech_pcm(6000)

    def speak(text):
        spoken.append(text)
        # Simulate another local status message arriving while the first is
        # speaking. It must wait behind this native response.
        tts.speak("A later local status update")
        manager.enqueue_native_audio(native)
        manager.finish_native_audio()

    release = tts.release_output

    def release_and_stop():
        release()
        tts.stop_event.set()
        released.set()

    tts._speak_one = speak
    tts.release_output = release_and_stop
    tts.speak("The first local acknowledgement")
    tts.start()
    try:
        assert released.wait(2)
        tts.join(2)
        assert spoken == ["The first local acknowledgement"]
        assert tts.queue.get_nowait() == "A later local status update"
        assert _drain_speech(playback) == native
    finally:
        tts.stop_event.set()
        tts.queue.put_nowait(None)
        tts.join(2)
    assert not tts.is_alive()


def test_interrupt_at_attempted_dequeue_cannot_miss_the_removed_utterance():
    manager, _playback, tts = _speech_manager()
    attempted, spoken = Event(), []

    def cancel():
        attempted.set()
        manager.interrupt_all()
        tts.stop_event.set()
        tts._utterance_ready.set()

    canceller = Thread(target=cancel)

    class DequeueBarrier(Queue):
        armed = True

        def get_nowait(self):
            text = super().get_nowait()
            if self.armed:
                self.armed = False
                # Start cancellation after text leaves the queue but before
                # run() can advertise it as waiting for the output device.
                canceller.start()
                assert attempted.wait(2)
            return text

    tts.queue = DequeueBarrier()
    # Give cancellation the whole acquisition boundary to settle; no real
    # provider or device can run while the test observes the state handoff.
    tts.acquire_output = lambda: tts._interrupted.wait(2) and False
    tts._speak_one = spoken.append
    tts.speak("Cancel the reply being dequeued")
    tts.start()
    try:
        assert attempted.wait(2)
        canceller.join(2)
        tts.join(2)
        assert not canceller.is_alive() and not tts.is_alive()
        assert tts._interrupted.is_set(), "Cancellation must see dequeued pending text"
        assert spoken == []
        assert not manager._tts_output_reserved
    finally:
        tts.stop_event.set()
        tts._utterance_ready.set()
        tts.join(2)
        if canceller.ident is not None:
            canceller.join(2)


def test_submission_when_idle_wait_is_cleared_keeps_its_wake_and_all_text():
    _manager, _playback, tts = _speech_manager()
    attempted, spoken, finished = Event(), [], Event()

    def submit():
        attempted.set()
        tts.speak("A reply arriving at the idle boundary")

    submitter = Thread(target=submit)

    class WakeBarrier(Event):
        armed = True

        def clear(self):
            super().clear()
            if self.armed:
                self.armed = False
                # run() still owns _submission_lock here. The producer must
                # set its wake only after the empty-queue clear completes.
                submitter.start()
                assert attempted.wait(2)

    def speak(text):
        spoken.append(text)
        tts.stop_event.set()
        finished.set()

    tts._utterance_ready = WakeBarrier()
    tts._speak_one = speak
    tts.start()
    try:
        assert finished.wait(2)
        submitter.join(2)
        tts.join(2)
        assert not submitter.is_alive() and not tts.is_alive()
        assert spoken == ["A reply arriving at the idle boundary"]
    finally:
        tts.stop_event.set()
        tts._utterance_ready.set()
        tts.join(2)
        if submitter.ident is not None:
            submitter.join(2)


class _SpeechIdleWaitProbe(Event):
    """Observe the worker reaching an idle/cancellation wait without sleeps."""

    def __init__(self):
        super().__init__()
        self.waits = Queue()

    def wait(self, timeout=None):
        self.waits.put_nowait(None)
        return super().wait(timeout)


def _speech_guarded(errors, work):
    def guarded():
        try:
            work()
        except BaseException as exc:
            errors.append(repr(exc))
    return guarded


def test_slow_interruption_cannot_requeue_old_pcm_under_new_generation():
    manager, playback, tts = _speech_manager()
    entered, allow_stop, cancel_done = Event(), Event(), Event()
    errors, requests = [], []
    tts._utterance_ready = probe = _SpeechIdleWaitProbe()
    tts._kokoro = tts._engine = tts._proc = None
    tts._active_backend = ""

    class SlowOldBackend:
        def stop(self):
            entered.set()
            if not allow_stop.wait(5):
                raise RuntimeError("fake old backend stop timed out")

    class Voice:
        SAMPLE_RATE = 24000

        def stream_pcm(self, text, emotion, on_chunk, should_stop):
            if should_stop():
                return None
            requests.append(text)
            on_chunk(_speech_pcm(6000))
            return _speech_pcm(6000)

    def acquire_after_cancel():
        if not cancel_done.wait(5):
            errors.append("cancellation did not finish before acquisition")
            return False
        return manager._acquire_tts_output()

    def cancel():
        try:
            manager.interrupt_all()
        finally:
            cancel_done.set()

    tts._piper = SlowOldBackend()
    tts.acquire_output = acquire_after_cancel
    tts._speak_one = lambda text: tts._speak_cloud(text, Voice(), "fake voice")
    tts._await_renderer = lambda _renderer: tts.stop_event.set()
    tts.speak("Old cancelled request")
    generation = playback._generation
    canceller = Thread(target=_speech_guarded(errors, cancel), daemon=True)
    canceller.start()
    try:
        assert entered.wait(5)
        tts.start()
        probe.waits.get(timeout=5)  # Worker has checked the cancellation barrier.
        assert requests == []
        allow_stop.set()
        canceller.join(5)
        assert not canceller.is_alive() and cancel_done.is_set()
        # A fresh reply is the positive control: cancellation must remove the
        # old request without leaving the next generation permanently silent.
        tts.speak("Fresh request after cancellation")
        tts.join(5)
        assert not tts.is_alive() and errors == []
        assert requests == ["Fresh request after cancellation"]
        with playback.queue.mutex:
            queued = list(playback.queue.queue)
        assert playback._generation == generation + 1
        assert queued == [(generation + 1, _speech_pcm(6000))]
    finally:
        allow_stop.set()
        cancel_done.set()
        tts.stop_event.set()
        tts._utterance_ready.set()
        canceller.join(5)
        if tts.ident is not None:
            tts.join(5)


def test_new_reply_waits_for_slow_backend_stop_and_survives_its_queue_flush():
    manager, _playback, tts = _speech_manager()
    entered, allow_stop, stopped = Event(), Event(), Event()
    errors, spoken = [], []
    tts._utterance_ready = probe = _SpeechIdleWaitProbe()
    tts._kokoro = tts._engine = tts._proc = None
    tts._active_backend = ""

    class SlowOldBackend:
        def stop(self):
            entered.set()
            if not allow_stop.wait(5):
                raise RuntimeError("fake backend stop timed out")
            stopped.set()

    def speak(text):
        if not stopped.is_set():
            errors.append("fresh provider started before old backend stopped")
        spoken.append(text)
        tts.stop_event.set()

    tts._piper = SlowOldBackend()
    tts._speak_one = speak
    tts.speak("Old queued text")
    canceller = Thread(target=_speech_guarded(errors, manager.interrupt_all), daemon=True)
    canceller.start()
    try:
        assert entered.wait(5)
        tts.start()
        probe.waits.get(timeout=5)
        tts.speak("New reply submitted during cancellation")
        probe.waits.get(timeout=5)  # New wake also checks the still-held barrier.
        assert spoken == []
        assert tts.queue.qsize() == 1
        allow_stop.set()
        canceller.join(5)
        tts.join(5)
        assert not canceller.is_alive() and not tts.is_alive()
        assert errors == []
        assert spoken == ["New reply submitted during cancellation"]
        assert tts._interrupting == 0
    finally:
        allow_stop.set()
        tts.stop_event.set()
        tts._utterance_ready.set()
        canceller.join(5)
        if tts.ident is not None:
            tts.join(5)


def test_concurrent_interruptions_keep_fresh_reply_blocked_until_both_stops_return():
    _manager, _playback, tts = _speech_manager()
    entered = {name: Event() for name in ("first", "second")}
    release = {name: Event() for name in ("first", "second")}
    errors, spoken = [], []
    tts._utterance_ready = probe = _SpeechIdleWaitProbe()
    tts._kokoro = tts._engine = tts._proc = None
    tts._active_backend = ""

    class ConcurrentBackend:
        def stop(self):
            from threading import current_thread
            name = current_thread().name
            entered[name].set()
            if not release[name].wait(5):
                raise RuntimeError(f"fake {name} cancellation timed out")

    def speak(text):
        spoken.append(text)
        tts.stop_event.set()

    tts._piper = ConcurrentBackend()
    tts._speak_one = speak
    cancellers = [Thread(target=_speech_guarded(errors, tts.interrupt), name=name, daemon=True)
                  for name in ("first", "second")]
    for thread in cancellers:
        thread.start()
    try:
        assert all(event.wait(5) for event in entered.values())
        assert tts._interrupting == 2
        tts.speak("Fresh reply after both cancellations")
        tts.start()
        probe.waits.get(timeout=5)
        release["first"].set()
        cancellers[0].join(5)
        assert not cancellers[0].is_alive()
        assert tts._interrupting == 1
        tts._utterance_ready.set()  # Force another check while the second stop waits.
        probe.waits.get(timeout=5)
        assert spoken == []
        release["second"].set()
        cancellers[1].join(5)
        tts.join(5)
        assert all(not thread.is_alive() for thread in cancellers)
        assert not tts.is_alive() and errors == []
        assert tts._interrupting == 0
        assert spoken == ["Fresh reply after both cancellations"]
    finally:
        for event in release.values():
            event.set()
        tts.stop_event.set()
        tts._utterance_ready.set()
        for thread in cancellers:
            thread.join(5)
        if tts.ident is not None:
            tts.join(5)


def test_fresh_reply_waits_for_manager_renderer_clear_and_keeps_all_new_pcm():
    manager, playback, tts = _speech_manager()
    stopped, allow_clear = Event(), Event()
    errors, requests = [], []
    tts.stop_event = acquisition_probe = _SpeechIdleWaitProbe()
    tts._charon = tts._piper = tts._kokoro = None
    tts._engine = tts._proc = None
    tts._active_backend = ""
    original_interrupt = tts.interrupt

    def stop_then_pause_before_manager_clear():
        original_interrupt()
        stopped.set()
        if not allow_clear.wait(5):
            raise RuntimeError("manager renderer-clear boundary timed out")

    class Voice:
        SAMPLE_RATE = 24000

        def stream_pcm(self, text, emotion, on_chunk, should_stop):
            if should_stop():
                return None
            requests.append(text)
            on_chunk(_speech_pcm(6000))
            return _speech_pcm(6000)

    tts.interrupt = stop_then_pause_before_manager_clear
    tts._speak_one = lambda text: tts._speak_cloud(text, Voice(), "fake voice")
    tts._await_renderer = lambda _renderer: tts.stop_event.set()
    tts.speak("Old request to cancel")
    generation = playback._generation
    canceller = Thread(target=_speech_guarded(errors, manager.interrupt_all), daemon=True)
    canceller.start()
    try:
        assert stopped.wait(5)
        assert tts._interrupting == 0  # Backend cancellation itself has finished.
        tts.speak("Fresh reply waiting for the complete manager cancellation")
        tts.start()
        # This is the output-acquisition poll, not the idle utterance wait:
        # it proves the new request has reached the manager's reservation.
        acquisition_probe.waits.get(timeout=5)
        assert requests == []
        assert playback.queue.empty()
        allow_clear.set()
        canceller.join(5)
        tts.join(5)
        assert not canceller.is_alive() and not tts.is_alive()
        assert errors == []
        assert requests == ["Fresh reply waiting for the complete manager cancellation"]
        assert playback._generation == generation + 1
        with playback.queue.mutex:
            queued = list(playback.queue.queue)
        assert queued == [(generation + 1, _speech_pcm(6000))]
        assert manager._output_interruptions == 0
    finally:
        allow_clear.set()
        tts.stop_event.set()
        tts._utterance_ready.set()
        canceller.join(5)
        if tts.ident is not None:
            tts.join(5)


def test_system_voice_cancel_during_prosody_cannot_publish_old_utterance(monkeypatch):
    from orion_core import voice_emotion

    monkeypatch.setattr(voice_emotion, "prosody_for", lambda *_: (150, 1.0))
    _manager, _playback, tts = _speech_manager()
    tts._piper = tts._kokoro = tts._proc = None
    tts._active_backend = "pyttsx3"
    entered, allow_prosody = Event(), Event()
    calls, errors = [], []

    class Engine:
        def setProperty(self, name, value):
            if name == "rate":
                entered.set()
                if not allow_prosody.wait(5):
                    errors.append("fake prosody setter timed out")

        def stop(self):
            calls.append("stop")

        def say(self, text):
            calls.append(("say", text))

        def runAndWait(self):
            calls.append("runAndWait")

    tts._engine = Engine()
    worker = Thread(target=_speech_guarded(errors,
        lambda: tts._speak_pyttsx3("Cancelled old utterance")), daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        tts.interrupt()
        assert calls == ["stop"]
        assert tts._interrupted.is_set()
        allow_prosody.set()
        worker.join(5)
        assert not worker.is_alive() and errors == []
        assert calls == ["stop"]  # No old text queued after the completed stop.
    finally:
        allow_prosody.set()
        worker.join(5)


def test_system_process_cancel_during_creation_kills_published_child_without_sending_text(monkeypatch):
    from threading import current_thread
    from orion_core import audio

    _manager, _playback, tts = _speech_manager()
    tts._piper = tts._kokoro = tts._engine = tts._proc = None
    tts._active_backend = "powershell"
    entered, allow_creation, killed, cancelled = Event(), Event(), Event(), Event()
    writes, killed_by, errors = [], [], []

    class Input:
        closed = False

        def write(self, text):
            writes.append(text)

        def close(self):
            self.closed = True
            # Keep the published child visible until interrupt() has killed it.
            if not killed.wait(5):
                errors.append("cancellation did not reach the published child")

    class Process:
        def __init__(self):
            self.stdin = Input()
            self.killed = False

        def poll(self):
            return 1 if self.killed else None

        def kill(self):
            self.killed = True
            killed_by.append(current_thread().name)
            killed.set()

    process = Process()

    def create(*args, **kwargs):
        entered.set()
        if not allow_creation.wait(5):
            errors.append("fake child creation timed out")
        return process

    def cancel():
        try:
            tts.interrupt()
        finally:
            cancelled.set()

    monkeypatch.setattr(audio.subprocess, "Popen", create)
    worker = Thread(target=_speech_guarded(errors,
        lambda: tts._speak_powershell("Cancelled old utterance")), daemon=True)
    canceller = Thread(target=_speech_guarded(errors, cancel),
                       name="test-system-voice-cancel", daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        assert tts._proc is None
        canceller.start()
        # interrupt() announces cancellation before waiting for publication.
        assert tts._interrupted.wait(5)
        assert not cancelled.is_set()
        allow_creation.set()
        canceller.join(5)
        worker.join(5)
        assert not canceller.is_alive() and not worker.is_alive()
        assert errors == [] and cancelled.is_set()
        assert writes == []
        assert process.killed and killed_by == ["test-system-voice-cancel"]
        assert process.stdin.closed and tts._proc is None
    finally:
        allow_creation.set()
        tts._interrupted.set()
        killed.set()
        worker.join(5)
        if canceller.ident is not None:
            canceller.join(5)

