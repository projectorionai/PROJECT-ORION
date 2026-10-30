"""
Regression tests for the realtime send discipline (Section 3).

These reproduce the conditions behind
    RuntimeError: Cannot enter into task <Task cancelling name='orion-send-realtime'>
— concurrent sends on the one live websocket and sends racing a channel
teardown — and prove the fix: every outbound call is serialised through a
single lock, and a send scheduled once the transport has closed becomes a
clean no-op (routing text turns to the provider fallback instead).

No live provider, audio device or event loop of the app is required: the send
methods are exercised on a bare worker instance with fake sessions, so the
tests are fully deterministic and offline.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.live_worker import GenAILiveWorker


class _Log:
    def emit(self, *_a, **_k) -> None:
        pass


class _Bus:
    log = _Log()


def _make_worker(session=None, closed=False) -> GenAILiveWorker:
    """A bare worker carrying only the state the send paths touch — no audio,
    no genai client, no __init__ side effects."""
    w = GenAILiveWorker.__new__(GenAILiveWorker)
    w.session = session
    w._send_closed = closed
    w._send_lock = asyncio.Lock()
    w.stop_event = asyncio.Event()
    w.bus = _Bus()
    w._session_loop_active = False
    return w


def test_enabled_skill_instructions_reach_live_config(tmp_path):
    from orion_core.providers import AIProviderProfile, OrionProviderSettings, ProviderRouter
    from orion_core.skills import SkillManager

    worker = _make_worker()
    worker.memory = SimpleNamespace(prompt_context=lambda **kwargs: "",
                                    recent_replies=lambda *args: [])
    profile = AIProviderProfile(name="synthetic", kind="gemini_live", model="test", api_key="test", enabled=True)
    settings = OrionProviderSettings(active_provider=profile.name,
                                     provider_order=[profile.name],
                                     providers={profile.name: profile})
    worker.router = ProviderRouter(settings, worker.bus, worker.memory)
    worker.router.skills = SkillManager(worker.bus, root=tmp_path / "skills")
    worker.dispatcher = SimpleNamespace(TOOL_DECLARATIONS=[])
    worker._search_tool_enabled = False
    worker._resumption_handle = None
    worker._transcription_hints_off = True

    def instruction():
        config = worker._build_config()
        return config["system_instruction"] if isinstance(config, dict) else config.system_instruction

    assert "[SKILL research-method]" in instruction()
    assert worker.router.skills.set_enabled("research-method", False).ok
    assert "[SKILL research-method]" not in instruction()


class ConcurrencyProbeSession:
    """Records the peak number of sends executing at once; a value above 1
    means two coroutines drove the socket concurrently (the bug)."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.calls = 0

    async def send_realtime_input(self, **_kwargs) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls += 1
        try:
            # Yield control so, without the lock, other sends would interleave.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            self.active -= 1


def test_concurrent_media_sends_are_serialised() -> None:
    session = ConcurrencyProbeSession()
    worker = _make_worker(session=session)

    async def scenario() -> None:
        media = {"data": b"x", "mime_type": "audio/pcm"}
        await asyncio.gather(*(worker._send_media(dict(media)) for _ in range(12)))

    asyncio.run(scenario())
    assert session.calls == 12
    assert session.peak == 1, f"sends overlapped (peak={session.peak}) — not serialised"


class RoutingProbeSession:
    """Records which typed realtime-input parameter each send used."""

    def __init__(self) -> None:
        self.kinds: list[str] = []

    async def send_realtime_input(self, **kwargs) -> None:
        self.kinds.append(next(iter(kwargs)) if kwargs else "none")


def test_media_routes_image_to_video_not_generic_audio_chunk() -> None:
    """Regression: a camera JPEG must be sent as a `video` frame, never the
    generic `media` chunk (which the native-audio model treats as audio and
    rejects with 1007 CONTENT_TYPE_AUDIO, dropping the whole live channel)."""
    session = RoutingProbeSession()
    worker = _make_worker(session=session)

    async def scenario() -> None:
        await worker._send_media({"data": b"a", "mime_type": "audio/pcm"})
        await worker._send_media({"data": b"i", "mime_type": "image/jpeg"})

    asyncio.run(scenario())
    assert session.kinds == ["audio", "video"], session.kinds


def test_send_after_close_is_a_noop() -> None:
    session = ConcurrencyProbeSession()
    worker = _make_worker(session=session, closed=True)
    asyncio.run(worker._send_media({"data": b"x", "mime_type": "audio/pcm"}))
    assert session.calls == 0, "a send on a closed transport must not touch the socket"


def test_send_when_stopping_is_a_noop() -> None:
    session = ConcurrencyProbeSession()
    worker = _make_worker(session=session, closed=False)
    worker.stop_event.set()
    asyncio.run(worker._send_media({"data": b"x", "mime_type": "audio/pcm"}))
    assert session.calls == 0


class HangingSession:
    """First send blocks forever (until cancelled); lets us cancel mid-send."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def send_realtime_input(self, **_kwargs) -> None:
        self.entered.set()
        await asyncio.Event().wait()  # never resolves


def test_cancellation_during_send_reraises_and_releases_lock() -> None:
    session = HangingSession()
    worker = _make_worker(session=session)

    async def scenario() -> None:
        task = asyncio.create_task(worker._send_media({"data": b"x", "mime_type": "audio/pcm"}))
        await session.entered.wait()          # in the middle of the send
        task.cancel()
        cancelled = False
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True
        assert cancelled, "cancellation must propagate, not be swallowed"
        # The lock must have been released so the channel stays usable.
        assert not worker._send_lock.locked(), "lock leaked after cancellation"

    asyncio.run(scenario())


class ClientContentSession:
    def __init__(self) -> None:
        self.turns = 0

    async def send_client_content(self, **_kwargs) -> None:
        self.turns += 1


def test_text_turn_routes_to_fallback_when_transport_closed() -> None:
    session = ClientContentSession()
    worker = _make_worker(session=session, closed=True)
    routed: list[str] = []

    async def fake_fallback(text: str, reason: str = "") -> None:
        routed.append(text)

    worker._submit_text_fallback = fake_fallback  # type: ignore[assignment]
    asyncio.run(worker._send_text_turn("what's the time?"))
    assert session.turns == 0, "must not send on a closed live transport"
    assert routed == ["what's the time?"], "turn should route to the text providers"


def test_text_turn_uses_live_channel_when_open() -> None:
    session = ClientContentSession()
    worker = _make_worker(session=session, closed=False)

    async def fake_fallback(text: str, reason: str = "") -> None:
        raise AssertionError("fallback must not run while the live channel is open")

    worker._submit_text_fallback = fake_fallback  # type: ignore[assignment]
    asyncio.run(worker._send_text_turn("hello"))
    assert session.turns == 1


def test_queued_media_does_not_cross_a_reconnect_or_shutdown_boundary():
    async def scenario():
        old = RoutingProbeSession()
        fresh = RoutingProbeSession()
        worker = _make_worker(session=old)
        await worker._send_lock.acquire()
        queued = asyncio.create_task(worker._send_media({"data": b"old turn", "mime_type": "audio/pcm"}))
        await asyncio.sleep(0)
        worker.session = fresh
        worker._send_lock.release()
        await queued
        assert old.kinds == [] and fresh.kinds == []
        await worker._send_media({"data": b"new turn", "mime_type": "audio/pcm"})
        assert fresh.kinds == ["audio"]
        await worker._send_lock.acquire()
        queued = asyncio.create_task(worker._send_media({"data": b"stop", "mime_type": "audio/pcm"}))
        await asyncio.sleep(0)
        worker.stop_event.set()
        worker._send_lock.release()
        await queued
        assert fresh.kinds == ["audio"]
        assert not worker._send_lock.locked()
    asyncio.run(scenario())


def test_unsupported_media_cannot_invalidate_the_live_channel():
    session = RoutingProbeSession()
    worker = _make_worker(session=session)
    assert asyncio.run(worker._send_media({"data": b"document", "mime_type": "application/pdf"})) is False
    assert asyncio.run(worker._send_media({"data": b"unknown"})) is False
    assert session.kinds == []


def test_an_old_sdk_never_relabels_an_image_as_generic_audio():
    class LegacySession:
        def __init__(self):
            self.generic = []

        async def send_realtime_input(self, *, media):
            self.generic.append(media)

    session = LegacySession()
    worker = _make_worker(session=session)
    assert asyncio.run(worker._send_media({"data": b"image", "mime_type": "image/jpeg"})) is False
    assert session.generic == []


def _tool_worker(session):
    worker = _make_worker(session=session)
    worker.mic = None
    worker.connected = True
    worker._turn_said = ""
    worker._mark_turn_progress = lambda: None
    worker._emit_state = lambda _state: None
    worker.speech = SimpleNamespace(output_active=lambda: False)
    worker._function_response = lambda call_id, name, payload: (call_id, name, payload)
    return worker


class _ToolSession(RoutingProbeSession):
    def __init__(self):
        super().__init__()
        self.responses = []

    async def send_tool_response(self, *, function_responses):
        self.responses.append(function_responses)


def test_tool_reply_rechecks_reconnect_and_cancellation_after_waiting_for_send_lock():
    async def scenario(reconnect):
        original = _ToolSession()
        replacement = _ToolSession()
        worker = _tool_worker(original)
        result_ready = asyncio.Event()

        async def result(_call):
            result_ready.set()
            return {"ok": True, "result": "done"}, None

        worker._run_tool_call = result
        call = SimpleNamespace(id="tool-1", name="web_search", args={})
        await worker._send_lock.acquire()
        task = asyncio.create_task(worker._handle_tool_call(SimpleNamespace(function_calls=[call])))
        await result_ready.wait()
        await asyncio.sleep(0)
        assert not task.done()
        if reconnect:
            worker.session = replacement
        else:
            worker._note_cancelled_tool_calls([call.id])
        worker._send_lock.release()
        await task
        assert original.responses == [] and replacement.responses == []

    asyncio.run(scenario(True))
    asyncio.run(scenario(False))


def test_late_tool_media_stays_with_the_channel_that_requested_it():
    async def scenario():
        original = _ToolSession()
        replacement = _ToolSession()
        worker = _tool_worker(original)
        started, finish = asyncio.Event(), asyncio.Event()

        async def result(_call):
            started.set()
            await finish.wait()
            return {"ok": True, "result": "screen"}, {"data": b"image", "mime_type": "image/jpeg"}

        worker._run_tool_call = result
        call = SimpleNamespace(id="tool-1", name="screen", args={})
        task = asyncio.create_task(worker._handle_tool_call(SimpleNamespace(function_calls=[call])))
        await started.wait()
        worker.session = replacement
        finish.set()
        await task
        assert original.kinds == replacement.kinds == []
        assert original.responses == replacement.responses == []

    asyncio.run(scenario())


def test_queued_old_tool_does_not_execute_after_reconnect():
    async def scenario():
        worker = _tool_worker(_ToolSession())
        worker._tool_call_lock = asyncio.Lock()
        await worker._tool_call_lock.acquire()
        ran = []

        async def handle(call):
            ran.append(call)

        worker._handle_tool_call = handle
        worker._start_tool_call(SimpleNamespace(function_calls=[]))
        await asyncio.sleep(0)
        queued = list(worker._tool_call_tasks)
        worker.session = _ToolSession()
        worker._tool_call_lock.release()
        await asyncio.gather(*queued)
        assert ran == []

    asyncio.run(scenario())


def test_go_away_keeps_a_resumption_update_in_the_same_message():
    class Session:
        async def receive(self):
            yield SimpleNamespace(
                go_away=SimpleNamespace(reason="scheduled handoff"),
                session_resumption_update=SimpleNamespace(resumable=True, new_handle="latest"))

    worker = _make_worker(session=Session())
    worker.conn_state = SimpleNamespace(set=lambda *_a: None)
    worker._resumption_handle = "older"
    asyncio.run(worker._receive_realtime())
    assert worker._resumption_handle == "latest"
    assert worker._last_close_reason == "go_away: scheduled handoff"


def test_cancelled_audio_message_still_completes_turn_and_unmutes_next_reply():
    class Session:
        async def receive(self):
            for chunk in (b"cancelled", b"next reply"):
                yield SimpleNamespace(
                    data=chunk,
                    server_content=SimpleNamespace(turn_complete=True, interrupted=False),
                    usage_metadata=None, go_away=None,
                    session_resumption_update=None, tool_call=None,
                )
            worker.stop_event.set()

    class Speech:
        def __init__(self):
            self.chunks = []
            self.finished = 0

        def enqueue_native_audio(self, chunk):
            self.chunks.append(chunk)

        def finish_native_audio(self):
            self.finished += 1

        def output_active(self):
            return False

    worker = _make_worker(session=Session())
    worker.speech = Speech()
    worker._drop_live_output = True
    worker._turn_active = True
    worker._pending_turn = "cancelled"
    worker._pending_turn_retries = 0
    worker._turn_progress_at = 0.0
    worker._trace = None
    worker._traced_playback = False
    worker.paused = False
    worker.connected = True
    worker.standby_mode = False
    worker._refresh_wake_window = lambda: None
    worker._record_live_usage = lambda _usage: None
    worker._emit_state = lambda _state: None

    asyncio.run(worker._receive_realtime())
    assert worker.speech.chunks == [b"next reply"]
    assert worker.speech.finished == 2
    assert worker._drop_live_output is False


def _audio_message_worker(messages, *, finish_speaking=True):
    """Exercise receive ordering without a renderer, microphone or provider."""
    class Session:
        async def receive(self):
            for message in messages:
                yield message
            worker.stop_event.set()

    class Speech:
        def __init__(self):
            self.events = []

        def enqueue_native_audio(self, chunk):
            self.events.append(("audio", chunk))

        def interrupt_all(self):
            self.events.append(("interrupt", None))

        def finish_native_audio(self):
            self.events.append(("finish", None))

        def output_active(self):
            return False

    worker = _make_worker(session=Session())
    worker.speech = Speech()
    worker._drop_live_output = False
    worker._turn_active = True
    worker._pending_turn = "synthetic request"
    worker._trace = None
    worker._traced_playback = False
    worker.paused = False
    worker.connected = True
    worker.standby_mode = False
    worker.finish_speaking = finish_speaking
    worker.conn_state = SimpleNamespace(set=lambda *_args: None)
    worker._voice_reply_allowed = lambda: True
    worker._refresh_wake_window = lambda: None
    worker._record_live_usage = lambda _usage: None
    worker._emit_state = lambda _state: None
    return worker


def test_go_away_delivers_bundled_final_audio_and_completion_before_reconnecting():
    message = SimpleNamespace(
        data=b"final syllable",
        go_away=SimpleNamespace(reason="scheduled handoff"),
        server_content=SimpleNamespace(turn_complete=True, interrupted=False),
        session_resumption_update=SimpleNamespace(resumable=True, new_handle="latest"),
    )
    worker = _audio_message_worker([message])
    asyncio.run(asyncio.wait_for(worker._receive_realtime(), timeout=1.0))
    assert worker.speech.events == [("audio", b"final syllable"), ("finish", None)]
    assert worker._pending_turn is None, "a completed answer must not replay after reconnect"
    assert worker._resumption_handle == "latest"
    assert worker._last_close_reason == "go_away: scheduled handoff"


def test_honoured_interruption_never_submits_cancelled_pcm_to_the_renderer():
    message = SimpleNamespace(
        data=b"cancelled response",
        server_content=SimpleNamespace(interrupted=True, turn_complete=True),
    )
    worker = _audio_message_worker([message], finish_speaking=False)
    asyncio.run(asyncio.wait_for(worker._receive_realtime(), timeout=1.0))
    assert worker.speech.events == [("interrupt", None), ("finish", None)]
    assert worker._pending_turn is None


def test_finish_speaking_preserves_pcm_bundled_with_ignored_interruption():
    message = SimpleNamespace(
        data=b"finish the sentence",
        server_content=SimpleNamespace(interrupted=True, turn_complete=True),
    )
    worker = _audio_message_worker([message], finish_speaking=True)
    asyncio.run(asyncio.wait_for(worker._receive_realtime(), timeout=1.0))
    assert worker.speech.events == [("audio", b"finish the sentence"), ("finish", None)]


def test_duplicate_session_loop_does_not_replace_the_active_microphone_or_queue(monkeypatch):
    import orion_core.live_worker as live

    stopped = []
    microphone = SimpleNamespace(stop=lambda **kwargs: stopped.append(kwargs))
    worker = _make_worker(session=object())
    worker.mic = microphone
    worker.out_queue = asyncio.Queue()
    original_queue = worker.out_queue
    worker._session_loop_active = True

    def forbidden(*args, **kwargs):
        raise AssertionError("a duplicate session must not create another microphone")

    monkeypatch.setattr(live, "MicrophoneEngine", forbidden)
    asyncio.run(asyncio.wait_for(worker._session_loop(), timeout=1.0))
    assert worker.mic is microphone
    assert worker.out_queue is original_queue
    assert not stopped
    assert worker._session_loop_active


def test_session_loop_after_stop_does_not_reopen_the_microphone(monkeypatch):
    import orion_core.live_worker as live

    worker = _make_worker(session=object())
    worker.mic = None
    worker.out_queue = asyncio.Queue()
    original_queue = worker.out_queue
    worker.stop_event.set()

    def forbidden(*args, **kwargs):
        raise AssertionError("a completed connection must not reopen capture after stop")

    monkeypatch.setattr(live, "MicrophoneEngine", forbidden)
    asyncio.run(asyncio.wait_for(worker._session_loop(), timeout=1.0))
    assert worker.mic is None
    assert worker.out_queue is original_queue
    assert not worker._session_loop_active


# ── a hung tool must not freeze the Live channel ─────────────────────────────

def test_a_tool_that_outlives_its_turn_is_answered_then_reported(monkeypatch):
    import orion_core.live_worker as lw

    finished = asyncio.Event()

    class Result:
        media = None

        def response_payload(self):
            return {"ok": True, "result": "Notepad saved."}

    class Dispatcher:
        async def dispatch_chain(self, name, args):
            await finished.wait()
            return Result()

    worker = _make_worker()
    worker.dispatcher = Dispatcher()
    said: list[str] = []
    worker.announce = said.append
    monkeypatch.setattr(lw.GenAILiveWorker, "LIVE_TOOL_ANSWER_S", 0.05)
    monkeypatch.setattr(lw, "_live_diag", lambda *a, **k: None)

    async def scenario():
        payload, media = await worker._run_tool_call(
            SimpleNamespace(name="desktop_control", args={}))
        # Running, not a success (Mark XXXVII): ok:True was spoken of as done.
        assert payload["status"] == "running" and "ok" not in payload
        assert "still running" in payload["result"]
        assert media is None
        finished.set()
        for _ in range(20):
            await asyncio.sleep(0)
        return said

    assert asyncio.run(scenario()) == ["The desktop control task has finished. Notepad saved."]


def test_a_quick_tool_is_answered_with_its_own_result(monkeypatch):
    import orion_core.live_worker as lw

    class Result:
        media = {"mime_type": "image/jpeg", "data": b"x"}

        def response_payload(self):
            return {"ok": True, "result": "done"}

    class Dispatcher:
        async def dispatch_chain(self, name, args):
            return Result()

    worker = _make_worker()
    worker.dispatcher = Dispatcher()
    monkeypatch.setattr(lw, "_live_diag", lambda *a, **k: None)
    payload, media = asyncio.run(worker._run_tool_call(SimpleNamespace(name="screen", args={})))
    assert payload == {"ok": True, "result": "done"} and media["data"] == b"x"


# ── a dropped channel is resumed, not benched ────────────────────────────────

def test_close_code_reads_the_websocket_code_off_an_api_error():
    from orion_core.live_worker import _close_code

    assert _close_code(SimpleNamespace(code=1006)) == 1006
    assert _close_code(SimpleNamespace(code=429)) is None
    assert _close_code(ValueError("x")) is None


def _run_worker_through(monkeypatch, failure, stopping=False, *, failures=1, session_ages=(),
                        api_key="AIza-test", configure=None):
    """Drive GenAILiveWorker.run() through one established session that ends
    with *failure*, and report what the recovery did."""
    import orion_core.live_worker as lw

    diags: list[tuple] = []
    monkeypatch.setattr(lw, "_live_diag", lambda event, **k: diags.append((event, k)))
    worker = _make_worker()
    worker._diags = diags
    worker._logs = []
    worker.bus = SimpleNamespace(log=SimpleNamespace(emit=worker._logs.append))
    worker._retry_delays = []
    worker._spoken = []
    worker._say = worker._spoken.append
    if session_ages:
        clock = [100.0]
        monkeypatch.setattr(lw, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    if failures > 1:
        async def retry_sleep(delay):
            worker._retry_delays.append(delay)
        monkeypatch.setattr(lw.asyncio, "sleep", retry_sleep)
    marked: list[str] = []
    profile = SimpleNamespace(name="gemini", model="models/live", api_key=api_key)
    worker.router = SimpleNamespace(
        live_profiles=lambda: [profile], active_key=lambda p: p.api_key,
        mark_failure=lambda p, exc: marked.append(p.name),
        has_text_fallback=lambda: False)

    class Session:
        pass

    class Connect:
        async def __aenter__(self):
            return Session()

        async def __aexit__(self, *exc):
            return False

    class Client:
        def __init__(self, **kwargs):
            worker._client_api_version = kwargs["http_options"]["api_version"]
            self.aio = SimpleNamespace(live=SimpleNamespace(
                connect=lambda model, config: Connect()))

    monkeypatch.setattr(lw.genai, "Client", Client)

    class ConnState:
        provider = ""

        def set(self, *a, **k):
            pass

        def in_reconnect_loop(self):
            return False

    worker.conn_state = ConnState()
    worker.speech = SimpleNamespace(start=lambda: None, stop=lambda: None)
    worker._turn_watchdog_task = object()
    worker.wake_mode_enabled = False
    worker._warn_if_key_malformed = lambda: None
    worker._emit_state = lambda _s: None
    worker._build_config = lambda: None
    worker._live_model_shift = {}
    worker._search_tool_enabled = False
    worker._resumption_handle = "handle-1"
    worker._no_live_notice_sent = False
    worker.mic = None
    worker._ensure_fallback_mic = lambda: None
    worker._offline_voice_ready = lambda: False
    worker.paused = False
    worker.dispatcher = SimpleNamespace(TOOL_DECLARATIONS=[])

    async def redrive():
        pass

    failed_sessions = 0

    async def session_loop():
        nonlocal failed_sessions
        if session_ages:
            clock[0] += session_ages[failed_sessions]
        failed_sessions += 1
        if failed_sessions >= failures:
            worker.stop_event.set()
        worker._stopping = stopping
        raise failure

    worker._redrive_pending_turn = redrive
    worker._session_loop = session_loop
    if configure is not None:
        configure(worker, Session)
    asyncio.run(worker.run())
    return worker, marked


def test_repeated_server_errors_back_off_after_a_successful_handshake(monkeypatch):
    # The 8 Oct 23:26:54 owner error resumed at 23:26:55. A handshake
    # is not proof that the server recovered: short sessions must not reset
    # the failure history and spin forever at a half-second retry interval.
    drop = RuntimeError("1011 None. Internal error encountered.")
    drop.code = 1011
    worker, marked = _run_worker_through(monkeypatch, drop, failures=9)
    assert marked == []
    assert worker._resumption_handle == "handle-1"
    assert worker._retry_delays[0] == 0.5
    assert worker._retry_delays[1] >= 2.0
    assert worker._retry_delays[-1] == 12.0
    assert all(a <= b for a, b in zip(worker._retry_delays, worker._retry_delays[1:]))
    assert len(worker._spoken) == 1


def test_a_recoverable_server_close_keeps_diagnostics_without_screen_tracebacks(monkeypatch):
    drop = RuntimeError("1011 None. Internal error encountered.")
    drop.code = 1011
    worker, marked = _run_worker_through(monkeypatch, drop)
    assert marked == []
    [record] = [fields for event, fields in worker._diags if event == "error"]
    assert record["code"] == 1011 and record["established"]
    assert "Internal error" in record["summary"]
    assert not any("exception trace" in line for line in worker._logs)
    assert any("reconnecting" in line or "resuming" in line for line in worker._logs)


def test_a_stable_session_restores_the_fast_resume_budget(monkeypatch):
    drop = RuntimeError("1011 None. Internal error encountered.")
    drop.code = 1011
    worker, marked = _run_worker_through(
        monkeypatch, drop, failures=4, session_ages=(0.1, 0.1, 31.0, 0.1))
    assert marked == []
    assert worker._retry_delays == [0.5, 2.0, 0.5]
    assert worker._resumption_handle == "handle-1"
    assert worker._spoken == []


def test_an_unexpected_live_exception_still_reports_its_trace(monkeypatch):
    worker, marked = _run_worker_through(monkeypatch, ValueError("bad application state"))
    assert marked == ["gemini"]
    assert any("exception trace" in line for line in worker._logs)


def test_an_abnormal_close_keeps_the_conversation_and_the_provider(monkeypatch):
    # Measured on the real server: a dropped socket surfaces as APIError 1006.
    # It used to bench Live for 45 s AND discard the resumption handle, so
    # ORION came back (late) with no memory of the conversation.
    drop = RuntimeError("1006 None. abnormal closure [internal]")
    drop.code = 1006
    worker, marked = _run_worker_through(monkeypatch, drop)
    assert marked == []
    assert worker._resumption_handle == "handle-1"


def test_the_close_at_shutdown_is_not_a_provider_fault(monkeypatch):
    # 7 Oct 22:15: "1000 None." journalled as an error seconds after "shut down".
    close = RuntimeError("1000 None.")
    close.code = 1000
    worker, marked = _run_worker_through(monkeypatch, close, stopping=True)
    assert marked == []
    assert [e for e, _k in worker._diags if e in ("error", "closed")] == ["closed"]
    assert worker._diags[-1][1]["reason"] == "shutdown"


def test_a_real_provider_fault_still_cools_the_provider(monkeypatch):
    fault = RuntimeError("quota exhausted for this project")
    fault.code = 429
    worker, marked = _run_worker_through(monkeypatch, fault)
    assert marked == ["gemini"]
    assert worker._resumption_handle is None


def test_a_mid_session_not_found_reconnects_on_the_same_model(monkeypatch):
    # MEASURED 2026-09-26: "1008 Requested entity was not found" 171 s into a
    # session walked ORION through two retired models before he came back.
    lost = RuntimeError("1008 None. Requested entity was not found.")
    lost.code = 1008
    worker, marked = _run_worker_through(monkeypatch, lost)
    assert marked == []
    assert worker._live_model_shift == {}
    assert worker._resumption_handle is None


def test_a_mid_session_credential_drop_reconnects_without_blaming_the_key(monkeypatch):
    # Mark XXXX.4 - 26 Oct 14:17: 384 s into a working session the channel closed
    # 1008 "missing required authentication credential"; the same key worked
    # 7 s later. ORION had said aloud that the key should be replaced.
    lapsed = RuntimeError("1008 None. Request is missing required authentication credential.")
    lapsed.code = 1008
    worker, marked = _run_worker_through(monkeypatch, lapsed)
    assert marked == []
    assert worker._resumption_handle is None
    assert any("credential lapsed mid-conversation" in line for line in worker._logs)
    assert not any("key" in line.lower() for line in worker._spoken)


def test_payload_refusals_recover_without_resuming_the_invalid_state(monkeypatch):
    # 13 Oct: each 1007 was logged as "fresh" but reused the same handle,
    # followed by another 1007 in the next resumed session.
    for code in (1007, 1009):
        fault = RuntimeError(f"{code} invalid frame payload")
        fault.code = code
        worker, marked = _run_worker_through(monkeypatch, fault)
        assert marked == []
        assert worker._resumption_handle is None


def test_authorization_api_keys_use_the_regular_live_api(monkeypatch):
    fault = RuntimeError("1006 socket closed")
    fault.code = 1006
    # The expressive voice (affective dialog) is a v1alpha feature (Mark XXXX.1);
    # without it a permanent key keeps the regular API.
    monkeypatch.setenv("ORION_EXPRESSIVE_VOICE", "0")
    worker, _ = _run_worker_through(monkeypatch, fault, api_key="AQ.authorization-test")
    assert worker._client_api_version == "v1beta"
    worker._warn_if_key_malformed = GenAILiveWorker._warn_if_key_malformed.__get__(worker)
    worker._logs.clear()
    worker._warn_if_key_malformed()
    assert worker._logs == []  # a permanent AQ key must not be called expired/malformed


def test_true_ephemeral_tokens_use_the_sdk_constrained_api(monkeypatch):
    fault = RuntimeError("1006 socket closed")
    fault.code = 1006
    worker, _ = _run_worker_through(monkeypatch, fault, api_key="auth_tokens/test-token")
    assert worker._client_api_version == "v1alpha"
    worker._warn_if_key_malformed = GenAILiveWorker._warn_if_key_malformed.__get__(worker)
    worker._logs.clear()
    worker._warn_if_key_malformed()
    assert len(worker._logs) == 1 and "short-lived" in worker._logs[0]
    assert "test-token" not in worker._logs[0]


def test_live_candidates_come_from_the_api_in_the_same_family():
    import orion_core.live_worker as lw

    class Model:
        def __init__(self, name, actions=("bidiGenerateContent",)):
            self.name, self.supported_actions = name, list(actions)

    class Client:
        class models:
            @staticmethod
            def list():
                return [Model("models/gemini-3.8-live"),
                        Model("models/gemini-2.5-flash-native-audio-latest"),
                        Model("models/gemini-3.5-transcribe-live"),
                        Model("models/gemini-2.5-flash", ("generateContent",))]

    worker = _make_worker()
    profile = SimpleNamespace(name="gemini",
                              model="models/gemini-2.5-flash-native-audio-preview-12-2025")
    got = asyncio.run(lw.GenAILiveWorker._live_candidates(worker, profile, Client()))
    assert got == ["models/gemini-2.5-flash-native-audio-preview-12-2025",
                   "models/gemini-2.5-flash-native-audio-latest",
                   "models/gemini-3.8-live"]


def test_guided_samples_are_held_from_live_audio_but_camera_frames_are_kept(monkeypatch):
    from orion_core import voice_enrolment

    monkeypatch.setattr(voice_enrolment, "holding_model", lambda: True)
    worker = _make_worker(session=object())
    worker.out_queue = asyncio.Queue()
    sent = []

    async def send(media):
        sent.append(media)
        worker.stop_event.set()

    worker._send_media = send
    worker.out_queue.put_nowait({"mime_type": "audio/pcm", "data": b"practice"})
    worker.out_queue.put_nowait({"mime_type": "image/jpeg", "data": b"camera"})
    asyncio.run(worker._send_realtime())
    assert sent == [{"mime_type": "image/jpeg", "data": b"camera"}]


def test_speaker_context_uses_the_send_lock_without_starting_another_reply():
    session = ConcurrencyProbeSession()
    notes = []

    async def send(**kwargs):
        notes.append(kwargs)
        await session.send_realtime_input()

    session.send_client_content = send
    worker = _make_worker(session=session)

    async def scenario():
        await asyncio.gather(worker._send_listener_note("owner note"),
                             worker._send_listener_note("other note"))

    asyncio.run(scenario())
    assert session.peak == 1
    assert [note["turn_complete"] for note in notes] == [False, False]


def test_confident_nonowner_cannot_dispatch_tools_but_typed_requests_still_work(monkeypatch):
    from orion_core import speaker_gate, voice_enrolment
    from orion_core.data import ToolResult

    monkeypatch.setattr(voice_enrolment, "_TURNS", SimpleNamespace(reply_allowed=lambda: False))
    worker = _make_worker()
    called = []

    async def dispatch(name, args):
        called.append((name, args))
        return ToolResult("done")

    worker.dispatcher = SimpleNamespace(dispatch_chain=dispatch)
    payload, media = asyncio.run(worker._dispatch_live_tool("diagnostics", {}))
    assert not payload["ok"] and media is None and called == []
    worker._typed_live_turn = True
    monkeypatch.setattr(speaker_gate, "voice_turn", lambda: (_ for _ in ()).throw(
        AssertionError("typed input was treated as a voice confirmation")))
    payload, _media = asyncio.run(worker._dispatch_live_tool("diagnostics", {}))
    assert payload["ok"] and called == [("diagnostics", {})]


def test_practice_commands_do_not_act_and_explicit_cancel_remains_usable(monkeypatch):
    from orion_core import voice_enrolment

    cancelled = []
    session = SimpleNamespace(listening=True, cancel=lambda: cancelled.append(True))
    monkeypatch.setattr(voice_enrolment, "_SESSION", session)
    worker = _make_worker()
    worker._strip_wake_words = lambda text: text.removeprefix("orion ")
    worker._on_local_transcript("Orion shut down now")
    assert cancelled == []
    worker._on_local_transcript("Orion cancel voice learning")
    assert cancelled == [True]

# -- Reconnect continuity continuation (13 October) --
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from orion_core import session_context as _continuity_context
from orion_core.live_worker import GenAILiveWorker
from orion_core.memory import OrionMemoryMatrix


class _ContinuityLog:
    def emit(self, *_args):
        pass


class _ContinuityStore:
    def list(self, **kwargs):
        return []

    def pending_approvals(self):
        return []


def _continuity_memory(rows=()):
    return SimpleNamespace(recall_episodes=lambda *_args, **_kwargs: list(rows))


def test_fresh_brief_preserves_real_recent_request_and_running_work():
    now = time.time()
    ledger = _continuity_context.ActivityLedger(clock_fn=lambda: now-2)
    ledger.started('research', {'topic': 'federal law'})
    brief = _continuity_context.build_brief(_continuity_memory(), now=now, store=_ContinuityStore(), ledger=ledger,
                                interrupted='Research federal law', reason='reconnect')
    assert 'federal law' in brief.text
    assert 'still running' in brief.text
    assert 'fresh session' in brief.text
    assert brief.tokens <= _continuity_context.MAX_TOKENS


def test_resumed_brief_has_only_changes_since_saved_point():
    now = time.time()
    rows = [dict(role='user', content='newer request', created_at=now-2),
            dict(role='user', content='old request', created_at=now-20)]
    brief = _continuity_context.build_update(_continuity_memory(rows), since=now-10, now=now, store=_ContinuityStore())
    assert 'newer request' in brief.text
    assert 'old request' not in brief.text
    empty = _continuity_context.build_update(_continuity_memory(), since=now, now=now, store=_ContinuityStore())
    assert empty.text == ''


@pytest.mark.parametrize('delay', [0.25, 1.0, 2.0])
def test_resumed_brief_preserves_completed_request_just_after_saved_point(delay):
    saved = 1_700_000_000.0
    rows = [
        dict(role='assistant', content='The receiver uses a directional antenna.',
             created_at=saved + delay + 0.1),
        dict(role='user', content='Explain the radio receiver I just assembled.',
             created_at=saved + delay),
        dict(role='user', content='The already restored request.', created_at=saved),
    ]
    brief = _continuity_context.build_update(
        _continuity_memory(rows), since=saved, now=saved + 10,
        store=_ContinuityStore(), interrupted='')
    assert 'Explain the radio receiver I just assembled.' in brief.text
    assert 'The receiver uses a directional antenna.' in brief.text
    assert 'The already restored request.' not in brief.text
    assert brief.counts['turns'] == 2
    assert brief.tokens <= _continuity_context.MAX_TOKENS


def test_resumed_brief_preserves_durable_iso_turn_one_second_after_saved_point():
    from datetime import datetime, timezone
    saved = 1_700_000_000.0
    stamp = datetime.fromtimestamp(saved + 1, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    rows = [dict(role='user', content='Review the completed notebook analysis.',
                 created_at=stamp)]
    brief = _continuity_context.build_update(
        _continuity_memory(rows), since=saved, now=saved + 10,
        store=_ContinuityStore())
    assert 'Review the completed notebook analysis.' in brief.text
    assert brief.counts['turns'] == 1


def test_resumed_brief_still_excludes_turns_at_or_before_saved_point():
    saved = 1_700_000_000.0
    rows = [dict(role='user', content='Restored boundary turn.', created_at=saved),
            dict(role='assistant', content='Restored earlier answer.', created_at=saved - 0.1)]
    brief = _continuity_context.build_update(
        _continuity_memory(rows), since=saved, now=saved + 10,
        store=_ContinuityStore())
    assert brief.kind == 'none' and brief.text == ''


def test_resumed_brief_deduplicates_pending_request_in_early_completed_episode():
    saved = 1_700_000_000.0
    request = 'Explain the radio receiver I just assembled.'
    rows = [dict(role='user', content=request, created_at=saved + 1)]
    brief = _continuity_context.build_update(
        _continuity_memory(rows), since=saved, now=saved + 10,
        interrupted=request.lower(), store=_ContinuityStore())
    assert request in brief.text
    assert brief.counts['turns'] == 1
    assert _continuity_context.INTERRUPTED_NOTE not in brief.text


def test_brief_bounds_and_redacts_credentials():
    rows = [dict(role='user', content='Bearer abc123secret '+('x'*2000), created_at=time.time())]*40
    brief = _continuity_context.build_brief(_continuity_memory(rows), store=_ContinuityStore())
    assert len(brief.text) <= _continuity_context.MAX_CHARS
    assert 'abc123secret' not in brief.text


def _continuity_worker():
    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    worker.bus = SimpleNamespace(log=_ContinuityLog())
    worker.memory = _continuity_memory()
    worker._resumption_handle = None
    worker._send_closed = False
    worker.stop_event = asyncio.Event()
    worker._send_lock = asyncio.Lock()
    worker.session = None
    worker.paused = False
    worker.connected = False
    worker.mic = None
    worker._turn_active = False
    worker._turn_said = ""
    worker.tool_busy = False
    worker.speech = SimpleNamespace(output_active=lambda: False, stop=lambda: None)
    worker._emit_state = lambda *_args: None
    worker._said_by_me = lambda _text: False
    worker.observe_for_reflex = lambda *_args, **_kwargs: None
    worker.observe_resolver_shadow = lambda *_args: None
    worker.announcements = []
    worker.announce = worker.announcements.append
    return worker


def test_context_preparation_fresh_resumed_and_changed_model(monkeypatch):
    monkeypatch.setattr(_continuity_context, '_store', lambda store: _ContinuityStore())
    worker = _continuity_worker()
    worker._pending_turn = 'finish my research'
    async def scenario():
        await worker._prepare_session_context('model-a')
        assert 'finish my research' in worker._session_context_text
        worker._resumption_handle = 'saved'
        worker._resumption_at = time.time()-10
        await worker._prepare_session_context('model-a')
        assert 'finish my research' in worker._session_context_text
        await worker._prepare_session_context('model-b')
        assert worker._resumption_handle is None
        assert 'different voice model' in worker._session_context_text
    asyncio.run(scenario())


def test_context_note_does_not_request_a_reply_or_send_when_closed():
    worker = _continuity_worker()
    calls = []
    async def send(**kwargs):
        calls.append(kwargs)
    worker.session = SimpleNamespace(send_client_content=send)
    async def scenario():
        await worker._send_context_note('background result')
        worker._send_closed = True
        await worker._send_context_note('ignored')
    asyncio.run(scenario())
    assert len(calls) == 1
    assert calls[0]['turn_complete'] is False


def test_resumed_startup_context_note_retry_preserves_initial_and_late_order(monkeypatch):
    from collections import deque
    initial = '[CONTEXT UPDATE - completed receiver request since the saved handle]'
    late = '[Background task result: antenna analysis finished]'
    delivered = []
    attempts = []
    drop = RuntimeError('1006 controlled session end')
    drop.code = 1006

    def configure(worker, session_type):
        worker._pending_context_notes = deque([late], maxlen=24)

        async def prepare(_model):
            worker._session_context_text = initial

        async def send(_session, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise ConnectionError('controlled initial context send failure')
            delivered.append(kwargs)

        async def session_loop():
            await worker._flush_context_notes()
            worker.stop_event.set()
            raise drop

        worker._prepare_session_context = prepare
        worker._session_loop = session_loop
        session_type.send_client_content = send

    worker, marked = _run_worker_through(monkeypatch, drop, configure=configure)
    assert marked == []
    assert [item['turns']['parts'][0]['text'] for item in delivered] == [initial, late]
    assert all(item['turn_complete'] is False for item in attempts)
    assert not worker._pending_context_notes


def test_resumed_startup_context_note_survives_failed_startup_flush(monkeypatch):
    from collections import deque
    initial = '[CONTEXT UPDATE - the completed research answer is not in the saved handle]'
    delivered = []
    queued_at_session_start = []
    healthy = False
    drop = RuntimeError('1006 controlled session end')
    drop.code = 1006

    def configure(worker, session_type):
        worker._pending_context_notes = deque(maxlen=24)

        async def prepare(_model):
            worker._session_context_text = initial

        async def send(_session, **kwargs):
            if not healthy:
                raise ConnectionError('controlled context transport failure')
            delivered.append(kwargs)

        async def session_loop():
            nonlocal healthy
            queued_at_session_start.extend(worker._pending_context_notes)
            healthy = True
            await worker._flush_context_notes()
            await worker._flush_context_notes()
            worker.stop_event.set()
            raise drop

        worker._prepare_session_context = prepare
        worker._session_loop = session_loop
        session_type.send_client_content = send

    worker, marked = _run_worker_through(monkeypatch, drop, configure=configure)
    assert marked == []
    assert queued_at_session_start == [initial]
    assert [item['turns']['parts'][0]['text'] for item in delivered] == [initial]
    assert delivered[0]['turn_complete'] is False
    assert not worker._pending_context_notes


def test_resumed_startup_context_note_success_is_delivered_once(monkeypatch):
    from collections import deque
    initial = '[CONTEXT UPDATE - completed receiver request]'
    late = '[Background task result: antenna analysis finished]'
    delivered = []
    drop = RuntimeError('1006 controlled session end')
    drop.code = 1006

    def configure(worker, session_type):
        worker._pending_context_notes = deque([late], maxlen=24)

        async def prepare(_model):
            worker._session_context_text = initial

        async def send(_session, **kwargs):
            delivered.append(kwargs)

        worker._prepare_session_context = prepare
        session_type.send_client_content = send

    worker, marked = _run_worker_through(monkeypatch, drop, configure=configure)
    assert marked == []
    assert [item['turns']['parts'][0]['text'] for item in delivered] == [initial, late]
    assert all(item['turn_complete'] is False for item in delivered)
    assert not worker._pending_context_notes


@pytest.mark.parametrize('fail_initial', [True, False])
def test_resumed_startup_queue_insertion_cannot_displace_inflight_head(monkeypatch, fail_initial):
    from collections import deque
    initial = '[CONTEXT UPDATE - completed receiver request]'
    late = '[Background task result: antenna analysis finished]'
    delivered = []
    drop = RuntimeError('1006 controlled session end')
    drop.code = 1006

    def configure(worker, session_type):
        worker._pending_context_notes = deque([late], maxlen=24)
        head_selected = asyncio.Event()
        original_send_note = worker._send_context_note
        first_initial = True
        in_flight_flush = None

        async def observe_send_note(text):
            if text == late:
                # The real flusher selected its existing deque head and owns
                # the flush lock. Its actual send now waits for the send lock.
                head_selected.set()
            return await original_send_note(text)

        async def prepare(_model):
            worker._session_context_text = initial

        async def send(_session, **kwargs):
            nonlocal first_initial, in_flight_flush
            text = kwargs['turns']['parts'][0]['text']
            if text == initial and first_initial:
                first_initial = False
                in_flight_flush = asyncio.create_task(worker._flush_context_notes())
                await head_selected.wait()
                if fail_initial:
                    raise ConnectionError('controlled initial send failure while late flush is queued')
            delivered.append(kwargs)

        async def session_loop():
            assert in_flight_flush is not None
            await in_flight_flush
            await worker._flush_context_notes()
            worker.stop_event.set()
            raise drop

        worker._prepare_session_context = prepare
        worker._send_context_note = observe_send_note
        worker._session_loop = session_loop
        session_type.send_client_content = send

    worker, marked = _run_worker_through(monkeypatch, drop, configure=configure)
    texts = [item['turns']['parts'][0]['text'] for item in delivered]
    assert marked == []
    assert texts.count(initial) == 1
    assert texts.count(late) == 1
    assert all(item['turn_complete'] is False for item in delivered)
    assert not worker._pending_context_notes


def test_context_note_rechecks_session_identity_after_send_lock_wait():
    worker = _continuity_worker()
    first, second = _ContinuitySession(), _ContinuitySession()
    worker.session = first

    async def scenario():
        await worker._send_lock.acquire()
        operation = asyncio.create_task(worker._send_context_note('old session background'))
        await _continuity_settle(lambda: bool(worker._send_lock._waiters))
        worker.session = second
        worker._send_lock.release()
        assert await operation is False
        assert first.notes == second.notes == []
        assert not worker._send_lock.locked()

    asyncio.run(scenario())


def test_context_note_flush_keeps_item_until_new_session_explicit_retry():
    from collections import deque
    worker = _continuity_worker()
    first, second = _ContinuitySession(), _ContinuitySession()
    worker.session = first
    note = 'completed background result, context only'
    worker._pending_context_notes = deque([note], maxlen=24)

    async def scenario():
        await worker._send_lock.acquire()
        operation = asyncio.create_task(worker._flush_context_notes())
        await _continuity_settle(lambda: bool(worker._send_lock._waiters))
        worker.session = second
        worker._send_lock.release()
        await operation
        assert first.notes == second.notes == []
        assert list(worker._pending_context_notes) == [note]
        await worker._flush_context_notes()
        await worker._flush_context_notes()
        assert first.notes == []
        assert len(second.notes) == 1
        assert second.notes[0]['turns']['parts'][0]['text'] == note
        assert second.notes[0]['turn_complete'] is False
        assert not worker._pending_context_notes

    asyncio.run(scenario())


def test_context_note_flush_cancellation_keeps_item_and_releases_lock():
    from collections import deque
    worker = _continuity_worker()
    note = 'completed research answer, context only'
    worker._pending_context_notes = deque([note], maxlen=24)

    async def scenario():
        entered = asyncio.Event()

        async def send(**_kwargs):
            entered.set()
            await asyncio.Event().wait()

        worker.session = SimpleNamespace(send_client_content=send)
        operation = asyncio.create_task(worker._flush_context_notes())
        await entered.wait()
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert list(worker._pending_context_notes) == [note]
        assert not worker._send_lock.locked()
        assert not worker._context_flush_lock.locked()
        worker.session = _ContinuitySession()
        await worker._flush_context_notes()
        assert len(worker.session.notes) == 1
        assert worker.session.notes[0]['turn_complete'] is False
        assert not worker._pending_context_notes

    asyncio.run(scenario())


def test_tool_activity_stays_running_until_detached_task_finishes():
    worker = _continuity_worker()
    async def scenario():
        event = asyncio.Event()
        async def dispatch(*_args):
            await event.wait()
            return {'ok': False, 'result': 'source unavailable'}, None
        worker._dispatch_live_tool = dispatch
        task = asyncio.create_task(worker._tracked_live_tool('research', {'topic': 'law'}))
        await asyncio.sleep(0)
        assert worker._activity_ledger.snapshot()[0].running
        event.set()
        await task
        done = worker._activity_ledger.snapshot()[0]
        assert not done.running and done.ok is False
        assert done.outcome == 'source unavailable'
    asyncio.run(scenario())


def test_scene_memory_persists_text_recall_forget_and_existing_episodes(tmp_path):
    matrix = OrionMemoryMatrix(tmp_path/'memory.db', tmp_path, SimpleNamespace(log=_ContinuityLog()))
    matrix.log_episode('user', 'preserve my existing conversation')
    now = time.time()
    data = dict(summary='A blue notebook on the desk', epoch=now)
    matrix.save_scene(captured_at='today', epoch=now, summary=data['summary'],
                      searchable='blue notebook desk', record=json.dumps(data))
    matrix.close()
    matrix = OrionMemoryMatrix(tmp_path/'memory.db', tmp_path, SimpleNamespace(log=_ContinuityLog()))
    try:
        rows = matrix.scene_memories('notebook')
        assert len(rows) == 1
        assert json.loads(rows[0]['record'])['summary'] == data['summary']
        assert not matrix.scene_memories('unrelated')
        assert matrix.forget_scenes(since=now-1) == 1
        assert not matrix.scene_memories()
        assert matrix.recall_episodes('preserve')[0]['content'] == 'preserve my existing conversation'
    finally:
        matrix.close()


def test_scene_memory_rejects_raw_image_data(tmp_path):
    matrix = OrionMemoryMatrix(tmp_path/'memory.db', tmp_path, SimpleNamespace(log=_ContinuityLog()))
    try:
        with pytest.raises(ValueError, match='not images'):
            matrix.save_scene(captured_at='today', epoch=time.time(), record='{"image":"base64 pixels"}')
        assert not matrix.scene_memories()
    finally:
        matrix.close()


class _ContinuitySession:
    def __init__(self):
        self.notes = []
        self.responses = []

    async def send_client_content(self, **kwargs):
        self.notes.append(kwargs)

    async def send_tool_response(self, **kwargs):
        self.responses.append(kwargs)


async def _continuity_settle(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    assert predicate(), "expected async progress did not occur"


def test_reconnect_waits_for_the_same_request_but_replays_an_unrelated_one():
    worker = _continuity_worker()
    worker.session = _ContinuitySession()
    sent, fallback = [], []
    async def send(text):
        sent.append(text)
    async def text_fallback(text, **kwargs):
        fallback.append(text)
    worker._send_text_turn = send
    worker._submit_text_fallback = text_fallback
    worker.router = SimpleNamespace(has_text_fallback=lambda: True)
    worker._activity_ledger = _continuity_context.ActivityLedger()
    worker._mark_turn("save the document")
    old = worker._activity_ledger.started("file_controller", {"action": "write"}, request_id=worker._turn_request_id)
    async def scenario():
        worker._pending_turn_retries = 1
        await worker._redrive_pending_turn()
        assert sent == fallback == []
        assert worker._pending_turn == "save the document"
        worker._mark_turn("what time is it")
        await worker._redrive_pending_turn()
        assert sent == ["what time is it"]
        assert old.running
    asyncio.run(scenario())


def test_watchdog_does_not_repeat_running_actions(monkeypatch):
    import orion_core.live_worker as live
    worker = _continuity_worker()
    worker._activity_ledger = _continuity_context.ActivityLedger()
    worker._mark_turn("save the document")
    worker._activity_ledger.started("file_controller", request_id=worker._turn_request_id)
    worker._turn_progress_at = time.monotonic() - 200
    worker.router = SimpleNamespace(has_text_fallback=lambda: True)
    async def fallback(*args, **kwargs):
        pytest.fail("watchdog must not repeat running actions")
    async def tick(_seconds):
        worker.stop_event.set()
    worker._submit_text_fallback = fallback
    monkeypatch.setattr(live.asyncio, "sleep", tick)
    asyncio.run(worker._turn_watchdog())
    assert worker._pending_turn == "save the document"


def test_fast_tool_finishing_after_reconnect_delivers_context_once():
    worker = _continuity_worker()
    first, second = _ContinuitySession(), _ContinuitySession()
    worker.session = first
    worker._mark_turn("research radio astronomy")
    async def scenario():
        release, entered = asyncio.Event(), asyncio.Event()
        async def dispatch(*args):
            entered.set()
            await release.wait()
            return {"ok": True, "result": "The telescope needs calibration."}, None
        worker._dispatch_live_tool = dispatch
        operation = asyncio.create_task(worker._run_tool_call(SimpleNamespace(name="research", args={})))
        await entered.wait()
        inner = next(iter(worker._live_tool_tasks))
        worker.session = second
        await worker._redrive_pending_turn()
        release.set()
        payload, _media = await operation
        assert payload["result"] == "The telescope needs calibration."
        await worker._flush_context_notes()
        worker._report_detached_tool("research", inner)
        await worker._flush_context_notes()
        assert len(second.notes) == 1
        assert second.notes[0]["turn_complete"] is False
        assert "calibration" in second.notes[0]["turns"]["parts"][0]["text"]
        assert len(worker.announcements) == 1
        assert worker._pending_turn is None
    asyncio.run(scenario())


def test_detached_result_queues_while_disconnected_and_is_not_repeated():
    worker = _continuity_worker()
    worker.session = _ContinuitySession()
    worker.LIVE_TOOL_ANSWER_S = 0.001
    async def scenario():
        release = asyncio.Event()
        async def dispatch(*args):
            await release.wait()
            return {"ok": True, "result": "Analysis complete."}, None
        worker._dispatch_live_tool = dispatch
        payload, _media = await worker._run_tool_call(SimpleNamespace(name="research", args={}))
        assert "still running" in payload["result"]
        inner = next(iter(worker._live_tool_tasks))
        worker.session = None
        release.set()
        await inner
        await _continuity_settle(lambda: bool(getattr(worker, "_pending_context_notes", None)))
        worker.session = _ContinuitySession()
        await asyncio.gather(worker._flush_context_notes(), worker._flush_context_notes())
        assert len(worker.session.notes) == 1
        assert len(worker.announcements) == 1
    asyncio.run(scenario())


def test_stale_queued_tool_message_is_skipped_before_any_dispatch():
    worker = _continuity_worker()
    worker.session = _ContinuitySession()
    async def forbidden(*args, **kwargs):
        pytest.fail("stale queued message must not execute")
    worker._handle_tool_call = forbidden
    async def scenario():
        worker._tool_call_lock = asyncio.Lock()
        await worker._tool_call_lock.acquire()
        worker._start_tool_call(SimpleNamespace(function_calls=[SimpleNamespace(name="write_file", args={}, id="old")]))
        tasks = list(worker._tool_call_tasks)
        worker.session = _ContinuitySession()
        worker._tool_call_lock.release()
        await asyncio.gather(*tasks)
        assert worker.session.responses == []
    asyncio.run(scenario())


def test_retired_queued_request_can_replay_if_no_action_started():
    worker = _continuity_worker()
    worker.session = _ContinuitySession()
    worker._mark_turn("save the document")
    replayed = []
    async def send(text):
        replayed.append(text)
    worker._send_text_turn = send
    worker.router = SimpleNamespace(has_text_fallback=lambda: False)
    async def scenario():
        worker._tool_call_lock = asyncio.Lock()
        await worker._tool_call_lock.acquire()
        worker._start_tool_call(SimpleNamespace(function_calls=[]))
        tasks = list(worker._tool_call_tasks)
        worker.session = _ContinuitySession()
        await worker._redrive_pending_turn()
        assert replayed == []
        worker._tool_call_lock.release()
        await asyncio.gather(*tasks)
        await _continuity_settle(lambda: bool(replayed))
        assert replayed == ["save the document"]
    asyncio.run(scenario())


def test_session_identity_is_rechecked_after_waiting_for_send_lock():
    worker = _continuity_worker()
    first, second = _ContinuitySession(), _ContinuitySession()
    worker.session = first
    worker._function_response = lambda call_id, name, payload: (call_id, name, payload)
    async def dispatch(*args):
        return {"ok": True, "result": "The file was saved."}, None
    worker._dispatch_live_tool = dispatch
    call = SimpleNamespace(function_calls=[SimpleNamespace(name="web_search", id="old-id", args={})])
    async def scenario():
        await worker._send_lock.acquire()
        operation = asyncio.create_task(worker._handle_tool_call(call))
        await _continuity_settle(lambda: bool(worker._send_lock._waiters))
        worker.session = second
        worker._send_lock.release()
        await operation
        await worker._flush_context_notes()
        assert first.responses == second.responses == []
        assert len(second.notes) == 1
    asyncio.run(scenario())


def test_interrupted_voice_buffer_is_persisted_once_and_used_in_next_brief(monkeypatch):
    monkeypatch.setattr(_continuity_context, "_store", lambda store: _ContinuityStore())
    worker = _continuity_worker()
    persisted = []
    worker._persist_episode = lambda role, text: persisted.append((role, text))
    async def interrupted(buffer):
        buffer.extend(["Research ", "federal law"])
        return
    worker._receive_realtime_stream = interrupted
    async def scenario():
        worker._mark_turn("Research federal law")
        await worker._receive_realtime()
        await worker._receive_realtime()
        assert persisted == [("user", "Research federal law")]
        worker._pending_turn = None
        await worker._prepare_session_context("model-a")
        assert "Research federal law" in worker._session_context_text
    asyncio.run(scenario())


def test_voice_buffer_is_preserved_on_receive_cancellation():
    worker = _continuity_worker()
    persisted = []
    worker._persist_episode = lambda role, text: persisted.append((role, text))
    async def cancelled(buffer):
        buffer.append("Explain that circuit")
        raise asyncio.CancelledError
    worker._receive_realtime_stream = cancelled
    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            await worker._receive_realtime()
    asyncio.run(scenario())
    assert persisted == [("user", "Explain that circuit")]


def test_late_results_redact_before_clipping_both_context_and_speech():
    worker = _continuity_worker()
    worker.session = _ContinuitySession()
    secret = "sk-" + "A" * 60
    raw = "x" * 390 + " " + secret
    async def scenario():
        task = asyncio.get_running_loop().create_future()
        task.set_result(({"ok": True, "result": raw}, None))
        worker._report_detached_tool("research", task)
        await worker._flush_context_notes()
        note = worker.session.notes[0]["turns"]["parts"][0]["text"]
        assert "sk-" not in note
        assert "sk-" not in worker.announcements[0]
        assert len(note) < 700
    asyncio.run(scenario())


def test_worker_stop_cancels_adopted_detached_tools_and_scene_work():
    from orion_core import background
    worker = _continuity_worker()
    worker.conn_state = SimpleNamespace(set=lambda *_args: None)
    worker.session = _ContinuitySession()
    worker.LIVE_TOOL_ANSWER_S = 0.001
    scene_stopped = []
    async def stop_scene():
        scene_stopped.append(True)
    worker.scene_recorder = SimpleNamespace(stop_pending=stop_scene)
    async def scenario():
        release = asyncio.Event()
        async def dispatch(*args):
            await release.wait()
            return {"ok": True, "result": "Never reached."}, None
        worker._dispatch_live_tool = dispatch
        await worker._run_tool_call(SimpleNamespace(name="research", args={}))
        inner = next(iter(worker._live_tool_tasks))
        assert inner in background._RUNNING
        await worker.stop()
        assert inner.cancelled()
        assert scene_stopped == [True]
        assert not worker.announcements
    asyncio.run(scenario())


def test_each_live_turn_gets_its_own_request_trace(monkeypatch):
    """16 Oct 2026 session.log: four different questions (14:14-14:25) were all
    stamped ORION_REQ_000001 - only the local recogniser began a trace, so a
    Live turn it missed reused the previous, finished request."""
    import orion_core.live_worker as live
    from orion_core.request_trace import Stage, TraceStore
    store = TraceStore()
    monkeypatch.setattr(live, "TRACES", store)
    worker = _continuity_worker()
    worker._trace = None
    worker._mark_turn("give me the latest on your own updates")
    first = worker._trace
    first.stage(Stage.LISTENING_RESTORED)              # that request finished
    worker._mark_turn("can you sing me a song")        # no local transcript for it
    second = worker._trace
    assert second is not first and second.request_id != first.request_id
    assert [e.stage for e in second.events] == [Stage.API_REQUEST_START.value]
    # Two unfinished Live turns are still two requests, not one.
    worker._mark_turn("how old are you now")
    assert worker._trace is not second
    # The local recogniser's transcript of the SAME utterance joins it (the two
    # arrive about a second apart, in either order) instead of doubling it.
    third = worker._trace
    worker._turn_trace(Stage.TRANSCRIPTION_COMPLETE).stage(Stage.TRANSCRIPTION_COMPLETE, "how old")
    assert worker._trace is third
    worker._trace = None
    worker._turn_trace(Stage.TRANSCRIPTION_COMPLETE).stage(Stage.TRANSCRIPTION_COMPLETE, "what else")
    heard = worker._trace
    worker._mark_turn("what else can we improve", origin="text")
    assert worker._trace is heard
    assert [e.stage for e in heard.events] == [Stage.TRANSCRIPTION_COMPLETE.value,
                                              Stage.API_REQUEST_START.value]
    # A stale unfinished one is left for TRACES.stalled() to report as dropped.
    monkeypatch.setattr(type(heard), "age_s", property(lambda self: 60.0))
    worker._mark_turn("are you there")
    assert worker._trace is not heard and not heard.complete
    assert len(store.all()) == 5


def test_the_expressive_voice_connects_on_the_api_that_has_it(monkeypatch):
    fault = RuntimeError("1006 socket closed")
    fault.code = 1006
    monkeypatch.setenv("ORION_EXPRESSIVE_VOICE", "1")
    worker, _ = _run_worker_through(monkeypatch, fault, api_key="AQ.authorization-test")
    assert worker._client_api_version == "v1alpha"


# ── the microphone stream as Live receives it (28 Oct 2026) ──────────────────

class _StreamSession:
    """Records every realtime input by its typed field."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(kwargs)


def test_a_paused_microphone_stream_is_closed_once_and_reopened_by_audio():
    """The stream paused whenever ORION spoke or ran a tool and Live was never
    told; the Live API asks for audio_stream_end on any pause over a second."""
    import time as _time
    session = _StreamSession()
    worker = _make_worker(session=session)

    async def scenario():
        await worker._send_media({"data": b"\x10\x00" * 512, "mime_type": "audio/pcm"})
        await worker._end_paused_audio_stream()        # under a second: still streaming
        assert not any("audio_stream_end" in kw for kw in session.sent)
        worker._audio_sent_at = _time.monotonic() - 1.5
        await worker._end_paused_audio_stream()
        await worker._end_paused_audio_stream()        # once per pause
        await worker._send_media({"data": b"\x10\x00" * 512, "mime_type": "audio/pcm"})

    asyncio.run(scenario())
    assert [next(iter(kw)) for kw in session.sent] == ["audio", "audio_stream_end", "audio"]
    assert worker._audio_streaming is True
    assert worker._audio_stream_ends == 1


def test_the_sender_closes_the_stream_while_the_microphone_is_quiet():
    import time as _time
    session = _StreamSession()
    worker = _make_worker(session=session)
    worker.out_queue = asyncio.Queue()
    worker._audio_streaming, worker._audio_sent_at = True, _time.monotonic() - 2.0
    original = session.send_realtime_input

    async def send(**kwargs):
        await original(**kwargs)
        worker.stop_event.set()

    session.send_realtime_input = send
    asyncio.run(worker._send_realtime())
    assert session.sent == [{"audio_stream_end": True}]


def test_what_reached_live_is_measured_for_a_missed_request():
    import time as _time
    worker = _make_worker(session=_StreamSession())
    loud = (8000).to_bytes(2, "little", signed=True) * 512
    for _ in range(100):
        worker._note_audio_sent(loud)
    seconds, level = worker._audio_evidence(_time.monotonic())
    assert abs(seconds - 100 * 512 / 16000) < 0.01
    assert -13.0 < level < -11.5                       # 8000 of 32768 is -12.2 dBFS
    assert worker._audio_evidence(_time.monotonic() + 60.0) == (0.0, None)


def test_a_context_note_waits_for_the_owner_to_finish_speaking(monkeypatch):
    """A note landing mid-utterance can cost Live the request."""
    import orion_core.live_worker as lw
    monkeypatch.setattr(lw, "_turn_detection", lambda: {"silence_ms": 0})
    calls = []

    async def send(**kwargs):
        calls.append(kwargs)

    worker = _make_worker(session=SimpleNamespace(send_client_content=send))
    worker._on_live_speech_activity(True)

    async def scenario():
        task = asyncio.create_task(worker._send_context_note("[background result]"))
        await asyncio.sleep(0.25)
        assert calls == [], "sent while the owner was still talking"
        worker._on_live_speech_activity(False)
        assert await asyncio.wait_for(task, 2.0) is True

    asyncio.run(scenario())
    assert calls and calls[0]["turn_complete"] is False


def test_a_channel_refresh_waits_for_quiet_and_is_rate_limited():
    worker = _make_worker(session=_StreamSession())
    worker.tool_busy = False
    worker.speech = SimpleNamespace(output_active=lambda: False)
    worker._turn_active = True
    assert worker._request_rotation("it stopped hearing the microphone") is True
    assert worker._request_rotation("again") is False, "one refresh a minute at most"
    assert worker._rotation_due() is False, "never across a turn in flight"
    worker._turn_active = False
    worker._owner_talking = True
    assert worker._rotation_due() is False, "never while the owner is talking"
    worker._owner_talking = False
    assert worker._rotation_due() is True
    assert worker._rotation_due() is False, "a refresh happens once"
    assert worker._last_close_reason.startswith("refresh:")


def test_a_due_refresh_ends_the_sender_so_the_channel_reconnects():
    worker = _make_worker(session=_StreamSession())
    worker.tool_busy = False
    worker.speech = SimpleNamespace(output_active=lambda: False)
    worker.out_queue = asyncio.Queue()
    worker._rotate_reason = "it stopped hearing the microphone"
    asyncio.run(asyncio.wait_for(worker._send_realtime(), 2.0))
    assert worker._rotate_reason == ""
