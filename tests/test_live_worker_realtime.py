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

        def enqueue_native_audio(self, chunk):
            self.chunks.append(chunk)

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
    assert worker._drop_live_output is False


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
        assert payload["ok"] is True and "still running" in payload["result"]
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


def _run_worker_through(monkeypatch, failure):
    """Drive GenAILiveWorker.run() through one established session that ends
    with *failure*, and report what the recovery did."""
    import orion_core.live_worker as lw

    monkeypatch.setattr(lw, "_live_diag", lambda *a, **k: None)
    worker = _make_worker()
    marked: list[str] = []
    profile = SimpleNamespace(name="gemini", model="models/live", api_key="AIza-test")
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

    async def session_loop():
        worker.stop_event.set()        # one session only
        raise failure

    worker._redrive_pending_turn = redrive
    worker._session_loop = session_loop
    asyncio.run(worker.run())
    return worker, marked


def test_an_abnormal_close_keeps_the_conversation_and_the_provider(monkeypatch):
    # Measured on the real server: a dropped socket surfaces as APIError 1006.
    # It used to bench Live for 45 s AND discard the resumption handle, so
    # ORION came back (late) with no memory of the conversation.
    drop = RuntimeError("1006 None. abnormal closure [internal]")
    drop.code = 1006
    worker, marked = _run_worker_through(monkeypatch, drop)
    assert marked == []
    assert worker._resumption_handle == "handle-1"


def test_a_real_provider_fault_still_cools_the_provider(monkeypatch):
    fault = RuntimeError("quota exhausted for this project")
    fault.code = 429
    worker, marked = _run_worker_through(monkeypatch, fault)
    assert marked == ["gemini"]
    assert worker._resumption_handle is None
