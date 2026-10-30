"""
Provider transport contracts, checked against real HTTP servers on 127.0.0.1.

These replace tests that asserted the JSON ORION *sent*. Those passed while
Ollama's /v1 endpoint silently dropped keep_alive and num_ctx, so they proved
nothing about memory. Here a fake Ollama and a fake OpenAI-compatible server
record what actually arrives, and the tests check what the router does with
the answers: which endpoint, which fields, which connection, which deadline.

The last test talks to a REAL installed Ollama and is skipped unless
ORION_LIVE_OLLAMA=1. It never pulls a model.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from orion_core import provider_transport as pt  # noqa: E402
from orion_core.providers import (  # noqa: E402
    AIProviderProfile, OrionProviderSettings, ProviderClosedError,
    ProviderDeadlineError, ProviderRouter, StreamInterruptedError)
from orion_core.request_trace import TRACES, bound  # noqa: E402


# ── scaffolding ───────────────────────────────────────────────────────────────

class _Signal:
    def __init__(self, sink=None):
        self._sink = sink

    def emit(self, *args, **_kw):
        if self._sink is not None:
            self._sink.append(args[0] if args else "")

    def connect(self, *_a, **_k):
        pass


class _Bus:
    def __init__(self):
        self.lines: list[str] = []

    def __getattr__(self, name):
        return _Signal(self.lines if name == "log" else None)


class _Memory:
    def prompt_context(self, limit=16):
        return ""


class _Ledger:
    def __init__(self):
        self.records = []

    def record(self, rec):
        self.records.append(rec)


class _Remote(AIProviderProfile):
    """A cloud provider for routing purposes, served from a local fake."""

    is_local = property(lambda self: False)


def _router(*profiles, ledger=None) -> ProviderRouter:
    settings = OrionProviderSettings(
        active_provider=profiles[0].name,
        provider_order=[p.name for p in profiles],
        providers={p.name: p for p in profiles})
    router = ProviderRouter(settings, _Bus(), memory=_Memory())
    if ledger is not None:
        router.attach_token_ledger(ledger)
    return router


def _cloud(name, url, key="sk-test", **kw) -> _Remote:
    return _Remote(name=name, kind="openai_compatible", model=kw.pop("model", "gpt-x"),
                   base_url=url, api_key=key, enabled=True, **kw)


def _local_ollama(url, **kw) -> AIProviderProfile:
    return AIProviderProfile(name="local_ollama", kind="openai_compatible",
                             model=kw.pop("model", "llama3.1"), base_url=url,
                             api_key="local", enabled=True, **kw)


def _peer(request) -> tuple:
    return tuple(request.transport.get_extra_info("peername") or ())


class FakeOllama:
    """The parts of Ollama's native API ORION uses, recording what arrives."""

    def __init__(self, pieces=("Hello", " there"), delay=0.0, hold=False):
        self.pieces = list(pieces)
        self.delay = delay
        self.hold = hold
        self.chats: list[dict] = []
        self.v1_calls = 0
        self.version_calls = 0
        self.unloads: list[str] = []
        self.loaded: dict[str, dict] = {}
        self.active = 0
        self.max_active = 0
        self.disconnected = False
        self.release: asyncio.Event | None = None
        self.started: asyncio.Event | None = None

    def app(self) -> web.Application:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        app = web.Application()
        app.router.add_get("/api/version", self._version)
        app.router.add_post("/api/chat", self._chat)
        app.router.add_get("/api/ps", self._ps)
        app.router.add_post("/api/generate", self._generate)
        app.router.add_post("/v1/chat/completions", self._v1)
        return app

    async def _version(self, _request):
        self.version_calls += 1
        return web.json_response({"version": "0.32.6"})

    async def _v1(self, _request):
        self.v1_calls += 1
        return web.json_response({"choices": [{"message": {"content": "wrong door"}}]})

    async def _ps(self, _request):
        return web.json_response({"models": [
            {"name": m, "model": m, **info} for m, info in self.loaded.items()]})

    async def _generate(self, request):
        body = await request.json()
        if body.get("keep_alive") == 0 and not body.get("prompt"):
            self.unloads.append(body["model"])
            self.loaded.pop(body["model"], None)
        return web.json_response({"model": body.get("model"), "done": True,
                                  "done_reason": "unload"})

    async def _chat(self, request):
        body = await request.json()
        self.chats.append({"json": body, "headers": dict(request.headers),
                           "peer": _peer(request)})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            model = body["model"] if ":" in body["model"] else body["model"] + ":latest"
            if body.get("keep_alive") != 0:
                self.loaded[model] = {"context_length": body["options"].get("num_ctx"),
                                      "keep_alive": body.get("keep_alive")}
            response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
            await response.prepare(request)
            for piece in self.pieces:
                await response.write(json.dumps({
                    "model": model, "message": {"role": "assistant", "content": piece},
                    "done": False}).encode() + b"\n")
                if self.delay:
                    await asyncio.sleep(self.delay)
            if self.hold:
                while not self.release.is_set():
                    # A blank line is a keep-alive the client skips. When the
                    # client hangs up, aiohttp either fails the write or
                    # cancels this handler, depending on version.
                    try:
                        await response.write(b"\n")
                        await asyncio.sleep(0.02)
                    except asyncio.CancelledError:
                        self.disconnected = True
                        raise
                    except Exception:
                        self.disconnected = True
                        return response
            await response.write(json.dumps({
                "model": model, "message": {"role": "assistant", "content": ""},
                "done": True, "done_reason": "stop", "total_duration": 2_000_000_000,
                "load_duration": 1_500_000_000, "prompt_eval_count": 12,
                "eval_count": 3}).encode() + b"\n")
            await response.write_eof()
            return response
        finally:
            self.active -= 1


class FakeOpenAI:
    """An OpenAI-compatible server that is NOT Ollama (/api/version -> 404)."""

    def __init__(self, text="fine", status=200, delay=0.0, pieces=None,
                 truncate=False, set_cookie=False):
        self.text = text
        self.status = status
        self.delay = delay
        self.pieces = pieces
        self.truncate = truncate
        self.set_cookie = set_cookie
        self.calls: list[dict] = []
        self.version_calls = 0

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_get("/api/version", self._version)
        return app

    async def _version(self, _request):
        self.version_calls += 1
        return web.Response(status=404, text="not found")

    async def _chat(self, request):
        body = await request.json()
        self.calls.append({"json": body, "headers": dict(request.headers),
                           "peer": _peer(request)})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.status >= 400:
            return web.Response(status=self.status, text="upstream overloaded")
        headers = {"Set-Cookie": f"session={body['model']}-secret; Path=/"} \
            if self.set_cookie else {}
        if body.get("stream"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", **headers})
            await response.prepare(request)
            for piece in self.pieces or [self.text]:
                chunk = {"choices": [{"delta": {"content": piece}}]}
                await response.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                await asyncio.sleep(0.01)
            if not self.truncate:
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        return web.json_response(
            {"id": f"r{len(self.calls)}", "model": body["model"],
             "choices": [{"message": {"content": self.text}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}},
            headers=headers)


async def _serve(fake) -> TestServer:
    server = TestServer(fake.app(), host="127.0.0.1")
    await server.start_server()
    return server


# ── the native Ollama adapter ─────────────────────────────────────────────────

def test_ollama_receives_context_keep_alive_and_images_on_its_native_api():
    fake = FakeOllama()
    ledger = _Ledger()

    async def flow():
        server = await _serve(fake)
        try:
            router = _router(_local_ollama(str(server.make_url("/v1"))), ledger=ledger)
            trace = TRACES.begin("test")
            with bound(trace):
                answer = await router._openai_compatible_chat(
                    router.settings.providers["local_ollama"], "look",
                    instruction="SYS", image_jpeg=b"fixture-image")
            await router.aclose()
            return answer, trace
        finally:
            await server.close()

    answer, trace = asyncio.run(flow())
    assert answer == "Hello there"
    assert fake.v1_calls == 0                       # never the compatible door
    sent = fake.chats[0]["json"]
    assert sent["options"]["num_ctx"] == ProviderRouter.LOCAL_NUM_CTX
    assert sent["keep_alive"] == f"{ProviderRouter.LOCAL_KEEP_ALIVE_S}s"
    assert sent["messages"][1]["images"] == [base64.b64encode(b"fixture-image").decode()]
    assert sent["options"]["num_predict"] > 0 and sent["stream"] is True
    assert "Authorization" not in fake.chats[0]["headers"]   # "local" is no key
    # Usage comes from Ollama's own counts, not a character estimate.
    usage = ledger.records[0]
    assert (usage.input_tokens, usage.output_tokens) == (12, 3)
    attempt = trace.marks_named("provider_attempt")[0].fields
    assert attempt["provider"] == "local_ollama" and attempt["transport"] == "ollama"
    assert attempt["ok"] is True and attempt["load_ms"] == 1500.0
    assert attempt["first_token_ms"] is not None and attempt["ms"] >= 0
    assert "look" not in json.dumps([m.fields for m in trace.marks])


def test_a_local_server_that_is_not_ollama_never_receives_ollama_fields():
    fake = FakeOpenAI(text="from lm studio")

    async def flow():
        server = await _serve(fake)
        try:
            profile = AIProviderProfile(
                name="local_lm_studio", kind="openai_compatible", model="local-model",
                base_url=str(server.make_url("/v1")), api_key="local", enabled=True)
            router = _router(profile)
            first = await router._openai_compatible_chat(profile, "hi", instruction="SYS")
            second = await router._openai_compatible_chat_stream(profile, "hi", instruction="SYS")
            await router.aclose()
            return first, second
        finally:
            await server.close()

    first, second = asyncio.run(flow())
    assert first == second == "from lm studio"
    assert fake.version_calls == 1                  # asked once, remembered
    for call in fake.calls:
        body = call["json"]
        assert "keep_alive" not in body and "options" not in body
        assert "num_ctx" not in json.dumps(body)


def test_native_use_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("ORION_OLLAMA_NATIVE", "0")
    fake = FakeOllama()

    async def flow():
        server = await _serve(fake)
        try:
            router = _router(_local_ollama(str(server.make_url("/v1"))))
            answer = await router._openai_compatible_chat(
                router.settings.providers["local_ollama"], "hi", instruction="SYS")
            await router.aclose()
            return answer
        finally:
            await server.close()

    assert asyncio.run(flow()) == "wrong door"
    assert fake.v1_calls == 1 and fake.chats == [] and fake.version_calls == 0


# ── sessions ─────────────────────────────────────────────────────────────────

def test_one_session_and_one_connection_serve_many_calls():
    fake = FakeOpenAI()

    async def flow():
        server = await _serve(fake)
        try:
            router = _router(_cloud("cloud", str(server.make_url("/v1"))))
            trace = TRACES.begin("test")
            with bound(trace):
                for _ in range(5):
                    await router.generate_text("hello", instruction="SYS")
            created = router._pool.created
            await router.aclose()
            return created, trace
        finally:
            await server.close()

    created, trace = asyncio.run(flow())
    assert created == 1
    assert len({call["peer"] for call in fake.calls}) == 1     # the same socket
    connects = [m.fields["connect_ms"] for m in trace.marks_named("provider_attempt")]
    assert connects[0] is not None and connects[1:] == [0.0] * 4   # set up once


def test_credentials_and_cookies_stay_with_their_own_profile():
    ok = FakeOpenAI(set_cookie=True)

    async def flow():
        good_server = await _serve(ok)
        try:
            # Two profiles on one host with different keys (a general profile
            # and the 'thoughts' one on the same account, say).
            first = _cloud("first", str(good_server.make_url("/v1")), key="key-one")
            second = _cloud("second", str(good_server.make_url("/v1")), key="key-two",
                            model="other-model")
            router = _router(first, second)
            await router._openai_compatible_chat(first, "a", instruction="SYS")
            await router._openai_compatible_chat(second, "b", instruction="SYS")
            await router._openai_compatible_chat(first, "c", instruction="SYS")
            await router.aclose()
        finally:
            await good_server.close()

    asyncio.run(flow())
    keys = [call["headers"].get("Authorization") for call in ok.calls]
    assert keys == ["Bearer key-one", "Bearer key-two", "Bearer key-one"]
    # The server set a cookie on every reply; none was ever sent back.
    assert all("Cookie" not in call["headers"] for call in ok.calls)


def test_sessions_are_never_shared_across_event_loops():
    fake = FakeOpenAI()
    sessions = []

    async def one_call(port):
        await router.generate_text("hi", instruction="SYS")
        sessions.append(await router._pool.get())

    async def serve_and_call():
        server = await _serve(fake)
        try:
            router.settings.providers["cloud"].base_url = str(server.make_url("/v1"))
            await one_call(server.port)
        finally:
            await server.close()

    router = _router(_cloud("cloud", "http://placeholder.invalid/v1"))
    asyncio.run(serve_and_call())
    asyncio.run(serve_and_call())
    assert router._pool.created == 2
    assert sessions[0] is not sessions[1]
    # The first loop's session was closed when that loop shut down, not
    # leaked until garbage collection.
    assert sessions[0].closed


def test_a_dead_pooled_connection_is_retried_once_on_a_fresh_one():
    """A server may close an idle keep-alive connection just as ORION reuses
    it. The request was never read, so it is sent again on a new connection
    instead of benching a healthy provider."""
    state = {"connections": 0, "answered": 0}
    clients = set()

    async def handle(reader, writer):
        clients.add(writer)
        state["connections"] += 1
        served = 0
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                length = int(re.search(rb"(?i)content-length:\s*(\d+)", head).group(1))
                await reader.readexactly(length)
                if served == 1:
                    writer.close()          # dropped while "idle"
                    return
                body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                             b"Content-Length: %d\r\n\r\n" % len(body) + body)
                await writer.drain()
                served += 1
                state["answered"] += 1
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            clients.discard(writer)

    async def flow():
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            router = _router(_cloud("cloud", f"http://127.0.0.1:{port}/v1"))
            texts = [await router._openai_compatible_chat(
                router.settings.providers["cloud"], "hi", instruction="SYS")
                for _ in range(2)]
            breaker = router.breaker_state(router.settings.providers["cloud"])
            await router.aclose()
            return texts, breaker
        finally:
            server.close()
            # asyncio.Server.close_clients was only added in Python 3.13.
            # Close actual writers so the supported 3.12 runtime also cleans up.
            active = list(clients)
            for writer in active:
                writer.close()
            await asyncio.gather(*(writer.wait_closed() for writer in active),
                                 return_exceptions=True)
            await server.wait_closed()

    texts, breaker = asyncio.run(flow())
    assert texts == ["ok", "ok"]
    assert state == {"connections": 2, "answered": 2}
    assert breaker == "closed"


# ── deadlines and timeouts ───────────────────────────────────────────────────

def test_one_deadline_bounds_every_attempt_and_is_not_charged_to_the_provider():
    slow, spare = FakeOpenAI(delay=10.0), FakeOpenAI(text="spare")

    async def flow():
        slow_server, spare_server = await _serve(slow), await _serve(spare)
        try:
            router = _router(_cloud("slow", str(slow_server.make_url("/v1"))),
                             _cloud("spare", str(spare_server.make_url("/v1"))))
            # The third held back for the spare (ProviderRouter.FALLBACK_RESERVE_S)
            # is less than an attempt needs here, so the spare is never started:
            # test_provider_fallback_reserve covers the spare answering in time.
            router.MIN_ATTEMPT_S = 0.6
            started = time.monotonic()
            with pytest.raises(ProviderDeadlineError):
                await router.generate_text("hi", instruction="SYS", deadline_s=1.0)
            elapsed = time.monotonic() - started
            state = router.breaker_state(router.settings.providers["slow"])
            await router.aclose()
            return elapsed, state
        finally:
            await slow_server.close()
            await spare_server.close()

    elapsed, state = asyncio.run(flow())
    assert elapsed < 1.8                 # not the profile's 30 s timeout
    assert state == "closed"             # running out of time is not a fault
    assert spare.calls == []             # nothing started that could not finish


def test_a_failed_provider_hands_the_remaining_time_to_the_next():
    broken, spare = FakeOpenAI(status=503), FakeOpenAI(text="from the spare")

    async def flow():
        a, b = await _serve(broken), await _serve(spare)
        try:
            router = _router(_cloud("broken", str(a.make_url("/v1"))),
                             _cloud("spare", str(b.make_url("/v1"))))
            profile, text = await router.generate_text("hi", instruction="SYS", deadline_s=5.0)
            await router.aclose()
            return profile.name, text, router.breaker_state(router.settings.providers["broken"])
        finally:
            await a.close()
            await b.close()

    assert asyncio.run(flow()) == ("spare", "from the spare", "open")


def test_timeouts_are_per_request_on_the_shared_session():
    slow, quick = FakeOpenAI(delay=5.0), FakeOpenAI(delay=0.8, text="in time")

    async def flow():
        a, b = await _serve(slow), await _serve(quick)
        try:
            router = _router(_cloud("slow", str(a.make_url("/v1"))))
            router.MIN_ATTEMPT_S = 0.1
            quick_profile = _cloud("quick", str(b.make_url("/v1")))
            # Same router, same pooled session: the short deadline of one
            # call must not become the other's timeout.
            results = await asyncio.gather(
                router.generate_text("x", instruction="SYS", deadline_s=0.5),
                router._openai_compatible_chat(quick_profile, "y", instruction="SYS"),
                return_exceptions=True)
            created = router._pool.created
            await router.aclose()
            assert created == 1
            return results
        finally:
            await a.close()
            await b.close()

    short, long = asyncio.run(flow())
    assert isinstance(short, ProviderDeadlineError)
    assert long == "in time"


# ── streaming, restart and cancellation ──────────────────────────────────────

def _stream_pair(truncate_first=True):
    first = FakeOpenAI(pieces=["Hel", "lo from A"], truncate=truncate_first)
    second = FakeOpenAI(pieces=["Hello", " from B"])
    return first, second


def test_a_stream_that_breaks_after_showing_text_restarts_cleanly():
    first, second = _stream_pair()
    shown, restarts = [], []

    async def flow():
        a, b = await _serve(first), await _serve(second)
        try:
            router = _router(_cloud("A", str(a.make_url("/v1"))),
                             _cloud("B", str(b.make_url("/v1"))))
            trace = TRACES.begin("test")
            with bound(trace):
                profile, text = await router.generate_text(
                    "hi", instruction="SYS", on_delta=shown.append,
                    on_restart=restarts.append)
            await router.aclose()
            return profile.name, text, trace
        finally:
            await a.close()
            await b.close()

    name, text, trace = asyncio.run(flow())
    assert (name, text) == ("B", "Hello from B")
    assert restarts == ["provider_failed:A"]
    # Everything shown after the restart is B's answer alone.
    after = shown[shown.index("Hello"):]
    assert "".join(after) == "Hello from B"
    assert shown[:2] == ["Hel", "lo from A"]
    assert trace.since_start_ms("first_token") is not None
    assert len(trace.marks_named("first_token")) == 1


def test_without_on_restart_a_broken_stream_raises_rather_than_splicing():
    first, second = _stream_pair()

    async def flow():
        a, b = await _serve(first), await _serve(second)
        try:
            router = _router(_cloud("A", str(a.make_url("/v1"))),
                             _cloud("B", str(b.make_url("/v1"))))
            with pytest.raises(StreamInterruptedError):
                await router.generate_text("hi", instruction="SYS", on_delta=lambda _p: None)
            await router.aclose()
        finally:
            await a.close()
            await b.close()

    asyncio.run(flow())
    assert second.calls == []


def test_a_failure_before_any_text_falls_through_without_a_restart():
    broken, second = FakeOpenAI(status=503), FakeOpenAI(pieces=["fine"])
    shown, restarts = [], []

    async def flow():
        a, b = await _serve(broken), await _serve(second)
        try:
            router = _router(_cloud("A", str(a.make_url("/v1"))),
                             _cloud("B", str(b.make_url("/v1"))))
            _p, text = await router.generate_text(
                "hi", instruction="SYS", on_delta=shown.append, on_restart=restarts.append)
            await router.aclose()
            return text
        finally:
            await a.close()
            await b.close()

    assert asyncio.run(flow()) == "fine"
    assert shown == ["fine"] and restarts == []


def test_cancelling_mid_stream_closes_the_connection_and_frees_the_slot():
    fake = FakeOllama(pieces=["partial"], hold=True)

    async def flow():
        server = await _serve(fake)
        try:
            router = _router(_local_ollama(str(server.make_url("/v1"))))
            shown = asyncio.Event()
            task = asyncio.create_task(router.generate_text(
                "hi", instruction="SYS", on_delta=lambda _p: shown.set()))
            await asyncio.wait_for(shown.wait(), 5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(100):
                if fake.disconnected:
                    break
                await asyncio.sleep(0.02)
            gates = router.admission_snapshot()
            busy = dict(router._inflight)
            fake.hold, fake.pieces = False, ["again"]
            _p, text = await router.generate_text("hi", instruction="SYS")
            await router.aclose()
            return gates, busy, text
        finally:
            fake.release.set()
            await server.close()

    gates, busy, text = asyncio.run(flow())
    assert fake.disconnected                        # Ollama stops generating
    assert all(g["active"] == 0 for g in gates.values())
    assert busy == {}
    assert text == "again"


# ── admission ────────────────────────────────────────────────────────────────

def test_interactive_waiters_go_before_background_ones():
    async def flow():
        gate = pt.EndpointGate(limit=1)
        await gate.acquire(pt.BACKGROUND)
        order = []

        async def wait(cls, label):
            await gate.acquire(cls)
            order.append(label)
            await asyncio.sleep(0)
            gate.release(cls)

        background = asyncio.create_task(wait(pt.BACKGROUND, "background"))
        await asyncio.sleep(0)
        interactive = asyncio.create_task(wait(pt.INTERACTIVE, "interactive"))
        await asyncio.sleep(0)
        gate.release(pt.BACKGROUND)
        await asyncio.gather(background, interactive)
        return order

    assert asyncio.run(flow()) == ["interactive", "background"]


def test_background_work_leaves_room_for_a_conversation():
    async def flow():
        gate = pt.EndpointGate(limit=2, background_limit=1)
        await gate.acquire(pt.BACKGROUND)
        with pytest.raises(asyncio.TimeoutError):
            await gate.acquire(pt.BACKGROUND, timeout=0.05)
        waited = await gate.acquire(pt.INTERACTIVE, timeout=0.05)
        return waited, gate.snapshot()

    waited, snap = asyncio.run(flow())
    assert waited == 0.0
    assert snap["active"] == 2 and snap["waiting"] == 0


def test_a_given_up_waiter_does_not_leak_its_slot():
    async def flow():
        gate = pt.EndpointGate(limit=1)
        await gate.acquire()
        with pytest.raises(asyncio.TimeoutError):
            await gate.acquire(timeout=0.01)
        gate.release()
        return await gate.acquire(timeout=0.5), gate.snapshot()

    waited, snap = asyncio.run(flow())
    assert waited == 0.0 and snap["active"] == 1


def test_local_generation_runs_one_at_a_time_with_conversation_first():
    fake = FakeOllama(pieces=["ok"], delay=0.15)

    async def flow():
        server = await _serve(fake)
        try:
            router = _router(_local_ollama(str(server.make_url("/v1"))))
            first = asyncio.create_task(router.generate_text(
                "background one", instruction="SYS", task="research"))
            await asyncio.wait_for(fake.started.wait(), 5.0)
            queued = asyncio.create_task(router.generate_text(
                "background two", instruction="SYS", task="research"))
            await asyncio.sleep(0.02)
            talk = asyncio.create_task(router.generate_text("conversation", instruction="SYS"))
            await asyncio.gather(first, queued, talk)
            await router.aclose()
        finally:
            await server.close()

    asyncio.run(flow())
    order = [c["json"]["messages"][1]["content"] for c in fake.chats]
    assert fake.max_active == 1
    assert order == ["background one", "conversation", "background two"]


# ── residency ────────────────────────────────────────────────────────────────


def test_shutdown_does_not_probe_a_configured_local_model_never_used():
    async def flow():
        router = _router(_local_ollama("http://127.0.0.1:11434/v1"))
        probes = []

        async def unexpected_probe(*_args):
            probes.append(_args)
            raise AssertionError("shutdown probed an unused model server")

        router._is_ollama = unexpected_probe
        assert await router.unload_local_models("shutdown") == []
        assert probes == []
        await router.aclose()

    asyncio.run(flow())

def test_background_work_never_shortens_a_conversations_residency():
    fake = FakeOllama(pieces=["ok"])

    async def flow():
        server = await _serve(fake)
        try:
            router = _router(_local_ollama(str(server.make_url("/v1"))))
            await router.generate_text("talk", instruction="SYS")
            await router.generate_thought("reflect")
            await router.aclose()
        finally:
            await server.close()

    asyncio.run(flow())
    first, second = (int(c["json"]["keep_alive"].rstrip("s")) for c in fake.chats)
    assert first == ProviderRouter.LOCAL_KEEP_ALIVE_S
    assert second >= first - 2 > ProviderRouter.LOCAL_BACKGROUND_KEEP_ALIVE_S


def test_pressure_unloads_idle_models_but_waits_for_an_active_request():
    fake = FakeOllama(pieces=["working"], hold=True)

    async def flow():
        server = await _serve(fake)
        try:
            router = _router(_local_ollama(str(server.make_url("/v1"))))
            task = asyncio.create_task(router.generate_text("long job", instruction="SYS"))
            await asyncio.wait_for(fake.started.wait(), 5.0)
            released_during = await router.relieve_memory_pressure("test pressure")
            unloads_during = list(fake.unloads)
            fake.release.set()
            await task
            for _ in range(100):
                if fake.unloads:
                    break
                await asyncio.sleep(0.02)
            fake.hold = False
            await router.generate_text("next", instruction="SYS")
            router.set_memory_pressure(False)
            await router.generate_text("after", instruction="SYS")
            await router.aclose()
            return released_during, unloads_during
        finally:
            fake.release.set()
            await server.close()

    released_during, unloads_during = asyncio.run(flow())
    assert released_during == [] and unloads_during == []   # never under a request
    assert fake.unloads == ["llama3.1:latest"]               # ...but right after it
    keep = [c["json"]["keep_alive"] for c in fake.chats]
    assert keep[1] == 0                                      # released after each reply
    assert keep[2] == f"{ProviderRouter.LOCAL_KEEP_ALIVE_S}s"


def test_only_orions_own_models_are_unloaded():
    fake = FakeOllama()

    async def flow():
        server = await _serve(fake)
        fake.loaded["someone-elses:7b"] = {"context_length": 8192}
        fake.loaded["llama3.1:latest"] = {"context_length": 4096}
        try:
            router = _router(_local_ollama(str(server.make_url("/v1"))))
            released = await router.unload_local_models("test")
            await router.aclose()
            return released
        finally:
            await server.close()

    assert asyncio.run(flow()) == ["llama3.1:latest"]
    assert "someone-elses:7b" in fake.loaded


# ── shutdown ─────────────────────────────────────────────────────────────────

def test_aclose_is_idempotent_and_refuses_new_calls():
    fake = FakeOpenAI()

    async def flow():
        server = await _serve(fake)
        try:
            router = _router(_cloud("cloud", str(server.make_url("/v1"))))
            await router.generate_text("hi", instruction="SYS")
            session = await router._pool.get()
            await router.aclose()
            await router.aclose()
            with pytest.raises(ProviderClosedError):
                await router.generate_text("again", instruction="SYS")
            return session.closed, router.breaker_state(router.settings.providers["cloud"])
        finally:
            await server.close()

    closed, breaker = asyncio.run(flow())
    assert closed is True and breaker == "closed"
    assert len(fake.calls) == 1


def test_the_first_success_reports_provider_ready(monkeypatch):
    from orion_core import startup_budget

    seen = []
    monkeypatch.setattr(startup_budget, "milestone", seen.append)
    fake = FakeOpenAI()

    async def flow():
        server = await _serve(fake)
        try:
            router = _router(_cloud("cloud", str(server.make_url("/v1"))))
            await router.generate_text("one", instruction="SYS")
            await router.generate_text("two", instruction="SYS")
            await router.aclose()
        finally:
            await server.close()

    asyncio.run(flow())
    assert seen == ["provider_ready"]


# ── live: the installed Ollama (opt-in) ──────────────────────────────────────

_OLLAMA = os.getenv("ORION_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{_OLLAMA}{path}", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.mark.skipif(os.getenv("ORION_LIVE_OLLAMA") != "1",
                    reason="set ORION_LIVE_OLLAMA=1 to check a real, installed Ollama")
def test_live_ollama_applies_orions_context_and_residency():
    """Against the real server: the model is loaded at ORION's context, stays
    resident for the keep-alive, and unload_local_models releases it. Uses a
    model already listed by /api/tags; never pulls one."""
    models = [m["name"] for m in _get("/api/tags").get("models", [])]
    if not models:
        pytest.skip("Ollama is running but has no models installed")
    model = models[0]

    async def flow():
        router = _router(_local_ollama(f"{_OLLAMA}/v1", model=model, timeout_s=300.0))
        _p, text = await router.generate_text("Reply with the single word: ready.",
                                              instruction="Answer in one word.")
        loaded = {m["name"]: m for m in _get("/api/ps").get("models", [])}
        released = await router.unload_local_models("live test")
        after = [m["name"] for m in _get("/api/ps").get("models", [])]
        await router.aclose()
        return text, loaded, released, after

    text, loaded, released, after = asyncio.run(flow())
    assert text.strip()
    key = model if ":" in model else f"{model}:latest"
    assert loaded[key]["context_length"] == ProviderRouter.LOCAL_NUM_CTX
    assert released == [key] and key not in after


def test_a_busy_gpu_is_a_deadline_not_a_provider_fault(monkeypatch):
    """Waiting for the GPU is bounded by the request's own deadline, and a
    saturated lease surfaces as ProviderDeadlineError — which is never charged
    to the local provider — rather than an unknown error the circuit breaker
    would count against Ollama."""
    import asyncio

    from orion_core import admission
    from orion_core.providers import ProviderDeadlineError, ProviderRouter

    released = []

    class _Ticket:
        async def wait(self, timeout=None):
            assert timeout == 0.25
            raise admission.Busy("the GPU is held by training")

        def release(self):
            released.append(True)

    monkeypatch.setattr(admission.ADMISSION, "acquire_lease",
                        lambda *a, **k: _Ticket())
    router = ProviderRouter.__new__(ProviderRouter)

    async def attempt():
        async with router._gpu_lease("interactive", timeout=0.25, provider="local_ollama"):
            raise AssertionError("must not generate without the GPU")

    try:
        asyncio.run(attempt())
    except ProviderDeadlineError as exc:
        assert "GPU is busy" in str(exc) and exc.attempts == ["local_ollama"]
    else:
        raise AssertionError("a busy GPU must raise ProviderDeadlineError")
    assert released == [True]


# ── local-only generation (the optional neural planner's path) ────────────────

def test_local_only_generation_is_structured_and_never_reaches_the_cloud():
    """A cloud profile first in the order is never contacted; the schema
    reaches Ollama as ``format``; a named model replaces the profile's own;
    and the attempt is booked under its own name, so a long planning reply
    does not raise the latency estimate that lets a spoken turn go local.
    An ordinary chat afterwards carries no ``format`` at all."""
    ollama, cloud = FakeOllama(pieces=('{"steps": []}',)), FakeOpenAI(text="cloud")
    schema = {"type": "object", "properties": {"steps": {"type": "array"}}}

    async def flow():
        local_server, cloud_server = await _serve(ollama), await _serve(cloud)
        try:
            router = _router(_cloud("paid", str(cloud_server.make_url("/v1"))),
                             _local_ollama(str(local_server.make_url("/v1"))))
            profile, text = await router.generate_local_text(
                "plan this", instruction="PLAN ONLY", task="planner",
                model="orion-planner:v2", response_format=schema, temperature=0.0,
                deadline_s=10)
            await router._openai_compatible_chat(
                router.settings.providers["local_ollama"], "hello", instruction="SYS")
            await router.aclose()
            return router, profile, text
        finally:
            await local_server.close()
            await cloud_server.close()

    router, profile, text = asyncio.run(flow())
    assert text == '{"steps": []}'
    assert cloud.calls == []
    planned, chatted = ollama.chats[0]["json"], ollama.chats[1]["json"]
    assert planned["format"] == schema and planned["model"] == "orion-planner:v2"
    assert planned["options"]["temperature"] == 0.0
    assert planned["messages"][0]["content"] == "PLAN ONLY"
    assert "format" not in chatted and chatted["model"] == "llama3.1"
    assert profile.name == "local_ollama:planner"
    assert "local_ollama:planner" in router._latency_ema


def test_local_only_generation_without_a_local_model_raises_instead_of_billing():
    from orion_core.providers import NoTextProviderError
    cloud = FakeOpenAI(text="cloud")

    async def flow():
        server = await _serve(cloud)
        try:
            router = _router(_cloud("paid", str(server.make_url("/v1"))))
            with pytest.raises(NoTextProviderError):
                await router.generate_local_text("plan this", instruction="PLAN ONLY")
            await router.aclose()
        finally:
            await server.close()

    asyncio.run(flow())
    assert cloud.calls == []


def _offline_rule_brain(router=None, telemetry=None, result=None):
    from types import SimpleNamespace
    from orion_core.data import ToolResult
    from orion_core.local_brain import LocalBrain

    calls = []

    async def dispatch(name, args):
        calls.append((name, args))
        return result if result is not None else ToolResult("Synthetic local result")

    brain = LocalBrain(_Bus(), memory=SimpleNamespace(),
                       dispatcher=SimpleNamespace(dispatch=dispatch),
                       knowledge=None, router=router, telemetry=telemetry)
    return brain, calls


@pytest.mark.parametrize("phrase,name,args", [
    ("hey open calculator", "open_app", {"app_name": "calculator"}),
    ("hello, open calculator", "open_app", {"app_name": "calculator"}),
    ("could you please open calculator", "open_app", {"app_name": "calculator"}),
    ("help me with reminders", "reminder", {"action": "list"}),
    ("help me with a list of reminders", "reminder", {"action": "list"}),
])
def test_offline_polite_task_reaches_dispatcher_once_instead_of_small_talk(phrase, name, args):
    brain, calls = _offline_rule_brain()
    assert asyncio.run(brain.try_known_response(phrase))
    assert calls == [(name, args)]


@pytest.mark.parametrize("allow_llm", [False, True])
def test_offline_rule_path_preserves_confirmation_gate_and_isolates_metrics(allow_llm):
    from types import SimpleNamespace
    from orion_core.data import ToolResult

    def broken_metric(*_):
        raise RuntimeError("synthetic optional metric fault")

    result = ToolResult("Please confirm this action.", media={"confirm": True})
    brain, calls = _offline_rule_brain(
        telemetry=SimpleNamespace(metrics=SimpleNamespace(incr=broken_metric)), result=result)
    assert asyncio.run(brain.respond("hey open calculator", allow_llm=allow_llm)) == result.text
    assert calls == [("open_app", {"app_name": "calculator"})]


@pytest.mark.parametrize("phrase", ["hello", "hey Orion", "what can you do?", "hey, what can you do?"])
def test_offline_standalone_greeting_and_capability_queries_still_answer(phrase):
    brain, calls = _offline_rule_brain()
    assert asyncio.run(brain.try_known_response(phrase))
    assert calls == []


def test_offline_unknown_preflight_leaves_model_fallback_and_counts_a_turn_once():
    from types import SimpleNamespace

    counts, generated = [], []

    async def generate(prompt, **kwargs):
        generated.append((prompt, kwargs))
        return object(), "An offline model answer."

    router = SimpleNamespace(local_text_profiles=lambda: [object()], generate_local_text=generate)
    brain, calls = _offline_rule_brain(
        router, SimpleNamespace(metrics=SimpleNamespace(incr=counts.append)))
    phrase = "Explain quantum entanglement in detail"
    assert asyncio.run(brain.try_known_response(phrase)) is None
    assert counts == generated == calls == []
    assert asyncio.run(brain.respond(phrase)) == "An offline model answer."
    assert counts == ["local_brain.turns"] and len(generated) == 1
    assert asyncio.run(brain.respond("")) == "I'm here."
    assert asyncio.run(brain.respond("", allow_llm=False)) is None


@pytest.mark.parametrize("url,local", [
    ("http://[::1]:11434/v1", True),
    ("http://localhost.invalid:11434/v1", False),
    ("http://127.0.0.1.invalid:11434/v1", False),
])
def test_offline_brain_uses_parsed_local_profiles_and_only_local_generation(url, local):
    from types import SimpleNamespace

    profile = AIProviderProfile(name="candidate", kind="openai_compatible", model="synthetic",
                                base_url=url, api_key="local", enabled=True)
    seen = []

    async def local_generate(prompt, **kwargs):
        seen.append((prompt, kwargs))
        return profile, "Local answer."

    router = SimpleNamespace(text_profiles=lambda: [profile], generate_local_text=local_generate,
                             generate_text=lambda *_a, **_k: pytest.fail("cloud-capable API used offline"))
    brain, _ = _offline_rule_brain(router)
    reply = asyncio.run(brain._try_local_llm("an open-ended thought"))
    assert reply == ("Local answer." if local else None)
    assert len(seen) == int(local)
    if seen:
        assert seen[0][1] == {"instruction": "", "task": "offline-chat"}


@pytest.mark.parametrize("known", [True, False])
def test_offline_worker_resolves_known_tasks_before_unavailable_model_but_keeps_unknown_fallback(known):
    import threading
    from types import SimpleNamespace
    from orion_core.live_worker import GenAILiveWorker

    provider_calls, spoken, persisted, rules = [], [], [], []

    async def preflight(text):
        rules.append(text)
        return "A local command result." if known else None

    async def provider(text):
        provider_calls.append(text)
        assert not known, "a dead configured provider delayed an already resolved local task"
        return SimpleNamespace(name="synthetic"), "A model answer."

    worker = SimpleNamespace(
        paused=False, stop_event=threading.Event(), _fallback_epoch=0, bus=_Bus(),
        router=SimpleNamespace(is_online=lambda: False, has_text_fallback=lambda: True,
                               generate_text=provider),
        local_brain=SimpleNamespace(try_known_response=preflight),
        _emit_state=lambda *_: None, _say=spoken.append,
        _persist_episode=lambda role, reply: persisted.append((role, reply)))
    asyncio.run(GenAILiveWorker._run_text_fallback(worker, "Synthetic request"))
    assert rules == ["Synthetic request"] and len(provider_calls) == int(not known)
    assert spoken == ["A local command result." if known else "A model answer."]
    assert persisted == [("orion", spoken[0])]


def test_cancelled_offline_known_task_does_not_speak_a_late_result():
    import threading
    from types import SimpleNamespace
    from orion_core.live_worker import GenAILiveWorker

    spoken = []
    worker = SimpleNamespace(paused=False, stop_event=threading.Event(), _fallback_epoch=0,
                             bus=_Bus(), _emit_state=lambda *_: None, _say=spoken.append)

    async def preflight(_):
        worker._fallback_epoch += 1
        return "Late result"

    worker.router = SimpleNamespace(is_online=lambda: False,
                                     has_text_fallback=lambda: pytest.fail("cancelled task retried"))
    worker.local_brain = SimpleNamespace(try_known_response=preflight)
    worker._persist_episode = lambda *_: pytest.fail("cancelled reply persisted")
    asyncio.run(GenAILiveWorker._run_text_fallback(worker, "Synthetic request"))
    assert spoken == []


@pytest.mark.parametrize("phrase", [
    "Do not close Notepad",
    "Please don't open calculator",
    "Could you please not close Notepad",
    "Explain how to open calculator",
    "Could you explain how to close Notepad",
    '"open calculator"',
    "`close Notepad`",
    "He said open calculator",
    "I said 'remember that the code is blue'",
    "Do not remember that the code is blue",
    "Never add a reminder in five minutes",
    "Explain memory allocation in Python",
    "Show me how to set a timer",
    "What is the command to call Jordan",
    "Do not save my workspace",
    "She asked me to remind her in five minutes",
    "What can you do to open calculator",
    "Don't tell me that's wrong",
    "Do not tell me the weather",
    "My friend said close Notepad",
    "My friend said that's wrong",
    "No, he said that's wrong",
    "No, I said 'that's wrong'",
])
def test_offline_mentioned_actions_are_unknown_and_never_mutate_in_either_reply_mode(phrase):
    from types import SimpleNamespace
    from orion_core.data import ToolResult

    generated, stored, corrected = [], [], []

    async def model(prompt, **_):
        generated.append(prompt)
        return object(), "A conversational model answer."

    async def correct(*args):
        corrected.append(args)
        return ToolResult("Synthetic correction")

    brain, calls = _offline_rule_brain(SimpleNamespace(
        local_text_profiles=lambda: [object()], generate_local_text=model))
    brain.memory = SimpleNamespace(remember=lambda *args: stored.append(args))
    brain.dispatcher.learning = SimpleNamespace(correct=correct)
    brain._last_fact_topic, brain._last_fact_answer = "Synthetic fact", "Earlier synthetic answer"
    assert asyncio.run(brain.try_known_response(phrase)) is None
    assert asyncio.run(brain.try_task(phrase)) is None
    assert generated == calls == stored == corrected == []
    assert asyncio.run(brain.respond(phrase)) == "A conversational model answer."
    assert generated == [phrase] and calls == stored == corrected == []


@pytest.mark.parametrize("phrase,tool,args", [
    ("thanks, open calculator", "open_app", {"app_name": "calculator"}),
    ("could you please close Notepad", "close_app", {"app_name": "Notepad"}),
    ("could you please remind me in five minutes to stretch", "reminder",
     {"action": "add", "phrase": "remind me in five minutes to stretch"}),
    ("please start a five minute timer", "reminder",
     {"action": "add", "phrase": "start a five minute timer"}),
    ("please save my workspace", "workspace_control", {"action": "save"}),
    ("what is on my screen?", "vision_analyse", {"action": "describe"}),
])
def test_direct_polite_offline_actions_and_screen_queries_dispatch_exactly_once(phrase, tool, args):
    brain, calls = _offline_rule_brain()
    assert asyncio.run(brain.try_known_response(phrase))
    assert calls == [(tool, args)]


@pytest.mark.parametrize("phrase,fact", [
    ("could you please remember that I prefer blue", "I prefer blue"),
    ("don't forget that I do not like tea", "I do not like tea"),
    ("remember that he said do not close Notepad", "he said do not close Notepad"),
    ("remember that it's actually blue", "it's actually blue"),
])
def test_direct_memory_request_preserves_negation_inside_the_requested_fact(phrase, fact):
    from types import SimpleNamespace

    brain, calls = _offline_rule_brain()
    stored = []
    brain.memory = SimpleNamespace(remember=lambda *args: stored.append(args))
    assert asyncio.run(brain.try_known_response(phrase)) == "Noted: " + fact + "."
    assert len(stored) == 1 and stored[0][2] == fact and calls == []


def test_offline_system_query_is_explicit_and_does_not_intercept_instructional_memory(monkeypatch):
    from types import SimpleNamespace

    read = []
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        cpu_percent=lambda **_: read.append("cpu") or 12,
        virtual_memory=lambda: SimpleNamespace(percent=25)))
    brain, calls = _offline_rule_brain()
    assert asyncio.run(brain.try_known_response("Explain memory allocation in Python")) is None
    assert read == calls == []
    reply = asyncio.run(brain.try_known_response("what is your CPU usage"))
    assert "12 percent" in reply and "25 percent" in reply
    assert read == ["cpu"] and calls == []
