"""
Tests for the phone-facing ORB surface: the /face page, the /api/events live
state mirror (SSE), and the upgraded PWA shell.

* /face serves the desktop's Three.js face (or the canvas fallback) with the
  postMessage bridge injected, and a CSP that admits only the jsdelivr CDN.
* /api/events requires a valid access token and streams the current state
  first; bus signals fan out to subscribers; a stalled queue never blocks.
* The PWA page keeps the auth flow and adds the orb iframe, speech synthesis
  and speech recognition wiring.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.remote import REMOTE_PAGE_HTML, RemoteGateway


class _Signal:
    def __init__(self):
        self.emitted = []
        self._handlers = []

    def emit(self, *payload):
        self.emitted.append(payload)
        for handler in self._handlers:
            handler(*payload)

    def connect(self, handler):
        self._handlers.append(handler)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _StubMemory:
    def log_episode(self, role, content):
        pass

    def prompt_context(self, limit=18):
        return ""


def _gateway(tmp_path) -> RemoteGateway:
    return RemoteGateway(None, _StubMemory(), _StubBus(), config_dir=tmp_path)


async def _client(gateway):
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(gateway._build_app()))
    await client.start_server()
    return client


def _access_token(gateway) -> str:
    code = gateway.auth.begin_pairing()
    device_id, refresh = gateway.auth.complete_pairing(code, "test-phone")
    token, _expires = gateway.auth.issue_access(device_id, refresh)
    return token


def test_pwa_page_has_orb_voice_and_auth():
    assert '<canvas id="vector"' in REMOTE_PAGE_HTML       # live signal stage
    assert "speechSynthesis" in REMOTE_PAGE_HTML          # voice replies
    assert "SpeechRecognition" in REMOTE_PAGE_HTML        # voice input
    assert "EventSource" in REMOTE_PAGE_HTML              # live orb mirror
    assert "/v1/auth/pair" in REMOTE_PAGE_HTML            # pairing flow kept


def test_face_route_serves_orb_with_bridge_and_cdn_csp(tmp_path):
    async def scenario():
        gateway = _gateway(tmp_path)
        client = await _client(gateway)
        try:
            res = await client.get("/face")
            body = await res.text()
            assert res.status == 200
            assert "addEventListener('message'" in body   # postMessage bridge
            csp = res.headers.get("Content-Security-Policy", "")
            assert "cdn.jsdelivr.net" in csp
            assert "default-src 'self'" in csp
        finally:
            await client.close()
    asyncio.run(scenario())


def test_events_requires_access_token(tmp_path):
    async def scenario():
        gateway = _gateway(tmp_path)
        client = await _client(gateway)
        try:
            res = await client.get("/api/events")
            assert res.status == 401
            res = await client.get("/api/events?token=garbage")
            assert res.status == 401
        finally:
            await client.close()
    asyncio.run(scenario())


def test_events_streams_state_then_bus_updates(tmp_path):
    async def scenario():
        gateway = _gateway(tmp_path)
        gateway._wire_bus_events()
        token = _access_token(gateway)
        client = await _client(gateway)
        try:
            res = await client.get(f"/api/events?token={token}")
            assert res.status == 200
            assert res.headers["Content-Type"].startswith("text/event-stream")
            line = await asyncio.wait_for(res.content.readline(), timeout=5)
            first = json.loads(line.decode().removeprefix("data: "))
            assert first == {"type": "state", "value": "STANDBY"}
            await res.content.readline()                  # blank separator
            gateway.bus.state.emit("LISTENING")
            line = await asyncio.wait_for(res.content.readline(), timeout=5)
            update = json.loads(line.decode().removeprefix("data: "))
            assert update == {"type": "state", "value": "LISTENING"}
        finally:
            await client.close()
    asyncio.run(scenario())


def test_stop_is_prompt_with_a_phone_stream_open(tmp_path, monkeypatch):
    """An open /api/events stream never ends by itself, and aiohttp's graceful
    shutdown waited 60 s for it — past the app's 25 s shutdown watchdog
    ("Still inside: remote gateway"). stop() must end the stream and return."""
    import socket
    import time as _time

    import aiohttp

    monkeypatch.setenv("ORION_REMOTE_FIREWALL", "0")
    monkeypatch.delenv("ORION_REMOTE_TLS", raising=False)

    async def scenario():
        gateway = _gateway(tmp_path)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            gateway.port = probe.getsockname()[1]
        gateway.host = "127.0.0.1"
        token = _access_token(gateway)
        await gateway.start()
        async with aiohttp.ClientSession() as session:
            res = await session.get(
                f"http://127.0.0.1:{gateway.port}/api/events?token={token}")
            assert res.status == 200
            await asyncio.wait_for(res.content.readline(), timeout=5)
            started = _time.monotonic()
            await asyncio.wait_for(gateway.stop(), timeout=10)
            assert _time.monotonic() - started < 5.0
            await gateway.stop()                          # idempotent
            res.close()
    asyncio.run(scenario())


def test_bus_amplitude_is_coalesced_not_per_sample(tmp_path):
    async def scenario():
        gateway = _gateway(tmp_path)
        gateway._wire_bus_events()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        gateway._event_subs.add(queue)
        for i in range(200):                              # audio-rate burst
            gateway.bus.amplitude.emit(i / 200)
        assert queue.qsize() == 0                         # nothing shipped yet
        ticker = asyncio.create_task(gateway._amplitude_ticker())
        try:
            item = await asyncio.wait_for(queue.get(), timeout=2)
            assert item["type"] == "amplitude"
            assert item["value"] == round(199 / 200, 3)   # only the latest
            assert queue.qsize() == 0
        finally:
            ticker.cancel()
    asyncio.run(scenario())


def test_full_queue_never_blocks_the_publisher(tmp_path):
    gateway = _gateway(tmp_path)
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    gateway._event_subs.add(queue)
    for _ in range(10):
        gateway._push_event({"type": "state", "value": "X"})   # must not raise
    assert queue.qsize() == 1


def test_close_reaches_a_stalled_stream_with_a_full_queue(tmp_path):
    gateway = _gateway(tmp_path)
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    gateway._event_subs.add(queue)
    gateway._push_event({"type": "state", "value": "X"})       # now full
    gateway._close_event_streams()
    assert queue.get_nowait() is None                         # the stop signal


def test_face_page_falls_back_without_webengine(tmp_path, monkeypatch):
    gateway = _gateway(tmp_path)
    import builtins
    real_import = builtins.__import__

    def deny_face(name, *args, **kwargs):
        if "face3d" in name:
            raise ImportError("no gui on this node")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_face)
    page = gateway._face_page()
    assert "canvas" in page                               # 2-D fallback orb
    assert "addEventListener('message'" in page           # same bridge contract


# ── the phone's face: the same page the desktop shows, with its library ──────

def test_face_page_fills_the_import_map_and_serves_its_library(tmp_path):
    async def scenario():
        gateway = _gateway(tmp_path)
        client = await _client(gateway)
        try:
            body = await (await client.get("/face")).text()
            assert "__THREE_BASE__" not in body and "__IDLE_FPS__" not in body
            files, version = gateway._three_files()
            if not files:                      # three.js not shipped: CDN fallback
                assert "cdn.jsdelivr.net" in body
                return
            assert f"/three/{version}/build/three.module.js" in body
            lib = await client.get(f"/three/{version}/build/three.module.js")
            assert lib.status == 200
            assert "javascript" in lib.headers.get("Content-Type", "")
            for bad in (f"/three/{version}/../../orion_core/remote.py",
                        f"/three/{version}/%2e%2e/%2e%2e/orion.py",
                        "/three/0.0.0/build/three.module.js"):
                assert (await client.get(bad)).status == 404, bad
        finally:
            await client.close()
    asyncio.run(scenario())


def test_the_catch_up_list_carries_the_owners_approval_token(tmp_path):
    async def scenario():
        gateway = _gateway(tmp_path)
        token = _access_token(gateway)
        device = gateway.auth.verify_access(token)
        conf = gateway.confirmations.create("send_email", {"to": "x"}, device, "Email x")
        client = await _client(gateway)
        try:
            res = await client.get("/api/confirmations",
                                   headers={"Authorization": f"Bearer {token}"})
            pending = (await res.json())["pending"]
            assert pending and pending[0]["id"] == conf.id
            assert pending[0]["token"] == conf.token
        finally:
            await client.close()
    asyncio.run(scenario())


def test_a_phone_action_goes_only_to_the_phone_that_asked(tmp_path):
    gateway = _gateway(tmp_path)
    mine, other = asyncio.Queue(), asyncio.Queue()
    gateway._event_subs.update({mine, other})
    gateway._event_devices.update({mine: "phone-a", other: "phone-b"})
    gateway._wire_bus_events()
    with gateway.tool_gate.for_device("phone-a"):
        gateway.bus.phone_action.emit({"kind": "call", "number": "123"})
    assert mine.qsize() == 1 and other.qsize() == 0
    # An action ORION starts himself goes to the phone used last.
    gateway._last_active_device = "phone-b"
    gateway.bus.phone_action.emit({"kind": "sms", "number": "123"})
    assert other.qsize() == 1 and mine.qsize() == 1


def test_the_page_only_repairs_when_the_desktop_refuses_the_device():
    assert "r.status===401||r.status===403" in REMOTE_PAGE_HTML
    assert "got==='denied'&&await pair()" in REMOTE_PAGE_HTML
    assert "new URLSearchParams(location.search).get('pair')" in REMOTE_PAGE_HTML
    assert "catchUpConfirms" in REMOTE_PAGE_HTML and "resolvedHere[d.id]" in REMOTE_PAGE_HTML


# ── the real page in a real browser ──────────────────────────────────────────
#
# Checking that the embedded script parses says nothing about what it does.
# These drive the page the phone loads in headless Chromium (Playwright),
# against a scripted gateway whose every event the test controls, plus one run
# against the real gateway. They skip cleanly where Playwright or its
# Chromium is not installed. A desktop Chromium is not the Android WebView:
# these establish behaviour, not device performance.

#: Credentials the page finds in localStorage, so it never asks to pair.
_SEED = ("localStorage.setItem('orion_device','dev-1');"
         "localStorage.setItem('orion_refresh','refresh-1');"
         "localStorage.setItem('orion_voice','0');")

#: A reply the size a phone really receives, not a one-word stub.
_REPLY = ("Here is what I found. " + "The forecast holds steady through the evening, "
          "with light wind from the west and a chance of rain after midnight. ") * 4


class _ScriptedGateway:
    """Just enough of the phone gateway for the real page to run, with each
    streamed reply played by a script the test supplies."""

    def __init__(self):
        self.scripts = deque()             # async (send, message) for the next turns
        self.events: asyncio.Queue = asyncio.Queue()
        self.event_streams = 0
        self.cancels = []
        self.history = []                  # [{"id", "role", "text"}] oldest first
        self.turns = 0

    def app(self):
        from aiohttp import web

        async def page(_request):
            return web.Response(text=REMOTE_PAGE_HTML, content_type="text/html")

        async def token(_request):
            return web.json_response({"ok": True, "access_token": "access-1",
                                      "expires_in": 900})

        async def health(_request):
            return web.json_response({"ok": True})

        async def confirmations(_request):
            return web.json_response({"ok": True, "pending": []})

        app = web.Application()
        app.router.add_get("/", page)
        app.router.add_post("/v1/auth/token", token)
        app.router.add_get("/api/health", health)
        app.router.add_get("/api/confirmations", confirmations)
        app.router.add_get("/api/events", self._events)
        app.router.add_get("/api/history", self._history)
        app.router.add_post("/api/chat/stream", self._stream)
        app.router.add_post("/api/chat/cancel", self._cancel)
        return app

    async def _events(self, request):
        from aiohttp import web
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        self.event_streams += 1
        try:
            await response.write(b'data: {"type":"state","value":"LISTENING"}\n\n')
            while (item := await self.events.get()) is not None:
                await response.write(f"data: {json.dumps(item)}\n\n".encode())
        except Exception:
            pass
        return response

    async def _history(self, request):
        from aiohttp import web
        before = int(request.query.get("before") or 0) or None
        limit = int(request.query.get("limit") or 50)
        older = [e for e in self.history if before is None or e["id"] < before]
        chosen = older[len(older) - limit:] if limit else []
        return web.json_response({"ok": True, "items": chosen, "more": len(older) > len(chosen),
                                  "next": len(self.history) + 1})

    async def _cancel(self, request):
        from aiohttp import web
        self.cancels.append((await request.json())["request_id"])
        return web.json_response({"ok": True, "status": "cancelled"})

    async def _stream(self, request):
        from aiohttp import web
        message = (await request.json())["message"]
        self.turns += 1
        request_id = f"REQ-{self.turns}"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

        async def send(**event):
            event.setdefault("request_id", request_id)
            await response.write(f"data: {json.dumps(event)}\n\n".encode())

        script = self.scripts.popleft() if self.scripts else self._quick
        try:
            await script(send, message)
        except Exception:
            pass                            # the page hung up (Stop)
        return response

    async def _quick(self, send, message):
        """An ordinary turn: start, one early delta, the whole reply."""
        user_hid = len(self.history) + 1
        reply = f"Reply to {message}. {_REPLY}"
        self.history += [{"id": user_hid, "role": "user", "text": message},
                         {"id": user_hid + 1, "role": "orion", "text": reply}]
        await send(type="start", seq=0, user_hid=user_hid)
        await send(type="delta", seq=1, text=reply[:40])
        await send(type="done", seq=2, reply=reply, provider="scripted", hid=user_hid + 1)


async def _open_page(browser, url):
    page = await browser.new_page(viewport={"width": 390, "height": 844})
    await page.add_init_script(_SEED)
    await page.goto(url)
    await page.wait_for_function("typeof access==='string'&&access.length>0")
    return page


async def _launch(playwright):
    try:
        return await playwright.chromium.launch()
    except Exception as exc:                       # browsers not installed
        pytest.skip(f"Playwright has no local Chromium: {str(exc).splitlines()[0][:120]}")


def _streaming_text(page):
    return page.evaluate(
        "(()=>{const n=document.querySelector('.msg.orion.streaming');return n?n.textContent:null})()")


async def _send(page, text):
    await page.fill("#m", text)
    await page.press("#m", "Enter")


def test_the_phone_page_streams_restarts_and_drops_stale_events():
    pytest.importorskip("playwright.async_api")
    from aiohttp.test_utils import TestServer
    from playwright.async_api import async_playwright

    async def scenario():
        gateway = _ScriptedGateway()
        server = TestServer(gateway.app())
        await server.start_server()
        try:
            async with async_playwright() as playwright:
                browser = await _launch(playwright)
                try:
                    page = await _open_page(browser, str(server.make_url("/")))
                    await _first_text_before_completion(page, gateway)
                    await _restart_clears_the_partial_answer(page, gateway)
                    await _duplicates_and_strays_are_ignored(page, gateway)
                    await _stop_cancels_the_reply(page, gateway)
                finally:
                    await gateway.events.put(None)
                    await browser.close()
        finally:
            await server.close()

    asyncio.run(scenario())


async def _first_text_before_completion(page, gateway):
    release = asyncio.Event()

    async def burst(send, message):
        await send(type="start", seq=0, user_hid=1)
        for i in range(200):                        # a fast model's burst of tokens
            await send(type="delta", seq=i + 1, text=f"w{i} ")
        await release.wait()
        await send(type="done", seq=201, reply="The final answer.", provider="scripted", hid=2)

    gateway.scripts.append(burst)
    # Count repaints of the streaming bubble: one per frame, not per token.
    await page.evaluate("""()=>{window.__paints=0;new MutationObserver(records=>{
        for(const r of records){const t=r.target;
          if(t.classList&&t.classList.contains('streaming'))window.__paints++;}})
        .observe(document.getElementById('log'),{subtree:true,childList:true,characterData:true});}""")
    await _send(page, "hello")
    await page.wait_for_function(
        "(()=>{const n=document.querySelector('.msg.orion.streaming');"
        "return n&&n.textContent.includes('w199')})()")
    # Visible text, and the reply is not finished: the stop control is live,
    # nothing has been announced, and the bubble is marked busy.
    assert (await _streaming_text(page)).startswith("w0 w1 w2")
    assert await page.is_visible("#stop")
    assert await page.text_content("#say") == ""
    assert await page.get_attribute(".msg.orion.streaming", "aria-busy") == "true"
    paints = await page.evaluate("window.__paints")
    assert 1 <= paints <= 20, paints                 # 200 tokens, a handful of frames
    assert await page.evaluate("document.activeElement.id") == "m"   # focus stays put
    release.set()
    await page.wait_for_function("document.getElementById('say').textContent==='The final answer.'")
    assert await _streaming_text(page) is None
    last = await page.evaluate("[...document.querySelectorAll('.msg.orion')].pop().textContent")
    assert last == "The final answer."
    assert not await page.is_visible("#stop")


async def _restart_clears_the_partial_answer(page, gateway):
    fail, finish = asyncio.Event(), asyncio.Event()

    async def restarting(send, message):
        await send(type="start", seq=0)
        await send(type="delta", seq=1, text="A wrong partial answer")
        await fail.wait()
        await send(type="restart", seq=2, reason="the first provider failed")
        await send(type="delta", seq=3, text="Right")
        await finish.wait()
        await send(type="done", seq=4, reply="Right answer.", provider="fallback")

    gateway.scripts.append(restarting)
    await _send(page, "try again")
    await page.wait_for_function(
        "(()=>{const n=document.querySelector('.msg.orion.streaming');"
        "return n&&n.textContent==='A wrong partial answer'})()")
    fail.set()
    await page.wait_for_function(
        "(()=>{const n=document.querySelector('.msg.orion.streaming');"
        "return n&&n.textContent==='Right'})()")
    traces = await page.evaluate("[...document.querySelectorAll('.trace')].map(t=>t.textContent)")
    assert any("Answer restarted" in t and "first provider failed" in t for t in traces)
    finish.set()
    await page.wait_for_function("document.getElementById('say').textContent==='Right answer.'")
    bubbles = await page.evaluate("[...document.querySelectorAll('.msg.orion')].map(n=>n.textContent)")
    assert not any("wrong partial" in b for b in bubbles)


async def _duplicates_and_strays_are_ignored(page, gateway):
    finish = asyncio.Event()

    async def noisy(send, message):
        await send(type="start", seq=0)
        await send(type="start", seq=0, request_id="REQ-OTHER")      # a second start
        await send(type="delta", seq=1, text="A")
        await send(type="delta", seq=1, text="A")                    # duplicate
        await send(type="delta", seq=2, text="B")
        await send(type="delta", seq=3, text="X", request_id="REQ-OLD")  # another request
        await send(type="delta", seq=2, text="B")                    # duplicate
        await send(type="delta", seq=1, text="stale")                # out of date
        await send(type="delta", seq=3, text="C")
        await finish.wait()
        await send(type="done", seq=4, reply="ABC", provider="scripted")

    gateway.scripts.append(noisy)
    await _send(page, "spell it")
    await page.wait_for_function(
        "(()=>{const n=document.querySelector('.msg.orion.streaming');"
        "return n&&n.textContent.length>=3})()")
    await page.wait_for_timeout(100)                # any stray would have landed by now
    assert await _streaming_text(page) == "ABC"
    finish.set()
    await page.wait_for_function("document.getElementById('say').textContent==='ABC'")


async def _stop_cancels_the_reply(page, gateway):
    async def endless(send, message):
        await send(type="start", seq=0)
        await send(type="delta", seq=1, text="Thinking it through")
        await asyncio.Event().wait()

    gateway.scripts.append(endless)
    await _send(page, "take your time")
    await page.wait_for_function(
        "(()=>{const n=document.querySelector('.msg.orion.streaming');"
        "return n&&n.textContent==='Thinking it through'})()")
    await page.click("#stop")
    await page.wait_for_function(
        "[...document.querySelectorAll('.msg.meta')].some(n=>n.textContent==='Stopped.')")
    deadline = time.monotonic() + 5
    while not gateway.cancels and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert gateway.cancels == [f"REQ-{gateway.turns}"]            # this request, by id
    assert await _streaming_text(page) is None                    # settled, text kept
    assert await page.evaluate(
        "[...document.querySelectorAll('.msg.orion')].some(n=>n.textContent==='Thinking it through')")
    assert not await page.is_visible("#stop")


def test_a_long_phone_session_keeps_the_page_bounded_and_the_reader_in_place():
    pytest.importorskip("playwright.async_api")
    from aiohttp.test_utils import TestServer
    from playwright.async_api import async_playwright

    async def turns(page, first, count):
        """*count* ordinary turns through the real form, at most four in flight."""
        await page.evaluate("""async ([first,count])=>{
            const f=document.getElementById('f'),m=document.getElementById('m');
            for(let i=first;i<first+count;i++){
              while(inflight.size>=4)await new Promise(r=>setTimeout(r,5));
              m.value='Message '+i+': could you check the weather for the weekend please';
              f.requestSubmit();await new Promise(r=>setTimeout(r,0));}
            while(inflight.size)await new Promise(r=>setTimeout(r,5));}""", [first, count])
        await page.wait_for_timeout(120)            # a few frames for the trim

    async def thoughts(gateway, first, count):
        for i in range(first, first + count):
            await gateway.events.put({"type": "thought", "text": f"Checking source {i}"})
            if i % 30 == 29:
                await asyncio.sleep(0.03)
        await asyncio.sleep(0.15)

    def census(page):
        return page.evaluate("""({msgs:log.querySelectorAll('.msg:not(.confirm)').length,
            traces:log.querySelectorAll('.trace').length,
            confirms:log.querySelectorAll('.confirm').length,
            total:log.children.length, stuck:nearBottom()})""")

    async def scenario():
        gateway = _ScriptedGateway()
        server = TestServer(gateway.app())
        await server.start_server()
        try:
            async with async_playwright() as playwright:
                browser = await _launch(playwright)
                try:
                    page = await _open_page(browser, str(server.make_url("/")))
                    deadline = time.monotonic() + 5
                    while not gateway.event_streams and time.monotonic() < deadline:
                        await asyncio.sleep(0.02)
                    # An approval raised early, then left waiting all session.
                    await gateway.events.put({
                        "type": "confirm", "id": "conf-1", "token": "tok-1",
                        "tool": "messaging", "summary": "send a message",
                        "parameterDiff": "{\"to\": \"Sam\"}"})
                    await page.wait_for_selector(".confirm")

                    # Hundreds of messages and activity lines, read at the bottom.
                    await turns(page, 0, 150)
                    await thoughts(gateway, 0, 200)
                    counts = await census(page)
                    assert counts["stuck"]
                    assert counts["msgs"] <= await page.evaluate("MAX_MSGS")
                    assert counts["traces"] <= await page.evaluate("MAX_TRACES")
                    assert counts["confirms"] == 1          # the approval survived
                    assert await page.is_visible("#earlier")

                    # Three replies are still being written when the reader
                    # scrolls up to an earlier message. Sending moves the view
                    # to the bottom (the reader just acted); what arrives after
                    # that must not move it, however much there is.
                    release = asyncio.Event()

                    async def held(send, message):
                        user_hid = len(gateway.history) + 1
                        reply = f"Reply to {message}. {_REPLY}"
                        gateway.history += [
                            {"id": user_hid, "role": "user", "text": message},
                            {"id": user_hid + 1, "role": "orion", "text": reply}]
                        await send(type="start", seq=0, user_hid=user_hid)
                        await release.wait()       # first text arrives once scrolled up
                        for i in range(6):
                            await send(type="delta", seq=1 + i, text=reply[40 * i:40 * i + 40])
                            await asyncio.sleep(0.02)
                        await send(type="done", seq=7, reply=reply, provider="scripted",
                                   hid=user_hid + 1)

                    gateway.scripts.extend([held, held, held])
                    for i in range(3):
                        await _send(page, f"Still writing {i}")
                    await page.wait_for_function(
                        "inflight.size===3&&[...inflight].every(s=>s.rid)")
                    anchor_top = await page.evaluate("""(()=>{
                        const n=[...log.querySelectorAll('.msg.user')].slice(-8)[0];n.id='anchor';
                        log.scrollTop=n.offsetTop-12;
                        return n.getBoundingClientRect().top})()""")
                    assert not (await census(page))["stuck"]
                    release.set()
                    await gateway.events.put({            # a new approval, raised now
                        "type": "confirm", "id": "conf-2", "token": "tok-2",
                        "tool": "outlook_mail", "summary": "send an email",
                        "parameterDiff": "{}"})
                    await thoughts(gateway, 200, 400)       # past the scrolled-up cap
                    await page.wait_for_function("inflight.size===0")
                    await page.wait_for_timeout(120)
                    counts = await census(page)
                    assert counts["total"] <= await page.evaluate("HARD_CAP")
                    assert counts["confirms"] == 2
                    assert await page.evaluate("document.getElementById('anchor')!==null")
                    moved = await page.evaluate(
                        "document.getElementById('anchor').getBoundingClientRect().top")
                    assert abs(moved - anchor_top) < 2, (anchor_top, moved)
                    assert not counts["stuck"]

                    # "Show earlier" pages older messages in above the reader
                    # without moving them either.
                    top_node = await page.evaluate("""(()=>{log.scrollTop=0;
                        const n=[...log.children].find(c=>c.dataset&&c.dataset.hid);n.id='top-msg';
                        return [+n.dataset.hid,n.getBoundingClientRect().top]})()""")
                    await page.click("#earlier")
                    await page.wait_for_function("log.querySelectorAll('.msg.hist').length>0")
                    loaded = await page.evaluate(
                        "[...log.querySelectorAll('.msg.hist')].map(n=>+n.dataset.hid)")
                    assert loaded and max(loaded) < top_node[0] and len(loaded) <= 50
                    after = await page.evaluate(
                        "document.getElementById('top-msg').getBoundingClientRect().top")
                    assert abs(after - top_node[1]) < 2

                    # Back at the bottom, the page trims to its bound again.
                    await page.evaluate("log.scrollTop=log.scrollHeight")
                    await page.wait_for_timeout(120)
                    counts = await census(page)
                    assert counts["msgs"] <= await page.evaluate("MAX_MSGS")
                    assert counts["confirms"] == 2
                finally:
                    await gateway.events.put(None)
                    await browser.close()
        finally:
            await server.close()

    asyncio.run(scenario())


def test_the_real_gateway_streams_to_the_real_page(tmp_path):
    """End to end: the page, the real gateway and its auth, and a router that
    streams. The first words must be on screen well before the answer ends."""
    pytest.importorskip("playwright.async_api")
    from aiohttp.test_utils import TestServer
    from playwright.async_api import async_playwright

    class _Router:
        def has_text_fallback(self):
            return True

        async def generate_text(self, prompt, system_extra="", *, on_delta=None, on_restart=None):
            words = [f"word{i} " for i in range(12)]
            for word in words:
                on_delta(word)
                await asyncio.sleep(0.06)
            return type("P", (), {"name": "streaming-model"})(), "".join(words).strip()

    async def scenario():
        gateway = RemoteGateway(_Router(), _StubMemory(), _StubBus(), config_dir=tmp_path)
        device_id, refresh = gateway.auth.complete_pairing(gateway.begin_pairing(), "phone")
        server = TestServer(gateway._build_app())
        await server.start_server()
        try:
            async with async_playwright() as playwright:
                browser = await _launch(playwright)
                try:
                    page = await browser.new_page(viewport={"width": 390, "height": 844})
                    await page.add_init_script(
                        f"localStorage.setItem('orion_device','{device_id}');"
                        f"localStorage.setItem('orion_refresh','{refresh}');"
                        "localStorage.setItem('orion_voice','0');")
                    await page.goto(str(server.make_url("/")))
                    # The gateway's real CSP forbids eval, which Playwright's
                    # wait_for_function needs; poll through evaluate instead,
                    # so the page is proven under the policy phones get.
                    deadline = time.monotonic() + 10
                    while not await page.evaluate(
                            "()=>typeof access==='string'&&access.length>0"):
                        assert time.monotonic() < deadline, "the page never authenticated"
                        await asyncio.sleep(0.05)
                    times = await page.evaluate("""async ()=>{const t0=performance.now();let first=0;
                        const seen=new MutationObserver(()=>{if(!first&&document.querySelector(
                          '.msg.orion.streaming')&&document.querySelector('.msg.orion.streaming')
                          .textContent)first=performance.now()-t0;});
                        seen.observe(log,{subtree:true,childList:true,characterData:true});
                        const m=document.getElementById('m');m.value='count for me';
                        document.getElementById('f').requestSubmit();
                        while(document.getElementById('say').textContent.indexOf('word11')<0)
                          await new Promise(r=>setTimeout(r,10));
                        seen.disconnect();return [first,performance.now()-t0];}""")
                    first, done = times
                    assert 0 < first < done - 300, times
                    assert await page.text_content("#model") == "ROUTE streaming-model"
                finally:
                    await browser.close()
        finally:
            await server.close()

    asyncio.run(scenario())
