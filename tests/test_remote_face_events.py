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
from pathlib import Path

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
    assert 'iframe id="face" src="/face"' in REMOTE_PAGE_HTML
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
