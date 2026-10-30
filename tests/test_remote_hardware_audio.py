"""Paired hardware access and the bounded background-audio protocol."""

from __future__ import annotations

import asyncio

from orion_core.remote import RemoteGateway


class _Signal:
    def emit(self, *args):
        pass


class _Bus:
    def __getattr__(self, name):
        signal = _Signal()
        setattr(self, name, signal)
        return signal


class _Memory:
    def log_episode(self, *args):
        pass

    def prompt_context(self, limit=18):
        return ""


async def _client(gateway):
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(gateway._build_app()))
    await client.start_server()
    return client


async def _token(client, gateway):
    code = gateway.begin_pairing()
    credentials = await (await client.post("/v1/auth/pair", json={"code": code})).json()
    response = await client.post("/v1/auth/token", json={
        "device_id": credentials["device_id"],
        "refresh_token": credentials["refresh_token"],
    })
    return (await response.json())["access_token"]


def test_hardware_requires_pairing_and_keeps_missing_sensors_explicit(tmp_path, monkeypatch):
    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        client = await _client(gateway)
        try:
            unauthorised = await client.get("/api/hardware")
            assert unauthorised.status == 401
            from orion_core import hardware_telemetry
            monkeypatch.setattr(hardware_telemetry, "sample", lambda: {
                "cpu": {"name": "AMD Ryzen 5 3600", "load_percent": 42,
                        "temperature_c": None},
                "gpu": {"name": "NVIDIA GeForce RTX 3060 Ti",
                        "load_percent": 28, "vram_used_mb": 2048,
                        "vram_total_mb": 8192, "temperature_c": 56},
            })
            token = await _token(client, gateway)
            response = await client.get("/api/hardware", headers={
                "Authorization": "Bearer " + token})
            data = await response.json()
            assert response.status == 200
            assert data["cpu"]["temperature_c"] is None
            assert data["gpu"]["vram_total_mb"] == 8192
        finally:
            await client.close()
    asyncio.run(flow())


def test_hardware_sampler_does_not_queue_behind_a_stuck_sensor(monkeypatch):
    from orion_core import hardware_telemetry

    class BusyLock:
        def acquire(self, blocking=True):
            assert blocking is False
            return False

    previous = {"sampled_at": 123, "cpu": {"load_percent": 42}}
    monkeypatch.setattr(hardware_telemetry, "_lock", BusyLock())
    monkeypatch.setattr(hardware_telemetry, "_last", previous)
    assert hardware_telemetry.sample() == {**previous, "stale": True}
    assert hardware_telemetry.sample() is not previous

    monkeypatch.setattr(hardware_telemetry, "_last", None)
    unavailable = hardware_telemetry.sample()
    assert unavailable["sampled_at"] is None
    assert unavailable["stale"] is True
    assert unavailable["gpu"]["available"] is False
    assert unavailable["cpu"]["temperature_c"] is None


def test_audio_socket_requires_pairing_and_returns_spoken_reply(tmp_path):
    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        gateway._transcribe_blocking = lambda body: ("Orion check status", "")
        async def answer(message, device_id, *, on_widget=None):
            assert message == "check status"
            assert device_id
            return "All systems ready.", "test"
        gateway._answer = answer
        client = await _client(gateway)
        try:
            unauthorised = await client.get("/v1/audio/stream")
            assert unauthorised.status == 401
            token = await _token(client, gateway)
            socket = await client.ws_connect("/v1/audio/stream", headers={
                "Authorization": "Bearer " + token})
            await socket.send_bytes(b"\x00\x00" * 1600)
            await socket.send_json({"type": "commit"})
            transcript = await socket.receive_json(timeout=3)
            speech = await socket.receive_json(timeout=3)
            assert transcript == {"type": "transcript", "text": "Orion check status"}
            assert speech["type"] == "tts_text"
            assert speech["text"] == "All systems ready."
            await socket.close()
        finally:
            await client.close()
    asyncio.run(flow())


def test_speaking_interruption_accepts_only_the_exact_wake_stop_phrase(tmp_path):
    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        heard = ["ORION stop"]
        gateway._transcribe_blocking = lambda body: (heard[0], "")
        async def never_dispatch(*args, **kwargs):
            raise AssertionError("A speaking interruption must never execute a tool")
        gateway._answer = never_dispatch
        client = await _client(gateway)
        try:
            token = await _token(client, gateway)
            socket = await client.ws_connect("/v1/audio/stream", headers={"Authorization": "Bearer " + token})
            for words, expected in (("ORION stop", "interrupt"),
                                    ("ORION delete all files", "status"),
                                    ("The guide says ORION stop", "status")):
                heard[0] = words
                await socket.send_bytes(b"\x00\x00" * 1600)
                await socket.send_json({"type": "commit", "interrupt_only": True})
                response = await socket.receive_json(timeout=3)
                assert response["type"] == expected
            await socket.close()
        finally:
            await client.close()
    asyncio.run(flow())


def test_background_voice_keeps_full_history_and_relays_display_cards(tmp_path):
    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        gateway._transcribe_blocking = lambda body: ("Orion explain the result", "")
        reply = "A complete detailed explanation. " * 500 + "FINAL WORDS."
        card = {"operation": "display", "title": "Result", "html": "<p>42</p>"}
        async def answer(message, device_id, *, on_widget):
            on_widget(card)
            return reply, "test"
        gateway._answer = answer
        client = await _client(gateway)
        try:
            token = await _token(client, gateway)
            socket = await client.ws_connect("/v1/audio/stream", headers={"Authorization": "Bearer " + token})
            await socket.send_bytes(b"\x00\x00" * 1600)
            await socket.send_json({"type": "commit"})
            assert (await socket.receive_json(timeout=3))["type"] == "transcript"
            response = await socket.receive_json(timeout=3)
            assert response["text"] == reply and response["widgets"] == [card]
            assert response["user_text"] == "explain the result"
            history = await (await client.get("/api/history?limit=10",
                headers={"Authorization": "Bearer " + token})).json()
            assert history["items"][-1]["text"] == reply
            assert history["items"][-1]["id"] == response["hid"]
            assert history["items"][-2]["id"] == response["user_hid"]
            await socket.close()
        finally:
            await client.close()
    asyncio.run(flow())


def test_audio_socket_stays_alive_during_a_slow_reply(tmp_path, monkeypatch):
    """Provider latency must not starve aiohttp's WebSocket PONG receiver."""
    from aiohttp import web

    response_class = web.WebSocketResponse

    def fast_heartbeat(*args, **kwargs):
        kwargs["heartbeat"] = 0.1
        return response_class(*args, **kwargs)

    monkeypatch.setattr(web, "WebSocketResponse", fast_heartbeat)

    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        gateway._transcribe_blocking = lambda body: ("Orion check status", "")

        async def answer(message, device_id, *, on_widget=None):
            await asyncio.sleep(0.5)
            return "All systems ready.", "test"

        gateway._answer = answer
        client = await _client(gateway)
        try:
            token = await _token(client, gateway)
            socket = await client.ws_connect("/v1/audio/stream", headers={
                "Authorization": "Bearer " + token})
            await socket.send_bytes(b"\x00\x00" * 1600)
            await socket.send_json({"type": "commit"})
            assert (await socket.receive_json(timeout=2))["type"] == "transcript"
            reply = await socket.receive_json(timeout=2)
            assert reply["type"] == "tts_text"
            assert reply["text"] == "All systems ready."
            await socket.send_json({"type": "ping"})
            assert (await socket.receive_json(timeout=2))["type"] == "pong"
            await socket.close()
        finally:
            await client.close()

    asyncio.run(flow())
