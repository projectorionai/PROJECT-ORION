"""Regression checks for live gateway revocation and bounded voice input."""

from __future__ import annotations

import asyncio
import io
import struct
import wave
from types import SimpleNamespace

import pytest

from orion_core.remote import RemoteGateway
import orion_core.remote_mobile_ws as mobile_ws


class _Log:
    def emit(self, *_args):
        pass


def _gateway(tmp_path):
    return RemoteGateway(None, object(), SimpleNamespace(log=_Log()), config_dir=tmp_path)


def _wav(rate: int, frames: int = 100) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)  # write a valid container, then change the rate
        writer.writeframes(b"\0\0" * frames)
    body = bytearray(output.getvalue())
    struct.pack_into("<I", body, 24, rate)
    return bytes(body)


def test_tunnel_rate_limit_isolated_by_verified_phone(tmp_path):
    gateway = _gateway(tmp_path)
    gateway._RATE_MAX = 1
    identities = []
    for _ in range(2):
        code = gateway.begin_pairing()
        device_id, refresh = gateway.auth.complete_pairing(code)
        token, _ = gateway.auth.issue_access(device_id, refresh)
        identities.append((device_id, token))

    def request(token: str):
        return SimpleNamespace(
            headers={"Authorization": f"Bearer {token}",
                     "X-Forwarded-For": "203.0.113.20"}, remote="127.0.0.1")

    first, second = (gateway._client_key(request(token)) for _, token in identities)
    assert first != second
    assert gateway._rate_ok(first)
    assert gateway._rate_ok(second)
    assert not gateway._rate_ok(first)

    forged = gateway._client_key(request("unverified-token"))
    assert forged == "peer:127.0.0.1"
    assert gateway.auth.revoke_device(identities[0][0])
    assert gateway._client_key(request(identities[0][1])) == forged


class _Transcriber:
    available = True

    def __init__(self):
        self.calls = []

    def transcribe_pcm(self, pcm, rate):
        self.calls.append((len(pcm), rate))
        return "heard"


@pytest.mark.parametrize("rate", [0, 1000, 384000])
def test_invalid_wav_rate_never_reaches_transcriber(tmp_path, rate):
    gateway = _gateway(tmp_path)
    transcriber = _Transcriber()
    gateway._transcriber = transcriber

    text, error = gateway._transcribe_blocking(_wav(rate))

    assert text == ""
    assert error == "unsupported audio sample rate"
    assert transcriber.calls == []


def test_wav_duration_is_bounded_before_transcription(tmp_path):
    gateway = _gateway(tmp_path)
    transcriber = _Transcriber()
    gateway._transcriber = transcriber
    # 241 seconds of 8 kHz mono PCM fits beneath the 8 MB HTTP body ceiling.
    body = _wav(8000, frames=8000 * 241)
    assert len(body) < gateway._MAX_AUDIO_BYTES

    text, error = gateway._transcribe_blocking(body)

    assert text == ""
    assert error == "audio too long"
    assert transcriber.calls == []


def test_revoked_device_cannot_receive_queued_private_sse_event(tmp_path):
    async def scenario():
        gateway = _gateway(tmp_path)
        code = gateway.begin_pairing()
        device_id, refresh = gateway.auth.complete_pairing(code)
        token, _ = gateway.auth.issue_access(device_id, refresh)

        class Stream:
            def __init__(self, headers):
                self.sent = []
                self.started = asyncio.Event()

            async def prepare(self, _request):
                pass

            async def write(self, data):
                self.sent.append(data)
                self.started.set()

        stream = None

        def make_stream(*, headers):
            nonlocal stream
            stream = Stream(headers)
            return stream

        gateway._web = SimpleNamespace(StreamResponse=make_stream)
        request = SimpleNamespace(query={"token": token})
        task = asyncio.create_task(gateway._handle_events(request))
        try:
            while stream is None:
                await asyncio.sleep(0)
            await asyncio.wait_for(stream.started.wait(), timeout=1)
            assert gateway.auth.revoke_device(device_id)
            gateway._push_device_event(device_id, {
                "type": "confirm", "token": "private-approval-token",
            })
            await asyncio.wait_for(task, timeout=1)
            assert all(b"private-approval-token" not in item for item in stream.sent)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_idle_mobile_websocket_closes_after_device_revocation(tmp_path, monkeypatch):
    async def scenario():
        from aiohttp import WSMsgType
        from aiohttp.test_utils import TestClient, TestServer

        gateway = _gateway(tmp_path)
        code = gateway.begin_pairing()
        device_id, refresh = gateway.auth.complete_pairing(code)
        token, _ = gateway.auth.issue_access(device_id, refresh)
        monkeypatch.setattr(mobile_ws, "AUTH_RECHECK_S", 0.05, raising=False)

        client = TestClient(TestServer(gateway._build_app()))
        await client.start_server()
        try:
            ws = await client.ws_connect("/v1/device/uplink", headers={
                "Authorization": f"Bearer {token}",
            })
            assert gateway.auth.revoke_device(device_id)
            message = await asyncio.wait_for(ws.receive(), timeout=0.5)
            assert message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED)
            assert ws.close_code == 1008
        finally:
            await client.close()

    asyncio.run(scenario())
