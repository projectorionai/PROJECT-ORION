"""Audio reaches the companion before generation ends, with no false success."""
import asyncio
import base64
import json
import struct

import pytest

from orion_core import remote_speech
from orion_core.remote import RemoteGateway
from orion_core.remote_speech import SpeechUnavailable, _speech_events
from tests.test_remote_hardware_audio import _Bus, _Memory, _client, _token


class Content:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_chunked(self, _size):
        for chunk in self.chunks:
            yield chunk


def test_provider_events_survive_split_lines_and_reject_truncated_events():
    async def flow():
        data = b'data: {"candidates": []}\r\n\r\ndata: [DONE]\n\n'
        assert [event async for event in _speech_events(Content(
            [data[index:index + 3] for index in range(0, len(data), 3)]))] == [{"candidates": []}]
        with pytest.raises(SpeechUnavailable, match="interrupted"):
            [event async for event in _speech_events(Content([data[:20]]))]
        monkey_max = remote_speech.MAX_STREAM_EVENT_BYTES
        try:
            remote_speech.MAX_STREAM_EVENT_BYTES = 20
            with pytest.raises(SpeechUnavailable, match="too large"):
                [event async for event in _speech_events(Content([b'data: ' + b'a' * 30]))]
        finally:
            remote_speech.MAX_STREAM_EVENT_BYTES = monkey_max
    asyncio.run(flow())


def test_provider_requires_successful_completion_and_fixed_pcm_rate(monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    pcm = b"\x01\x02" * 100
    events = []

    class Response:
        status = 200
        @property
        def content(self):
            return Content([("data: " + json.dumps(event) + "\n\n").encode() for event in events])
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass

    class Client:
        def post(self, url, **kwargs):
            assert url.endswith(":streamGenerateContent")
            assert kwargs["params"] == {"alt": "sse"}
            assert kwargs["allow_redirects"] is False
            request = kwargs["json"]
            assert request["contents"][0]["parts"][0]["text"] == "Hello"
            assert "British" in request["contents"][0]["parts"][0]["speech_metadata"]["style"]
            assert request["generationConfig"]["responseFormat"]["audio"]["sampleRate"] == 24000
            return Response()

    @asynccontextmanager
    async def client(_router):
        yield Client()

    monkeypatch.setattr(remote_speech, "_speech_client", client)
    router = SimpleNamespace(settings=SimpleNamespace(providers={"gemini":
        SimpleNamespace(enabled=True, api_key="test-key")}))

    def packet(rate=24000, finish=None):
        candidate = {"content": {"parts": [{"inlineData": {
            "data": base64.b64encode(pcm).decode(), "mimeType": f"audio/L16;rate={rate}"}}]}}
        if finish: candidate["finishReason"] = finish
        return {"candidates": [candidate]}

    async def flow():
        events[:] = [packet(finish="STOP")]
        assert b"".join([part async for part in remote_speech.stream_charon(router, "Hello")]) == pcm
        events[:] = [packet()]
        with pytest.raises(SpeechUnavailable, match="interrupted"):
            [part async for part in remote_speech.stream_charon(router, "Hello")]
        events[:] = [packet(finish="OTHER")]
        with pytest.raises(SpeechUnavailable, match="finish"):
            [part async for part in remote_speech.stream_charon(router, "Hello")]
        events[:] = [packet(rate=16000, finish="STOP")]
        with pytest.raises(SpeechUnavailable, match="rate"):
            [part async for part in remote_speech.stream_charon(router, "Hello")]
    asyncio.run(flow())


def test_legacy_charon_request_does_not_follow_a_provider_redirect(monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    class Response:
        status = 302

        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass

    class Client:
        def post(self, _url, **kwargs):
            assert kwargs["allow_redirects"] is False
            return Response()

    @asynccontextmanager
    async def client(_router):
        yield Client()

    monkeypatch.setattr(remote_speech, "_speech_client", client)
    router = SimpleNamespace(settings=SimpleNamespace(providers={"gemini":
        SimpleNamespace(enabled=True, api_key="test-key")}))
    with pytest.raises(SpeechUnavailable, match="HTTP 302"):
        asyncio.run(remote_speech.synthesise_charon(router, "Hello"))


def test_authenticated_phone_receives_audio_before_the_provider_finishes(tmp_path, monkeypatch):
    async def flow():
        finish = asyncio.Event()
        closed = asyncio.Event()
        pcm = b"\x01\x02" * 2400

        async def speech(_router, text):
            assert text == "Hello"
            try:
                yield pcm
                await finish.wait()
                yield pcm
            finally:
                closed.set()

        monkeypatch.setattr(remote_speech, "stream_charon", speech)
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        client = await _client(gateway)
        try:
            assert (await client.post("/api/speech/stream", json={"text": "Hello"})).status == 401
            token = await _token(client, gateway)
            headers = {"Authorization": "Bearer " + token}
            response = await asyncio.wait_for(client.post("/api/speech/stream", headers=headers,
                json={"text": "Hello"}), timeout=2)
            assert response.headers["X-Orion-PCM-Format"] == "s16le-mono-v1"
            assert response.headers["Cache-Control"] == "no-store, no-transform"
            assert struct.unpack("!I", await response.content.readexactly(4))[0] == len(pcm)
            assert await response.content.readexactly(len(pcm)) == pcm
            assert not finish.is_set()
            # A second request from this device must not double-generate speech.
            assert (await client.post("/api/speech/stream", headers=headers,
                json={"text": "Hello"})).status == 429
            finish.set()
            assert await response.read() == struct.pack("!I", len(pcm)) + pcm + bytes(4)
            assert not gateway._speech_devices
            assert closed.is_set()
        finally:
            finish.set()
            await client.close()
    asyncio.run(flow())


def test_disconnect_cancels_generation_and_releases_busy_device(tmp_path, monkeypatch):
    async def flow():
        closed = asyncio.Event()
        async def speech(_router, _text):
            try:
                yield b"\x00\x00" * 100
                await asyncio.Event().wait()
            finally:
                closed.set()
        monkeypatch.setattr(remote_speech, "stream_charon", speech)
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        client = await _client(gateway)
        try:
            token = await _token(client, gateway)
            response = await client.post("/api/speech/stream", headers={"Authorization": "Bearer " + token},
                json={"text": "Hello"})
            response.close()
            await asyncio.wait_for(closed.wait(), timeout=2)
            # Generator finalisation signals before its caller finishes cleanup.
            deadline = asyncio.get_running_loop().time() + 1
            while gateway._speech_devices and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert not gateway._speech_devices
        finally:
            await client.close()
    asyncio.run(flow())
