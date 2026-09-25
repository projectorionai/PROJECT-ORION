"""
HTTP test for the phone voice-input path (/api/transcribe).

The Android WebView can record audio but has no in-browser SpeechRecognition, so
it uploads a 16 kHz mono PCM WAV and the desktop transcribes it offline.  These
tests exercise the endpoint end to end with a stub transcriber (so no real
Whisper model is loaded): auth is enforced, a WAV is decoded and transcribed,
stereo is down-mixed to mono, and bad input is rejected cleanly.
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.remote import RemoteGateway


class _Signal:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Memory:
    def log_episode(self, *a):
        pass

    def prompt_context(self, limit=18):
        return ""


class _FakeTranscriber:
    """Stand-in for OfflineTranscriber — records what it was handed."""

    def __init__(self, text="hello orion", available=True):
        self._text = text
        self._available = available
        self.last_pcm = None
        self.last_rate = None

    @property
    def available(self):
        return self._available

    def transcribe_pcm(self, pcm, sample_rate=16000):
        self.last_pcm = pcm
        self.last_rate = sample_rate
        return self._text


def _gateway(tmp_path, transcriber=_FakeTranscriber()):
    gw = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
    gw._transcriber = transcriber          # skip the real (slow) model build
    return gw


def _wav(nsamples=16000, channels=1, rate=16000, sampwidth=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(b"\x11\x22" * nsamples * channels)
    return buf.getvalue()


async def _client(gateway):
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(gateway._build_app()))
    await client.start_server()
    return client


async def _pair_and_token(client, gateway):
    code = gateway.begin_pairing()
    creds = await (await client.post("/v1/auth/pair", json={"code": code})).json()
    tok = (await (await client.post("/v1/auth/token", json={
        "device_id": creds["device_id"],
        "refresh_token": creds["refresh_token"]})).json())["access_token"]
    return tok


def test_transcribe_returns_text_for_paired_device(tmp_path):
    async def flow():
        stub = _FakeTranscriber(text="lights on please")
        gateway = _gateway(tmp_path, stub)
        client = await _client(gateway)
        try:
            tok = await _pair_and_token(client, gateway)
            r = await client.post(
                "/api/transcribe", data=_wav(),
                headers={"Authorization": "Bearer " + tok,
                         "Content-Type": "audio/wav"})
            assert r.status == 200
            body = await r.json()
            assert body["ok"] is True
            assert body["text"] == "lights on please"
            # A clean mono 16-bit clip reaches the transcriber unchanged.
            assert stub.last_pcm == b"\x11\x22" * 16000
            assert stub.last_rate == 16000
        finally:
            await client.close()

    import asyncio
    asyncio.run(flow())


def test_transcribe_requires_authentication(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        client = await _client(gateway)
        try:
            r = await client.post("/api/transcribe", data=_wav(),
                                  headers={"Content-Type": "audio/wav"})
            assert r.status == 401
        finally:
            await client.close()

    import asyncio
    asyncio.run(flow())


def test_transcribe_downmixes_stereo(tmp_path):
    async def flow():
        stub = _FakeTranscriber(text="ok")
        gateway = _gateway(tmp_path, stub)
        client = await _client(gateway)
        try:
            tok = await _pair_and_token(client, gateway)
            r = await client.post(
                "/api/transcribe", data=_wav(nsamples=8000, channels=2),
                headers={"Authorization": "Bearer " + tok,
                         "Content-Type": "audio/wav"})
            assert r.status == 200
            assert (await r.json())["ok"] is True
            # Two interleaved channels collapse to one → half the sample bytes.
            assert len(stub.last_pcm) == 8000 * 2
        finally:
            await client.close()

    import asyncio
    asyncio.run(flow())


def test_transcribe_rejects_non_wav(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        client = await _client(gateway)
        try:
            tok = await _pair_and_token(client, gateway)
            r = await client.post(
                "/api/transcribe", data=b"this is not audio",
                headers={"Authorization": "Bearer " + tok,
                         "Content-Type": "audio/wav"})
            assert r.status == 400
            assert (await r.json())["ok"] is False
        finally:
            await client.close()

    import asyncio
    asyncio.run(flow())


def test_transcribe_reports_missing_engine(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path, _FakeTranscriber(available=False))
        client = await _client(gateway)
        try:
            tok = await _pair_and_token(client, gateway)
            r = await client.post(
                "/api/transcribe", data=_wav(),
                headers={"Authorization": "Bearer " + tok,
                         "Content-Type": "audio/wav"})
            assert r.status == 503
            assert "engine" in (await r.json())["error"]
        finally:
            await client.close()

    import asyncio
    asyncio.run(flow())


def test_transcribe_rejects_oversized_upload(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        client = await _client(gateway)
        try:
            tok = await _pair_and_token(client, gateway)
            big = b"\x00" * (RemoteGateway._MAX_AUDIO_BYTES + 1)
            r = await client.post(
                "/api/transcribe", data=big,
                headers={"Authorization": "Bearer " + tok,
                         "Content-Type": "audio/wav"})
            assert r.status == 413
        finally:
            await client.close()

    import asyncio
    asyncio.run(flow())
