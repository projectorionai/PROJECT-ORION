"""Paired speech selects a documented voice without weakening authentication."""

import asyncio
import struct
from types import SimpleNamespace

import pytest

from orion_core import remote_speech
from orion_core.remote import RemoteGateway
from tests.test_remote_hardware_audio import _Bus, _Memory, _client, _token


def _router():
    return SimpleNamespace(settings=SimpleNamespace(providers={"gemini":
        SimpleNamespace(enabled=True, api_key="test-key")}))


@pytest.mark.parametrize("model", ["gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts"])
def test_all_documented_voices_use_the_selected_provider_voice(model, monkeypatch):
    monkeypatch.setenv("ORION_REMOTE_TTS_MODEL", model)
    assert len(remote_speech.GEMINI_VOICES) == 30
    for voice in remote_speech.GEMINI_VOICES:
        _key, _model, payload = remote_speech._charon_request(_router(), "First.\n\nSecond.", voice)
        config = payload["generationConfig"]["speechConfig"]["voiceConfig"]
        actual = config.get("voice") or config["prebuiltVoiceConfig"]["voiceName"]
        assert actual == voice
        spoken = payload["contents"][0]["parts"][0]["text"]
        assert "First. Second." in spoken
    with pytest.raises(ValueError):
        remote_speech._charon_request(_router(), "Hello", "Charon\ninjected")


def test_paired_client_selects_voice_and_invalid_names_never_reach_provider(tmp_path, monkeypatch):
    calls = []

    async def speech(_router, text, voice=None):
        calls.append((text, voice))
        yield b"\x01\x02" * 100

    monkeypatch.setattr(remote_speech, "stream_charon", speech)

    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        client = await _client(gateway)
        try:
            assert (await client.post("/api/speech/stream", json={"text": "Hello", "voice": "Orus"})).status == 401
            token = await _token(client, gateway)
            headers = {"Authorization": "Bearer " + token}
            for bad in ["Phone", "invalid", {"voice": "Orus"}, "Orus\n"]:
                assert (await client.post("/api/speech/stream", headers=headers,
                    json={"text": "Hello", "voice": bad})).status == 400
            assert not calls
            response = await client.post("/api/speech/stream", headers=headers,
                json={"text": "Hello", "voice": "Orus"})
            assert response.status == 200
            assert response.headers["X-Orion-Voice"] == "Orus"
            assert await response.read() == struct.pack("!I", 200) + b"\x01\x02" * 100 + bytes(4)
            assert calls == [("Hello", "Orus")]
            assert not gateway._speech_devices
        finally:
            await client.close()

    asyncio.run(flow())


def test_complete_stream_without_pcm_is_unavailable(tmp_path, monkeypatch):
    async def speech(_router, _text):
        if False:
            yield b""

    monkeypatch.setattr(remote_speech, "stream_charon", speech)

    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        client = await _client(gateway)
        try:
            token = await _token(client, gateway)
            response = await client.post("/api/speech/stream", headers={"Authorization": "Bearer " + token},
                json={"text": "Hello"})
            assert response.status == 503
            assert not gateway._speech_devices
        finally:
            await client.close()

    asyncio.run(flow())
