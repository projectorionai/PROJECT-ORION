"""
ElevenLabs streaming — audio must start on the first chunk, and it must go
through ORION's one verified output stream.

The old path fetched the whole utterance, then called sd.play() on a private
stream.  That produced a silence proportional to the length of the reply, a
device open/close between every sentence, and speech that could come out of a
different speaker than the one ORION had just verified.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from orion_core import voice_elevenlabs as el


class _Sig:
    def __init__(self):
        self.messages = []

    def emit(self, *a):
        self.messages.append(a[0] if len(a) == 1 else a)

    def connect(self, *a):
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


class _FakeResponse(io.BytesIO):
    """Behaves like urlopen's response: status + chunked read + context mgr."""

    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(el, "resolve_api_key", lambda: "key-123")
    monkeypatch.setattr(el, "resolve_voice_id", lambda: "voice-abc")
    monkeypatch.setattr(el, "CACHE_DIR", tmp_path / "tts_cache")
    # Neutralise any character preset in the real config so the delivery
    # settings under test are purely the emotion mapping (this test pins the
    # 'calm' stability); a preset would legitimately override it.
    monkeypatch.setattr(el, "active_preset", lambda: "")
    return el.ElevenLabsVoice()


def _pcm(n_bytes: int) -> bytes:
    return bytes(range(256)) * (n_bytes // 256)


# ── streaming delivers progressively ─────────────────────────────────────────

def test_chunks_are_delivered_as_they_arrive_not_at_the_end(configured, monkeypatch):
    payload = _pcm(4096 * 5)
    captured = []

    def fake_urlopen(req, timeout=None):
        assert req.full_url.endswith("/stream?output_format=pcm_24000"), req.full_url
        return _FakeResponse(payload)

    monkeypatch.setattr(el.urllib.request, "urlopen", fake_urlopen)

    result = configured.stream_pcm("hello there", on_chunk=captured.append)

    assert result == payload
    assert len(captured) == 5, "audio must arrive in chunks, not one blob"
    assert b"".join(captured) == payload


def test_the_streaming_endpoint_is_used(configured, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        seen["headers"] = {k.lower(): v for k, v in req.headers.items()}
        return _FakeResponse(_pcm(4096))

    monkeypatch.setattr(el.urllib.request, "urlopen", fake_urlopen)
    configured.stream_pcm("hi", emotion="calm")

    assert "/stream" in seen["url"]
    assert seen["body"]["model_id"] == el.DEFAULT_MODEL_ID
    assert seen["body"]["voice_settings"]["stability"] == 0.70   # calm
    assert seen["headers"]["xi-api-key"] == "key-123"


def test_an_interruption_stops_the_download_mid_stream(configured, monkeypatch):
    payload = _pcm(4096 * 10)
    monkeypatch.setattr(el.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResponse(payload))
    captured = []
    # Stop after the third chunk.
    result = configured.stream_pcm(
        "a long sentence", on_chunk=captured.append,
        should_stop=lambda: len(captured) >= 3)

    assert len(captured) == 3
    assert result is None, "a truncated utterance must not be reported as complete"


def test_a_truncated_utterance_is_never_cached(configured, monkeypatch):
    payload = _pcm(4096 * 6)
    monkeypatch.setattr(el.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResponse(payload))
    captured = []
    configured.stream_pcm("cut me off", on_chunk=captured.append,
                          should_stop=lambda: len(captured) >= 2)
    assert not list(el.CACHE_DIR.glob("*.pcm")) if el.CACHE_DIR.exists() else True


# ── caching behaves identically to a fresh stream ────────────────────────────

def test_a_cache_hit_is_still_delivered_in_chunks(configured, monkeypatch):
    payload = _pcm(4096 * 4)
    monkeypatch.setattr(el.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResponse(payload))
    configured.stream_pcm("cache me")          # populate

    def explode(*a, **k):
        raise AssertionError("a cache hit must not touch the network")

    monkeypatch.setattr(el.urllib.request, "urlopen", explode)
    captured = []
    result = configured.stream_pcm("cache me", on_chunk=captured.append)

    assert result == payload
    assert len(captured) == 4


# ── failure never raises, and never loses the voice ──────────────────────────

def test_a_network_failure_before_any_audio_returns_none(configured, monkeypatch):
    def boom(*a, **k):
        raise OSError("connection reset")

    monkeypatch.setattr(el.urllib.request, "urlopen", boom)
    assert configured.stream_pcm("hello", on_chunk=lambda c: None) is None


def test_a_failure_after_some_audio_keeps_what_played(configured, monkeypatch):
    class HalfBroken(_FakeResponse):
        def __init__(self):
            super().__init__(_pcm(4096 * 3))
            self.reads = 0

        def read(self, n=-1):
            self.reads += 1
            if self.reads > 2:
                raise OSError("stream died")
            return super().read(n)

    monkeypatch.setattr(el.urllib.request, "urlopen",
                        lambda req, timeout=None: HalfBroken())
    captured = []
    result = configured.stream_pcm("partial", on_chunk=captured.append)
    assert len(captured) == 2
    assert result == b"".join(captured), (
        "audio that already played must be reported so the sentence is not "
        "spoken again in a second voice")


def test_an_unconfigured_voice_returns_none_without_calling_out(monkeypatch):
    monkeypatch.setattr(el, "resolve_api_key", lambda: "")
    monkeypatch.setattr(el, "resolve_voice_id", lambda: "")
    monkeypatch.setattr(el.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("must not call the API"))
    assert el.ElevenLabsVoice().stream_pcm("hi") is None


# ── it feeds the shared renderer, not a private stream ───────────────────────

def _synth():
    from orion_core.audio import SpeechSynthesiser
    return SpeechSynthesiser(_StubBus())


def test_elevenlabs_audio_goes_into_the_shared_renderer_queue(monkeypatch):
    from orion_core.audio import SpeechSynthesiser

    synth = _synth()
    enqueued = []
    renderer = SimpleNamespace(
        enqueue=enqueued.append,
        is_active=lambda: False,
        queue=SimpleNamespace(empty=lambda: True),
    )
    synth.attach_renderer(renderer)
    monkeypatch.setattr(synth._elevenlabs, "available", lambda: True)

    def fake_stream(text, emotion="neutral", on_chunk=None, should_stop=None):
        for i in range(3):
            on_chunk(b"\x01\x02" * 100)
        return b"\x01\x02" * 300

    monkeypatch.setattr(synth._elevenlabs, "stream_pcm", fake_stream)
    monkeypatch.setattr(SpeechSynthesiser, "_await_renderer", lambda *a, **k: None)

    assert synth._speak_elevenlabs("hello") is True
    # The silence compressor works in whole 10 ms frames, so it may re-block
    # the stream — what matters is that every byte reaches the renderer, in
    # order, and that none of this loud audio is trimmed.
    assert enqueued, "audio must reach the renderer"
    assert b"".join(enqueued) == b"\x01\x02" * 300


def test_sd_play_is_not_used_when_a_renderer_is_attached(monkeypatch):
    from orion_core import audio as audio_mod
    from orion_core.audio import SpeechSynthesiser

    synth = _synth()
    synth.attach_renderer(SimpleNamespace(
        enqueue=lambda c: None,
        is_active=lambda: False,
        queue=SimpleNamespace(empty=lambda: True)))
    monkeypatch.setattr(synth._elevenlabs, "available", lambda: True)
    monkeypatch.setattr(synth._elevenlabs, "stream_pcm",
                        lambda text, emotion="neutral", on_chunk=None, should_stop=None:
                        (on_chunk(b"\x00" * 512), b"\x00" * 512)[1])
    monkeypatch.setattr(SpeechSynthesiser, "_await_renderer", lambda *a, **k: None)
    monkeypatch.setattr(audio_mod.sd, "play",
                        lambda *a, **k: pytest.fail(
                            "ElevenLabs must not open its own output stream"))

    synth._speak_elevenlabs("hello")


def test_total_elevenlabs_failure_falls_back_to_the_local_engine(monkeypatch):
    synth = _synth()
    synth.attach_renderer(SimpleNamespace(
        enqueue=lambda c: None,
        is_active=lambda: False,
        queue=SimpleNamespace(empty=lambda: True)))
    monkeypatch.setattr(synth._elevenlabs, "available", lambda: True)
    monkeypatch.setattr(
        synth._elevenlabs, "stream_pcm",
        lambda text, emotion="neutral", on_chunk=None, should_stop=None: None)

    assert synth._speak_elevenlabs("hello") is False, (
        "no audio at all must fall through to pyttsx3/PowerShell")


def test_the_queue_manager_wires_the_renderer_automatically():
    from orion_core.audio import (
        AudioPlaybackThread, SpeechQueueManager, SpeechSynthesiser,
    )
    from orion_core.audio_state import AudioStateMachine

    bus = _StubBus()
    machine = AudioStateMachine(bus)
    playback = AudioPlaybackThread(bus, machine)
    tts = SpeechSynthesiser(bus, machine)
    SpeechQueueManager(bus, playback, tts, machine)

    assert tts._renderer is playback, (
        "ORION must have exactly one output stream for his whole voice")
