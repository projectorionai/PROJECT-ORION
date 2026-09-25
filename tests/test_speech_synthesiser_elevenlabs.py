"""
Tests for SpeechSynthesiser's ElevenLabs wiring (Mark XX design-spec,
"ElevenLabs Integration"): ElevenLabs is tried first for each utterance
when configured, falling back to the existing pyttsx3/PowerShell chain for
just that one utterance if it produces no audio — never a hard dependency,
never silent. SpeechSynthesiser had zero direct test coverage before this
(only an import-existence smoke check elsewhere), so this also covers the
pre-existing interrupt()/hold() contract this change had to preserve.

Methods are called directly rather than running the real background
Thread/queue loop, and sounddevice.play/stop are monkeypatched so no real
audio device is touched.

Headless — no display, no audio hardware, no network required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import audio as audio_module
from orion_core.audio import SpeechSynthesiser


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubElevenLabs:
    SAMPLE_RATE = 24000

    def __init__(self, available: bool = False, pcm: bytes | None = None) -> None:
        self._available = available
        self._pcm = pcm
        self.calls: list[tuple[str, str]] = []

    def available(self) -> bool:
        return self._available

    def synthesize_pcm(self, text: str, emotion: str = "neutral") -> bytes | None:
        self.calls.append((text, emotion))
        return self._pcm


def _synth() -> SpeechSynthesiser:
    return SpeechSynthesiser(_StubBus())


# ── _speak_one: dispatch + fallback ─────────────────────────────────────────

def test_speak_one_uses_elevenlabs_when_available(monkeypatch):
    synth = _synth()
    synth._elevenlabs = _StubElevenLabs(available=True, pcm=b"\x00\x01" * 100)
    calls = {"pyttsx3": 0, "powershell": 0}
    monkeypatch.setattr(synth, "_speak_pyttsx3", lambda t: calls.__setitem__("pyttsx3", calls["pyttsx3"] + 1))
    monkeypatch.setattr(synth, "_speak_powershell", lambda t: calls.__setitem__("powershell", calls["powershell"] + 1))
    monkeypatch.setattr(audio_module.sd, "play", lambda *a, **k: None)

    synth._speak_one("hello")

    assert synth._active_backend == "elevenlabs"
    assert calls == {"pyttsx3": 0, "powershell": 0}


def test_speak_one_falls_back_to_pyttsx3_when_elevenlabs_produces_no_audio(monkeypatch):
    synth = _synth()
    synth._elevenlabs = _StubElevenLabs(available=True, pcm=None)
    synth._engine = object()   # pyttsx3 engine present
    calls = []
    monkeypatch.setattr(synth, "_speak_pyttsx3", lambda t: calls.append(t))

    synth._speak_one("hello")

    assert calls == ["hello"]
    assert synth._active_backend == "pyttsx3"


def test_speak_one_falls_back_to_powershell_when_no_pyttsx3_engine(monkeypatch):
    synth = _synth()
    synth._elevenlabs = _StubElevenLabs(available=True, pcm=None)
    synth._engine = None
    calls = []
    monkeypatch.setattr(synth, "_speak_powershell", lambda t: calls.append(t))

    synth._speak_one("hello")

    assert calls == ["hello"]
    assert synth._active_backend == "powershell"


def test_speak_one_skips_elevenlabs_entirely_when_unavailable(monkeypatch):
    synth = _synth()
    synth._elevenlabs = _StubElevenLabs(available=False)
    synth._engine = object()
    calls = []
    monkeypatch.setattr(synth, "_speak_pyttsx3", lambda t: calls.append(t))

    synth._speak_one("hello")

    assert calls == ["hello"]
    assert synth._elevenlabs.calls == []   # never even asked to synthesise
    assert synth._active_backend == "pyttsx3"


# ── _speak_elevenlabs ────────────────────────────────────────────────────────

def test_speak_elevenlabs_returns_false_with_no_pcm(monkeypatch):
    synth = _synth()
    synth._elevenlabs = _StubElevenLabs(available=True, pcm=None)
    play_calls = []
    monkeypatch.setattr(audio_module.sd, "play", lambda *a, **k: play_calls.append((a, k)))

    assert synth._speak_elevenlabs("hello") is False
    assert play_calls == []


def test_speak_elevenlabs_plays_pcm_with_the_right_samplerate(monkeypatch):
    synth = _synth()
    synth._elevenlabs = _StubElevenLabs(available=True, pcm=b"\x00\x01" * 500)
    play_calls = []
    monkeypatch.setattr(audio_module.sd, "play", lambda data, samplerate=None, **k: play_calls.append((samplerate, k)))

    result = synth._speak_elevenlabs("hello there")

    assert result is True
    assert len(play_calls) == 1
    samplerate, kwargs = play_calls[0]
    assert samplerate == 24000
    assert kwargs.get("blocking") is True


def test_speak_elevenlabs_respects_a_pending_interruption(monkeypatch):
    synth = _synth()
    synth._elevenlabs = _StubElevenLabs(available=True, pcm=b"\x00\x01" * 100)
    synth._interrupted.set()

    result = synth._speak_elevenlabs("hello")

    assert result is True
    assert synth._elevenlabs.calls == []   # never even asked to synthesise


def test_speak_elevenlabs_recovers_from_a_playback_exception(monkeypatch):
    synth = _synth()
    synth._elevenlabs = _StubElevenLabs(available=True, pcm=b"\x00\x01" * 100)

    def _boom(*a, **k):
        raise RuntimeError("no audio device")

    monkeypatch.setattr(audio_module.sd, "play", _boom)

    result = synth._speak_elevenlabs("hello")   # must not raise
    assert result is True   # still counted as "settled" — the utterance was attempted


# ── interrupt() / hold(): sounddevice only stopped for the elevenlabs backend ─

def test_interrupt_stops_sounddevice_when_elevenlabs_was_active(monkeypatch):
    synth = _synth()
    synth._active_backend = "elevenlabs"
    stop_calls = []
    monkeypatch.setattr(audio_module.sd, "stop", lambda: stop_calls.append(1))

    synth.interrupt()

    assert stop_calls == [1]


def test_interrupt_does_not_touch_sounddevice_for_pyttsx3(monkeypatch):
    synth = _synth()
    synth._active_backend = "pyttsx3"
    stop_calls = []
    monkeypatch.setattr(audio_module.sd, "stop", lambda: stop_calls.append(1))

    synth.interrupt()

    assert stop_calls == []


def test_hold_replays_the_full_utterance_for_elevenlabs(monkeypatch):
    synth = _synth()
    synth._speaking.set()
    synth._active_backend = "elevenlabs"
    synth._current_text = "the full sentence that was interrupted"
    synth._word_location = 10   # would only matter for the word-boundary engines
    monkeypatch.setattr(audio_module.sd, "stop", lambda: None)

    was_speaking = synth.hold()

    assert was_speaking is True
    assert synth._resume_text == "the full sentence that was interrupted"


def test_hold_does_not_touch_sounddevice_for_pyttsx3(monkeypatch):
    synth = _synth()
    synth._speaking.set()
    synth._active_backend = "pyttsx3"
    synth._current_text = "hello world"
    stop_calls = []
    monkeypatch.setattr(audio_module.sd, "stop", lambda: stop_calls.append(1))

    synth.hold()

    assert stop_calls == []
