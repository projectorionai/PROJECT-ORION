"""The goodbye starts at once (SpeechQueueManager.say_goodbye).

Measured 29 Oct 2026 in the owner's session: "Clean shutdown after 18.6 s ...
16.00 s farewell (speaking the goodbye)". The quit came 3 s after a long
security note was queued, so the goodbye waited behind that note - and behind
a Gemini TTS request for it - until app._await_farewell's ceiling. The
goodbye itself went through the everyday voice chain too, which can ask the
network or load Kokoro's model before a word is said.

Now the goodbye drops what was still queued, plays Charon's cached line
straight into the renderer when he has one, otherwise uses a voice that can
start immediately, and nothing queued after it is said.

Headless: no audio device, no network, no model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import voice_charon  # noqa: E402
from orion_core.audio import SpeechQueueManager, SpeechSynthesiser  # noqa: E402
from orion_core.audio_state import AudioStateMachine  # noqa: E402


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _Bus:
    def __getattr__(self, name):
        return _Signal()


class _Playback:
    """The renderer's surface that the queue manager touches."""

    def __init__(self) -> None:
        self.queue: Queue = Queue()
        self.played: list[bytes] = []
        self.cleared = 0
        self._generation = 0

    def enqueue(self, chunk: bytes) -> None:
        self.played.append(bytes(chunk))

    def enqueue_for_generation(self, chunk: bytes, generation: int) -> None:
        self.enqueue(chunk)

    def clear(self) -> None:
        self.cleared += 1

    def is_active(self) -> bool:
        return False

    def has_pending(self) -> bool:
        return False

    def held(self) -> bool:
        return False


class _Kokoro:
    def __init__(self, loaded: bool) -> None:
        self._loaded = loaded
        self.loads = 0

    def available(self) -> bool:
        return True

    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self.loads += 1

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def _charon(tmp_path, monkeypatch, asked: list[str]) -> voice_charon.CharonVoice:
    monkeypatch.setattr(voice_charon, "_ON", True)

    def synthesise(text: str) -> bytes:
        asked.append(text)                          # a Gemini TTS request
        return b"\x01\x02" * 2400

    return voice_charon.CharonVoice(key_source=lambda: "k", online=lambda: True,
                                    synthesise=synthesise, cache=tmp_path)


def _manager(charon=None, kokoro=None):
    bus = _Bus()
    machine = AudioStateMachine(bus)
    tts = SpeechSynthesiser(bus, machine=machine)
    tts._charon = charon
    tts._kokoro = kokoro
    playback = _Playback()
    return SpeechQueueManager(bus, playback, tts, machine), tts, playback


def test_a_cached_goodbye_goes_straight_to_the_renderer(tmp_path, monkeypatch):
    asked: list[str] = []
    charon = _charon(tmp_path, monkeypatch, asked)
    charon._remember("Goodbye, sir.", b"\x05\x06" * 4800)
    manager, tts, playback = _manager(charon)
    tts.speak("A long security note that was still waiting to be said.")
    assert tts.queue_depth() == 1

    assert manager.say_goodbye("Goodbye, sir.") == "charon"

    assert playback.played == [b"\x05\x06" * 4800]   # his voice, now
    assert tts.queue_depth() == 0                    # the stale note is gone
    assert playback.cleared >= 1                     # and nothing old still plays
    assert asked == []                               # no network request


def test_nothing_queued_after_the_goodbye_is_said(tmp_path, monkeypatch):
    asked: list[str] = []
    charon = _charon(tmp_path, monkeypatch, asked)
    charon._remember("Goodbye, sir.", b"\x05\x06" * 4800)
    manager, tts, playback = _manager(charon)
    manager.say_goodbye("Goodbye, sir.")
    manager.speak_text("Security note: something started listening on a port.")
    manager.enqueue_native_audio(b"\x00\x01" * 960)
    assert tts.queue_depth() == 0
    assert playback.played == [b"\x05\x06" * 4800]
    assert not manager.is_busy()


def test_an_uncached_goodbye_never_asks_the_network_or_loads_a_model(tmp_path, monkeypatch):
    asked: list[str] = []
    charon = _charon(tmp_path, monkeypatch, asked)
    kokoro = _Kokoro(loaded=False)
    manager, tts, playback = _manager(charon, kokoro)
    assert manager.say_goodbye("Until later, sir.") == "local"
    assert tts.queue_depth() == 1 and playback.played == []

    spoke: list[str] = []
    monkeypatch.setattr(tts, "_speak_pyttsx3", spoke.append)
    monkeypatch.setattr(tts, "_drain_renderer", lambda: None)
    tts._engine = object()
    tts._speak_one(tts.queue.get_nowait())

    assert spoke == ["Until later, sir."]
    assert asked == [] and kokoro.loads == 0
    assert tts._active_backend == "pyttsx3"


def test_an_already_loaded_kokoro_may_say_it(tmp_path, monkeypatch):
    asked: list[str] = []
    charon = _charon(tmp_path, monkeypatch, asked)
    kokoro = _Kokoro(loaded=True)
    manager, tts, _playback = _manager(charon, kokoro)
    used: list[str] = []
    monkeypatch.setattr(tts, "_speak_cloud",
                        lambda text, voice, label: used.append(label) or True)
    tts.speak_parting("Until later, sir.")
    tts._speak_one(tts.queue.get_nowait())
    assert used == ["Kokoro"] and asked == [] and kokoro.loads == 0


def test_a_muted_voice_says_no_goodbye(tmp_path, monkeypatch):
    manager, tts, playback = _manager()
    manager.muted = True
    assert manager.say_goodbye("Goodbye.") == ""
    assert tts.queue_depth() == 0 and playback.played == []


def test_every_shutdown_path_uses_it():
    root = Path(__file__).resolve().parents[1] / "orion_core"
    app = (root / "app.py").read_text(encoding="utf-8")
    assert '"say_goodbye"' in app
    worker = (root / "live_worker.py").read_text(encoding="utf-8")
    assert worker.count("self._say_goodbye(") >= 2
