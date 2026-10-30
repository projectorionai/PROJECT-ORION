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

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

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


# ── Charon: ORION's own voice when Live is not there (voice_charon) ──────────

def test_charon_speaks_first_and_is_remembered_for_offline(tmp_path, monkeypatch):
    """10 Oct: "When ORION goes offline I want ORION to still have a voice ...
    Charon's voice - I don't want to be stuck with system default voice"."""
    import io
    import wave
    from orion_core import voice_charon
    monkeypatch.setattr(voice_charon, "_ON", True)
    asked: list[str] = []
    online = {"up": True}

    def synth(text):
        asked.append(text)
        return b"\x01\x02" * 12000                    # half a second of PCM

    voice = voice_charon.CharonVoice(key_source=lambda: "k", online=lambda: online["up"],
                                     synthesise=synth, cache=tmp_path)
    chunks: list[bytes] = []
    assert voice.stream_pcm("Good evening, sir.", on_chunk=chunks.append) is not None
    assert b"".join(chunks) == b"\x01\x02" * 12000 and asked == ["Good evening, sir."]
    online["up"] = False                              # no internet now
    assert voice.stream_pcm("good evening,  SIR.") is not None, "cached line not spoken offline"
    assert voice.stream_pcm("Something never said before.") is None   # falls back
    assert asked == ["Good evening, sir."]

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(48000)
        wav.writeframes(b"\x00\x10" * 4800)
    assert len(voice_charon.to_pcm(buffer.getvalue(), "audio/wav")) == 2400 * 2   # 48 -> 24 kHz


def test_charon_failure_stands_down_and_the_local_voice_speaks(monkeypatch):
    from orion_core import voice_charon
    monkeypatch.setattr(voice_charon, "_ON", True)

    def broken(text):
        raise OSError("network unreachable")

    voice = voice_charon.CharonVoice(key_source=lambda: "k", synthesise=broken,
                                     cache=Path("Z:/nonexistent-orion-cache"))
    assert voice.stream_pcm("Hello there.") is None
    assert voice._cooldown_until > 0                  # no 8 s wait on every line

    synth = _synth()
    synth._renderer = object()
    synth._charon = voice
    synth._elevenlabs = _StubElevenLabs(available=False)
    spoke = []
    monkeypatch.setattr(synth, "_speak_pyttsx3", lambda t: spoke.append(t))
    synth._engine = object()
    monkeypatch.setattr(synth, "_drain_renderer", lambda: None)
    synth._speak_one("Hello there.")
    assert spoke == ["Hello there."] and synth._active_backend == "pyttsx3"


def test_charon_is_off_unless_orion_switched_it_on(monkeypatch):
    from orion_core import voice_charon
    monkeypatch.setattr(voice_charon, "_ON", False)
    assert not voice_charon.CharonVoice(key_source=lambda: "k").available()


def test_offline_charon_segments_do_not_split_decimal_numbers(tmp_path, monkeypatch):
    from orion_core import voice_charon

    monkeypatch.setattr(voice_charon, "_ON", True)
    text = "The value is 3.5. Good evening."
    pieces = [text[start:end] for start, end in voice_charon.sentence_spans(text)]
    assert pieces == ["The value is 3.5.", "Good evening."]
    voice = voice_charon.CharonVoice(key_source=lambda: "", cache=tmp_path)
    voice._remember("Good evening.", b"\x01\x00" * 100)
    segments = voice.offline_plan(text)
    assert [s.text for s in segments] == pieces
    assert segments[0].pcm is None and segments[1].pcm


def test_prewarm_stops_before_provider_when_daily_budget_cannot_persist(tmp_path, monkeypatch):
    from orion_core import voice_charon

    calls = []
    voice = voice_charon.CharonVoice(key_source=lambda: "k", cache=tmp_path,
                                     synthesise=lambda text: calls.append(text) or b"\x01\x00")
    warm = voice_charon.CharonPrewarm(voice, lines=lambda: ["Good evening."],
                                     online=lambda: True, idle_s=0,
                                     state_path=tmp_path / "budget.json")
    monkeypatch.setattr(warm, "_save_state", lambda state: False)
    assert warm.fill() == "state-unavailable"
    assert calls == []


def test_prewarm_malformed_counter_cannot_reset_daily_budget(tmp_path):
    import json
    from orion_core import voice_charon

    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"day": "today", "requests": "broken", "skip": None}))
    calls = []
    voice = voice_charon.CharonVoice(key_source=lambda: "k", cache=tmp_path,
                                     synthesise=lambda text: calls.append(text) or b"\x01\x00")
    warm = voice_charon.CharonPrewarm(voice, lines=lambda: ["Good evening."],
                                     online=lambda: True, idle_s=0, today=lambda: "today",
                                     state_path=path)
    assert warm.fill() == "daily-cap"
    assert calls == []


def test_prewarm_cache_survives_and_daily_cap_survives_restart(tmp_path):
    from orion_core import voice_charon

    calls = []
    voice = voice_charon.CharonVoice(key_source=lambda: "k", cache=tmp_path,
                                     synthesise=lambda text: calls.append(text) or b"\x01\x00")
    kwargs = dict(lines=lambda: ["Good evening.", "Welcome back."], online=lambda: True,
                  interval_s=0, idle_s=0, daily_cap=1, today=lambda: "today")
    assert voice_charon.CharonPrewarm(voice, **kwargs).fill() == "daily-cap"
    assert voice_charon.CharonPrewarm(voice, **kwargs).fill() == "daily-cap"
    assert calls == ["Good evening."] and voice.is_cached("Good evening.")


def test_piper_pending_cancel_kills_owned_process_and_bounds_reap(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from orion_core import voice_piper

    model = tmp_path / "voice.onnx"
    (tmp_path / "piper.exe").write_bytes(b"not executed")
    calls = []
    process = SimpleNamespace(stdin=None, stdout=None, stderr=None, returncode=None)
    process.poll = lambda: None
    process.kill = lambda: calls.append("kill")

    def communicate(*args, **kwargs):
        calls.append(("communicate", kwargs.get("timeout")))
        return b"", b""

    process.communicate = communicate
    monkeypatch.setattr(voice_piper.subprocess, "Popen", lambda *a, **k: process)
    voice = voice_piper.PiperLocalVoice(directory=tmp_path)
    assert voice._run_exe(model, "Hello", should_stop=lambda: True) is None
    assert calls == ["kill", ("communicate", voice_piper.EXE_REAP_S)]
    assert voice._process is None


def test_piper_warmup_has_only_one_loader(tmp_path, monkeypatch):
    import threading
    from orion_core import voice_piper

    model = tmp_path / "voice.onnx"
    loaded = threading.Event()
    release = threading.Event()
    calls = []

    def loader(_model):
        calls.append(1)
        loaded.set()
        release.wait(2)
        return object()

    voice = voice_piper.PiperLocalVoice(directory=tmp_path, engine_loader=loader)
    monkeypatch.setattr(voice, "available", lambda: True)
    monkeypatch.setattr(voice, "_discover", lambda: (model, "package"))
    try:
        assert voice.warm_async()
        assert loaded.wait(1)
        assert not voice.warm_async()
    finally:
        release.set()
        thread = getattr(voice, "_warm_thread", None)
        if thread is not None:
            thread.join(2)
    assert calls == [1]


def test_synth_interrupt_cancels_piper_without_local_fallback(monkeypatch):
    from types import SimpleNamespace

    synth = _synth()
    calls = []
    synth._piper = SimpleNamespace(stop=lambda: calls.append("stop"))
    synth._active_backend = "piper"
    monkeypatch.setattr(audio_module.sd, "stop", lambda: None)
    synth.interrupt()
    assert calls == ["stop"] and synth._interrupted.is_set()


# ── Charon: the 16 Oct refusal, stand-downs, the warm-up, one shared voice ───

_DEADLINE = "Manually set deadline 8s is too short. Minimum allowed deadline is 10s."


def _genai_error(code: int, status: str, message: str):
    """The google-genai SDK's own error, as the service's refusal arrives."""
    errors = pytest.importorskip("google.genai.errors")
    kind = errors.ServerError if code >= 500 else errors.ClientError
    return kind(code, {"error": {"code": code, "status": status, "message": message}})


def _tone(seconds: float = 0.2, hz: float = 200.0) -> bytes:
    """Audible 24 kHz PCM (the renderer's silence trimming leaves it alone)."""
    import numpy as np
    t = np.arange(int(seconds * 24000)) / 24000.0
    return (np.sin(2 * np.pi * hz * t) * 8000).astype("<i2").tobytes()


class _Gemini:
    """Stands in for Gemini TTS at voice_charon's transport seam: records each
    request, then streams PCM or raises what the service would. ``fail`` is
    an exception, or a function of the request's text returning one (or None)."""

    def __init__(self, fail=None, pcm: bytes | None = None) -> None:
        self.fail = fail
        self.pcm = pcm if pcm is not None else _tone()
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, model, contents, voice_name, emit, should_stop):
        self.calls.append((model, contents, voice_name))
        error = self.fail(contents) if callable(self.fail) else self.fail
        if error is not None:
            raise error
        for start in range(0, len(self.pcm), 960):
            emit(self.pcm[start:start + 960])
        return True


class _Renderer:
    """ORION's one output stream, as _speak_cloud feeds it."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def enqueue(self, chunk: bytes) -> None:
        self.chunks.append(chunk)


def test_charon_sends_its_own_server_deadline_not_one_made_from_the_8s_timeout():
    """16 Oct 2026, 0 of 164 lines in his voice: the SDK turned the client's
    8 s timeout into ``X-Server-Timeout: 8`` and the service refused every
    request. The deadline is sent explicitly and the client's own per-read
    timeout stays short, so a dead network still fails fast. Built against
    the installed SDK; no request is made."""
    pytest.importorskip("google.genai")
    from orion_core import voice_charon

    voice = voice_charon.CharonVoice(key_source=lambda: "test-key")
    client = voice._client_for("test-key")
    assert voice._client_for("test-key") is client
    api = client._api_client
    if hasattr(api, "_build_request"):
        request = api._build_request(
            "post", "models/gemini-3.8-flash-lite-tts:streamGenerateContent", {}, None)
        headers, timeout = request.headers, request.timeout
    else:
        headers, timeout = api._http_options.headers, api._http_options.timeout / 1000.0
    assert headers["X-Server-Timeout"] == str(voice_charon.SERVER_DEADLINE_S)
    assert int(headers["X-Server-Timeout"]) >= 10
    assert timeout == voice_charon.CLIENT_TIMEOUT_MS / 1000.0


def test_the_deadline_refusal_is_config_billing_is_auth_and_a_quota_stays_quota():
    from orion_core import voice_charon

    deadline = _genai_error(400, "INVALID_ARGUMENT", _DEADLINE)
    assert voice_charon.failure_kind(deadline) == "config"          # was "other"
    assert "Minimum allowed deadline is 10s" in voice_charon.describe_failure(deadline)
    billing = _genai_error(400, "FAILED_PRECONDITION",
                           "Gemini API free tier is not available in your country. "
                           "Please enable billing on your project in Google AI Studio.")
    assert voice_charon.failure_kind(billing) == "auth"
    spent = _genai_error(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota, "
                         "please check your plan and billing details.")
    assert voice_charon.failure_kind(spent) == "quota"
    assert voice_charon.failure_kind(
        _genai_error(403, "PERMISSION_DENIED", "Method does not allow unregistered callers.")) == "auth"
    assert voice_charon.failure_kind(
        _genai_error(503, "UNAVAILABLE", "The model is overloaded.")) == "network"
    assert voice_charon.failure_kind(
        OSError("[WinError 10054] An existing connection was forcibly closed")) == "network"
    assert voice_charon.failure_kind(ValueError("an unexpected reply")) == "other"


def test_one_blip_stands_charon_down_for_seconds_not_minutes(tmp_path, monkeypatch):
    """17 Oct: any failure but a quota stood Charon down for 2-15 minutes and
    the system voice said every new line meanwhile."""
    from orion_core import voice_charon
    monkeypatch.setattr(voice_charon, "_ON", True)
    gemini = _Gemini(fail=OSError("[WinError 10054] An existing connection was forcibly closed"))
    voice = voice_charon.CharonVoice(key_source=lambda: "k", transport=gemini, cache=tmp_path)

    assert voice.stream_pcm("The report is ready.") is None
    assert voice.last_reason.startswith("network")
    assert 0 < voice._cooldown_until - time.monotonic() <= voice_charon.BLIP_COOLDOWN_S
    assert not voice.reachable()

    def waits(error, times: int) -> list[int]:
        out = []
        for _ in range(times):
            voice._stand_down(error)
            out.append(round(voice._cooldown_until - time.monotonic()))
        return out

    # Still failing: each failure in a row doubles it, up to the network ceiling.
    assert waits(gemini.fail, 4) == [10, 20, 30, 30]

    # The connection is back: once the stand-down lapses, a line that works
    # ends the run, and the next blip is seconds again.
    gemini.fail = None
    voice._cooldown_until = time.monotonic() - 1.0
    assert voice.stream_pcm("The report is ready.") and voice.last_source == "gemini"
    assert voice._failures == 0 and voice.reachable()
    assert waits(OSError("connection reset"), 1) == [round(voice_charon.BLIP_COOLDOWN_S)]

    # A refusal that is the same for every line (16 Oct) reaches the long
    # stand-down - but only by failing again and again.
    voice._worked()
    climb = waits(_genai_error(400, "INVALID_ARGUMENT", _DEADLINE), 9)
    assert climb[0] == round(voice_charon.BLIP_COOLDOWN_S) and climb == sorted(climb)
    assert climb[-1] == voice_charon.CONFIG_COOLDOWN_S

    # A refused key and billing are not blips: the whole cooldown at once.
    for lasting in (_genai_error(403, "PERMISSION_DENIED", "Method does not allow unregistered callers."),
                    _genai_error(400, "FAILED_PRECONDITION", "Please enable billing on your project.")):
        voice._worked()
        assert waits(lasting, 1) == [voice_charon.CONFIG_COOLDOWN_S]


def test_a_cached_line_is_said_in_charon_while_he_is_stood_down(tmp_path, monkeypatch):
    """Whatever stood him down, a line already in his cache never goes to
    another voice: stream_pcm looks there before anything else."""
    from orion_core import voice_charon
    monkeypatch.setattr(voice_charon, "_ON", True)
    monkeypatch.delenv("ORION_CHARON_VOICE", raising=False)
    gemini = _Gemini(fail=_genai_error(403, "PERMISSION_DENIED",
                                       "Method does not allow unregistered callers."))
    voice = voice_charon.CharonVoice(key_source=lambda: "k", transport=gemini, cache=tmp_path)
    line = _tone(0.5, 180.0)
    voice._remember("Good evening, sir.", line)
    assert voice.stream_pcm("Something he has never said.") is None
    assert not voice.reachable()                         # a refused key: 15 minutes
    asked = len(gemini.calls)

    chunks: list[bytes] = []
    assert voice.stream_pcm("Good evening, sir.", on_chunk=chunks.append) == line
    assert voice.last_source == "cache" and b"".join(chunks) == line
    assert len(gemini.calls) == asked                    # nothing asked while stood down

    synth = _synth()
    synth._charon = voice
    synth._renderer = _Renderer()
    synth._elevenlabs = _StubElevenLabs(available=False)
    synth._engine = object()
    spoke: list[str] = []
    monkeypatch.setattr(synth, "_speak_pyttsx3", spoke.append)
    monkeypatch.setattr(synth, "_drain_renderer", lambda: None)
    synth._speak_one("Good evening, sir.")
    assert synth._active_backend == "charon" and spoke == [] and synth._renderer.chunks

    # One new sentence with it: his line in his voice, only the new one handed on.
    synth._renderer.chunks.clear()
    synth._speak_one("Good evening, sir. The build has finished.")
    assert spoke == ["The build has finished."] and synth._renderer.chunks
    assert len(gemini.calls) == asked


def test_the_speaking_path_uses_the_one_shared_charon_voice(monkeypatch):
    """audio.py built its own CharonVoice, although the warm-up and render_pcm
    were meant to share the speaking path's: one client, one quota state, one
    set of stand-downs - and its notices now reach the session log."""
    from orion_core import voice_charon
    shared = voice_charon.CharonVoice(key_source=lambda: "k")
    monkeypatch.setattr(voice_charon, "_SHARED", shared)
    logged: list[str] = []
    bus = _StubBus()
    bus.log = SimpleNamespace(emit=logged.append, connect=lambda *a, **k: None)
    synth = SpeechSynthesiser(bus)
    assert synth._charon is shared and voice_charon.shared_voice() is shared
    shared._stand_down(_genai_error(403, "PERMISSION_DENIED", "Method does not allow unregistered callers."))
    assert any("could not speak - auth" in line for line in logged)


def test_render_pcm_shares_his_voice_and_never_keeps_a_styled_line(tmp_path, monkeypatch):
    """render_pcm is the singing lane's way into his voice: the same voice,
    quota and stand-downs as speech; a styled line is never cached; a cached
    unstyled line costs no request."""
    from orion_core import voice_charon
    monkeypatch.delenv("ORION_CHARON_VOICE", raising=False)
    gemini = _Gemini()
    voice = voice_charon.CharonVoice(key_source=lambda: "k", transport=gemini, cache=tmp_path)
    monkeypatch.setattr(voice_charon, "_SHARED", voice)
    monkeypatch.setattr(voice_charon, "_ON", False)
    with pytest.raises(voice_charon.CharonUnavailable) as off:
        voice_charon.render_pcm("Hello.")
    assert off.value.kind == "off" and gemini.calls == []

    monkeypatch.setattr(voice_charon, "_ON", True)
    voice._stand_down(OSError("connection reset"))       # the speaking path had a blip
    chunks: list[bytes] = []
    pcm = voice_charon.render_pcm("Happy. Birthday.", style="Sing it softly",
                                  on_chunk=chunks.append)
    assert pcm == gemini.pcm and b"".join(chunks) == pcm
    assert gemini.calls[-1][1] == "Sing it softly: Happy. Birthday."
    assert voice.cache_stats()["files"] == 0             # a styled line is never kept
    assert voice.reachable()                             # a line that worked ended it

    voice._remember("Good evening.", _tone(0.1))
    asked = len(gemini.calls)
    assert voice_charon.render_pcm("Good evening.") == _tone(0.1)
    assert len(gemini.calls) == asked

    gemini.fail = _genai_error(400, "INVALID_ARGUMENT", _DEADLINE)
    with pytest.raises(voice_charon.CharonUnavailable) as refused:
        voice_charon.render_pcm("Something new.")
    assert refused.value.kind == "config"


def test_prewarm_resumes_a_version_1_day_as_version_2(tmp_path):
    """The owner's warm-up file on 17 Oct was version 1: 96 refused requests
    counted against the day, every refused line on the skip list, nothing
    cached. Version 2 bounds the day's count by what the cache really holds
    and drops those skips, so the warm-up resumes the same day; a version 2
    count is the day's real count and is kept."""
    from orion_core import voice_charon
    lines = ["Good evening.", "Welcome back.", "Goodbye, sir.", "Right away."]
    path = tmp_path / "prewarm.json"
    path.write_text(json.dumps({"day": "today", "requests": 96, "blocked": "",
                                "skip": [line.lower() for line in lines]}), encoding="utf-8")
    gemini = _Gemini()
    voice = voice_charon.CharonVoice(key_source=lambda: "k", transport=gemini, cache=tmp_path)
    voice._remember("Good evening.", _tone(0.1))           # one line really was cached
    kwargs = dict(lines=lambda: lines, online=lambda: True, interval_s=0, idle_s=0,
                  daily_cap=4, today=lambda: "today", state_path=path)

    assert voice_charon.CharonPrewarm(voice, **kwargs).fill() == "complete"
    assert [c for _m, c, _v in gemini.calls] == [voice_charon.spoken_form(x) for x in lines[1:]]
    assert {m for m, _c, _v in gemini.calls} == {voice_charon.MODELS[0]}   # the warm-up model only
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["v"] == voice_charon.PREWARM_STATE_VERSION == 2
    assert state["requests"] == 4 and state["skip"] == []    # min(96, 1 cached) + 3 warmed

    lines.append("Of course.")
    assert voice_charon.CharonPrewarm(voice, **kwargs).fill() == "daily-cap"
    assert len(gemini.calls) == 3


def test_prewarm_pauses_after_three_refusals_in_a_row_and_skips_nothing(tmp_path):
    """16 Oct: 120 refusals in a row, each counted against the day and its
    line skipped. Now three failures in a row pause the warm-up - nothing
    counted, nothing skipped, the reason logged once, the speaking path left
    alone; when lines work again only the ones that failed just before are
    left for tomorrow."""
    from orion_core import voice_charon
    deadline = _genai_error(400, "INVALID_ARGUMENT", _DEADLINE)
    gemini = _Gemini(fail=deadline)
    logged: list[str] = []
    voice = voice_charon.CharonVoice(key_source=lambda: "k", transport=gemini, cache=tmp_path)
    lines = ["Good evening.", "Welcome back.", "Goodbye, sir.", "Right away.", "Of course."]
    path = tmp_path / "prewarm.json"
    warm = voice_charon.CharonPrewarm(voice, lines=lambda: lines, online=lambda: True,
                                      interval_s=0, idle_s=0, today=lambda: "today",
                                      log=logged.append, state_path=path)
    assert warm.fill() == "failing"
    assert len(gemini.calls) == voice_charon.PREWARM_MAX_FAILURES == 3
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["requests"] == 0 and state["skip"] == [] and state["v"] == 2
    assert sum("refused - config" in line for line in logged) == 1
    assert voice.reachable()

    gemini.fail = lambda text: deadline if text in {"Good evening.", "Welcome back."} else None
    assert warm.fill() == "skipped"
    state = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(state["skip"]) == ["good evening.", "welcome back."]
    assert state["requests"] == 3
    assert all(voice.is_cached(line) for line in lines[2:])


@pytest.mark.parametrize("faulty_policy", [False, True])
def test_forced_offline_charon_blocks_every_uncached_entry_point_and_keeps_cached_speech(
        tmp_path, monkeypatch, faulty_policy):
    from orion_core import voice_charon

    def offline():
        if faulty_policy:
            raise RuntimeError("synthetic connectivity fault")
        return False

    def transport(*_):
        pytest.fail("offline voice contacted a cloud transport")

    monkeypatch.setattr(voice_charon, "_ON", True)
    monkeypatch.delenv("ORION_CHARON_VOICE", raising=False)
    voice = voice_charon.CharonVoice(key_source=lambda: "synthetic", online=offline,
                                     transport=transport, cache=tmp_path, honorific="")
    monkeypatch.setattr(voice_charon, "_SHARED", voice)
    assert not voice.reachable() and not voice.network_available()
    assert voice.stream_pcm("An uncached offline sentence.") is None
    for invoke in (
        lambda: voice_charon.render_pcm("An uncached offline sentence."),
        lambda: voice.prewarm_one("An uncached offline sentence."),
    ):
        with pytest.raises(voice_charon.CharonUnavailable) as unavailable:
            invoke()
        assert unavailable.value.kind == "network"
    pcm = _tone(.1)
    voice._remember("A cached sentence.", pcm)
    chunks = []
    assert voice.stream_pcm("A cached sentence.", on_chunk=chunks.append) == pcm
    assert b"".join(chunks) == pcm
    assert voice_charon.render_pcm("A cached sentence.") == pcm
    assert voice.prewarm_one("A cached sentence.") == len(pcm)


def test_speech_prewarm_uses_the_same_network_policy_as_foreground_voice(tmp_path, monkeypatch):
    import threading
    from orion_core import voice_charon

    monkeypatch.setattr(voice_charon, "_ON", True)
    monkeypatch.delenv("ORION_CHARON_VOICE", raising=False)
    voice = voice_charon.CharonVoice(
        key_source=lambda: "synthetic", online=lambda: False, cache=tmp_path,
        synthesise=lambda *_: pytest.fail("offline prewarm invoked synthesis"))
    synth = SpeechSynthesiser.__new__(SpeechSynthesiser)
    synth._charon = voice
    synth._prewarm = None
    synth.stop_event = threading.Event()
    synth._busy_for_prewarm = lambda: False
    synth.bus = _StubBus()
    started = []

    class Prewarm:
        def __init__(self, same_voice, *, online, **_):
            assert same_voice is voice
            self.online = online

        def start(self):
            started.append(self.online())

    monkeypatch.setattr(voice_charon, "CharonPrewarm", Prewarm)
    monkeypatch.setattr(voice_charon, "network_up", lambda: pytest.fail("raw connectivity probe used"))
    synth._start_prewarm()
    assert started == [False]


# ── singing: the sing tool and his voice tuned to a melody (singing.py) ──────

def _spoken_words(text: str, style: str = "") -> bytes:
    """A stand-in for Charon reading "Happy. Birthday. To. You.": per syllable
    a voiced 110 Hz tone with harmonics, a short dip between syllables and a
    pause between words - what segment_syllables looks for in his speech."""
    import numpy as np
    from orion_core import singing
    del style
    sr = singing.SAMPLE_RATE
    pieces = [np.zeros(int(0.2 * sr))]
    for word in text.replace(".", " ").split():
        for _ in range(max(1, singing.count_syllables(word))):
            t = np.arange(int(0.16 * sr)) / sr
            voiced = sum(np.sin(2 * np.pi * 110.0 * k * t) / k for k in range(1, 6))
            pieces.append(0.25 * voiced * np.sin(np.pi * t / t[-1]) ** 0.5)
            pieces.append(np.zeros(int(0.03 * sr)))
        pieces.append(np.zeros(int(0.22 * sr)))
    return (np.concatenate(pieces) * 32767).astype("<i2").tobytes()


def test_singing_tunes_his_words_to_the_melody_over_a_backing():
    import numpy as np
    from orion_core import singing
    asked: list[str] = []

    def charon(text: str, style: str = "") -> bytes:
        asked.append(text)
        return _spoken_words(text, style)

    song = singing.public_domain_song("happy birthday", "Alex")
    performance = singing.perform(song, [singing.Voice("Charon", charon)], keep_vocal=True)

    # Two lyric lines per request, each word read on its own.
    assert asked == ["Happy. Birthday. To. You. Happy. Birthday. To. You.",
                     "Happy. Birthday. Dear. Alex. Happy. Birthday. To. You."]
    assert performance.voice == "Charon" and performance.key.endswith("major")
    pcm = np.frombuffer(performance.pcm, dtype="<i2").astype(np.float64) / 32768.0
    assert abs(len(pcm) / singing.SAMPLE_RATE - performance.seconds) < 0.01
    assert performance.seconds > 15 and np.sqrt(np.mean(pcm ** 2)) > 0.02
    # The intro bar is the piano alone, before the first sung note.
    first = min(start for start, _end, _hz in performance.notes)
    assert first > 1.0 and np.abs(pcm[:int((first - 0.1) * singing.SAMPLE_RATE)]).max() > 0.01
    # Every long note is sung at its pitch: TD-PSOLA moves the pitch and keeps
    # his formants. The dry vocal's f0 over the middle of each note.
    f0 = singing.analyse(performance.vocal)["f0"]
    frames = singing.SAMPLE_RATE / singing.HOP
    errors = []
    for start, end, hz in performance.notes:
        if end - start < 0.4:
            continue
        middle = f0[int((start + 0.3 * (end - start)) * frames):int((end - 0.3 * (end - start)) * frames)]
        middle = middle[middle > 0]
        errors.append(abs(float(np.median(middle)) / hz - 1.0) if middle.size else 1.0)
    assert len(errors) >= 15
    assert max(errors) < 0.02, errors          # MEASURED 18 Oct: within 0.4% (7 cents)


def test_the_sing_tool_sings_through_his_voice_output_and_keeps_the_song(tmp_path, monkeypatch):
    import wave
    from orion_core import singing, voice_charon

    class _Speech:
        muted = False

        def __init__(self) -> None:
            self.chunks: list[bytes] = []
            self.finished = 0

        def enqueue_native_audio(self, chunk: bytes) -> None:
            self.chunks.append(chunk)

        def finish_native_audio(self) -> None:
            self.finished += 1

    speech = _Speech()
    host = SimpleNamespace(live_worker=SimpleNamespace(speech=speech), ai=None,
                           identity=SimpleNamespace(preferences={"honorific": "sir"}))
    monkeypatch.setattr(voice_charon, "_ON", True)             # a real run: his voice is on
    monkeypatch.setattr(singing, "voice_chain", lambda: [singing.Voice("Charon", _spoken_words)])
    monkeypatch.setattr(singing, "songs_folder", lambda: tmp_path)
    monkeypatch.setitem(singing._LAST, "performance", None)
    monkeypatch.setitem(singing._LAST, "path", None)

    result = asyncio.run(singing.handle(host, {"song": "happy birthday", "name": "Alex"}))
    assert result.ok, result.text
    assert "Now singing 'Happy Birthday'" in result.text and "dear Alex" in result.text
    played = b"".join(speech.chunks)
    assert played == singing._LAST["performance"].pcm and speech.finished == 1
    assert max(len(chunk) for chunk in speech.chunks) <= singing.CHUNK_BYTES

    again = asyncio.run(singing.handle(host, {"action": "encore"}))
    assert again.ok and b"".join(speech.chunks) == played * 2
    kept = asyncio.run(singing.handle(host, {"action": "save"}))
    saved = list(tmp_path.glob("*.wav"))
    assert kept.ok and len(saved) == 1 and saved[0].name.endswith("Happy Birthday.wav")
    with wave.open(str(saved[0]), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
        assert wav.readframes(wav.getnframes()) == played

    speech.muted = True
    muted = asyncio.run(singing.handle(host, {"action": "replay"}))
    assert not muted.ok and "no voice output" in muted.text


def test_a_song_is_planned_from_lyrics_a_songwriter_or_his_own_songs():
    from orion_core import singing

    song, origin = asyncio.run(singing.plan_song(SimpleNamespace(ai=None), {
        "lyrics": "The kettle sings, the morning's grey / A proper cup will save the day",
        "style": "shanty"}))
    assert origin == "your words" and song.mood == "shanty"
    assert song.text_lines == ["The kettle sings the morning's grey", "A proper cup will save the day"]
    syllables = sum(singing.count_syllables(w.text) for line in song.lines for w in line)
    assert sum(len(w.notes) for line in song.lines for w in line) == syllables
    assert song.lines[-1][-1].notes[-1][-1][0] == 0.0         # it ends on the tonic

    class _Songwriter:
        def __init__(self) -> None:
            self.asked: list[tuple[str, str]] = []

        async def generate_text(self, prompt, *, instruction=None, task="", max_tokens=None,
                                deadline_s=None):
            self.asked.append((prompt, task))
            return None, json.dumps({"title": "Tea Time", "lines": [
                "The kettle sings upon the stove", "A cup of tea is what I love",
                "With milk and sugar, warm and sweet", "A cup of tea is such a treat"]})

    writer = _Songwriter()
    song, origin = asyncio.run(singing.plan_song(
        SimpleNamespace(ai=SimpleNamespace(router=writer)), {"topic": "tea", "style": "upbeat"}))
    assert origin == "newly written" and song.title == "Tea Time" and len(song.lines) == 4
    assert writer.asked[0][1] == "sing.compose" and "about tea" in writer.asked[0][0]
    assert "Never reproduce lyrics" in writer.asked[0][0]

    class _Offline:
        async def generate_text(self, *args, **kwargs):
            raise RuntimeError("no provider is reachable")

    song, origin = asyncio.run(singing.plan_song(
        SimpleNamespace(ai=SimpleNamespace(router=_Offline())), {"topic": "rain on the window"}))
    assert song.title == "Silver Rain" and origin.startswith("one of my own")


def test_sing_is_a_declared_routed_serial_tool_found_by_its_words():
    from collections import deque
    from orion_core import concurrency, tool_gateway, tool_resolver
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.tool_vocabulary import collisions, terms_for

    declaration = next(d for d in TOOL_DECLARATIONS if d["name"] == "sing")
    assert {"action", "lyrics", "song", "topic", "style", "name", "save"} <= set(
        declaration["parameters"]["properties"])
    assert "copyrighted" in declaration["description"]

    dispatcher = OrionDispatcher.__new__(OrionDispatcher)
    dispatcher.bus = _StubBus()
    dispatcher.telemetry = None
    dispatcher.active_tools = 0
    dispatcher.recent_tools = deque(maxlen=60)
    assert "sing" in dispatcher.handler_table()
    status = asyncio.run(dispatcher.dispatch("sing", {"action": "status"}))
    assert status.ok and "I can sing" in status.text and "Happy Birthday" in status.text
    off = asyncio.run(dispatcher.dispatch("sing", {"song": "happy birthday"}))
    assert not off.ok and "isn't switched on" in off.text      # no test ever sings aloud

    assert concurrency.classify("sing") is concurrency.ToolClass.SERIAL
    assert concurrency.plan_batches(["web_search", "sing", "resource_status"]) == [[0], [1], [2]]
    assert tool_resolver.classify_tool("sing") == "voice"
    assert "lullaby" in terms_for("sing")
    assert not [hit for hit in collisions([d["name"] for d in TOOL_DECLARATIONS]) if "sing" in hit]
    for request in ("sing me a song", "can you sing happy birthday to my mum", "sing a lullaby"):
        scores = tool_resolver.lexical_scores(request, TOOL_DECLARATIONS)
        assert max(scores, key=scores.get) == "sing", request
    # A direct Live tool since 23 Oct 2026: one hop away, "sing me a song" got
    # "not quite equipped". use_tool('sing') (the persona's wording) still works.
    assert "sing" in tool_gateway.LIVE_CORE_TOOLS


def test_the_persona_says_he_can_sing_and_how(tmp_path, monkeypatch):
    """16 Oct: "My vocal capabilities are, shall we say, strictly
    conversational." Nothing in his persona said he could sing."""
    import orion_core.identity as idm
    from orion_core.bus import OrionBus
    from orion_core.singing import SINGING_DIRECTIVE

    monkeypatch.setattr(idm, "IDENTITY_PATH", tmp_path / "identity.json")
    monkeypatch.setattr(idm, "CONFIG_DIR", tmp_path)
    persona = idm.IdentityManager(OrionBus()).persona_text()
    assert SINGING_DIRECTIVE in persona
    assert persona.lstrip().startswith("SPOKEN-NAME RULE")
    assert "use_tool with name 'sing'" in persona and "never refuse" in persona
    # 27 Oct: "sing it, in your own voice" had him RECITE Twinkle Twinkle.
    assert "ONLY through the sing tool" in SINGING_DIRECTIVE
    assert "in your own voice" not in SINGING_DIRECTIVE


def test_a_request_to_sing_is_told_from_talk_about_singing():
    from orion_core.singing import is_sing_request, sing_args
    for asked in ("Can you sing a song for me?", "Ryan, sing me a song",
                  "could you please sing happy birthday to Sam", "sing", "Sing!",
                  "I want you to sing", "will you sing a lullaby about the sea",
                  "give me a little song"):
        assert is_sing_request(asked), asked
    for said in ("I can't sing", "do you like singing", "what songs can you sing",
                 "sing along with me later", "my sister sings in a choir"):
        assert not is_sing_request(said), said
    assert sing_args("could you please sing happy birthday to Sam") == {
        "song": "happy birthday", "name": "Sam"}
    assert sing_args("sing twinkle twinkle") == {"song": "twinkle twinkle little star"}
    assert sing_args("sing a lullaby about the sea") == {"style": "lullaby", "topic": "the sea"}
    assert sing_args("Can you sing a song for me?") == {}


def test_lyrics_recited_in_a_reply_are_recognised_and_kept():
    from orion_core.singing import lyric_lines, reads_lyrics
    reply = ("Ooh, I'd love to! Here's a little tune. *Twinkle, twinkle, little star,* "
             "*Shining bright, wherever you are.* *Like a diamond in the sky,*")
    assert reads_lyrics(reply)
    assert lyric_lines(reply) == ["Twinkle, twinkle, little star", "Shining bright, wherever you are.",
                                  "Like a diamond in the sky"]
    assert not reads_lyrics("That is *very* kind of you, sir.")


def test_typed_sing_goes_straight_to_the_singer_and_nothing_is_read_over_it():
    import inspect
    from orion_core.live_worker import GenAILiveWorker
    from orion_core.reflex import match_reflex
    m = match_reflex("sing me a lullaby")
    assert m is not None and m.tool == "sing" and m.args == {"style": "lullaby"}
    assert match_reflex("can you sing happy birthday to Sam").args["name"] == "Sam"
    assert match_reflex("sing along later") is None
    src = inspect.getsource(GenAILiveWorker._handle_reflex)
    assert 'm.tool == "sing"' in src
    loop = inspect.getsource(GenAILiveWorker._receive_realtime_stream)
    assert "reads_lyrics(" in loop and "_sing_instead(" in loop and "not lyrics_muted" in loop


# ── Kokoro: ORION's offline voice (voice_kokoro), 18 Oct ─────────────────────
# "a downloaded male voice embedded into ORION's offline voice". No real model
# is needed below: a tiny model spec stands in for the 354 MB files and a fake
# engine for kokoro_onnx; the one real-synthesis test skips without the model.

class _FakeKokoroEngine:
    """Stands in for kokoro_onnx.Kokoro: records each piece, returns a tone
    (0.06 s of it per character) at Kokoro's 24 kHz."""

    voices = {"bm_george": None, "bm_lewis": None, "am_michael": None}

    def __init__(self, fail_on: str = "", every_piece_fails: bool = False) -> None:
        self.calls: list[tuple[str, str, float, str]] = []
        self.fail_on = fail_on
        self.every_piece_fails = every_piece_fails

    def create(self, text, voice, speed=1.0, lang="en-us"):
        import numpy as np
        self.calls.append((text, voice, speed, lang))
        if (self.every_piece_fails or (self.fail_on and self.fail_on in text)
                or not any(ch.isalnum() for ch in text)):
            raise ValueError(f"No phonemes of {text!r} are in the model vocabulary")
        samples = int(len(text) * 0.06 * 24000)
        t = np.arange(samples) / 24000.0
        return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 24000


def _tiny_kokoro_model(folder: Path, monkeypatch, *, wrong: bool = False):
    """The model files as a tiny pinned spec, so nothing here needs 354 MB."""
    import hashlib
    from orion_core import voice_kokoro
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, body in ((voice_kokoro.MODEL_FILE, b"onnx-model"),
                       (voice_kokoro.VOICES_FILE, b"voices")):
        (folder / name).write_bytes(body)
        pinned = body if not wrong else body[::-1]           # same size, other bytes
        files[name] = voice_kokoro.ModelFile(len(body), hashlib.sha256(pinned).hexdigest())
    spec = voice_kokoro.ModelSpec(name="kokoro-v1.0", release="https://example.invalid",
                                  files=files)
    monkeypatch.setattr(voice_kokoro, "MODEL", spec)
    return spec


def _kokoro_voice(tmp_path, monkeypatch, engine=None, *, on: bool = True, model: bool = True,
                  clock=None):
    from orion_core import voice_kokoro
    monkeypatch.setattr(voice_kokoro, "_ON", on)
    monkeypatch.delenv("ORION_KOKORO_VOICE", raising=False)
    folder = tmp_path / "models" / "kokoro-v1.0"
    if model:
        _tiny_kokoro_model(folder, monkeypatch)
    loads: list[Path] = []
    engine = engine if engine is not None else _FakeKokoroEngine()

    def loader(found: Path):
        loads.append(found)
        return engine

    voice = voice_kokoro.KokoroVoice(dirs=[folder], engine_loader=loader,
                                     settings=tmp_path / "voices" / "kokoro.json", clock=clock)
    return voice, engine, loads


def test_kokoro_starts_with_a_short_piece_and_never_splits_a_phrase():
    from orion_core import voice_kokoro as vk
    assert vk.plan_chunks("   ") == [] and vk.plan_chunks("") == []
    assert vk.plan_chunks("Good evening, sir.") == [("Good evening, sir.", 0.0)]
    # A long first sentence: split at its clause, so speech starts sooner.
    plan = vk.plan_chunks("It's Sunday the eleventh of October, seven forty-two in the evening.")
    assert plan == [("It's Sunday the eleventh of October,", vk.CLAUSE_PAUSE_S),
                    ("seven forty-two in the evening.", 0.0)]
    # Decimals, initials and titles are not sentence ends; lines are.
    assert vk.split_sentences("Dr. Harper said 3.5 hours. J. Smith agreed!\nNew line here") == [
        "Dr. Harper said 3.5 hours.", "J. Smith agreed!", "New line here"]
    # Nothing to split at: one piece, however long.
    long_plain = "I couldn't find a file called budget forecast in your Documents folder."
    assert vk.plan_chunks(long_plain) == [(long_plain, 0.0)]
    # A short opener then a long sentence: the next piece is cut to what can
    # be made while the opener plays, at a clause or before a conjunction.
    reply = ("Yes. I've looked at your calendar for the week, and there are three meetings "
             "on Tuesday, two of them with the design team, which you might want to move "
             "if the deadline slips again.")
    plan = vk.plan_chunks(reply)
    pieces = [piece for piece, _pause in plan]
    assert pieces[0] == "Yes." and len(pieces) >= 3
    assert " ".join(pieces) == " ".join(reply.split())               # every word, in order
    assert all(pause > 0 for _piece, pause in plan[:-1]) and plan[-1][1] == 0.0
    lead = 0.0
    for index, (piece, pause) in enumerate(plan):
        if index:
            budget = max(vk.MIN_BUDGET_CHARS, lead * vk.MADE_CHARS_PER_S)
            assert len(piece) <= budget or vk._split_at(piece, budget) is None, piece
        lead = (max(0.0, lead - len(piece) / vk.MADE_CHARS_PER_S)
                + len(piece) / vk.SPOKEN_CHARS_PER_S + pause)


def test_kokoro_voice_choice_persists_and_rejects_a_stranger(tmp_path):
    from orion_core import voice_kokoro as vk
    settings = tmp_path / "voices" / "kokoro.json"
    assert vk.current_voice(settings) == vk.DEFAULT_VOICE == "bm_george"
    assert vk.language_for("bm_george") == "en-gb" and vk.language_for("am_michael") == "en-us"
    # Mark XXXVIII: ORION is a man - a woman's voice ("Alice") is refused like a
    # name Kokoro does not have.
    names = ("George", "bm-lewis", "BM_DANIEL", "michael", "Alice", "Zelda", "")
    assert [vk.normalise_voice(n) for n in names] == [
        "bm_george", "bm_lewis", "bm_daniel", "am_michael", None, None, None]
    message, ok = vk.choose_voice("Lewis", settings)
    assert ok and "bm_lewis" in message
    assert json.loads(settings.read_text(encoding="utf-8")) == {"voice": "bm_lewis"}
    assert vk.KokoroVoice(settings=settings).voice == "bm_lewis"     # what speaks next
    message, ok = vk.choose_voice("Zelda", settings)
    assert not ok and "bm_george" in message
    assert vk.current_voice(settings) == "bm_lewis"                   # a refusal changes nothing
    settings.write_text("{not json", encoding="utf-8")
    assert vk.current_voice(settings) == "bm_george"                  # a broken file: the default


def test_kokoro_is_off_in_tests_and_needs_its_pinned_model(tmp_path, monkeypatch):
    from orion_core import voice_charon, voice_kokoro, voice_piper
    voice, _engine, loads = _kokoro_voice(tmp_path, monkeypatch, on=False)
    assert not voice.available() and voice.stream_pcm("Hello.") is None and not loads
    monkeypatch.setattr(voice_kokoro, "_ON", True)
    assert voice.available() and not voice.loaded() and not loads     # lazy: nothing loaded yet
    monkeypatch.setenv("ORION_KOKORO_VOICE", "0")
    assert not voice.available()
    monkeypatch.delenv("ORION_KOKORO_VOICE")

    absent, _engine, _loads = _kokoro_voice(tmp_path / "elsewhere", monkeypatch, model=False)
    assert not absent.available() and "none of" in absent.describe()
    folder = tmp_path / "models" / "kokoro-v1.0"
    _tiny_kokoro_model(folder, monkeypatch)
    assert voice_kokoro.verify(folder) == [] and voice_kokoro.verify(folder, full=True) == []
    (folder / voice_kokoro.VOICES_FILE).write_bytes(b"voice")         # one byte short
    assert voice_kokoro.verify(folder) and voice_kokoro.model_dir([folder]) is None
    _tiny_kokoro_model(folder, monkeypatch, wrong=True)               # right sizes, other bytes
    assert voice_kokoro.verify(folder) == []
    assert any("SHA-256" in problem for problem in voice_kokoro.verify(folder, full=True))

    # A real run switches it on with Charon (orion.py -> voice_charon.enable).
    for module in (voice_charon, voice_kokoro, voice_piper):
        monkeypatch.setattr(module, "_ON", False)
    voice_charon.enable()
    assert voice_kokoro._ON and voice_piper._ON and voice_charon._ON


def test_kokoro_streams_each_piece_as_it_is_made_in_renderer_pcm(tmp_path, monkeypatch):
    import numpy as np
    from orion_core import voice_kokoro as vk
    voice, engine, loads = _kokoro_voice(tmp_path, monkeypatch)
    vk.choose_voice("bm_lewis", tmp_path / "voices" / "kokoro.json")
    seen_at: list[int] = []
    chunks: list[bytes] = []

    def on_chunk(chunk: bytes) -> None:
        seen_at.append(len(engine.calls))
        chunks.append(chunk)

    text = "Good evening, sir. All systems are running normally tonight."
    pcm = voice.stream_pcm(text, "excited", on_chunk=on_chunk)
    assert pcm == b"".join(chunks) and len(loads) == 1
    assert seen_at[0] == 1, "the first piece must play before the next is made"
    assert all(len(chunk) <= vk.CHUNK_BYTES for chunk in chunks)
    assert [call[0] for call in engine.calls] == [p for p, _ in vk.plan_chunks(text)]
    assert {call[1] for call in engine.calls} == {"bm_lewis"}
    assert {call[3] for call in engine.calls} == {"en-gb"}
    assert engine.calls[0][2] == vk.delivery("excited")[0] > vk.delivery("neutral")[0]
    samples = np.frombuffer(pcm, dtype="<i2")
    assert 0 < np.abs(samples).max() <= int(vk.PEAK_LIMIT * 32767) + 1
    assert voice.timing["audio_s"] == len(pcm) / 2 / vk.SAMPLE_RATE and "first_ms" in voice.timing

    # A piece with nothing to say is skipped, the rest still spoken; nothing
    # speakable at all is no voice (None), so the next voice takes the line.
    engine.fail_on = "running"
    assert voice.stream_pcm(text) is not None
    assert voice.stream_pcm("...") is None
    engine.every_piece_fails = True
    assert voice.stream_pcm(text) is None
    engine.every_piece_fails = False
    engine.fail_on = ""
    # An interruption while the first piece plays: no piece after it is made.
    engine.calls.clear()
    played: list[bytes] = []
    assert voice.stream_pcm(text, on_chunk=played.append, should_stop=lambda: bool(played))
    assert len(engine.calls) == 1 and len(played) == 1


def test_kokoro_loads_once_at_a_quiet_moment_and_lets_go_when_idle(tmp_path, monkeypatch):
    import threading
    from orion_core import voice_kokoro as vk
    now = [1000.0]
    voice, engine, loads = _kokoro_voice(tmp_path, monkeypatch, clock=lambda: now[0])
    logged: list[str] = []
    state = {"needed": False, "quiet": False}
    voice.attach(quiet=lambda: state["quiet"], needed=lambda: state["needed"], log=logged.append)
    voice.watch_once()
    assert not loads                               # Charon can speak: nothing to load
    state["needed"] = True
    voice.watch_once()
    assert not loads                               # needed, but ORION is talking
    state["quiet"] = True
    voice.watch_once()
    now[0] += vk.QUIET_SETTLE_S - 0.5
    voice.watch_once()
    assert not loads                               # quiet, but not for long enough yet
    now[0] += 1.0
    voice.watch_once()
    assert len(loads) == 1 and voice.loaded() and "quiet moment" in logged[-1]

    # Idle and Charon back: let the model go - but never while it is speaking.
    state["needed"] = False
    now[0] += vk.IDLE_RELEASE_S + 1
    voice._busy = 1
    voice.watch_once()
    assert voice.loaded()
    voice._busy = 0
    voice.watch_once()
    assert not voice.loaded() and "let go" in logged[-1]

    # Two speakers at once still load it once (thread-safe).
    gate = threading.Event()

    def slow(found):
        loads.append(found)
        gate.wait(2)
        return engine

    voice._loader = slow
    threads = [threading.Thread(target=voice.load) for _ in range(2)]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join(3)
    assert len(loads) == 2 and voice.loaded()

    # A model that will not load is not retried on every line.
    voice.release()

    def broken(found):
        raise RuntimeError("bad model")

    voice._loader = broken
    assert voice.stream_pcm("Hello there.") is None and not voice.available()
    assert voice.last_error == "bad model"
    now[0] += vk.LOAD_RETRY_S + 1
    assert voice.available()
    voice.close()
    assert not voice.loaded()


def _offline_synth(monkeypatch, kokoro, renderer=None):
    synth = _synth()
    synth._renderer = renderer if renderer is not None else _Renderer()
    synth._charon = None
    synth._kokoro = kokoro
    synth._elevenlabs = _StubElevenLabs(available=True, pcm=b"\x01\x00" * 2400)
    elevenlabs: list[str] = []
    monkeypatch.setattr(synth, "_speak_elevenlabs", lambda text: elevenlabs.append(text) or True)
    piper_calls: list[str] = []
    synth._piper = SimpleNamespace(SAMPLE_RATE=24000, available=lambda: True, stop=lambda: None,
                                   warm_async=lambda: piper_calls.append("warm"),
                                   stream_pcm=lambda text, **kw: piper_calls.append(text))
    synth._engine = object()
    sapi: list[str] = []
    monkeypatch.setattr(synth, "_speak_pyttsx3", sapi.append)
    monkeypatch.setattr(synth, "_drain_renderer", lambda: None)
    logged: list[str] = []
    synth.bus = SimpleNamespace(log=SimpleNamespace(emit=logged.append),
                                amplitude=SimpleNamespace(emit=lambda *a: None))
    return synth, elevenlabs, piper_calls, sapi, logged


def test_kokoro_speaks_what_charon_cannot_before_elevenlabs_piper_and_sapi(tmp_path, monkeypatch):
    voice, engine, loads = _kokoro_voice(tmp_path, monkeypatch)
    synth, elevenlabs, piper_calls, sapi, logged = _offline_synth(monkeypatch, voice)
    synth._speak_one("Hello there. I am working offline.")
    assert synth._active_backend == "kokoro" and synth._renderer.chunks
    assert not elevenlabs and not piper_calls and not sapi
    synth._speak_one("And again.")
    announced = [line for line in logged if line.startswith("VOICE: offline voice Kokoro")]
    assert announced == ["VOICE: offline voice Kokoro bm_george (CPU)"]   # once, on first use
    assert len(loads) == 1


def test_without_kokoro_the_chain_falls_back_as_before(tmp_path, monkeypatch):
    absent, _engine, loads = _kokoro_voice(tmp_path, monkeypatch, model=False)
    synth, elevenlabs, _piper, sapi, _logged = _offline_synth(monkeypatch, absent)
    synth._speak_one("Hello there.")
    assert elevenlabs == ["Hello there."] and synth._active_backend == "elevenlabs" and not loads

    # Kokoro there but with nothing to say: the next voice says it.
    mute, _engine, _loads = _kokoro_voice(tmp_path, monkeypatch,
                                          engine=_FakeKokoroEngine(every_piece_fails=True))
    synth, elevenlabs, piper_calls, sapi, _logged = _offline_synth(monkeypatch, mute)
    synth._elevenlabs = _StubElevenLabs(available=False)
    synth._piper = None
    synth._speak_one("Hello there.")
    assert sapi == ["Hello there."] and synth._active_backend == "pyttsx3"

    # A model that fails to load says why once and the line is still spoken.
    def broken(found):
        raise RuntimeError("bad model")

    mute._loader = broken
    mute.release()
    synth, _elevenlabs, _piper, sapi, logged = _offline_synth(monkeypatch, mute)
    synth._elevenlabs = _StubElevenLabs(available=False)
    synth._piper = None
    synth._speak_one("Hello there.")
    synth._speak_one("Hello again.")
    assert sapi == ["Hello there.", "Hello again."]
    assert sum("could not load" in line for line in logged) == 1


class _BusyRenderer(_Renderer):
    """A renderer still playing the previous line until it is waited for."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        from queue import Queue
        self.queue: Queue = Queue()
        self.playing = True
        self.events = events

    def is_active(self) -> bool:
        return self.playing

    def enqueue(self, chunk: bytes) -> None:
        if not self.chunks:
            self.events.append("audio")
        super().enqueue(chunk)


def test_kokoro_is_loaded_only_once_what_is_playing_has_finished(tmp_path, monkeypatch):
    """Building its session holds Python's lock up to 0.5 s - longer than the
    renderer's 0.3 s buffer - so it never happens while audio is playing."""
    events: list[str] = []
    voice, engine, loads = _kokoro_voice(tmp_path, monkeypatch)
    renderer = _BusyRenderer(events)
    synth, *_rest = _offline_synth(monkeypatch, voice, renderer=renderer)
    original = voice._loader
    voice._loader = lambda found: (events.append("load"), original(found))[1]

    def waited(renderer_, timeout=300.0):
        events.append("drained")
        renderer_.playing = False

    monkeypatch.setattr(synth, "_await_renderer", waited)
    synth._speak_one("Hello there.")
    assert events == ["drained", "load", "audio"]
    events.clear()
    renderer.playing = True
    synth._speak_one("Hello again.")                   # loaded: nothing waits for the renderer
    assert events == [] and synth._active_backend == "kokoro"


def test_offline_charon_parts_load_kokoro_before_his_part_plays(tmp_path, monkeypatch):
    from orion_core import voice_charon
    events: list[str] = []
    voice, engine, loads = _kokoro_voice(tmp_path, monkeypatch)
    original = voice._loader
    voice._loader = lambda found: (events.append("load"), original(found))[1]
    renderer = _BusyRenderer(events)
    renderer.playing = False
    synth, _elevenlabs, piper_calls, _sapi, _logged = _offline_synth(monkeypatch, voice,
                                                                     renderer=renderer)
    cached = _tone(0.3)
    synth._charon = SimpleNamespace(
        SAMPLE_RATE=24000, available=lambda: True, reachable=lambda: False,
        stream_pcm=lambda *a, **k: None,
        offline_plan=lambda text: [voice_charon.Segment("Good evening, sir.", cached),
                                   voice_charon.Segment("The build has finished.")])
    synth._speak_one("Good evening, sir. The build has finished.")
    assert events[:2] == ["load", "audio"]             # loaded before his part played
    assert [call[0] for call in engine.calls] == ["The build has finished."]
    assert not piper_calls and synth._charon_cannot_speak()


def test_interrupt_and_shutdown_reach_kokoro(monkeypatch):
    synth = _synth()
    calls: list[str] = []
    synth._kokoro = SimpleNamespace(stop=lambda: calls.append("stop"),
                                    close=lambda: calls.append("close"))
    synth._piper = None
    synth._active_backend = "kokoro"
    stopped: list[int] = []
    monkeypatch.setattr(audio_module.sd, "stop", lambda: stopped.append(1))
    synth.interrupt()
    assert calls == ["stop"] and stopped == [1]
    calls.clear()
    synth.stop()
    assert calls == ["stop", "close"]


def test_kokoro_real_synthesis_when_the_model_is_here(tmp_path, monkeypatch):
    """The real model and engine, when this machine has them (skipped
    otherwise): a short line becomes ORION-level 24 kHz PCM. Plays nothing."""
    import numpy as np
    import build_standalone
    from orion_core import voice_kokoro as vk
    if vk.packages_missing():
        pytest.skip(f"the Kokoro engine is not installed: {vk.packages_missing()}")
    folder = vk.model_dir([*vk.search_dirs(), *build_standalone._kokoro_candidates(None)])
    if folder is None:
        pytest.skip("the Kokoro model is not on this machine")
    monkeypatch.setattr(vk, "_ON", True)
    voice = vk.KokoroVoice(dirs=[folder], settings=tmp_path / "kokoro.json")
    try:
        chunks: list[bytes] = []
        pcm = voice.stream_pcm("Good evening, sir.", on_chunk=chunks.append)
        assert pcm and b"".join(chunks) == pcm
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        assert 0.6 < samples.size / vk.SAMPLE_RATE < 4.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        assert 0.03 < rms < 0.3 and float(np.abs(samples).max()) <= vk.PEAK_LIMIT + 0.01
        assert voice.timing["first_ms"] < 5000
    finally:
        voice.close()
