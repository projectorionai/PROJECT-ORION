"""
Tests for voice_elevenlabs.py — ElevenLabs as a real, swappable voice
provider for SpeechSynthesiser's local/fallback path (Mark XX design-spec,
"ElevenLabs Integration"). Pure and dependency-free beyond urllib/numpy, so
tested directly with no Qt/threading involved.
"""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from orion_core import voice_elevenlabs as ev


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "API_CONFIG_PATH", tmp_path / "api_keys.json")
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "tts_cache")


# ── configuration ────────────────────────────────────────────────────────────

def test_resolve_defaults_to_blank(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert ev.resolve_api_key() == ""
    assert ev.resolve_voice_id() == ""


def test_set_voice_persists_both_fields(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test", voice_id="v123")
    assert ev.resolve_api_key() == "sk-test"
    assert ev.resolve_voice_id() == "v123"


def test_set_voice_updates_one_field_without_clearing_the_other(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test", voice_id="v123")
    ev.set_voice(voice_id="v456")
    assert ev.resolve_api_key() == "sk-test"   # untouched
    assert ev.resolve_voice_id() == "v456"


def test_set_voice_preserves_the_rest_of_api_keys_json(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.API_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ev.API_CONFIG_PATH.write_text('{"providers": {"gemini": {"api_key": "g"}}}', encoding="utf-8")
    ev.set_voice(api_key="sk-test", voice_id="v123")
    import json
    data = json.loads(ev.API_CONFIG_PATH.read_text(encoding="utf-8"))
    assert data["providers"]["gemini"]["api_key"] == "g"
    assert data["elevenlabs"]["api_key"] == "sk-test"


def test_describe_reflects_configuration_state(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert "not set" in ev.describe()
    ev.set_voice(api_key="sk-test", voice_id="v123")
    assert "configured" in ev.describe()
    assert "v123" in ev.describe()


# ── pronunciation ────────────────────────────────────────────────────────────

def test_pronunciation_dictionary_includes_the_default_entry(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    table = ev.pronunciation_dictionary()
    assert table["ORION"] == "OH-ree-on"


def test_pronunciation_dictionary_merges_custom_overrides(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev._save_elevenlabs_section({"pronunciation": {"GIF": "jiff"}})
    table = ev.pronunciation_dictionary()
    assert table["ORION"] == "OH-ree-on"
    assert table["GIF"] == "jiff"


def test_apply_pronunciation_substitutes_case_insensitively_at_word_boundaries():
    text = "orion, can you hear me? ORION!"
    result = ev.apply_pronunciation(text, {"ORION": "OH-ree-on"})
    assert result == "OH-ree-on, can you hear me? OH-ree-on!"


def test_apply_pronunciation_does_not_touch_substrings():
    result = ev.apply_pronunciation("orionic gadget", {"ORION": "OH-ree-on"})
    assert result == "orionic gadget"   # word-boundary match only, not a substring hit


def test_apply_pronunciation_with_empty_table_is_a_no_op():
    assert ev.apply_pronunciation("orion", {}) == "orion"


# ── emotion -> voice settings ────────────────────────────────────────────────

def test_voice_settings_for_known_emotion():
    settings = ev.voice_settings_for_emotion("excited")
    assert settings == ev._EMOTION_VOICE_SETTINGS["excited"]


def test_voice_settings_for_unknown_emotion_falls_back_to_neutral():
    assert ev.voice_settings_for_emotion("bewildered") == ev._EMOTION_VOICE_SETTINGS["neutral"]


def test_voice_settings_returns_a_copy_not_the_shared_dict():
    a = ev.voice_settings_for_emotion("calm")
    a["stability"] = 999.0
    assert ev.voice_settings_for_emotion("calm")["stability"] != 999.0


# ── ElevenLabsVoice.available() ─────────────────────────────────────────────

def test_available_is_false_with_nothing_configured(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert ev.ElevenLabsVoice().available() is False


def test_available_is_false_with_only_a_key(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test")
    assert ev.ElevenLabsVoice().available() is False


def test_available_is_true_with_both_key_and_voice(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test", voice_id="v123")
    assert ev.ElevenLabsVoice().available() is True


# ── synthesize_pcm ────────────────────────────────────────────────────────────

def test_synthesize_pcm_returns_none_when_unavailable(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert ev.ElevenLabsVoice().synthesize_pcm("hello") is None


def test_synthesize_pcm_returns_none_for_blank_text(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test", voice_id="v123")
    assert ev.ElevenLabsVoice().synthesize_pcm("   ") is None


def test_synthesize_pcm_calls_the_api_and_caches_the_result(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test", voice_id="v123")
    voice = ev.ElevenLabsVoice()
    calls = []
    monkeypatch.setattr(voice, "_request_pcm", lambda *a: calls.append(a) or b"\x01\x02\x03\x04")

    first = voice.synthesize_pcm("hello there")
    assert first == b"\x01\x02\x03\x04"
    assert len(calls) == 1

    second = voice.synthesize_pcm("hello there")   # same text/voice/emotion -> cache hit
    assert second == b"\x01\x02\x03\x04"
    assert len(calls) == 1   # no second HTTP call


def test_synthesize_pcm_returns_none_and_does_not_raise_on_network_failure(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test", voice_id="v123")
    voice = ev.ElevenLabsVoice()

    def _boom(*a):
        raise urllib.error.URLError("no connection")

    monkeypatch.setattr(voice, "_request_pcm", _boom)
    assert voice.synthesize_pcm("hello") is None


def test_synthesize_pcm_applies_pronunciation_before_the_cache_key(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test", voice_id="v123")
    voice = ev.ElevenLabsVoice()
    seen_text = []
    monkeypatch.setattr(voice, "_request_pcm",
                        lambda text, *_a: seen_text.append(text) or b"\x00\x00")
    voice.synthesize_pcm("ORION reporting")
    assert seen_text == ["OH-ree-on reporting"]


def test_cache_evicts_the_oldest_file_beyond_capacity(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(ev, "MAX_CACHE_FILES", 2)
    ev.set_voice(api_key="sk-test", voice_id="v123")
    voice = ev.ElevenLabsVoice()
    calls = []
    monkeypatch.setattr(voice, "_request_pcm",
                        lambda text, *_a: calls.append(text) or text.encode())
    voice.synthesize_pcm("first")
    voice.synthesize_pcm("second")
    voice.synthesize_pcm("third")
    assert len(list(ev.CACHE_DIR.glob("*.pcm"))) == 2


# ── apply_fade ────────────────────────────────────────────────────────────────

def test_apply_fade_preserves_length():
    samples = np.full(1000, 1000, dtype=np.int16)
    faded = ev.apply_fade(samples, fade_frames=100)
    assert len(faded) == len(samples)


def test_apply_fade_ramps_the_very_first_and_last_sample_toward_zero():
    samples = np.full(1000, 1000, dtype=np.int16)
    faded = ev.apply_fade(samples, fade_frames=100)
    assert faded[0] == 0
    assert faded[-1] == 0
    assert faded[500] == 1000   # middle untouched


def test_apply_fade_on_empty_array_does_not_raise():
    result = ev.apply_fade(np.array([], dtype=np.int16), fade_frames=100)
    assert len(result) == 0


def test_apply_fade_on_short_array_does_not_raise():
    samples = np.array([500, 500, 500], dtype=np.int16)
    result = ev.apply_fade(samples, fade_frames=100)
    assert len(result) == 3


# ── list_voices ───────────────────────────────────────────────────────────────

def test_list_voices_returns_empty_with_no_api_key(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert ev.list_voices() == []


def test_list_voices_parses_the_account_voice_list(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test")

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            import json
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    payload = {"voices": [{"voice_id": "v1", "name": "Rachel"}, {"voice_id": "v2", "name": "Adam"}]}
    monkeypatch.setattr(ev.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))

    voices = ev.list_voices()
    assert voices == [{"voice_id": "v1", "name": "Rachel"}, {"voice_id": "v2", "name": "Adam"}]


def test_list_voices_returns_empty_and_never_raises_on_request_failure(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test")

    def _boom(*a, **k):
        raise urllib.error.URLError("no connection")

    monkeypatch.setattr(ev.urllib.request, "urlopen", _boom)
    assert ev.list_voices() == []
