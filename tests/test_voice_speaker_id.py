"""
Tests for voice_speaker_id.py (Mark XX architectural-audit pass, Track J)
— real speaker recognition (a SPECIFIC enrolled individual, not just
gender). Runs against the real resemblyzer encoder (no mocking — same
"drive the real thing" discipline as test_debugger.py's real pdb
subprocess), using synthetic waveforms shaped enough like speech to survive
resemblyzer's own VAD. Two waveforms built from the SAME base frequency
(different noise) stand in for "the same speaker, a different sample";
waveforms with a different base frequency stand in for "a different
speaker" — verified empirically to land on opposite sides of
MATCH_THRESHOLD before these assertions were written.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from orion_core import voice_speaker_id as vid

_SR = 16000


def _voice(seed: int, base_freq: float, seconds: float = 3.0) -> np.ndarray:
    """A synthetic, speech-shaped waveform: a couple of harmonic components
    plus noise, amplitude-modulated so it isn't a flat tone (a flat tone
    can get stripped entirely by the VAD, same as true silence)."""
    t = np.arange(int(_SR * seconds)) / _SR
    rng = np.random.default_rng(seed)
    sig = (0.3 * np.sin(2 * np.pi * base_freq * t)
           + 0.15 * np.sin(2 * np.pi * base_freq * 2.3 * t)
           + 0.05 * rng.standard_normal(len(t)))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return (sig * envelope).astype(np.float32)


def _silence(seconds: float = 2.0) -> np.ndarray:
    return np.zeros(int(_SR * seconds), dtype=np.float32)


def _isolate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(vid, "PROFILES_PATH", tmp_path / "voice_profiles.json")
    monkeypatch.setattr(vid, "CONFIG_DIR", tmp_path)


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


# ── availability ─────────────────────────────────────────────────────────────

def test_available_is_true_with_the_real_dependency_installed():
    # This session installed resemblyzer/librosa specifically for this
    # track — if this is False, every other test below will also fail,
    # so it's worth asserting on its own for a clear failure signal.
    assert vid.available() is True


# ── consent gate ─────────────────────────────────────────────────────────────

def test_enroll_without_consent_is_refused(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    result = service.enroll("Sam", [_voice(1, 120)], _SR, consent=False)
    assert not result.ok
    assert "consent" in result.text.lower()
    assert service.list_profiles() == []


def test_enroll_without_a_name_is_refused(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    result = service.enroll("", [_voice(1, 120)], _SR, consent=True)
    assert not result.ok


def test_enroll_with_no_samples_is_refused(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    result = service.enroll("Sam", [], _SR, consent=True)
    assert not result.ok


# ── enrolment + persistence ──────────────────────────────────────────────────

def test_enroll_with_consent_stores_a_profile(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    result = service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    assert result.ok
    assert service.list_profiles() == ["Sam"]


def test_enrolled_profile_persists_to_disk_as_json(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    assert vid.PROFILES_PATH.exists()
    import json
    data = json.loads(vid.PROFILES_PATH.read_text(encoding="utf-8"))
    assert "Sam" in data
    assert len(data["Sam"]["embedding"]) == 256


def test_profiles_survive_a_new_service_instance(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    first = vid.SpeakerIdentificationService()
    first.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    second = vid.SpeakerIdentificationService()
    assert second.list_profiles() == ["Sam"]


def test_enrolling_multiple_samples_averages_into_one_centroid(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    result = service.enroll(
        "Sam", [_voice(1, 120), _voice(2, 120), _voice(3, 120)], _SR, consent=True)
    assert result.ok
    assert "3 sample" in result.text


def test_multiple_trusted_people_can_be_enrolled_individually(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    service.enroll("Alex", [_voice(9, 220)], _SR, consent=True)
    assert service.list_profiles() == ["Alex", "Sam"]


def test_enroll_with_unprocessable_audio_fails_cleanly(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    result = service.enroll("Sam", [_silence()], _SR, consent=True)
    assert not result.ok
    assert service.list_profiles() == []


# ── identification ───────────────────────────────────────────────────────────

def test_identify_recognises_the_enrolled_speaker_on_a_new_sample(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    name, score = service.identify(_voice(2, 120), _SR)   # same base freq, different noise
    assert name == "Sam"
    assert score >= vid.MATCH_THRESHOLD


def test_identify_does_not_match_a_different_speaker(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    name, score = service.identify(_voice(3, 220), _SR)   # different base freq
    assert name == ""
    assert score < vid.MATCH_THRESHOLD


def test_identify_picks_the_closest_of_several_enrolled_people(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    service.enroll("Alex", [_voice(9, 220)], _SR, consent=True)
    name, score = service.identify(_voice(2, 120), _SR)
    assert name == "Sam"


def test_identify_with_no_profiles_returns_blank(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    name, score = service.identify(_voice(1, 120), _SR)
    assert name == ""
    assert score == 0.0


def test_identify_with_silence_never_raises(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    name, score = service.identify(_silence(), _SR)
    assert name == ""
    assert score == 0.0


# ── removal (right to be forgotten) ─────────────────────────────────────────

def test_remove_profile_deletes_it(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    result = service.remove_profile("Sam")
    assert result.ok
    assert service.list_profiles() == []


def test_remove_profile_persists_the_deletion(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    first = vid.SpeakerIdentificationService()
    first.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    first.remove_profile("Sam")
    second = vid.SpeakerIdentificationService()
    assert second.list_profiles() == []


def test_remove_profile_for_an_unknown_name_fails_cleanly(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    result = service.remove_profile("Nobody")
    assert not result.ok


# ── describe() ───────────────────────────────────────────────────────────────

def test_describe_with_no_profiles(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    assert "no voice profiles" in service.describe().lower()


def test_describe_lists_enrolled_names(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService()
    service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    assert "Sam" in service.describe()


# ── bus logging (best-effort, never fatal) ──────────────────────────────────

def test_enroll_logs_to_the_bus_when_one_is_attached(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService(bus=_StubBus())
    result = service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    assert result.ok


def test_service_works_with_no_bus_attached(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService(bus=None)
    result = service.enroll("Sam", [_voice(1, 120)], _SR, consent=True)
    assert result.ok
