"""
Tests for the voice_speaker_id dispatcher tool (dispatch_desktop.py) — the
voice/text-reachable surface over voice_speaker_id.py. enroll's actual
microphone recording is stubbed (no real hardware in CI); consent-gating,
routing and the enroll/list/remove/status actions are exercised for real
against a real SpeakerIdentificationService writing to a tmp_path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from orion_core import voice_speaker_id as vid
from orion_core.dispatcher import OrionDispatcher


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


def _isolate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(vid, "PROFILES_PATH", tmp_path / "voice_profiles.json")
    monkeypatch.setattr(vid, "CONFIG_DIR", tmp_path)


def _dispatcher(speaker_id=None) -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _StubBus()
    d.voiceprint_service = speaker_id
    return d


def _fake_clip(seconds: float = 5.0) -> np.ndarray:
    # Same speech-shaped synthetic waveform used in test_voice_speaker_id.py.
    sr = 16000
    t = np.arange(int(sr * seconds)) / sr
    sig = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.05 * np.random.default_rng(1).standard_normal(len(t))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return (sig * envelope).astype(np.float32)


async def test_reports_unavailable_with_no_service():
    d = _dispatcher(None)
    result = await d.voice_speaker_id_tool({"action": "status"})
    assert not result.ok


async def test_enroll_without_a_name_is_refused(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher(vid.SpeakerIdentificationService(bus=_StubBus()))
    result = await d.voice_speaker_id_tool({"action": "enroll", "consent": True})
    assert not result.ok


async def test_enroll_without_consent_is_refused_before_recording(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(vid, "record_enrollment_clip", lambda *a, **k: calls.append(1) or _fake_clip())
    d = _dispatcher(vid.SpeakerIdentificationService(bus=_StubBus()))
    result = await d.voice_speaker_id_tool({"action": "enroll", "name": "Jordan"})
    assert not result.ok
    assert "consent" in result.text.lower()
    assert calls == []   # never touched the microphone without consent


async def test_enroll_with_consent_records_and_stores_a_profile(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(vid, "record_enrollment_clip", lambda *a, **k: _fake_clip())
    service = vid.SpeakerIdentificationService(bus=_StubBus())
    d = _dispatcher(service)
    result = await d.voice_speaker_id_tool(
        {"action": "enroll", "name": "Jordan", "consent": True})
    assert result.ok
    assert service.list_profiles() == ["Jordan"]


async def test_enroll_surfaces_a_microphone_failure_cleanly(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("no input device")

    monkeypatch.setattr(vid, "record_enrollment_clip", _boom)
    d = _dispatcher(vid.SpeakerIdentificationService(bus=_StubBus()))
    result = await d.voice_speaker_id_tool(
        {"action": "enroll", "name": "Jordan", "consent": True})
    assert not result.ok
    assert "microphone" in result.text.lower()


async def test_list_reports_enrolled_names(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService(bus=_StubBus())
    service.enroll("Jordan", [_fake_clip()], 16000, consent=True)
    d = _dispatcher(service)
    result = await d.voice_speaker_id_tool({"action": "list"})
    assert result.ok
    assert "Jordan" in result.text


async def test_list_with_nobody_enrolled(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher(vid.SpeakerIdentificationService(bus=_StubBus()))
    result = await d.voice_speaker_id_tool({"action": "list"})
    assert result.ok
    assert "no voice profiles" in result.text.lower()


async def test_remove_deletes_a_profile(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService(bus=_StubBus())
    service.enroll("Jordan", [_fake_clip()], 16000, consent=True)
    d = _dispatcher(service)
    result = await d.voice_speaker_id_tool({"action": "remove", "name": "Jordan"})
    assert result.ok
    assert service.list_profiles() == []


async def test_status_reports_the_service_description(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher(vid.SpeakerIdentificationService(bus=_StubBus()))
    result = await d.voice_speaker_id_tool({"action": "status"})
    assert result.ok


async def test_unknown_action_fails_cleanly(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher(vid.SpeakerIdentificationService(bus=_StubBus()))
    result = await d.voice_speaker_id_tool({"action": "explode"})
    assert not result.ok
    assert "enroll" in result.text


def test_tool_is_registered_in_the_handler_table():
    d = _dispatcher()
    assert d.handler_table()["voice_speaker_id"] == d.voice_speaker_id_tool


# ── acting on who spoke, not merely recording it ────────────────────────────
#
# Identification that changes nothing is what was already there: presence was
# read, used to colour the tone of a reply, and never allowed to decide
# anything. These cover the three actions that make it a decision — which of
# the enrolled voices is the user's, and whether to answer anyone else.

def _isolate_voiceprints(tmp_path, monkeypatch):
    """Point the gate's store at scratch files.

    It is a process-wide singleton over the user's real enrolments, so a test
    that forgot this would answer from — and write to — whoever is actually
    enrolled on the machine running it.
    """
    from orion_core import voiceprint

    store = voiceprint.Voiceprints(
        path=tmp_path / "voiceprints.json",
        profiles_path=tmp_path / "voice_profiles.json")
    monkeypatch.setattr(voiceprint, "_STORE", store)
    return store


async def test_owner_marks_whose_voice_is_the_users(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    store = _isolate_voiceprints(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService(bus=_StubBus())
    service.enroll("Jordan", [_fake_clip()], 16000, consent=True)
    monkeypatch.setattr(store, "profiles_path", tmp_path / "voice_profiles.json")

    d = _dispatcher(service)
    result = await d.voice_speaker_id_tool({"action": "owner", "name": "Jordan"})
    assert result.ok, result.text
    assert store.owner == "Jordan"


async def test_owner_refuses_a_name_nobody_enrolled(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _isolate_voiceprints(tmp_path, monkeypatch)
    d = _dispatcher(vid.SpeakerIdentificationService(bus=_StubBus()))
    result = await d.voice_speaker_id_tool({"action": "owner", "name": "Nobody"})
    assert not result.ok
    assert "nobody" in result.text.lower()


async def test_only_me_refuses_with_no_voice_enrolled(tmp_path, monkeypatch):
    """It would ignore everybody, including the person who asked."""
    _isolate(tmp_path, monkeypatch)
    store = _isolate_voiceprints(tmp_path, monkeypatch)
    d = _dispatcher(vid.SpeakerIdentificationService(bus=_StubBus()))
    result = await d.voice_speaker_id_tool({"action": "only_me"})
    assert not result.ok
    assert "enrol" in result.text.lower()
    assert store.only_owner is False


async def test_only_me_then_anyone(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    store = _isolate_voiceprints(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService(bus=_StubBus())
    service.enroll("Jordan", [_fake_clip()], 16000, consent=True)

    d = _dispatcher(service)
    assert (await d.voice_speaker_id_tool({"action": "only_me"})).ok
    assert store.only_owner is True

    result = await d.voice_speaker_id_tool({"action": "anyone"})
    assert result.ok
    assert store.only_owner is False


async def test_status_says_whether_he_is_answering_everyone(tmp_path, monkeypatch):
    """The gate is invisible otherwise, and an assistant that has quietly
    stopped listening to the room is a confusing thing to debug."""
    _isolate(tmp_path, monkeypatch)
    store = _isolate_voiceprints(tmp_path, monkeypatch)
    service = vid.SpeakerIdentificationService(bus=_StubBus())
    service.enroll("Jordan", [_fake_clip()], 16000, consent=True)
    d = _dispatcher(service)

    assert "anyone" in (await d.voice_speaker_id_tool({"action": "status"})).text.lower()
    await d.voice_speaker_id_tool({"action": "only_me"})
    assert "only" in (await d.voice_speaker_id_tool({"action": "status"})).text.lower()


async def test_the_new_actions_are_listed_when_one_is_wrong(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _isolate_voiceprints(tmp_path, monkeypatch)
    d = _dispatcher(vid.SpeakerIdentificationService(bus=_StubBus()))
    result = await d.voice_speaker_id_tool({"action": "explode"})
    for action in ("owner", "only_me", "anyone"):
        assert action in result.text
