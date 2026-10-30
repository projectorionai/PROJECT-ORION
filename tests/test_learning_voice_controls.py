"""Explicit lesson controls save the coach, keeping global audio controls distinct."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from orion_core import live_worker as lw, speaker_gate
from orion_core.learning_coach import LearningCoach
from orion_core.voice_interrupt import VoiceInterruptManager, lesson_control_action


@pytest.mark.parametrize("text,action", [
    ("pause my lesson", "coach_pause"), ("resume my lesson", "coach_resume"),
    ("Orion pause my lesson", "coach_pause"),
    ("Orion, resume my lesson!", "coach_resume"),
    ("pause the lesson", "coach_pause"), ("resume the lesson", "coach_resume"),
    ("Orion pause", ""), ("Orion resume", ""), ("pause", ""),
    ("Explain how I can pause my lesson", ""),
    ("Don't pause my lesson", ""), ("pause my lesson and delete it", ""),
])
def test_lesson_matcher_requires_the_whole_explicit_control(text, action):
    assert lesson_control_action(text) == action


def _worker(tmp_path):
    worker = lw.GenAILiveWorker.__new__(lw.GenAILiveWorker)
    coach = LearningCoach(tmp_path / "lesson.json")
    coach.handle("start", pace_minutes=0)
    worker.calls, worker.spoken, worker.episodes, worker.global_controls = [], [], [], []
    async def dispatch(name, args, **kwargs):
        worker.calls.append((name, args, speaker_gate.VOICE_TURN.get()))
        return coach.handle(args["action"])
    worker.dispatcher = SimpleNamespace(dispatch_chain=dispatch)
    worker.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda _: None))
    worker.stop_event = SimpleNamespace(is_set=lambda: False)
    worker.standby_mode = False
    worker.quiet_mode = False
    worker.paused = False
    worker.connected = False
    worker.wake_mode_enabled = False
    worker._wake_gate_open = lambda: True
    worker._note_user_present = lambda: None
    worker._refresh_wake_window = lambda: None
    worker._is_own_echo = lambda _: False
    worker._say = worker.spoken.append
    worker._persist_episode = lambda role, text: worker.episodes.append((role, text))
    worker.interrupts = SimpleNamespace(note_trigger=lambda: None)
    worker.pause = lambda: worker.global_controls.append("pause")
    worker.resume = lambda: worker.global_controls.append("resume")
    return worker, coach


@pytest.mark.parametrize("prefix", ["", "Orion ", "Orion, "])
@pytest.mark.parametrize("connected", [False, True])
def test_spoken_lesson_pause_resume_reach_durable_coach(tmp_path, monkeypatch, prefix, connected):
    worker, coach = _worker(tmp_path)
    worker.connected = connected
    worker._handle_quiet_command = lambda _: pytest.fail("lesson swallowed by quiet control")
    monkeypatch.setattr(lw.background, "spawn", lambda coro: asyncio.run(coro))
    worker._on_local_transcript(prefix + "pause my lesson")
    assert "paused" in LearningCoach(coach.state_path).handle("status").text
    worker._on_local_transcript(prefix + "resume my lesson")
    assert "active" in LearningCoach(coach.state_path).handle("status").text
    assert [(name, args["action"]) for name, args, _ in worker.calls] == [
        ("cyber_curriculum", "coach_pause"), ("cyber_curriculum", "coach_resume")]
    assert all(is_voice for _, _, is_voice in worker.calls)
    assert not worker.global_controls and not worker.paused


def test_typed_lesson_controls_also_bypass_global_quiet(tmp_path):
    worker, _ = _worker(tmp_path)
    worker._handle_quiet_command = lambda _: pytest.fail("lesson swallowed by quiet control")
    asyncio.run(worker.submit_text("pause my lesson"))
    asyncio.run(worker.submit_text("resume my lesson"))
    assert [args["action"] for _, args, _ in worker.calls] == ["coach_pause", "coach_resume"]
    assert not any(is_voice for _, _, is_voice in worker.calls)
    assert not worker.global_controls


def test_closed_wake_gate_rejects_unaddressed_lesson_command(tmp_path, monkeypatch):
    worker, _ = _worker(tmp_path)
    worker.wake_mode_enabled = True
    worker._wake_gate_open = lambda: False
    monkeypatch.setattr(lw.background, "spawn", lambda coro: asyncio.run(coro))
    worker._on_local_transcript("pause my lesson")
    assert not worker.calls
    worker._on_local_transcript("Orion pause my lesson")
    assert worker.calls[0][1] == {"action": "coach_pause"}


@pytest.mark.parametrize("blocked", ["standby", "shutdown"])
def test_lesson_interrupt_does_not_open_standby_or_run_during_shutdown(tmp_path, monkeypatch, blocked):
    worker, _ = _worker(tmp_path)
    worker.standby_mode = blocked == "standby"
    worker.stop_event = SimpleNamespace(is_set=lambda: blocked == "shutdown")
    monkeypatch.setattr(lw.background, "spawn", lambda coro: asyncio.run(coro))
    worker._on_interrupt_phrase("coach_pause")
    assert not worker.calls and not worker.global_controls


def test_global_audio_controls_still_use_the_existing_path(tmp_path):
    worker, _ = _worker(tmp_path)
    for text, expected in (("Orion pause", "pause"), ("Orion resume", "resume")):
        action = VoiceInterruptManager.classify(text)
        assert action == expected
        worker._on_interrupt_phrase(action)
    assert worker.global_controls == ["pause", "resume"]
    assert not worker.calls


@pytest.mark.parametrize("phrase, expected", [
    ("orion pause my lesson", "coach_pause"),
    ("orion resume my lesson", "coach_resume"),
])
def test_command_grammar_distinguishes_lesson_control_while_speaking(tmp_path, monkeypatch, phrase, expected):
    worker, _ = _worker(tmp_path)
    manager = VoiceInterruptManager(worker.bus, None)
    manager._command_rec = SimpleNamespace(
        AcceptWaveform=lambda _: True,
        # The real recogniser enables SetWords(True), and the hearing guard
        # requires measured final-word confidence. Keep the production gate.
        Result=lambda: json.dumps({"text": phrase, "result": [
            {"word": word, "conf": 0.95} for word in phrase.split()]}),
        Reset=lambda: None)
    assert "orion pause my lesson" in manager.describe()["phrases"]
    action = manager.feed(b"synthetic audio")
    assert action == expected
    monkeypatch.setattr(lw.background, "spawn", lambda coro: asyncio.run(coro))
    worker._on_interrupt_phrase(action)
    assert worker.calls[0][1] == {"action": expected}
    assert not worker.global_controls


@pytest.mark.parametrize("payload", [
    {"text": "orion pause my lesson"},
    {"text": "orion pause my lesson", "result": [{"conf": 0.4}]},
    {"partial": "orion pause my lesson"},
    {"text": "orion pause my lesson", "result": [{"conf": float("nan")}]},
])
def test_lesson_grammar_does_not_bypass_the_final_confidence_gate(payload):
    manager = VoiceInterruptManager(SimpleNamespace(), None)
    manager._command_rec = SimpleNamespace(
        AcceptWaveform=lambda _: True, Result=lambda: json.dumps(payload), Reset=lambda: None)
    assert manager.feed(b"synthetic audio") == ""


def test_duplicate_local_detection_has_one_dispatch_and_receipt(tmp_path):
    worker, _ = _worker(tmp_path)
    async def both():
        await asyncio.gather(worker._handle_lesson_control("coach_pause", voice=True),
                             worker._handle_lesson_control("coach_pause", voice=True))
    asyncio.run(both())
    assert len(worker.calls) == 1 and len(worker.spoken) == 1


def test_dispatch_failure_does_not_claim_the_lesson_was_saved(tmp_path):
    worker, coach = _worker(tmp_path)
    async def fail(*args, **kwargs):
        raise OSError("synthetic unavailable tool")
    worker.dispatcher.dispatch_chain = fail
    assert asyncio.run(worker._handle_lesson_control("coach_pause"))
    assert "couldn't update" in worker.spoken[0]
    assert "active" in coach.handle("status").text


def test_missing_dispatcher_uses_normal_text_route_for_explicit_voice_lesson(tmp_path, monkeypatch):
    worker, _ = _worker(tmp_path)
    worker.dispatcher = SimpleNamespace()
    requests = []
    async def submit(text):
        requests.append(text)
    worker.submit_text = submit
    monkeypatch.setattr(lw.background, "spawn", lambda coro: asyncio.run(coro))
    worker._on_local_transcript("pause my lesson")
    assert requests == ["pause my lesson"]
    assert not worker.global_controls
