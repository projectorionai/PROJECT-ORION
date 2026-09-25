"""
A spoken go-ahead for a sensitive action must be the owner's voice.

The guard is off by default and only ever refuses one thing: a call that
releases an action (confirm / consent / submit), issued through the VOICE
channel, when the last utterance was confidently someone else. Typed calls,
uncertain verdicts, stale verdicts and ordinary calls all pass.

Also pinned here: voiceprint settings are parsed once per change rather than
on every audio chunk, and judge() records its verdict for the guard.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import dispatcher as dispatcher_module
from orion_core import speaker_gate, voiceprint
from orion_core.dispatcher import OrionDispatcher
from orion_core.voiceprint import Verdict

STRANGER = Verdict(False, 0.31, "", "a different voice", confident=True)
OWNER = Verdict(True, 0.93, "SampleUser", "recognised", confident=True)
UNSURE = Verdict(False, 0.66, "", "not a voice ORION knows", confident=False)


@pytest.fixture(autouse=True)
def _fresh_gate():
    speaker_gate.clear()
    yield
    speaker_gate.clear()


def _on():
    return True


def _in_voice_turn(fn):
    with speaker_gate.voice_turn():
        return fn()


# ── the decision ─────────────────────────────────────────────────────────────

def test_a_strangers_spoken_go_ahead_is_refused():
    speaker_gate.note_verdict(STRANGER)
    reason = _in_voice_turn(lambda: speaker_gate.refusal("messaging", {"confirm": True}, _on))
    assert reason is not None
    assert "owner's own voice" in reason and "0.31" in reason


@pytest.mark.parametrize("args", [{"consent": True}, {"submit": True}, {"confirm": "true"}])
def test_every_release_argument_counts(args):
    speaker_gate.note_verdict(STRANGER)
    assert _in_voice_turn(lambda: speaker_gate.refusal("t", args, _on)) is not None


@pytest.mark.parametrize("args", [{}, {"confirm": False}, {"confirm": "false"}, {"text": "hi"}])
def test_a_call_that_releases_nothing_passes(args):
    speaker_gate.note_verdict(STRANGER)
    assert _in_voice_turn(lambda: speaker_gate.refusal("t", args, _on)) is None


def test_typed_requests_are_never_checked():
    speaker_gate.note_verdict(STRANGER)
    assert speaker_gate.refusal("messaging", {"confirm": True}, _on) is None


def test_the_guard_switched_off_passes_everything():
    speaker_gate.note_verdict(STRANGER)
    assert _in_voice_turn(
        lambda: speaker_gate.refusal("t", {"confirm": True}, lambda: False)) is None


@pytest.mark.parametrize("verdict", [OWNER, UNSURE])
def test_the_owner_or_an_uncertain_verdict_passes(verdict):
    speaker_gate.note_verdict(verdict)
    assert _in_voice_turn(lambda: speaker_gate.refusal("t", {"confirm": True}, _on)) is None


def test_no_recent_verdict_passes():
    assert _in_voice_turn(lambda: speaker_gate.refusal("t", {"confirm": True}, _on)) is None
    speaker_gate.note_verdict(STRANGER, now=time.monotonic() - speaker_gate.RECENT_S - 1)
    assert _in_voice_turn(lambda: speaker_gate.refusal("t", {"confirm": True}, _on)) is None


def test_a_broken_setting_read_fails_open():
    speaker_gate.note_verdict(STRANGER)

    def broken():
        raise OSError("settings unreadable")
    assert _in_voice_turn(lambda: speaker_gate.refusal("t", {"confirm": True}, broken)) is None


def test_the_voice_mark_reaches_tasks_created_inside_it():
    async def child():
        return speaker_gate.VOICE_TURN.get()

    async def flow():
        with speaker_gate.voice_turn():
            inside = await asyncio.create_task(child())
        outside = await asyncio.create_task(child())
        return inside, outside
    assert asyncio.run(flow()) == (True, False)


# ── through the dispatcher ───────────────────────────────────────────────────

def _dispatcher():
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.mcp_host = None
    return d


def test_the_dispatcher_refuses_before_routing(monkeypatch):
    monkeypatch.setattr(dispatcher_module, "_action_guard_on", lambda: True)
    speaker_gate.note_verdict(STRANGER)
    d = _dispatcher()

    async def flow():
        with speaker_gate.voice_turn():
            refused = await d.dispatch("no_such_tool", {"confirm": True})
        typed = await d.dispatch("no_such_tool", {"confirm": True})
        return refused, typed
    refused, typed = asyncio.run(flow())
    assert refused.ok is False and "owner's own voice" in refused.text
    # Typed, the same call goes on to normal routing (here: an unknown tool).
    assert "Unknown dispatch target" in typed.text


def test_the_dispatcher_passes_when_the_guard_is_off(monkeypatch):
    monkeypatch.setattr(dispatcher_module, "_action_guard_on", lambda: False)
    speaker_gate.note_verdict(STRANGER)

    async def flow():
        with speaker_gate.voice_turn():
            return await _dispatcher().dispatch("no_such_tool", {"confirm": True})
    assert "Unknown dispatch target" in asyncio.run(flow()).text


# ── voiceprint settings ──────────────────────────────────────────────────────

@pytest.fixture()
def prints(tmp_path, monkeypatch):
    store = voiceprint.Voiceprints(path=tmp_path / "settings.json",
                                   profiles_path=tmp_path / "profiles.json")
    monkeypatch.setattr(voiceprint, "_STORE", store)
    return store


def _enrol_on_disk(store, name="SampleUser"):
    store.profiles_path.write_text(json.dumps({name: {"embedding": [1.0] + [0.0] * 255}}),
                                   encoding="utf-8")


def test_guard_actions_needs_an_enrolled_voice(prints):
    ok, message = prints.set_guard_actions(True)
    assert ok is False and "enrol" in message.lower()
    _enrol_on_disk(prints)
    ok, _ = prints.set_guard_actions(True)
    assert ok and prints.guard_actions and prints.judging
    assert prints.only_owner is False          # listening to everyone still


def test_settings_are_parsed_once_per_change(prints, monkeypatch):
    _enrol_on_disk(prints)
    prints.set_guard_actions(True)
    calls = {"n": 0}
    real_loads = json.loads

    def counting(*a, **k):
        calls["n"] += 1
        return real_loads(*a, **k)
    monkeypatch.setattr(voiceprint.json, "loads", counting)
    for _ in range(50):
        assert prints.judging
    assert calls["n"] <= 2, "the audio loop's question re-parsed the files"


def test_a_change_written_elsewhere_is_seen_at_once(prints):
    _enrol_on_disk(prints)
    assert prints.guard_actions is False
    # Another part of ORION (or another store instance) writes the file.
    prints.path.write_text(json.dumps({"owner": "SampleUser", "guard_actions": True,
                                       "note": "written elsewhere"}), encoding="utf-8")
    assert prints.guard_actions is True


def test_loaded_data_is_a_copy(prints):
    _enrol_on_disk(prints)
    data = prints._load()
    data["profiles"].clear()
    data["owner"] = "somebody else"
    assert prints.names() == ["SampleUser"]


def test_judge_answers_yes_without_computing_when_no_gate_is_on(prints, monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("no gate is on; nothing should be computed")
    monkeypatch.setattr(prints, "is_owner", fail)
    assert voiceprint.judge(b"\x00\x00" * 1600).is_owner is True
    assert speaker_gate.recent_verdict() is None


def test_judge_records_its_verdict_for_the_guard(prints, monkeypatch):
    _enrol_on_disk(prints)
    prints.set_guard_actions(True)
    monkeypatch.setattr(prints, "is_owner", lambda wav, sr: STRANGER)
    verdict = voiceprint.judge(b"\x00\x00" * 1600)
    assert verdict is STRANGER
    assert speaker_gate.recent_verdict() is STRANGER
    # The listening gate is still off: the offline path keeps transcribing.
    assert voiceprint.should_listen(b"\x00\x00" * 1600).is_owner is True
