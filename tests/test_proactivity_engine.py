"""
ProactivityEngine (Mark XXVI, Phase 2).

The decision is pure (gather + plan), so it is tested with no timers or audio:
cooldown must debounce a standing condition, the shared speech gate must hold a
remark while the user is in FOCUS, multiple nudges must coalesce into one, and the
engine must never speak over ORION's own voice.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.focus import FocusEngine, FocusStore  # noqa: E402
from orion_core.proactive_policy import Attention, ProactivePolicy, Urgency  # noqa: E402
from orion_core.proactivity_engine import (  # noqa: E402
    STUDY_DUE_THRESHOLD,
    ProactiveSnapshot,
    ProactivityEngine,
    engine_enabled,
    gather,
    plan,
)
from orion_core.study import StudyEngine, StudyStore  # noqa: E402


def _open_policy() -> ProactivePolicy:
    return ProactivePolicy()          # fresh, never the shared singleton


# ── the fake bus ──────────────────────────────────────────────────────────────

class _Sig:
    def __init__(self):
        self.emitted: list = []

    def emit(self, *a):
        self.emitted.append(a[0] if len(a) == 1 else a)

    def connect(self, *a):
        pass


class _Bus:
    def __init__(self):
        self.speak_request = _Sig()
        self.speaking = _Sig()
        self.log = _Sig()


# ── gather ────────────────────────────────────────────────────────────────────

def test_a_break_due_block_and_a_backlog_each_produce_a_nudge():
    snap = ProactiveSnapshot(focus_break_due=True, focus_label="revise",
                             study_due=STUDY_DUE_THRESHOLD)
    kinds = {n.kind for n in gather(snap)}
    assert kinds == {"focus_break_due", "study_due"}


def test_a_small_backlog_is_below_the_bar():
    assert gather(ProactiveSnapshot(study_due=STUDY_DUE_THRESHOLD - 1)) == []


def test_nothing_happening_says_nothing():
    assert gather(ProactiveSnapshot()) == []


# ── plan: cooldown, gate, coalesce ────────────────────────────────────────────

def test_a_fresh_nudge_is_spoken_when_attention_is_open():
    pol = _open_policy()
    p = plan(gather(ProactiveSnapshot(focus_break_due=True, focus_label="x")),
             clock=1000.0, last_spoken={}, should_speak=pol.should_speak)
    assert "break" in p.utterance
    assert p.spoken_kinds == ["focus_break_due"]


def test_cooldown_debounces_a_standing_condition():
    pol = _open_policy()
    nudges = gather(ProactiveSnapshot(focus_break_due=True))
    # spoken 100 s ago; the break cooldown is 300 s, so it must stay silent.
    p = plan(nudges, clock=1100.0, last_spoken={"focus_break_due": 1000.0},
             should_speak=pol.should_speak)
    assert p.utterance == ""
    assert p.spoken_kinds == []


def test_cooldown_expires_and_the_nudge_returns():
    pol = _open_policy()
    nudges = gather(ProactiveSnapshot(focus_break_due=True))
    p = plan(nudges, clock=1000.0 + 400.0, last_spoken={"focus_break_due": 1000.0},
             should_speak=pol.should_speak)
    assert "break" in p.utterance


def test_focus_attention_holds_an_actionable_nudge():
    pol = _open_policy()
    pol.set_attention(Attention.FOCUS)
    p = plan(gather(ProactiveSnapshot(focus_break_due=True, study_due=99)),
             clock=1.0, last_spoken={}, should_speak=pol.should_speak)
    assert p.utterance == ""
    assert set(p.held_kinds) == {"focus_break_due", "study_due"}


def test_multiple_nudges_coalesce_into_one_utterance():
    pol = _open_policy()
    p = plan(gather(ProactiveSnapshot(focus_break_due=True, focus_label="thesis",
                                      study_due=40)),
             clock=1.0, last_spoken={}, should_speak=pol.should_speak)
    assert "break" in p.utterance and "40 flashcards" in p.utterance
    assert set(p.spoken_kinds) == {"focus_break_due", "study_due"}


# ── snapshot from real engines ────────────────────────────────────────────────

def test_snapshot_reads_a_break_due_block_and_the_due_count(tmp_path):
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    focus.start("write up", minutes=1,
                now=datetime.now(timezone.utc) - timedelta(minutes=3))
    study = StudyEngine(StudyStore(tmp_path / "s.db"))
    for i in range(STUDY_DUE_THRESHOLD + 1):
        study.add(f"q{i}", "a")
    engine = ProactivityEngine(_Bus(), focus=focus, study=study, policy=_open_policy())
    snap = engine.snapshot()
    assert snap.focus_break_due is True
    assert snap.focus_label == "write up"
    assert snap.study_due >= STUDY_DUE_THRESHOLD


# ── the engine tick ───────────────────────────────────────────────────────────

def _engine_with_break(tmp_path):
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    focus.start("deep work", minutes=1,
                now=datetime.now(timezone.utc) - timedelta(minutes=3))
    bus = _Bus()
    return ProactivityEngine(bus, focus=focus, policy=_open_policy()), bus


def test_tick_speaks_then_stays_quiet_within_cooldown(tmp_path):
    engine, bus = _engine_with_break(tmp_path)
    engine.tick(clock=100.0)
    assert len(bus.speak_request.emitted) == 1
    assert "break" in bus.speak_request.emitted[0]
    engine.tick(clock=180.0)                    # 80 s later, cooldown 300 s
    assert len(bus.speak_request.emitted) == 1  # nothing new


def test_tick_never_speaks_over_orions_own_voice(tmp_path):
    engine, bus = _engine_with_break(tmp_path)
    engine._speaking = True
    result = engine.tick(clock=100.0)
    assert result.utterance != ""               # it decided there was something
    assert bus.speak_request.emitted == []       # but did not say it


def test_tick_with_nothing_due_is_silent(tmp_path):
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))    # no block running
    bus = _Bus()
    engine = ProactivityEngine(bus, focus=focus, policy=_open_policy())
    engine.tick(clock=1.0)
    assert bus.speak_request.emitted == []


# ── policy classification + flag ──────────────────────────────────────────────

def test_the_new_kinds_are_actionable_not_ambient():
    from orion_core.proactive_policy import URGENCY_BY_KIND
    assert URGENCY_BY_KIND["focus_break_due"] == Urgency.ACTIONABLE
    assert URGENCY_BY_KIND["study_due"] == Urgency.ACTIONABLE


def test_the_engine_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ORION_PROACTIVE_ENGINE", raising=False)
    assert engine_enabled() is False
    monkeypatch.setenv("ORION_PROACTIVE_ENGINE", "1")
    assert engine_enabled() is True
