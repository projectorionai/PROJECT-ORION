"""
Tests for Companion Intelligence (execution-directive priority #5).

Covers the continuity ledger (activities collapse per-day, humanised recency),
goal progress + auto-achievement at 100%, habit streak arithmetic across
days, session continuity, the assembled continuity brief (project awareness
via a cognition snapshot), and the agent's passive 'working on X' observer.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.companion import CompanionAgent, CompanionEngine, _humanise_when


class _FakeCognition:
    def __init__(self, tasks=None):
        self._tasks = tasks or {}

    def snapshot(self):
        return {"pending_tasks": self._tasks}


class _FakeMemory:
    def __init__(self):
        self.records = {}

    def remember(self, tier, key, value, project=""):
        self.records[key] = value
        return "ok"


@pytest.fixture
def engine(tmp_path):
    eng = CompanionEngine(memory=_FakeMemory(), db_path=tmp_path / "companion.db")
    yield eng
    eng.close()


def _stamp(days_ago: int) -> str:
    at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── activities ────────────────────────────────────────────────────────────────

def test_log_activity_and_recent(engine):
    rec = engine.log_activity("worked_on", "ExampleStore", "supplier shortlist")
    assert rec["subject"] == "ExampleStore"
    acts = engine.recent_activities()
    assert len(acts) == 1
    assert acts[0]["kind"] == "worked_on"


def test_same_day_same_subject_collapses(engine):
    engine.log_activity("worked_on", "ExampleStore")
    engine.log_activity("worked_on", "ExampleStore")
    assert len(engine.recent_activities()) == 1


def test_activity_written_to_long_term_memory(engine):
    engine.log_activity("studied", "neuroscience")
    assert any("neuroscience" in v for v in engine.memory.records.values())


def test_humanise_when_buckets():
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)   # a Friday
    assert _humanise_when("2026-07-17T09:00:00Z", now) == "earlier today"
    assert _humanise_when("2026-07-16T09:00:00Z", now) == "yesterday"
    assert _humanise_when("2026-07-13T09:00:00Z", now) == "earlier this week"
    assert _humanise_when("2026-07-06T09:00:00Z", now) == "last week"


# ── goals ─────────────────────────────────────────────────────────────────────

def test_goal_progress_and_achievement_at_100(engine):
    engine.upsert_goal("Ship command deck", progress=40)
    goals = engine.goals()
    assert goals and goals[0].progress == 40
    engine.upsert_goal("Ship command deck", progress=100)
    assert engine.goals() == []                       # no longer open
    achieved = engine.goals(include_achieved=True)
    assert achieved and achieved[0].status == "achieved"
    assert any("Goal achieved" in a["title"] for a in engine.achievements())


def test_goal_progress_clamped(engine):
    g = engine.upsert_goal("Overshoot", progress=250)
    assert g.progress == 100


# ── habits ────────────────────────────────────────────────────────────────────

def test_habit_streak_consecutive_days(engine):
    today = datetime.now(timezone.utc).date()
    for back in (0, 1, 2):
        engine.mark_habit("study", (today - timedelta(days=back)).strftime("%Y-%m-%d"))
    assert engine.habit_streak("study") == 3


def test_habit_streak_broken_by_gap(engine):
    today = datetime.now(timezone.utc).date()
    engine.mark_habit("gym", today.strftime("%Y-%m-%d"))
    engine.mark_habit("gym", (today - timedelta(days=3)).strftime("%Y-%m-%d"))
    assert engine.habit_streak("gym") == 1


def test_habit_streak_survives_unmarked_today(engine):
    today = datetime.now(timezone.utc).date()
    for back in (1, 2):
        engine.mark_habit("read", (today - timedelta(days=back)).strftime("%Y-%m-%d"))
    assert engine.habit_streak("read") == 2


# ── sessions ──────────────────────────────────────────────────────────────────

def test_session_begin_end_and_last(tmp_path):
    eng = CompanionEngine(db_path=tmp_path / "c.db")
    eng.begin_session()
    eng.end_session("built the ingestion engine")
    # A "next boot" engine sees the prior session.
    eng2 = CompanionEngine(db_path=tmp_path / "c.db")
    eng2.begin_session()
    last = eng2.last_session()
    assert last is not None
    assert "ingestion" in last["summary"]
    eng.close()
    eng2.close()


# ── the brief ─────────────────────────────────────────────────────────────────

def test_continuity_brief_mentions_activity_and_open_task(tmp_path):
    cognition = _FakeCognition(tasks={
        "t1": {"title": "finish command deck", "status": "pending",
               "updated_at": "2026-07-16T10:00:00Z"},
    })
    eng = CompanionEngine(cognition=cognition, db_path=tmp_path / "c.db")
    eng.log_activity("worked_on", "ExampleStore")
    brief = eng.continuity_brief()
    assert "ExampleStore" in brief
    assert "finish command deck" in brief
    eng.close()


def test_continuity_brief_falls_back_to_goal(engine):
    engine.log_activity("studied", "neuroscience")
    engine.upsert_goal("Learn BCI decoding", progress=30)
    brief = engine.continuity_brief()
    assert "neuroscience" in brief
    assert "Learn BCI decoding" in brief and "30%" in brief


def test_brief_empty_when_no_history(engine):
    assert engine.continuity_brief() == ""


# ── the agent ─────────────────────────────────────────────────────────────────

def test_agent_observes_working_on(engine):
    agent = CompanionAgent(engine)
    rec = agent.observe("I'm working on ExampleStore today and it's going well")
    assert rec and rec["subject"].startswith("ExampleStore")
    assert engine.recent_activities()[0]["kind"] == "worked_on"


def test_agent_observes_studying(engine):
    agent = CompanionAgent(engine)
    rec = agent.observe("I was studying neuroscience this morning")
    assert rec and rec["kind"] == "studied"


def test_agent_ignores_junk_subjects(engine):
    agent = CompanionAgent(engine)
    assert agent.observe("I'm working on it right now") is None
    assert agent.observe("nothing about work here") is None


def test_status_counts(engine):
    engine.log_activity("worked_on", "ExampleStore")
    engine.upsert_goal("Ship", progress=10)
    engine.mark_habit("study")
    engine.record_achievement("First ingest")
    s = engine.status()
    assert s["activities"] == 1 and s["open_goals"] == 1
    assert s["achievements"] == 1 and s["habits"] == 1
