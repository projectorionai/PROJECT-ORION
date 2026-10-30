"""
Deep Focus (Mark XXV).

A focus block must measure honestly: elapsed/remaining time, when the break is
due, distractions logged, and a streak that only counts COMPLETED days. Starting
a new block must never leave an old one dangling.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from orion_core.focus import (  # noqa: E402
    DEFAULT_BREAK,
    DEFAULT_WORK,
    FocusEngine,
    FocusSession,
    FocusStore,
    human_minutes,
    resolve_cadence,
)

T0 = datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc)


@pytest.fixture()
def engine(tmp_path):
    return FocusEngine(FocusStore(tmp_path / "focus.db"))


# ── cadence resolution ───────────────────────────────────────────────────────

def test_presets_resolve():
    assert resolve_cadence("deep") == (50, 10)
    assert resolve_cadence("pomodoro") == (25, 5)
    assert resolve_cadence("long") == (90, 20)


def test_explicit_minutes_win_and_scale_the_break():
    work, brk = resolve_cadence("pomodoro", minutes=45)
    assert work == 45
    assert brk == 9            # 45/5, since no break was asked for


def test_unknown_preset_falls_back_to_the_default():
    assert resolve_cadence("nonsense") == (DEFAULT_WORK, DEFAULT_BREAK)


# ── the session clock ────────────────────────────────────────────────────────

def test_elapsed_and_remaining_track_wall_time():
    s = FocusSession(label="x", planned_minutes=50, started_at=T0.isoformat())
    assert s.elapsed_minutes(T0 + timedelta(minutes=20)) == pytest.approx(20, abs=0.1)
    assert s.remaining_minutes(T0 + timedelta(minutes=20)) == pytest.approx(30, abs=0.1)


def test_break_becomes_due_at_the_planned_length():
    s = FocusSession(label="x", planned_minutes=50, started_at=T0.isoformat())
    assert not s.is_break_due(T0 + timedelta(minutes=49))
    assert s.is_break_due(T0 + timedelta(minutes=51))


# ── engine lifecycle ─────────────────────────────────────────────────────────

def test_start_creates_a_running_block(engine):
    s = engine.start("revise action potentials", preset="deep", now=T0)
    assert s.id is not None
    active = engine.active()
    assert active is not None and active.label == "revise action potentials"
    assert active.ended_at == ""


def test_starting_again_finishes_the_previous_block(engine):
    first = engine.start("a", minutes=25, now=T0)
    # start a second only 10 min in — the first had NOT reached its length
    engine.start("b", minutes=25, now=T0 + timedelta(minutes=10))
    reloaded = engine.store.all()
    prev = next(s for s in reloaded if s.id == first.id)
    assert prev.ended_at != ""
    assert prev.completed is False           # abandoned early, doesn't count
    assert engine.active().label == "b"


def test_complete_ends_the_block_and_counts(engine):
    engine.start("deep work", minutes=50, now=T0)
    done = engine.complete(now=T0 + timedelta(minutes=50))
    assert done.completed is True
    assert engine.active() is None


def test_cancel_does_not_count(engine):
    engine.start("x", now=T0)
    engine.cancel(now=T0 + timedelta(minutes=5))
    assert engine.active() is None
    assert engine.stats()["completed"] == 0


def test_interruptions_are_logged(engine):
    engine.start("x", now=T0)
    engine.interrupt()
    engine.interrupt()
    assert engine.active().interruptions == 2


def test_interrupt_with_nothing_running_is_safe(engine):
    assert engine.interrupt() is None


# ── streak ───────────────────────────────────────────────────────────────────

def _seed_completed_day(engine, day: datetime):
    s = FocusSession(label="x", planned_minutes=50, started_at=day.isoformat(),
                     ended_at=(day + timedelta(minutes=50)).isoformat(), completed=True)
    engine.store.add(s)


def test_streak_counts_consecutive_completed_days(engine):
    today = datetime(2026, 8, 12, 10, tzinfo=timezone.utc)
    for d in range(3):
        _seed_completed_day(engine, today - timedelta(days=d))
    assert engine.streak(now=today) == 3


def test_streak_breaks_on_a_missed_day(engine):
    today = datetime(2026, 8, 12, 10, tzinfo=timezone.utc)
    _seed_completed_day(engine, today)
    _seed_completed_day(engine, today - timedelta(days=2))   # gap at day-1
    assert engine.streak(now=today) == 1


def test_streak_survives_a_yet_empty_today_if_yesterday_counted(engine):
    today = datetime(2026, 8, 12, 10, tzinfo=timezone.utc)
    _seed_completed_day(engine, today - timedelta(days=1))
    assert engine.streak(now=today) == 1


def test_a_cancelled_day_does_not_extend_the_streak(engine):
    today = datetime(2026, 8, 12, 10, tzinfo=timezone.utc)
    engine.start("x", now=today)
    engine.cancel(now=today + timedelta(minutes=3))
    assert engine.streak(now=today) == 0


# ── stats ────────────────────────────────────────────────────────────────────

def test_stats_summarise_the_window(engine):
    engine.start("a", minutes=25, now=T0)
    engine.complete(now=T0 + timedelta(minutes=25))
    engine.start("b", minutes=25, now=T0 + timedelta(minutes=30))
    engine.interrupt()
    engine.cancel(now=T0 + timedelta(minutes=40))
    st = engine.stats(days=7, now=T0 + timedelta(minutes=41))
    assert st["sessions"] == 2
    assert st["completed"] == 1
    assert st["completion_rate"] == 50
    assert st["interruptions"] == 1
    assert st["focus_minutes"] >= 25


# ── spoken helpers ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,expected", [
    (50, "50 minutes"), (1, "1 minute"), (60, "1 hour"), (80, "1 h 20 m")])
def test_human_minutes(m, expected):
    assert human_minutes(m) == expected


# ── tool wiring ──────────────────────────────────────────────────────────────

def test_the_focus_tool_is_registered():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"focus"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "focus")
    for key in ("action", "label", "preset", "minutes"):
        assert key in tool["parameters"]["properties"]


class _Bag:
    def __init__(self, dbpath):
        self.focus = FocusEngine(FocusStore(dbpath))


def _tool(stub, args):
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    return ProductivityDispatchMixin.focus_tool(stub, args)


def test_tool_start_status_done_flow(tmp_path):
    stub = _Bag(tmp_path / "f.db")
    r = _tool(stub, {"action": "start", "label": "revise neuro", "preset": "pomodoro"})
    assert r.ok and "revise neuro" in r.text and "25 minute" in r.text

    r = _tool(stub, {"action": "status"})
    assert "left" in r.text or "break" in r.text

    r = _tool(stub, {"action": "interrupt"})
    assert "distraction" in r.text.lower()

    r = _tool(stub, {"action": "done"})
    assert "streak" in r.text.lower()

    r = _tool(stub, {"action": "stats"})
    assert "block" in r.text.lower() and "streak" in r.text.lower()


def test_tool_start_needs_an_intention(tmp_path):
    stub = _Bag(tmp_path / "f.db")
    r = _tool(stub, {"action": "start"})
    assert not r.ok


# ── the daily brief folds in the learning loop (Mark XXV enhancement) ─────────

def test_catch_up_surfaces_due_cards_and_an_overrun_focus_block(tmp_path):
    from types import SimpleNamespace

    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    from orion_core.study import StudyEngine, StudyStore

    bag = SimpleNamespace()
    # Empty rewind + standing-questions stubs so catch_up reads nothing real.
    bag._rewind = SimpleNamespace(reload=lambda: None, decisions=lambda limit=5: [])
    bag._standing_questions = SimpleNamespace(due=lambda: [], list=lambda: [])
    # A due card and a focus block already past its length.
    bag.study = StudyEngine(StudyStore(tmp_path / "s.db"))
    bag.study.add("What is myelin?", "an insulating sheath")     # never reviewed → due
    bag.focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    bag.focus.start("write up results", minutes=1,
                    now=datetime.now(timezone.utc) - timedelta(minutes=3))

    result = KnowledgeDispatchMixin.catch_up_tool(bag, {})
    assert "flashcard" in result.text
    assert "break" in result.text


# ── the Focus page: a task in small steps, a dial, a gentle stop (30 Oct 2026) ──

def test_a_task_written_as_one_line_becomes_its_steps():
    from orion_core.focus import chunk_task
    assert chunk_task("outline the essay, write the intro then add two sources") == [
        "Outline the essay", "Write the intro", "Add two sources"]
    assert chunk_task("1. read chapter 3\n2. make notes and test myself on them") == [
        "Read chapter 3", "Make notes", "Test myself on them"]
    # "and" inside one thing is not two steps, and nothing is made up.
    assert chunk_task("buy salt and pepper") == ["Buy salt and pepper"]
    assert chunk_task("  ") == []
    assert len(chunk_task(", ".join(f"step {i}" for i in range(20)))) == 8


def test_the_steps_are_still_there_after_a_restart(tmp_path):
    path = tmp_path / "focus.db"
    store = FocusStore(path)
    store.save_plan("Lab report", [("Outline the method", True), ("Write the results", False),
                                   ("  ", False)])
    store.close()
    again = FocusStore(path)
    assert again.load_plan() == ("Lab report", [("Outline the method", True),
                                                ("Write the results", False)])
    again.save_plan("Revise", [])             # a task with no steps yet is kept too
    assert again.load_plan() == ("Revise", [])


def test_the_dial_reads_idle_running_due_and_break(engine):
    from orion_core.gui.focus_deck import clock_text, dial_state, gentle_stop_counts
    assert clock_text(65.4) == "01:05" and clock_text(-3) == "00:00"
    idle = dial_state(None, T0, idle_minutes=25)
    assert idle["mode"] == "idle" and idle["text"] == "25:00"
    session = engine.start("write up", minutes=20, now=T0)
    running = dial_state(session, T0 + timedelta(minutes=5), idle_minutes=25)
    assert running["mode"] == "running" and running["text"] == "15:00"
    assert running["progress"] == pytest.approx(0.25)
    assert dial_state(session, T0 + timedelta(minutes=21), idle_minutes=25)["mode"] == "due"
    on_break = dial_state(None, T0, idle_minutes=25, break_until=T0 + timedelta(minutes=2),
                          break_seconds=300)
    assert on_break["mode"] == "break" and on_break["text"] == "02:00"
    # Stopping past halfway still counts the block; before it, it does not.
    assert gentle_stop_counts(session, T0 + timedelta(minutes=11))
    assert not gentle_stop_counts(session, T0 + timedelta(minutes=4))
    assert not gentle_stop_counts(None, T0)


def test_the_focus_page_starts_a_block_with_its_break_nudge(engine):
    from types import SimpleNamespace

    from PyQt6.QtWidgets import QApplication
    globals().setdefault("_QT_APP", QApplication.instance() or QApplication([]))
    from orion_core.gui.focus_deck import BREAK_TEXT, HALFWAY_TEXT, MODES, FocusDeckView

    added: list[tuple[str, float]] = []
    reminders = SimpleNamespace(
        add=lambda text, minutes: added.append((text, minutes)) or SimpleNamespace(ok=True),
        active=lambda: [], cancel=lambda rid: None)
    page = FocusDeckView(SimpleNamespace(), engine, reminders=reminders)
    page.step_input.setText("outline the method, write the results")
    page.add_steps()
    assert [text for text, _done in page.steps] == ["Outline the method", "Write the results"]
    assert engine.store.load_plan()[1] == [("Outline the method", False),
                                           ("Write the results", False)]
    page.choose(next(m for m in MODES if m.body_double))
    page.start()
    session = engine.active()
    assert session is not None and session.label == "Outline the method"
    assert session.kind == "body double" and session.planned_minutes == 25
    assert added == [(BREAK_TEXT, 25), (HALFWAY_TEXT, 12.5)]
    assert page.dial.state()["mode"] == "running"
    page.stop()                                  # straight away: not counted
    assert engine.active() is None and "Stopped after" in page.status.text()
    page.deleteLater()
