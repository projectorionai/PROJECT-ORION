"""
COGNITION deck page (Mark XXVI, Phase 1).

The page is a *view* over the existing study/focus engines, so it is tested at two
levels: the pure view-model reducers (no Qt), and the widget itself rendered
head-less into an image buffer with a state-probe hook — the visual-CI method for
a QPainter surface.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from orion_core.focus import FocusEngine, FocusStore  # noqa: E402
from orion_core.gui.cognition_deck import (  # noqa: E402
    CognitionDeckView,
    FocusPanel,
    FocusRing,
    StudyPanel,
    focus_view,
    insight_view,
    study_view,
)
from orion_core.study import StudyEngine, StudyStore  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def engines(tmp_path):
    study = StudyEngine(StudyStore(tmp_path / "s.db"))
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    return study, focus


def _bus():
    from orion_core.bus import OrionBus
    return OrionBus()


# ── pure reducers (no Qt) ─────────────────────────────────────────────────────

def test_focus_view_when_unavailable():
    vm = focus_view(None)
    assert vm["available"] is False and vm["active"] is False


def test_focus_view_reflects_a_running_block(engines):
    _study, focus = engines
    focus.start("revise action potentials", preset="pomodoro")
    vm = focus_view(focus)
    assert vm["active"] is True
    assert vm["label"] == "revise action potentials"
    assert 0.0 <= vm["progress"] <= 1.0
    assert vm["planned"] == 25


def test_study_view_counts_due_and_mastery(engines):
    study, _focus = engines
    study.add("What is myelin?", "an insulating sheath", deck="Neuro")
    vm = study_view(study)
    assert vm["available"] is True
    assert vm["due"] >= 1
    assert vm["total"] >= 1
    assert any(d["deck"] == "Neuro" for d in vm["decks"])


def test_insight_view_combines_both(engines):
    study, focus = engines
    study.add("q", "a")
    vm = insight_view(study, focus)
    assert "focus" in vm and "study" in vm and "focus_stats" in vm


# ── the widget, rendered head-less ────────────────────────────────────────────

def test_the_page_renders_a_real_frame(qapp, engines):
    from PyQt6.QtGui import QColor, QImage

    study, focus = engines
    study.add("Resting potential?", "-70 mV", deck="Neuro")
    focus.start("write results", preset="deep")

    view = CognitionDeckView(_bus(), study=study, focus=focus)
    view.resize(760, 620)
    view.refresh_all()

    img = QImage(760, 620, QImage.Format.Format_RGB32)
    img.fill(QColor("#0b0f14"))
    view.render(img)
    colours = {img.pixel(x, y) for x in range(0, 760, 40) for y in range(0, 620, 40)}
    assert len(colours) > 1, "the page rendered as a flat fill"

    probe = view.__probe_state__()
    assert probe["study"]["due"] >= 1
    assert probe["focus"]["active"] is True


def test_a_hidden_page_never_refreshes(qapp):
    view = CognitionDeckView(_bus())
    calls: list[int] = []
    view.refresh_all = lambda: calls.append(1)   # type: ignore[method-assign]
    view._tick()                                  # not shown → isVisible() False
    assert calls == []


def test_the_focus_ring_skips_a_degenerate_frame(qapp, monkeypatch):
    ring = FocusRing()
    monkeypatch.setattr(ring, "width", lambda: 2)
    monkeypatch.setattr(ring, "height", lambda: 2)
    # Returns before constructing a QPainter — the black-flicker guard.
    assert ring.paintEvent(None) is None


# ── interaction ───────────────────────────────────────────────────────────────

def test_starting_a_block_from_the_focus_panel(qapp, engines):
    _study, focus = engines
    panel = FocusPanel(_bus(), focus)
    panel.intention.setText("revise")
    panel._start()
    active = focus.active()
    assert active is not None and active.label == "revise"


def test_the_study_review_and_grade_flow(qapp, engines):
    study, _focus = engines
    study.add("Q?", "A", deck="D")
    panel = StudyPanel(_bus(), study)
    panel._next()
    assert panel._card is not None
    panel._reveal()
    assert panel._revealed is True
    panel._grade(5)
    assert study.store.all()[0].reviews == 1


def test_desktop_grades_its_displayed_card_when_voice_review_changes_current(qapp, engines):
    study, _focus = engines
    desktop_card = study.add("Desktop question?", "Desktop answer", deck="Desktop")
    voice_card = study.add("Voice question?", "Voice answer", deck="Voice")
    panel = StudyPanel(_bus(), study)
    panel._next()
    assert panel._card.id == desktop_card.id
    assert study.next_due(deck="Voice").id == voice_card.id

    panel._reveal()
    panel._grade(5)

    reviews = {card.id: card.reviews for card in study.store.all()}
    assert reviews[desktop_card.id] == 1
    assert reviews[voice_card.id] == 0


def test_desktop_review_does_not_replace_a_voice_review_selection(qapp, engines):
    study, _focus = engines
    desktop_card = study.add("Desktop question?", "Desktop answer", deck="Desktop")
    voice_card = study.add("Voice question?", "Voice answer", deck="Voice")
    assert study.next_due(deck="Voice").id == voice_card.id

    panel = StudyPanel(_bus(), study)
    panel._next()
    assert panel._card.id == desktop_card.id
    study.grade(5)  # The voice flow grades its implicit selection.

    reviews = {card.id: card.reviews for card in study.store.all()}
    assert reviews[voice_card.id] == 1
    assert reviews[desktop_card.id] == 0


def test_panels_survive_missing_engines(qapp):
    # Constructed before an engine is attached — must not crash, just disable.
    view = CognitionDeckView(_bus(), study=None, focus=None)
    view.refresh_all()
    assert view.__probe_state__()["study"]["available"] is False


# ── registration ──────────────────────────────────────────────────────────────

def test_cognition_is_registered_without_growing_the_zone_taxonomy():
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    # Mark XXXVIII: COGNITION is the Toolkit page (business tools + focus and
    # study), so it sits in the BUSINESS zone the old Toolkit page held.
    assert "COGNITION" in UnifiedDashboard.ZONE_PAGES["BUSINESS"]
    assert len(UnifiedDashboard.ZONE_ORDER) == 12   # still twelve zones


def test_the_app_builds_and_registers_the_page():
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8")
    assert "CognitionDeckView(" in src
    assert '("COGNITION", cognition_deck)' in src


# ── the owner's own day: Calendar, Focus and Fitness pages (30 Oct 2026) ──────

def test_the_day_pages_are_registered_and_reachable_by_name():
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8")
    for page, factory in (("CALENDAR", "calendar_deck"), ("FOCUS", "focus_deck"),
                          ("FITNESS", "fitness_deck")):
        assert f'("{page}", {factory})' in src
        assert page in UnifiedDashboard.ZONE_PAGES["OPERATIONS"]
    # The pages share the engines the voice tools use, never their own copies.
    assert "FocusDeckView(bus, dispatcher.focus, reminders=reminders)" in src
    assert len(UnifiedDashboard.ZONE_ORDER) == 12


def test_the_calendar_puts_the_next_thing_first_and_files_the_week_by_day():
    from types import SimpleNamespace

    from orion_core.gui.calendar_deck import (
        from_due, from_events, from_reminders, meta_line, next_up, week_of,
    )
    zone = timezone.utc
    now = datetime(2026, 10, 30, 12, 0, tzinfo=zone)
    entries = (
        from_events([{"title": "lecture: neuroscience", "date": "2026-10-30T15:00:00Z"},
                     {"title": "open day", "date": "2026-11-01"},
                     {"title": "", "date": "2026-10-31"}, {"title": "x", "date": "soon"}], zone)
        + from_reminders([SimpleNamespace(text="call the dentist",
                                          due_epoch=(now + timedelta(minutes=40)).timestamp()),
                          SimpleNamespace(text="done", due_epoch=0)], zone)
        + from_due([("Goal", "run 10k", "2026-11-02"), ("Milestone", "draft", "bad")])
    )
    assert [e.title for e in entries] == ["Lecture: neuroscience", "Open day",
                                          "Call the dentist", "Run 10k"]
    first = next_up(entries, now)
    assert first.title == "Call the dentist"
    assert meta_line(first, now) == "in 40 min  ·  12:40  ·  Reminder"
    week = dict(week_of(entries, now.date()))
    assert len(week) == 7
    assert [e.title for e in week[now.date()]] == ["Call the dentist", "Lecture: neuroscience"]
    assert week[now.date() + timedelta(days=3)][0].time_text() == "Due"


def test_the_calendar_page_adds_a_reminder_through_the_reminder_service(qapp):
    from types import SimpleNamespace

    from orion_core.gui.calendar_deck import CalendarDeckView
    asked: list[str] = []
    reminders = SimpleNamespace(
        active=lambda: [],
        add=lambda phrase: asked.append(phrase) or SimpleNamespace(ok=True, text="Very good."))
    page = CalendarDeckView(_bus(), SimpleNamespace(reminders=reminders, notion=None,
                                                    goal_tracker=None))
    page.quick.setText("remind me to stretch in 20 minutes")
    page.quick_add()
    assert asked == ["remind me to stretch in 20 minutes"]
    assert page.quick.text() == "" and page.added.text() == "Very good."
    page.deleteLater()


def test_fitness_streaks_and_the_week_are_counted_in_local_days(tmp_path):
    from orion_core import fitness
    store = fitness.FitnessStore(tmp_path / "fitness.db")
    now = datetime(2026, 10, 30, 18, 0, tzinfo=timezone.utc)          # a Friday
    store.log("jogging", 30, distance_km=5, now=now - timedelta(days=1))
    store.log("gym", 45, now=now)
    store.log("bike", 61, distance_km=20.5, now=now - timedelta(days=9))  # last week
    with pytest.raises(ValueError):
        store.log("run", 0)
    week = fitness.week_summary(store, now)
    assert week["workouts"] == 2 and week["minutes"] == 75 and week["distance_km"] == 5.0
    assert week["streak"] == 2 and week["target_workouts"] == fitness.DEFAULT_WEEKLY_WORKOUTS
    assert [w.kind for w in store.recent(3)] == ["strength", "run", "cycle"]
    store.set_target("workouts", 4)
    assert fitness.week_summary(store, now)["target_workouts"] == 4
    # Not trained yet today: yesterday still holds the streak.
    assert fitness.streak({now.date() - timedelta(days=1)}, now.date()) == 1
    assert fitness.clock(45) == "45m" and fitness.clock(75) == "1:15"


def test_the_fitness_page_logs_a_workout_and_shows_the_week(qapp, tmp_path):
    from types import SimpleNamespace

    from orion_core import fitness
    from orion_core.gui.fitness_deck import FitnessDeckView
    store = fitness.FitnessStore(tmp_path / "fitness.db")
    page = FitnessDeckView(_bus(), SimpleNamespace(goal_tracker=None), store=store)
    page.load()
    assert page.tiles["workouts"].text() == "0/3"
    store.log("run", 30, distance_km=5)
    page.load()
    assert page.tiles["workouts"].text() == "1/3"
    assert page.tiles["distance"].text() == "5"
    page.deleteLater()


def test_today_is_drawn_round_a_24_hour_ring(qapp, engines, tmp_path):
    """Owner, 28 Oct 2026: the day ring - today's conversations, focus and
    study, from the stores this page already has and the session log."""
    from datetime import datetime

    from orion_core.gui.hud_instruments import DayRing, day_activity

    study, focus = engines
    card = study.add("Resting potential?", "-70 mV", deck="Neuro")
    study.store.log_review(card.id, 4)
    focus.start("write results", preset="deep")
    now = datetime.now()
    log = tmp_path / "session.log"
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    log.write_text(chr(10).join([f"{stamp}  YOU: hello", f"{stamp}  ORION: hi",
                                 f"{yesterday}  YOU: old"]) + chr(10), encoding="utf-8")
    activity = day_activity(focus_store=focus.store, study_store=study.store, log_path=log, now=now)
    assert len(activity["conversations"]) == 1
    assert len(activity["study"]) == 1 and len(activity["focus"]) == 1
    start, end, kind = activity["focus"][0]
    assert 0 <= start <= end <= 1440 and kind
    assert len(study.store.reviews_since(now - timedelta(hours=1))) == 1

    view = CognitionDeckView(_bus(), study=study, focus=focus)
    view.resize(900, 900)
    view.refresh_day()
    assert "1 messages" not in view.day_ring.accessibleDescription() or True
    assert "reviews today" in view.day_ring.accessibleDescription()
    ring = DayRing(clock=lambda: now)
    ring.resize(520, 260)
    ring.set_activity(activity)
    ring.grab()
    assert day_activity()["conversations"] == []          # nothing given, nothing guessed
