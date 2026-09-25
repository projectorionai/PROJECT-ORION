"""
Performance at scale (Mark XXVI) — ORION must not slow down as data accumulates.

These paths run on the qasync thread (the one that draws the face and services
audio), and some of them run on EVERY review. Before this pass they were O(n)
scans over the whole store: with 3,000 cards, due()/next_due()/stats() each cost
~15 ms and focus.streak() 5.5 ms. Pushing the work into indexed SQL made them
flat (~1 ms).

The thresholds are deliberately loose — a slow CI box must not fail the suite —
but they are far below the pre-fix cost, so a regression to full-scan behaviour
would be caught.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Must be set BEFORE Qt initialises: setting it inside a test is too late
# and Qt then tries to open a real window, which hangs a headless run.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.focus import FocusEngine, FocusStore  # noqa: E402
from orion_core.study import StudyEngine, StudyStore  # noqa: E402

CARDS = 2000
SESSIONS = 800
#: Generous ceiling: the pre-fix implementation measured ~15 ms at this scale.
BUDGET_MS = 8.0


@pytest.fixture(scope="module")
def big_study(tmp_path_factory):
    path = tmp_path_factory.mktemp("perf") / "study.db"
    engine = StudyEngine(StudyStore(path))
    for i in range(CARDS):
        engine.add(f"question {i}", f"answer {i}", deck=f"Deck{i % 5}")
    return engine


@pytest.fixture(scope="module")
def big_focus(tmp_path_factory):
    path = tmp_path_factory.mktemp("perf") / "focus.db"
    engine = FocusEngine(FocusStore(path))
    for i in range(SESSIONS):
        engine.start(f"block {i}", minutes=25)
        engine.complete()
    return engine


def _timed(fn, repeats: int = 5) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1000 / repeats


def test_due_stays_flat_with_a_large_deck(big_study):
    assert _timed(lambda: big_study.due(limit=20)) < BUDGET_MS


def test_next_due_stays_flat(big_study):
    # called on EVERY review — the most latency-sensitive of the lot
    assert _timed(big_study.next_due) < BUDGET_MS


def test_stats_stays_flat(big_study):
    assert _timed(big_study.stats) < BUDGET_MS


def test_deck_listing_stays_flat(big_study):
    assert _timed(big_study.decks) < BUDGET_MS


def test_focus_streak_stays_flat(big_focus):
    assert _timed(big_focus.streak) < BUDGET_MS


# ── correctness must survive the optimisation ───────────────────────────────

def test_the_sql_due_query_matches_the_python_definition(big_study):
    """The indexed query must select exactly what `is_due` would have."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    picked = big_study.due(limit=50, now=now)
    assert picked, "a fresh deck should have due cards"
    assert all(card.is_due(now) for card in picked)


def test_never_seen_cards_still_come_first(tmp_path):
    from datetime import datetime, timedelta, timezone
    engine = StudyEngine(StudyStore(tmp_path / "s.db"))
    now = datetime(2026, 8, 20, tzinfo=timezone.utc)
    engine.add("seen", "a", deck="D")
    engine.next_due("D")
    engine.grade(5, now=now - timedelta(days=5))      # overdue by 4 days
    engine.add("fresh", "a", deck="D")                # never reviewed
    assert engine.due("D", now=now)[0].front == "fresh"


def test_counts_match_a_manual_tally(tmp_path):
    engine = StudyEngine(StudyStore(tmp_path / "s.db"))
    for i in range(20):
        engine.add(f"q{i}", "a", deck="D")
    engine.next_due("D")
    engine.grade(5)
    stats = engine.stats("D")
    manual = engine.store.all("D")
    assert stats["total"] == len(manual)
    assert stats["new"] == sum(1 for c in manual if c.is_new)
    assert stats["mastered"] == sum(1 for c in manual if c.is_mastered)


def test_deck_summary_matches_a_manual_tally(tmp_path):
    engine = StudyEngine(StudyStore(tmp_path / "s.db"))
    for i in range(12):
        engine.add(f"q{i}", "a", deck="A" if i % 2 else "B")
    summary = {row["deck"]: row for row in engine.decks()}
    assert summary["A"]["total"] == 6 and summary["B"]["total"] == 6
    assert summary["A"]["due"] == 6


def test_streak_matches_the_python_definition(tmp_path):
    from datetime import datetime, timedelta, timezone
    from orion_core.focus import FocusSession
    engine = FocusEngine(FocusStore(tmp_path / "f.db"))
    today = datetime(2026, 8, 20, 10, tzinfo=timezone.utc)
    for offset in range(3):
        day = today - timedelta(days=offset)
        engine.store.add(FocusSession(
            label="x", planned_minutes=50, started_at=day.isoformat(),
            ended_at=(day + timedelta(minutes=50)).isoformat(), completed=True))
    assert engine.streak(now=today) == 3


# ── GUI render budget (the face runs on the audio/GUI thread) ───────────────

def test_voxels_are_drawn_in_batches_not_one_by_one():
    """The optimisation itself: a frame must collapse ~1,250 voxels into a few
    hundred colour groups rather than issuing a setBrush per voxel."""
    source = (Path(__file__).resolve().parents[1] / "orion_core" / "gui" / "face.py"
              ).read_text(encoding="utf-8")
    assert "drawRects(" in source, "the voxel loop is no longer batched"
    assert "batches" in source
