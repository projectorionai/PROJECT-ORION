"""
Wellbeing (Mark XXVI, Phase 3) — check-ins, trends, and correlation with focus.

The correlation is the point (it links wellbeing to the focus data ORION already
holds), so the Pearson maths and its honesty about sample size are tested hardest.
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
from orion_core.wellbeing import (  # noqa: E402
    WellbeingEngine,
    WellbeingStore,
    pearson,
    resolve_mood,
)

NOW = datetime(2026, 8, 20, 12, tzinfo=timezone.utc)


@pytest.fixture()
def engine(tmp_path):
    return WellbeingEngine(WellbeingStore(tmp_path / "wb.db"))


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_pearson_of_a_perfect_line_is_one():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == -1.0


def test_pearson_is_none_without_variance_or_points():
    assert pearson([1, 1, 1], [2, 3, 4]) is None
    assert pearson([1], [2]) is None


def test_mood_labels_and_numbers_resolve():
    assert resolve_mood("great") == 2
    assert resolve_mood("low") == -1
    assert resolve_mood("2") == 2
    assert resolve_mood(-5) == -2          # clamped
    assert resolve_mood(None) is None


# ── check-in + trend ──────────────────────────────────────────────────────────

def test_a_checkin_clamps_and_stores(engine):
    c = engine.checkin(energy=9, mood="good", stress=3, sleep_hours=7.5,
                       factors=["caffeine"])
    assert c.energy == 5                    # clamped to 1..5
    assert c.mood_valence == 1
    assert c.sleep_hours == 7.5
    assert c.factors == ["caffeine"]


def test_trend_reports_averages_and_direction(engine):
    # energy rising over the window
    for i, e in enumerate([2, 2, 3, 4, 5]):
        engine.checkin(energy=e, at=NOW - timedelta(days=5 - i))
    tr = engine.trend(days=14, now=NOW)
    assert tr["checkins"] == 5
    assert tr["avg_energy"] == pytest.approx(3.2, abs=0.1)
    assert tr["energy_direction"] == "up"


def test_today_summarises_only_today(engine):
    engine.checkin(energy=4, at=NOW)
    engine.checkin(energy=2, at=NOW - timedelta(days=2))
    t = engine.today(now=NOW)
    assert t["checkins"] == 1 and t["avg_energy"] == 4.0


# ── correlation with focus ────────────────────────────────────────────────────

def test_correlate_links_energy_to_focus_minutes(tmp_path):
    wb = WellbeingEngine(WellbeingStore(tmp_path / "wb.db"))
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    # Four days: higher energy on days with more focus minutes.
    plan = [(1, 10), (2, 20), (4, 40), (5, 55)]
    for i, (energy, minutes) in enumerate(plan):
        day = NOW - timedelta(days=len(plan) - i)
        wb.checkin(energy=energy, at=day)
        s = focus.start("work", minutes=minutes, now=day)
        focus.complete(now=day + timedelta(minutes=minutes))
    cors = wb.correlate(focus=focus, days=30, now=NOW)
    energy_minutes = next(c for c in cors if c["pair"] == "energy vs focus minutes")
    assert energy_minutes["coefficient"] > 0.8
    assert energy_minutes["n"] == 4


def test_correlate_is_empty_without_overlap(engine):
    engine.checkin(energy=3, at=NOW)
    assert engine.correlate(focus=None, days=30, now=NOW) == []


# ── tool wiring ───────────────────────────────────────────────────────────────

def test_the_wellbeing_tool_is_registered():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"wellbeing"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "wellbeing")
    for key in ("action", "energy", "mood", "sleep_hours"):
        assert key in tool["parameters"]["properties"]


class _Bag:
    def __init__(self, dbpath):
        self.wellbeing = WellbeingEngine(WellbeingStore(dbpath))
        self.focus = None


def _tool(stub, args):
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    return ProductivityDispatchMixin.wellbeing_tool(stub, args)


def test_tool_checkin_and_today(tmp_path):
    stub = _Bag(tmp_path / "wb.db")
    r = _tool(stub, {"action": "checkin", "energy": 4, "mood": "good", "sleep_hours": 7})
    assert r.ok and "energy 4/5" in r.text
    r = _tool(stub, {"action": "today"})
    assert "energy" in r.text.lower()
