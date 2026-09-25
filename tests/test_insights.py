"""
Cross-domain insights (Mark XXVI) — the patterns only ORION can see.

Sixteen databases, and until now nothing looked across them. The risk with an
engine like this is that it becomes a horoscope: confident-sounding nonsense from
five data points. So the tests care most about what it REFUSES — thin evidence,
weak correlations, missing stores — and about the wording never claiming cause.
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
from orion_core.insights import (  # noqa: E402
    MIN_DAYS,
    MIN_R,
    Finding,
    collect_series,
    find_patterns,
    render,
)
from orion_core.study import StudyEngine, StudyStore  # noqa: E402
from orion_core.wellbeing import WellbeingEngine, WellbeingStore  # noqa: E402

NOW = datetime(2026, 8, 20, 12, tzinfo=timezone.utc)


def _day(offset: int) -> datetime:
    return NOW - timedelta(days=offset)


@pytest.fixture()
def stores(tmp_path):
    return (FocusEngine(FocusStore(tmp_path / "f.db")),
            WellbeingEngine(WellbeingStore(tmp_path / "w.db")),
            StudyEngine(StudyStore(tmp_path / "s.db")))


def _seed(focus, wellbeing, plan):
    """plan = [(sleep_hours, focus_minutes), …] one entry per day."""
    for index, (sleep, minutes) in enumerate(plan):
        when = _day(len(plan) - index)
        wellbeing.checkin(sleep_hours=sleep, energy=3, at=when)
        focus.start("work", minutes=minutes, now=when)
        focus.complete(now=when + timedelta(minutes=minutes))


# ── the finding itself ───────────────────────────────────────────────────────

def test_a_positive_finding_reads_naturally():
    sentence = Finding("sleep_hours", "focus_minutes", 0.91, 12).sentence()
    assert "more hours slept" in sentence
    assert "higher" in sentence and "r=+0.91" in sentence


def test_a_negative_finding_says_lower():
    sentence = Finding("stress", "focus_minutes", -0.7, 9).sentence()
    assert "lower" in sentence and "r=-0.70" in sentence


def test_a_possessive_label_is_phrased_properly():
    # "On days with more your energy" would be gibberish.
    assert "more your" not in Finding("energy", "focus_minutes", 0.8, 8).sentence()


def test_strength_bands():
    assert Finding("a", "b", 0.85, 9).strength == "strong"
    assert Finding("a", "b", 0.65, 9).strength == "clear"
    assert Finding("a", "b", 0.5, 9).strength == "modest"


# ── real series from real stores ─────────────────────────────────────────────

def test_series_are_built_from_the_live_stores(stores):
    focus, wellbeing, study = stores
    _seed(focus, wellbeing, [(7.0, 50), (6.0, 30)])
    series = collect_series(focus=focus, wellbeing=wellbeing, days=60, now=NOW)
    assert "focus_minutes" in series and "sleep_hours" in series
    assert len(series["focus_minutes"]) == 2


def test_a_planted_relationship_is_found(stores):
    focus, wellbeing, study = stores
    _seed(focus, wellbeing, [(5.0, 15), (8.0, 70), (6.0, 30), (7.5, 60),
                             (5.5, 20), (8.5, 80), (6.5, 35), (7.0, 55)])
    series = collect_series(focus=focus, wellbeing=wellbeing, days=60, now=NOW)
    findings = find_patterns(series)
    match = next((f for f in findings
                  if {f.left, f.right} == {"sleep_hours", "focus_minutes"}), None)
    assert match is not None, "the planted sleep/focus relationship was missed"
    assert match.coefficient > 0.8


def test_study_accuracy_comes_off_the_review_log(stores):
    focus, wellbeing, study = stores
    card = study.add("q", "a")
    for offset, quality in ((3, 5), (3, 5), (2, 1), (2, 1)):
        study.store.log_review(card.id, quality, at=_day(offset))
    series = collect_series(study=study, days=60, now=NOW)
    accuracy = series["study_accuracy"]
    assert len(accuracy) == 2
    assert max(accuracy.values()) == 100.0 and min(accuracy.values()) == 0.0


# ── what it REFUSES (the part that stops it being a horoscope) ──────────────

def test_too_few_days_is_refused(stores):
    focus, wellbeing, _study = stores
    _seed(focus, wellbeing, [(5.0, 10), (8.0, 80)])          # 2 days
    series = collect_series(focus=focus, wellbeing=wellbeing, days=60, now=NOW)
    assert find_patterns(series) == [], "a 2-day 'pattern' must never be reported"


def test_a_weak_correlation_is_refused():
    # plenty of days, no relationship
    series = {
        "sleep_hours": {NOW.date() - timedelta(days=i): float(i % 3) for i in range(12)},
        "focus_minutes": {NOW.date() - timedelta(days=i): float((i * 7) % 5) for i in range(12)},
    }
    for finding in find_patterns(series):
        assert abs(finding.coefficient) >= MIN_R


def test_the_evidence_bar_is_configurable_and_enforced():
    days = [NOW.date() - timedelta(days=i) for i in range(MIN_DAYS - 1)]
    series = {"sleep_hours": {d: float(i) for i, d in enumerate(days)},
              "focus_minutes": {d: float(i) * 10 for i, d in enumerate(days)}}
    assert find_patterns(series) == []          # perfect correlation, too few days


def test_missing_stores_cost_one_series_not_the_report(stores):
    focus, _wellbeing, _study = stores
    _seed(focus, _wellbeing, [(7.0, 40)])
    series = collect_series(focus=focus, wellbeing=None, study=None, finance=None,
                            days=60, now=NOW)
    assert "focus_minutes" in series and "sleep_hours" not in series


def test_a_broken_store_does_not_take_the_report_down():
    class _Broken:
        @property
        def store(self):
            raise RuntimeError("database is gone")
    series = collect_series(focus=_Broken(), days=60, now=NOW)
    assert series == {}


def test_no_data_at_all_is_handled():
    assert collect_series(days=60, now=NOW) == {}
    assert find_patterns({}) == []


# ── the report ───────────────────────────────────────────────────────────────

def test_the_report_admits_when_there_is_nothing_to_say():
    text = render([], {"focus_minutes": {}})
    assert "no pattern is strong enough" in text.lower()
    assert "keep logging" in text.lower()


def test_the_report_never_claims_causation():
    findings = [Finding("sleep_hours", "focus_minutes", 0.9, 10)]
    text = render(findings, {"sleep_hours": {}, "focus_minutes": {}})
    assert "correlations, not proof of cause" in text
    for banned in ("because", "causes", "proves"):
        assert banned not in text.lower().split("not proof")[0]


def test_the_report_ranks_the_strongest_first():
    findings = find_patterns({
        "sleep_hours": {NOW.date() - timedelta(days=i): float(i) for i in range(10)},
        "focus_minutes": {NOW.date() - timedelta(days=i): float(i) * 10 for i in range(10)},
        "energy": {NOW.date() - timedelta(days=i): float(i % 4) for i in range(10)},
    })
    coefficients = [abs(f.coefficient) for f in findings]
    assert coefficients == sorted(coefficients, reverse=True)


# ── tool wiring ──────────────────────────────────────────────────────────────

def test_the_wellbeing_tool_exposes_patterns():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "wellbeing")
    assert "patterns" in tool["description"]
    src = (Path(__file__).resolve().parents[1] / "orion_core"
           / "dispatch_productivity.py").read_text(encoding="utf-8")
    assert "from .insights import collect_series, find_patterns, render" in src


def test_the_tool_runs_end_to_end(tmp_path):
    from types import SimpleNamespace
    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    wellbeing = WellbeingEngine(WellbeingStore(tmp_path / "w.db"))
    _seed(focus, wellbeing, [(5.0, 15), (8.0, 70), (6.0, 30), (7.5, 60),
                             (5.5, 20), (8.5, 80), (6.5, 35)])
    stub = SimpleNamespace(wellbeing=wellbeing, focus=focus, study=None, finance=None)
    result = ProductivityDispatchMixin.wellbeing_tool(stub, {"action": "patterns"})
    assert result.ok
    assert "pattern" in result.text.lower()


# ── daily rhythm: WHEN does the work actually land? ─────────────────────────

from orion_core.insights import (  # noqa: E402
    MIN_SESSIONS,
    DaypartProfile,
    daily_rhythm,
    render_rhythm,
    _part_of_day,
)


def _block(focus, when, minutes, completed):
    focus.start("work", minutes=minutes, now=when)
    if completed:
        focus.complete(now=when + timedelta(minutes=minutes))
    else:
        focus.interrupt()
        focus.cancel(now=when + timedelta(minutes=max(1, minutes // 4)))


@pytest.mark.parametrize("hour,part", [
    (6, "early morning"), (10, "morning"), (14, "afternoon"),
    (19, "evening"), (22, "night"), (2, "night"),
])
def test_hours_map_to_parts_of_day(hour, part):
    assert _part_of_day(hour) == part


def test_rhythm_separates_a_good_slot_from_a_bad_one(tmp_path):
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)
    for i in range(6):
        day = base - timedelta(days=i)
        _block(focus, day.replace(hour=9), 50, True)
        _block(focus, day.replace(hour=22), 50, False)
    profiles = {p.part: p for p in daily_rhythm(focus, days=90,
                                                now=base + timedelta(hours=1))}
    assert profiles["morning"].completion_rate == 100
    assert profiles["night"].completion_rate == 0
    assert profiles["night"].interruptions == 6


def test_the_rhythm_report_names_the_better_slot(tmp_path):
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)
    for i in range(5):
        day = base - timedelta(days=i)
        _block(focus, day.replace(hour=9), 50, True)
        _block(focus, day.replace(hour=22), 50, False)
    text = render_rhythm(daily_rhythm(focus, days=90, now=base + timedelta(hours=1)))
    assert "morning" in text and "hard work" in text


def test_thin_evidence_refuses_to_name_a_pattern(tmp_path):
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)
    _block(focus, base.replace(hour=9), 50, True)          # a single block
    text = render_rhythm(daily_rhythm(focus, days=90, now=base + timedelta(hours=1)))
    assert "Too few blocks" in text
    assert str(MIN_SESSIONS) in text


def test_no_history_says_so(tmp_path):
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    assert "No focus history" in render_rhythm(daily_rhythm(focus))


def test_an_even_performer_is_not_given_a_false_verdict(tmp_path):
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)
    for i in range(4):
        day = base - timedelta(days=i)
        _block(focus, day.replace(hour=9), 50, True)
        _block(focus, day.replace(hour=14), 50, True)
    text = render_rhythm(daily_rhythm(focus, days=90, now=base + timedelta(hours=1)))
    assert "fairly evenly" in text


def test_a_broken_focus_store_yields_no_rhythm():
    class _Broken:
        @property
        def store(self):
            raise RuntimeError("gone")
    assert daily_rhythm(_Broken()) == []


def test_profile_maths():
    profile = DaypartProfile("morning", sessions=4, completed=3,
                             minutes=200.0, interruptions=2)
    assert profile.completion_rate == 75 and profile.average_minutes == 50


def test_the_focus_tool_exposes_the_rhythm(tmp_path):
    from types import SimpleNamespace
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "focus")
    assert "rhythm" in tool["description"]

    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)
    for i in range(4):
        _block(focus, (base - timedelta(days=i)).replace(hour=9), 50, True)
    stub = SimpleNamespace(focus=focus)
    result = ProductivityDispatchMixin.focus_tool(stub, {"action": "rhythm"})
    assert result.ok and "morning" in result.text
