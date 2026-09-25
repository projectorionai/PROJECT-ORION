"""Tests for StartupBudget (Mark X.12 §5.1)."""

from __future__ import annotations

from orion_core.startup_budget import StartupBudget


def _fixed_clock(ticks):
    it = iter(ticks)
    return lambda: next(it)


def test_phase_deltas_and_total_are_computed_from_marks():
    # ticks: t0=0.00 (construction), then three marks.
    budget = StartupBudget(
        target_seconds=1.0, slow_phase_seconds=0.2,
        clock=_fixed_clock([0.00, 0.05, 0.35, 0.40]),
    )
    assert round(budget.mark("memory"), 2) == 0.05
    assert round(budget.mark("windows"), 2) == 0.30
    assert round(budget.mark("services"), 2) == 0.05

    assert [p.name for p in budget.phases] == ["memory", "windows", "services"]
    assert round(budget.total, 2) == 0.40
    assert not budget.over_budget()


def test_slow_phases_are_flagged():
    budget = StartupBudget(
        slow_phase_seconds=0.2,
        clock=_fixed_clock([0.0, 0.05, 0.35, 0.40]),
    )
    budget.mark("fast")     # +0.05
    budget.mark("heavy")    # +0.30  -> slow
    budget.mark("fast2")    # +0.05

    slow = [p.name for p in budget.slow_phases()]
    assert slow == ["heavy"]
    assert "<- slow" in budget.report()
    assert "heavy" in budget.report()


def test_over_budget_is_detected():
    budget = StartupBudget(
        target_seconds=2.0,
        clock=_fixed_clock([0.0, 3.0]),   # a single 3s phase blows a 2s target
    )
    budget.mark("everything")
    assert budget.over_budget()
    assert "OVER" in budget.report()


def test_total_before_any_mark_uses_live_clock():
    budget = StartupBudget(clock=_fixed_clock([0.0, 0.5]))
    # No marks yet: total falls through to (clock() - t0) == 0.5.
    assert round(budget.total, 2) == 0.5


def test_launch_timestamp_includes_time_before_budget_construction():
    budget = StartupBudget(started_at=10.0, clock=_fixed_clock([11.3, 11.5]))
    assert round(budget.mark("launcher + imports + Qt"), 2) == 1.3
    assert round(budget.mark("core window visible"), 2) == 0.2
    assert round(budget.total, 2) == 1.5


def test_breach_summary_names_the_slowest_phases():
    budget = StartupBudget(
        target_seconds=1.0, slow_phase_seconds=0.1,
        clock=_fixed_clock([0.0, 0.05, 0.85, 1.30]),
    )
    budget.mark("fast")     # +0.05, not slow
    budget.mark("heavy")    # +0.80, slow
    budget.mark("medium")   # +0.45, slow

    summary = budget.breach_summary()
    assert "1.30s > 1s target" in summary
    assert "heavy" in summary
    assert "medium" in summary
    # slowest first
    assert summary.index("heavy") < summary.index("medium")


def test_breach_summary_with_no_single_standout_phase():
    # Over budget from many small phases, none of which individually
    # crosses the slow-phase threshold — breach_summary() must not crash
    # and should say so plainly instead of naming a false culprit.
    budget = StartupBudget(
        target_seconds=1.0, slow_phase_seconds=0.5,
        clock=_fixed_clock([0.0, 0.3, 0.6, 0.9, 1.2]),
    )
    for name in ("a", "b", "c", "d"):
        budget.mark(name)
    assert budget.over_budget()
    assert budget.slow_phases() == []
    assert "no single phase stands out" in budget.breach_summary()
