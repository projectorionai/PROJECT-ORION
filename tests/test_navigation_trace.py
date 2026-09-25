"""
Tests for navigation_trace.py (Mark XXI, Track D7) — the rolling log of
UI-targeting attempts built specifically so the navigation-accuracy fixes
that follow (fuzzy matching, OCR fallback, bidirectional scroll, informative
failures) can be measured against real data instead of a feeling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.navigation_trace import NavigationTrace


class _Signal:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def emit(self, *a) -> None:
        self.calls.append(a)


class _StubBus:
    def __init__(self) -> None:
        self.dashboard_event = _Signal()


# ── record() ─────────────────────────────────────────────────────────────────

def test_record_returns_the_attempt():
    trace = NavigationTrace()
    attempt = trace.record(query="Save", strategy="uia", found=True)
    assert attempt.query == "Save"
    assert attempt.found is True


def test_record_appends_to_the_log():
    trace = NavigationTrace()
    trace.record(query="Save", strategy="uia", found=True)
    trace.record(query="Cancel", strategy="uia", found=False)
    assert len(trace.recent(10)) == 2


def test_record_emits_on_the_bus_when_attached():
    bus = _StubBus()
    trace = NavigationTrace(bus=bus)
    trace.record(query="Save", strategy="uia", found=True)
    assert len(bus.dashboard_event.calls) == 1
    assert bus.dashboard_event.calls[0][0] == "navigation"


def test_record_never_raises_when_bus_emit_is_broken():
    class _BrokenBus:
        class dashboard_event:
            @staticmethod
            def emit(*a):
                raise RuntimeError("boom")

    trace = NavigationTrace(bus=_BrokenBus())
    trace.record(query="Save", strategy="uia", found=True)   # must not raise


def test_record_works_with_no_bus_attached():
    trace = NavigationTrace(bus=None)
    trace.record(query="Save", strategy="uia", found=True)   # must not raise


def test_log_is_bounded_by_maxlen():
    trace = NavigationTrace(maxlen=5)
    for i in range(20):
        trace.record(query=f"item-{i}", strategy="uia", found=True)
    assert len(trace.recent(100)) == 5
    assert trace.recent(100)[0]["query"] == "item-15"   # oldest 15 evicted


# ── recent() / failures() ───────────────────────────────────────────────────

def test_recent_returns_the_last_n_in_order():
    trace = NavigationTrace()
    for i in range(5):
        trace.record(query=f"item-{i}", strategy="uia", found=True)
    recent = trace.recent(2)
    assert [r["query"] for r in recent] == ["item-3", "item-4"]


def test_failures_only_returns_not_found_attempts():
    trace = NavigationTrace()
    trace.record(query="Save", strategy="uia", found=True)
    trace.record(query="Ghost Button", strategy="uia", found=False, candidates=["Save", "Save As"])
    trace.record(query="Cancel", strategy="ocr", found=True)
    misses = trace.failures()
    assert len(misses) == 1
    assert misses[0]["query"] == "Ghost Button"
    assert misses[0]["candidates"] == ["Save", "Save As"]


def test_failures_respects_the_limit():
    trace = NavigationTrace()
    for i in range(10):
        trace.record(query=f"miss-{i}", strategy="uia", found=False)
    assert len(trace.failures(limit=3)) == 3


# ── summary() ────────────────────────────────────────────────────────────────

def test_summary_with_no_attempts():
    trace = NavigationTrace()
    assert "no navigation attempts" in trace.summary().lower()


def test_summary_reports_found_and_verified_counts():
    trace = NavigationTrace()
    trace.record(query="Save", strategy="uia", found=True, verified=True)
    trace.record(query="Cancel", strategy="uia", found=True, verified=False)
    trace.record(query="Ghost", strategy="uia", found=False)
    summary = trace.summary()
    assert "3 navigation attempt(s)" in summary
    assert "2 found" in summary
    assert "1 verified" in summary


def test_summary_breaks_down_by_strategy():
    trace = NavigationTrace()
    trace.record(query="Save", strategy="uia", found=True)
    trace.record(query="Weird Icon", strategy="ocr", found=True)
    summary = trace.summary()
    assert "ocr: 1" in summary
    assert "uia: 1" in summary


def test_summary_lists_recent_failures_with_candidates():
    trace = NavigationTrace()
    trace.record(query="Ghost Button", strategy="uia", found=False,
                 candidates=["Save", "Save As", "Save All"])
    summary = trace.summary()
    assert "Ghost Button" in summary
    assert "Save" in summary


# ── report() ─────────────────────────────────────────────────────────────────

def test_report_with_no_attempts():
    trace = NavigationTrace()
    assert "no navigation attempts" in trace.report().lower()


def test_report_includes_a_per_attempt_line():
    trace = NavigationTrace()
    trace.record(query="Save", strategy="uia", found=True, verified=True)
    report = trace.report()
    assert "Save" in report
    assert "found" in report.lower()


def test_report_respects_the_attempt_limit():
    trace = NavigationTrace()
    for i in range(30):
        trace.record(query=f"item-{i}", strategy="uia", found=True)
    report = trace.report(limit=5)
    # Only the most recent 5 should have their own detail line; earlier
    # ones (item-0..item-24) should not appear as an attempt line.
    assert "item-29" in report
    assert "item-0 " not in report and not report.count("item-0'")
