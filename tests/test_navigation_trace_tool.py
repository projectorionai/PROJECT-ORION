"""
Tests for the navigation_trace dispatcher tool (dispatch_desktop.py) — the
voice/text-reachable surface over NavigationTrace (Mark XXI, Track D7).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dispatcher import OrionDispatcher
from orion_core.navigation_trace import NavigationTrace


def _dispatcher(trace=None) -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.navigation_trace = trace
    return d


def test_reports_unavailable_with_no_trace_attached():
    d = _dispatcher(None)
    result = d.navigation_trace_tool({"action": "summary"})
    assert not result.ok


def test_summary_is_the_default_action():
    trace = NavigationTrace()
    trace.record(query="Save", strategy="uia", found=True, verified=True)
    d = _dispatcher(trace)
    result = d.navigation_trace_tool({})
    assert result.ok
    assert "1 navigation attempt" in result.text


def test_report_includes_recent_attempts():
    trace = NavigationTrace()
    trace.record(query="Save", strategy="uia", found=True, verified=True)
    d = _dispatcher(trace)
    result = d.navigation_trace_tool({"action": "report"})
    assert result.ok
    assert "Save" in result.text
    assert "Recent attempts" in result.text


def test_failures_action_lists_only_misses():
    trace = NavigationTrace()
    trace.record(query="Save", strategy="uia", found=True, verified=True)
    trace.record(query="Ghost Button", strategy="ocr", found=False,
                 candidates=["Save", "Save As"])
    d = _dispatcher(trace)
    result = d.navigation_trace_tool({"action": "failures"})
    assert result.ok
    assert "Ghost Button" in result.text
    assert "Save As" in result.text
    assert "\nSave\n" not in result.text and "- 'Save'" not in result.text


def test_failures_action_with_no_failures_says_so():
    trace = NavigationTrace()
    trace.record(query="Save", strategy="uia", found=True, verified=True)
    d = _dispatcher(trace)
    result = d.navigation_trace_tool({"action": "failures"})
    assert result.ok
    assert "no failed" in result.text.lower()


def test_unknown_action_fails_cleanly():
    d = _dispatcher(NavigationTrace())
    result = d.navigation_trace_tool({"action": "explode"})
    assert not result.ok
    assert "summary" in result.text


def test_limit_is_respected_in_report():
    trace = NavigationTrace()
    for i in range(10):
        trace.record(query=f"item-{i}", strategy="uia", found=True, verified=True)
    d = _dispatcher(trace)
    result = d.navigation_trace_tool({"action": "report", "limit": 2})
    assert "item-9" in result.text
    assert "item-0 " not in result.text


def test_tool_is_registered_in_the_handler_table():
    d = _dispatcher()
    assert d.handler_table()["navigation_trace"] == d.navigation_trace_tool
