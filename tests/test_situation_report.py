"""
Situation report (Mark XXIII) — the "catch me up" synthesiser.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.situation_report import (  # noqa: E402
    Priority, Category, build_report,
)


# ── ranking + rendering ──────────────────────────────────────────────────────

def test_empty_report_says_all_quiet():
    r = build_report()
    assert r.is_empty
    assert "quiet" in r.render().lower()
    assert not r.worth_speaking()


def test_alerts_lead_and_criticals_are_urgent():
    r = build_report(
        alerts=[{"text": "disk nearly full", "critical": True}],
        decisions=[{"text": "ship the exe tomorrow"}])
    text = r.render()
    # the alert section renders before the decisions section
    assert text.index("attention") < text.index("decided")
    assert r.top_priority == Priority.URGENT
    assert r.worth_speaking()


def test_watch_deltas_render_with_their_points():
    r = build_report(watch_deltas=[
        {"question": "neural interfaces", "points": ["paper A", "paper B", "paper C"]}])
    text = r.render()
    assert "neural interfaces" in text
    assert "paper A" in text
    # three+ new points is important enough to surface
    assert r.top_priority >= Priority.IMPORTANT


def test_headline_is_the_single_most_important_item():
    r = build_report(
        alerts=[{"text": "memory pressure high", "critical": True}],
        tasks=[{"text": "reply to email"}])
    assert "memory pressure" in r.headline()


def test_sections_are_ordered_alert_watch_decision_task():
    r = build_report(
        tasks=[{"text": "t"}], decisions=[{"text": "d"}],
        watch_deltas=[{"question": "w", "points": ["p"]}],
        alerts=[{"text": "a"}])
    text = r.render()
    assert text.index("attention") < text.index("watching") < \
        text.index("decided") < text.index("Open items")


def test_worth_speaking_gate_respects_the_floor():
    # only routine items → not worth interrupting for
    r = build_report(tasks=[{"text": "tidy up"}])
    assert not r.worth_speaking()
    # an important delta → worth speaking
    r2 = build_report(watch_deltas=[{"question": "q", "points": ["a", "b", "c"]}])
    assert r2.worth_speaking()


def test_plain_strings_are_accepted():
    r = build_report(alerts=["something odd"], decisions=["we chose X"])
    text = r.render()
    assert "something odd" in text and "we chose X" in text


def test_sections_are_capped():
    r = build_report(alerts=[f"a{i}" for i in range(20)], max_per_section=3)
    assert sum(1 for i in r.items if i.category == Category.ALERT) == 3


# ── the tool wiring ──────────────────────────────────────────────────────────

def test_tool_and_schema_present():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"catch_up"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "catch_up")
    assert "action" in tool["parameters"]["properties"]
