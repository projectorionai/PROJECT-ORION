"""
Standing questions (CAP-05) — cadence, due list, and the delta.

    "Register the things I care about, and surface what's new."
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.standing_questions import (  # noqa: E402
    StandingQuestions, resolve_cadence,
)


@pytest.fixture()
def store(tmp_path):
    s = StandingQuestions(path=tmp_path / "sq.db")
    yield s
    s.close()


# ── cadence ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,hours", [
    ("daily", 24.0), ("weekly", 168.0), ("hourly", 1.0),
    ("6", 6.0), (12, 12.0), (None, 24.0), ("nonsense", 24.0),
])
def test_cadence_resolves(value, hours):
    assert resolve_cadence(value) == hours


# ── management ───────────────────────────────────────────────────────────────

def test_add_and_list(store):
    q = store.add("What's new in neural interfaces?", "daily")
    assert q.id is not None
    assert q.cadence_hours == 24.0
    assert [x.text for x in store.list()] == ["What's new in neural interfaces?"]


def test_add_is_idempotent_on_text(store):
    store.add("AI chip supply", "daily")
    store.add("AI chip supply", "weekly")       # same text updates cadence
    items = store.list()
    assert len(items) == 1
    assert items[0].cadence_hours == 168.0


def test_remove_deactivates(store):
    q = store.add("crypto regulation")
    assert store.remove(q.id)
    assert store.list() == []
    assert len(store.list(include_inactive=True)) == 1


def test_empty_text_is_rejected(store):
    with pytest.raises(ValueError):
        store.add("   ")


# ── due ──────────────────────────────────────────────────────────────────────

def test_a_new_question_is_due_immediately(store):
    store.add("fusion energy milestones")
    assert len(store.due()) == 1


def test_a_recently_checked_question_is_not_due(store):
    now = datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)
    q = store.add("robotics news", "daily")
    store.record_check(q.id, "Boston Dynamics did a thing.", now=now)
    assert store.due(now=now + timedelta(hours=1)) == []
    assert len(store.due(now=now + timedelta(hours=25))) == 1


# ── delta ────────────────────────────────────────────────────────────────────

def test_first_check_is_all_new(store):
    q = store.add("space launches")
    delta = store.record_check(q.id, "Starship flew.\nA new lunar lander was announced.")
    assert delta.has_news
    assert len(delta.new_points) == 2


def test_second_check_reports_only_what_changed(store):
    q = store.add("space launches")
    store.record_check(q.id, "Starship flew.\nA new lunar lander was announced.")
    delta = store.record_check(
        q.id, "Starship flew.\nA new lunar lander was announced.\n"
              "SpaceX set a reuse record.")
    assert len(delta.new_points) == 1
    assert "reuse record" in delta.new_points[0]


def test_reworded_repeats_are_not_counted_as_new(store):
    q = store.add("markets")
    store.record_check(q.id, "The central bank raised interest rates today.")
    delta = store.record_check(q.id, "The central bank raised interest rates today!")
    assert not delta.has_news       # fuzzy match treats it as the same point


def test_describe_no_news(store):
    q = store.add("weather patterns")
    store.record_check(q.id, "It rained a lot in the north this week overall.")
    delta = store.record_check(q.id, "It rained a lot in the north this week overall.")
    assert "nothing new" in delta.describe()


# ── run_due with an injected researcher ──────────────────────────────────────

async def test_run_due_researches_and_returns_deltas(store):
    store.add("AI safety papers", "daily")

    async def researcher(question: str) -> str:
        return f"New paper on {question}. A second finding about alignment."

    deltas = await store.run_due(researcher)
    assert len(deltas) == 1
    assert deltas[0].has_news
    # after running, the question is no longer due
    assert store.due() == []


async def test_run_due_survives_a_failing_researcher(store):
    store.add("quantum computing")

    async def broken(question: str) -> str:
        raise RuntimeError("research backend down")

    deltas = await store.run_due(broken)
    assert deltas == []
    # a failed run leaves it still due to try again
    assert len(store.due()) == 1


# ── the tool wiring ──────────────────────────────────────────────────────────

def test_tool_and_schema_present():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"standing_questions"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "standing_questions")
    props = tool["parameters"]["properties"]
    assert "action" in props and "question" in props
