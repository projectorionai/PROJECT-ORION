"""
Rewind (CAP-04) — the scrubbable day, read from the verbatim transcripts.

    "What did we decide about the exe yesterday?"
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.rewind import RewindTimeline, Turn  # noqa: E402


def _write(dir_: Path, name: str, rows: list[tuple[str, str, str]]) -> None:
    lines = [json.dumps({"at": at, "role": role, "content": content})
             for at, role, content in rows]
    (dir_ / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def convo(tmp_path):
    _write(tmp_path, "2026-08-09_a.jsonl", [
        ("2026-08-09T10:00:00+00:00", "user", "Should we ship the exe today?"),
        ("2026-08-09T10:00:20+00:00", "assistant", "Let's ship the exe tomorrow instead, once tests pass."),
        ("2026-08-09T10:05:00+00:00", "user", "Fine. What about the chess board?"),
    ])
    _write(tmp_path, "2026-08-10_b.jsonl", [
        ("2026-08-10T09:00:00+00:00", "user", "Morning. Any news on Mars?"),
        ("2026-08-10T09:00:15+00:00", "assistant", "We'll cover Mars in the briefing."),
    ])
    return tmp_path


@pytest.fixture()
def rw(convo):
    return RewindTimeline(convo)


# ── loading + timeline ───────────────────────────────────────────────────────

def test_loads_all_transcripts_in_time_order(rw):
    turns = rw.timeline(limit=100)
    assert len(turns) == 5
    assert turns[0].content.startswith("Should we ship")
    assert turns[-1].content.startswith("We'll cover Mars")


def test_timeline_filters_to_a_day(rw):
    day = rw.timeline("2026-08-09")
    assert len(day) == 3
    assert all(t.at.date().isoformat() == "2026-08-09" for t in day)


def test_days_lists_active_dates(rw):
    days = [d.isoformat() for d in rw.days()]
    assert days == ["2026-08-09", "2026-08-10"]


def test_yesterday_and_today_resolve(convo):
    # a transcript stamped for *today* so the alias has something to find
    now = datetime.now(timezone.utc)
    _write(convo, "today.jsonl", [
        (now.isoformat(), "user", "a fresh message logged right now"),
    ])
    rw = RewindTimeline(convo)
    today = rw.timeline("today")
    assert any("fresh message" in t.content for t in today)


# ── search ───────────────────────────────────────────────────────────────────

def test_search_finds_the_moment(rw):
    hits = rw.search("chess board")
    assert hits
    assert "chess board" in hits[0].content.lower()
    assert hits[0].clock.startswith("2026-08-09")


def test_search_requires_all_terms(rw):
    assert rw.search("mars briefing")          # both appear across the Mars turn
    assert rw.search("mars unicorn") == []     # 'unicorn' never appears


def test_search_empty_query_is_empty(rw):
    assert rw.search("") == []


# ── decisions ────────────────────────────────────────────────────────────────

def test_decisions_finds_settled_turns_only(rw):
    decisions = rw.decisions()
    texts = " ".join(d.content for d in decisions)
    assert "Let's ship the exe" in texts
    assert "We'll cover Mars" in texts
    # a plain question is not a decision
    assert not any("Should we ship the exe today?" == d.content for d in decisions)


def test_decisions_can_be_filtered_by_topic(rw):
    exe = rw.decisions("exe")
    assert exe and all("exe" in d.content.lower() for d in exe)
    assert not any("Mars" in d.content for d in exe)


# ── recap ────────────────────────────────────────────────────────────────────

async def test_extractive_recap_counts_and_lists_decisions(rw):
    text = await rw.recap("2026-08-09")
    assert "turns" in text
    assert "exe" in text.lower()


async def test_recap_uses_the_model_when_present(rw):
    async def fake(prompt: str) -> str:
        assert "RECAP:" in prompt
        return "You decided to ship the exe tomorrow and discussed chess."
    text = await rw.recap("2026-08-09", generate=fake)
    assert "ship the exe" in text


async def test_recap_of_an_empty_day_says_so(rw):
    text = await rw.recap("2020-01-01")
    assert "No conversation" in text


# ── around ───────────────────────────────────────────────────────────────────

def test_around_returns_a_window_near_a_time(rw):
    near = rw.around("2026-08-09T10:00:10+00:00", window=2)
    assert near
    assert any("exe" in t.content.lower() for t in near)


# ── the tool wiring ──────────────────────────────────────────────────────────

def test_tool_and_schema_present():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"rewind"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "rewind")
    props = tool["parameters"]["properties"]
    assert "action" in props and "query" in props


# ── long-term semantic recall (Mark XXIII) ───────────────────────────────────

def test_relevant_ranks_by_relevance_not_strict_and_match(rw):
    # a query where no single turn has ALL the words, but one is clearly best
    hits = rw.relevant("decide about shipping the exe")
    assert hits
    assert "exe" in hits[0].content.lower()          # the exe turn ranks top


def test_relevant_returns_partial_matches_search_would_miss(rw):
    # strict search needs every term; relevant does not
    assert rw.search("exe unicorn dragon") == []      # 'unicorn' never appears
    assert rw.relevant("exe unicorn dragon")          # still finds the exe turns


def test_relevant_rarer_terms_win(rw):
    # 'chess' is rarer than 'the'/'about'; the chess turn should surface
    hits = rw.relevant("what about the chess board")
    assert hits and "chess" in hits[0].content.lower()


def test_relevant_empty_query_is_empty(rw):
    assert rw.relevant("") == []
    assert rw.relevant("the a an of") == []           # all stopwords


def test_rewind_tool_has_a_recall_action():
    import inspect
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    src = inspect.getsource(KnowledgeDispatchMixin.rewind_tool)
    assert "recall" in src and "timeline.relevant" in src
