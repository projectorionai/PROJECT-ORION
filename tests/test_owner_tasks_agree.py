"""The briefing, the awareness tool and every overdue rule read the same tasks.

The owner's session, 25 Oct 2026 11:52 (Mark 37.2): the briefing read the
Notion board - which is also his calendar - and said "several overdue tasks
require attention" (a finished trip, a house move, a team day, two
untitled rows). "Can you clear the several due tasks?" reached
awareness(clear_overdue), which looked only at ORION's own (empty) list and
answered "Nothing is overdue"; ORION then told him nothing had been overdue.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from orion_core import briefing as briefing_module
from orion_core import owner_tasks
from orion_core.briefing import MorningBriefingService
from orion_core.cognition import CognitiveStateManager
from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
from orion_core.dispatch_web import WebDispatchMixin
from orion_core.notion import NotionService


def _bus():
    return SimpleNamespace(log=SimpleNamespace(emit=Mock()),
                           dashboard_event=SimpleNamespace(emit=Mock()))


def _days(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


def _page(key, title, start, end=None, status=None):
    props = {"Name": {"title": [{"plain_text": title}] if title else []},
             "Date": {"date": {"start": start, "end": end} if start else None}}
    if status is not None:
        props["Status"] = {"status": {"name": status}}
    return {"id": key, "url": f"https://notion.example/{key}", "properties": props}


def _owner_board(with_status: bool):
    """The owner's rows as the briefing read them out (dates relative to today)."""
    status = "Not started" if with_status else None
    return [
        _page("trip", "Away (Lisbon trip)", _days(-55), _days(-41), status),
        _page("move", "House Move and New Job", _days(-35), None, status),
        _page("boys", "Team Day / Planning / Workshop Day", _days(-14), None, status),
        _page("u1", "", _days(-10), None, status),
        _page("u2", "", _days(-9), None, status),
        _page("next", "Dentist", _days(3), None, status),
    ]


def _notion(rows, *, with_status=True, fail=False, writes=None):
    service = NotionService(_bus(), {"token": "synthetic", "tasks_database_id": "db",
                                     "calendar_database_id": "db"})
    properties = {"Name": {"type": "title", "title": {}}, "Date": {"type": "date", "date": {}}}
    if with_status:
        properties["Status"] = {"type": "status", "status": {
            "options": [{"id": "todo", "name": "Not started"}, {"id": "done", "name": "Done"}],
            "groups": [{"name": "Complete", "option_ids": ["done"]}]}}

    async def request(method, path, payload=None):
        if fail:
            raise RuntimeError("Notion API 503: service unavailable")
        if method == "GET" and path == "/databases/db":
            return {"properties": properties}
        if method == "POST" and path == "/databases/db/query":
            return {"results": list(rows), "has_more": False}
        if writes is not None:
            writes.append((method, path, payload))
        raise AssertionError(f"unexpected Notion write {method} {path}")

    service._request = AsyncMock(side_effect=request)
    return service


def _host(tmp_path, notion):
    host = WebDispatchMixin()
    host.bus = _bus()
    host.cognition = CognitiveStateManager(host.bus, Mock(), path=tmp_path / "state.json")
    notion.cognition = host.cognition
    host.notion = notion
    host.cognitive_loop = None
    return host


def _briefing(tmp_path, monkeypatch, host):
    monkeypatch.setattr(briefing_module, "NEWS_CACHE_PATH", tmp_path / "news_cache.db")
    monkeypatch.setattr(briefing_module, "BRIEFING_STATE_PATH", tmp_path / "briefing_state.json")
    service = MorningBriefingService(_bus(), host.notion, SimpleNamespace(available=False))
    service.cognition = host.cognition
    return service


def _awareness(host, args):
    return asyncio.run(KnowledgeDispatchMixin.awareness_tool(host, dict(args)))


# ── the owner's scenario, end to end ─────────────────────────────────────────

@pytest.mark.parametrize("action", ["clear_overdue", "clear_due_tasks"])
def test_tasks_the_briefing_calls_overdue_are_the_ones_clear_overdue_clears(
        tmp_path, monkeypatch, action):
    writes: list = []
    host = _host(tmp_path, _notion(_owner_board(True), writes=writes))
    briefing = _briefing(tmp_path, monkeypatch, host)

    section = asyncio.run(briefing._tasks_section())
    assert "5 overdue" in section
    assert "Away (Lisbon trip)" in section and "(OVERDUE)" in section
    assert "Dentist" in section            # open, not overdue

    result = _awareness(host, {"action": action})
    assert result.ok, result.text
    assert "Nothing is overdue" not in result.text
    assert set(result.receipt["cleared_board"]) == {
        "Away (Lisbon trip)", "House Move and New Job",
        "Team Day / Planning / Workshop Day", "(untitled)"}
    assert writes == []                    # the board itself is left as it is

    # Every reader now agrees: nothing overdue, the future task still open.
    after = asyncio.run(briefing._tasks_section())
    assert "overdue" not in after.lower() and "Dentist" in after
    listing = _awareness(host, {"action": "list_tasks"})
    assert "Dentist" in listing.text and "OVERDUE" not in listing.text
    again = _awareness(host, {"action": "clear_overdue"})
    assert again.ok and again.text.startswith("Nothing is overdue")
    # ...and it survives a restart.
    restored = CognitiveStateManager(_bus(), Mock(), path=host.cognition.path)
    restored._load_locked()
    assert restored.is_dismissed("Away (Lisbon trip)")
    assert not restored.is_dismissed("Dentist")


def test_an_unreadable_board_is_never_reported_as_nothing_overdue(tmp_path):
    host = _host(tmp_path, _notion([], fail=True))
    result = _awareness(host, {"action": "clear_overdue"})
    assert not result.ok
    assert "could not be read" in result.text
    assert "Do not tell the user nothing was overdue" in result.text


def test_list_tasks_and_situation_include_the_board_the_briefing_reads(tmp_path):
    host = _host(tmp_path, _notion(_owner_board(True)))
    host.cognition.add_task("Write report", due=_days(-1))
    listing = _awareness(host, {"action": "list_tasks"})
    assert "Write report" in listing.text and "Away (Lisbon trip)" in listing.text
    assert "(6 overdue)" in listing.text
    host.cognitive_loop = SimpleNamespace(situation_report=lambda: owner_tasks_result())
    report = _awareness(host, {"action": "situation"})
    assert "6 overdue" in report.text and "1 on ORION's list" in report.text


def test_board_tasks_from_before_the_fresh_start_are_never_read_out(tmp_path, monkeypatch):
    """28 Oct 2026: ORION's own tasks were cleared for university, but the
    board's old overdue rows kept opening every briefing. A fresh start hides
    every board task that fell due before it; a later one still counts."""
    start = tmp_path / "fresh_start.json"
    start.write_text('{"since": "%s"}' % _days(-1), encoding="utf-8")
    monkeypatch.setattr(owner_tasks, "FRESH_START_PATH", start)
    assert owner_tasks.fresh_start() == datetime.fromisoformat(_days(-1))
    host = _host(tmp_path, _notion(_owner_board(True)))
    view = asyncio.run(owner_tasks.gather(host.notion, host.cognition))
    assert [row["title"] for row in view.rows] == ["Dentist"]
    assert not any(row["overdue"] for row in view.rows)
    # A range that ends after the fresh start is still current.
    assert not owner_tasks.before_fresh_start(_days(-3), end=_days(2),
                                              since=owner_tasks.fresh_start())
    # No file, or an unreadable one, hides nothing.
    start.write_text("not json", encoding="utf-8")
    assert owner_tasks.fresh_start() is None
    assert len(asyncio.run(owner_tasks.gather(host.notion, host.cognition)).rows) == 6


def owner_tasks_result():
    from orion_core.data import ToolResult
    return ToolResult("Cognitive awareness report:\n- Active project: none")


def test_local_overdue_tasks_are_cleared_too(tmp_path):
    host = _host(tmp_path, _notion([]))
    host.cognition.add_task("Pay invoice", due=_days(-2))
    host.cognition.add_task("Book flights", due=_days(5))
    result = _awareness(host, {"action": "clear_overdue"})
    assert result.ok and result.receipt["cleared_local"] == ["Pay invoice"]
    assert [t["title"] for t in host.cognition.list_tasks()] == ["Book flights"]


# ── a calendar used as the task board ───────────────────────────────────────

def test_past_entries_in_a_calendar_without_a_done_field_are_events_not_overdue_tasks():
    service = _notion(_owner_board(False), with_status=False)
    rows = asyncio.run(service.open_tasks(limit=12))
    assert [r["title"] for r in rows] == ["Dentist"]


def test_a_board_with_a_done_field_keeps_its_past_due_open_tasks():
    service = _notion(_owner_board(True))
    rows = asyncio.run(service.open_tasks(limit=12))
    assert len(rows) == 6 and rows[0]["date_end"] == _days(-41)


# ── one overdue rule everywhere ──────────────────────────────────────────────

def test_a_range_is_due_when_it_ends_and_a_bare_date_at_the_end_of_the_day():
    noon = datetime.combine(date.today(), datetime.min.time()) + timedelta(hours=11)
    assert not owner_tasks.is_overdue(_days(0), now=noon)
    assert owner_tasks.is_overdue(_days(-1), now=noon)
    assert not owner_tasks.is_overdue(_days(-3), end=_days(2), now=noon)
    assert owner_tasks.is_overdue(f"{_days(0)} 09:00", now=noon)
    assert owner_tasks.hours_until("not a date") is None
    aware = datetime.now().astimezone()
    assert owner_tasks.is_overdue((aware - timedelta(hours=1)).isoformat())
    assert not owner_tasks.is_overdue((aware + timedelta(hours=1)).isoformat())


def test_every_task_reader_agrees_a_task_due_today_is_not_overdue():
    from orion_core.briefing_engine import PriorityMonitor
    from orion_core.cognitive_loop import CognitiveLoopManager
    from orion_core.executive_core import PriorityEngine, StrategicEngine
    from orion_core.momentum import MomentumEngine

    state = {"pending_tasks": {"t": {"title": "Ship it", "due": _days(0)}},
             "active_projects": {}, "goals": {}, "open_workflows": {}, "research_sessions": {}}
    assert not CognitiveStateManager.task_overdue({"due": _days(0)})
    assert not any("OVERDUE" in p for p in PriorityMonitor().due_soon(state))
    assert not any("are overdue" in r for r in StrategicEngine().recommendations(state))
    assert PriorityEngine().rank(state)[0]["reason"] != "OVERDUE"
    assert not MomentumEngine._is_overdue(_days(0))
    loop = CognitiveLoopManager.__new__(CognitiveLoopManager)
    loop.cognition = SimpleNamespace(snapshot=lambda: state)
    assert not any("OVERDUE" in d for d in loop._due_soon())

    state["pending_tasks"]["t"]["due"] = _days(-1)
    assert CognitiveStateManager.task_overdue({"due": _days(-1)})
    assert any("OVERDUE" in p for p in PriorityMonitor().due_soon(state))
    assert any("are overdue" in r for r in StrategicEngine().recommendations(state))
    assert PriorityEngine().rank(state)[0]["reason"] == "OVERDUE"
    assert MomentumEngine._is_overdue(_days(-1))
    assert any("OVERDUE" in d for d in loop._due_soon())
