"""Launch welcomes vary across restarts without inventing context."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from orion_core import temporal
from orion_core.live_worker import GenAILiveWorker
from orion_core.time_service import TimeService


class Signal:
    def emit(self, *args):
        pass


class Bus:
    def __getattr__(self, name):
        return Signal()


@pytest.fixture
def presence(tmp_path, monkeypatch):
    monkeypatch.setattr(temporal, "STATE_PATH", tmp_path / "temporal.json")
    monkeypatch.setattr(temporal, "TIME", TimeService(clock=lambda: datetime(
        2026, 10, 6, 11, 30, tzinfo=timezone.utc)))
    monkeypatch.setattr(temporal.random, "choice", lambda choices: choices[0])
    return temporal.TemporalPresence(Bus())


def feeds(service, values):
    async def enrich(now, budget=None):
        return list(values)
    service._enrich = enrich


def test_identical_facts_produce_different_welcomes_across_restarts(presence):
    values = ["— an overcast autumn morning at 17 degrees", "Your calendar is clear today."]
    greetings = []
    for _ in range(5):
        service = temporal.TemporalPresence(Bus())
        feeds(service, values)
        greetings.append(asyncio.run(service.compose_greeting()))
    assert len(set(greetings)) == 5
    assert all("morning" in text and "17 degrees" in text for text in greetings)
    assert all("a overcast" not in text for text in greetings)
    saved = json.loads(temporal.STATE_PATH.read_text())
    assert saved["greeting_history"] == greetings
    assert len(saved["greeting_openings"]) == 4


def test_live_prompt_contains_all_current_facts_and_recent_examples(presence):
    values = ["— an overcast autumn morning at 17 degrees", "Your calendar is clear today.",
              "It has been 1 day since we last spoke.", "The FTSE 100 is up 0.5 percent."]
    feeds(presence, values)
    previous = asyncio.run(presence.compose_greeting())
    current = asyncio.run(presence.compose_greeting())
    prompt = presence.greeting_instruction(current)
    data = json.loads(prompt.split("\n", 1)[1])
    assert len(data["current_facts"]["observations"]) == 4
    assert previous in data["recent_greeting_examples"]
    assert data["current_facts"]["date"] == "Tuesday the 6th of October"
    assert data["current_facts"]["time"] == "eleven thirty in the morning"
    assert "in your own words" in prompt
    assert "word for word" not in prompt
    assert "don't invent" in prompt
    assert len(current) < len(" ".join(values)) + 80


def test_unavailable_feeds_still_greet_at_the_correct_time(presence, monkeypatch):
    monkeypatch.setattr(temporal, "TIME", TimeService(clock=lambda: datetime(
        2026, 10, 6, 0, 31, tzinfo=timezone.utc)))

    async def unavailable(now):
        raise TimeoutError("feed unavailable")

    presence._enrich = unavailable
    greeting = asyncio.run(presence.compose_greeting())
    assert "evening" in greeting
    assert "Good morning" not in greeting
    assert "calendar" not in greeting
    assert presence._greeting_facts["period"] == "night"


@pytest.mark.parametrize("briefed,fresh", [(False, False), (True, False), (True, True)])
def test_all_live_startup_paths_use_natural_grounded_greetings(presence, briefed, fresh):
    feeds(presence, ["Your calendar is clear today."])
    captured = []

    async def send(instruction):
        captured.append(instruction)

    async def fresh_news():
        return fresh

    worker = SimpleNamespace(
        temporal=presence, connected=True, stop_event=SimpleNamespace(is_set=lambda: False),
        dispatcher=SimpleNamespace(briefing=SimpleNamespace(
            already_briefed_today=lambda: briefed,
            has_fresh_news_since_last_briefing=fresh_news,
            briefing_recency_phrase=lambda: "ten minutes ago")),
        bus=Bus(), session=object(), awaiting_briefing=False,
        _refresh_wake_window=lambda: None, _compose_greeting=presence.compose_greeting,
        _emit_state=lambda state: None, _send_text_turn=send,
    )
    asyncio.run(GenAILiveWorker.offer_startup_briefing(worker))
    assert len(captured) == 1
    assert "current_facts" in captured[0]
    assert "in your own words" in captured[0]
    assert "word for word" not in captured[0]
    assert worker.awaiting_briefing is (not briefed or fresh)
    if briefed and not fresh:
        assert "Do NOT offer another briefing" in captured[0]
        assert "ten minutes ago" in captured[0]
    else:
        assert "Do NOT deliver the briefing yet" in captured[0]
        assert "Only if they agree" in captured[0]


# ── calendar facts are today's (audit 6 Oct 2026, A05) ──────────────────────

def _calendar(rows, available=True):
    """The real NotionService.events_on over a synthetic database that applies
    the same inclusive date filter Notion does."""
    from orion_core.notion import NotionService
    notion = NotionService(Bus(), {"token": "synthetic-token" if available else "",
                                  "calendar_database_id": "synthetic-calendar"})

    async def schema(_database):
        return {"date": "Date", "title": "Name"}

    async def query(_database, **kwargs):
        lower = kwargs["filter_payload"]["and"][0]["date"]["on_or_after"]
        upper = kwargs["filter_payload"]["and"][1]["date"]["on_or_before"]
        found = [r for r in rows if lower <= r["date"][:10] <= upper]
        return found[:kwargs["limit"]]

    notion._database_schema = schema
    notion._query_database = query
    return notion


def _calendar_says(presence, rows, available=True):
    presence.notion = _calendar(rows, available)
    return asyncio.run(presence._calendar_fragment())


def test_a_tomorrow_only_event_is_not_today(presence):
    rows = [{"date": "2026-10-07", "title": "Dentist", "status": ""}]
    assert _calendar_says(presence, rows) == "Your calendar is clear today."


def test_today_is_counted_and_tomorrow_left_out(presence):
    rows = [{"date": "2026-10-06", "title": "Stand-up", "status": ""},
            {"date": "2026-10-06T15:00:00.000+01:00", "title": "Gym", "status": ""},
            {"date": "2026-10-07", "title": "Dentist", "status": ""}]
    assert _calendar_says(presence, rows) == "You have 2 calendar entries today."


def test_a_capped_list_is_not_reported_as_a_count(presence):
    rows = [{"date": "2026-10-06", "title": f"Call {i}", "status": ""} for i in range(9)]
    assert _calendar_says(presence, rows) == (
        f"You have more than {presence.CALENDAR_LIMIT} calendar entries today.")


def test_an_unreadable_calendar_is_never_called_clear(presence):
    assert _calendar_says(presence, [], available=False) == ""
    presence.notion = _calendar([])

    async def broken(_database, **_kwargs):
        raise RuntimeError("401 unauthorised")
    presence.notion._query_database = broken
    assert asyncio.run(presence._calendar_fragment()) == ""


# ── one slow feed costs only itself (audit 6 Oct 2026, A06) ─────────────────

def _feeds(presence, monkeypatch, **fragments):
    started = []

    def make(name, value, delay):
        async def feed(*_args):
            started.append(name)
            await asyncio.sleep(delay)
            if isinstance(value, Exception):
                raise value
            return value
        return feed
    for name in ("weather", "calendar", "last_seen", "day", "market"):
        value, delay = fragments.get(name, ("", 0))
        monkeypatch.setattr(presence, f"_{name}_fragment", make(name, value, delay))
    return started


def test_a_slow_feed_does_not_erase_the_ones_that_answered(presence, monkeypatch):
    presence.TIME_BUDGET_S = 0.05
    _feeds(presence, monkeypatch, weather=("It is overcast at 17 degrees.", 0),
           market=("The market is up.", 2), calendar=(RuntimeError("down"), 0))
    asyncio.run(presence.compose_greeting())
    assert presence._greeting_facts["observations"] == ["It is overcast at 17 degrees."]


def test_every_feed_late_still_greets_and_leaves_nothing_running(presence, monkeypatch):
    presence.TIME_BUDGET_S = 0.05
    _feeds(presence, monkeypatch, weather=("Sunny.", 2), market=("Up.", 2))

    async def greet():
        text = await presence.compose_greeting()
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return text, pending
    text, pending = asyncio.run(greet())
    assert presence._greeting_facts["observations"] == []
    assert "Tuesday the 6th of October" in text
    assert pending == [], "a late feed outlived the greeting"


# ── ORION's own lines, said differently (voice_variety) ─────────────────────

def test_a_line_is_never_one_of_its_kinds_last_few(tmp_path):
    import random
    from orion_core.voice_variety import AVOID_RECENT, BANK, Variety
    v = Variety(tmp_path / "variety.json")
    picks = [v.pick("wake", hour=14, rng=random.Random(i)) for i in range(40)]
    for i in range(AVOID_RECENT, len(picks)):
        assert picks[i] not in picks[i - AVOID_RECENT:i], picks[i - AVOID_RECENT:i + 1]
    assert len(set(picks)) >= len(BANK["wake"]) - AVOID_RECENT
    # ...and the memory survives a restart.
    again = Variety(tmp_path / "variety.json")
    assert again.pick("wake", hour=14) not in picks[-AVOID_RECENT:]


def test_late_at_night_the_shorter_lines_are_preferred():
    import random
    from orion_core.voice_variety import BANK, Variety
    v = Variety(None)
    longest = max(BANK["wake"], key=len)
    picks = {v.pick("wake", hour=2, rng=random.Random(i)) for i in range(30)}
    lengths = sorted(len(line) for line in BANK["wake"])
    assert longest not in picks and max(len(p) for p in picks) <= lengths[len(lengths) // 2 + 1]


def test_the_neural_network_steers_away_from_what_was_just_said():
    """With vectors ready, the pick is weighted toward the line least like
    the last things said - and nothing is embedded on the caller's thread."""
    import random
    import threading
    import numpy as np
    from orion_core.voice_variety import Variety

    calls = []

    class _Encoder:
        def ready(self):
            return True

        def encode_one(self, text):
            calls.append(threading.current_thread() is threading.main_thread())
            vec = np.zeros(4)
            vec[0 if "listen" in text.lower() else 1] = 1.0      # two 'meanings'
            return vec

    v = Variety(None, encoder=_Encoder())
    v.note_spoken("I'm listening carefully.")
    v.pick("wake", hour=14)                     # starts the embedding worker
    for _ in range(200):
        if len(v._vectors) >= 13:
            break
        threading.Event().wait(0.01)
    assert calls and not any(calls), "embedded on the caller's thread"
    listening = 0
    for i in range(60):                          # each time, just after saying it
        v._recent.clear()
        v._spoken.clear()
        v._spoken.append("I'm listening carefully.")
        listening += "listen" in v.pick("wake", hour=14, rng=random.Random(i)).lower()
    assert listening <= 2, f"picked a 'listening' line {listening}x right after saying one"


def test_the_style_note_names_the_habits_it_measured():
    import random
    from orion_core.voice_variety import Variety
    replies = (["Certainly, sir. I have opened the file for you."] * 4
               + ["Right then, would you like me to check the weather?"] * 3
               + ["Good afternoon. It is a bright day."] * 5
               + ["Done - the music is paused."] * 4)
    note = Variety(None).style_note(replies, rng=random.Random(3))
    assert "'Certainly, sir…' (4x)" in note and "'Right then…' (3x)" in note
    assert "Good afternoon" not in note, "greetings are varied separately"
    assert "'sir' in 25%" in note
    assert "This session," in note


@pytest.mark.parametrize("completion_kind", ["checkbox", "status"])
def test_completed_notion_rows_disappear_from_default_lists_after_relaunch(completion_kind):
    """A fresh service must reread persisted completion, without relying on
    the previous session's schema, dismissed-task memory, or displayed rows.
    The real board has an old Progress select beside its Complete checkbox.
    """
    import copy
    from unittest.mock import Mock

    from orion_core.briefing import MorningBriefingService
    from orion_core.notion import NotionService

    field = "Complete" if completion_kind == "checkbox" else "State"
    unfinished = False if completion_kind == "checkbox" else {"name": "To do"}
    done_name = "Done" if completion_kind == "checkbox" else "Delivered"
    schema = {"properties": {
        "Name": {"type": "title", "title": {}},
        "Due": {"type": "date", "date": {}},
        "Progress": {"type": "select", "select": {"options": [
            {"name": "Not started"}, {"name": "In progress"}, {"name": "Complete"},
        ]}},
        field: {"type": completion_kind, completion_kind: {} if completion_kind == "checkbox" else {
            "options": [{"id": "todo", "name": "To do"}, {"id": "sent", "name": "Delivered"}],
            "groups": [{"name": "Complete", "option_ids": ["sent"]}],
        }},
    }}
    persisted_page = {"id": "persisted-task", "url": "https://notion.example/persisted-task",
        "properties": {
            "Name": {"title": [{"plain_text": "Finished setup"}]},
            "Due": {"date": {"start": "2026-10-17"}},
            "Progress": {"select": {"name": "Not started"}},
            field: {completion_kind: unfinished},
        }}
    requests = []

    async def request(method, path, payload=None):
        requests.append((method, path, copy.deepcopy(payload)))
        if method == "GET" and path == "/databases/synthetic-board":
            return copy.deepcopy(schema)
        if method == "POST" and path == "/databases/synthetic-board/query":
            return {"results": [copy.deepcopy(persisted_page)], "has_more": False}
        if method == "PATCH" and path == "/pages/persisted-task":
            persisted_page["properties"].update(copy.deepcopy(payload["properties"]))
            return copy.deepcopy(persisted_page)
        raise AssertionError(f"Unexpected synthetic Notion request: {method} {path}")

    def launch():
        signal = SimpleNamespace(emit=Mock())
        service = NotionService(SimpleNamespace(dashboard_event=signal), {"token": "synthetic-token"})
        service.tasks_db = "synthetic-board"
        service.calendar_db = ""
        service._request = request
        return service

    first_session = launch()
    assert [row["title"] for row in asyncio.run(first_session.open_tasks())] == ["Finished setup"]
    result = asyncio.run(first_session.complete_task("Finished setup"))
    assert result.ok and result.receipt["completed"] is True
    assert persisted_page["properties"][field][completion_kind] == (
        True if completion_kind == "checkbox" else {"name": done_name})

    restarted = launch()
    assert restarted._schema_cache == {}
    assert asyncio.run(restarted.open_tasks()) == []
    rendered = asyncio.run(restarted.list_tasks())
    assert rendered.ok and rendered.text == "The task board is clear."
    restarted.bus.dashboard_event.emit.assert_called_once_with("tasks", [])
    briefing = SimpleNamespace(notion=restarted, cognition=None)
    assert asyncio.run(MorningBriefingService._tasks_section(briefing)) == "Tasks: The task board is clear."
    # With no dedicated calendar, dated tasks must not return through agenda
    # or today's greeting after disappearing from the default task list.
    assert asyncio.run(restarted.events_on(datetime(2026, 10, 17).date())) == []
    agenda = asyncio.run(restarted.upcoming_events())
    assert agenda.ok and "calendar is clear" in agenda.text

    # Removing completed rows from ORION's active view needs no archive/write
    # during startup and leaves history available through an explicit request.
    assert [row["title"] for row in asyncio.run(restarted.open_tasks(include_done=True))] == ["Finished setup"]
    writes = [entry for entry in requests if entry[0] == "PATCH"]
    assert len(writes) == 1
    assert writes[0][2] == {"properties": {field: {completion_kind: (
        True if completion_kind == "checkbox" else {"name": done_name})}}}


def test_dedicated_notion_calendar_keeps_its_events_when_status_is_done():
    """Task completion filters must not change a separate calendar's policy."""
    from orion_core.notion import NotionService

    service = NotionService(Bus(), {"token": "synthetic-token"})
    service.tasks_db = "synthetic-tasks"
    service.calendar_db = "synthetic-calendar"

    async def request(method, path, payload=None):
        if method == "GET" and path == "/databases/synthetic-calendar":
            return {"properties": {"Name": {"type": "title", "title": {}},
                "Date": {"type": "date", "date": {}},
                "Status": {"type": "status", "status": {"options": [{"name": "Done"}]}},
            }}
        assert method == "POST" and path == "/databases/synthetic-calendar/query"
        return {"results": [{"id": "calendar-entry", "properties": {
            "Name": {"title": [{"plain_text": "Recorded event"}]},
            "Date": {"date": {"start": "2026-10-17"}},
            "Status": {"status": {"name": "Done"}},
        }}], "has_more": False}

    service._request = request
    assert [row["title"] for row in asyncio.run(service.events_on(datetime(2026, 10, 17).date()))] == ["Recorded event"]
    agenda = asyncio.run(service.upcoming_events())
    assert agenda.ok and "Recorded event" in agenda.text


@pytest.mark.parametrize("requested_due,returned_due", [
    ("2026-10-20T13:00:27+02:00", "2026-10-20T13:00:27.000+02:00"),
    ("2026-10-20T11:00:27Z", "2026-10-20T11:00:27.000+00:00"),
])
def test_verified_notion_edit_preserves_deadline_offset_and_seconds_in_memory_receipt(requested_due, returned_due):
    from unittest.mock import AsyncMock
    from orion_core.notion import NotionService

    service = NotionService(Bus(), {"token": "synthetic-token"})
    service.tasks_db = "synthetic-tasks"
    service._database_schema = AsyncMock(return_value={
        "title": "Name", "date": "Due", "status": "Status", "status_kind": "status"})
    service._find_task = AsyncMock(return_value={"id": "one", "title": "Keep deadline", "date": "",
                                                "status": "To do", "url": ""})
    service._request = AsyncMock(return_value={"id": "one", "properties": {
        "Name": {"title": [{"plain_text": "Keep deadline"}]},
        "Due": {"date": {"start": returned_due}},
    }})
    result = asyncio.run(service.update_task("Keep deadline", due=requested_due))
    assert result.ok and result.receipt["date"] == returned_due
    parsed = service._parse_page(service._request.return_value, service._database_schema.return_value)
    assert parsed["date"] == returned_due[:16].replace("T", " ")
    assert parsed["date_raw"] == returned_due


def test_cleared_notion_deadline_receipt_does_not_retain_previous_raw_date():
    from unittest.mock import AsyncMock
    from orion_core.notion import NotionService

    service = NotionService(Bus(), {"token": "synthetic-token"})
    service.tasks_db = "synthetic-tasks"
    service._database_schema = AsyncMock(return_value={
        "title": "Name", "date": "Due", "status": "Status", "status_kind": "status"})
    service._find_task = AsyncMock(return_value={"id": "one", "title": "Clear deadline",
        "date": "2026-10-20 13:00", "date_raw": "2026-10-20T13:00:27+02:00", "status": "To do", "url": ""})
    service._request = AsyncMock(return_value={"id": "one", "properties": {"Due": {"date": None}}})
    result = asyncio.run(service.update_task("Clear deadline", due=""))
    assert result.ok and result.receipt["date"] == ""


@pytest.mark.parametrize("action", ["rename", "complete"])
def test_accepted_notion_write_retains_page_identity_when_readback_is_unavailable(action):
    from unittest.mock import AsyncMock
    from orion_core.notion import NotionService

    service = NotionService(Bus(), {"token": "synthetic-token"})
    service.tasks_db = "synthetic-tasks"
    service._database_schema = AsyncMock(return_value={"title": "Name", "date": "Due", "status": "Status",
        "status_kind": "status", "done_options": ["Done"], "status_options": ["To do", "Done"]})
    service._find_task = AsyncMock(return_value={"id": "one", "title": "Original title", "date": "",
                                                "status": "To do", "url": ""})
    service._request = AsyncMock(side_effect=[{"id": "one"}, OSError("readback unavailable")])
    result = asyncio.run(service.complete_task("Original title") if action == "complete" else
                         service.update_task("Original title", title="Changed title", due="2026-10-20T13:00:27+02:00"))
    assert not result.ok and "accepted" in result.text and "could not be verified" in result.text
    assert result.receipt["verification_pending"] is True
    assert result.receipt["board_write_accepted"] is True
    assert result.receipt["page_id"] == "one" and result.receipt["original_task"] == "Original title"
    assert "completed" not in result.receipt and "terminal" not in result.receipt and "task" not in result.receipt
    if action == "rename":
        assert result.receipt["requested_task"] == "Changed title"
        assert result.receipt["requested_date"] == "2026-10-20T13:00:27+02:00"
    else:
        assert result.receipt["requested_task"] == "Original title"
        assert result.receipt["requested_status"] == "Done"
    assert service._request.await_args_list[-1].args == ("GET", "/pages/one")


def test_rejected_notion_write_does_not_claim_the_board_accepted_it():
    from unittest.mock import AsyncMock
    from orion_core.notion import NotionService

    service = NotionService(Bus(), {"token": "synthetic-token"})
    service.tasks_db = "synthetic-tasks"
    service._database_schema = AsyncMock(return_value={"title": "Name", "date": "Due", "status": "Status", "status_kind": "status"})
    service._find_task = AsyncMock(return_value={"id": "one", "title": "Original title", "date": "", "status": "To do", "url": ""})
    service._request = AsyncMock(side_effect=RuntimeError("Notion API 403: update denied"))
    result = asyncio.run(service.update_task("Original title", title="Changed title"))
    assert not result.ok and result.receipt is None


def _notion_with_restored_local_completion(tmp_path, *, remaining=False):
    from unittest.mock import Mock
    from orion_core.cognition import CognitiveStateManager
    from orion_core.notion import NotionService

    memory = Mock()
    state_path = tmp_path / "continuity.json"
    first_session = CognitiveStateManager(Bus(), memory, path=state_path)
    for index in range(3):
        first_session.dismiss_task(f"Finished locally {index}", "completed by user")
    restored = CognitiveStateManager(Bus(), memory, path=state_path)
    restored._load_locked()
    service = NotionService(SimpleNamespace(dashboard_event=SimpleNamespace(emit=Mock())),
                            {"token": "synthetic-token"})
    service.tasks_db, service.calendar_db = "synthetic-tasks", ""
    service.cognition = restored
    day = "2026-10-17"
    calls = []

    def page(title, key):
        return {"id": key, "properties": {
            "Name": {"title": [{"plain_text": title}]},
            "Complete": {"checkbox": False},
            "Due": {"date": {"start": day}},
        }}

    async def request(method, path, payload=None):
        calls.append((method, path, payload))
        if method == "GET":
            return {"properties": {"Name": {"type": "title", "title": {}},
                "Complete": {"type": "checkbox", "checkbox": {}},
                "Due": {"type": "date", "date": {}},
            }}
        assert method == "POST" and path.endswith("/query"), "Visibility filtering must not write to Notion"
        if payload.get("start_cursor") == "after-local-completions":
            return {"results": [page("Still open remotely", "open")], "has_more": False}
        return {"results": [page(f"Finished locally {index}", f"done-{index}") for index in range(3)],
                "has_more": remaining, "next_cursor": "after-local-completions" if remaining else None}

    service._request = request
    return service, calls


def test_restored_local_completion_hides_remote_open_rows_from_every_default_feed(tmp_path):
    from orion_core.briefing import MorningBriefingService

    service, requests = _notion_with_restored_local_completion(tmp_path)
    assert asyncio.run(service.open_tasks()) == []
    result = asyncio.run(service.list_tasks())
    assert result.ok and result.text == "The task board is clear."
    service.bus.dashboard_event.emit.assert_called_with("tasks", [])
    briefing = SimpleNamespace(notion=service, cognition=service.cognition)
    assert asyncio.run(MorningBriefingService._tasks_section(briefing)) == "Tasks: The task board is clear."
    assert asyncio.run(service.events_on(datetime(2026, 10, 17).date())) == []
    agenda = asyncio.run(service.upcoming_events())
    assert agenda.ok and "calendar is clear" in agenda.text
    assert len(asyncio.run(service.open_tasks(include_done=True))) == 3
    assert all(method in {"GET", "POST"} for method, _, _ in requests)


def test_local_completion_filter_runs_before_limits_for_tasks_and_task_calendar(tmp_path):
    service, _ = _notion_with_restored_local_completion(tmp_path, remaining=True)
    assert [row["title"] for row in asyncio.run(service.open_tasks(limit=1))] == ["Still open remotely"]
    events = asyncio.run(service.events_on(datetime(2026, 10, 17).date(), limit=1))
    assert [row["title"] for row in events] == ["Still open remotely"]
    agenda = asyncio.run(service.upcoming_events(limit=1))
    assert agenda.ok and "Still open remotely" in agenda.text and "Finished locally" not in agenda.text


def test_dedicated_calendar_ignores_local_task_completion_markers(tmp_path):
    service, _ = _notion_with_restored_local_completion(tmp_path)
    service.calendar_db = "dedicated-calendar"
    events = asyncio.run(service.events_on(datetime(2026, 10, 17).date()))
    assert len(events) == 3
    agenda = asyncio.run(service.upcoming_events())
    assert agenda.ok and "Finished locally 0" in agenda.text


@pytest.mark.parametrize("calendar_reference", ["empty", "same", "same_uuid_other_format", "distinct"])
def test_calendar_database_identity_controls_task_completion_filtering(calendar_reference):
    from unittest.mock import Mock
    from orion_core.notion import NotionService

    service = NotionService(Bus(), {"token": "synthetic-token"})
    service.tasks_db = "0123456789abcdef0123456789abcdef"
    service.calendar_db = {
        "empty": "", "same": service.tasks_db,
        "same_uuid_other_format": "01234567-89AB-CDEF-0123-456789ABCDEF",
        "distinct": "fedcba9876543210fedcba9876543210",
    }[calendar_reference]
    service.cognition = SimpleNamespace(is_dismissed=Mock(side_effect=lambda title: title == "Finished locally"))
    calls = []

    def page(title, done):
        return {"id": title, "properties": {"Name": {"title": [{"plain_text": title}]},
            "Complete": {"checkbox": done}, "Due": {"date": {"start": "2026-10-17"}}}}

    async def request(method, path, payload=None):
        calls.append((method, path))
        if method == "GET":
            return {"properties": {"Name": {"type": "title", "title": {}},
                "Complete": {"type": "checkbox", "checkbox": {}}, "Due": {"type": "date", "date": {}}}}
        assert method == "POST" and path.endswith("/query")
        if payload.get("start_cursor") == "remaining":
            return {"results": [page("Still open", False)], "has_more": False}
        return {"results": [page("Finished remotely", True), page("Finished locally", False)],
                "has_more": True, "next_cursor": "remaining"}

    service._request = request
    shared_tasks = calendar_reference != "distinct"
    limit = 1 if shared_tasks else 10
    rows = asyncio.run(service.events_on(datetime(2026, 10, 17).date(), limit=limit))
    expected = ["Still open"] if shared_tasks else ["Finished remotely", "Finished locally", "Still open"]
    assert [row["title"] for row in rows] == expected
    agenda = asyncio.run(service.upcoming_events(limit=limit))
    assert agenda.ok and "Still open" in agenda.text
    assert ("Finished remotely" not in agenda.text) is shared_tasks
    assert ("Finished locally" not in agenda.text) is shared_tasks
    assert all(method in {"GET", "POST"} for method, _ in calls)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.status, self._payload = status, payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _fake_session(payload=None, error=None):
    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def get(self, url):
            if error is not None:
                raise error
            return _FakeResponse(payload)
    return Session


def test_saved_location_is_refreshed_at_every_start(presence, monkeypatch):
    # 23 Oct: the saved place (Teignmouth) was never looked up again, so the
    # greeting kept naming it after the owner moved to Colchester.
    presence._state.update({"latitude": 50.5455, "longitude": -3.4911, "city": "Teignmouth"})
    monkeypatch.setattr(temporal, "ClientSession", _fake_session({
        "latitude": 51.8959, "longitude": 0.8919, "city": "Colchester",
        "region": "England", "country_name": "United Kingdom"}))
    routed = []
    router = SimpleNamespace(set_locality=routed.append)
    locality = asyncio.run(presence.prime_locality(router))
    assert locality == "Colchester, England, United Kingdom"
    assert routed == [locality]
    assert presence.coordinates() == (51.8959, 0.8919)


def test_failed_lookup_keeps_the_last_known_place(presence, monkeypatch):
    presence._state.update({"latitude": 50.5455, "longitude": -3.4911, "city": "Teignmouth"})
    monkeypatch.setattr(temporal, "ClientSession", _fake_session(error=OSError("offline")))
    assert asyncio.run(presence.prime_locality(None)) == "Teignmouth"



def test_untitled_notion_tasks_can_be_completed_and_archived():
    """28 Oct 2026: "clear the tasks in Notion" found two untitled rows, and
    searching the board for "(untitled)" matched neither."""
    from unittest.mock import AsyncMock
    from orion_core.notion import NotionService
    service = NotionService(Bus(), {"token": "synthetic-token"})
    service.tasks_db = "synthetic-tasks"
    service._database_schema = AsyncMock(return_value={
        "title": "Name", "date": "Due", "status": "Done", "status_kind": "checkbox",
        "done_options": []})
    rows = [{"id": "a", "title": "(untitled)", "status": "Not started", "date": "", "url": ""},
            {"id": "b", "title": "(untitled)", "status": "Not started", "date": "", "url": ""}]
    service._query_database = AsyncMock(return_value=rows)
    service._verified_update = AsyncMock(side_effect=lambda task, schema, props: dict(task, status="Done"))
    result = asyncio.run(service.complete_task("(untitled)"))
    assert result.ok and result.text == "Marked 2 untitled Notion tasks complete."
    assert service._query_database.call_args[0][1] == {"property": "Name", "title": {"is_empty": True}}
    service._request = AsyncMock(return_value={"archived": True})
    archived = asyncio.run(service.archive_task_exact("untitled"))
    assert archived.ok and archived.text == "Archived 2 untitled Notion tasks."


def test_notion_takes_the_short_names_a_model_uses_for_its_actions():
    from orion_core.action_aliases import normalise_action
    assert normalise_action("notion_workspace", "create") == "create_task"
    assert normalise_action("notion_workspace", "add") == "create_task"
    assert normalise_action("notion_workspace", "complete") == "complete_task"
    assert normalise_action("notion_workspace", "list_tasks") == "list_tasks"
