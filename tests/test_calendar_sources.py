"""
Calendar sources and conflict-aware scheduling end to end (brief §2.4).

The scheduling engine existed and was tested, but nothing ever called it and
no calendar was ever read, so "find me 90 minutes this week" had no answer.
These tests pin the ICS reader against what real calendars emit, and the
executive's find_time/book_slot actions on top of it.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import calendar_sources as cs  # noqa: E402
from orion_core.calendar_sources import BusyBlock, notion_busy, outlook_busy, parse_ics  # noqa: E402
from orion_core.data import ToolResult  # noqa: E402

LONDON = ZoneInfo("Europe/London")
# Monday 5 October 2026, 08:00 London.
MONDAY = datetime(2026, 10, 5, 8, 0, tzinfo=LONDON)
WEEK_END = MONDAY + timedelta(days=7)


def _ics(*events: str) -> str:
    body = "\r\n".join(f"BEGIN:VEVENT\r\n{e.strip()}\r\nEND:VEVENT" for e in events)
    return f"BEGIN:VCALENDAR\r\nVERSION:2.0\r\n{body}\r\nEND:VCALENDAR\r\n"


def _blocks(text: str):
    return parse_ics(text, MONDAY, WEEK_END, LONDON)


def _hours(blocks):
    return [(b.start.strftime("%a %H:%M"), b.end.strftime("%H:%M")) for b in blocks]


# ── single events and time values ───────────────────────────────────────────

def test_utc_and_zoned_and_floating_times():
    blocks = _blocks(_ics(
        "UID:a\nDTSTART:20261005T090000Z\nDTEND:20261005T100000Z\nSUMMARY:UTC",
        "UID:b\nDTSTART;TZID=Europe/Paris:20261005T140000\nDTEND;TZID=Europe/Paris:20261005T150000\nSUMMARY:Paris",
        "UID:c\nDTSTART:20261006T110000\nDTEND:20261006T113000\nSUMMARY:Floating",
    ))
    # 09:00Z is 10:00 BST; 14:00 Paris is 13:00 London; floating is local.
    assert _hours(blocks) == [("Mon 10:00", "11:00"), ("Mon 13:00", "14:00"),
                              ("Tue 11:00", "11:30")]


def test_outlook_windows_zone_names_are_understood():
    blocks = _blocks(_ics(
        "UID:w\nDTSTART;TZID=Eastern Standard Time:20261005T090000\n"
        "DTEND;TZID=Eastern Standard Time:20261005T093000\nSUMMARY:NY call"))
    assert _hours(blocks) == [("Mon 14:00", "14:30")]


def test_folded_lines_and_escaped_text():
    text = _ics("UID:f\nDTSTART:20261005T120000\nDTEND:20261005T130000\n"
                "SUMMARY:Lunch with\r\n  the team\\, finally")
    assert _blocks(text)[0].title == "Lunch with the team, finally"


def test_duration_instead_of_dtend():
    blocks = _blocks(_ics("UID:d\nDTSTART:20261007T090000\nDURATION:PT1H30M"))
    assert _hours(blocks) == [("Wed 09:00", "10:30")]


@pytest.mark.parametrize("marker", [
    "TRANSP:TRANSPARENT", "STATUS:CANCELLED", "X-MICROSOFT-CDO-BUSYSTATUS:FREE"])
def test_free_and_cancelled_entries_do_not_block(marker):
    assert _blocks(_ics(
        f"UID:x\nDTSTART:20261005T090000\nDTEND:20261005T100000\n{marker}")) == []


def test_all_day_entries_block_only_when_marked_busy():
    birthday = "UID:b\nDTSTART;VALUE=DATE:20261006\nDTEND;VALUE=DATE:20261007\nSUMMARY:Birthday"
    leave = ("UID:o\nDTSTART;VALUE=DATE:20261008\nDTEND;VALUE=DATE:20261009\n"
             "X-MICROSOFT-CDO-BUSYSTATUS:OOF\nSUMMARY:Leave")
    blocks = _blocks(_ics(birthday, leave))
    assert [b.title for b in blocks] == ["Leave"]
    assert blocks[0].end - blocks[0].start == timedelta(days=1)


def test_alarms_inside_an_event_do_not_leak_properties():
    text = _ics("UID:al\nDTSTART:20261005T090000\nDTEND:20261005T100000\nSUMMARY:Real\n"
                "BEGIN:VALARM\nTRIGGER:-PT15M\nSUMMARY:Alarm text\nEND:VALARM")
    assert [b.title for b in _blocks(text)] == ["Real"]


def test_a_malformed_event_is_skipped_not_fatal():
    text = _ics("UID:bad\nDTSTART:not-a-date\nDTEND:20261005T100000",
                "UID:good\nDTSTART:20261005T090000\nDTEND:20261005T100000")
    assert len(_blocks(text)) == 1


# ── recurrence ──────────────────────────────────────────────────────────────

def test_weekly_by_day_with_an_exception_and_a_moved_occurrence():
    series = ("UID:standup\nDTSTART;TZID=Europe/London:20260907T093000\n"
              "DTEND;TZID=Europe/London:20260907T094500\n"
              "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR\n"
              "EXDATE;TZID=Europe/London:20261007T093000\nSUMMARY:Stand-up")
    moved = ("UID:standup\nRECURRENCE-ID;TZID=Europe/London:20261009T093000\n"
             "DTSTART;TZID=Europe/London:20261009T160000\n"
             "DTEND;TZID=Europe/London:20261009T161500\nSUMMARY:Stand-up (moved)")
    blocks = _blocks(_ics(series, moved))
    # The window ends Monday 12 Oct at 08:00, before that day's stand-up.
    assert _hours(blocks) == [("Mon 09:30", "09:45"),     # Wed excluded
                              ("Fri 16:00", "16:15")]     # Fri moved from 09:30


def test_daily_with_count_stops():
    blocks = _blocks(_ics("UID:c\nDTSTART:20261005T170000\nDTEND:20261005T173000\n"
                          "RRULE:FREQ=DAILY;COUNT=3"))
    assert len(blocks) == 3


def test_until_is_inclusive_of_its_day():
    blocks = _blocks(_ics("UID:u\nDTSTART:20261005T120000\nDTEND:20261005T130000\n"
                          "RRULE:FREQ=DAILY;UNTIL=20261007"))
    assert [b.start.day for b in blocks] == [5, 6, 7]


def test_a_series_begun_years_ago_still_reaches_this_week():
    blocks = _blocks(_ics("UID:old\nDTSTART:20180101T080000Z\nDTEND:20180101T083000Z\n"
                          "RRULE:FREQ=DAILY"))
    assert len(blocks) == 7, "fast-forward failed; the expansion budget ran out first"


def test_monthly_last_friday():
    window = parse_ics(_ics("UID:m\nDTSTART:20260130T150000\nDTEND:20260130T160000\n"
                            "RRULE:FREQ=MONTHLY;BYDAY=-1FR"),
                       datetime(2026, 10, 1, tzinfo=LONDON),
                       datetime(2026, 11, 30, tzinfo=LONDON), LONDON)
    assert [b.start.date().isoformat() for b in window] == ["2026-10-30", "2026-11-27"]


def test_biweekly_interval():
    blocks = parse_ics(_ics("UID:i\nDTSTART:20260907T100000\nDTEND:20260907T110000\n"
                            "RRULE:FREQ=WEEKLY;INTERVAL=2"),
                       MONDAY, MONDAY + timedelta(days=21), LONDON)
    assert [b.start.day for b in blocks] == [5, 19]


# ── Outlook and Notion adapters ─────────────────────────────────────────────

class _OutlookItem:
    def __init__(self, start, end, subject, busy=2):
        self.Start, self.End, self.Subject, self.BusyStatus = start, end, subject, busy


class _OutlookItems:
    def __init__(self, items):
        self._items = items
        self._i = 0
        self.IncludeRecurrences = False
        self.restriction = ""

    def Sort(self, key):  # noqa: N802
        pass

    def Restrict(self, text):  # noqa: N802
        self.restriction = text
        return self

    def GetFirst(self):  # noqa: N802
        self._i = 0
        return self.GetNext()

    def GetNext(self):  # noqa: N802
        if self._i >= len(self._items):
            return None
        item = self._items[self._i]
        self._i += 1
        return item


class _Namespace:
    def __init__(self, items):
        self.items = _OutlookItems(items)

    def GetDefaultFolder(self, number):  # noqa: N802
        assert number == 9
        return type("Folder", (), {"Items": self.items})()


def test_outlook_reads_busy_items_and_skips_free_ones():
    namespace = _Namespace([
        _OutlookItem(datetime(2026, 10, 5, 10), datetime(2026, 10, 5, 11), "Review"),
        _OutlookItem(datetime(2026, 10, 5, 12), datetime(2026, 10, 5, 13), "Lunch", busy=0),
        _OutlookItem(datetime(2026, 10, 6, 9), datetime(2026, 10, 6, 9, 30), "Tentative", busy=1),
    ])
    blocks = outlook_busy(namespace, MONDAY, WEEK_END, LONDON)
    assert [b.title for b in blocks] == ["Review", "Tentative"]
    assert namespace.items.IncludeRecurrences is True
    assert "[Start] <" in namespace.items.restriction


def test_notion_date_only_entries_are_deadlines_not_appointments():
    pages = [
        {"properties": {"When": {"date": {"start": "2026-10-06T14:00:00+01:00",
                                          "end": "2026-10-06T15:30:00+01:00"}},
                        "Name": {"title": [{"plain_text": "Client call"}]}}},
        {"properties": {"When": {"date": {"start": "2026-10-07"}},
                        "Name": {"title": [{"plain_text": "Report due"}]}}},
        {"properties": {"When": {"date": {"start": "2026-10-08T10:00:00"}},
                        "Name": {"title": [{"plain_text": "No end given"}]}}},
    ]
    blocks = notion_busy(pages, "When", "Name", MONDAY, WEEK_END, LONDON)
    assert [(b.title, b.end - b.start) for b in blocks] == [
        ("Client call", timedelta(minutes=90)), ("No end given", timedelta(hours=1))]


def test_feed_names_hide_the_secret_part_of_a_private_url():
    url = "https://calendar.google.com/calendar/ical/me%40x.com/private-abc123/basic.ics"
    assert cs.describe_feed(url) == "calendar.google.com"
    assert "private-abc123" not in cs.describe_feed(url)


def test_configured_feeds_read_calendars_json(tmp_path):
    (tmp_path / "calendars.json").write_text(
        '{"ics": ["https://example.com/a.ics", "  ", "C:/cal.ics"]}', encoding="utf-8")
    assert cs.configured_feeds(tmp_path) == ["https://example.com/a.ics", "C:/cal.ics"]
    assert cs.configured_feeds(tmp_path / "missing") == []


# ── find_time / book_slot ───────────────────────────────────────────────────

class _Signal:
    def emit(self, *a):
        pass


class _Bus:
    def __getattr__(self, name):
        return _Signal()


class _Cognition:
    def __init__(self):
        self.tasks = []

    def add_task(self, title, project, when):
        self.tasks.append((title, when))


class _Memory:
    active_project = ""


class _Outlook:
    available = True

    def __init__(self, blocks):
        self.blocks = blocks

    async def busy_blocks(self, start, end, zone):
        return ToolResult("ok", evidence=[{"block": b} for b in self.blocks])


class _BrokenNotion:
    available = True

    async def busy_blocks(self, start, end, zone):
        return ToolResult("Notion calendar query failed: 401 unauthorised", ok=False)


@pytest.fixture()
def assistant(monkeypatch, tmp_path):
    from orion_core.executive import ExecutiveAssistantMode
    from orion_core.time_service import TIME

    TIME.set_clock(lambda: MONDAY)
    monkeypatch.setattr(cs, "configured_feeds", lambda *a: [])
    busy = [BusyBlock(datetime(2026, 10, 5, 9, tzinfo=LONDON),
                      datetime(2026, 10, 5, 12, tzinfo=LONDON), "Workshop", "Outlook")]
    mode = ExecutiveAssistantMode(_Bus(), _Memory(), _Cognition(),
                                  notion=_BrokenNotion(), outlook=_Outlook(busy))
    yield mode
    TIME.set_clock(None)


def test_find_time_skips_busy_time_and_names_what_it_could_not_read(assistant):
    result = asyncio.run(assistant.find_time(minutes=90, days=2, title="the proposal"))
    assert result.ok
    assert "Mon 05 Oct, 12:00–13:30" in result.text      # first gap after the workshop
    assert "Outlook (1 busy)" in result.text
    assert "Not checked: Notion" in result.text
    assert "Nothing is booked" in result.text


def test_find_time_honours_a_deadline_and_working_hours(assistant):
    result = asyncio.run(assistant.find_time(minutes=60, days=7, deadline="2026-10-05",
                                             hours="8-17"))
    starts = [e["start"][11:16] for e in result.evidence]
    assert starts == ["08:00", "12:00"]                    # Monday only, around 09–12


def test_an_unreadable_deadline_is_a_question_not_a_guess(assistant):
    result = asyncio.run(assistant.find_time(minutes=60, deadline="next-ish"))
    assert not result.ok and "couldn't read the deadline" in result.text


def test_book_slot_schedules_the_chosen_option(assistant):
    asyncio.run(assistant.find_time(minutes=30, days=2, title="Write-up"))
    booked = asyncio.run(assistant.book_slot(2))
    assert booked.ok
    title, when = assistant.cognition.tasks[-1]
    # Option 1 is Monday after the workshop; option 2 the next free gap,
    # Tuesday at the start of the working day.
    assert title == "Write-up" and when == "2026-10-06 09:00"
    again = asyncio.run(assistant.book_slot(1))
    assert not again.ok                                    # proposals are spent


def test_with_no_calendar_the_answer_says_so(monkeypatch):
    from orion_core.executive import ExecutiveAssistantMode
    from orion_core.time_service import TIME

    TIME.set_clock(lambda: MONDAY)
    try:
        monkeypatch.setattr(cs, "configured_feeds", lambda *a: [])
        mode = ExecutiveAssistantMode(_Bus(), _Memory(), _Cognition())
        result = asyncio.run(mode.find_time(minutes=60, days=1))
        assert "No calendar could be read" in result.text
    finally:
        TIME.set_clock(None)


@pytest.mark.parametrize("text, expected", [
    ("8-17", (8, 17)), ("08:00–17:30", (8, 17)), ("9 to 6pm", (9, 18)),
    ("9 to 6", (9, 18)), ("", None), ("whenever", None)])
def test_working_hours_parsing(text, expected):
    from orion_core.executive import _parse_hours
    assert _parse_hours(text) == expected
