"""Reminder inputs, elapsed-time scheduling and actual delivery outcomes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from orion_core.dispatch_productivity import ProductivityDispatchMixin
from orion_core.reminders import ReminderService
from orion_core.telephony import Channel, Dispatch
from orion_core.time_service import TIME


def london_2026():
    """Real 2026 UK transitions without depending on optional Windows tzdata."""
    transitions = [int(datetime(2026, month, day, 1, tzinfo=timezone.utc).timestamp())
                   for month, day in ((3, 29), (10, 25))]
    header = b"TZif\x00" + b"\x00" * 15 + struct.pack(">6l", 0, 0, 0, 2, 2, 8)
    records = struct.pack(">2l", *transitions) + bytes((1, 0))
    records += struct.pack(">lbb", 0, 0, 0) + struct.pack(">lbb", 3600, 1, 4)
    return ZoneInfo.from_file(BytesIO(header + records + b"GMT\x00BST\x00"), key="UK-2026")


class Signal:
    def __init__(self):
        self.sent = []

    def emit(self, *args):
        self.sent.append(args)


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(TIME, "_clock", lambda: datetime(2026, 9, 27, 12, tzinfo=timezone.utc))
    bus = SimpleNamespace(**{name: Signal() for name in (
        "banner", "log", "speak_request", "dashboard_event", "phone_action")})
    return ReminderService(bus)


@pytest.mark.parametrize("clock", ["24:00", "25:00", "99:00", "10:60", "13pm", "0am",
                                    "3:7", "3:000", "3pmjunk", "tomorrow 3pm"])
def test_invalid_clock_does_not_schedule(service, clock):
    result = service.add(text="check the oven", at=clock)
    assert not result.ok
    assert service.active() == []
    assert service._counter == 0
    assert service.bus.banner.sent == []


@pytest.mark.parametrize("clock", ["25:00", "10:60", "13pm", "3:7", "3pmjunk", "3 pmjunk", "3 :07"])
def test_invalid_clock_in_phrase_does_not_schedule(service, clock):
    assert not service.add(phrase=f"remind me at {clock} to check the oven").ok
    assert service.active() == []


@pytest.mark.parametrize("minutes", ["nonsense", [], float("nan"), float("inf"),
                                      -float("inf"), 1e300, 0, -1])
def test_invalid_delay_returns_tool_failure_without_mutating_state(service, minutes):
    dispatcher = SimpleNamespace(reminders=service)
    result = ProductivityDispatchMixin.reminder_tool(
        dispatcher, {"action": "add", "text": "check the oven", "minutes": minutes})
    assert not result.ok
    assert service.active() == []
    assert service._counter == 0


@pytest.mark.parametrize("clock,seconds", [("12am", 12 * 3600), ("12pm", 24 * 3600),
                                           ("15:30", 3.5 * 3600), ("at 3PM", 3 * 3600)])
def test_valid_clock_inputs_keep_their_meaning(service, clock, seconds):
    assert service._seconds_until_clock(clock) == seconds


@pytest.mark.parametrize("now,clock,seconds", [
    (datetime(2026, 3, 29, 0, 30), "3am", 90 * 60),
    (datetime(2026, 10, 25, 0, 30), "3am", 210 * 60),
    (datetime(2026, 10, 25, 1, 45), "1:30am", 45 * 60),
    (datetime(2026, 10, 25, 1, 45, fold=1), "1:30am", (23 * 60 + 45) * 60),
    (datetime(2026, 3, 29, 0, 30), "1:30am", 24 * 3600),
])
def test_absolute_deadlines_measure_elapsed_time_across_dst(service, monkeypatch, now, clock, seconds):
    local = now.replace(tzinfo=london_2026())
    monkeypatch.setattr(TIME, "_clock", lambda: local)
    assert service._seconds_until_clock(clock) == seconds


def test_relative_delay_reports_the_actual_wall_time_across_dst(service, monkeypatch):
    now = datetime(2026, 3, 29, 0, 30, tzinfo=london_2026())
    monkeypatch.setattr(TIME, "_clock", lambda: now)
    assert service.add(text="check the oven", minutes=90).ok
    assert service.active()[0].wall_due == "03:00"


@pytest.mark.parametrize("now,phrase,expected", [
    (datetime(2026, 10, 5, 11), "on Monday at 10", datetime(2026, 10, 12, 10)),
    (datetime(2026, 10, 5, 9), "on Monday at 10", datetime(2026, 10, 5, 10)),
    (datetime(2026, 10, 5, 9), "next Monday at 10", datetime(2026, 10, 12, 10)),
    (datetime(2026, 10, 5, 11), "on Tuesday at 10", datetime(2026, 10, 6, 10)),
    (datetime(2026, 10, 5, 11), "tomorrow at 10", datetime(2026, 10, 6, 10)),
    (datetime(2026, 12, 31, 11), "tomorrow at 10", datetime(2027, 1, 1, 10)),
])
def test_named_weekdays_and_relative_days_keep_the_requested_date(service, monkeypatch, now, phrase, expected):
    now = now.replace(tzinfo=timezone.utc)
    monkeypatch.setattr(TIME, "_clock", lambda: now)
    body, delay = service._parse(f"remind me {phrase} to call my mother", None, "")
    assert body == "call my mother"
    assert datetime.fromtimestamp(now.timestamp() + delay, timezone.utc) == expected.replace(tzinfo=timezone.utc)


@pytest.mark.parametrize("now,offset,expected", [
    (datetime(2026, 3, 28, 12), 1, None),
    (datetime(2026, 10, 25, 1, 45), 0, 45 * 60),
    (datetime(2026, 10, 25, 1, 45, fold=1), 0, -15 * 60),
])
def test_explicit_days_validate_gaps_and_select_the_future_fold(service, monkeypatch, now, offset, expected):
    now = now.replace(tzinfo=london_2026())
    monkeypatch.setattr(TIME, "_clock", lambda: now)
    assert service._seconds_until_day(offset, 1, 30) == expected


def test_a_passed_explicit_today_is_rejected_instead_of_becoming_tomorrow(service):
    result = service.add(phrase="remind me today at 10 to check the oven")
    assert not result.ok and "passed" in result.text
    assert not service.active()


@pytest.mark.parametrize("clock", ["13pm", "0am", "24am"])
def test_an_invalid_meridiem_is_not_accepted_with_a_day_hint(service, clock):
    assert not service.add(phrase=f"remind me tomorrow at {clock} to check the oven").ok
    assert not service.active()


def test_dispatcher_keeps_explicit_delivery_channel_and_recipient(service):
    dispatcher = SimpleNamespace(reminders=service)
    result = ProductivityDispatchMixin.reminder_tool(dispatcher, {
        "action": "add", "text": "check the oven", "minutes": 1,
        "channel": "SMS", "recipient": "Alex"})
    assert result.ok
    reminder = service.active()[0]
    assert (reminder.channel, reminder.recipient) == ("SMS", "Alex")


@pytest.mark.parametrize("body", ["bring medicine", "recall memories", "check text messages"])
def test_words_inside_the_subject_do_not_choose_a_remote_channel(service, body):
    assert service.add(phrase=f"remind me in 5 minutes to {body}").ok
    reminder = service.active()[0]
    assert reminder.channel == "DESKTOP"
    assert reminder.text == body


@pytest.mark.parametrize("channel", [Channel.SMS, Channel.VOICE_CALL])
@pytest.mark.parametrize("outcome", ["success", "refused", "exception"])
def test_remote_delivery_observes_the_result_before_discarding_local_fallback(service, channel, outcome):
    async def scenario():
        method = AsyncMock(return_value=Dispatch(outcome == "success", channel, outcome))
        if outcome == "exception":
            method.side_effect = RuntimeError("transport unavailable")
        service.telephony = SimpleNamespace(call=method, text=method)
        assert service.add(text="check the oven", minutes=1, channel=channel, recipient="Alex").ok
        service._fire(service.active()[0])
        # Complete the task on this test's loop; no actual phone gateway runs.
        tasks = [task for task in asyncio.all_tasks()
                 if task.get_name() == "orion-reminder-dispatch"]
        assert len(tasks) == 1
        await asyncio.gather(*tasks)
        method.assert_awaited_once()
        assert method.await_args.args[0] == "Alex"
        spoken = service.bus.speak_request.sent
        assert spoken == ([] if outcome == "success" else [("A reminder: check the oven.",)])
        assert len(service.bus.dashboard_event.sent) == 1

    asyncio.run(scenario())


def test_remote_delivery_without_an_event_loop_falls_back_locally(service):
    service.telephony = SimpleNamespace(call=AsyncMock())
    service.add(text="check the oven", minutes=1, channel="VOICE_CALL")
    service._fire(service.active()[0])
    assert service.bus.speak_request.sent == [("A reminder: check the oven.",)]
    service.telephony.call.assert_not_awaited()
