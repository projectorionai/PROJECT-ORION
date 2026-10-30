"""
Reminder times use ORION's clock (TimeService), not the machine's.

ORION tells the time in his own zone; a machine in another zone (a UTC cloud
node) made "remind me at 3pm" fire at 3pm machine time - an hour out - and
said back a due time from the wrong clock.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.reminders import ReminderService
from orion_core.time_service import TIME

ORION_ZONE = timezone(timedelta(hours=1))


class _Signal:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


@pytest.fixture()
def orion_at_1411():
    TIME.set_clock(lambda: datetime(2026, 9, 25, 14, 11, tzinfo=ORION_ZONE))
    yield
    TIME.set_clock(None)


def test_at_3pm_counts_down_on_orions_clock(orion_at_1411):
    service = ReminderService.__new__(ReminderService)
    seconds = service._seconds_until_clock("at 3pm")
    assert seconds == pytest.approx(49 * 60, abs=1)


def test_a_time_already_passed_today_is_tomorrow(orion_at_1411):
    service = ReminderService.__new__(ReminderService)
    seconds = service._seconds_until_clock("at 9am")
    assert seconds == pytest.approx((24 * 60 - 5 * 60 - 11) * 60, abs=1)


@pytest.mark.parametrize("phrase,seconds,body", [
    ("remind me at 3 to call mum", 40 * 60, "call mum"),
    ("remind me at quarter past 4 to leave", 115 * 60, "leave"),
    ("remind me at 11 to go to bed", 520 * 60, "go to bed"),
    ("remind me at 9 to take my tablets", 400 * 60, "take my tablets"),
    ("remind me at 2 to water the plants", 1420 * 60, "water the plants"),
    ("wake me up at half 6", 970 * 60, "your alarm"),
    ("remind me in 5 to check the oven", 300, "check the oven"),
    ("remind me in five to check the oven", 300, "check the oven"),
    ("remind me tomorrow at 3 to call mum", 1480 * 60, "call mum"),
    ("remind me at 03:00 to leave", 760 * 60, "leave"),
])
def test_spoken_time_uses_context_before_resolving_deadline(monkeypatch, phrase, seconds, body):
    monkeypatch.setattr(TIME, "_clock", lambda: datetime(2026, 10, 7, 14, 20, tzinfo=ORION_ZONE))
    service = ReminderService.__new__(ReminderService)
    parsed_body, delay = service._parse(phrase, None, "")
    assert parsed_body == body
    assert delay == seconds


def test_noon_stays_noon_and_unitless_delay_does_not_consume_subject(monkeypatch):
    monkeypatch.setattr(TIME, "_clock", lambda: datetime(2026, 10, 7, 23, 30, tzinfo=ORION_ZONE))
    service = ReminderService.__new__(ReminderService)
    assert service._parse("remind me at noon to eat", None, "") == ("eat", 12.5 * 3600)
    assert service._parse("remind me in 5 chapters to check the ending", None, "")[1] is None
    assert service._parse("remind me in 5 days to call mum", None, "") == ("call mum", 5 * 86400)


def test_structured_clock_keeps_waking_context(monkeypatch):
    monkeypatch.setattr(TIME, "_clock", lambda: datetime(2026, 10, 7, 14, 20, tzinfo=ORION_ZONE))
    service = ReminderService.__new__(ReminderService)
    assert service._parse("wake me up", None, "6:30")[1] == 970 * 60
