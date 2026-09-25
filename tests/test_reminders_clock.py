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
