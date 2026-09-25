"""Tests for the conflict-aware scheduling engine (Mark X.12 §2.4)."""

from __future__ import annotations

from datetime import datetime, timedelta

from orion_core.scheduling import (
    TimeSlot,
    find_slots,
    free_intervals,
)

# A fixed reference week: Mon 2026-07-20 … Fri 2026-07-24.
MON = datetime(2026, 7, 20, 0, 0)      # a Monday
TUE = MON + timedelta(days=1)
SAT = MON + timedelta(days=5)


def _at(day, hour, minute=0):
    return day.replace(hour=hour, minute=minute)


def test_free_intervals_respects_working_hours_and_busy():
    busy = [(_at(MON, 10), _at(MON, 11))]      # a 10–11 meeting on Monday
    free = free_intervals(busy, _at(MON, 0), _at(MON, 23, 59))
    # Working day is 09–18, split by the meeting into 09–10 and 11–18.
    assert TimeSlot(_at(MON, 9), _at(MON, 10)) in free
    assert TimeSlot(_at(MON, 11), _at(MON, 18)) in free
    # Nothing before 09 or after 18.
    assert all(s.start.hour >= 9 and s.end.hour <= 18 for s in free)


def test_weekends_are_excluded_by_default():
    free = free_intervals([], _at(SAT, 0), _at(SAT, 23, 59))
    assert free == []          # Saturday is not a default workday


def test_find_slots_proposes_earliest_first_one_per_gap():
    busy = [
        (_at(MON, 9), _at(MON, 17)),      # Monday almost fully booked
        (_at(TUE, 9), _at(TUE, 10)),      # Tuesday 9–10 busy
    ]
    slots = find_slots(busy, _at(MON, 0), _at(TUE, 23, 59),
                       duration=timedelta(minutes=90), limit=3)
    # Monday's only free stretch (17–18) is too short for 90m → skipped.
    # Tuesday's first fit is 10:00–11:30.
    assert slots[0] == TimeSlot(_at(TUE, 10), _at(TUE, 11, 30))
    assert all(s.duration == timedelta(minutes=90) for s in slots)


def test_slots_are_aligned_to_granularity():
    busy = [(_at(MON, 9), _at(MON, 9, 20))]    # frees at 09:20
    slots = find_slots(busy, _at(MON, 0), _at(MON, 23, 59),
                       duration=timedelta(minutes=30),
                       granularity=timedelta(minutes=15), limit=1)
    # 09:20 rounds up to 09:30 (next 15-min boundary).
    assert slots[0].start == _at(MON, 9, 30)


def test_deadline_excludes_later_slots():
    slots = find_slots([], _at(MON, 0), _at(TUE, 23, 59),
                       duration=timedelta(hours=1),
                       deadline=_at(MON, 10), limit=5)
    # Only a slot that ENDS by Monday 10:00 qualifies → 09:00–10:00.
    assert slots == [TimeSlot(_at(MON, 9), _at(MON, 10))]


def test_no_slot_when_everything_is_busy():
    busy = [(_at(MON, 9), _at(MON, 18))]
    slots = find_slots(busy, _at(MON, 0), _at(MON, 23, 59),
                       duration=timedelta(minutes=30))
    assert slots == []


def test_zero_or_negative_duration_yields_nothing():
    assert find_slots([], _at(MON, 0), _at(TUE, 0), timedelta(0)) == []


def test_limit_none_returns_all_gaps_that_fit():
    busy = [(_at(MON, 12), _at(MON, 13)), (_at(TUE, 12), _at(TUE, 13))]
    slots = find_slots(busy, _at(MON, 0), _at(TUE, 23, 59),
                       duration=timedelta(minutes=30), limit=None)
    # Free gaps: Mon 9–12, Mon 13–18, Tue 9–12, Tue 13–18 → 4 proposals.
    assert len(slots) == 4
