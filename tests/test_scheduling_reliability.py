"""Scheduling must preserve actual durations and avoid busy instants at DST."""
import asyncio
from datetime import datetime, timedelta, timezone
from io import BytesIO
import struct
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from orion_core.data import ToolResult
from orion_core.plugin_manifest import PluginManifest
from orion_core.plugin_runner import PluginScheduler
from orion_core.scheduling import TimeSlot, find_slots, free_intervals


def london_2026():
    transitions = [int(datetime(2026, month, day, 1, tzinfo=timezone.utc).timestamp())
                   for month, day in ((3, 29), (10, 25))]
    header = b"TZif\x00" + b"\x00" * 15 + struct.pack(">6l", 0, 0, 0, 2, 2, 8)
    records = struct.pack(">2l", *transitions) + bytes((1, 0))
    records += struct.pack(">lbb", 0, 0, 0) + struct.pack(">lbb", 3600, 1, 4)
    return ZoneInfo.from_file(BytesIO(header + records + b"GMT\x00BST\x00"), key="UK-2026")


def test_no_proposals_are_requested_with_a_zero_limit():
    day = datetime(2026, 10, 6)
    assert find_slots([], day, day + timedelta(days=1), timedelta(minutes=30), limit=0) == []


@pytest.mark.parametrize("granularity", [timedelta(0), timedelta(minutes=-15)])
def test_invalid_granularity_is_rejected_before_proposing_a_busy_slot(granularity):
    day = datetime(2026, 10, 6)
    busy = [(day.replace(hour=9), day.replace(hour=9, minute=20))]
    with pytest.raises(ValueError, match="granularity"):
        find_slots(busy, day, day + timedelta(days=1), timedelta(minutes=30), granularity=granularity)


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_limits_are_rejected(limit):
    day = datetime(2026, 10, 6)
    with pytest.raises(ValueError, match="limit"):
        find_slots([], day, day + timedelta(days=1), timedelta(minutes=30), limit=limit)


def test_duration_counts_elapsed_minutes_between_autumn_folds():
    zone = london_2026()
    slot = TimeSlot(datetime(2026, 10, 25, 1, 45, tzinfo=zone, fold=0),
                    datetime(2026, 10, 25, 1, 15, tzinfo=zone, fold=1))
    assert slot.duration == timedelta(minutes=30)


def test_busy_interval_spanning_autumn_fold_is_not_discarded_as_backwards():
    day = datetime(2026, 10, 25, tzinfo=london_2026())
    busy = TimeSlot(day.replace(hour=1, minute=45, fold=0), day.replace(hour=1, minute=15, fold=1))
    free = free_intervals([(busy.start, busy.end)], day, day.replace(hour=4),
                          work_hours=(0, 4), workdays=frozenset({6}))
    assert len(free) == 2
    assert all(not slot.overlaps(busy) for slot in free)


@pytest.mark.parametrize("month,day,hours", [(3, 29, 23), (10, 25, 25)])
def test_whole_local_day_has_the_actual_elapsed_length(month, day, hours):
    start = datetime(2026, month, day, tzinfo=london_2026())
    free = free_intervals([], start, start + timedelta(days=1),
                          work_hours=(0, 24), workdays=frozenset({6}))
    assert len(free) == 1
    assert free[0].duration == timedelta(hours=hours)


def test_requested_spring_duration_is_elapsed_time():
    day = datetime(2026, 3, 29, tzinfo=london_2026())
    slots = find_slots([], day, day.replace(hour=4), timedelta(hours=2),
                       work_hours=(0, 4), workdays=frozenset({6}), limit=1)
    assert slots == [TimeSlot(day, day.replace(hour=3))]
    assert slots[0].duration == timedelta(hours=2)


def test_repeated_clock_time_does_not_mean_overlapping_instants():
    day = datetime(2026, 10, 25, tzinfo=london_2026())
    first = TimeSlot(day.replace(hour=1, minute=10), day.replace(hour=1, minute=20))
    second = TimeSlot(day.replace(hour=1, minute=10, fold=1), day.replace(hour=1, minute=20, fold=1))
    assert not first.overlaps(second)


def test_fold_deadline_limits_the_actual_end_instant():
    day = datetime(2026, 10, 25, tzinfo=london_2026())
    start = day.replace(hour=1, minute=45)
    deadline = day.replace(hour=1, minute=15, fold=1)
    slots = find_slots([], start, day.replace(hour=3), timedelta(minutes=30),
                       work_hours=(0, 4), workdays=frozenset({6}), deadline=deadline)
    assert slots == [TimeSlot(start, deadline)]


def test_busy_dates_in_another_zone_are_compared_as_instants():
    day = datetime(2026, 7, 20, tzinfo=timezone(timedelta(hours=1)))
    busy = [(datetime(2026, 7, 20, 8, tzinfo=timezone.utc), datetime(2026, 7, 20, 9, tzinfo=timezone.utc))]
    slots = find_slots(busy, day, day.replace(hour=18), timedelta(hours=1))
    assert slots[0].start.hour == 10


def test_missing_work_hour_boundary_moves_to_real_spring_time():
    day = datetime(2026, 3, 29, tzinfo=london_2026())
    free = free_intervals([], day, day.replace(hour=4), work_hours=(1, 3), workdays=frozenset({6}))
    assert free == [TimeSlot(day.replace(hour=2), day.replace(hour=3))]
    assert free[0].duration == timedelta(hours=1)


def test_mixed_naive_and_aware_dates_get_a_clear_error():
    day = datetime(2026, 10, 6)
    with pytest.raises(ValueError, match="timezone"):
        find_slots([], day, day.replace(hour=18, tzinfo=timezone.utc), timedelta(minutes=30))


def test_datetime_upper_boundary_does_not_overflow_the_day_loop():
    free = free_intervals([], datetime(9999, 12, 31), datetime.max,
                          work_hours=(0, 24), workdays=frozenset(range(7)))
    assert free == [TimeSlot(datetime(9999, 12, 31), datetime.max)]


def test_plugin_runner_does_not_fire_the_second_fold_during_the_first():
    day = datetime(2026, 10, 25, tzinfo=london_2026())
    clock = [day.replace(hour=1, minute=45)]
    calls = []
    async def dispatch(name, _args):
        calls.append(name)
        return ToolResult("done")
    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), clock=lambda: clock[0])
    scheduled = scheduler.add(PluginManifest.from_dict({"name": "fold", "module": "fold.py", "schedule": "30 1 * * *"}))
    assert scheduled.next_due.fold == 1
    asyncio.run(scheduler._tick())
    assert calls == []
    clock[0] = day.replace(hour=1, minute=30, fold=1)
    asyncio.run(scheduler._tick())
    assert calls == ["fold"]


def test_subtraction_only_reads_busy_intervals_near_the_working_window():
    from orion_core.scheduling import _subtract

    origin = datetime(2026, 1, 1)
    intervals = [(origin + timedelta(hours=hour),
                  origin + timedelta(hours=hour, minutes=10)) for hour in range(4096)]

    class CountingBusy:
        def __init__(self):
            self.reads = 0

        def __len__(self):
            return len(intervals)

        def __getitem__(self, index):
            self.reads += 1
            return intervals[index]

    busy = CountingBusy()
    start, end = intervals[2048]
    window = (start - timedelta(minutes=5), end + timedelta(minutes=5))

    assert _subtract(window, busy) == [(window[0], start), (end, window[1])]
    assert busy.reads <= 32, f"one gap inspected {busy.reads} calendar entries"


def test_subtraction_keeps_half_open_boundaries_with_indexed_busy_lookup():
    from orion_core.scheduling import _subtract

    day = datetime(2026, 10, 6, tzinfo=timezone.utc)
    busy = [(day.replace(hour=8), day.replace(hour=9)),
            (day.replace(hour=10), day.replace(hour=11))]
    window = (day.replace(hour=9), day.replace(hour=10))

    assert _subtract(window, busy) == [window]
