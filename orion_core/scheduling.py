"""
Conflict-aware scheduling (Mark X.12 §2.4).

Calendar integration today only *reads* and *creates* events. This adds the
missing piece consistent with `momentum.py`'s shipping-coach framing:
"find me 90 minutes this week for X" cross-references busy intervals (Notion
calendar + Outlook calendar + tracked deadlines) and proposes free slots inside
working hours, respecting an optional deadline — it never books anything itself
(the caller keeps it CONFIRM-tier: propose, the user confirms).

This module is the pure engine — merge busy intervals, carve working-hour
windows, subtract the busy, and emit aligned candidate slots. Where the busy
intervals come from (which calendars) and how a chosen slot is booked are the
integration on top; everything here is deterministic and unit-tested.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

# Monday=0 … Sunday=6; the default working week.
DEFAULT_WORKDAYS = frozenset({0, 1, 2, 3, 4})
DEFAULT_WORK_HOURS = (9, 18)          # 09:00–18:00 local
DEFAULT_GRANULARITY = timedelta(minutes=15)


def _instant(when: datetime) -> datetime:
    return when.astimezone(timezone.utc) if when.utcoffset() is not None else when


def _require_same_awareness(*values: datetime) -> None:
    if len({value.utcoffset() is not None for value in values}) > 1:
        raise ValueError("Scheduling dates must either all include timezones or all be naive")


def _local_boundary(wall: datetime, *, closing: bool = False) -> datetime:
    """Use the full repeated hour; move missing work-hour boundaries forward."""
    if wall.utcoffset() is None:
        return wall
    candidates = [wall.replace(fold=fold).astimezone(timezone.utc).astimezone(wall.tzinfo)
                  for fold in (0, 1)]
    valid = [candidate for candidate in candidates
             if candidate.replace(tzinfo=None) == wall.replace(tzinfo=None)]
    if valid:
        return (max if closing else min)(valid, key=_instant)
    return max(candidates, key=lambda candidate: candidate.replace(tzinfo=None))


@dataclass(frozen=True)
class TimeSlot:
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        _require_same_awareness(self.start, self.end)
        return _instant(self.end) - _instant(self.start)

    def overlaps(self, other: "TimeSlot") -> bool:
        _require_same_awareness(self.start, self.end, other.start, other.end)
        return _instant(self.start) < _instant(other.end) and _instant(other.start) < _instant(self.end)


def _merge(intervals: Iterable[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Merge overlapping/adjacent (start, end) intervals into a minimal set."""
    normalised = [(_instant(s), _instant(e)) for s, e in intervals]
    ordered = sorted((s, e) for s, e in normalised if s < e)
    merged: list[tuple[datetime, datetime]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _working_windows(window_start: datetime, window_end: datetime,
                     work_hours: tuple[int, int], workdays: frozenset[int]
                     ) -> list[tuple[datetime, datetime]]:
    """The working-hour windows within [window_start, window_end], one per
    eligible day, clipped to the search window."""
    start_hour, end_hour = work_hours
    windows: list[tuple[datetime, datetime]] = []
    day = window_start.replace(hour=0, minute=0, second=0, microsecond=0)
    last = window_end.astimezone(window_start.tzinfo) if window_start.utcoffset() is not None else window_end
    while day.date() <= last.date():
        if day.weekday() in workdays:
            day_open = _local_boundary(day.replace(hour=start_hour))
            if end_hour == 24:
                try:
                    day_close = _local_boundary(day + timedelta(days=1), closing=True)
                except OverflowError:
                    day_close = last
            else:
                day_close = _local_boundary(day.replace(hour=end_hour), closing=True)
            lo = max(_instant(day_open), _instant(window_start))
            hi = min(_instant(day_close), _instant(window_end))
            if lo < hi:
                windows.append((lo, hi))
        if day.date() == last.date():
            break
        day += timedelta(days=1)
    return windows


def _subtract(window: tuple[datetime, datetime],
             busy: Sequence[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Subtract sorted, merged busy intervals without scanning the calendar.

    Disjoint intervals have increasing ends. Locate the first possible
    overlap, then stop at the first interval beyond this working window.
    """
    free: list[tuple[datetime, datetime]] = []
    cursor, end = window
    first = bisect_right(busy, cursor, key=lambda interval: interval[1])
    for index in range(first, len(busy)):
        b_start, b_end = busy[index]
        if b_start >= end:
            break
        if b_start > cursor:
            free.append((cursor, min(b_start, end)))
        cursor = max(cursor, b_end)
        if cursor >= end:
            break
    if cursor < end:
        free.append((cursor, end))
    return free


def _ceil_to(dt: datetime, granularity: timedelta) -> datetime:
    day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    # An elapsed-time grid from local midnight includes both autumn folds and
    # never manufactures a missing spring clock time.
    origin = _instant(_local_boundary(day_start))
    steps = math.ceil((_instant(dt) - origin) / granularity)
    aligned = origin + steps * granularity
    return aligned.astimezone(dt.tzinfo) if dt.utcoffset() is not None else aligned


def free_intervals(busy: Iterable[tuple[datetime, datetime]],
                   window_start: datetime, window_end: datetime, *,
                   work_hours: tuple[int, int] = DEFAULT_WORK_HOURS,
                   workdays: frozenset[int] = DEFAULT_WORKDAYS
                   ) -> list[TimeSlot]:
    """Free stretches in the start date's timezone; busy intervals use instants."""
    _require_same_awareness(window_start, window_end)
    start_hour, end_hour = work_hours
    if (type(start_hour) is not int or type(end_hour) is not int
            or not 0 <= start_hour < end_hour <= 24):
        raise ValueError("Working hours must be increasing whole hours between 0 and 24")
    if any(type(day) is not int or not 0 <= day <= 6 for day in workdays):
        raise ValueError("Workdays must be weekday integers from 0 to 6")
    intervals = list(busy)
    for start, end in intervals:
        _require_same_awareness(window_start, start, end)
    if _instant(window_start) >= _instant(window_end):
        return []
    merged = _merge(intervals)
    result: list[TimeSlot] = []
    for window in _working_windows(window_start, window_end, work_hours, workdays):
        for lo, hi in _subtract(window, merged):
            if lo < hi:
                if window_start.utcoffset() is not None:
                    lo, hi = lo.astimezone(window_start.tzinfo), hi.astimezone(window_start.tzinfo)
                result.append(TimeSlot(lo, hi))
    return result


def find_slots(busy: Iterable[tuple[datetime, datetime]],
               window_start: datetime, window_end: datetime,
               duration: timedelta, *,
               work_hours: tuple[int, int] = DEFAULT_WORK_HOURS,
               workdays: frozenset[int] = DEFAULT_WORKDAYS,
               granularity: timedelta = DEFAULT_GRANULARITY,
               deadline: Optional[datetime] = None,
               limit: Optional[int] = 3) -> list[TimeSlot]:
    """Propose up to *limit* free slots of *duration*, earliest first, spread one
    per free gap so the options span the window rather than clustering. Slots are
    aligned to elapsed steps after local midnight and never end after *deadline*.
    Durations are elapsed time, including on days when the clocks change."""
    if granularity <= timedelta(0):
        raise ValueError("Scheduling granularity must be positive")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("Scheduling limit must be a non-negative integer or None")
    _require_same_awareness(window_start, window_end, *([deadline] if deadline else []))
    if limit == 0:
        return []
    if duration <= timedelta(0):
        return []
    hard_end = min(window_end, deadline, key=_instant) if deadline else window_end
    proposals: list[TimeSlot] = []
    for gap in free_intervals(busy, window_start, hard_end,
                              work_hours=work_hours, workdays=workdays):
        try:
            start = _ceil_to(gap.start, granularity)
            end = _instant(start) + duration
        except OverflowError:
            continue
        if start.utcoffset() is not None:
            end = end.astimezone(start.tzinfo)
        if _instant(end) <= _instant(gap.end):
            proposals.append(TimeSlot(start, end))
            if limit is not None and len(proposals) >= limit:
                break
    return proposals
