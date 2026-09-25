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
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence

# Monday=0 … Sunday=6; the default working week.
DEFAULT_WORKDAYS = frozenset({0, 1, 2, 3, 4})
DEFAULT_WORK_HOURS = (9, 18)          # 09:00–18:00 local
DEFAULT_GRANULARITY = timedelta(minutes=15)


@dataclass(frozen=True)
class TimeSlot:
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def overlaps(self, other: "TimeSlot") -> bool:
        return self.start < other.end and other.start < self.end


def _merge(intervals: Iterable[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Merge overlapping/adjacent (start, end) intervals into a minimal set."""
    ordered = sorted((s, e) for s, e in intervals if s < e)
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
    last = window_end
    while day <= last:
        if day.weekday() in workdays:
            day_open = day.replace(hour=start_hour)
            day_close = day.replace(hour=end_hour)
            lo = max(day_open, window_start)
            hi = min(day_close, window_end)
            if lo < hi:
                windows.append((lo, hi))
        day += timedelta(days=1)
    return windows


def _subtract(window: tuple[datetime, datetime],
             busy: Sequence[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Return the parts of *window* not covered by any *busy* interval."""
    free: list[tuple[datetime, datetime]] = []
    cursor, end = window
    for b_start, b_end in busy:
        if b_end <= cursor or b_start >= end:
            continue
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
    steps = math.ceil((dt - day_start) / granularity)
    return day_start + steps * granularity


def free_intervals(busy: Iterable[tuple[datetime, datetime]],
                   window_start: datetime, window_end: datetime, *,
                   work_hours: tuple[int, int] = DEFAULT_WORK_HOURS,
                   workdays: frozenset[int] = DEFAULT_WORKDAYS
                   ) -> list[TimeSlot]:
    """Every free stretch inside working hours across the search window."""
    merged = _merge(busy)
    result: list[TimeSlot] = []
    for window in _working_windows(window_start, window_end, work_hours, workdays):
        for lo, hi in _subtract(window, merged):
            if lo < hi:
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
    aligned up to *granularity* and never end after *deadline*."""
    if duration <= timedelta(0):
        return []
    hard_end = min(window_end, deadline) if deadline else window_end
    proposals: list[TimeSlot] = []
    for gap in free_intervals(busy, window_start, hard_end,
                              work_hours=work_hours, workdays=workdays):
        start = _ceil_to(gap.start, granularity)
        end = start + duration
        if end <= gap.end and (deadline is None or end <= deadline):
            proposals.append(TimeSlot(start, end))
            if limit is not None and len(proposals) >= limit:
                break
    return proposals
