"""
What time someone means when they don't say a.m. or p.m.

MEASURED, 7 Oct 2026 at 14:20, through the reminder parser:

    "remind me at 3 to call mum"            -> 03:00 tomorrow   (meant 15:00 today)
    "remind me at quarter past 4 to leave"  -> 04:15 tomorrow   (meant 16:15 today)
    "remind me at 11 to go to bed"          -> 11:00 tomorrow   (meant 23:00 tonight)
    "remind me at 9 to take my tablets"     -> 09:00 tomorrow   (the next 9 is 21:00 today)

A bare hour was read on the 24-hour clock and rolled to tomorrow once it had
passed. People don't talk like that: "at 3" in the afternoon is three this
afternoon. ``likely_hour`` picks between the two readings of a bare hour the
way a person would:

  * an explicit a.m./p.m. ("3pm", "in the morning", "tonight") is obeyed;
  * waking up, alarms and the like mean the morning; bed and "tonight" mean
    the evening;
  * otherwise the SOONEST reading that falls in waking hours (07:00-23:00),
    and failing that the soonest at all - so "at 2" at 14:20 is two tomorrow
    afternoon, not two in the morning.

Pure: no clock of its own, no I/O. The reminder parser and the reflex rules
pass ``now`` in.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

#: When a reminder is plausible without anything saying otherwise.
WAKING_HOURS = range(7, 23)
#: A reading this close is meant even at night: "at 1" at half eleven is in
#: ninety minutes, not lunchtime tomorrow.
SOON = timedelta(hours=3)

# "3pm" has no word boundary between the 3 and the pm, so a.m./p.m. are found
# by not being inside a word rather than by \b.
_PM = re.compile(r"(?<![a-z])p\.?m\b\.?|\b(in the (afternoon|evening)|this (afternoon|evening)|"
                 r"tonight|at night)\b", re.IGNORECASE)
_AM = re.compile(r"(?<![a-z])a\.?m\b\.?|\b(in the morning|this morning|tomorrow morning)\b",
                 re.IGNORECASE)
_MORNING_ACTS = re.compile(r"\b(wake|woken|waking|alarm|get up|getting up|up early|breakfast)\b",
                           re.IGNORECASE)
_EVENING_ACTS = re.compile(r"\b(bed|bedtime|sleep|tonight|dinner|supper)\b", re.IGNORECASE)


def readings(hour: int) -> list[int]:
    """The 24-hour readings of a spoken hour: 3 -> [3, 15]; 12 -> [12, 0];
    15 -> [15] (already unambiguous)."""
    if hour == 12:
        return [12, 0]
    if 1 <= hour <= 11:
        return [hour, hour + 12]
    return [hour % 24]


def next_at(hour24: int, minute: int, now: datetime) -> datetime:
    """The next moment the clock reads hour24:minute, from *now*."""
    target = now.replace(hour=hour24, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def likely_hour(hour: int, minute: int, now: datetime, text: str = "") -> int:
    """The 24-hour hour someone means by *hour* (as spoken) in *text*."""
    options = readings(int(hour))
    if len(options) == 1:
        return options[0]
    am, pm = options[0], options[1]             # for 12: noon, then midnight
    if int(hour) == 12:
        am, pm = 0, 12
    if _PM.search(text):
        return pm
    if _AM.search(text):
        return am
    if _MORNING_ACTS.search(text):
        return am
    if _EVENING_ACTS.search(text) and pm in range(18, 24):
        return pm
    upcoming = sorted(options, key=lambda h: next_at(h, minute, now))
    if next_at(upcoming[0], minute, now) - now <= SOON:
        return upcoming[0]
    for h in upcoming:
        if h in WAKING_HOURS:
            return h
    return upcoming[0]


def likely_moment(hour: int, minute: int, now: datetime, text: str = "") -> datetime:
    """The next occurrence of the time someone means (see likely_hour)."""
    return next_at(likely_hour(hour, minute, now, text), minute, now)


__all__ = ["WAKING_HOURS", "likely_hour", "likely_moment", "next_at", "readings"]
