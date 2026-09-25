"""
Plugins that run on their own, without being asked.

Every plugin ORION has runs only when something calls it. That makes anything
recurring — check the news each morning, sweep the inbox at six, compile a
dossier overnight — depend on the user remembering to ask, which is precisely
the work an assistant is supposed to take away.

So a manifest may now declare when it wants to run:

    "schedule": "cron(0 6 * * *)"      06:00 every day
    "schedule": "cron(*/15 * * * *)"   every quarter of an hour
    "schedule": "cron(0 9 * * 1-5)"    09:00 on weekdays
    "interval_seconds": 1800           every half hour, from start-up

Wall-clock, deliberately
------------------------
Cron fields are matched against local wall-clock time, not against elapsed
monotonic time, because "six in the morning" has to mean six in the morning
after the clocks change. The consequence is that a missed minute is a missed
run rather than a run fired late in a burst: if the machine was asleep at 06:00
the 06:00 job does not go off at 09:14 when it wakes. A briefing delivered
three hours late is worse than one not delivered, and the catch-up path is
``catch_up``, which exists and is honest about the gap.

The one subtlety in cron
------------------------
When both day-of-month and day-of-week are restricted, standard cron runs the
job when **either** matches, not both. ``0 0 13 * 5`` is "midnight on the 13th,
and also every Friday" — not "Friday the 13th". Getting this backwards is the
classic cron bug, and it is silent: the job simply runs less often than anyone
expected and nobody notices for a month.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

#: Fields, in cron's order, with the values each may take.
_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day", 1, 31),
    ("month", 1, 12),
    ("weekday", 0, 6),          # 0 = Sunday, as crontab(5) has it
)

#: Names people write instead of numbers.
_NAMES: dict[str, int] = {
    "sun": 0, "mon": 1, "tue": 2, "tues": 2, "wed": 3, "thu": 4, "thur": 4,
    "thurs": 4, "fri": 5, "sat": 6,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

#: Shorthands cron accepts in place of five fields.
_ALIASES: dict[str, str] = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
}

#: How far ahead ``next_due`` will look before giving up. Four years covers a
#: leap-day-only schedule; beyond that the expression matches nothing real and
#: saying so beats looping.
_SEARCH_LIMIT_MINUTES = 366 * 4 * 24 * 60


class ScheduleError(ValueError):
    """A schedule that cannot mean anything. Carries a sentence for the user."""


def _expand(spec: str, low: int, high: int, field: str) -> set[int]:
    """One cron field as the set of values it matches."""
    values: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip().lower()
        if not part:
            raise ScheduleError(f"{field}: empty value in {spec!r}")

        step = 1
        if "/" in part:
            part, _, step_text = part.partition("/")
            if not step_text.isdigit() or int(step_text) < 1:
                raise ScheduleError(f"{field}: step must be a positive number")
            step = int(step_text)
            part = part.strip() or "*"

        if part == "*":
            start, end = low, high
        elif "-" in part.lstrip("-"):
            start_text, _, end_text = part.partition("-")
            start, end = _value(start_text, field), _value(end_text, field)
        else:
            start = end = _value(part, field)

        if start > end:
            # "fri-mon" and "22-2" wrap round the end of the range, which is
            # what someone writing them means.
            wrapped = list(range(start, high + 1)) + list(range(low, end + 1))
            values.update(wrapped[::step])
            continue
        if not (low <= start <= high and low <= end <= high):
            raise ScheduleError(
                f"{field}: {part!r} is outside {low}-{high}")
        values.update(range(start, end + 1, step))

    if not values:
        raise ScheduleError(f"{field}: {spec!r} matches nothing")
    return values


def _value(text: str, field: str) -> int:
    text = text.strip().lower()
    if text in _NAMES:
        return _NAMES[text]
    try:
        number = int(text)
    except ValueError:
        raise ScheduleError(f"{field}: {text!r} is not a number or a name")
    # Both 0 and 7 mean Sunday in crontab(5).
    if field == "weekday" and number == 7:
        return 0
    return number


@dataclass(frozen=True)
class Schedule:
    """When a plugin wants to run.

    Either a cron expression or a fixed interval — never both, because two
    answers to "when next?" is a bug waiting to be argued about.
    """

    expression: str = ""
    interval_seconds: float = 0.0
    minutes: frozenset[int] = frozenset()
    hours: frozenset[int] = frozenset()
    days: frozenset[int] = frozenset()
    months: frozenset[int] = frozenset()
    weekdays: frozenset[int] = frozenset()
    #: Whether day-of-month and day-of-week were BOTH restricted. See the
    #: module docstring: cron then means "either", not "both".
    day_or_weekday: bool = False

    @property
    def is_interval(self) -> bool:
        return self.interval_seconds > 0

    def __str__(self) -> str:
        if self.is_interval:
            return f"every {_readable_interval(self.interval_seconds)}"
        return f"cron({self.expression})"

    # ── matching ─────────────────────────────────────────────────────────────

    def matches(self, when: datetime) -> bool:
        """Whether a cron schedule fires in *when*'s minute."""
        if self.is_interval:
            return False
        if when.minute not in self.minutes or when.hour not in self.hours:
            return False
        if when.month not in self.months:
            return False
        # crontab(5): Python's Monday=0 needs mapping to cron's Sunday=0.
        weekday = (when.weekday() + 1) % 7
        day_ok = when.day in self.days
        weekday_ok = weekday in self.weekdays
        if self.day_or_weekday:
            return day_ok or weekday_ok
        return day_ok and weekday_ok

    def next_due(self, after: datetime) -> datetime | None:
        """The next wall-clock minute this fires, strictly after *after*.

        Searched minute by minute from the next whole minute. Four years is
        enough for a 29-February-only schedule; past that the expression
        matches nothing real, and returning None says so rather than looping.
        """
        if self.is_interval:
            return after + timedelta(seconds=self.interval_seconds)
        moment = (after.replace(second=0, microsecond=0)
                  + timedelta(minutes=1))
        for _ in range(_SEARCH_LIMIT_MINUTES):
            if self.matches(moment):
                return moment
            moment += timedelta(minutes=1)
        return None


def _readable_interval(seconds: float) -> str:
    seconds = int(seconds)
    if seconds % 3600 == 0 and seconds >= 3600:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''}"
    if seconds % 60 == 0 and seconds >= 60:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{seconds} second{'s' if seconds != 1 else ''}"


_CRON_CALL = re.compile(r"\Acron\s*\((?P<body>.*)\)\Z", re.IGNORECASE | re.S)


def parse_cron(expression: str) -> Schedule:
    """A cron expression as a :class:`Schedule`.

    Accepts the five-field form with or without a ``cron(...)`` wrapper, and
    the ``@daily`` family.
    """
    text = str(expression or "").strip()
    match = _CRON_CALL.match(text)
    if match:
        text = match.group("body").strip()
    if not text:
        raise ScheduleError("the schedule is empty")

    alias = _ALIASES.get(text.lower())
    if alias:
        text = alias

    parts = text.split()
    if len(parts) != 5:
        raise ScheduleError(
            f"a cron schedule needs 5 fields "
            f"(minute hour day month weekday), got {len(parts)}: {text!r}")

    sets: list[frozenset[int]] = []
    for spec, (field, low, high) in zip(parts, _FIELDS):
        sets.append(frozenset(_expand(spec, low, high, field)))

    day_restricted = parts[2].strip() != "*"
    weekday_restricted = parts[4].strip() != "*"
    return Schedule(
        expression=text,
        minutes=sets[0], hours=sets[1], days=sets[2],
        months=sets[3], weekdays=sets[4],
        day_or_weekday=day_restricted and weekday_restricted,
    )


def parse_schedule(schedule: Any = "", interval_seconds: Any = 0) -> Schedule | None:
    """Read whichever of the two a manifest declared. ``None`` if neither.

    Declaring both is refused rather than silently preferring one: two answers
    to "when does this run?" is a disagreement someone will lose an afternoon
    to.
    """
    text = str(schedule or "").strip()
    try:
        interval = float(interval_seconds or 0)
    except (TypeError, ValueError):
        raise ScheduleError("'interval_seconds' must be a number")

    if text and interval > 0:
        raise ScheduleError(
            "declare either 'schedule' or 'interval_seconds', not both")
    if interval > 0:
        if interval < 30:
            raise ScheduleError(
                "'interval_seconds' must be at least 30 — anything faster is "
                "a loop, not a schedule")
        return Schedule(interval_seconds=interval)
    if text:
        return parse_cron(text)
    return None


def describe(schedule: Schedule | None) -> str:
    """A plain-English line for the log and the plugin list."""
    if schedule is None:
        return "on request only"
    if schedule.is_interval:
        return f"every {_readable_interval(schedule.interval_seconds)}"
    upcoming = schedule.next_due(datetime.now())
    when = upcoming.strftime("%a %d %b %H:%M") if upcoming else "never again"
    return f"cron({schedule.expression}) — next {when}"


__all__ = [
    "Schedule", "ScheduleError", "describe", "parse_cron", "parse_schedule",
]
