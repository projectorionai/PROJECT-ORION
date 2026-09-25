"""
TimeService — the ONE authoritative source of "what time is it" for all of ORION.

Why this module exists
----------------------
Before it, seven modules each decided the time of day for themselves, with
three different band tables:

    live_worker.py   morning < 12, afternoon < 17, evening
    briefing.py      morning < 12, afternoon < 17, evening
    local_brain.py   morning < 12, afternoon < 18, evening
    temporal.py      morning < 12, afternoon < 17, evening
                     ...and, for the weather clause, "morning" if hour < 12
                     else "day" — with the season pulled from the month.

That last one is the bug the user actually hit: at 00:31 on an August night
``hour < 12`` is true, so ORION greeted them with "Good morning" and described
"a clear summer morning".  Midnight is not the morning and it is certainly not
daytime.  Worse, the three band tables disagreed at 17:00–17:59, so the HUD,
the briefing and the offline brain could each name a different part of the same
day.

Everything now derives from this module.  One clock, one set of bands, one
season, one answer.

Bands
-----
    night      22:00 – 04:59      (crosses midnight)
    morning    05:00 – 11:59
    afternoon  12:00 – 17:59
    evening    18:00 – 21:59

``time_of_day()`` names the real part of the day (so 00:31 is "night");
``greeting_period()`` is the narrower thing you *say* — English has no
"good night" that means hello, so the small hours greet as "evening".
Keeping them separate is what lets ORION say "Good evening" at 00:31 while
still knowing, everywhere else, that it is the middle of the night.

Timezone
--------
The user is in the UK, so BST/GMT and the transition days have to be right.
Windows' own local time already applies the correct UK offset, and
``datetime.now().astimezone()`` returns it as an aware datetime with the live
DST offset baked in — that is the primary source, and it needs no packages.

``zoneinfo`` with an explicit ``Europe/London`` is *preferred* when the tzdata
is actually present, because it stays correct even if the machine's own zone is
ever changed away from the UK.  It is strictly an upgrade: this machine has no
``tzdata`` module installed (verified — ``ZoneInfoNotFoundError``), so the
import is guarded and the OS clock carries it.  Never let a missing optional
package be the reason ORION cannot tell the time.

Clock changes
-------------
Nothing is cached.  Every call reads the clock, so a system clock correction, a
DST transition or simply passing midnight is picked up on the next question
rather than being frozen at whatever it was when ORION started.  ``now()`` is
the single seam a test overrides, via ``set_clock()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Callable, Optional

# The UK zone, when the platform can actually supply it.  Guarded: a machine
# without tzdata (this one) must still tell the time perfectly well.
_UK_ZONE: Optional[tzinfo]
try:  # pragma: no cover - depends on whether tzdata is installed
    from zoneinfo import ZoneInfo

    _UK_ZONE = ZoneInfo("Europe/London")
except Exception:  # ZoneInfoNotFoundError, ImportError, anything
    _UK_ZONE = None


# ── the bands ─────────────────────────────────────────────────────────────────
# Start hour of each period, in order.  NIGHT wraps midnight and is therefore
# handled as "at or after NIGHT_START, or before MORNING_START".
MORNING_START = 5
AFTERNOON_START = 12
EVENING_START = 18
NIGHT_START = 22

MORNING = "morning"
AFTERNOON = "afternoon"
EVENING = "evening"
NIGHT = "night"

#: Meteorological seasons by month, northern hemisphere.
_SEASONS_NORTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}
#: The same months, southern hemisphere — six months out of phase.
_SEASONS_SOUTH = {
    month: {"winter": "summer", "summer": "winter",
            "spring": "autumn", "autumn": "spring"}[name]
    for month, name in _SEASONS_NORTH.items()
}

_ORDINALS = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}


def ordinal(day: int) -> str:
    """7 -> '7th', 21 -> '21st'.  Used by every spoken date in ORION."""
    return f"{day}{_ORDINALS.get(int(day), 'th')}"


def classify(moment: datetime | time | int) -> str:
    """Name the part of the day for *moment* — the single band table.

    Accepts a datetime, a time, or a bare hour, so every caller in the codebase
    can use it without first shaping its value.
    """
    if isinstance(moment, datetime):
        hour = moment.hour
    elif isinstance(moment, time):
        hour = moment.hour
    else:
        hour = int(moment)
    hour %= 24
    if hour >= NIGHT_START or hour < MORNING_START:
        return NIGHT
    if hour < AFTERNOON_START:
        return MORNING
    if hour < EVENING_START:
        return AFTERNOON
    return EVENING


def greeting_for(moment: datetime | time | int) -> str:
    """The period you GREET with — 'morning' / 'afternoon' / 'evening'.

    Distinct from :func:`classify`: there is no English greeting "good night"
    that means hello, so the small hours greet as evening.  This is why ORION
    can correctly say "Good evening" at 00:31 while every other subsystem still
    knows the true period is night.
    """
    period = classify(moment)
    return EVENING if period == NIGHT else period


@dataclass(frozen=True)
class TimeSnapshot:
    """One consistent read of the clock.

    Everything a caller might need is captured together, so a subsystem can
    never straddle midnight mid-sentence by asking for the date and then the
    hour a moment later.
    """

    moment: datetime          # timezone-aware local time
    period: str               # night / morning / afternoon / evening
    greeting: str             # morning / afternoon / evening
    season: str               # winter / spring / summer / autumn
    is_daytime: bool
    tz_name: str              # "GMT" / "BST" / whatever the OS reports
    utc_offset: str           # "+00:00" / "+01:00"

    @property
    def date(self) -> date:
        return self.moment.date()

    @property
    def time(self) -> time:
        return self.moment.timetz()


class TimeService:
    """The authoritative clock.  Construct once; share it everywhere.

    A module-level default instance is provided (:data:`TIME`) because most of
    ORION reaches for the time from deep inside call stacks that have no
    dependency-injection seam.  Tests override the clock rather than the
    instance, via :meth:`set_clock`, so a fake time is genuinely global and no
    subsystem can accidentally read the real one mid-test.
    """

    def __init__(self, hemisphere: str = "north",
                 clock: Callable[[], datetime] | None = None) -> None:
        self._hemisphere = "south" if str(hemisphere).lower().startswith("s") else "north"
        self._clock: Callable[[], datetime] | None = clock

    # ── the clock seam ────────────────────────────────────────────────────────

    def set_clock(self, clock: Callable[[], datetime] | None) -> None:
        """Override the clock (tests, simulated protocol runs).  None restores
        the real one."""
        self._clock = clock

    def set_hemisphere(self, hemisphere: str) -> None:
        """Set from resolved locality, so 'summer' is right in Sydney too."""
        self._hemisphere = "south" if str(hemisphere).lower().startswith("s") else "north"

    def now(self) -> datetime:
        """The current local time, timezone-aware, DST already applied.

        Never cached.  Reading the clock every time is what makes ORION notice
        midnight, a DST change and a system clock correction without a restart.
        """
        if self._clock is not None:
            moment = self._clock()
            # A naive fake clock is still usable — attach the local zone rather
            # than rejecting it, so tests can pass plain datetimes.
            if moment.tzinfo is None:
                return moment.replace(tzinfo=self._zone())
            return moment
        if _UK_ZONE is not None:
            return datetime.now(_UK_ZONE)
        return datetime.now().astimezone()

    def _zone(self) -> tzinfo:
        if _UK_ZONE is not None:
            return _UK_ZONE
        return datetime.now().astimezone().tzinfo or timezone.utc

    # ── the API named in the spec ─────────────────────────────────────────────

    def current_datetime(self) -> datetime:
        return self.now()

    def current_date(self) -> date:
        return self.now().date()

    def current_time(self) -> time:
        return self.now().timetz()

    def timezone(self) -> str:
        """The live zone abbreviation — 'GMT' in winter, 'BST' in summer."""
        moment = self.now()
        name = moment.tzname() or ""
        if name:
            return name
        offset = moment.utcoffset() or timedelta(0)
        return "UTC" if not offset else f"UTC{self.utc_offset()}"

    def utc_offset(self) -> str:
        offset = self.now().utcoffset() or timedelta(0)
        total = int(offset.total_seconds())
        sign = "-" if total < 0 else "+"
        total = abs(total)
        return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"

    def time_of_day(self) -> str:
        """night / morning / afternoon / evening — the TRUE period."""
        return classify(self.now())

    def greeting_period(self) -> str:
        """morning / afternoon / evening — what ORION says out loud."""
        return greeting_for(self.now())

    def season(self) -> str:
        table = _SEASONS_SOUTH if self._hemisphere == "south" else _SEASONS_NORTH
        return table.get(self.now().month, "")

    def is_morning(self) -> bool:
        return self.time_of_day() == MORNING

    def is_afternoon(self) -> bool:
        return self.time_of_day() == AFTERNOON

    def is_evening(self) -> bool:
        return self.time_of_day() == EVENING

    def is_night(self) -> bool:
        return self.time_of_day() == NIGHT

    def is_daytime(self) -> bool:
        """True for morning and afternoon only.

        This is the predicate the weather clause needed: "a clear summer day"
        at 00:31 was wrong precisely because nothing ever asked whether it was
        actually daytime.
        """
        return self.time_of_day() in (MORNING, AFTERNOON)

    def is_weekend(self) -> bool:
        return self.now().weekday() >= 5

    # ── composed forms every subsystem shares ─────────────────────────────────

    def snapshot(self) -> TimeSnapshot:
        """One atomic read — use this when you need more than a single field."""
        moment = self.now()
        period = classify(moment)
        table = _SEASONS_SOUTH if self._hemisphere == "south" else _SEASONS_NORTH
        offset = moment.utcoffset() or timedelta(0)
        total = int(offset.total_seconds())
        sign = "-" if total < 0 else "+"
        total = abs(total)
        return TimeSnapshot(
            moment=moment,
            period=period,
            greeting=EVENING if period == NIGHT else period,
            season=table.get(moment.month, ""),
            is_daytime=period in (MORNING, AFTERNOON),
            tz_name=moment.tzname() or "UTC",
            utc_offset=f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}",
        )

    def spoken_now(self) -> str:
        """'twenty-seven minutes past ten in the evening' — words, never digits.

        Digit clocks get misread aloud by every voice channel ('22:27' spoken
        as 'twenty twenty-seven'), which is why the spoken forms exist at all.
        """
        from .utils import spoken_time
        moment = self.now()
        return spoken_time(moment.hour, moment.minute)

    def spoken_date(self) -> str:
        """'Friday the 8th of August'."""
        moment = self.now()
        return (f"{moment.strftime('%A')} the {ordinal(moment.day)} "
                f"of {moment.strftime('%B')}")

    def describe(self) -> str:
        """A short human line for logs and the diagnostics panel."""
        snap = self.snapshot()
        return (f"{snap.moment.strftime('%A %d %B %Y, %H:%M:%S')} "
                f"{snap.tz_name} (UTC{snap.utc_offset}) — "
                f"{snap.period}, {snap.season}")

    def prompt_line(self) -> str:
        """The authoritative time grounding injected into every system prompt.

        The model must never infer the date, the season or the part of the day;
        ORION has claimed 2012 and 2021 in error, and claimed a summer daytime
        at half past midnight.  Stating all of it explicitly, with the period
        named, is what stops that.
        """
        from .utils import spoken_time, spoken_year
        snap = self.snapshot()
        moment = snap.moment
        return (
            "AUTHORITATIVE CURRENT DATE AND TIME — this is correct and takes "
            "absolute precedence over any assumption you hold: it is now "
            f"{moment.strftime('%A, %d %B %Y, %H:%M')} {snap.tz_name} "
            f"(the year is {moment.strftime('%Y')}). It is currently the "
            f"{snap.period.upper()} — "
            f"{'daytime' if snap.is_daytime else 'NOT daytime, it is dark out'} — "
            f"in {snap.season}. Never describe the present as a different part "
            "of the day, a different season, or a different year than these. "
            "When you reason about recency ('today', 'this week', 'latest'), "
            "anchor to this moment. WHEN SPEAKING ALOUD, always render clock "
            "times and years as words, never digit by digit: right now you "
            f"would say the time as '{spoken_time(moment.hour, moment.minute)}' "
            f"and the year as '{spoken_year(moment.year)}'. Never read a "
            "24-hour clock as digits — '22:27' spoken as 'twenty-two "
            "twenty-seven' or misread as 'twenty twenty-seven' is wrong; say "
            "'ten twenty-seven in the evening'."
        )


#: The shared instance.  Import this, do not build your own.
TIME = TimeService()


# ── module-level conveniences (the common one-liners) ─────────────────────────

def now() -> datetime:
    return TIME.now()


def time_of_day() -> str:
    return TIME.time_of_day()


def greeting_period() -> str:
    return TIME.greeting_period()


def season() -> str:
    return TIME.season()


def is_daytime() -> bool:
    return TIME.is_daytime()


def snapshot() -> TimeSnapshot:
    return TIME.snapshot()
