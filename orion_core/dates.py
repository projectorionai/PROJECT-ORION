"""
Dates — exact answers to the calendar questions people ask.

"How many days until Christmas?", "what day is the 4th of May?", "what's the
date three weeks from today?", "how long until 5pm?". A language model is
reliably unreliable at these (an off-by-one on the count, the wrong weekday),
and ORION had nothing local to ask: the calculator did sums, units and world
time, not the calendar. This answers them from the clock, exactly.

It understands:

    targets   named days (Christmas, Christmas Eve, Boxing Day, New Year's Day
              and Eve, Valentine's Day, Halloween, Bonfire Night, St Patrick's
              Day, Easter Sunday, Good Friday, Easter Monday, Mothering Sunday
              (UK), Father's Day, Remembrance Sunday), written dates ("4 May",
              "the 1st of January", "December 25th", "25/12", "25/12/2027"),
              weekdays ("Friday", "next Friday", "this Friday"), "today",
              "tomorrow", "yesterday", and offsets ("3 weeks from today",
              "in 10 days", "100 days from now", "2 months from today")
    questions how many days / weeks / months until or since; what day (of the
              week) is it on; what date is it; how long until a clock time

UK conventions: day before month in numeric dates ("4/5" is 4 May), dates
said as "Friday 25 December". A named day that has already passed this year
means next year's. Returns None for anything that is not a calendar question,
so the calculator carries on as before.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

_MONTHS = {name: i for i, names in enumerate((
    (), ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
    ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
    ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
    ("december", "dec")), 0) for name in names}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                 "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50, "a hundred": 100,
                 "hundred": 100}
_ORDINAL = r"(?:st|nd|rd|th)?"


def easter(year: int) -> date:
    """Western Easter Sunday (the anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _named(year: int) -> dict[str, tuple[str, date]]:
    e = easter(year)
    return {
        "christmas eve": ("Christmas Eve", date(year, 12, 24)),
        "christmas day": ("Christmas Day", date(year, 12, 25)),
        "christmas": ("Christmas Day", date(year, 12, 25)),
        "xmas": ("Christmas Day", date(year, 12, 25)),
        "boxing day": ("Boxing Day", date(year, 12, 26)),
        "new year's eve": ("New Year's Eve", date(year, 12, 31)),
        "new years eve": ("New Year's Eve", date(year, 12, 31)),
        "new year's day": ("New Year's Day", date(year, 1, 1)),
        "new years day": ("New Year's Day", date(year, 1, 1)),
        "new year": ("New Year's Day", date(year, 1, 1)),
        "valentine's day": ("Valentine's Day", date(year, 2, 14)),
        "valentines day": ("Valentine's Day", date(year, 2, 14)),
        "valentines": ("Valentine's Day", date(year, 2, 14)),
        "st patrick's day": ("St Patrick's Day", date(year, 3, 17)),
        "st patricks day": ("St Patrick's Day", date(year, 3, 17)),
        "halloween": ("Halloween", date(year, 10, 31)),
        "bonfire night": ("Bonfire Night", date(year, 11, 5)),
        "guy fawkes night": ("Bonfire Night", date(year, 11, 5)),
        "easter sunday": ("Easter Sunday", e),
        "easter monday": ("Easter Monday", e + timedelta(days=1)),
        "good friday": ("Good Friday", e - timedelta(days=2)),
        "easter": ("Easter Sunday", e),
        "mothering sunday": ("Mothering Sunday", e - timedelta(days=21)),
        "mother's day": ("Mothering Sunday", e - timedelta(days=21)),
        "mothers day": ("Mothering Sunday", e - timedelta(days=21)),
        "father's day": ("Father's Day", _nth_weekday(year, 6, 6, 3)),
        "fathers day": ("Father's Day", _nth_weekday(year, 6, 6, 3)),
        "remembrance sunday": ("Remembrance Sunday", _nth_weekday(year, 11, 6, 2)),
    }


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _upcoming(make, today: date) -> date | None:
    """This year's date, or next year's when this year's has passed."""
    this = make(today.year)
    if this is not None and this >= today:
        return this
    return make(today.year + 1)


def _number(word: str) -> int | None:
    word = word.strip().lower()
    if word.isdigit():
        return int(word)
    return _NUMBER_WORDS.get(word)


def resolve(text: str, today: date, *, past: bool = False) -> tuple[str, date] | None:
    """(how to say it, the date) for the target named in *text*."""
    t = " " + re.sub(r"\s+", " ", text.lower().replace("’", "'")) + " "
    # offsets: "3 weeks from today", "in 10 days", "2 months ago"
    m = re.search(r"\b(?:in\s+)?(?P<n>\d{1,4}|a hundred|" + "|".join(_NUMBER_WORDS) + r")\s+"
                  r"(?P<unit>day|week|month|year)s?\s*(?P<dir>from (?:today|now)|time|ago|"
                  r"after today|before today|from now)?", t)
    # "in 10 days" is an offset; "how many days are in a year" is not.
    if m and (m.group("dir") or re.search(r"(?<!are )(?<!is )\bin\s+" + re.escape(m.group("n"))
                                          + r"\s+" + m.group("unit"), t)):
        n = _number(m.group("n"))
        if n is not None:
            sign = -1 if m.group("dir") in ("ago", "before today") else 1
            unit = m.group("unit")
            if unit == "day":
                target = today + timedelta(days=sign * n)
            elif unit == "week":
                target = today + timedelta(weeks=sign * n)
            else:
                months = n * (12 if unit == "year" else 1) * sign
                y, mo = divmod(today.month - 1 + months, 12)
                year, month = today.year + y, mo + 1
                target = _safe_date(year, month, today.day) or _safe_date(year, month, 28)
            return f"{n} {unit}{'s' if n != 1 else ''} {'ago' if sign < 0 else 'from today'}", target
    for word, offset in (("today", 0), ("tomorrow", 1), ("yesterday", -1)):
        if re.search(rf"\b{word}\b", t) and not re.search(r"\bfrom today\b", t):
            if not re.search(rf"\b(?:{_MONTH_ALT}|{'|'.join(_WEEKDAYS)})\b", t):
                return word, today + timedelta(days=offset)
    # named days
    names = _named(today.year)
    for key in sorted(names, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", t):
            label = names[key][0]
            if past:
                when = names[key][1] if names[key][1] <= today else _named(today.year - 1)[key][1]
            else:
                when = _upcoming(lambda y: _named(y)[key][1], today)
            return label, when
    # written dates: "4 may", "the 1st of january", "may 4th", "25/12(/2027)"
    year_m = re.search(r"\b(20\d\d)\b", t)
    year = int(year_m.group(1)) if year_m else None
    m = (re.search(rf"\b(?P<d>\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?(?P<m>{_MONTH_ALT})\b", t)
         or re.search(rf"\b(?P<m>{_MONTH_ALT})\s+(?:the\s+)?(?P<d>\d{{1,2}}){_ORDINAL}\b", t))
    if m:
        day, month = int(m.group("d")), _MONTHS[m.group("m")]
    else:
        m = re.search(r"\b(?P<d>\d{1,2})/(?P<m>\d{1,2})(?:/(?P<y>\d{2,4}))?\b", t)
        if m:
            day, month = int(m.group("d")), int(m.group("m"))
            if m.group("y"):
                y = int(m.group("y"))
                year = y + 2000 if y < 100 else y
        else:
            day = month = 0
    if month:
        if year is not None:
            when = _safe_date(year, month, day)
        elif past:
            when = _safe_date(today.year, month, day)
            if when is not None and when > today:
                when = _safe_date(today.year - 1, month, day)
        else:
            when = _upcoming(lambda y: _safe_date(y, month, day), today)
        if when is None:
            return None
        return f"{when.day} {when.strftime('%B')}", when
    # weekdays: "next friday", "this friday", "friday". "Next Friday" is the
    # coming one; the answer names the date, so either reading is checkable.
    m = re.search(rf"\b(?P<which>next|this|last|coming)?\s*(?P<wd>{'|'.join(_WEEKDAYS)})\b", t)
    if m:
        index = _WEEKDAYS.index(m.group("wd"))
        which = m.group("which") or ""
        if which == "last" or past:
            delta = (today.weekday() - index) % 7 or 7
            return f"last {m.group('wd').title()}", today - timedelta(days=delta)
        delta = (index - today.weekday()) % 7
        if delta == 0 and which in ("next", "coming"):
            delta = 7
        return f"{'next ' if which == 'next' else ''}{m.group('wd').title()}", today + timedelta(days=delta)
    return None


def _say_date(day: date, today: date) -> str:
    text = f"{day.strftime('%A')} {day.day} {day.strftime('%B')}"
    return text + (f" {day.year}" if day.year != today.year else "")


def _span(days: int, suffix: str = "") -> str:
    """'81 days (11 weeks and 4 days)', '3 weeks (21 days)', '5 days' -
    *suffix* (' ago') goes after the main figure, before the detail."""
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''}{suffix}"
    weeks, rest = divmod(days, 7)
    if not rest:
        return f"{weeks} weeks{suffix} ({days} days)"
    return (f"{days} days{suffix} ({weeks} week{'s' if weeks != 1 else ''} and "
            f"{rest} day{'s' if rest != 1 else ''})")


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


_QUESTION = re.compile(
    r"\b(how many (?:days|weeks|months|sleeps)|how long (?:until|till|til|to|before|since|ago)|"
    r"days? (?:until|till|to|left)|weeks? (?:until|till|to)|"
    r"what (?:day|date)|which day|what's the date|whats the date|what is the date|"
    r"(?:from|after|before) today|from now|ago\b|when is|when's|"
    r"(?:today|tomorrow|yesterday)'?s date)", re.I)
_CLOCK = re.compile(r"how long (?:until|till|til|before|to)\s+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>am|pm)?\b", re.I)


def answer(text: str, now: datetime | None = None) -> str | None:
    """The exact answer to a calendar question, or None if it isn't one."""
    raw = str(text or "").strip()
    if not raw:
        return None
    now = now or datetime.now()
    today = now.date()
    lowered = raw.lower().rstrip("?.! ")
    if re.fullmatch(r"(?:what(?:'s|s| is) (?:the |today's |todays )?date(?: today)?|"
                    r"what day is it(?: today)?|what(?:'s|s| is) today)", lowered):
        return f"Today is {_say_date(today, today)} {today.year}."
    if re.search(r"\b(?:next|upcoming|coming)\s+bank\s+holiday|bank\s+holiday\s+coming\s+up", lowered) or \
            re.fullmatch(r"(?:when(?:'s| is)|what(?:'s| is))\s+(?:the\s+)?(?:next\s+)?bank holiday", lowered):
        # GOV.UK's list, as cached by bank_holidays.refresh (the greeting
        # keeps it fresh) - never a network call from here.
        try:
            from .bank_holidays import next_holiday
            found = next_holiday(today)
        except Exception:
            found = None
        if found is None:
            return None
        title, when = found
        delta = (when - today).days
        lead = "today" if delta == 0 else "tomorrow" if delta == 1 else f"in {_span(delta)}"
        return f"The next bank holiday is {title}, on {_say_date(when, today)} — {lead}."
    if not _QUESTION.search(raw):
        return None

    clock = _CLOCK.search(lowered)
    if clock:
        hour, minute = int(clock.group("h")), int(clock.group("m") or 0)
        ap = clock.group("ap")
        if ap == "pm" and hour < 12:
            hour += 12
        elif ap == "am" and hour == 12:
            hour = 0
        elif ap is None and hour < 12 and now.hour >= hour:
            hour += 12                      # "until 5" in the afternoon means 5 pm
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        mins = int((target - now).total_seconds() // 60)
        h, m = divmod(mins, 60)
        parts = ([f"{h} hour{'s' if h != 1 else ''}"] if h else []) + \
                ([f"{m} minute{'s' if m != 1 else ''}"] if m or not h else [])
        return f"{' and '.join(parts)} until {target.strftime('%H:%M')}."

    past = bool(re.search(r"\b(since|ago|was|did)\b", lowered))
    found = resolve(lowered, today, past=past)
    if found is None:
        return None
    label, when = found
    delta = (when - today).days
    said = _say_date(when, today)
    if re.search(r"\bhow many (?:days|weeks|months|sleeps)|how long|days? (?:until|till|to|left)|"
                 r"weeks? (?:until|till|to)", lowered):
        if delta == 0:
            return f"{_cap(label)} is today, {said}."
        if delta < 0:
            return f"{_cap(label)} was {_span(-delta, ' ago')}, on {said}."
        if "sleeps" in lowered:
            return f"{delta} sleep{'s' if delta != 1 else ''} until {label} ({said})."
        if "month" in lowered and delta >= 28:
            months = (when.year - today.year) * 12 + when.month - today.month - (when.day < today.day)
            return (f"{_cap(label)} is {months} month{'s' if months != 1 else ''} away "
                    f"({delta} days), on {said}.")
        return f"{_cap(label)} is in {_span(delta)}, on {said}."
    when_word = "is" if delta >= 0 else "was"
    if label in ("today", "tomorrow", "yesterday"):
        return f"{_cap(label)} {when_word} {said}."
    if label.endswith(("from today", "ago")):
        return f"{_cap(label)} {when_word} {said}."
    if re.sub(r"^(?:next|last) ", "", label.lower()) in _WEEKDAYS:      # not "Easter Sunday"
        return f"{_cap(label)} {when_word} {when.day} {when.strftime('%B')}, {_relative(delta)}."
    if re.match(r"\d", label):
        return f"{_cap(said)}, {_relative(delta)}."
    return f"{_cap(label)} {when_word} on {said}, {_relative(delta)}."


def _relative(delta: int) -> str:
    if delta == 0:
        return "that's today"
    if delta == 1:
        return "that's tomorrow"
    if delta == -1:
        return "that was yesterday"
    return f"in {_span(delta)}" if delta > 0 else _span(-delta, " ago")


__all__ = ["answer", "easter", "resolve"]
