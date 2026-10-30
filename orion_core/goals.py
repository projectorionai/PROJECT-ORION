"""
Personal goals — one coherent system for what the user is working towards.

Before this module ORION had four half-goals, none of which could answer "am I
going to make it?":

  * ``companion.py`` kept a title and a hand-typed 0-100 percentage;
  * ``cognition.GoalManager`` kept a title, a priority and a progress float in
    the cognitive-state JSON, scored for relevance but never measured;
  * ``missions.py`` keeps goal *checklists* inside a mission (project work,
    not personal targets — left exactly as it is);
  * two forged plugins (``fitness_goal_tracker`` / ``fitness_progress_monitor``)
    were the owner asking for the real thing and getting a record writer.

This is the real thing. A goal has an outcome, a reason it matters, a
measurable target (metric, unit, baseline, target value), a deadline, an area
and a priority; milestones and key results; linked habits, tasks and
reminders. Progress is logged in ordinary speech — "I ran 5k today", "log two
hours of revision" — parsed into the right goal's unit, and from the log ORION
can say whether the goal is on track, behind or at risk and when, at the
current pace, it will actually be finished.

Design rules this file keeps:

  * **One store.** ``config/goals.db`` (SQLite, WAL). The older stores are
    imported once, read-only and idempotently — a goal the user deleted here
    is never resurrected from the ledger it came from.
  * **Never on the GUI thread.** Every method that touches the database is
    synchronous and is called through ``asyncio.to_thread`` (the tool) or a
    worker thread (the deck page). ``attention_cached`` exists for callers
    on the loop: it returns the last computed summary and refreshes it on a
    daemon thread, so the loop never waits on SQLite.
  * **No new scheduler.** Reminders go through ``ReminderService``. Nudges
    are a banner from ``GoalNudges`` (an idle pass that admission and a quiet
    conversation gate, never spoken) and a line in the morning briefing.
  * **Honest arithmetic.** A projection is only made from logged progress; a
    goal with nothing logged is "not started", never "on track".
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .constants import CONFIG_DIR
from .db import connect
from .utils import first_line

#: Redirected by conftest so the suite never opens the owner's goals.
GOALS_FILENAME = "goals.db"

AREAS: tuple[str, ...] = ("health", "study", "career", "finance", "projects", "personal")
PRIORITIES: dict[str, int] = {"high": 1, "medium": 2, "low": 3}
_PRIORITY_NAMES = {v: k for k, v in PRIORITIES.items()}
CADENCE_DAYS: dict[str, int] = {"daily": 1, "weekly": 7, "fortnightly": 14, "monthly": 30, "none": 0}

#: Pace labels, worst first — the order the briefing and the deck sort by.
PACE_ORDER: tuple[str, ...] = ("overdue", "at risk", "behind", "not started", "stalled",
                               "on track", "in progress", "ahead", "target reached", "done")


def default_path() -> Path:
    return CONFIG_DIR / GOALS_FILENAME


def _now() -> datetime:
    from .time_service import TIME
    return TIME.now()


def _stamp(moment: datetime | None = None) -> str:
    return (moment or _now()).astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_stamp(value: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _local_date(stamp: str, now: datetime) -> date | None:
    moment = _parse_stamp(stamp)
    if moment is None:
        return None
    return moment.astimezone(now.tzinfo or timezone.utc).date()


def _clean(value: Any, limit: int = 200) -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split())
    return text[:limit]


def _float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(str(value).replace(",", "").replace("£", "").replace("$", "").replace("€", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    # bool("false") is True — the confirm-deletes trap (audit 23 Sep 2026).
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "y", "confirm", "confirmed"}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")[:64]


# ── dates said the way people say them ────────────────────────────────────────

_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), start=1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9


def parse_deadline(value: Any, today: date | None = None) -> str:
    """'2026-12-31', '31 December', 'end of the month', 'in 6 weeks' → ISO, or ''.

    Deliberately small: the model usually passes an ISO date, and this only
    has to cover what a person types into the deck or says in one breath."""
    text = _clean(value, 60).lower().strip(" .")
    if not text:
        return ""
    today = today or _now().date()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        pass
    m = re.fullmatch(r"(?:in\s+)?(\d+)\s*(day|week|month|year)s?(?:\s+time)?", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "day":
            return (today + timedelta(days=n)).isoformat()
        if unit == "week":
            return (today + timedelta(weeks=n)).isoformat()
        months = n * (12 if unit == "year" else 1)
        year, month = divmod(today.month - 1 + months, 12)
        return _month_end(today.year + year, month + 1).isoformat()
    if text in {"end of the month", "end of month", "this month"}:
        return _month_end(today.year, today.month).isoformat()
    if text in {"end of the year", "end of year", "this year"}:
        return date(today.year, 12, 31).isoformat()
    if text in {"end of the week", "end of week", "this week"}:
        return (today + timedelta(days=6 - today.weekday())).isoformat()
    if text in {"next month", "end of next month"}:
        year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        return _month_end(year, month).isoformat()
    m = re.fullmatch(r"(?:by\s+|on\s+|the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)\s*(\d{4})?", text)
    if m and m.group(2) in _MONTHS:
        return _dated(int(m.group(1)), _MONTHS[m.group(2)], m.group(3), today)
    m = re.fullmatch(r"(?:by\s+|on\s+)?([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?", text)
    if m and m.group(1) in _MONTHS:
        return _dated(int(m.group(2)), _MONTHS[m.group(1)], m.group(3), today)
    m = re.fullmatch(r"(?:by\s+|end of\s+)?([a-z]+)\s*(\d{4})?", text)
    if m and m.group(1) in _MONTHS:
        year = int(m.group(2)) if m.group(2) else today.year
        if not m.group(2) and _MONTHS[m.group(1)] < today.month:
            year += 1
        return _month_end(year, _MONTHS[m.group(1)]).isoformat()
    return ""


def _dated(day: int, month: int, year: str | None, today: date) -> str:
    try:
        chosen = date(int(year) if year else today.year, month, day)
    except ValueError:
        return ""
    if not year and chosen < today:
        try:
            chosen = date(today.year + 1, month, day)
        except ValueError:
            return ""
    return chosen.isoformat()


def _month_end(year: int, month: int) -> date:
    following = date(year + (month // 12), month % 12 + 1, 1)
    return following - timedelta(days=1)


def _say_date(iso: str, today: date | None = None) -> str:
    """'31 Dec' this year, '31 Dec 2027' otherwise — speakable, never 2026-12-31."""
    try:
        d = date.fromisoformat(iso[:10])
    except (TypeError, ValueError):
        return iso
    today = today or _now().date()
    return f"{d.day} {d.strftime('%b')}" + ("" if d.year == today.year else f" {d.year}")


# ── units ─────────────────────────────────────────────────────────────────────
#
# A unit is (dimension, factor to the dimension's base). Money never converts
# between currencies — an exchange rate is not ORION's to guess.

_UNITS: dict[str, tuple[str, float]] = {
    "km": ("distance", 1.0), "mi": ("distance", 1.609344), "m": ("distance", 0.001),
    "h": ("time", 1.0), "min": ("time", 1.0 / 60.0),
    "kg": ("mass", 1.0), "lb": ("mass", 0.45359237),
    "gbp": ("gbp", 1.0), "usd": ("usd", 1.0), "eur": ("eur", 1.0),
}
_ALIASES: dict[str, str] = {}
for _canon, _words in {
    "km": ("km", "kms", "kilometre", "kilometres", "kilometer", "kilometers", "k"),
    "mi": ("mi", "mile", "miles"),
    "m": ("metre", "metres", "meter", "meters"),
    "h": ("h", "hr", "hrs", "hour", "hours"),
    "min": ("min", "mins", "minute", "minutes"),
    "kg": ("kg", "kgs", "kilo", "kilos", "kilogram", "kilograms"),
    "lb": ("lb", "lbs"),
    "gbp": ("£", "gbp", "quid", "pound", "pounds"),
    "usd": ("$", "usd", "dollar", "dollars"),
    "eur": ("€", "eur", "euro", "euros"),
}.items():
    for _w in _words:
        _ALIASES[_w] = _canon

_DISPLAY = {"km": "km", "mi": "miles", "m": "m", "h": "hours", "min": "minutes",
            "kg": "kg", "lb": "lb"}
_CURRENCY_SIGN = {"gbp": "£", "usd": "$", "eur": "€"}

#: Count units: whatever follows the number. Normalised to a plural word.
_COUNT_SINGULAR = {
    "pages": "page", "words": "word", "sessions": "session", "reps": "rep",
    "chapters": "chapter", "lessons": "lesson", "cards": "card", "workouts": "workout",
    "problems": "problem", "applications": "application", "videos": "video",
    "posts": "post", "times": "time", "books": "book", "steps": "step", "days": "day",
    "glasses": "glass", "litres": "litre", "classes": "class", "lectures": "lecture",
    "essays": "essay", "commits": "commit", "runs": "run", "swims": "swim",
    "rides": "ride", "walks": "walk", "articles": "article", "papers": "paper",
    "modules": "module", "questions": "question", "exercises": "exercise",
    "interviews": "interview", "calls": "call", "episodes": "episode", "laps": "lap",
}
_COUNT_PLURAL = {v: k for k, v in _COUNT_SINGULAR.items()}


def canonical_unit(unit: Any) -> str:
    """'kilometres' → 'km', 'Hours' → 'h', '£' → 'gbp', 'page' → 'pages'."""
    word = _clean(unit, 30).lower().strip(" .")
    if not word:
        return ""
    if word in _ALIASES:
        return _ALIASES[word]
    if word in _COUNT_PLURAL:
        return _COUNT_PLURAL[word]
    if word in _COUNT_SINGULAR:
        return word
    return word


def unit_dimension(unit: str) -> str:
    unit = canonical_unit(unit)
    if unit in _UNITS:
        return _UNITS[unit][0]
    return f"count:{unit}" if unit else ""


def convert(amount: float, from_unit: str, to_unit: str) -> float | None:
    """*amount* in *from_unit* expressed in *to_unit*, or None if incompatible."""
    a, b = canonical_unit(from_unit), canonical_unit(to_unit)
    if a == b:
        return amount
    if not a or not b:
        return amount if not a else None
    if a in _UNITS and b in _UNITS and _UNITS[a][0] == _UNITS[b][0]:
        return amount * _UNITS[a][1] / _UNITS[b][1]
    return None


def fmt_qty(value: float | None, unit: str) -> str:
    """'£1,200', '5 km', '1.5 hours', '30 pages', '1 page'."""
    if value is None:
        return "—"
    unit = canonical_unit(unit)
    # Spoken precision: "9.8 km a week", not "9.78"; pennies only for money.
    places = 2 if unit in _CURRENCY_SIGN or abs(value) < 1 else 1 if abs(value) < 100 else 0
    rounded = round(value, places)
    number = (f"{rounded:,.0f}" if abs(rounded - round(rounded)) < 1e-9
              else f"{rounded:,.2f}".rstrip("0").rstrip("."))
    if unit in _CURRENCY_SIGN:
        return f"{_CURRENCY_SIGN[unit]}{number}"
    if not unit:
        return number
    label = _DISPLAY.get(unit, unit)
    if unit in _COUNT_SINGULAR and abs(rounded - 1) < 1e-9:
        label = _COUNT_SINGULAR[unit]
    elif unit == "h" and abs(rounded - 1) < 1e-9:
        label = "hour"
    elif unit == "min" and abs(rounded - 1) < 1e-9:
        label = "minute"
    elif unit == "mi" and abs(rounded - 1) < 1e-9:
        label = "mile"
    return f"{number} {label}"


# ── progress in ordinary speech ───────────────────────────────────────────────

_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "ninety": 90, "hundred": 100, "half": 0.5, "couple": 2,
}
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_QTY_RE = re.compile(
    r"(?P<pre>[£$€])?\s*(?P<num>" + _NUMBER + r")\s*(?P<mult>k\b)?\s*(?P<unit>[a-z]+)?",
    re.IGNORECASE)
_WORD_QTY_RE = re.compile(
    r"\b(?P<num>" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True))
    + r")(?:\s+(?:and\s+a\s+half))?\s+(?:of\s+)?(?:an?\s+)?(?P<unit>[a-z]+)", re.IGNORECASE)
#: Readings, not increments: "I weigh 82 kg", "I've saved £1,200 in total".
_ABSOLUTE_RE = re.compile(
    r"\b(?:in total|total(?:\s+of)?|altogether|now (?:at|on)|up to|reached|i'?m at|i am at|"
    r"weigh(?:ed)?(?:\s+in)?(?:\s+at)?|stands? at|so far|my best is|new pb|personal best|"
    r"pb of|maxed|max(?:ed)? out at|currently)\b", re.IGNORECASE)

#: Spoken verb forms → one activity word, so "ran" finds a goal called "Run…".
_LEMMAS: dict[str, str] = {
    "ran": "run", "running": "run", "runs": "run", "jog": "run", "jogged": "run",
    "jogging": "run", "marathon": "run", "parkrun": "run",
    "revised": "revise", "revising": "revise", "revision": "revise",
    "studied": "study", "studying": "study", "studies": "study",
    "reading": "read", "reads": "read",
    "wrote": "write", "writing": "write", "written": "write", "words": "write",
    "saved": "save", "saving": "save", "savings": "save", "put": "save",
    "benched": "bench", "benching": "bench", "lifted": "lift", "lifting": "lift",
    "squatted": "squat", "squats": "squat", "deadlifted": "deadlift", "deadlifts": "deadlift",
    "swam": "swim", "swimming": "swim", "swims": "swim",
    "cycled": "cycle", "cycling": "cycle", "biked": "cycle", "bike": "cycle",
    "rode": "cycle", "ride": "cycle",
    "walked": "walk", "walking": "walk", "walks": "walk", "steps": "walk",
    "coded": "code", "coding": "code", "programming": "code", "programmed": "code",
    "practised": "practise", "practiced": "practise", "practice": "practise",
    "practising": "practise", "practicing": "practise",
    "trained": "train", "training": "train",
    "meditated": "meditate", "meditation": "meditate", "meditating": "meditate",
    "weighed": "weigh", "weight": "weigh", "weighing": "weigh", "weigh-in": "weigh",
    "learnt": "learn", "learned": "learn", "learning": "learn",
    "paid": "pay", "repaid": "pay", "paying": "pay", "debt": "pay",
    "applied": "apply", "applications": "apply", "application": "apply",
    "exercised": "exercise", "workout": "exercise", "workouts": "exercise", "gym": "exercise",
    "slept": "sleep", "sleeping": "sleep", "drank": "drink", "drinking": "drink",
    "invested": "invest", "investing": "invest", "built": "build", "building": "build",
    "shipped": "ship", "shipping": "ship", "lost": "lose", "losing": "lose",
    "pages": "read", "books": "read", "chapters": "read", "kilometres": "run",
}
_STOP = frozenset("""
i i'm im ive i've me my mine we our the a an and or of to for in on at by with from
today yesterday tonight this morning evening afternoon week just did do done got
have has had was were is are be been so far now log logged add added please orion
another more some about around nearly almost over than then again also really
goal goals target towards toward progress
""".split())

#: Words that hint at a life area, for a goal the words alone cannot place.
_AREA_HINTS: dict[str, frozenset[str]] = {
    "health": frozenset("run swim cycle walk exercise bench lift squat deadlift weigh sleep "
                        "meditate drink lose fitness health train stretch yoga".split()),
    "study": frozenset("revise study learn exam lecture flashcard essay dissertation "
                       "module course read homework".split()),
    "finance": frozenset("save pay invest money budget debt emergency fund".split()),
    "career": frozenset("apply interview cv job network promotion portfolio".split()),
    "projects": frozenset("code build ship write launch release project".split()),
}


def _lemma(word: str) -> str:
    word = word.lower().strip("'’.,!?")
    if word in _LEMMAS:
        return _LEMMAS[word]
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _words(text: str) -> set[str]:
    return {_lemma(w) for w in re.findall(r"[a-z][a-z'’-]+", str(text or "").lower())
            if w not in _STOP and len(w) > 1} - _STOP


#: Verbs that make a bare "5k" money ("saved 5k") or a distance ("ran 5k").
_MONEY_VERBS = frozenset({"save", "pay", "invest", "earn", "spend", "raise", "fund"})
_DISTANCE_VERBS = frozenset({"run", "walk", "cycle", "swim", "row", "hike"})
#: Unit words carry no meaning about WHICH goal — "hours" fits every timed goal.
_UNIT_WORDS = frozenset(w for w in _ALIASES if w.isalpha()) | frozenset(
    {"minut", "hour", "half", "kilo", "kilometre", "mile", "metre", "pound", "dollar", "euro"})


@dataclass
class ParsedProgress:
    """What an utterance said: how much, in what, and whether it is a reading."""
    amount: float | None = None
    unit: str = ""
    absolute: bool = False
    thousands: bool = False         # "5k" with nothing to say whether km or thousands
    money: bool = False             # a bare amount said with a money verb ("saved 1k")
    words: set[str] = field(default_factory=set)
    raw: str = ""


def parse_progress(text: str) -> ParsedProgress:
    """'I ran 5k today' → 5 'km'; 'log 2 hours of revision' → 2 'h';
    'an hour and a half of piano' → 1.5 'h'; 'I weigh 82.5kg' → 82.5 'kg' absolute."""
    raw = _clean(text, 300)
    lowered = raw.lower()
    words = _words(raw)
    parsed = ParsedProgress(raw=raw, words=words - _UNIT_WORDS,
                            absolute=bool(_ABSOLUTE_RE.search(lowered)))
    time_total = 0.0
    time_seen = False
    best: tuple[float, str, bool] | None = None
    # "an hour and a half" / "half an hour" before the digit pass.
    if re.search(r"\bhalf an hour\b", lowered):
        time_total, time_seen = time_total + 0.5, True
    m = re.search(r"\b(an?|one|two|three|four|five)\s+hours?\s+and\s+a\s+half\b", lowered)
    if m:
        time_total, time_seen = time_total + _WORD_NUMBERS[m.group(1)] + 0.5, True
    for match in _QTY_RE.finditer(lowered):
        amount = float(match.group("num").replace(",", ""))
        unit_word = (match.group("unit") or "").lower()
        if unit_word in _STOP or _lemma(unit_word) in _STOP:
            unit_word = ""                    # "5k today": "today" is not a unit
        pre = match.group("pre") or ""
        thousands = bool(match.group("mult"))
        if unit_word == "grand":
            thousands, unit_word = True, ""
        unit = canonical_unit(pre) if pre else canonical_unit(unit_word) if unit_word else ""
        if unit in {"h", "min"}:
            time_total += amount if unit == "h" else amount / 60.0
            time_seen = True
            continue
        if thousands and (pre or unit):
            amount, thousands = amount * 1000, False          # "£5k", "5k steps"
        elif thousands:
            # A bare "5k" is a distance to a runner and thousands to a saver.
            # The verb decides when it can; otherwise the goal it lands on does
            # (see GoalsEngine._resolve_amount).
            if words & _MONEY_VERBS:
                amount, thousands, parsed.money = amount * 1000, False, True
            else:
                unit = "km"
                thousands = not (words & _DISTANCE_VERBS)
        elif not unit and not pre and words & _MONEY_VERBS:
            parsed.money = True
        if best is None:
            best = (amount, unit, thousands)
    if not time_seen:
        for match in _WORD_QTY_RE.finditer(lowered):
            word, unit_word = match.group("num").lower(), match.group("unit").lower()
            unit = canonical_unit(unit_word)
            if unit in {"h", "min"} or unit in _COUNT_SINGULAR or unit in _UNITS:
                amount = float(_WORD_NUMBERS.get(word, 1))
                if "and a half" in match.group(0).lower():
                    amount += 0.5
                if unit in {"h", "min"}:
                    time_total += amount if unit == "h" else amount / 60.0
                    time_seen = True
                elif best is None:
                    best = (amount, unit, False)
    if time_seen and (best is None or not best[1]):
        parsed.amount, parsed.unit = round(time_total, 4), "h"
    elif best is not None:
        parsed.amount, parsed.unit, parsed.thousands = best
    elif time_seen:
        parsed.amount, parsed.unit = round(time_total, 4), "h"
    return parsed


# ── records ───────────────────────────────────────────────────────────────────

@dataclass
class Goal:
    id: int
    title: str
    outcome: str = ""
    why: str = ""
    area: str = "personal"
    priority: int = 2
    metric: str = ""
    unit: str = ""
    baseline: float | None = None
    target: float | None = None
    mode: str = "cumulative"        # cumulative (sum the logs) | level (latest reading)
    deadline: str = ""
    start_date: str = ""
    cadence: str = "weekly"
    status: str = "active"          # active | paused | completed | archived | deleted
    manual_pct: float | None = None
    reflection: str = ""
    source: str = ""
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    last_checkin_at: str = ""

    @property
    def priority_name(self) -> str:
        return _PRIORITY_NAMES.get(self.priority, "medium")

    @property
    def measurable(self) -> bool:
        return self.target is not None

    def label(self) -> str:
        return f"#{self.id} {self.title}"


@dataclass
class Milestone:
    id: int
    goal_id: int
    title: str
    due: str = ""
    target_value: float | None = None
    done: bool = False
    done_at: str = ""
    position: int = 0


@dataclass
class KeyResult:
    id: int
    goal_id: int
    title: str
    unit: str = ""
    target: float | None = None
    current: float = 0.0
    done: bool = False


@dataclass
class Entry:
    id: int
    goal_id: int
    at: str
    kind: str                        # add | set | note
    amount: float | None = None
    unit: str = ""
    note: str = ""
    kr_id: int | None = None


@dataclass
class Link:
    id: int
    goal_id: int
    kind: str                        # habit | task | reminder
    ref: str
    label: str = ""
    created_at: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT '',
    why TEXT NOT NULL DEFAULT '',
    area TEXT NOT NULL DEFAULT 'personal',
    priority INTEGER NOT NULL DEFAULT 2,
    metric TEXT NOT NULL DEFAULT '',
    unit TEXT NOT NULL DEFAULT '',
    baseline REAL,
    target REAL,
    mode TEXT NOT NULL DEFAULT 'cumulative',
    deadline TEXT NOT NULL DEFAULT '',
    start_date TEXT NOT NULL DEFAULT '',
    cadence TEXT NOT NULL DEFAULT 'weekly',
    status TEXT NOT NULL DEFAULT 'active',
    manual_pct REAL,
    reflection TEXT NOT NULL DEFAULT '',
    source TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT '',
    last_checkin_at TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_goals_source ON goals(source) WHERE source IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE TABLE IF NOT EXISTS milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    due TEXT NOT NULL DEFAULT '',
    target_value REAL,
    done INTEGER NOT NULL DEFAULT 0,
    done_at TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_milestones_goal ON milestones(goal_id);
CREATE TABLE IF NOT EXISTS key_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    unit TEXT NOT NULL DEFAULT '',
    target REAL,
    current REAL NOT NULL DEFAULT 0,
    done INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_key_results_goal ON key_results(goal_id);
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER NOT NULL,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    amount REAL,
    unit TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    kr_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_entries_goal_at ON entries(goal_id, at);
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_links_goal ON links(goal_id);
CREATE TABLE IF NOT EXISTS nudges (
    key TEXT PRIMARY KEY,
    day TEXT NOT NULL,
    at TEXT NOT NULL
);
"""

_GOAL_FIELDS = ("title", "outcome", "why", "area", "priority", "metric", "unit", "baseline",
                "target", "mode", "deadline", "start_date", "cadence", "status", "manual_pct",
                "reflection", "source", "created_at", "updated_at", "completed_at",
                "last_checkin_at")


class GoalStore:
    """SQLite persistence. Thread-safe; every call is short and indexed."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = connect(self.path)
        self._lock = threading.RLock()
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except Exception:
                pass

    # goals
    @staticmethod
    def _goal(row: sqlite3.Row) -> Goal:
        data = {k: row[k] for k in row.keys()}
        return Goal(**{k: (v if v is not None or k in {"baseline", "target", "manual_pct", "source"} else "")
                       for k, v in data.items()})

    def insert_goal(self, goal: Goal) -> Goal:
        values = [getattr(goal, f) for f in _GOAL_FIELDS]
        values[_GOAL_FIELDS.index("source")] = goal.source or None
        with self._lock, self._db:
            cur = self._db.execute(
                f"INSERT INTO goals ({', '.join(_GOAL_FIELDS)}) VALUES ({', '.join('?' * len(_GOAL_FIELDS))})",
                values)
            goal.id = int(cur.lastrowid)
        return goal

    def insert_goal_if_new_source(self, goal: Goal) -> bool:
        """Imported goals only: a source key seen before (even on a goal the
        user has since deleted) is never inserted twice."""
        values = [getattr(goal, f) for f in _GOAL_FIELDS]
        with self._lock, self._db:
            cur = self._db.execute(
                f"INSERT OR IGNORE INTO goals ({', '.join(_GOAL_FIELDS)}) "
                f"VALUES ({', '.join('?' * len(_GOAL_FIELDS))})", values)
            if cur.rowcount:
                goal.id = int(cur.lastrowid)
            return bool(cur.rowcount)

    def update_goal(self, goal_id: int, **fields: Any) -> None:
        fields = {k: v for k, v in fields.items() if k in _GOAL_FIELDS}
        if not fields:
            return
        fields["updated_at"] = _stamp()
        with self._lock, self._db:
            self._db.execute(
                f"UPDATE goals SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                [*fields.values(), int(goal_id)])

    def get_goal(self, goal_id: int) -> Goal | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM goals WHERE id=?", (int(goal_id),)).fetchone()
        return self._goal(row) if row else None

    def goals(self, statuses: Iterable[str] | None = ("active", "paused")) -> list[Goal]:
        with self._lock:
            if statuses is None:
                rows = self._db.execute("SELECT * FROM goals WHERE status != 'deleted' "
                                        "ORDER BY priority, id").fetchall()
            else:
                wanted = tuple(statuses)
                rows = self._db.execute(
                    f"SELECT * FROM goals WHERE status IN ({', '.join('?' * len(wanted))}) "
                    "ORDER BY priority, id", wanted).fetchall()
        return [self._goal(r) for r in rows]

    def sources(self) -> set[str]:
        with self._lock:
            rows = self._db.execute("SELECT source FROM goals WHERE source IS NOT NULL").fetchall()
        return {r[0] for r in rows}

    # milestones
    def add_milestone(self, goal_id: int, title: str, due: str = "",
                      target_value: float | None = None) -> Milestone:
        with self._lock, self._db:
            pos = self._db.execute("SELECT COALESCE(MAX(position), 0) + 1 FROM milestones "
                                   "WHERE goal_id=?", (goal_id,)).fetchone()[0]
            cur = self._db.execute(
                "INSERT INTO milestones (goal_id, title, due, target_value, position) VALUES (?,?,?,?,?)",
                (goal_id, title, due, target_value, pos))
        return Milestone(int(cur.lastrowid), goal_id, title, due, target_value, False, "", pos)

    def milestones(self, goal_id: int) -> list[Milestone]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM milestones WHERE goal_id=? ORDER BY position, id",
                                    (goal_id,)).fetchall()
        return [Milestone(r["id"], r["goal_id"], r["title"], r["due"], r["target_value"],
                          bool(r["done"]), r["done_at"], r["position"]) for r in rows]

    def set_milestone_done(self, milestone_id: int, done: bool = True) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE milestones SET done=?, done_at=? WHERE id=?",
                             (1 if done else 0, _stamp() if done else "", milestone_id))

    def clear_open_milestones(self, goal_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM milestones WHERE goal_id=? AND done=0", (goal_id,))

    # key results
    def add_key_result(self, goal_id: int, title: str, unit: str = "",
                       target: float | None = None) -> KeyResult:
        with self._lock, self._db:
            cur = self._db.execute("INSERT INTO key_results (goal_id, title, unit, target) VALUES (?,?,?,?)",
                                   (goal_id, title, unit, target))
        return KeyResult(int(cur.lastrowid), goal_id, title, unit, target)

    def key_results(self, goal_id: int) -> list[KeyResult]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM key_results WHERE goal_id=? ORDER BY id",
                                    (goal_id,)).fetchall()
        return [KeyResult(r["id"], r["goal_id"], r["title"], r["unit"], r["target"],
                          float(r["current"] or 0.0), bool(r["done"])) for r in rows]

    def update_key_result(self, kr_id: int, current: float, done: bool) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE key_results SET current=?, done=? WHERE id=?",
                             (current, 1 if done else 0, kr_id))

    # entries
    def add_entry(self, goal_id: int, kind: str, amount: float | None = None, unit: str = "",
                  note: str = "", kr_id: int | None = None, at: str = "") -> Entry:
        at = at or _stamp()
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO entries (goal_id, at, kind, amount, unit, note, kr_id) VALUES (?,?,?,?,?,?,?)",
                (goal_id, at, kind, amount, unit, note, kr_id))
            self._db.execute("UPDATE goals SET last_checkin_at=?, updated_at=? WHERE id=?",
                             (at, _stamp(), goal_id))
        return Entry(int(cur.lastrowid), goal_id, at, kind, amount, unit, note, kr_id)

    def entries(self, goal_id: int, since: str = "") -> list[Entry]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM entries WHERE goal_id=? AND at>=? ORDER BY at, id",
                (goal_id, since)).fetchall()
        return [Entry(r["id"], r["goal_id"], r["at"], r["kind"], r["amount"], r["unit"],
                      r["note"], r["kr_id"]) for r in rows]

    # links
    def add_link(self, goal_id: int, kind: str, ref: str, label: str = "") -> Link:
        at = _stamp()
        with self._lock, self._db:
            cur = self._db.execute("INSERT INTO links (goal_id, kind, ref, label, created_at) "
                                   "VALUES (?,?,?,?,?)", (goal_id, kind, ref, label, at))
        return Link(int(cur.lastrowid), goal_id, kind, ref, label, at)

    def links(self, goal_id: int) -> list[Link]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM links WHERE goal_id=? ORDER BY id",
                                    (goal_id,)).fetchall()
        return [Link(r["id"], r["goal_id"], r["kind"], r["ref"], r["label"], r["created_at"])
                for r in rows]

    # nudges: the local day each one was last raised
    def nudged_days(self, keys: Iterable[str]) -> dict[str, str]:
        wanted = tuple(dict.fromkeys(str(k) for k in keys))
        if not wanted:
            return {}
        with self._lock:
            rows = self._db.execute(
                f"SELECT key, day FROM nudges WHERE key IN ({', '.join('?' * len(wanted))})",
                wanted).fetchall()
        return {r["key"]: r["day"] for r in rows}

    def mark_nudged(self, keys: Iterable[str], day: str) -> None:
        at = _stamp()
        with self._lock, self._db:
            self._db.executemany("INSERT OR REPLACE INTO nudges (key, day, at) VALUES (?,?,?)",
                                 [(str(k), day, at) for k in dict.fromkeys(keys)])


# ── the engine ────────────────────────────────────────────────────────────────

class GoalsEngine:
    """Goals with outcomes, measures, pace and plans. Synchronous: call it
    through ``asyncio.to_thread`` from the loop (see the module docstring)."""

    #: How long the proactive survey may reuse a computed summary.
    ATTENTION_MAX_AGE_S = 300.0

    def __init__(self, store: GoalStore | None = None, *, reminders: Any = None,
                 router: Any = None, companion: Any = None, cognition: Any = None,
                 bus: Any = None, import_legacy: bool = True) -> None:
        self.store = store or GoalStore()
        self.reminders = reminders
        self.router = router
        self.companion = companion
        self.cognition = cognition
        self.bus = bus
        self._attention: dict[str, Any] | None = None
        self._attention_at = 0.0
        self._refreshing = False
        self._refresh_lock = threading.Lock()
        self.imported = 0
        if import_legacy:
            try:
                self.imported = self.import_legacy()
            except Exception as exc:                    # never stops ORION starting
                self._log(f"GOALS: legacy import skipped - {first_line(exc, 120)}")

    # ── finding a goal ────────────────────────────────────────────────────────

    def find(self, ref: Any, *, include_closed: bool = False) -> Goal | None:
        """By id ('3', '#3'), exact title, slug, or the best word overlap."""
        text = _clean(ref, 200)
        if not text:
            return None
        pool = self.store.goals(None if include_closed else ("active", "paused"))
        digits = text.lstrip("#").strip()
        if digits.isdigit():
            wanted = int(digits)
            for goal in pool:
                if goal.id == wanted:
                    return goal
            goal = self.store.get_goal(wanted)
            return goal if goal is not None and goal.status != "deleted" and include_closed else None
        lowered = text.lower()
        for goal in pool:
            if goal.title.lower() == lowered or _slug(goal.title) == _slug(text):
                return goal
        for goal in pool:
            if lowered in goal.title.lower():
                return goal
        words = _words(text)
        best, best_score = None, 0.0
        for goal in pool:
            gw = _words(" ".join((goal.title, goal.metric, goal.outcome)))
            if not gw:
                continue
            score = len(words & gw) / max(1, len(words))
            if score > best_score:
                best, best_score = goal, score
        return best if best_score >= 0.5 else None

    def _require(self, ref: Any, *, include_closed: bool = False) -> Goal:
        goal = self.find(ref, include_closed=include_closed)
        if goal is None:
            names = ", ".join(f"'{g.title}'" for g in self.store.goals()[:6])
            raise LookupError(
                f"I couldn't find a goal matching '{_clean(ref, 60)}'."
                + (f" Your goals are: {names}." if names else " You have no goals yet."))
        return goal

    # ── create and edit ───────────────────────────────────────────────────────

    def create(self, title: str, *, outcome: str = "", why: str = "", area: str = "",
               priority: Any = "medium", metric: str = "", unit: str = "",
               baseline: Any = None, target: Any = None, deadline: Any = "",
               cadence: str = "weekly", mode: str = "", start_date: str = "") -> Goal:
        title = _clean(title, 160)
        if not title:
            raise ValueError("A goal needs a name — what are you working towards?")
        now = _now()
        unit_c = canonical_unit(unit)
        baseline_v, target_v = _float(baseline), _float(target)
        if target_v is None and unit_c:
            raise ValueError("You gave a unit but no target — how much is the goal?")
        goal = Goal(
            id=0, title=title, outcome=_clean(outcome, 400), why=_clean(why, 400),
            area=self._area(area, " ".join((title, outcome, metric)), unit_c),
            priority=self._priority(priority), metric=_clean(metric, 60), unit=unit_c,
            baseline=baseline_v if baseline_v is not None else (0.0 if target_v is not None else None),
            target=target_v, deadline=parse_deadline(deadline, now.date()) if deadline else "",
            start_date=(parse_deadline(start_date, now.date()) if start_date else now.date().isoformat()),
            cadence=self._cadence(cadence),
            mode=self._mode(mode, unit_c, metric, title, baseline_v, target_v),
            created_at=_stamp(now), updated_at=_stamp(now))
        if deadline and not goal.deadline:
            raise ValueError(f"I couldn't read '{_clean(deadline, 40)}' as a date — try 2026-12-31 "
                             "or '31 December'.")
        self.store.insert_goal(goal)
        self._invalidate()
        return goal

    def update(self, ref: Any, **fields: Any) -> Goal:
        goal = self._require(ref, include_closed=True)
        changes: dict[str, Any] = {}
        for key in ("title", "outcome", "why", "metric", "reflection"):
            if fields.get(key) not in (None, ""):
                changes[key] = _clean(fields[key], 400 if key != "title" else 160)
        if fields.get("area") not in (None, ""):
            changes["area"] = self._area(fields["area"], goal.title)
        if fields.get("priority") not in (None, ""):
            changes["priority"] = self._priority(fields["priority"])
        if fields.get("unit") not in (None, ""):
            changes["unit"] = canonical_unit(fields["unit"])
        for key in ("baseline", "target"):
            value = _float(fields.get(key))
            if value is not None:
                changes[key] = value
        if fields.get("deadline") not in (None, ""):
            iso = parse_deadline(fields["deadline"])
            if not iso:
                raise ValueError(f"I couldn't read '{_clean(fields['deadline'], 40)}' as a date.")
            changes["deadline"] = iso
        if fields.get("cadence") not in (None, ""):
            changes["cadence"] = self._cadence(fields["cadence"])
        if fields.get("mode") not in (None, ""):
            changes["mode"] = "level" if str(fields["mode"]).lower().startswith(("level", "latest", "reading")) else "cumulative"
        status = str(fields.get("status") or "").lower()
        if status in {"active", "paused"}:
            changes["status"] = status
        pct = _float(fields.get("progress") if fields.get("progress") is not None else fields.get("percent"))
        if pct is not None:
            changes["manual_pct"] = max(0.0, min(100.0, pct))
        if "target" in changes and goal.baseline is None and "baseline" not in changes:
            changes["baseline"] = 0.0
        if not changes:
            raise ValueError("Nothing to change — say what to update (deadline, target, priority…).")
        self.store.update_goal(goal.id, **changes)
        self._invalidate()
        return self.store.get_goal(goal.id) or goal

    @staticmethod
    def _priority(value: Any) -> int:
        text = str(value or "").strip().lower()
        if text in PRIORITIES:
            return PRIORITIES[text]
        if text in {"1", "top", "urgent", "critical"}:
            return 1
        if text in {"3", "someday", "minor"}:
            return 3
        return 2

    @staticmethod
    def _cadence(value: Any) -> str:
        text = str(value or "weekly").strip().lower()
        aliases = {"day": "daily", "every day": "daily", "week": "weekly", "every week": "weekly",
                   "biweekly": "fortnightly", "fortnight": "fortnightly", "month": "monthly",
                   "every month": "monthly", "never": "none", "off": "none"}
        text = aliases.get(text, text)
        return text if text in CADENCE_DAYS else "weekly"

    @staticmethod
    def _area(value: Any, context: str = "", unit: str = "") -> str:
        text = str(value or "").strip().lower()
        aliases = {"fitness": "health", "exercise": "health", "education": "study",
                   "learning": "study", "university": "study", "work": "career", "job": "career",
                   "money": "finance", "savings": "finance", "project": "projects",
                   "business": "projects", "life": "personal"}
        text = aliases.get(text, text)
        if text in AREAS:
            return text
        words = _words(context)
        for area, hints in _AREA_HINTS.items():
            if words & hints:
                return area
        # The unit says what the words did not: kilograms and kilometres are
        # the body, pounds sterling are money.
        dimension = unit_dimension(unit) if unit else ""
        if dimension in {"mass", "distance"}:
            return "health"
        if dimension in {"gbp", "usd", "eur"}:
            return "finance"
        return "personal"

    @staticmethod
    def _mode(value: Any, unit: str, metric: str, title: str,
              baseline: float | None, target: float | None) -> str:
        text = str(value or "").strip().lower()
        if text.startswith(("level", "latest", "reading")):
            return "level"
        if text.startswith(("cumulative", "sum", "total")):
            return "cumulative"
        # A body weight, a lift, a pace or a score is a READING: logging 82 kg
        # twice does not make 164. Distance, time and money accumulate.
        words = _words(" ".join((metric, title)))
        if words & {"weigh", "lose", "bench", "squat", "deadlift", "lift", "pace", "score",
                    "grade", "rating", "elo", "max", "pb", "personal", "best"}:
            return "level"
        if baseline is not None and target is not None and target < baseline:
            return "level"
        return "cumulative"

    # ── progress ──────────────────────────────────────────────────────────────

    def log_progress(self, ref: Any = None, *, amount: Any = None, unit: str = "",
                     text: str = "", note: str = "", key_result: Any = None,
                     absolute: Any = None) -> tuple[Goal, Entry, str]:
        """Record progress, from structured values or a sentence. Returns the
        goal, the entry and a one-line confirmation (with the new pace)."""
        parsed = parse_progress(text) if text else ParsedProgress()
        value = _float(amount)
        unit_c = canonical_unit(unit) if unit else ""
        if value is None:
            value, unit_c = parsed.amount, unit_c or parsed.unit
        is_absolute = _truthy(absolute) if absolute is not None else parsed.absolute
        goal = self.find(ref) if ref not in (None, "") else None
        if ref not in (None, "") and goal is None:
            goal = self._require(ref)
        if goal is None:
            goal = self._match_goal(parsed, unit_c, text)
        value, unit_c = self._resolve_amount(goal, value, unit_c, parsed)
        kr = self._find_key_result(goal, key_result) if key_result not in (None, "") else None
        if value is None:
            # "Went for a run" against a goal measured in sessions counts one;
            # against one measured in km it is a check-in, and we ask how far.
            if goal.unit in _COUNT_SINGULAR or goal.unit in {"times", "days", "sessions", "workouts"}:
                value, unit_c = 1.0, goal.unit
            else:
                entry = self.store.add_entry(goal.id, "note", note=_clean(note or text, 300),
                                             kr_id=kr.id if kr else None)
                self._mark_linked_habits(goal)
                self._invalidate()
                ask = (f" How much was it, in {_DISPLAY.get(goal.unit, goal.unit)}?"
                       if goal.measurable and goal.unit else "")
                return goal, entry, f"Check-in noted for '{goal.title}'.{ask}"
        target_unit = (kr.unit if kr and kr.unit else goal.unit) or unit_c
        converted = convert(value, unit_c, target_unit) if unit_c and target_unit else value
        if converted is None:
            raise ValueError(
                f"'{goal.title}' is measured in {_DISPLAY.get(goal.unit, goal.unit)}, and "
                f"{fmt_qty(value, unit_c)} doesn't convert to that.")
        kind = "set" if (is_absolute or goal.mode == "level") and not kr else "add"
        entry = self.store.add_entry(goal.id, kind, round(converted, 4), target_unit,
                                     _clean(note or text, 300), kr.id if kr else None)
        if kr is not None:
            current = kr.current + converted if not is_absolute else converted
            self.store.update_key_result(kr.id, current, kr.target is not None and current >= kr.target)
        self._mark_linked_habits(goal)
        self._invalidate()
        status = self.assess(goal)
        said = fmt_qty(converted, target_unit)
        verb = "Recorded" if kind == "set" else "Logged"
        message = f"{verb} {said} for '{goal.title}'"
        if kr is not None:
            message += f" (key result '{kr.title}')"
        if goal.measurable:
            # "88 kg of 80 kg" reads as nonsense for a goal that goes DOWN.
            joint = " of " if goal.target >= (goal.baseline or 0.0) else ", target "
            message += (f" — now {fmt_qty(status['current'], goal.unit)}{joint}"
                        f"{fmt_qty(goal.target, goal.unit)} ({status['percent']}%)")
        message += f". {status['pace'].capitalize()}."
        if status.get("projected") and status["pace"] not in {"target reached", "done"}:
            message += f" At this pace you'll finish around {_say_date(status['projected'])}."
        if status["pace"] == "target reached":
            message += " That's the target — say the word and I'll mark it complete."
        return goal, entry, message

    def _match_goal(self, parsed: ParsedProgress, unit: str, text: str) -> Goal:
        goals = [g for g in self.store.goals(("active",))]
        if not goals:
            raise LookupError("You have no active goals to log against yet — tell me the goal first.")
        words = parsed.words or _words(text)
        dimension = unit_dimension(unit) if unit else ""
        scored: list[tuple[float, Goal]] = []
        for goal in goals:
            gw = _words(" ".join((goal.title, goal.metric, goal.outcome,
                                  _DISPLAY.get(goal.unit, goal.unit))))
            score = 2.0 * len(words & gw)
            gdim = unit_dimension(goal.unit) if goal.unit else ""
            money_goal = gdim in {"gbp", "usd", "eur"}
            if parsed.money:
                score += 3.0 if money_goal else -1.0
            elif dimension and gdim:
                if dimension == gdim:
                    score += 3.0
                elif parsed.thousands and money_goal:
                    score += 2.0                     # a bare "5k" on a savings goal
                else:
                    score -= 3.0
            if words & _AREA_HINTS.get(goal.area, frozenset()):
                score += 1.0
            scored.append((score, goal))
        scored.sort(key=lambda pair: (-pair[0], pair[1].priority, pair[1].id))
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else float("-inf")
        if best_score > 0 and best_score - runner_up >= 1.0:
            return best
        if len(goals) == 1 and best_score >= 0:
            return best
        contenders = [g for s, g in scored if s >= max(best_score, 0) - 0.5][:3] or [g for _s, g in scored[:3]]
        names = " or ".join(f"'{g.title}'" for g in contenders)
        raise LookupError(f"Which goal is that for — {names}?")

    @staticmethod
    def _resolve_amount(goal: Goal, value: float | None, unit: str,
                        parsed: ParsedProgress) -> tuple[float | None, str]:
        if value is None:
            return None, unit
        gdim = unit_dimension(goal.unit) if goal.unit else ""
        if parsed.thousands and unit == "km" and gdim in {"gbp", "usd", "eur"}:
            return value * 1000.0, goal.unit
        if unit in {"gbp"} and gdim == "mass" and "£" not in parsed.raw:
            return value, "lb"                       # "pounds" on a weight goal
        if not unit:
            return value, goal.unit
        return value, unit

    def _find_key_result(self, goal: Goal, ref: Any) -> KeyResult | None:
        text = _clean(ref, 120).lower().lstrip("#")
        for kr in self.store.key_results(goal.id):
            if text == str(kr.id) or text in kr.title.lower():
                return kr
        return None

    def _mark_linked_habits(self, goal: Goal) -> None:
        if self.companion is None:
            return
        for link in self.store.links(goal.id):
            if link.kind == "habit":
                try:
                    self.companion.mark_habit(link.ref)
                except Exception:
                    pass

    # ── measurement and pace ──────────────────────────────────────────────────

    def current_value(self, goal: Goal, entries: list[Entry] | None = None) -> float | None:
        if not goal.measurable:
            return None
        entries = self.store.entries(goal.id) if entries is None else entries
        current = goal.baseline if goal.baseline is not None else 0.0
        readings = [e for e in entries if e.kr_id is None and e.kind in {"add", "set"} and e.amount is not None]
        if goal.mode == "level":
            return readings[-1].amount if readings else current
        for e in readings:
            current = e.amount if e.kind == "set" else current + e.amount
        return current

    def fraction(self, goal: Goal, current: float | None, milestones: list[Milestone],
                 key_results: list[KeyResult]) -> float:
        if goal.measurable and current is not None:
            base = goal.baseline if goal.baseline is not None else 0.0
            span = goal.target - base
            if abs(span) < 1e-12:
                return 1.0 if current == goal.target else 0.0
            return max(0.0, min(1.0, (current - base) / span))
        if goal.manual_pct is not None:
            return max(0.0, min(1.0, goal.manual_pct / 100.0))
        measured = [kr for kr in key_results if kr.target]
        if measured:
            return sum(min(1.0, kr.current / kr.target) for kr in measured) / len(measured)
        if milestones:
            return sum(1 for m in milestones if m.done) / len(milestones)
        return 0.0

    def assess(self, goal: Goal, now: datetime | None = None) -> dict[str, Any]:
        """Everything about where a goal stands, as plain data."""
        now = now or _now()
        today = now.date()
        goal = self.store.get_goal(goal.id) or goal
        entries = self.store.entries(goal.id)
        milestones = self.store.milestones(goal.id)
        key_results = self.store.key_results(goal.id)
        current = self.current_value(goal, entries)
        fraction = self.fraction(goal, current, milestones, key_results)
        try:
            start = date.fromisoformat(goal.start_date[:10]) if goal.start_date else None
        except ValueError:
            start = None
        start = start or _local_date(goal.created_at, now) or today
        deadline = None
        if goal.deadline:
            try:
                deadline = date.fromisoformat(goal.deadline[:10])
            except ValueError:
                deadline = None
        progressed = [e for e in entries if e.kind in {"add", "set"}] or (
            [] if goal.manual_pct in (None, 0) else [None])
        last_activity = (entries[-1].at if entries else goal.last_checkin_at) or goal.created_at
        last_date = _local_date(last_activity, now) or start
        idle_days = max(0, (today - last_date).days)
        cadence_days = CADENCE_DAYS.get(goal.cadence, 7)
        checkin_due = bool(cadence_days) and goal.status == "active" and idle_days >= cadence_days

        days_elapsed = max(0, (today - start).days)
        days_left = (deadline - today).days if deadline else None
        expected = None
        projected = ""
        needed_per_week = None
        if deadline:
            total = max(1, (deadline - start).days)
            expected = max(0.0, min(1.0, days_elapsed / total))
        # A projection needs a history to project from. One log on the first
        # day said "finished by Tuesday" for a three-month goal; three days of
        # elapsed time (or three separate days of logs) is the least that
        # makes a rate mean anything.
        logged_days = {_local_date(e.at, now) for e in entries if e.kind in {"add", "set"}}
        history = days_elapsed >= 3 or len(logged_days) >= 3
        if fraction > 0 and progressed and fraction < 1 and history:
            rate = fraction / max(1, days_elapsed)
            if rate > 0:
                projected = (today + timedelta(days=math.ceil((1 - fraction) / rate))).isoformat()
        if goal.measurable and current is not None and deadline and fraction < 1:
            remaining = abs(goal.target - current)
            weeks = max(1.0, (days_left or 0) / 7.0) if days_left and days_left > 0 else 1.0
            needed_per_week = remaining / weeks

        if goal.status == "completed":
            pace = "done"
        elif fraction >= 1.0:
            pace = "target reached"
        elif deadline is None:
            if not progressed:
                pace = "not started"
            elif idle_days > max(14, 2 * (cadence_days or 7)):
                pace = "stalled"
            else:
                pace = "in progress"
        elif days_left is not None and days_left < 0:
            pace = "overdue"
        elif not progressed:
            pace = "not started" if (expected or 0) < 0.15 and (days_left or 0) > 7 else (
                "behind" if (expected or 0) < 0.3 else "at risk")
        else:
            gap = (expected or 0.0) - fraction
            on_time = bool(projected) and date.fromisoformat(projected) <= deadline
            if fraction - (expected or 0) > 0.1 and (on_time or not projected):
                pace = "ahead"
            elif on_time or gap <= 0.05:
                pace = "on track"
            elif gap <= 0.2:
                pace = "behind"
            else:
                pace = "at risk"
        next_milestone = next((m for m in milestones if not m.done), None)
        due_milestones = [m for m in milestones if not m.done and m.due and m.due[:10] <= today.isoformat()]
        week_ago = (now - timedelta(days=7)).astimezone(timezone.utc).isoformat(timespec="seconds")
        week_entries = [e for e in entries if e.at >= week_ago and e.kr_id is None]
        week_amount = sum(e.amount or 0.0 for e in week_entries if e.kind == "add")
        return {
            "goal": goal, "current": current, "fraction": fraction,
            "percent": int(round(fraction * 100)), "pace": pace, "expected": expected,
            "projected": projected, "days_left": days_left, "needed_per_week": needed_per_week,
            "checkin_due": checkin_due, "idle_days": idle_days, "entries": len(entries),
            "milestones": milestones, "key_results": key_results,
            "next_milestone": next_milestone, "due_milestones": due_milestones,
            "week_entries": len(week_entries), "week_amount": week_amount,
            "links": self.store.links(goal.id),
        }

    # ── milestones, key results, links ────────────────────────────────────────

    def add_milestone(self, ref: Any, title: str, due: Any = "", target_value: Any = None) -> Milestone:
        goal = self._require(ref)
        title = _clean(title, 160)
        if not title:
            raise ValueError("What's the milestone?")
        iso = parse_deadline(due) if due else ""
        if due and not iso:
            raise ValueError(f"I couldn't read '{_clean(due, 40)}' as a date.")
        milestone = self.store.add_milestone(goal.id, title, iso, _float(target_value))
        self._invalidate()
        return milestone

    def complete_milestone(self, ref: Any, milestone: Any) -> Milestone:
        goal = self._require(ref)
        text = _clean(milestone, 160).lower().lstrip("#")
        for m in self.store.milestones(goal.id):
            if not m.done and (text == str(m.id) or (text and text in m.title.lower())
                               or text in {"next", "current", ""}):
                self.store.set_milestone_done(m.id)
                self._invalidate()
                m.done = True
                return m
        raise LookupError(f"No open milestone on '{goal.title}' matches '{_clean(milestone, 60)}'.")

    def add_key_result(self, ref: Any, title: str, unit: str = "", target: Any = None) -> KeyResult:
        goal = self._require(ref)
        title = _clean(title, 160)
        if not title:
            raise ValueError("What's the key result?")
        kr = self.store.add_key_result(goal.id, title, canonical_unit(unit), _float(target))
        self._invalidate()
        return kr

    def link(self, ref: Any, kind: str, label: str, *, when: str = "", minutes: Any = None,
             at: str = "", due: str = "") -> tuple[Link, str]:
        """Attach a habit, a task or a reminder. Reminders and tasks are made
        in the systems that own them; this keeps the reference."""
        goal = self._require(ref)
        kind = str(kind or "").strip().lower()
        label = _clean(label, 160)
        if kind == "habit":
            name = (label or goal.title).lower()
            link = self.store.add_link(goal.id, "habit", name, name)
            return link, (f"Linked the habit '{name}' to '{goal.title}' — logging progress "
                          "now ticks it off for the day.")
        if kind == "task":
            if not label:
                raise ValueError("What's the task?")
            if self.cognition is not None:
                try:
                    self.cognition.add_task(label, project=_slug(goal.title), due=parse_deadline(due) or "")
                except Exception as exc:
                    self._log(f"GOALS: task not added to awareness - {first_line(exc, 80)}")
            link = self.store.add_link(goal.id, "task", label, label)
            return link, f"Task '{label}' linked to '{goal.title}'."
        if kind == "reminder":
            if self.reminders is None:
                raise RuntimeError("The reminder service isn't running, so I can't set that.")
            text = label or f"check in on your goal: {goal.title}"
            before = {r.id for r in self._active_reminders()}
            mins = _float(minutes)
            if mins is not None or at:
                result = self.reminders.add(text=text, minutes=mins, at=_clean(at, 20))
            else:
                phrase = f"remind me {_clean(when, 60) or 'tomorrow at 9am'} to {text}"
                result = self.reminders.add(phrase=phrase)
            if not getattr(result, "ok", False):
                raise ValueError(getattr(result, "text", "That reminder wasn't set."))
            new = [r for r in self._active_reminders() if r.id not in before]
            ref_id = str(new[-1].id) if new else ""
            link = self.store.add_link(goal.id, "reminder", ref_id, text)
            return link, f"{result.text} It's linked to '{goal.title}'."
        raise ValueError("Link a 'habit', a 'task' or a 'reminder'.")

    def _active_reminders(self) -> list[Any]:
        try:
            return list(self.reminders.active())
        except Exception:
            return []

    # ── closing a goal ────────────────────────────────────────────────────────

    def complete(self, ref: Any, reflection: str = "") -> Goal:
        goal = self._require(ref)
        stamp = _stamp()
        self.store.update_goal(goal.id, status="completed", completed_at=stamp,
                               reflection=_clean(reflection, 800) or goal.reflection)
        if self.companion is not None:
            try:
                self.companion.record_achievement(f"Goal achieved: {goal.title}",
                                                  _clean(reflection, 240))
            except Exception:
                pass
        self._invalidate()
        return self.store.get_goal(goal.id) or goal

    def archive(self, ref: Any, reflection: str = "") -> Goal:
        goal = self._require(ref, include_closed=True)
        self.store.update_goal(goal.id, status="archived",
                               reflection=_clean(reflection, 800) or goal.reflection)
        self._invalidate()
        return self.store.get_goal(goal.id) or goal

    def delete(self, ref: Any) -> Goal:
        """Hidden everywhere, recoverable with restore. A slip of the tongue
        should never cost months of logged progress."""
        goal = self._require(ref, include_closed=True)
        self.store.update_goal(goal.id, status="deleted")
        self._invalidate()
        return goal

    def restore(self, ref: Any) -> Goal:
        text = _clean(ref, 160)
        pool = self.store.goals(("deleted", "archived", "completed"))
        digits = text.lstrip("#")
        goal = next((g for g in pool if (digits.isdigit() and g.id == int(digits))
                     or g.title.lower() == text.lower() or text.lower() in g.title.lower()), None)
        if goal is None:
            raise LookupError(f"No deleted or archived goal matches '{text}'.")
        self.store.update_goal(goal.id, status="active", completed_at="")
        self._invalidate()
        return self.store.get_goal(goal.id) or goal

    # ── reading it back ───────────────────────────────────────────────────────

    def overview(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Every active and paused goal, assessed, worst pace first."""
        now = now or _now()
        rows = [self.assess(g, now) for g in self.store.goals(("active", "paused"))]
        rank = {p: i for i, p in enumerate(PACE_ORDER)}
        rows.sort(key=lambda r: (rank.get(r["pace"], 99), r["goal"].priority, r["goal"].id))
        return rows

    def describe(self, ref: Any, now: datetime | None = None) -> str:
        goal = self._require(ref, include_closed=True)
        s = self.assess(goal, now)
        g = s["goal"]
        lines = [f"{g.title} — {g.area}, {g.priority_name} priority, {s['pace']}."]
        if g.outcome:
            lines.append(f"Outcome: {g.outcome}")
        if g.why:
            lines.append(f"Why it matters: {g.why}")
        if g.measurable:
            joint = " of " if g.target >= (g.baseline or 0.0) else ", target "
            lines.append(f"Progress: {fmt_qty(s['current'], g.unit)}{joint}{fmt_qty(g.target, g.unit)}"
                         f" ({s['percent']}%)" + (f", from a baseline of {fmt_qty(g.baseline, g.unit)}"
                                                  if g.baseline else "") + ".")
        else:
            lines.append(f"Progress: {s['percent']}%.")
        if g.deadline:
            left = s["days_left"]
            lines.append(f"Deadline: {_say_date(g.deadline)}"
                         + (f" ({left} day{'s' if left != 1 else ''} left)" if left is not None and left >= 0 else "")
                         + (f"; at this pace, done around {_say_date(s['projected'])}" if s["projected"] else "")
                         + ".")
        if s["needed_per_week"] and s["pace"] in {"behind", "at risk", "not started", "on track"}:
            lines.append(f"Needed: about {fmt_qty(s['needed_per_week'], g.unit)} a week from here.")
        for m in s["milestones"][:6]:
            lines.append(f"  {'✓' if m.done else '○'} {m.title}" + (f" (by {_say_date(m.due)})" if m.due else ""))
        for kr in s["key_results"][:4]:
            lines.append(f"  KR: {kr.title} — {fmt_qty(kr.current, kr.unit)}"
                         + (f" of {fmt_qty(kr.target, kr.unit)}" if kr.target is not None else ""))
        links = s["links"]
        if links:
            lines.append("Linked: " + "; ".join(f"{l.kind} '{l.label or l.ref}'" for l in links[:5]) + ".")
        if s["checkin_due"]:
            lines.append(f"Check-in due — nothing logged for {s['idle_days']} day(s).")
        if g.reflection:
            lines.append(f"Reflection: {g.reflection}")
        return "\n".join(lines)

    def review(self, days: int = 7, now: datetime | None = None) -> str:
        """The weekly review: what moved, what slipped, and what to do next."""
        now = now or _now()
        days = max(1, min(90, int(days or 7)))
        since = (now - timedelta(days=days)).astimezone(timezone.utc).isoformat(timespec="seconds")
        rows = self.overview(now)
        finished = [g for g in self.store.goals(("completed",)) if g.completed_at >= since]
        if not rows and not finished:
            return "No goals to review yet. Tell me what you're working towards and I'll track it."
        lines = [f"Goal review — the last {days} day{'s' if days != 1 else ''}:"]
        moved, quiet = [], []
        for r in rows:
            g = r["goal"]
            recent = [e for e in self.store.entries(g.id, since) if e.kr_id is None]
            added = sum(e.amount or 0.0 for e in recent if e.kind == "add")
            line = f"• {g.title}: {r['percent']}% — {r['pace']}"
            if recent:
                line += (f"; +{fmt_qty(added, g.unit)} this period" if added and g.unit
                         else f"; {len(recent)} log(s) this period")
                moved.append(line)
            else:
                quiet.append(line + "; nothing logged")
        lines.extend(moved or ["• Nothing was logged against any goal."])
        lines.extend(quiet)
        done_ms = []
        for r in rows:
            done_ms.extend(f"{m.title} ({r['goal'].title})" for m in r["milestones"]
                           if m.done and m.done_at >= since)
        if finished or done_ms:
            wins = [f"completed '{g.title}'" for g in finished] + [f"milestone {m}" for m in done_ms]
            lines.append("Wins: " + "; ".join(wins[:6]) + ".")
        advice = []
        for r in rows:
            g = r["goal"]
            if r["pace"] in {"at risk", "behind", "overdue"}:
                if r["needed_per_week"] and g.unit:
                    advice.append(f"'{g.title}' needs about {fmt_qty(r['needed_per_week'], g.unit)} "
                                  "a week to land on time")
                elif r["pace"] == "overdue":
                    advice.append(f"'{g.title}' is past its deadline — reset the date or archive it")
                else:
                    advice.append(f"'{g.title}' needs a concrete next step this week")
            elif r["pace"] in {"not started", "stalled"}:
                advice.append(f"'{g.title}' hasn't moved — pick the smallest first step")
        if advice:
            lines.append("Next week: " + "; ".join(advice[:4]) + ".")
        else:
            lines.append("Everything is on pace. Keep the cadence.")
        return "\n".join(lines)

    # ── what deserves attention (briefing + proactive survey) ─────────────────

    def attention(self, now: datetime | None = None) -> dict[str, Any]:
        """Goals at risk and today's goal actions, as plain data."""
        now = now or _now()
        rows = self.overview(now)
        at_risk = [r for r in rows if r["pace"] in {"at risk", "overdue", "behind"}
                   and r["goal"].status == "active"]
        checkins = [r for r in rows if r["checkin_due"]]
        actions: list[str] = []
        for r in rows:
            g = r["goal"]
            for m in r["due_milestones"][:2]:
                actions.append(f"milestone '{m.title}' for '{g.title}' is due")
        for r in at_risk[:3]:
            g = r["goal"]
            if r["needed_per_week"] and g.unit:
                actions.append(f"'{g.title}' needs about {fmt_qty(r['needed_per_week'], g.unit)} this week")
        for r in checkins[:3]:
            if r not in at_risk:
                actions.append(f"check in on '{r['goal'].title}'")
        return {
            "active": len(rows),
            "at_risk": [{"id": r["goal"].id, "title": r["goal"].title, "pace": r["pace"],
                         "percent": r["percent"]} for r in at_risk],
            "checkins_due": [{"id": r["goal"].id, "title": r["goal"].title,
                              "idle_days": r["idle_days"]} for r in checkins],
            "actions": actions[:6],
            "at": _stamp(now),
        }

    def refresh_attention(self) -> dict[str, Any]:
        summary = self.attention()
        with self._refresh_lock:
            self._attention, self._attention_at = summary, time.monotonic()
            self._refreshing = False
        return summary

    def attention_cached(self) -> dict[str, Any]:
        """For callers ON the event loop: never touches SQLite. Returns the
        last summary and, when it has gone stale, refreshes it on a daemon
        thread for the next caller."""
        with self._refresh_lock:
            summary = self._attention
            stale = summary is None or time.monotonic() - self._attention_at > self.ATTENTION_MAX_AGE_S
            start = stale and not self._refreshing
            if start:
                self._refreshing = True
        if start:
            def _work() -> None:
                try:
                    self.refresh_attention()
                except Exception:
                    with self._refresh_lock:
                        self._refreshing = False
            threading.Thread(target=_work, name="orion-goals-attention", daemon=True).start()
        return dict(summary or {})

    def briefing_text(self, now: datetime | None = None) -> str:
        """One or two sentences for the morning briefing; '' when there are no goals."""
        summary = self.attention(now)
        if not summary["active"]:
            return ""
        parts = [f"Goals: {summary['active']} active."]
        if summary["at_risk"]:
            parts.append("Needs attention — " + "; ".join(
                f"'{r['title']}' ({r['pace']}, {r['percent']}%)" for r in summary["at_risk"][:3]) + ".")
        else:
            parts.append("All on pace.")
        if summary["actions"]:
            parts.append("Today — " + "; ".join(summary["actions"][:3]) + ".")
        return " ".join(parts)

    #: Days before the same nudge is raised again. A goal merely behind is
    #: raised less often than one at risk; a pace not listed is never raised.
    NUDGE_REPEAT_DAYS: dict[str, int] = {"overdue": 1, "at risk": 1, "behind": 3, "milestone": 1}
    _NUDGE_LABEL = {"at risk": "Goal at risk", "behind": "Goal behind pace"}

    def take_nudges(self, now: datetime | None = None) -> list[str]:
        """A line for each active goal that is overdue, at risk or behind, and
        each milestone due, that has not been raised within its repeat window
        (NUDGE_REPEAT_DAYS, in local days). They are recorded as raised in the
        same call, in goals.db, so a restart does not raise them again.
        Blocking (SQLite): GoalNudges calls it on a worker thread."""
        now = now or _now()
        today = now.date()
        candidates: list[tuple[str, str, str]] = []
        for r in self.overview(now):
            g = r["goal"]
            if g.status != "active":
                continue
            if r["pace"] in self.NUDGE_REPEAT_DAYS:
                candidates.append((f"goal:{g.id}:{r['pace']}", r["pace"], self._nudge_text(r, today)))
            for m in r["due_milestones"][:2]:
                when = "today" if m.due[:10] == today.isoformat() else f"since {_say_date(m.due, today)}"
                candidates.append((f"milestone:{m.id}", "milestone",
                                   f"Milestone due {when}: '{m.title}' for '{g.title}'."))
        if not candidates:
            return []
        raised = self.store.nudged_days(key for key, _kind, _text in candidates)
        fresh: list[tuple[str, str]] = []
        for key, kind, text in candidates:
            try:
                gap = (today - date.fromisoformat(raised[key])).days if key in raised else None
            except ValueError:
                gap = None
            if gap is None or gap >= self.NUDGE_REPEAT_DAYS[kind]:
                fresh.append((key, text))
        if fresh:
            self.store.mark_nudged([key for key, _text in fresh], today.isoformat())
        return [text for _key, text in fresh]

    @classmethod
    def _nudge_text(cls, r: dict[str, Any], today: date) -> str:
        g = r["goal"]
        if r["pace"] == "overdue":
            return (f"Goal overdue: '{g.title}' was due {_say_date(g.deadline, today)} "
                    f"and is at {r['percent']}%.")
        text = f"{cls._NUDGE_LABEL.get(r['pace'], 'Goal')}: '{g.title}' is at {r['percent']}%"
        if r["needed_per_week"] and g.unit and g.deadline:
            text += (f" - about {fmt_qty(r['needed_per_week'], g.unit)} a week lands it by "
                     f"{_say_date(g.deadline, today)}")
        return text + "."

    def _invalidate(self) -> None:
        with self._refresh_lock:
            self._attention_at = 0.0
        if self.bus is not None:
            try:
                self.bus.dashboard_event.emit("goals", {"changed": True})
            except Exception:
                pass

    # ── plans ─────────────────────────────────────────────────────────────────

    def offline_plan(self, goal: Goal, now: datetime | None = None) -> dict[str, Any]:
        """A deterministic breakdown: evenly paced checkpoints to the deadline
        for a measured goal, the four stages every goal goes through otherwise."""
        now = now or _now()
        today = now.date()
        s = self.assess(goal, now)
        deadline = None
        if goal.deadline:
            try:
                deadline = date.fromisoformat(goal.deadline[:10])
            except ValueError:
                deadline = None
        horizon = deadline if deadline and deadline > today else today + timedelta(weeks=8)
        span = max(4, (horizon - today).days)
        milestones: list[dict[str, Any]] = []
        if goal.measurable and s["current"] is not None:
            start_v, end_v = s["current"], goal.target
            for i, share in enumerate((0.25, 0.5, 0.75, 1.0), start=1):
                value = start_v + (end_v - start_v) * share
                due = today + timedelta(days=round(span * share))
                label = (f"Reach {fmt_qty(value, goal.unit)}" if i < 4
                         else f"Hit the target: {fmt_qty(end_v, goal.unit)}")
                milestones.append({"title": label, "due": due.isoformat(), "target": round(value, 2)})
            weekly = abs(end_v - start_v) / max(1.0, span / 7.0)
            actions = [f"Schedule {fmt_qty(weekly, goal.unit)} a week into the calendar now",
                       "Log every session as you go — the pace check depends on it",
                       "Decide the minimum version for a bad week, so the streak survives it"]
        else:
            stages = ("Define done: one sentence that means this goal is achieved",
                      "First concrete step completed",
                      "Halfway review: what's working, what to cut",
                      "Finish and reflect")
            for share, title in zip((0.1, 0.35, 0.6, 1.0), stages):
                milestones.append({"title": title,
                                   "due": (today + timedelta(days=round(span * share))).isoformat(),
                                   "target": None})
            actions = ["Write the definition of done", "Do the smallest first step today",
                       "Set a weekly check-in"]
        return {"milestones": milestones, "actions": actions,
                "risk": "Losing the weekly rhythm — protect one fixed slot a week.",
                "source": "offline"}

    @staticmethod
    def parse_model_plan(answer: str) -> dict[str, Any] | None:
        """The model's JSON plan, tolerating prose or code fences around it."""
        match = re.search(r"\{.*\}", str(answer or ""), re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return None
        rows = []
        for item in (data.get("milestones") or [])[:8]:
            if isinstance(item, dict) and _clean(item.get("title"), 160):
                rows.append({"title": _clean(item.get("title"), 160),
                             "due": parse_deadline(item.get("due")) if item.get("due") else "",
                             "target": _float(item.get("target"))})
            elif isinstance(item, str) and item.strip():
                rows.append({"title": _clean(item, 160), "due": "", "target": None})
        if not rows:
            return None
        actions = [_clean(a, 160) for a in (data.get("first_actions") or data.get("actions") or [])
                   if isinstance(a, str) and a.strip()][:5]
        return {"milestones": rows, "actions": actions,
                "risk": _clean(data.get("risk"), 240), "source": "model"}

    def plan_prompt(self, goal: Goal) -> tuple[str, str]:
        s = self.assess(goal)
        today = _now().date().isoformat()
        persona = ("SPECIALIST MODE — Goal coach. Break a personal goal into 3-6 ordered "
                   "milestones, each with a due date between today and the deadline and, for a "
                   "measured goal, the value to have reached. Then the first three today-sized "
                   "actions and the single biggest risk. Reply with JSON only: "
                   '{"milestones":[{"title":"","due":"YYYY-MM-DD","target":null}],'
                   '"first_actions":[""],"risk":""}. British English.')
        facts = [f"Today: {today}", f"Goal: {goal.title}"]
        if goal.outcome:
            facts.append(f"Outcome: {goal.outcome}")
        if goal.why:
            facts.append(f"Why: {goal.why}")
        if goal.measurable:
            facts.append(f"Measure: {fmt_qty(s['current'], goal.unit)} now, target "
                         f"{fmt_qty(goal.target, goal.unit)} ({goal.mode})")
        if goal.deadline:
            facts.append(f"Deadline: {goal.deadline}")
        facts.append(f"Area: {goal.area}; pace: {s['pace']}")
        return "\n".join(facts), persona

    def save_plan(self, goal: Goal, plan: dict[str, Any], replace: bool = False) -> int:
        """Store the plan's milestones. Existing open milestones are kept
        unless *replace* — the user's own milestones outrank a generated plan."""
        existing = [m for m in self.store.milestones(goal.id) if not m.done]
        if existing and not replace:
            return 0
        if replace:
            self.store.clear_open_milestones(goal.id)
        for row in plan["milestones"]:
            self.store.add_milestone(goal.id, row["title"], row.get("due") or "", row.get("target"))
        self._invalidate()
        return len(plan["milestones"])

    @staticmethod
    def render_plan(goal: Goal, plan: dict[str, Any], saved: int) -> str:
        head = f"Plan for '{goal.title}'" + (" (offline — I'll sharpen it when a model is reachable)"
                                             if plan.get("source") == "offline" else "") + ":"
        lines = [head]
        for i, row in enumerate(plan["milestones"], start=1):
            lines.append(f"{i}. {row['title']}" + (f" — by {_say_date(row['due'])}" if row.get("due") else ""))
        if plan.get("actions"):
            lines.append("First actions: " + "; ".join(plan["actions"]) + ".")
        if plan.get("risk"):
            lines.append(f"Biggest risk: {plan['risk']}")
        lines.append(f"Saved {saved} milestone(s) to the goal." if saved else
                     "The goal already has open milestones, so I kept yours — say replace to swap them.")
        return "\n".join(lines)

    # ── one store: the older goal records ─────────────────────────────────────

    def import_legacy(self, config_dir: Path | None = None) -> int:
        """Bring goals from the companion ledger, the cognitive state and the
        forged fitness tracker into this store — read-only, once each."""
        base = Path(config_dir) if config_dir is not None else self.store.path.parent
        seen = self.store.sources()
        found: list[Goal] = []
        now = _now()
        stamp = _stamp(now)
        companion_db = base / "companion.db"
        if companion_db.exists():
            for row in _read_rows(companion_db, "SELECT slug, title, progress, target, status FROM goals"):
                key = f"companion:{row[0]}"
                if key in seen or row[4] == "dropped":
                    continue
                found.append(Goal(0, _clean(row[1], 160) or "Goal", manual_pct=float(row[2] or 0),
                                  deadline=parse_deadline(row[3]) if row[3] else "",
                                  status="completed" if row[4] == "achieved" else "active",
                                  area=self._area("", row[1]), source=key,
                                  start_date=now.date().isoformat(), created_at=stamp, updated_at=stamp))
        cognition_json = base / "cognitive_state.json"
        if cognition_json.exists():
            try:
                state = json.loads(cognition_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state = {}
            for gid, raw in ((state.get("goals") or {}) if isinstance(state, dict) else {}).items():
                if not isinstance(raw, dict) or str(raw.get("status")) == "archived":
                    continue
                key = f"cognition:{gid}"
                if key in seen:
                    continue
                prio = _float(raw.get("priority"))
                found.append(Goal(0, _clean(raw.get("title"), 160) or "Goal",
                                  outcome=_clean(raw.get("description"), 400),
                                  priority=1 if (prio or 0.5) >= 0.7 else 3 if (prio or 0.5) < 0.4 else 2,
                                  manual_pct=round(100 * max(0.0, min(1.0, _float(raw.get("progress")) or 0.0)), 1),
                                  status="completed" if raw.get("status") == "completed" else "active",
                                  area=self._area("", raw.get("title") or ""), source=key,
                                  start_date=now.date().isoformat(), created_at=stamp, updated_at=stamp))
        outcomes_db = base / "tool_outcomes.db"
        if outcomes_db.exists():
            for row in _read_rows(outcomes_db, "SELECT key, payload FROM records WHERE kind='fitness_goal'"):
                key = f"fitness_goal:{row[0]}"
                if key in seen:
                    continue
                try:
                    payload = json.loads(row[1])
                except (TypeError, ValueError):
                    continue
                found.append(Goal(0, _clean(payload.get("goal"), 160) or "Fitness goal", area="health",
                                  manual_pct=float(payload.get("progress") or 0),
                                  deadline=parse_deadline(payload.get("deadline")) if payload.get("deadline") else "",
                                  status="completed" if payload.get("status") == "Completed" else "active",
                                  source=key, start_date=now.date().isoformat(),
                                  created_at=stamp, updated_at=stamp))
        added = sum(1 for goal in found if self.store.insert_goal_if_new_source(goal))
        if added:
            self._log(f"GOALS: imported {added} goal(s) from the older goal records.")
            self._invalidate()
        return added

    def _log(self, message: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(message)
            except Exception:
                pass

    def close(self) -> None:
        self.store.close()


def _read_rows(path: Path, query: str) -> list[tuple]:
    """Read another store without the chance of writing it: a read-only URI,
    and a missing table is simply nothing to import."""
    uri = "file:" + path.resolve().as_posix() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=2.0)) as db:
            return [tuple(r) for r in db.execute(query).fetchall()]
    except sqlite3.Error:
        return []


# ── the `goals` tool ──────────────────────────────────────────────────────────
#
# One action table, documented here because a "goals coach" specialist will
# call it too. Every branch is synchronous (call it through asyncio.to_thread)
# except "plan", which needs the model and lives in plan_goal() below.

ACTIONS: dict[str, tuple[str, ...]] = {
    "create": ("create", "add", "new", "set", "set_goal"),
    "list": ("list", "overview", "status", "all", "goals"),
    "show": ("show", "details", "describe", "get", "detail"),
    "update": ("update", "edit", "change"),
    "pause": ("pause",),
    "resume": ("resume", "unpause", "reactivate"),
    "log_progress": ("log_progress", "log", "progress", "checkin", "check_in", "record", "track"),
    "add_milestone": ("add_milestone", "milestone"),
    "complete_milestone": ("complete_milestone", "milestone_done", "tick_milestone"),
    "add_key_result": ("add_key_result", "key_result", "kr"),
    "link_habit": ("link_habit", "habit"),
    "link_task": ("link_task", "task"),
    "link_reminder": ("link_reminder", "reminder", "remind"),
    "review": ("review", "weekly_review", "week"),
    "attention": ("attention", "at_risk", "today", "nudges"),
    "plan": ("plan", "break_down", "breakdown", "decompose"),
    "complete": ("complete", "done", "achieved", "finish"),
    "archive": ("archive", "shelve"),
    "delete": ("delete", "remove", "drop"),
    "restore": ("restore", "undelete", "unarchive"),
}
_ACTION_OF = {alias: name for name, aliases in ACTIONS.items() for alias in aliases}


def resolve_action(value: Any) -> str:
    text = str(value or "list").strip().lower().replace(" ", "_").replace("-", "_")
    return _ACTION_OF.get(text, text)


def _goal_ref(args: dict[str, Any]) -> Any:
    for key in ("goal", "id", "goal_id", "name"):
        if args.get(key) not in (None, ""):
            return args[key]
    return ""


def _overview_text(engine: GoalsEngine, include_closed: bool = False) -> str:
    rows = engine.overview()
    if not rows:
        closed = engine.store.goals(("completed", "archived"))
        return ("No active goals. " + (f"{len(closed)} completed or archived. " if closed else "")
                + "Tell me what you're working towards — an outcome, a measure and a date — "
                  "and I'll track the pace.")
    lines = [f"{len(rows)} active goal{'s' if len(rows) != 1 else ''}:"]
    for r in rows:
        g = r["goal"]
        measure = (f"{fmt_qty(r['current'], g.unit)} of {fmt_qty(g.target, g.unit)} ({r['percent']}%)"
                   if g.measurable else f"{r['percent']}%")
        tail = f", due {_say_date(g.deadline)}" if g.deadline else ""
        paused = " [paused]" if g.status == "paused" else ""
        lines.append(f"• #{g.id} {g.title}{paused} — {measure}, {r['pace']}{tail}")
    return "\n".join(lines)


def tool_action(engine: GoalsEngine, args: dict[str, Any]) -> Any:
    """Run one `goals` action (anything but "plan"). Returns a ToolResult."""
    from .data import ToolResult
    args = dict(args or {})
    action = resolve_action(args.get("action"))
    ref = _goal_ref(args)
    title = _clean(args.get("title") or "", 160)
    try:
        if action == "create":
            name = title or _clean(args.get("text") or args.get("goal") or "", 160)
            goal = engine.create(
                name, outcome=args.get("outcome") or "", why=args.get("why") or "",
                area=args.get("area") or "", priority=args.get("priority") or "medium",
                metric=args.get("metric") or "", unit=args.get("unit") or "",
                baseline=args.get("baseline"), target=args.get("target"),
                deadline=args.get("deadline") or "", cadence=args.get("cadence") or "weekly",
                mode=args.get("mode") or "")
            measure = (f" Target: {fmt_qty(goal.target, goal.unit)}" + (
                f" from {fmt_qty(goal.baseline, goal.unit)}" if goal.baseline else "") + "."
                if goal.measurable else " No measure yet — give me a target and unit and I can track pace.")
            due = f" Deadline {_say_date(goal.deadline)}." if goal.deadline else " No deadline set."
            return ToolResult(f"Goal #{goal.id} set: '{goal.title}' ({goal.area}, "
                              f"{goal.priority_name} priority).{measure}{due} "
                              f"I'll check in {goal.cadence if goal.cadence != 'none' else 'only when asked'}.")
        if action == "list":
            return ToolResult(_overview_text(engine))
        if action == "show":
            if not ref:
                return ToolResult(_overview_text(engine))
            return ToolResult(engine.describe(ref))
        if action == "update":
            if not ref:
                return ToolResult("Which goal should I change?", ok=False)
            fields = {k: args.get(k) for k in ("title", "outcome", "why", "area", "priority", "metric",
                                               "unit", "baseline", "target", "deadline", "cadence",
                                               "mode", "progress", "reflection")}
            goal = engine.update(ref, **fields)
            return ToolResult(f"Updated '{goal.title}'. " + engine.describe(goal.id).split("\n", 1)[0])
        if action in {"pause", "resume"}:
            goal = engine.update(ref, status="paused" if action == "pause" else "active")
            return ToolResult(f"'{goal.title}' is {'paused — no check-ins until you resume it' if action == 'pause' else 'active again'}.")
        if action == "log_progress":
            text = _clean(args.get("text") or args.get("note") or "", 300)
            if args.get("amount") in (None, "") and not text and args.get("progress") in (None, ""):
                return ToolResult("What did you do? Say it naturally — 'I ran 5k' — or give an amount.", ok=False)
            if args.get("progress") not in (None, "") and args.get("amount") in (None, ""):
                goal = engine.update(ref, progress=args.get("progress"))
                return ToolResult(f"'{goal.title}' set to {engine.assess(goal)['percent']}%.")
            _goal, _entry, message = engine.log_progress(
                ref or None, amount=args.get("amount"), unit=args.get("unit") or "", text=text,
                key_result=args.get("key_result"), absolute=args.get("absolute"))
            return ToolResult(message)
        if action == "add_milestone":
            m = engine.add_milestone(ref, title or _clean(args.get("text"), 160), args.get("due") or "",
                                     args.get("target"))
            return ToolResult(f"Milestone added: {m.title}" + (f" (by {_say_date(m.due)})" if m.due else "") + ".")
        if action == "complete_milestone":
            m = engine.complete_milestone(ref, title or args.get("milestone") or "next")
            return ToolResult(f"Milestone done: {m.title}. " + engine.describe(m.goal_id).split("\n", 1)[0])
        if action == "add_key_result":
            kr = engine.add_key_result(ref, title or _clean(args.get("text"), 160), args.get("unit") or "",
                                       args.get("target"))
            return ToolResult(f"Key result added to the goal: {kr.title}"
                              + (f", target {fmt_qty(kr.target, kr.unit)}" if kr.target is not None else "") + ".")
        if action in {"link_habit", "link_task", "link_reminder"}:
            kind = action.split("_", 1)[1]
            _link, message = engine.link(ref, kind, title or _clean(args.get("text"), 160),
                                         when=_clean(args.get("when") or "", 60), due=args.get("due") or "")
            return ToolResult(message)
        if action == "review":
            days = int(_float(args.get("days")) or 7)
            return ToolResult(engine.review(days))
        if action == "attention":
            text = engine.briefing_text()
            return ToolResult(text or "No active goals yet.")
        if action == "complete":
            goal = engine.complete(ref, _clean(args.get("reflection") or args.get("text") or "", 800))
            ask = "" if goal.reflection else " What worked, and what would you do differently? I'll keep the reflection with it."
            return ToolResult(f"'{goal.title}' marked complete — well done.{ask}")
        if action == "archive":
            goal = engine.archive(ref, _clean(args.get("reflection") or args.get("text") or "", 800))
            return ToolResult(f"'{goal.title}' archived. Its history is kept; restore it any time.")
        if action == "delete":
            goal = engine._require(ref, include_closed=True)
            if not _truthy(args.get("confirm")):
                logs = len(engine.store.entries(goal.id))
                return ToolResult(f"Delete '{goal.title}' and its {logs} progress log(s)? It stays "
                                  "recoverable with restore. Call again with confirm=true to go ahead.",
                                  ok=False)
            engine.delete(goal.id)
            return ToolResult(f"Deleted '{goal.title}'. Say 'restore {goal.title}' if that was a mistake.")
        if action == "restore":
            goal = engine.restore(ref or title)
            return ToolResult(f"'{goal.title}' is active again.")
        if action == "plan":
            return ToolResult("Planning needs the async path — call plan_goal().", ok=False)
    except (LookupError, ValueError, RuntimeError) as exc:
        return ToolResult(str(exc), ok=False)
    return ToolResult(
        f"Unsupported goals action: {action}. Use create, list, show, update, log_progress, "
        "add_milestone, complete_milestone, add_key_result, link_habit, link_task, link_reminder, "
        "review, plan, complete, archive, delete (confirm) or restore.", ok=False)


async def plan_goal(engine: GoalsEngine, args: dict[str, Any]) -> Any:
    """Break a goal into milestones: the model when one is reachable through
    ORION's router, the deterministic plan otherwise. DB work stays off the loop."""
    import asyncio

    from .data import ToolResult
    ref = _goal_ref(args) or args.get("title") or args.get("text")
    try:
        goal = await asyncio.to_thread(engine._require, ref)
    except LookupError as exc:
        return ToolResult(str(exc), ok=False)
    plan: dict[str, Any] | None = None
    router = engine.router
    if router is not None:
        try:
            if router.has_text_fallback():
                prompt, persona = await asyncio.to_thread(engine.plan_prompt, goal)
                _profile, answer = await router.generate_text(prompt, system_extra=persona)
                plan = engine.parse_model_plan(answer)
        except Exception as exc:
            engine._log(f"GOALS: model plan failed - {first_line(exc, 80)}")
    if plan is None:
        plan = await asyncio.to_thread(engine.offline_plan, goal)
    saved = await asyncio.to_thread(engine.save_plan, goal, plan, _truthy(args.get("replace")))
    return ToolResult(engine.render_plan(goal, plan, saved))


def briefing_section(engine: GoalsEngine | None = None) -> str:
    """The morning briefing's goals line. Opens the store only if it already
    exists, so a user who has never set a goal gets no phantom section."""
    if engine is None:
        if not default_path().exists():
            return ""
        engine = GoalsEngine(import_legacy=False)
        try:
            return engine.briefing_text()
        finally:
            engine.close()
    return engine.briefing_text()


# ── nudges: goals that need attention, raised while ORION is idle ─────────────

class GoalNudges:
    """Raises at-risk goals and due milestones as a banner when ORION is idle.

    Nothing is spoken: a banner, a log line and a 'goals' dashboard event. A
    pass runs only when admission has no reason to defer background work
    (memory pressure, a request being answered, a busy processor) and the
    conversation is quiet - ORION neither speaking nor mid-turn - so a nudge
    never lands on top of a conversation. take_nudges limits each item to once
    a day (a goal merely behind, every few days) and records it in goals.db.
    Each pass also refreshes the summary attention_cached serves.
    """

    LOOP_NAME = "goal_nudges"
    FIRST_DELAY_S = 180.0
    INTERVAL_S = 1800.0
    #: How soon to look again when a pass had to wait for a quiet moment.
    RETRY_S = 60.0

    def __init__(self, engine: GoalsEngine, bus: Any = None, *, admission: Any = None,
                 quiet: Any = None) -> None:
        self.engine = engine
        self.bus = bus if bus is not None else engine.bus
        self._admission = admission
        self._quiet = quiet
        self._stopped = False

    def _admission_obj(self) -> Any:
        if self._admission is not None:
            return self._admission
        from .admission import ADMISSION
        return ADMISSION

    def _listen(self) -> Any:
        """A quiet-conversation probe fed by ORION's states, or None."""
        try:
            from .semantic import ConversationQuiet
            tracker = ConversationQuiet()
            self.bus.state.connect(tracker.note_state)
        except Exception:
            return None
        return tracker.quiet

    def ready(self) -> bool:
        """Whether a pass may run now."""
        try:
            if not self._admission_obj().should_run(self.LOOP_NAME):
                return False
        except Exception:
            return False
        if self._quiet is None:
            return True
        try:
            return bool(self._quiet())
        except Exception:
            return True                 # a broken probe must not silence goals for good

    def check(self, now: datetime | None = None) -> list[str]:
        """One pass (blocking): refresh the attention summary, take the nudges."""
        self.engine.refresh_attention()
        return self.engine.take_nudges(now)

    def announce(self, texts: list[str]) -> None:
        """On the event loop: log every nudge, show the first as a banner."""
        if not texts or self.bus is None:
            return
        for text in texts:
            self.engine._log(f"GOALS: {text}")
        head = texts[0] if len(texts) == 1 else (
            f"{texts[0]} (+{len(texts) - 1} more - ask how your goals are going)")
        for signal, payload in (("banner", (head[:180], 2)),
                                ("dashboard_event", ("goals", {"nudges": list(texts)}))):
            try:
                getattr(self.bus, signal).emit(*payload)
            except Exception:
                pass

    async def run(self) -> None:
        import asyncio
        admission = self._admission_obj()
        try:
            admission.register_loop(self.LOOP_NAME, cls="background")
        except Exception:
            pass
        if self._quiet is None:
            self._quiet = self._listen()
        try:
            await asyncio.sleep(self.FIRST_DELAY_S)
            while not self._stopped:
                delay = self.RETRY_S
                if self.ready():
                    try:
                        self.announce(await asyncio.to_thread(self.check))
                    except Exception as exc:
                        self.engine._log(f"GOALS: nudge pass skipped - {first_line(exc, 100)}")
                    delay = self.INTERVAL_S
                await asyncio.sleep(delay)
        finally:
            try:
                admission.unregister_loop(self.LOOP_NAME)
            except Exception:
                pass

    def stop(self) -> None:
        self._stopped = True


__all__ = [
    "AREAS", "PRIORITIES", "CADENCE_DAYS", "PACE_ORDER", "Goal", "Milestone", "KeyResult",
    "Entry", "Link", "GoalStore", "GoalsEngine", "GoalNudges", "ParsedProgress", "parse_progress",
    "parse_deadline", "canonical_unit", "convert", "fmt_qty", "briefing_section", "default_path",
    "ACTIONS", "resolve_action", "tool_action", "plan_goal",
]
