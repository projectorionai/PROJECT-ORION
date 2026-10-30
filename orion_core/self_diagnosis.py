"""
self_diagnosis.py - "why did you stop talking at 23:41?" answered from ORION's
own records, quoted exactly.

ORION already keeps the evidence: the session log (every NET:/VOICE:/PERF:/
FACE:/TOOL: line, see session_log.py, rotated into session.log.1..3), the Live
voice journal (live_sessions.jsonl: connects, go_away, close codes such as
1008/1011), the shutdown report (last_shutdown.txt, marked FORCED when the
watchdog had to kill a hung exit), the boot report, faulthandler.log and the
crash reports. Nothing read them back for a question about a moment, so a
model asked "what happened at 23:41?" could only guess - and a guess that
names a cause with confidence is worse than "I don't know".

This module turns a question or a time into a window, gathers every record in
it from ALL of those files, classifies and ranks what it finds, and returns a
short plain-English explanation followed by the exact lines, verbatim (secrets
redacted) with their timestamps, so the model can quote them rather than
paraphrase them. When nothing in the window explains anything it says so,
names the nearest evidence and the files it searched, and never invents a
cause.

Pure functions over paths; every read is bounded (large files are tail-read)
and nothing here touches the event loop, so it is safe on a worker thread
(the diagnostics tool runs it through ``asyncio.to_thread``).
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

from .session_log import redact

#: Output caps: what a model can quote back without drowning the answer.
MAX_LINES = 25
MAX_CHARS = 3_500
#: One quoted line is cut here (marked) - a traceback is not evidence twice.
LINE_CAP = 300
#: Bounded reads. session.log rotates at 2 MB, so this covers a whole file.
LOG_TAIL_BYTES = 2_500_000
JOURNAL_TAIL_BYTES = 1_500_000
FAULT_TAIL_BYTES = 600_000
REPORT_HEAD_BYTES = 4_096
SESSION_LOG = "session.log"
ROTATED_KEEP = 3

#: Lines that show the voice conversation was alive.
ACTIVITY_TAGS = frozenset({"VOICE", "YOU", "ORION", "SR"})
#: A silence at least this long around the moment asked about is reported.
QUIET_GAP = timedelta(minutes=3)


# ── where the evidence lives ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Sources:
    diagnostics: Path       # config/diagnostics
    crash_reports: Path     # config/crash_reports

    @classmethod
    def under(cls, config_dir: Path | str) -> "Sources":
        root = Path(config_dir)
        return cls(root / "diagnostics", root / "crash_reports")


def default_sources() -> Sources:
    from .constants import CONFIG_DIR
    return Sources.under(CONFIG_DIR)


# ── the window ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime
    focus: datetime | None      # the moment named, when one was
    label: str                  # "around 23:41 on Fri 9 Oct"


_MONTHS = {name: index for index, names in enumerate((
    ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
    ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
    ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
    ("dec", "december")), start=1) for name in names}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAYS = {name: index for index, name in enumerate(
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))}
_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
                 "twenty": 20, "thirty": 30, "forty": 40, "few": 3, "couple": 2,
                 "couple of": 2}

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({_MONTH_RE})\b")
_MONTH_DAY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b")
_WEEKDAY = re.compile(r"\b(?:on\s+|last\s+)?(monday|tuesday|wednesday|thursday|friday|"
                      r"saturday|sunday)\b")
_CLOCK = re.compile(r"(?<![\d.:])([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\s*(am|pm)?(?![\d:])")
_DOT_CLOCK = re.compile(r"(?<![\d.:])([01]?\d|2[0-3])\.([0-5]\d)\s*(am|pm)\b|"
                        r"\b(?:at|around|about|by|from|since|after|before)\s+"
                        r"([01]?\d|2[0-3])\.([0-5]\d)(?![\d.])")
_HOUR_AMPM = re.compile(r"(?<![\d.:])(1[0-2]|0?[1-9])\s*(am|pm)\b")
_AT_HOUR = re.compile(r"\b(?:at|around|about|by)\s+(1\d|2[0-3]|0?\d)(?:\s*o'?clock)?(?![\d:.%])"
                      r"(?!\s*(?:sec|min|hour|hr|ms|percent|times|days?\b))")
_AGO = re.compile(r"\b(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|"
                  r"twenty|thirty|forty|few|couple(?: of)?)\s*"
                  r"(sec(?:ond)?s?|min(?:ute)?s?|h(?:ou)?rs?|hours?)\s+ago\b")
_JUST_NOW = re.compile(r"\b(?:just now|just then|a moment ago|moments ago|seconds ago|"
                       r"right now|a (?:sec|second) ago|just happened|a minute ago)\b")
_APPROX = re.compile(r"\b(?:around|about|roughly|approximately|approx|circa|sometime|"
                     r"ish|or so)\b|~")
_PARTS = {"morning": (5, 12), "afternoon": (12, 18), "evening": (18, 24), "night": (18, 30)}


def _normalise(text: str) -> str:
    text = " " + str(text or "").lower() + " "
    for dotted, plain in (("a.m.", "am"), ("p.m.", "pm"), ("a.m", "am"), ("p.m", "pm")):
        text = text.replace(dotted, plain)
    return re.sub(r"\s+", " ", text)


def _day_label(moment: datetime) -> str:
    return f"{moment:%a} {moment.day} {moment:%b}"


def _span_label(start: datetime, end: datetime) -> str:
    if start.date() == end.date():
        return f"{start:%H:%M} to {end:%H:%M} on {_day_label(start)}"
    return f"{start:%H:%M} {_day_label(start)} to {end:%H:%M} {_day_label(end)}"


def _explicit_day(text: str, today: date) -> date | None:
    def _past(year: int, month: int, day: int) -> date | None:
        try:
            found = date(year, month, day)
        except ValueError:
            return None
        if found > today:
            try:
                found = date(year - 1, month, day)
            except ValueError:
                return None
        return found

    match = _ISO_DATE.search(text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    match = _DAY_MONTH.search(text)
    if match:
        return _past(today.year, _MONTHS[match.group(2)], int(match.group(1)))
    match = _MONTH_DAY.search(text)
    if match:
        return _past(today.year, _MONTHS[match.group(1)], int(match.group(2)))
    match = _WEEKDAY.search(text)
    if match:
        back = (today.weekday() - _WEEKDAYS[match.group(1)]) % 7
        return today - timedelta(days=back)
    return None


def _clock(text: str) -> tuple[int, int, int, str | None, bool] | None:
    """(hour, minute, second, am/pm, seconds_given) for the first time named."""
    match = _CLOCK.search(text)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0),
                match.group(4), match.group(3) is not None)
    match = _DOT_CLOCK.search(text)
    if match:
        if match.group(1) is not None:
            return int(match.group(1)), int(match.group(2)), 0, match.group(3), False
        return int(match.group(4)), int(match.group(5)), 0, None, False
    match = _HOUR_AMPM.search(text)
    if match:
        return int(match.group(1)), 0, 0, match.group(2), False
    match = _AT_HOUR.search(text)
    if match:
        return int(match.group(1)), 0, 0, None, False
    if re.search(r"\b(?:noon|midday)\b", text):
        return 12, 0, 0, "pm", False
    if re.search(r"\bmidnight\b", text):
        return 0, 0, 0, None, False
    return None


def _hours(hour: int, ampm: str | None, part: str | None) -> list[int]:
    if ampm == "am":
        return [hour % 12]
    if ampm == "pm":
        return [hour % 12 + 12]
    if hour == 0 or hour > 12:
        return [hour]
    if part == "morning":
        return [hour % 12]
    if part in {"afternoon", "evening"}:
        return [hour % 12 + 12]
    return [hour % 12, hour % 12 + 12]


def resolve_window(text: str, now: datetime | None = None) -> Window:
    """The stretch of time a question is about.

    A named clock time resolves to its most recent occurrence that is not in
    the future ("23:41" asked at 10:00 is yesterday's); 12-hour forms without
    am/pm take whichever reading is most recent, unless "this morning", "last
    night" and the like settle it. "just now", "N minutes ago", "earlier",
    "this morning", "last night", "yesterday" and dates are understood; with
    nothing at all it is the last 30 minutes.
    """
    now = (now or datetime.now()).replace(microsecond=0)
    today = now.date()
    said = _normalise(text)
    approximate = bool(_APPROX.search(said))

    part: str | None = None
    anchor: date | None = None
    if re.search(r"\blast night\b", said):
        anchor, part = today - timedelta(days=1), "night"
    elif re.search(r"\b(?:tonight|this evening)\b", said):
        anchor, part = today, "evening"
    elif re.search(r"\bthis morning\b", said):
        anchor, part = today, "morning"
    elif re.search(r"\bthis afternoon\b", said):
        anchor, part = today, "afternoon"
    else:
        explicit = _explicit_day(said, today)
        if re.search(r"\byesterday\b", said):
            anchor = today - timedelta(days=1)
        elif explicit is not None:
            anchor = explicit
        elif re.search(r"\btoday\b", said):
            anchor = today
        for name in ("morning", "afternoon", "evening", "night"):
            if anchor is not None and re.search(rf"\b{name}\b", said):
                part = name
                break

    span: tuple[datetime, datetime] | None = None
    if anchor is not None and part is not None:
        low, high = _PARTS[part]
        base = datetime.combine(anchor, dtime())
        span = (base + timedelta(hours=low), min(base + timedelta(hours=high), now))

    clock = _clock(said)
    if clock is not None:
        hour, minute, second, ampm, exact = clock
        days = ([anchor, anchor + timedelta(days=1)] if anchor is not None and part == "night"
                else [anchor] if anchor is not None
                else [today, today - timedelta(days=1)])
        candidates = [datetime.combine(day, dtime(h, minute, second))
                      for day in days for h in _hours(hour, ampm, part)]
        soon = now + timedelta(minutes=1)
        inside = [c for c in candidates if c <= soon
                  and (span is None or span[0] <= c < span[0] + timedelta(hours=12))]
        past = [c for c in candidates if c <= soon]
        focus = max(inside) if inside else max(past) if past else min(candidates)
        if approximate:
            before, after = timedelta(minutes=15), timedelta(minutes=15)
        elif exact:
            before, after = timedelta(minutes=5), timedelta(minutes=10)
        else:
            before, after = timedelta(minutes=6), timedelta(minutes=15)
        prefix = "around" if approximate or not exact else "at"
        stamp = f"{focus:%H:%M:%S}" if exact else f"{focus:%H:%M}"
        return Window(focus - before, focus + after, focus,
                      f"{prefix} {stamp} on {_day_label(focus)}")

    match = _AGO.search(said)
    if match or re.search(r"\bhalf an hour ago\b", said):
        if match:
            raw = match.group(1)
            count = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, 1)
            unit = match.group(2)
            delta = (timedelta(hours=count) if unit.startswith("h")
                     else timedelta(seconds=count) if unit.startswith("s")
                     else timedelta(minutes=count))
        else:
            delta = timedelta(minutes=30)
        focus = now - delta
        slack = max(timedelta(minutes=5), delta * 0.3)
        return Window(focus - slack, min(now, focus + slack), focus,
                      f"about {_spoken(delta)} ago ({focus:%H:%M} on {_day_label(focus)})")
    if _JUST_NOW.search(said):
        return Window(now - timedelta(minutes=10), now, None,
                      f"just now (the last 10 minutes, to {now:%H:%M})")
    if span is not None:
        return Window(span[0], span[1], None,
                      f"{'last night' if part == 'night' and anchor != today else part} "
                      f"({_span_label(*span)})")
    if anchor is not None:
        start = datetime.combine(anchor, dtime())
        return Window(start, min(start + timedelta(days=1), now), None,
                      f"{'today' if anchor == today else _day_label(start)} "
                      f"({_span_label(start, min(start + timedelta(days=1), now))})")
    if re.search(r"\bearlier\b|\bbefore\b|\ba (?:while|bit) ago\b", said):
        start = now - timedelta(hours=4)
        return Window(start, now, None, f"earlier (the last 4 hours, {_span_label(start, now)})")
    start = now - timedelta(minutes=30)
    return Window(start, now, None, f"the last 30 minutes ({_span_label(start, now)})")


def _spoken(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return f"{int(delta.total_seconds())} seconds"
    if minutes < 90:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{minutes / 60:.1f} hours"


# ── reading, bounded ─────────────────────────────────────────────────────────

def _text(data: bytes) -> str:
    # Every one of these files is written in text mode, so on Windows its lines
    # end "\r\n". Bytes decoded as they are keep the "\r", and a "...=====$"
    # pattern (faulthandler's session marker) then never matches - which is
    # how a real access violation read as "0 fatal dumps". read_text() would
    # translate them; a bounded read has to do it itself.
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def _tail(path: Path, limit: int) -> str:
    """The last *limit* bytes of *path* as text, starting at a whole line."""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        start = max(0, size - limit)
        handle.seek(start)
        data = handle.read(limit)
    text = _text(data)
    if start > 0:
        cut = text.find("\n")
        text = text[cut + 1:] if cut >= 0 else ""
    return text


def _head(path: Path, limit: int = REPORT_HEAD_BYTES) -> str:
    with open(path, "rb") as handle:
        return _text(handle.read(limit))


# ── what a line means ────────────────────────────────────────────────────────

@dataclass
class Event:
    at: datetime
    source: str
    text: str                       # the line as it is in the file
    kind: str = ""
    weight: int = 0
    tag: str = ""
    detail: str = ""
    inferred: bool = False          # date reconstructed from the old time-only form
    until: datetime | None = None   # a fault known only to lie between at.. and until
    started: datetime | None = None # the session a faulthandler dump belongs to
    back: datetime | None = None    # a closed channel's recorded reconnect


_TAG = re.compile(r"^([A-Z][A-Z0-9_]{1,15}):")
# Any PERF line's first duration and what follows it, in either unit:
#   "PERF: event loop blocked 412 ms by open_stream() at audio.py:42"
#   "PERF: event loop blocked 412 ms in open() at camera.py:88"
#   "PERF: event loop STILL blocked after 3.0 s by open_stream() at audio.py:42"
_PERF_BLOCK = re.compile(r"(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|secs?|seconds?)\b\s*"
                         r"(?:(?:blocked|stalled|late)\s+)?"
                         r"(?:(?:by|in|at|from|during|inside)\s+)?(?P<culprit>.*)$", re.I)
# "  (+3 shorter block(s) not listed)" - a count, not part of the culprit.
_PERF_MORE = re.compile(r"\s*\(\+(\d+) shorter block\(s\)[^)]*\)\s*$", re.I)
_PERF_STILL = re.compile(r"\bstill blocked\b", re.I)
_CLOSE_CODE = re.compile(r"\b(1000|1001|1006|1007|1008|1009|1011|1012|1013|1014)\b")
_REENTRY = re.compile(r"cannot enter into task\s+(.+?)\s+-\s+(.+?)\s+pumped.*?\bat\s+(.+?);", re.I)

#: (kind, base weight, pattern) - first match wins, most specific first.
_RULES: tuple[tuple[str, int, re.Pattern[str]], ...] = tuple(
    (kind, weight, re.compile(pattern, re.I)) for kind, weight, pattern in (
        ("crash", 95, r"windows fatal exception|fatal python error|^CRASH:|access violation"),
        ("forced_shutdown", 90, r"forced by the shutdown watchdog"),
        ("reentry", 100, r"cannot enter into task"),
        ("voice_released", 75, r"live audio stopped arriving|voice channel was released"),
        ("live_missed", 80, r"live never answered"),
        ("live_refresh", 40, r"refreshing the live channel"),
        ("live_error", 85, r"websocket error|error code 1\d{3}|abnormal closure|"
                           r"requested entity was not found|missing required authentication|"
                           r"internal error (?:occurred|encountered)"),
        ("no_provider", 75, r"no live audio provider|no live audio or text|"
                            r"text providers exhausted|every provider is rate-limited|"
                            r"no text provider"),
        ("offline", 70, r"\b(?:went|gone|is|now|are|running) offline\b|no internet|"
                        r"internet (?:connection )?(?:lost|down|unavailable)|"
                        r"network (?:is )?unreachable|getaddrinfo failed|name resolution"),
        ("reconnect", 25, r"leaving standby|standby released"),     # coming back
        ("muted", 55, r"voice muted"),
        ("standby", 50, r"\bstandby\b"),
        ("go_away", 35, r"go_?away"),
        ("channel_closed", 55, r"live channel closed|channel (?:closed|dropped)|"
                               r"temporary connection issue|temporarily unavailable"),
        ("reconnect", 25, r"channel synchronised|re-?establish|reconnect|"
                          r"re-sending the interrupted"),
        ("face", 45, r"^FACE:.*(?:webgl context lost|renderer stalled|stopp|did not start|"
                     r"cannot draw|fail|black|could not)"),
        ("audio_device", 50, r"^AUDIO:.*(?:microphone|output device|output renderer|"
                             r"voice output|would not open|could not open|device)"),
        ("tool_error", 45, r"^TOOL:.*(?:failed|not delivered|cancelled|reconnected while)|"
                           r"tool response not delivered"),
        ("shutdown", 40, r"shutdown requested|shutdown directive|restart directive|"
                         r"clean shutdown|shutting down"),
        ("session_start", 35, r"===== session \d{4}-|process start to ready"),
        ("error", 20, r"\b(?:error|exception|failed|failure|fault|refused|timed out)\b"),
    ))
#: One cheap screen before the classifier: most log lines are routine.
_INTERESTING = re.compile("|".join(f"(?:{pattern.pattern})" for _k, _w, pattern in _RULES)
                          + r"|^PERF:", re.I)
#: Markers of recovery: never offered as the cause of anything.
_RECOVERY = frozenset({"reconnect"})


def _perf_weight(ms: float) -> int:
    # 100-249 ms blocks reach only the file: real, but rarely the story.
    return (88 if ms >= 5000 else 78 if ms >= 2000 else 62 if ms >= 1000
            else 42 if ms >= 400 else 30 if ms >= 250 else 15)


def _loop_block(body: str) -> tuple[str, int, str] | None:
    """A PERF line as (kind, weight, "<how long>|<culprit>"), whatever its
    exact wording - the watchdog's format is allowed to change."""
    found = _PERF_BLOCK.search(body)
    if not found:
        return None
    number, unit = float(found.group("n")), found.group("unit").lower()
    ms = number if unit == "ms" else number * 1000.0
    culprit = _PERF_MORE.sub("", found.group("culprit")).strip().rstrip(".")
    label = f"{ms:.0f} ms" if unit == "ms" else f"{found.group('n')} s"
    if _PERF_STILL.search(body):                # written by the watchdog mid-freeze
        label = f"over {label} (still blocked when this line was written)"
    return "loop_block", _perf_weight(ms), f"{label}|{culprit or 'an unnamed call'}"


#: Log lines repeat (the stamp is already gone), and the screen below is one
#: large case-insensitive alternation: MEASURED 26 Oct, 47,000 lines of a big
#: session.log cost 1.3 s in classify alone. The answer depends only on the
#: text, so a repeated line is classified once.
@functools.lru_cache(maxsize=8192)
def classify(body: str) -> tuple[str, int, str]:
    """(kind, weight, detail) for one log line's text (after its stamp)."""
    body = body.strip()
    tag_match = _TAG.match(body)
    tag = tag_match.group(1) if tag_match else ""
    if tag == "PERF":
        block = _loop_block(body)
        if block is not None:
            return block
    if not _INTERESTING.search(body):
        return "", 0, ""
    for kind, weight, pattern in _RULES:
        if not pattern.search(body):
            continue
        detail = ""
        if kind == "reentry":
            found = _REENTRY.search(body)
            if found:
                detail = f"{found.group(1)}|{found.group(2)}|{found.group(3)}"
        elif kind in {"live_error", "channel_closed", "error"} and tag == "NET":
            code = _CLOSE_CODE.search(body)
            if code:
                kind, weight, detail = "live_error", 85, code.group(1)
        return kind, weight, detail
    return "", 0, ""


# ── the session log ──────────────────────────────────────────────────────────

_NEW_STAMP = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\s")
_OLD_STAMP = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s")
_HEADER = re.compile(r"===== session (\d{4})-(\d{2})-(\d{2}) \d{2}:\d{2}:\d{2} \(pid \d+\) =====")
_ROLLOVER_S = 3600          # time going backwards by more than this is midnight


@dataclass
class _Row:
    raw: str
    body: str
    secs: int | None
    day: date | None
    exact: bool
    parent: int = -1        # a continuation line's entry


def parse_session_log(text: str, source: str, *, mtime: float | None = None,
                      carry: tuple[date, int] | None = None
                      ) -> tuple[list[Event], tuple[date, int] | None]:
    """Every line of one session log as an Event, oldest first.

    Lines in the old time-only form get their date from the session header
    before them (rolling over at midnight), from *carry* (the last dated line
    of the previous, older file), or - for a prefix with neither - by walking
    back from the first dated line, or from the file's modification time.
    Returns the events and the carry for the next, newer file.
    """
    rows: list[_Row] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        match = _NEW_STAMP.match(raw)
        if match:
            y, mo, d, h, mi, s = (int(g) for g in match.groups())
            try:
                day = date(y, mo, d)
            except ValueError:
                day = None
            if day is not None:
                rows.append(_Row(raw, raw[match.end():].strip(), h * 3600 + mi * 60 + s, day, True))
                continue
        match = _OLD_STAMP.match(raw)
        if match:
            h, mi, s = (int(g) for g in match.groups())
            body = raw[match.end():].strip()
            header = _HEADER.search(body)
            day = None
            if header:
                try:
                    day = date(int(header.group(1)), int(header.group(2)), int(header.group(3)))
                except ValueError:
                    day = None
            rows.append(_Row(raw, body, h * 3600 + mi * 60 + s, day, False))
            continue
        if rows:                                    # an indented continuation
            parent = rows[-1].parent if rows[-1].parent >= 0 else len(rows) - 1
            rows.append(_Row(raw, raw.strip(), None, None, False, parent))

    entries = [index for index, row in enumerate(rows) if row.parent < 0]
    current, previous = (carry if carry else (None, None))
    for index in entries:
        row = rows[index]
        if row.day is not None:
            current, previous = row.day, row.secs
            continue
        if current is not None:
            if previous is not None and row.secs + _ROLLOVER_S < previous:
                current = current + timedelta(days=1)
            row.day, previous = current, row.secs
    undated = [index for index in entries if rows[index].day is None]
    if undated:
        first_dated = next((index for index in entries if rows[index].day is not None), None)
        if first_dated is None and mtime is not None:
            last = rows[entries[-1]]
            modified = datetime.fromtimestamp(mtime)
            seconds = modified.hour * 3600 + modified.minute * 60 + modified.second
            last.day = (modified.date() if last.secs <= seconds + 120
                        else modified.date() - timedelta(days=1))
            first_dated = entries[-1]
        if first_dated is not None:
            position = entries.index(first_dated)
            later = rows[first_dated]
            for index in reversed(entries[:position]):
                row = rows[index]
                day = later.day
                if row.secs > later.secs + _ROLLOVER_S:
                    day = day - timedelta(days=1)
                row.day, later = day, row

    events: list[Event] = []
    for row in rows:
        owner = rows[row.parent] if row.parent >= 0 else row
        if owner.day is None or owner.secs is None:
            continue
        at = datetime.combine(owner.day, dtime()) + timedelta(seconds=owner.secs)
        kind, weight, detail = classify(row.body)
        tag_match = _TAG.match(owner.body)
        if row.parent >= 0 and kind in {"error", ""}:
            kind, weight = "", 0                    # a traceback line is context, not a finding
        events.append(Event(at, source, row.raw, kind, weight,
                            tag_match.group(1) if tag_match else "", detail,
                            inferred=not owner.exact))
    last_dated = next((rows[i] for i in reversed(entries) if rows[i].day is not None), None)
    out_carry = (last_dated.day, last_dated.secs) if last_dated is not None else carry
    return events, out_carry


# ── the other records ────────────────────────────────────────────────────────

_CODE_MEANING = {
    "1000": "a normal close",
    "1001": "the server going away",
    "1006": "the connection dropped with no close frame (network or server side)",
    "1007": "the server refused data ORION sent (malformed payload)",
    "1008": "the server closed it on a policy violation",
    "1009": "the server refused a message ORION sent as too large",
    "1011": "the server reported an internal error",
    "1012": "the server restarting",
    "1013": "the server asking to try again later",
    "1014": "a bad gateway",
}


def _code_meaning(code: str, summary: str) -> str:
    meaning = _CODE_MEANING.get(code, "an unexpected close")
    low = summary.lower()
    if code == "1008":
        if "authentication" in low or "credential" in low or "oauth" in low:
            meaning = "the server rejected its sign-in credential (missing or expired)"
        elif "not found for api version" in low or "is not found" in low:
            meaning = "the requested voice model is not available"
        elif "entity was not found" in low:
            meaning = "the server no longer recognised the session it was resuming"
    return meaning


def parse_live_journal(text: str, source: str) -> list[Event]:
    events: list[Event] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
            at = datetime.fromisoformat(str(row.get("at")))
        except Exception:
            continue
        if at.tzinfo is not None:
            at = at.astimezone().replace(tzinfo=None)
        event = str(row.get("event") or "")
        kind, weight, detail = "", 0, ""
        if event == "error":
            code = str(row.get("code") or "")
            kind, weight = "live_error", 85
            detail = f"{code}|{row.get('summary') or row.get('type') or ''}"
        elif event == "closed":
            reason = str(row.get("reason") or "")
            if "go_away" in reason:
                kind, weight = "go_away", 35
            elif reason == "shutdown":
                kind, weight = "shutdown", 40
            else:
                kind, weight = "channel_closed", 55
            detail = reason
        elif event == "connected":
            kind, weight = "reconnect", 25
            detail = "resumed" if row.get("resumed") else "fresh"
        elif event == "no_provider":
            kind, weight = "no_provider", 75
        elif event.startswith("tool_"):
            kind, weight, detail = "tool_error", 40, f"{event} {row.get('tool') or ''}".strip()
        events.append(Event(at, source, raw, kind, weight, "LIVE", detail))
    return events


def parse_shutdown_report(text: str, source: str) -> list[Event]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    try:
        at = datetime.strptime(lines[0][:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return []
    forced = "FORCED" in lines[0]
    events = [Event(at, source, lines[0], "forced_shutdown" if forced else "shutdown",
                    90 if forced else 40, "SHUTDOWN")]
    inside = next((line for line in lines[1:4] if line.strip().startswith("Still inside")), "")
    if inside:
        events[0].detail = inside.split(":", 1)[-1].strip()
        events.append(Event(at, source, inside.strip(), "forced_shutdown", 89, "SHUTDOWN",
                            events[0].detail))
    return events


def parse_boot_report(text: str, source: str) -> list[Event]:
    first = text.splitlines()[0].rstrip() if text.strip() else ""
    try:
        at = datetime.strptime(first[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return []
    return [Event(at, source, first, "session_start", 30, "BOOT")]


def parse_fault_log(text: str, source: str, mtime: float | None) -> list[Event]:
    """faulthandler.log: one event per session slice that holds a fatal dump.

    faulthandler records no time of its own. The last slice's dump is the
    file's last write; an older one lies somewhere before the next session
    armed - both are said plainly rather than given a false precision."""
    try:
        from .crash_reporter import _ARMED, _FATAL, analyse_fault_block
    except Exception:
        return []
    marks = list(_ARMED.finditer(text))
    events: list[Event] = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        block = text[mark.end():end]
        found = analyse_fault_block(block)
        if not found or _HANDLED_CODES.search(str(found.get("error") or "")):
            continue                        # a first-chance exception something handled
        try:
            armed = datetime.strptime(mark.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        last = index + 1 == len(marks)
        until: datetime | None = None
        if last and mtime is not None:
            at = datetime.fromtimestamp(mtime).replace(microsecond=0)
        else:
            at = armed
            if not last:
                try:
                    until = datetime.strptime(marks[index + 1].group(1), "%Y%m%d_%H%M%S")
                except ValueError:
                    until = None
        fatal = _FATAL.search(block)
        line = fatal.group(0).strip() if fatal else "fatal error"
        detail = str(found.get("summary") or "")
        exiting = bool(found.get("shutting_down"))
        if exiting:
            detail += " (the shutdown watchdog was running: it faulted while exiting)"
        # The quoted line is the dump's own fatal line; which session it
        # belongs to goes in the source label rather than into the quote.
        events.append(Event(at, f"{source} (session armed {armed:%Y-%m-%d %H:%M:%S})", line,
                            "crash", 70 if exiting else 95 if until is None else 60, "FAULT",
                            detail, inferred=until is not None, until=until, started=armed))
    return events


#: faulthandler sees first-chance exceptions too; these are ones Windows or a
#: library then handled (0x8001010d from COM was logged on 23 Sep 2026 in a
#: session that carried on). crash_reporter refuses to call them crashes.
_HANDLED_CODES = re.compile(r"0x(?:8001010d|406d1388|e06d7363|40010006|40010008)\b", re.I)
#: Plain names for the native codes a dump reports.
_NATIVE_CODES = {"0xc0000374": "heap corruption", "0xc0000005": "access violation",
                 "0xc00000fd": "stack overflow", "0xc0000409": "stack buffer overrun",
                 "0xc0000420": "assertion failure", "0x80000003": "breakpoint"}


def _place_faults(events: list[Event]) -> None:
    """Pin an older dump to its session's last record.

    Only the newest dump has a time (the file's last write); an older one is
    known only to lie between its session's start and the next start, which
    can be days. A fatal error ends the session, so it came after the last
    thing that session recorded: the range becomes that record plus ten
    minutes. With no record to pin it to, it stays wide and is weak evidence.
    """
    records = [event.at for event in events if event.tag != "FAULT"]
    for fault in events:
        if fault.tag != "FAULT" or fault.until is None or fault.started is None:
            continue
        inside = [at for at in records if fault.started <= at < fault.until]
        if inside:
            last = max(inside)
            fault.at, fault.until = last, min(fault.until, last + timedelta(minutes=10))
        else:
            fault.weight = min(fault.weight, 30)


_REPORT_STAMP = re.compile(r"^(\d{8})_(\d{6})")


def parse_crash_report(text: str, source: str, fallback: datetime | None) -> list[Event]:
    def field(name: str) -> str:
        match = re.search(rf"^{name}\s*:\s*(.*)$", text, re.M)
        return match.group(1).strip() if match else ""

    stamp = field("time").split(" (recorded")[0].strip()
    try:
        at = datetime.strptime(stamp[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        if fallback is None:
            return []
        at = fallback
    kind = field("kind")
    summary_line = next((line.strip() for line in text.splitlines()
                         if line.startswith("summary")), "")
    weight = (95 if kind in {"native-crash", "qt-fatal", "qt-gpu"}
              else 70 if kind == "native-exit" else 45 if kind == "python" else 50)
    if not summary_line:
        return []
    # The file name carries the kind; the quoted line stays verbatim.
    return [Event(at, source, summary_line, "crash", weight, "CRASH", f"{kind}|{field('summary')}")]


# ── gathering ────────────────────────────────────────────────────────────────

@dataclass
class _Searched:
    name: str
    first: datetime | None = None
    last: datetime | None = None
    present: bool = True
    note: str = ""

    def describe(self) -> str:
        if not self.present:
            return f"{self.name} (absent)"
        if self.first is None:
            return f"{self.name} ({self.note or 'nothing dated'})"
        if self.first.date() == self.last.date():
            span = f"{_day_label(self.first)} {self.first:%H:%M}-{self.last:%H:%M}"
        else:
            span = f"{_day_label(self.first)} {self.first:%H:%M} to {_day_label(self.last)} {self.last:%H:%M}"
        return f"{self.name} ({span}{', ' + self.note if self.note else ''})"


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def gather(window: Window, sources: Sources) -> tuple[list[Event], list[_Searched]]:
    """Every dated record from every source, oldest first, plus what was read."""
    events: list[Event] = []
    searched: list[_Searched] = []

    def note(name: str, found: list[Event], present: bool = True, extra: str = "") -> None:
        dated = [event.at for event in found]
        searched.append(_Searched(name, min(dated) if dated else None,
                                  max(dated) if dated else None, present, extra))

    diag = sources.diagnostics
    carry: tuple[date, int] | None = None
    for index in range(ROTATED_KEEP, -1, -1):           # oldest first
        name = SESSION_LOG if index == 0 else f"{SESSION_LOG}.{index}"
        path = diag / name
        if not path.is_file():
            if index == 0:
                note(name, [], present=False)
            continue
        try:
            text = _tail(path, LOG_TAIL_BYTES)
            found, carry = parse_session_log(text, name, mtime=_mtime(path), carry=carry)
        except OSError as exc:
            note(name, [], extra=f"unreadable: {type(exc).__name__}")
            continue
        events.extend(found)
        inferred = any(event.inferred for event in found)
        note(name, found, extra="time-only lines, dates inferred" if inferred else "")

    for name in ("live_sessions.jsonl.1", "live_sessions.jsonl"):
        path = diag / name
        if not path.is_file():
            if name == "live_sessions.jsonl":
                note(name, [], present=False)
            continue
        try:
            found = parse_live_journal(_tail(path, JOURNAL_TAIL_BYTES), name)
        except OSError:
            found = []
        events.extend(found)
        note(name, found)

    for name, parser in (("last_shutdown.txt", parse_shutdown_report),
                         ("last_boot.txt", parse_boot_report)):
        path = diag / name
        if not path.is_file():
            note(name, [], present=False)
            continue
        try:
            found = parser(_head(path), name)
        except OSError:
            found = []
        events.extend(found)
        note(name, found)

    crash_dir = sources.crash_reports
    fault = crash_dir / "faulthandler.log"
    if fault.is_file():
        try:
            found = parse_fault_log(_tail(fault, FAULT_TAIL_BYTES), fault.name, _mtime(fault))
        except OSError:
            found = []
        events.extend(found)
        searched.append(_Searched(fault.name, note=f"{len(found)} fatal dump(s)"))
    else:
        note(fault.name, [], present=False)

    reports = 0
    try:
        files = sorted(p for p in crash_dir.glob("*.txt") if p.name != "README.txt")
    except OSError:
        files = []
    near_start, near_end = window.start - timedelta(days=1), window.end + timedelta(days=1)
    for path in files[-60:]:
        match = _REPORT_STAMP.match(path.name)
        stamp = None
        if match:
            try:
                stamp = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
            except ValueError:
                stamp = None
        if stamp is not None and not near_start <= stamp <= near_end:
            continue
        try:
            found = parse_crash_report(_head(path), path.name, stamp)
        except OSError:
            continue
        reports += 1
        events.extend(found)
    if crash_dir.is_dir():
        searched.append(_Searched("crash_reports", note=f"{reports} report(s) within a day"))
    else:
        searched.append(_Searched("crash_reports", present=False))
    _place_faults(events)
    events.sort(key=lambda event: event.at)
    return events, searched


# ── ranking and the answer ───────────────────────────────────────────────────

_VOICE_WORDS = re.compile(r"talk|quiet|silent|speak|spoke|voice|answer|respond|repl|hear|"
                          r"listen|ignor|cut out|mute|deaf|stopp?ed", re.I)
_FREEZE_WORDS = re.compile(r"freez|froze|lag|stutter|stuck|slow|hang|hung|unresponsive|jank|"
                           r"flash|flicker", re.I)
_CRASH_WORDS = re.compile(r"crash|clos|die|died|disappear|vanish|quit|exit|shut", re.I)
#: "Why did you crash?" is about the crash itself, not a close code beside it.
_CRASH_ONLY = re.compile(r"crash|\bdie\b|died|disappear|vanish|fatal|segfault", re.I)
_BOOSTS = (
    (_CRASH_ONLY, {"crash"}, 15),
    (_VOICE_WORDS, {"reentry", "live_error", "voice_released", "channel_closed", "no_provider",
                    "audio_device", "standby", "muted", "offline", "shutdown",
                    "forced_shutdown"}, 10),
    (_FREEZE_WORDS, {"loop_block", "face", "reentry"}, 15),
    (_CRASH_WORDS, {"crash", "forced_shutdown", "shutdown", "session_start"}, 15),
)


def _in_window(event: Event, window: Window) -> bool:
    if event.until is not None:
        return event.at <= window.end and event.until >= window.start
    return window.start <= event.at <= window.end


#: A re-entry victim that carries the conversation (the receive loop, audio).
_VOICE_TASK = re.compile(r"receive|live|voice|audio|speech|speak|mic|listen|send|session", re.I)


def _victim(event: Event) -> str:
    return event.detail.split("|", 1)[0] if event.kind == "reentry" and "|" in event.detail else ""


def _score(event: Event, window: Window, question: str) -> float:
    score = float(event.weight)
    for words, kinds, boost in _BOOSTS:
        if event.kind in kinds and words.search(question):
            score += boost
    victim = _victim(event)
    if victim and _VOICE_WORDS.search(question) and not _VOICE_TASK.search(victim):
        # A reminders or sentinel task dying is real, but it does not explain
        # a silent voice as well as a close code beside it does.
        score -= 25
    if window.focus is not None and event.until is None:
        minutes = (event.at - window.focus).total_seconds() / 60.0
        score += max(0.0, 20.0 - 3.0 * -minutes) if minutes < 0 else max(0.0, 20.0 - 2.0 * minutes)
    elif window.end > window.start:
        score += 10.0 * (event.at - window.start) / (window.end - window.start)
    return score


def _quiet_gap(events: list[Event], window: Window) -> tuple[Event | None, Event | None] | None:
    """The longest silence of the conversation around the moment asked about."""
    session = [event for event in events if event.tag in ACTIVITY_TAGS
               and event.kind not in {"standby", "muted"}      # "silent in standby" is not talk
               and window.start - timedelta(minutes=30) <= event.at <= window.end + timedelta(hours=1)]
    if not session:
        return None
    lo = (window.focus - timedelta(minutes=2)) if window.focus else window.start
    hi = (window.focus + timedelta(minutes=5)) if window.focus else window.end
    best: tuple[Event | None, Event | None] | None = None
    best_len = QUIET_GAP
    pairs = list(zip([None] + session, session + [None]))
    for before, after in pairs:
        if before is None:                  # a silence needs a last-heard line
            continue
        start = before.at
        end = after.at if after else window.end
        if end < lo or start > hi:
            continue
        if end - start >= best_len:
            best, best_len = (before, after), end - start
    return best


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


_LEADING_STAMP = re.compile(r"^(?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2}:\d{2}\s+")


def _sentence(event: Event) -> str:
    """The plain-English reading of one finding (first person: the model
    speaks it as ORION)."""
    t = f"{event.at:%H:%M:%S}"
    text = event.text.strip()
    if text.startswith("{"):                    # a journal row: say what it was
        text = f"{event.kind.replace('_', ' ')} {event.detail}".strip()
    body = _clip(redact(_LEADING_STAMP.sub("", text)), 140)
    kind = event.kind
    if kind == "reentry":
        # The strongest evidence there is for "why did you stop talking": a
        # LOOP: line names the task that died and the call that killed it.
        parts = event.detail.split("|")
        if len(parts) == 3:
            victim, pumper, culprit = parts
            if "stepped again" in event.text:
                # Mark XXXX.4 onwards the refused step is retried, so the task
                # was held up for as long as that step lasted - not lost.
                return (f"At {t} {victim} was held up: {pumper} pumped the Windows message "
                        f"queue inside its own step (at {culprit}), and {victim} waited until "
                        "that step ended before it carried on.")
            how = (f"{pumper} pumped the Windows message queue inside its own step (at "
                   f"{culprit}), so Python refused to resume {victim} ('Cannot enter into task') "
                   "and it never ran again")
            if re.search(r"receive", victim, re.I):
                return (f"At {t} the task that receives my Live voice channel, {victim}, died: "
                        f"{how}. From then on nothing said to me was heard or answered until "
                        "the channel was rebuilt.")
            if _VOICE_TASK.search(victim):
                return (f"At {t} {victim}, part of my voice pipeline, died: {how}. Until it was "
                        "replaced, that part of listening or speaking stopped.")
            return f"At {t} the task {victim} died: {how}."
        return (f"At {t} an asyncio task died with 'Cannot enter into task': a nested Windows "
                "message pump re-entered the event loop, and the task it hit never resumed "
                "(when the voice goes silent, that task is usually the Live receive loop).")
    if kind == "live_error":
        code, _, summary = event.detail.partition("|")
        if not code.isdigit():
            found = _CLOSE_CODE.search(event.text)
            code = found.group(1) if found else ""
        summary = summary or event.text
        if code:
            return (f"At {t} the Live voice channel closed with error {code}: "
                    f"{_code_meaning(code, summary)}.")
        return f"At {t} the Live voice channel failed: {_clip(redact(summary), 120)}."
    if kind == "voice_released":
        return f"At {t} Live audio stopped arriving and the voice channel was released."
    if kind == "live_missed":
        heard = re.search(r"never answered '(.*?)' \(heard here \d+s ago; (.*?)\)", event.text)
        if heard:
            return (f"At {t} my cloud voice missed '{heard.group(1)}' - my own recogniser heard "
                    f"it, so I answered it as text ({heard.group(2)}).")
        return f"At {t} my cloud voice missed a request my own recogniser heard: {body}"
    if kind == "live_refresh":
        return (f"At {t} I refreshed my voice channel because it had stopped hearing you - the "
                "same conversation, on a fresh connection.")
    if kind == "no_provider":
        return f"At {t} no live voice provider was available, so voice fell back or stopped."
    if kind == "channel_closed":
        if event.detail == "closing for shutdown":
            return f"At {t} the Live voice channel closed because I was shutting down."
        reason = event.detail or "receive ended"
        if event.back is not None:
            return (f"At {t} the Live voice channel closed ({reason}); it was re-established "
                    f"at {event.back:%H:%M:%S}.")
        return (f"At {t} the Live voice channel closed ({reason}), and no reconnect was "
                "recorded in the next five minutes.")
    if kind == "go_away":
        return (f"At {t} the server rotated the voice channel (a routine go_away, roughly every "
                "nine minutes) - normally a gap of a second or two.")
    if kind == "loop_block":
        ms, _, culprit = event.detail.partition("|")
        return (f"At {t} my main thread was blocked for {ms} by {culprit} - while it is blocked "
                "I cannot listen, speak or draw the face.")
    if kind == "face":
        return f"At {t} the face reported a problem: {body}"
    if kind == "audio_device":
        return f"At {t} the audio devices changed or failed: {body}"
    if kind == "tool_error":
        return f"At {t} a tool call failed or was abandoned: {body}"
    if kind == "forced_shutdown":
        inside = f" (still inside: {event.detail})" if event.detail else ""
        return f"At {t} shutdown hung and was FORCED by the watchdog{inside}."
    if kind == "shutdown":
        return f"At {t} I was shutting down: {body}"
    if kind == "crash":
        kind_name, _, summary = event.detail.partition("|")
        what = summary or kind_name or event.text
        for code, name in _NATIVE_CODES.items():
            what = re.sub(rf"(?i)\b{code}\b(?! \()", f"{code} ({name})", what, count=1)
        what = _clip(redact(what), 220)
        if event.until is not None:
            started = (f" that started {event.started:%H:%M} on {_day_label(event.started)}"
                       if event.started is not None else "")
            return (f"A fatal native error ended my session{started}, some time from "
                    f"{_span_label(event.at, event.until)} (faulthandler records no time; "
                    f"{event.at:%H:%M} is that session's last record): {what}")
        if kind_name == "native-exit" or "faulted while exiting" in event.detail:
            return f"At {t} I faulted natively while exiting (the session's work was over): {what}"
        if kind_name == "python":
            return f"At {t} an internal Python error was recorded: {what}"
        return f"At {t} I crashed: {what}"
    if kind == "offline":
        return f"At {t} the network went offline: {body}"
    if kind == "standby":
        return f"At {t} I was in or entering standby (deliberately silent): {body}"
    if kind == "muted":
        return f"At {t} my voice was muted - replies went to text instead: {body}"
    if kind == "session_start":
        return f"At {t} a new session of mine started - the previous one had ended."
    return f"At {t} an error was logged: {body}"


def _quote(event: Event) -> str:
    text = redact(event.text.rstrip())
    if len(text) > LINE_CAP:
        text = text[:LINE_CAP].rstrip() + " [...]"
    where = event.source
    if event.inferred and event.until is None:
        where += f", date inferred {event.at:%Y-%m-%d}"
    return f"- {where} | {text}"


def explain(question: str = "", when: str = "", *, sources: Sources | None = None,
            now: datetime | None = None) -> str:
    """The answer to "what happened at <time>?" - never raises, never guesses."""
    try:
        return _explain(question, when, sources or default_sources(), now)
    except Exception as exc:                     # a diagnostic must not become the incident
        return (f"I could not read my own logs just now ({type(exc).__name__}: "
                f"{_clip(redact(str(exc)), 160)}), so I cannot say what happened.")


def _now() -> datetime:
    """The clock (a seam for tests that drive the whole tool path)."""
    return datetime.now()


def _mark_shutdown_effects(events: list[Event]) -> None:
    """What a shutdown causes is not the cause. A voice channel that closes as
    ORION shuts down is relabelled and demoted, so "had to be re-established"
    is never said of a channel nobody wanted back; a native fault on the way
    out (the session's work already over) is demoted below the shutdown that
    preceded it. *events* is sorted by time."""
    stops = [event.at for event in events if event.kind in {"shutdown", "forced_shutdown"}]
    if not stops:
        return

    def near(at: datetime, after: int) -> bool:
        return any(stop - timedelta(seconds=30) <= at <= stop + timedelta(seconds=after)
                   for stop in stops)

    for event in events:
        if event.kind == "channel_closed" and near(event.at, 120):
            event.detail, event.weight = "closing for shutdown", 20
        elif (event.kind == "crash" and event.until is None and near(event.at, 300)
              and ("faulted while exiting" in event.detail
                   or event.detail.startswith("native-exit|"))):
            event.weight = min(event.weight, 35)


def _mark_reconnects(events: list[Event]) -> None:
    """Note when each closed channel actually came back, so a close is never
    described as recovered unless a reconnect was recorded after it."""
    backs = [event.at for event in events if event.kind in _RECOVERY]
    for event in events:
        if event.kind == "channel_closed" and event.detail != "closing for shutdown":
            event.back = next((at for at in backs
                               if event.at <= at <= event.at + timedelta(minutes=5)), None)


def _explain(question: str, when: str, sources: Sources, now: datetime | None) -> str:
    now = (now or _now()).replace(microsecond=0)
    window = resolve_window(f"{when} {question}".strip() if when else question, now)
    events, searched = gather(window, sources)
    _mark_shutdown_effects(events)
    _mark_reconnects(events)
    searched_line = "Searched: " + "; ".join(item.describe() for item in searched) + "."
    if not events:
        return (f"I have no records to read for {window.label}: there is no session log, Live "
                "journal, shutdown report or crash record in config/diagnostics or "
                "config/crash_reports yet (the session log is written only by a real run of "
                f"ORION). I cannot tell what happened, and I will not guess.\n{searched_line}")

    asked = f"{question} {when}"
    inside = [event for event in events if _in_window(event, window)]
    findings = [event for event in inside if event.weight > 0]
    causes = sorted((event for event in findings if event.kind not in _RECOVERY),
                    key=lambda event: _score(event, window, asked), reverse=True)
    gap = _quiet_gap(events, window) if (_VOICE_WORDS.search(asked) or not causes) else None

    if not causes:
        return _nothing_found(window, events, inside, gap, searched_line)

    primary = causes[0]
    summary = [f"What most likely happened {window.label} "
               f"(searched {_span_label(window.start, window.end)}):",
               _sentence(primary)]
    others: list[str] = []
    seen_kinds = {primary.kind}
    for event in causes[1:]:
        if event.kind in seen_kinds or event.weight < 35:
            continue
        seen_kinds.add(event.kind)
        others.append(_sentence(event))
        if len(others) == 2:
            break
    if others:
        summary.append("Also in that window: " + " ".join(others))
    blocks = [(event, _block_ms(event)) for event in findings if event.kind == "loop_block"]
    if len(blocks) >= 2:
        worst, worst_ms = max(blocks, key=lambda pair: pair[1])
        summary.append(f"Main-thread blocks logged in that window: {len(blocks)}, worst "
                       f"{worst_ms:.0f} ms at {worst.at:%H:%M:%S} "
                       f"({worst.detail.partition('|')[2]}).")
    must: list[Event] = [primary]
    if gap is not None:
        before, after = gap
        if before is not None and after is not None:
            minutes = (after.at - before.at).total_seconds() / 60
            summary.append(f"Silence: no voice activity (VOICE/YOU/ORION lines) from "
                           f"{before.at:%H:%M:%S} to {after.at:%H:%M:%S} - {minutes:.0f} minutes.")
        elif before is not None:
            summary.append(f"Silence: no voice activity after {before.at:%H:%M:%S} in the log.")
        must.extend(event for event in (before, after) if event is not None)
    # The channel coming back AFTER the cause: after a dump's whole time range
    # (a session's own connect before it crashed is not a recovery), and not
    # at all after a shutdown, whose next connect is simply the next run.
    def after_cause(event: Event) -> bool:
        if primary.until is not None:
            return event.at > primary.until
        return event.at >= primary.at

    recovery = None
    if primary.kind not in {"shutdown", "forced_shutdown"}:
        recovery = next((event for event in events if event.kind in _RECOVERY
                         and after_cause(event)
                         and event.at <= window.end + timedelta(hours=1)), None)
    if recovery is not None:
        if recovery.text.lstrip().startswith("{"):
            what = ("the Live voice channel connected again"
                    + (" (session resumed)" if recovery.detail == "resumed" else ""))
        else:
            what = _clip(redact(_LEADING_STAMP.sub("", recovery.text.strip())), 120)
        summary.append(f"Recovery: at {recovery.at:%H:%M:%S} ({recovery.source}) - {what}")
        must.append(recovery)
    if any(event.inferred and event.until is None for event in must + causes[:5]):
        summary.append("(Some lines are from the older time-only log format; their dates "
                       "were inferred from the session headers.)")

    head = "\n".join(summary)
    chosen: list[Event] = []
    for event in must + causes:
        if event not in chosen:
            chosen.append(event)
    return _compose(head, chosen, searched_line)


def _block_ms(event: Event) -> float:
    found = _PERF_BLOCK.search(event.text)
    if not found:
        return 0.0
    number = float(found.group("n"))
    return number if found.group("unit").lower() == "ms" else number * 1000.0


#: At most this many quoted lines that differ only in their numbers.
_SAME_LINE_QUOTES = 3


def _signature(event: Event) -> str:
    return event.kind + ":" + re.sub(r"\d+", "#", _LEADING_STAMP.sub("", event.text.strip()))[:100]


def _compose(head: str, ranked: list[Event], searched_line: str) -> str:
    """Head, then up to MAX_LINES quoted lines (chronological), then the
    sources - the whole kept under MAX_CHARS. A line repeated with only its
    numbers changed (forty PERF lines naming one culprit) is quoted at most
    three times, so repetition cannot crowd out the rest of the evidence."""
    lead = ("Exact log lines (verbatim, oldest first, secrets redacted) - quote these "
            "rather than paraphrasing their times")
    footer = "\n" + _clip(searched_line, 700)
    budget = MAX_CHARS - len(head) - len(lead) - len(footer) - 60
    picked: list[Event] = []
    used = repeats = 0
    seen: dict[str, int] = {}
    for event in ranked:
        if len(picked) >= MAX_LINES:
            break
        key = _signature(event)
        if seen.get(key, 0) >= _SAME_LINE_QUOTES:
            repeats += 1
            continue
        cost = len(_quote(event)) + 1
        if used + cost > budget:
            continue
        picked.append(event)
        seen[key] = seen.get(key, 0) + 1
        used += cost
    picked.sort(key=lambda event: event.at)
    body = "\n".join(_quote(event) for event in picked)
    if repeats:
        lead += f" ({repeats} near-identical repeat(s) not quoted)"
    text = f"{head}\n{lead}:\n{body}{footer}" if picked else f"{head}{footer}"
    return text[:MAX_CHARS]


def _nothing_found(window: Window, events: list[Event], inside: list[Event],
                   gap: tuple[Event | None, Event | None] | None, searched_line: str) -> str:
    lines = [f"I found nothing in my records that explains {window.label} "
             f"(searched {_span_label(window.start, window.end)}): no voice-channel drop, "
             "error, stall, crash or shutdown was logged then. I will not guess at a cause."]
    quoted: list[Event] = []
    if gap is not None and gap[0] is not None and gap[1] is not None:
        minutes = (gap[1].at - gap[0].at).total_seconds() / 60
        lines.append(f"There is a {minutes:.0f}-minute stretch with no voice activity "
                     f"({gap[0].at:%H:%M:%S} to {gap[1].at:%H:%M:%S}), but nothing logged says why.")
        quoted.extend(gap)
    if inside:
        focus = window.focus or window.end
        nearest = sorted(inside, key=lambda event: abs((event.at - focus).total_seconds()))[:4]
        lines.append(f"The log has {len(inside)} routine line(s) in that window; the closest:")
        quoted.extend(event for event in nearest if event not in quoted)
    else:
        first, last = events[0].at, events[-1].at
        if window.end < first or window.start > last:
            lines.append(f"My records do not cover that time: they run from "
                         f"{first:%H:%M} {_day_label(first)} to {last:%H:%M} {_day_label(last)}.")
        else:
            lines.append("Nothing at all was logged in that window.")
    notable = [event for event in events if event.weight >= 40 and event.kind not in _RECOVERY
               and not _in_window(event, window)]
    if notable:
        centre = window.focus or window.end
        nearest = sorted(notable, key=lambda event: abs((event.at - centre).total_seconds()))[:2]
        for event in nearest:
            gap_min = (event.at - centre).total_seconds() / 60
            direction = "after" if gap_min > 0 else "before"
            lines.append(f"Nearest notable evidence, {_spoken(timedelta(minutes=abs(gap_min)))} "
                         f"{direction}: {_sentence(event)}")
            if event not in quoted:
                quoted.append(event)
    return _compose("\n".join(lines), quoted, searched_line)


__all__ = ["Window", "Event", "Sources", "resolve_window", "classify", "gather",
           "parse_session_log", "parse_live_journal", "explain", "default_sources",
           "MAX_LINES", "MAX_CHARS"]
