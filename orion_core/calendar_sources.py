"""
Calendar sources — where ORION learns when you are busy.

``scheduling.py`` (Mark X.12 §2.4) was written as "the pure engine" for "find
me 90 minutes this week for X", with the calendars it should consult left as
"the integration on top". That integration was never written: nothing imported
the engine, and the 2026-10-03 audit found the capability absent from every
tool. This module is that missing half: it turns each calendar ORION can reach
into the busy intervals the engine subtracts.

Sources, each optional and each reported when it could not be read:

* **ICS feeds** — every major calendar publishes one (Outlook.com and Microsoft
  365 "publish calendar", Google's "secret address in iCal format", iCloud
  public calendars), so this one source covers them all without an account
  integration per vendor. Listed in ``config/calendars.json``::

      {"ics": ["https://calendar.google.com/calendar/ical/…/basic.ics",
               "C:/Users/me/Documents/work.ics"]}

* **Outlook desktop** — the default calendar over COM, recurrences expanded.
* **Notion** — the configured calendar database's dated entries.

The ICS reader implements what real calendars actually emit: folded lines,
UTC/zoned/floating times (including Outlook's Windows zone names), all-day
events, DURATION, RRULE (DAILY, WEEKLY with BYDAY, MONTHLY by day or by
ordinal weekday, YEARLY; INTERVAL, COUNT, UNTIL), EXDATE and moved occurrences
(RECURRENCE-ID). Free, transparent, tentative-free and cancelled entries do not
block time; an all-day entry blocks only when it is marked busy or
out-of-office, because most all-day entries (birthdays, holidays, reminders)
are not appointments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterable, Optional

#: Upper bound on occurrences expanded from one recurring event.
MAX_OCCURRENCES = 2000

#: Outlook writes Windows zone names into TZID; zoneinfo knows IANA names.
WINDOWS_ZONES = {
    "gmt standard time": "Europe/London",
    "greenwich standard time": "Atlantic/Reykjavik",
    "w. europe standard time": "Europe/Berlin",
    "romance standard time": "Europe/Paris",
    "central europe standard time": "Europe/Budapest",
    "central european standard time": "Europe/Warsaw",
    "e. europe standard time": "Europe/Chisinau",
    "fle standard time": "Europe/Kiev",
    "gtb standard time": "Europe/Bucharest",
    "eastern standard time": "America/New_York",
    "central standard time": "America/Chicago",
    "mountain standard time": "America/Denver",
    "pacific standard time": "America/Los_Angeles",
    "india standard time": "Asia/Kolkata",
    "china standard time": "Asia/Shanghai",
    "tokyo standard time": "Asia/Tokyo",
    "aus eastern standard time": "Australia/Sydney",
    "utc": "UTC",
    "coordinated universal time": "UTC",
}

_WEEKDAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}


@dataclass(frozen=True)
class BusyBlock:
    start: datetime
    end: datetime
    title: str = ""
    source: str = ""


# ── time values ─────────────────────────────────────────────────────────────

def _zone_named(name: str, fallback: tzinfo) -> tzinfo:
    name = str(name or "").strip().strip('"')
    if not name:
        return fallback
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(WINDOWS_ZONES.get(name.lower(), name))
    except Exception:
        return fallback


def _parse_value(value: str, params: dict[str, str], zone: tzinfo
                 ) -> tuple[datetime, bool]:
    """An ICS DATE or DATE-TIME as an aware datetime, and whether all-day."""
    value = value.strip()
    if params.get("VALUE", "").upper() == "DATE" or re.fullmatch(r"\d{8}", value):
        day = datetime.strptime(value[:8], "%Y%m%d").date()
        return datetime.combine(day, time(0, 0), tzinfo=zone), True
    utc = value.endswith("Z")
    moment = datetime.strptime(value.rstrip("Z")[:15], "%Y%m%dT%H%M%S")
    if utc:
        return moment.replace(tzinfo=timezone.utc), False
    return moment.replace(tzinfo=_zone_named(params.get("TZID", ""), zone)), False


def _parse_duration(text: str) -> Optional[timedelta]:
    match = re.fullmatch(
        r"([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        text.strip())
    if not match:
        return None
    sign, weeks, days, hours, minutes, seconds = match.groups()
    span = timedelta(weeks=int(weeks or 0), days=int(days or 0), hours=int(hours or 0),
                     minutes=int(minutes or 0), seconds=int(seconds or 0))
    return -span if sign == "-" else span


# ── ICS structure ───────────────────────────────────────────────────────────

def _unfold(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        elif raw:
            lines.append(raw)
    return lines


def _split_property(line: str) -> tuple[str, dict[str, str], str]:
    head, _, value = line.partition(":")
    parts = head.split(";")
    params: dict[str, str] = {}
    for part in parts[1:]:
        key, _, val = part.partition("=")
        params[key.upper()] = val
    return parts[0].upper(), params, value


def _events(text: str) -> list[list[tuple[str, dict[str, str], str]]]:
    events: list[list[tuple[str, dict[str, str], str]]] = []
    current: Optional[list[tuple[str, dict[str, str], str]]] = None
    depth = 0
    for line in _unfold(text):
        name, params, value = _split_property(line)
        if name == "BEGIN" and value.upper() == "VEVENT":
            current, depth = [], 0
            continue
        if current is None:
            continue
        if name == "BEGIN":                  # VALARM and friends: skip their props
            depth += 1
            continue
        if name == "END":
            if value.upper() == "VEVENT" and depth == 0:
                events.append(current)
                current = None
            else:
                depth = max(0, depth - 1)
            continue
        if depth == 0:
            current.append((name, params, value))
    return events


def _first(props, name: str):
    return next(((p, v) for n, p, v in props if n == name), None)


# ── recurrence ──────────────────────────────────────────────────────────────

def _rule(text: str) -> dict[str, str]:
    return {k.upper(): v for k, _, v in (part.partition("=") for part in text.split(";")) if k}


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> Optional[date]:
    if nth > 0:
        first = date(year, month, 1)
        day = first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (nth - 1))
        return day if day.month == month else None
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    day = last - timedelta(days=(last.weekday() - weekday) % 7 + 7 * (-nth - 1))
    return day if day.month == month else None


class _SeriesEnded(Exception):
    """COUNT or UNTIL reached. Not StopIteration: raised inside a generator
    that becomes a RuntimeError (PEP 479)."""


def _occurrences(start: datetime, rule: dict[str, str], window_start: datetime,
                 until_bound: datetime) -> Iterable[datetime]:
    """Occurrence starts of *rule* from *start* (inclusive), in order, up to
    *until_bound*. A series with no COUNT is fast-forwarded to just before
    *window_start*: a daily stand-up begun years ago would otherwise use up
    the expansion budget long before reaching the week being searched."""
    freq = rule.get("FREQ", "").upper()
    interval = max(1, int(rule.get("INTERVAL", "1") or 1))
    count = int(rule["COUNT"]) if rule.get("COUNT", "").isdigit() else None
    until: Optional[datetime] = None
    if rule.get("UNTIL"):
        try:
            until, _ = _parse_value(rule["UNTIL"], {}, start.tzinfo or timezone.utc)
            if len(rule["UNTIL"].strip()) == 8:          # a date: inclusive of that day
                until = until + timedelta(days=1) - timedelta(seconds=1)
        except ValueError:
            until = None
    byday = [d.strip().upper() for d in rule.get("BYDAY", "").split(",") if d.strip()]
    produced = 0
    skip = count is None and window_start > start      # safe only without COUNT

    def admit(moment: datetime) -> bool:
        nonlocal produced
        if moment < start:
            return False
        if until is not None and moment > until:
            raise _SeriesEnded
        if count is not None and produced >= count:
            raise _SeriesEnded
        produced += 1
        return True

    def run() -> Iterable[datetime]:
        cycles = 0
        if freq == "DAILY":
            moment = start
            if skip:
                jumps = max(0, (window_start - start).days // interval - 1)
                moment = start + timedelta(days=jumps * interval)
            while moment <= until_bound and cycles < MAX_OCCURRENCES:
                if admit(moment):
                    yield moment
                moment += timedelta(days=interval)
                cycles += 1
        elif freq == "WEEKLY":
            days = sorted({_WEEKDAYS[d[-2:]] for d in byday if d[-2:] in _WEEKDAYS}) \
                or [start.weekday()]
            week = start - timedelta(days=start.weekday())
            if skip:
                jumps = max(0, (window_start - week).days // (7 * interval) - 1)
                week += timedelta(weeks=jumps * interval)
            while week <= until_bound and cycles < MAX_OCCURRENCES:
                for wd in days:
                    moment = week + timedelta(days=wd)
                    if admit(moment):
                        yield moment
                week += timedelta(weeks=interval)
                cycles += 1
        elif freq == "MONTHLY":
            year, month = start.year, start.month
            if skip:
                elapsed = (window_start.year - year) * 12 + (window_start.month - month)
                jumps = max(0, elapsed // interval - 1)
                month += jumps * interval
                year += (month - 1) // 12
                month = (month - 1) % 12 + 1
            while cycles < MAX_OCCURRENCES:
                candidates: list[datetime] = []
                if byday:
                    for spec in byday:
                        m = re.fullmatch(r"([+-]?\d+)?(MO|TU|WE|TH|FR|SA|SU)", spec)
                        if not m:
                            continue
                        nth = int(m.group(1) or 1)
                        day = _nth_weekday(year, month, _WEEKDAYS[m.group(2)], nth)
                        if day is not None:
                            candidates.append(start.replace(year=day.year, month=day.month,
                                                            day=day.day))
                else:
                    try:
                        candidates.append(start.replace(year=year, month=month))
                    except ValueError:          # the 31st in a 30-day month: skipped
                        pass
                if start.replace(year=year, month=month, day=1) > until_bound:
                    return
                for moment in sorted(candidates):
                    if admit(moment):
                        yield moment
                month += interval
                year += (month - 1) // 12
                month = (month - 1) % 12 + 1
                cycles += 1
        elif freq == "YEARLY":
            year = start.year
            if skip:
                year += max(0, (window_start.year - year) // interval - 1) * interval
            while cycles < MAX_OCCURRENCES:
                try:
                    moment: Optional[datetime] = start.replace(year=year)
                except ValueError:              # 29 February
                    moment = None
                if moment is not None:
                    if moment > until_bound:
                        return
                    if admit(moment):
                        yield moment
                year += interval
                cycles += 1
        else:
            if admit(start):
                yield start

    try:
        yield from run()
    except _SeriesEnded:
        return


# ── public: ICS ─────────────────────────────────────────────────────────────

def parse_ics(text: str, window_start: datetime, window_end: datetime,
              zone: tzinfo, source: str = "ics") -> list[BusyBlock]:
    """Busy blocks from ICS *text* that overlap [window_start, window_end)."""
    events = _events(text)
    moved: set[tuple[str, datetime]] = set()
    for props in events:
        rid = _first(props, "RECURRENCE-ID")
        uid = _first(props, "UID")
        if rid and uid:
            try:
                moved.add((uid[1], _parse_value(rid[1], rid[0], zone)[0]
                           .astimezone(timezone.utc)))
            except ValueError:
                pass

    blocks: list[BusyBlock] = []
    for props in events:
        try:
            block_list = _event_blocks(props, window_start, window_end, zone, source, moved)
        except (ValueError, KeyError):
            continue                               # one malformed event is skipped
        blocks.extend(block_list)
    return sorted(blocks, key=lambda b: b.start)


def _event_blocks(props, window_start, window_end, zone, source, moved) -> list[BusyBlock]:
    status = (_first(props, "STATUS") or ({}, ""))[1].upper()
    transp = (_first(props, "TRANSP") or ({}, ""))[1].upper()
    busy_status = (_first(props, "X-MICROSOFT-CDO-BUSYSTATUS") or ({}, ""))[1].upper()
    if status == "CANCELLED" or transp == "TRANSPARENT" or busy_status in {"FREE", "WORKINGELSEWHERE"}:
        return []
    started = _first(props, "DTSTART")
    if started is None:
        return []
    start, all_day = _parse_value(started[1], started[0], zone)
    if all_day and not (busy_status in {"BUSY", "OOF"} or transp == "OPAQUE"):
        return []
    ended = _first(props, "DTEND")
    if ended is not None:
        end = _parse_value(ended[1], ended[0], zone)[0]
    else:
        span = _parse_duration((_first(props, "DURATION") or ({}, ""))[1] or "")
        end = start + (span if span is not None else (timedelta(days=1) if all_day else timedelta(0)))
    length = end - start
    if length <= timedelta(0):
        return []
    title = (_first(props, "SUMMARY") or ({}, ""))[1].replace("\\,", ",").replace("\\n", " ")
    uid = (_first(props, "UID") or ({}, ""))[1]
    is_override = _first(props, "RECURRENCE-ID") is not None
    excluded: set[datetime] = set()
    for name, params, value in props:
        if name == "EXDATE":
            for item in value.split(","):
                try:
                    excluded.add(_parse_value(item, params, zone)[0].astimezone(timezone.utc))
                except ValueError:
                    continue

    rule = _first(props, "RRULE")
    starts: Iterable[datetime]
    if rule is not None and not is_override:
        starts = _occurrences(start, _rule(rule[1]), window_start, window_end)
    else:
        starts = [start]
    blocks: list[BusyBlock] = []
    for occurrence in starts:
        key = occurrence.astimezone(timezone.utc)
        if key in excluded or (not is_override and rule is not None and (uid, key) in moved):
            continue
        finish = occurrence + length
        if finish <= window_start or occurrence >= window_end:
            continue
        blocks.append(BusyBlock(occurrence.astimezone(zone), finish.astimezone(zone),
                                title, source))
    return blocks


# ── configured feeds ────────────────────────────────────────────────────────

def configured_feeds(config_dir: Path | None = None) -> list[str]:
    """ICS URLs and paths from ``calendars.json``; [] when none are set."""
    if config_dir is None:
        from .constants import CONFIG_DIR
        config_dir = CONFIG_DIR
    try:
        data = json.loads((Path(config_dir) / "calendars.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    feeds = data.get("ics") if isinstance(data, dict) else None
    return [str(f).strip() for f in (feeds or []) if str(f).strip()]


def read_feed(location: str, timeout: float = 15.0) -> str:
    """The text of one ICS feed: an http(s) URL or a local file. Blocking."""
    if re.match(r"^(?:https?|webcal)://", location, re.IGNORECASE):
        import urllib.request
        url = re.sub(r"^webcal://", "https://", location, flags=re.IGNORECASE)
        request = urllib.request.Request(url, headers={"User-Agent": "ORION-calendar/1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (scheme checked above)
            return response.read(5_000_000).decode("utf-8", errors="replace")
    return Path(location).expanduser().read_text(encoding="utf-8", errors="replace")


def describe_feed(location: str) -> str:
    """A feed's name for messages, without the secret part of a private URL."""
    match = re.match(r"^(?:https?|webcal)://([^/]+)", location, re.IGNORECASE)
    return match.group(1) if match else Path(location).name


# ── Outlook (COM) ───────────────────────────────────────────────────────────

def outlook_busy(namespace: Any, window_start: datetime, window_end: datetime,
                 zone: tzinfo) -> list[BusyBlock]:
    """Busy blocks from the default Outlook calendar. Runs inside a COM thread.

    ``IncludeRecurrences`` expands recurring meetings, but then the collection
    has no meaningful Count and a ``for`` loop over it never ends — so it is
    walked with GetFirst/GetNext and stopped at the end of the window.
    """
    folder = namespace.GetDefaultFolder(9)          # olFolderCalendar
    items = folder.Items
    items.Sort("[Start]")
    items.IncludeRecurrences = True
    local_start = window_start.astimezone(zone).replace(tzinfo=None)
    local_end = window_end.astimezone(zone).replace(tzinfo=None)
    fmt = "%m/%d/%Y %I:%M %p"
    restricted = items.Restrict(
        f"[Start] < '{local_end.strftime(fmt)}' AND [End] > '{local_start.strftime(fmt)}'")
    blocks: list[BusyBlock] = []
    item = restricted.GetFirst()
    guard = 0
    while item is not None and guard < MAX_OCCURRENCES:
        guard += 1
        try:
            if int(getattr(item, "BusyStatus", 2)) not in (0, 4):     # free / elsewhere
                begin = _naive_local(item.Start).replace(tzinfo=zone)
                finish = _naive_local(item.End).replace(tzinfo=zone)
                if begin >= window_end:
                    break
                if finish > window_start and finish > begin:
                    blocks.append(BusyBlock(begin, finish, str(getattr(item, "Subject", "")),
                                            "Outlook"))
        except Exception:
            pass
        item = restricted.GetNext()
    return blocks


def _naive_local(value: Any) -> datetime:
    """pywintypes datetimes are local wall-clock times; strip any tzinfo."""
    return datetime(value.year, value.month, value.day, value.hour, value.minute,
                    getattr(value, "second", 0))


# ── Notion ──────────────────────────────────────────────────────────────────

def notion_busy(pages: Iterable[dict[str, Any]], date_property: str,
                title_property: str, window_start: datetime, window_end: datetime,
                zone: tzinfo, default_length: timedelta = timedelta(hours=1)
                ) -> list[BusyBlock]:
    """Busy blocks from raw Notion pages' date property.

    A date-only entry is a deadline or an all-day marker, not an appointment,
    and does not block time. A start without an end blocks *default_length*.
    """
    blocks: list[BusyBlock] = []
    for page in pages:
        props = page.get("properties") or {}
        holder = (props.get(date_property) or {}).get("date") or {}
        start_text = str(holder.get("start") or "")
        if len(start_text) <= 10:
            continue
        try:
            start = _iso(start_text, zone)
            end_text = str(holder.get("end") or "")
            end = _iso(end_text, zone) if len(end_text) > 10 else start + default_length
        except ValueError:
            continue
        if end <= window_start or start >= window_end or end <= start:
            continue
        title_parts = (props.get(title_property) or {}).get("title") or []
        title = "".join(str(p.get("plain_text") or "") for p in title_parts if isinstance(p, dict))
        blocks.append(BusyBlock(start.astimezone(zone), end.astimezone(zone), title, "Notion"))
    return blocks


def _iso(text: str, zone: tzinfo) -> datetime:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=zone)
