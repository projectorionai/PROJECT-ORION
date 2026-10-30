"""
ReminderService — spoken reminders, alarms and countdowns (a JARVIS staple).

The user can say (or type) things like:
    "remind me in 30 minutes to check the ad campaign"
    "remind me at 15:00 to call the supplier"
    "set an alarm for 7am"

A single background loop checks due reminders once a second and, when one
fires, announces it through the proactive-voice channel (ORION speaks it) and
raises a HUD banner. Pending reminders and cancellations are stored atomically
in config/reminders.json. On restart their wall-clock deadlines are rebased to
the new process's monotonic clock; overdue reminders are announced locally.

The dispatcher exposes ``reminder`` (add / list / cancel); natural-language
phrases are parsed here so the live model or the LocalBrain can pass either a
structured delay or a raw phrase.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .atomic_io import atomic_write_text
from .constants import CONFIG_DIR
from .utils import first_line
from .clock_sense import likely_hour

REMINDERS_PATH = CONFIG_DIR / "reminders.json"
_DEFAULT_PATH = object()
_MAX_STORE_BYTES = 8 * 1024 * 1024


# ── how people actually say a time ───────────────────────────────────────────
#
# The parser below understood "in <whole number> <unit>" and "at H[:MM][am|pm]"
# only. Measured 5 Oct 2026: "in half an hour", "in an hour", "in twenty
# minutes", "at noon", "half past four" and "tomorrow morning" set NO time;
# "tonight at 8" was set for 8 in the MORNING, and "on Monday at 10" for 10
# today with "on monday" left in the reminder. Spoken forms are rewritten into
# the two shapes the parser knows, and a day/period layer sits on top.

_UNITS_WORD = r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)"
_SMALL = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_NUMBER_WORD = (r"(?:(?:" + "|".join(_TENS) + r")(?:[\s-](?:" + "|".join(list(_SMALL)[1:10])
                + r"))?|" + "|".join(sorted(_SMALL, key=len, reverse=True)) + r")")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
#: (default hour, minute, the period is afternoon-or-later).
_PERIODS = {"morning": (9, 0, False), "afternoon": (15, 0, True),
            "evening": (18, 0, True), "tonight": (20, 0, True), "night": (20, 0, True)}


def _word_number(text: str) -> int | None:
    parts = re.split(r"[\s-]+", text.strip().lower())
    if len(parts) == 1:
        return _SMALL.get(parts[0], _TENS.get(parts[0]))
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _SMALL:
        return _TENS[parts[0]] + _SMALL[parts[1]]
    return None


def spoken_to_numeric(raw: str) -> str:
    """Rewrite spoken durations and clock times into digits.

    "in half an hour" -> "in 30 minutes"; "in twenty minutes" -> "in 20
    minutes"; "at half past four" -> "at 4:30"; "at noon" -> "at 12:00".
    Only phrases attached to a time unit or "at"/"past"/"to" are touched, so
    the reminder's own words are left alone.
    """
    text = str(raw or "")
    flags = re.IGNORECASE
    # Timers and alarms say "for": "a timer for 5 minutes" is "in 5 minutes",
    # "a 20 minute timer" likewise, "an alarm for 7am" is "at 7am".
    if re.search(r"\b(?:timer|countdown)\b", text, flags):
        text = re.sub(r"\b(" + r"\d+(?:\.\d+)?|" + _NUMBER_WORD + r"|an?)[\s-]*"
                      + r"(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+(timer|countdown)\b",
                      r"\3 in \1 \2", text, flags=flags)
        text = re.sub(r"\bfor\s+(?=(?:\d|" + _NUMBER_WORD + r"|an?\s|half\s|a\s+couple|a\s+few))",
                      "in ", text, flags=flags)
    if re.search(r"\b(?:alarm|wake me)\b", text, flags):
        text = re.sub(r"\bfor\s+(?=\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b)", "at ", text, flags=flags)
    # Durations — only after "in", so the reminder's own words ("buy one hour
    # of parking") are left exactly as said.
    text = re.sub(r"\bin\s+(?:an?|one)\s+hour\s+and\s+a\s+half\b", "in 90 minutes", text, flags=flags)
    text = re.sub(r"\bin\s+(\d+)\s+and\s+a\s+half\s+hours?\b",
                  lambda m: f"in {int(m.group(1)) * 60 + 30} minutes", text, flags=flags)
    text = re.sub(r"\bin\s+half\s+an?\s+hour\b", "in 30 minutes", text, flags=flags)
    text = re.sub(r"\bin\s+half\s+a\s+minute\b", "in 30 seconds", text, flags=flags)
    text = re.sub(r"\bin\s+(?:a\s+)?quarter\s+(?:of\s+)?an?\s+hour\b", "in 15 minutes",
                  text, flags=flags)
    text = re.sub(r"\bin\s+a\s+couple\s+of\s+" + _UNITS_WORD, r"in 2 \1", text, flags=flags)
    text = re.sub(r"\bin\s+a\s+few\s+" + _UNITS_WORD, r"in 3 \1", text, flags=flags)
    text = re.sub(r"\bin\s+an?\s+" + _UNITS_WORD, r"in 1 \1", text, flags=flags)
    text = re.sub(r"\bin\s+(" + _NUMBER_WORD + r")\s+" + _UNITS_WORD,
                  lambda m: f"in {_word_number(m.group(1))} {m.group(2)}"
                  if _word_number(m.group(1)) is not None else m.group(0), text, flags=flags)
    text = re.sub(r"\bin\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?)\b",
                  lambda m: (f"in {round(float(m.group(1)) * 60)} minutes"
                             if "." in m.group(1) else m.group(0)), text, flags=flags)
    # 7 Oct's "in 5 to check the oven" had no unit. Only accept a complete
    # time clause; "in 5 chapters" is part of the subject, not a countdown.
    text = re.sub(r"\bin\s+(\d+|" + _NUMBER_WORD + r")(?=\s+to\b|\s*[,.!?]?$)",
                  lambda m: f"in {int(m.group(1)) if m.group(1).isdigit() else _word_number(m.group(1))} minutes",
                  text, flags=flags)
    # Clock times.
    text = re.sub(r"\b(?:at\s+)?(?:12\s+)?(?:noon|midday)\b", "at 12:00", text, flags=flags)
    text = re.sub(r"\b(?:at\s+)?midnight\b", "at 0:00", text, flags=flags)

    def hour_of(word: str) -> int | None:
        word = word.strip().lower()
        if word.isdigit():
            return int(word)
        return _SMALL.get(word)

    def past_to(m: re.Match[str]) -> str:
        minute_word, relation, hour_word = m.group(1).lower(), m.group(2).lower(), m.group(3)
        hour = hour_of(hour_word)
        minutes = {"half": 30, "quarter": 15}.get(minute_word)
        if minutes is None:
            minutes = int(minute_word) if minute_word.isdigit() else _word_number(minute_word)
        if hour is None or minutes is None or not 0 < minutes < 60 or not 0 < hour <= 12:
            return m.group(0)
        if relation == "to":
            hour, minutes = (hour - 1) or 12, 60 - minutes
        return f"at {hour}:{minutes:02d}"

    text = re.sub(r"\b(?:at\s+)?(half|quarter|\d{1,2}|" + _NUMBER_WORD + r")\s+(past|to)\s+"
                  r"(\d{1,2}|" + "|".join(list(_SMALL)[1:13]) + r")\b", past_to, text, flags=flags)
    text = re.sub(r"\b(?:at\s+)?half\s+(\d{1,2}|" + "|".join(list(_SMALL)[1:13]) + r")\b",
                  lambda m: f"at {hour_of(m.group(1))}:30", text, flags=flags)
    text = re.sub(r"\b(?:at\s+)?(\d{1,2}|" + "|".join(list(_SMALL)[1:13]) + r")\s+o'?clock\b",
                  lambda m: f"at {hour_of(m.group(1))}", text, flags=flags)
    return text


def day_hint(raw: str) -> tuple[int | None, str, list[str]]:
    """(days ahead or None, period word or '', the phrases to strip from the body).

    Days ahead is relative to *today* — weekdays resolve later, against the
    clock. Period is morning/afternoon/evening/tonight/night or ''.
    """
    text = str(raw or "").lower()
    strip: list[str] = []
    days: int | None = None
    period = ""
    if re.search(r"\btomorrow\b", text):
        days = 1
        strip.append(r"\btomorrow\b")
    elif re.search(r"\btoday\b", text):
        days = 0
        strip.append(r"\btoday\b")
    weekday = re.search(r"\b(?:on\s+|next\s+|this\s+)?(" + "|".join(_WEEKDAYS) + r")\b", text)
    if weekday and days is None:
        # Resolved later, against the clock: -1..-7 is "the next such day",
        # -8..-14 is "NEXT Monday" (a week on when said on a Monday).
        index = _WEEKDAYS.index(weekday.group(1))
        days = (-8 - index) if weekday.group(0).startswith("next") else (-1 - index)
        strip.append(re.escape(weekday.group(0)))
    found = re.search(r"\b(?:this\s+|in\s+the\s+|tomorrow\s+)?(morning|afternoon|evening|tonight)\b"
                      r"|\b(?:at|tomorrow)\s+(night)\b", text)
    if found:
        period = found.group(1) or found.group(2)
        strip.append(re.escape(found.group(0).replace("tomorrow ", "")))
        if period == "tonight" and days is None:
            days = 0
    return days, period, strip


@dataclass
class Reminder:
    text: str
    due_at: float                      # monotonic deadline
    wall_due: str = ""                 # human-readable due time
    id: int = 0
    fired: bool = False
    #: How it should reach you. A reminder that says how it wants to arrive is
    #: far more useful than one that always arrives the same way: an eight
    #: a.m. briefing should ring your phone, a "stand up" nudge should not.
    channel: str = "DESKTOP"
    #: Who to ring or text, when the channel leaves this machine. A name from
    #: the contact book, or a number already in it.
    recipient: str = ""
    due_epoch: float = 0.0             # portable deadline; monotonic is process-local
    restored_overdue: bool = False


class ReminderService:
    _REL_RE = re.compile(
        r"in\s+(\d+)\s*(second|sec|minute|min|hour|hr|day)s?", re.IGNORECASE
    )
    _AT_RE = re.compile(
        r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?![\w:]|\s*(?:am|pm|:))", re.IGNORECASE
    )
    _CLOCK_RE = re.compile(
        r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE
    )
    _UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60,
                     "hour": 3600, "hr": 3600, "day": 86400}

    def __init__(self, bus: OrionBus, telemetry: Any | None = None, *,
                 path: Path | str | None | object = _DEFAULT_PATH) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._telemetry_failures: set[tuple[str, str]] = set()
        self._reminders: list[Reminder] = []
        self._counter = 0
        self._lock = RLock()
        self._mutation_lock = RLock()  # Serialise saves without blocking GUI reads.
        self.path = Path(REMINDERS_PATH if path is _DEFAULT_PATH else path) if path is not None else None
        self._storage_error = ""
        self._checkpoint_error = False
        self._next_checkpoint_attempt = 0.0
        self._running = False
        self._stop = asyncio.Event()
        #: Set post-construction by app.py when a telephony server is
        #: connected. Absent, VOICE_CALL and SMS reminders fall back to being
        #: spoken here rather than failing.
        self.telephony: Any = None
        self._load()

    def _observe(self, section: str, method: str, *args: Any) -> None:
        """A failed optional diagnostic must never block a saved reminder."""
        if self.telemetry is None:
            return
        try:
            getattr(getattr(self.telemetry, section), method)(*args)
        except Exception as exc:
            key = (section, method)
            with self._lock:
                first_failure = key not in self._telemetry_failures
                self._telemetry_failures.add(key)
            if first_failure:
                logging.getLogger(__name__).warning(
                    "Optional reminder telemetry %s.%s failed (%s)",
                    section, method, type(exc).__name__)

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            if self.path.stat().st_size > _MAX_STORE_BYTES:
                raise ValueError("Reminder store is too large")
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("Unsupported reminder store format")
            rows, counter = data["reminders"], data["counter"]
            if not isinstance(rows, list) or type(counter) is not int or counter < 0:
                raise ValueError("Invalid reminder store metadata")
            from .time_service import TIME
            now, monotonic = TIME.now(), time.monotonic()
            pending, ids = [], set()
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Invalid reminder record")
                rid, epoch = row["id"], row["due_epoch"]
                if type(rid) is not int or not 0 < rid <= counter or rid in ids:
                    raise ValueError("Invalid or duplicate reminder id")
                if type(epoch) not in (int, float) or not math.isfinite(epoch):
                    raise ValueError("Invalid reminder deadline")
                if (not isinstance(row["text"], str) or not row["text"].strip()
                        or not isinstance(row.get("recipient", ""), str)
                        or row.get("channel", "DESKTOP") not in {"DESKTOP", "MOBILE_PUSH", "SMS", "VOICE_CALL"}):
                    raise ValueError("Invalid reminder content or delivery channel")
                due_wall = datetime.fromtimestamp(epoch, now.tzinfo).strftime("%H:%M")
                delay = epoch - now.timestamp()
                pending.append(Reminder(row["text"], monotonic + max(0.0, delay), due_wall,
                                        rid, channel=row.get("channel", "DESKTOP"),
                                        recipient=row.get("recipient", ""), due_epoch=epoch,
                                        restored_overdue=delay < 0))
                ids.add(rid)
            self._reminders, self._counter = pending, counter
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            self._storage_error = first_line(exc)
            self.bus.log.emit(f"REMINDER: saved reminders could not be read — {self._storage_error}")

    def _save(self, pending: list[Reminder], counter: int) -> None:
        if self._storage_error:
            raise OSError(f"Saved reminders could not be read: {self._storage_error}")
        if self.path is None:
            return
        with self._lock:
            rows = [{"id": r.id, "text": r.text, "due_epoch": r.due_epoch,
                     "channel": r.channel, "recipient": r.recipient}
                    for r in pending if not r.fired]
        content = json.dumps({"version": 1, "counter": counter, "reminders": rows}, ensure_ascii=False)
        if len(content.encode("utf-8")) > _MAX_STORE_BYTES:
            raise OSError("Reminder store is too large")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, content, encoding="utf-8")

    def _checkpoint(self) -> None:
        with self._mutation_lock:
            with self._lock:
                pending, counter = list(self._reminders), self._counter
            self._save(pending, counter)

    # ── scheduling ────────────────────────────────────────────────────────────

    def add(self, text: str = "", minutes: float | None = None,
            at: str = "", phrase: str = "", channel: Any = "",
            recipient: str = "") -> ToolResult:
        """Add a reminder from structured args or a natural-language phrase.

        ``channel`` says how it should arrive: DESKTOP (spoken here, the way
        it always worked), MOBILE_PUSH, SMS, or VOICE_CALL. It is also read
        out of the phrase itself, because "remind me at 8 and call me" is how
        people actually ask.
        """
        raw = phrase or text
        wanted, raw = self._read_channel(raw, channel)
        try:
            body, delay = self._parse(raw, minutes, at)
        except (TypeError, ValueError, OverflowError):
            delay = None
        if delay is None or not math.isfinite(delay):
            return ToolResult(
                "When should I remind you? Try 'in 20 minutes' or 'at 3pm'.", ok=False
            )
        if delay <= 0:
            return ToolResult("That time has already passed.", ok=False)
        # Said back to the user, so ORION's clock: the machine's can be in
        # another zone (a cloud node on UTC).
        from .time_service import TIME
        now = TIME.now()
        try:
            # A countdown is elapsed time, even when the local clock skips
            # or repeats an hour between now and the deadline.
            due_wall = datetime.fromtimestamp(now.timestamp() + delay, now.tzinfo).strftime("%H:%M")
        except (ValueError, OverflowError, OSError):
            return ToolResult("That reminder time is too far in the future.", ok=False)
        with self._mutation_lock:
            with self._lock:
                counter, pending = self._counter + 1, list(self._reminders)
            reminder = Reminder(text=body or "your reminder", due_at=time.monotonic() + delay,
                                wall_due=due_wall, id=counter, due_epoch=now.timestamp() + delay,
                                channel=wanted.value, recipient=recipient)
            try:
                self._save([*pending, reminder], counter)
            except OSError as exc:
                return ToolResult(f"I couldn't save that reminder; it wasn't set: {first_line(exc)}", ok=False)
            with self._lock:
                self._counter = counter
                self._reminders.append(reminder)
        self._observe("metrics", "incr", "reminder.added")
        pretty = self._pretty_delay(delay)
        self.bus.banner.emit(f"REMINDER SET — {due_wall}", 1)
        how = {"VOICE_CALL": " — I'll ring you",
               "SMS": " — I'll text you",
               "MOBILE_PUSH": " — on your phone"}.get(wanted.value, "")
        if reminder.text == "your timer":
            return ToolResult(f"Timer set - {pretty.removeprefix('in ')} (until {due_wall}){how}.")
        if reminder.text == "your alarm":
            return ToolResult(f"Alarm set for {due_wall}{how}.")
        return ToolResult(
            f"Very good. I'll remind you {pretty} (at {due_wall}){how}: "
            f"{reminder.text}.")

    def _read_channel(self, raw: str, explicit: Any) -> tuple[Any, str]:
        """Work out how this reminder should arrive, and strip the words that
        said so out of its text.

        Left in, "remind me at 8 and call me" would have ORION read the phrase
        "and call me" back at you as though it were the reminder.
        """
        from .telephony import Channel

        text = str(raw or "")
        if explicit:
            return Channel.parse(explicit), text

        spoken = [
            (Channel.VOICE_CALL, ("call me", "ring me", "phone me",
                                  "give me a call", "by phone")),
            (Channel.SMS, ("text me", "message me", "send me a text",
                           "by sms")),
            (Channel.MOBILE_PUSH, ("notify me", "on my phone",
                                   "push it to my phone")),
        ]
        chosen = None
        for channel, phrases in spoken:
            for phrase in phrases:
                pattern = r"\b" + re.escape(phrase) + r"\b"
                if re.search(pattern, text, flags=re.IGNORECASE):
                    if chosen is None:
                        chosen = channel
                    # Every matching phrase is removed, not just the first.
                    # "notify me on my phone" is two of them, and leaving the
                    # second in had ORION read "on my phone about lunch" back
                    # as though that were the reminder.
                    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        if chosen is None:
            return Channel.DESKTOP, text
        text = re.sub(r"\s+(and|then|,)\s*$", "", text.strip())
        text = re.sub(r"^\s*(and|then|,)\s+", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ,.;&")
        return chosen, text

    def _parse(self, raw: str, minutes: float | None, at: str) -> tuple[str, float | None]:
        raw = str(raw or "").strip()
        context = raw
        # Explicit structured delay wins.
        if minutes is not None:
            body = self._strip_time_words(raw)
            return body, float(minutes) * 60.0
        if at:
            _days, _period, strip = day_hint(raw)
            body_source = raw
            for pattern in strip:
                body_source = re.sub(pattern, " ", body_source, flags=re.IGNORECASE)
            body = self._strip_time_words(body_source)
            return body, self._seconds_until_clock(at, context)
        raw = spoken_to_numeric(raw)
        # Relative "in N units".
        m = self._REL_RE.search(raw)
        if m:
            qty = int(m.group(1))
            unit = m.group(2).lower()
            secs = qty * self._UNIT_SECONDS.get(unit, 60)
            return self._extract_body(raw), float(secs)
        days, period, strip = day_hint(raw)
        body_source = raw
        for pattern in strip:
            body_source = re.sub(pattern, " ", body_source, flags=re.IGNORECASE)
        # Absolute "at HH[:MM][am/pm]".
        m = self._AT_RE.search(raw)
        if m and (days is not None or period):
            hour, minute = int(m.group(1)), int(m.group(2) or 0)
            meridiem = (m.group(3) or "").lower()
            if meridiem and not 1 <= hour <= 12:
                return self._extract_body(body_source), None
            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            elif not meridiem and period and _PERIODS[period][2] and hour < 12:
                hour += 12                    # "tonight at 8" is 20:00
            elif not meridiem:
                from .time_service import TIME
                # A named day describes a day's schedule, not whichever
                # reading is closest to the current time. Preserve "Monday
                # at 10" as morning even when this Monday's 10 has passed.
                reference = TIME.now().replace(hour=7, minute=0, second=0, microsecond=0)
                hour = self._resolve_spoken_hour(m, reference, context)
            return self._extract_body(body_source), self._seconds_until_day(days, hour, minute)
        if m:
            return self._extract_body(raw), self._seconds_until_clock(m.group(0), context)
        if days is not None or period:
            hour, minute, _late = _PERIODS.get(period, (9, 0, False))
            return self._extract_body(body_source), self._seconds_until_day(days, hour, minute)
        return self._extract_body(raw), None

    def _seconds_until_day(self, days: int | None, hour: int, minute: int) -> float | None:
        """Seconds until *hour:minute* on the day *days* names: 0 today, 1
        tomorrow, -1-weekday for the next such weekday, None for the next
        time that clock comes round. Explicit days never silently change."""
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        from .time_service import TIME
        now = TIME.now()
        weekday = days is not None and days < 0
        if days is not None and -14 <= days <= -8:
            wanted = -8 - days
            days = (wanted - now.weekday()) % 7 or 7
        elif days is not None and -7 <= days < 0:
            wanted = -1 - days
            days = (wanted - now.weekday()) % 7
        elif weekday:
            return None
        offsets = (days, days + 7) if weekday else ((0, 1) if days is None else (days,))
        for offset in offsets:
            delays = self._wall_time_delays(now, offset, hour, minute)
            future = [delay for delay in delays if delay > 0]
            if future:
                return min(future)
            if days is not None and not weekday:
                # A passed explicit today is still today; add() explains that
                # it has passed. A missing spring-transition time has no delay.
                return max(delays) if delays else None
        return None

    @staticmethod
    def _wall_time_delays(now: datetime, days: int, hour: int, minute: int) -> list[float]:
        """Valid instants for a local wall time, including both autumn folds."""
        target = (now + timedelta(days=days)).replace(
            hour=hour, minute=minute, second=0, microsecond=0)
        instants = set()
        for fold in (0, 1):
            stamp = target.replace(fold=fold).timestamp()
            local = datetime.fromtimestamp(stamp, now.tzinfo)
            if local.replace(tzinfo=None) == target.replace(tzinfo=None):
                instants.add(stamp)
        return sorted(stamp - now.timestamp() for stamp in instants)

    @staticmethod
    def _resolve_spoken_hour(match: re.Match[str], now: datetime, text: str) -> int:
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        # Zero-padded clocks and named noon/midnight are already explicit.
        if (match.group(1).startswith("0") or re.search(r"\b(noon|midday|midnight)\b", text, re.I)
                or not 1 <= hour <= 12 or minute > 59):
            return hour
        return likely_hour(hour, minute, now, text)

    def _seconds_until_clock(self, at: str, context: str = "") -> float | None:
        m = self._CLOCK_RE.fullmatch(at.strip())
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridiem = (m.group(3) or "").lower()
        if minute > 59 or (not meridiem and hour > 23):
            return None
        if meridiem and not 1 <= hour <= 12:
            return None
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        # "At 3pm" means 3pm on ORION's clock, the one he tells the time by.
        # The machine's can be in another zone (a UTC cloud node), where this
        # used to fire an hour out.
        from .time_service import TIME
        now = TIME.now()
        days, period, _strip = day_hint(context)
        if not meridiem:
            if period and _PERIODS[period][2] and hour < 12:
                hour += 12
            else:
                reference = now.replace(hour=7, minute=0, second=0, microsecond=0) if days is not None else now
                hour = self._resolve_spoken_hour(m, reference, context or at)
        if days is not None or period:
            return self._seconds_until_day(days, hour, minute)
        for days in (0, 1):
            future = [delay for delay in self._wall_time_delays(now, days, hour, minute) if delay > 0]
            if future:
                return min(future)
        return None

    def _extract_body(self, raw: str) -> str:
        timer = bool(re.search(r"\b(?:timer|countdown)\b", raw, re.IGNORECASE))
        alarm = bool(re.search(r"\b(?:alarm|wake me)\b", raw, re.IGNORECASE))
        body = re.sub(r"^\s*(remind me|set (?:a|an|me an?) (?:reminder|alarm|timer|countdown)|"
                      r"start (?:a|an) (?:timer|countdown)|wake me(?: up)?|remind|alarm|timer|"
                      r"countdown)\s*", "", raw, flags=re.IGNORECASE)
        body = re.sub(r"\bto\b", "", body, count=1) if body.lower().strip().startswith("to") else body
        body = self._REL_RE.sub("", body)
        body = self._AT_RE.sub("", body)
        body = re.sub(r"^\s*(?:to|for)\s+", "", body, flags=re.IGNORECASE)
        body = re.sub(r"^\s*(?:timer|countdown)\b\s*(?:for\s+)?", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s{2,}", " ", body)
        default = "your timer" if timer else "your alarm" if alarm else "your reminder"
        return body.strip(" ,.") or default

    def _strip_time_words(self, raw: str) -> str:
        return self._extract_body(raw)

    @staticmethod
    def _pretty_delay(secs: float) -> str:
        if secs < 90:
            return f"in {int(secs)} seconds"
        if secs < 5400:
            return f"in {round(secs / 60)} minutes"
        if secs < 172800:
            return f"in {round(secs / 3600, 1)} hours"
        return f"in {round(secs / 86400, 1)} days"

    # ── management ────────────────────────────────────────────────────────────

    @property
    def storage_error(self) -> str:
        """Non-empty when saved state is unavailable; callers must not show clear."""
        return self._storage_error

    def active(self) -> list[Reminder]:
        with self._lock:
            return [r for r in self._reminders if not r.fired]

    def list_text(self) -> ToolResult:
        if self._storage_error:
            return ToolResult(f"I couldn't read the saved reminders: {self._storage_error}", ok=False)
        pending = self.active()
        if not pending:
            return ToolResult("No reminders are set.")
        lines = ["Pending reminders:"]
        for r in sorted(pending, key=lambda x: x.due_at):
            remaining = max(0, int(r.due_at - time.monotonic()))
            lines.append(f"- #{r.id} at {r.wall_due} ({self._pretty_delay(remaining)}): {r.text}")
        return ToolResult("\n".join(lines))

    def cancel(self, reminder_id: int | None = None) -> ToolResult:
        with self._mutation_lock:
            with self._lock:
                removed = [r for r in self.active() if reminder_id is None or r.id == int(reminder_id)]
                if reminder_id is not None and not removed:
                    return ToolResult(f"No active reminder #{reminder_id}.", ok=False)
                kept, counter = [r for r in self._reminders if r not in removed], self._counter
            try:
                self._save(kept, counter)
            except OSError as exc:
                return ToolResult(f"I couldn't save the cancellation; reminders are unchanged: {first_line(exc)}", ok=False)
            with self._lock:
                for r in removed:
                    r.fired = True
                self._reminders = kept
            return ToolResult(f"Cleared {len(removed)} reminder(s)." if reminder_id is None
                              else f"Reminder #{reminder_id} cancelled.")

    # ── background loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        try:
            self._observe("health", "register", "reminders")
            while not self._stop.is_set():
                now = time.monotonic()
                changed = False
                for r in sorted(self.active(), key=lambda item: item.due_at)[:3]:
                    with self._lock:
                        if not r.fired and now >= r.due_at:
                            try:
                                self._fire(r)
                            except Exception as exc:
                                r.due_at = now + 30.0
                                self.bus.log.emit(f"REMINDER: delivery failed; retrying in 30 seconds — {first_line(exc)}")
                            else:
                                r.fired = True
                                changed = True
                if changed or (self._checkpoint_error and now >= self._next_checkpoint_attempt):
                    try:
                        await asyncio.to_thread(self._checkpoint)
                    except OSError as exc:
                        if not self._checkpoint_error:
                            self.bus.log.emit(f"REMINDER: delivery state could not be saved — {first_line(exc)}")
                        self._checkpoint_error = True
                        self._next_checkpoint_attempt = now + 30.0
                    else:
                        self._checkpoint_error = False
                        with self._lock:
                            self._reminders = [r for r in self._reminders if not r.fired]
                pending_count = len(self.active())
                self._observe("health", "beat", "reminders",
                              "WARN" if self._checkpoint_error or self._storage_error else "OK",
                              f"{pending_count} pending")
                self._observe("metrics", "gauge", "reminders.pending", float(pending_count))
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            with self._lock:
                self._running = False

    def _fire(self, reminder: Reminder) -> None:
        """Deliver a reminder on the channel it asked for.

        The banner and the dashboard event happen whatever the channel —
        they are free, they are local, and a reminder that rang your phone
        should still be visible on screen when you get back to it. Only the
        *spoken* delivery is routed.
        """
        prefix = "MISSED REMINDER" if reminder.restored_overdue else "REMINDER"
        self.bus.banner.emit(f"⏰ {prefix}: {reminder.text}", 4)
        self.bus.dashboard_event.emit("reminder_fired", reminder.text)
        self._observe("metrics", "incr", "reminder.fired")

        if reminder.restored_overdue:
            self.bus.speak_request.emit(f"While I was offline, a reminder came due: {reminder.text}.")
            return  # An overdue call/text is announced here, never sent late.

        from .telephony import Channel

        channel = Channel.parse(reminder.channel)
        if channel is Channel.MOBILE_PUSH:
            try:
                self.bus.phone_action.emit(
                    {"kind": "share", "text": f"Reminder: {reminder.text}"})
                return
            except Exception:
                pass                       # fall through and say it here
        elif channel in {Channel.VOICE_CALL, Channel.SMS}:
            if self._dispatch_remote(reminder, channel):
                return
            # Falling back to speaking it is deliberate: a reminder that could
            # not be delivered the way it was asked for must still be
            # delivered, not silently dropped.
            self._remote_fallback(reminder, channel)
            return

        if reminder.text == "your timer":
            self.bus.speak_request.emit("Time's up - your timer has finished.")
        elif reminder.text == "your alarm":
            from .time_service import TIME
            self.bus.speak_request.emit(f"It's {TIME.now().strftime('%H:%M')} - your alarm.")
        else:
            self.bus.speak_request.emit(f"A reminder: {reminder.text}.")

    def _remote_fallback(self, reminder: Reminder, channel: Any) -> None:
        self.bus.log.emit(
            f"REMINDER: couldn't reach you by "
            f"{channel.value.lower().replace('_', ' ')} — saying it here.")
        self.bus.speak_request.emit(f"A reminder: {reminder.text}.")

    def _dispatch_remote(self, reminder: Reminder, channel: Any) -> bool:
        """Start an observed delivery, falling back locally if it fails."""
        gateway = getattr(self, "telephony", None)
        if gateway is None:
            return False
        from .telephony import Channel

        who = reminder.recipient or "me"
        message = f"This is ORION with a reminder. {reminder.text}."

        async def deliver() -> None:
            try:
                result = (await gateway.call(who, message, reason="reminder")
                          if channel is Channel.VOICE_CALL
                          else await gateway.text(who, message))
            except Exception:
                self._remote_fallback(reminder, channel)
                return
            if not getattr(result, "ok", False):
                self._remote_fallback(reminder, channel)

        coroutine = deliver()
        try:
            from . import background

            return background.spawn(coroutine, name="orion-reminder-dispatch") is not None
        except Exception as exc:
            coroutine.close()
            self.bus.log.emit(f"REMINDER: telephony dispatch failed - {type(exc).__name__}")
            return False

    def stop(self) -> None:
        self._stop.set()
