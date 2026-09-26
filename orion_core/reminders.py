"""
ReminderService — spoken reminders, alarms and countdowns (a JARVIS staple).

The user can say (or type) things like:
    "remind me in 30 minutes to check the ad campaign"
    "remind me at 15:00 to call the supplier"
    "set an alarm for 7am"

A single background loop checks due reminders once a second and, when one
fires, announces it through the proactive-voice channel (ORION speaks it) and
raises a HUD banner.  Active reminders persist to memory so they survive a
restart within their window.

The dispatcher exposes ``reminder`` (add / list / cancel); natural-language
phrases are parsed here so the live model or the LocalBrain can pass either a
structured delay or a raw phrase.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .bus import OrionBus
from .data import ToolResult


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


class ReminderService:
    _REL_RE = re.compile(
        r"in\s+(\d+)\s*(second|sec|minute|min|hour|hr|day)s?", re.IGNORECASE
    )
    _AT_RE = re.compile(
        r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE
    )
    _UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60,
                     "hour": 3600, "hr": 3600, "day": 86400}

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._reminders: list[Reminder] = []
        self._counter = 0
        self._stop = asyncio.Event()
        #: Set post-construction by app.py when a telephony server is
        #: connected. Absent, VOICE_CALL and SMS reminders fall back to being
        #: spoken here rather than failing.
        self.telephony: Any = None

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
        body, delay = self._parse(raw, minutes, at)
        if delay is None:
            return ToolResult(
                "When should I remind you? Try 'in 20 minutes' or 'at 3pm'.", ok=False
            )
        if delay <= 0:
            return ToolResult("That time has already passed.", ok=False)
        self._counter += 1
        # Said back to the user, so ORION's clock: the machine's can be in
        # another zone (a cloud node on UTC).
        from .time_service import TIME
        due_wall = (TIME.now() + timedelta(seconds=delay)).strftime("%H:%M")
        reminder = Reminder(text=body or "your reminder", due_at=time.monotonic() + delay,
                            wall_due=due_wall, id=self._counter,
                            channel=wanted.value, recipient=recipient)
        self._reminders.append(reminder)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("reminder.added")
        pretty = self._pretty_delay(delay)
        self.bus.banner.emit(f"REMINDER SET — {due_wall}", 1)
        how = {"VOICE_CALL": " — I'll ring you",
               "SMS": " — I'll text you",
               "MOBILE_PUSH": " — on your phone"}.get(wanted.value, "")
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

        lowered = text.lower()
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
                if phrase in text.lower():
                    if chosen is None:
                        chosen = channel
                    # Every matching phrase is removed, not just the first.
                    # "notify me on my phone" is two of them, and leaving the
                    # second in had ORION read "on my phone about lunch" back
                    # as though that were the reminder.
                    index = text.lower().index(phrase)
                    text = text[:index] + text[index + len(phrase):]
        if chosen is None:
            return Channel.DESKTOP, text
        text = re.sub(r"\s+(and|then|,)\s*$", "", text.strip())
        text = re.sub(r"^\s*(and|then|,)\s+", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ,.;&")
        return chosen, text

    def _parse(self, raw: str, minutes: float | None, at: str) -> tuple[str, float | None]:
        raw = str(raw or "").strip()
        # Explicit structured delay wins.
        if minutes is not None:
            body = self._strip_time_words(raw)
            return body, float(minutes) * 60.0
        if at:
            body = self._strip_time_words(raw)
            return body, self._seconds_until_clock(at)
        # Relative "in N units".
        m = self._REL_RE.search(raw)
        if m:
            qty = int(m.group(1))
            unit = m.group(2).lower()
            secs = qty * self._UNIT_SECONDS.get(unit, 60)
            return self._extract_body(raw), float(secs)
        # Absolute "at HH[:MM][am/pm]".
        m = self._AT_RE.search(raw)
        if m:
            return self._extract_body(raw), self._seconds_until_clock(m.group(0))
        return self._extract_body(raw), None

    def _seconds_until_clock(self, at: str) -> float | None:
        m = self._AT_RE.search(at) or re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", at, re.IGNORECASE)
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridiem = (m.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        # "At 3pm" means 3pm on ORION's clock, the one he tells the time by.
        # The machine's can be in another zone (a UTC cloud node), where this
        # used to fire an hour out.
        from .time_service import TIME
        now = TIME.now()
        target = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)   # next occurrence
        return (target - now).total_seconds()

    def _extract_body(self, raw: str) -> str:
        body = re.sub(r"^\s*(remind me|set (?:a|an) (?:reminder|alarm)|remind|alarm)\s*", "",
                      raw, flags=re.IGNORECASE)
        body = re.sub(r"\bto\b", "", body, count=1) if body.lower().strip().startswith("to") else body
        body = self._REL_RE.sub("", body)
        body = self._AT_RE.sub("", body)
        body = re.sub(r"^\s*to\s+", "", body, flags=re.IGNORECASE)
        return body.strip(" ,.") or "your reminder"

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

    def active(self) -> list[Reminder]:
        return [r for r in self._reminders if not r.fired]

    def list_text(self) -> ToolResult:
        pending = self.active()
        if not pending:
            return ToolResult("No reminders are set.")
        lines = ["Pending reminders:"]
        for r in sorted(pending, key=lambda x: x.due_at):
            remaining = max(0, int(r.due_at - time.monotonic()))
            lines.append(f"- #{r.id} at {r.wall_due} ({self._pretty_delay(remaining)}): {r.text}")
        return ToolResult("\n".join(lines))

    def cancel(self, reminder_id: int | None = None) -> ToolResult:
        if reminder_id is None:
            count = len(self.active())
            self._reminders = [r for r in self._reminders if r.fired]
            return ToolResult(f"Cleared {count} reminder(s).")
        for r in self._reminders:
            if r.id == int(reminder_id) and not r.fired:
                r.fired = True
                return ToolResult(f"Reminder #{reminder_id} cancelled.")
        return ToolResult(f"No active reminder #{reminder_id}.", ok=False)

    # ── background loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        if self.telemetry is not None:
            self.telemetry.health.register("reminders")
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                for r in self._reminders:
                    if not r.fired and now >= r.due_at:
                        r.fired = True
                        self._fire(r)
                if self.telemetry is not None:
                    self.telemetry.health.beat("reminders", "OK", f"{len(self.active())} pending")
                    self.telemetry.metrics.gauge("reminders.pending", float(len(self.active())))
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    def _fire(self, reminder: Reminder) -> None:
        """Deliver a reminder on the channel it asked for.

        The banner and the dashboard event happen whatever the channel —
        they are free, they are local, and a reminder that rang your phone
        should still be visible on screen when you get back to it. Only the
        *spoken* delivery is routed.
        """
        self.bus.banner.emit(f"⏰ REMINDER: {reminder.text}", 4)
        self.bus.dashboard_event.emit("reminder_fired", reminder.text)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("reminder.fired")

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
            self.bus.log.emit(
                f"REMINDER: couldn't reach you by "
                f"{channel.value.lower().replace('_', ' ')} — saying it here.")

        self.bus.speak_request.emit(f"A reminder: {reminder.text}.")

    def _dispatch_remote(self, reminder: Reminder, channel: Any) -> bool:
        """Ring or text. Returns whether it was handed off successfully."""
        gateway = getattr(self, "telephony", None)
        if gateway is None:
            return False
        from .telephony import Channel

        who = reminder.recipient or "me"
        message = f"This is ORION with a reminder. {reminder.text}."
        try:
            from . import background

            coroutine = (gateway.call(who, message, reason="reminder")
                         if channel is Channel.VOICE_CALL
                         else gateway.text(who, message))
            background.spawn(coroutine, name="orion-reminder-dispatch")
            return True
        except Exception as exc:
            self.bus.log.emit(f"REMINDER: telephony dispatch failed - {exc}")
            return False

    def stop(self) -> None:
        self._stop.set()
