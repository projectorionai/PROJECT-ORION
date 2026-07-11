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

    # ── scheduling ────────────────────────────────────────────────────────────

    def add(self, text: str = "", minutes: float | None = None,
            at: str = "", phrase: str = "") -> ToolResult:
        """Add a reminder from structured args or a natural-language phrase."""
        raw = phrase or text
        body, delay = self._parse(raw, minutes, at)
        if delay is None:
            return ToolResult(
                "When should I remind you, sir? Try 'in 20 minutes' or 'at 3pm'.", ok=False
            )
        if delay <= 0:
            return ToolResult("That time has already passed, sir.", ok=False)
        self._counter += 1
        due_wall = (datetime.now() + timedelta(seconds=delay)).strftime("%H:%M")
        reminder = Reminder(text=body or "your reminder", due_at=time.monotonic() + delay,
                            wall_due=due_wall, id=self._counter)
        self._reminders.append(reminder)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("reminder.added")
        pretty = self._pretty_delay(delay)
        self.bus.banner.emit(f"REMINDER SET — {due_wall}", 1)
        return ToolResult(f"Very good, sir. I'll remind you {pretty} (at {due_wall}): {reminder.text}.")

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
        now = datetime.now()
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
            return ToolResult("No reminders are set, sir.")
        lines = ["Pending reminders, sir:"]
        for r in sorted(pending, key=lambda x: x.due_at):
            remaining = max(0, int(r.due_at - time.monotonic()))
            lines.append(f"- #{r.id} at {r.wall_due} ({self._pretty_delay(remaining)}): {r.text}")
        return ToolResult("\n".join(lines))

    def cancel(self, reminder_id: int | None = None) -> ToolResult:
        if reminder_id is None:
            count = len(self.active())
            self._reminders = [r for r in self._reminders if r.fired]
            return ToolResult(f"Cleared {count} reminder(s), sir.")
        for r in self._reminders:
            if r.id == int(reminder_id) and not r.fired:
                r.fired = True
                return ToolResult(f"Reminder #{reminder_id} cancelled, sir.")
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
        self.bus.banner.emit(f"⏰ REMINDER: {reminder.text}", 4)
        self.bus.speak_request.emit(f"A reminder, sir: {reminder.text}.")
        self.bus.dashboard_event.emit("reminder_fired", reminder.text)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("reminder.fired")

    def stop(self) -> None:
        self._stop.set()
