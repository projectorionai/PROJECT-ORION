"""
Calendar: today and the six days after it, from what ORION already reads -
the Notion calendar, the reminders he is holding, and goal deadlines and
milestones - with the next thing at the top and one line to add a reminder.

The Notion read is ORION's own (NotionService.upcoming_events, which also
feeds the Widgets page); reminders go through the same ReminderService the
voice uses. Network and disk work runs off the GUI thread.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ..constants import SPACE
from .cards import card, clear_layout, label, off_thread, run_async

DAYS_SHOWN = 7
#: The Notion calendar is a network read: at most this often while open.
NOTION_EVERY_S = 300.0
REFRESH_MS = 60_000


@dataclass(frozen=True)
class Entry:
    day: date
    title: str
    source: str                      # Notion | Reminder | Goal | Milestone
    at: datetime | None = None       # None: all day, or due that day

    @property
    def style(self) -> str:
        return {"Reminder": "entryReminder", "Goal": "entryGoal",
                "Milestone": "entryGoal"}.get(self.source, "entry")

    def time_text(self) -> str:
        if self.at is not None:
            return self.at.strftime("%H:%M")
        return "Due" if self.source in {"Goal", "Milestone"} else "All day"


def _now() -> datetime:
    from ..time_service import TIME
    return TIME.now()


def _sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text[:1].upper() + text[1:]


# ── turning each source into entries (pure) ──────────────────────────────────

def from_events(rows: Any, zone: Any) -> list[Entry]:
    """Notion rows ({"title", "date"}) as entries; a bare date is all day."""
    out: list[Entry] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        stamp, title = str(row.get("date") or "").strip(), _sentence(row.get("title"))
        if not stamp or not title:
            continue
        try:
            if len(stamp) <= 10:
                out.append(Entry(date.fromisoformat(stamp), title, "Notion"))
                continue
            at = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        at = (at if at.tzinfo else at.replace(tzinfo=zone)).astimezone(zone)
        out.append(Entry(at.date(), title, "Notion", at))
    return out


def from_reminders(reminders: Any, zone: Any) -> list[Entry]:
    out: list[Entry] = []
    for reminder in reminders or []:
        epoch = float(getattr(reminder, "due_epoch", 0) or 0)
        text = _sentence(getattr(reminder, "text", ""))
        if epoch <= 0 or not text:
            continue
        at = datetime.fromtimestamp(epoch, zone)
        out.append(Entry(at.date(), text, "Reminder", at))
    return out


def from_due(rows: Any) -> list[Entry]:
    """(source, title, due ISO date) rows - goal deadlines and milestones."""
    out: list[Entry] = []
    for source, title, due in rows or []:
        try:
            day = date.fromisoformat(str(due or "")[:10])
        except ValueError:
            continue
        if str(title or "").strip():
            out.append(Entry(day, _sentence(title), source))
    return out


def _order(entry: Entry) -> tuple:
    # Something due "that day" is due by its end, after the day's timed entries.
    return (entry.day, 0 if entry.at is not None else 1,
            entry.at.timestamp() if entry.at is not None else 0.0, entry.title.casefold())


def week_of(entries: list[Entry], today: date, days: int = DAYS_SHOWN
            ) -> list[tuple[date, list[Entry]]]:
    buckets: dict[date, list[Entry]] = {today + timedelta(days=i): [] for i in range(days)}
    for entry in entries:
        if entry.day in buckets:
            buckets[entry.day].append(entry)
    return [(day, sorted(items, key=_order)) for day, items in buckets.items()]


def next_up(entries: list[Entry], now: datetime) -> Entry | None:
    """The next timed entry still to come, or the soonest thing due."""
    grace = now - timedelta(minutes=5)
    ahead = [e for e in entries
             if (e.at is not None and e.at >= grace) or (e.at is None and e.day >= now.date())]
    return min(ahead, key=_order) if ahead else None


def relative(entry: Entry, now: datetime) -> str:
    """'in 25 min', 'in 2 h 10 min', 'Tomorrow', 'Wednesday'."""
    days = (entry.day - now.date()).days
    if entry.at is not None and days == 0:
        minutes = int((entry.at - now).total_seconds() // 60)
        if minutes <= 0:
            return "Now"
        if minutes < 60:
            return f"in {minutes} min"
        hours, rest = divmod(minutes, 60)
        return f"in {hours} h {rest} min" if rest else f"in {hours} h"
    if days == 0:
        return "Today"
    if days == 1:
        return "Tomorrow"
    return entry.day.strftime("%A")


def meta_line(entry: Entry, now: datetime) -> str:
    parts = [relative(entry, now)]
    if entry.at is not None:
        parts.append(entry.at.strftime("%H:%M"))
    parts.append(entry.source)
    return "  ·  ".join(parts)


# ── the page ──────────────────────────────────────────────────────────────────

class CalendarDeckView(QWidget):
    def __init__(self, bus: Any, dispatcher: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus, self.dispatcher = bus, dispatcher
        self._loading = False
        self._adding = False
        self._events: list[Entry] = []
        self._events_at = 0.0
        self.entries: list[Entry] = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        page = QVBoxLayout(body)
        page.setContentsMargins(SPACE.SM, SPACE.SM, SPACE.SM, SPACE.SM)
        page.setSpacing(SPACE.LG)

        top = QHBoxLayout()
        top.setSpacing(SPACE.LG)
        next_card, next_box = card("Next Up")
        self.next_title = label("", "heroText", wrap=True)
        self.next_meta = label("", "heroMeta")
        next_box.addWidget(self.next_title)
        next_box.addWidget(self.next_meta)
        next_box.addStretch(1)
        top.addWidget(next_card, 3)

        add_card, add_box = card()
        row = QHBoxLayout()
        row.setSpacing(SPACE.SM)
        self.quick = QLineEdit()
        self.quick.setPlaceholderText("Remind me to…")
        self.quick.setAccessibleName("New reminder")
        self.quick.returnPressed.connect(self.quick_add)
        self.add_button = QPushButton("Add")
        self.add_button.setObjectName("primaryButton")
        self.add_button.setAccessibleName("Add the reminder")
        self.add_button.clicked.connect(self.quick_add)
        row.addWidget(self.quick, 1)
        row.addWidget(self.add_button)
        add_box.addStretch(1)
        add_box.addLayout(row)
        self.added = label("", "mutedLabel", wrap=True)
        add_box.addWidget(self.added)
        add_box.addStretch(1)
        top.addWidget(add_card, 2)
        page.addLayout(top)

        self.week = QHBoxLayout()
        self.week.setSpacing(SPACE.SM + SPACE.XS)
        page.addLayout(self.week, 1)
        self._day_boxes: list[QVBoxLayout] = []

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        try:
            bus.dashboard_event.connect(self._on_event)
        except Exception:
            pass
        self.show_entries([], _now())

    # ── reading ──────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self.isVisible():
            self.load()

    def _on_event(self, channel: str, _payload: Any) -> None:
        if channel in {"reminder_fired", "goals", "tasks_dirty"} and self.isVisible():
            self.load()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self.load()

    def load(self) -> None:
        if self._loading or sip.isdeleted(self):
            return
        self._loading = True
        run_async(self, self._gather, done=self._loaded, failed=self._load_failed,
                  name="orion-calendar-page")

    async def _gather(self) -> tuple[datetime, list[Entry]]:
        now = _now()
        zone = now.tzinfo
        if time.monotonic() - self._events_at >= NOTION_EVERY_S or not self._events_at:
            self._events = await self._notion_events(zone)
            self._events_at = time.monotonic()
        local = await asyncio.to_thread(self._read_local, zone)
        return now, self._events + local

    async def _notion_events(self, zone: Any) -> list[Entry]:
        notion = getattr(self.dispatcher, "notion", None)
        try:
            if notion is None or not notion.available:
                return []
        except Exception:
            return []
        rows: list[Any] = []

        def capture(channel: str, payload: Any) -> None:
            if channel == "events" and isinstance(payload, list):
                rows[:] = payload

        self.bus.dashboard_event.connect(capture)
        try:
            await notion.upcoming_events(days=DAYS_SHOWN, limit=40)
        except Exception:
            return []
        finally:
            try:
                self.bus.dashboard_event.disconnect(capture)
            except Exception:
                pass
        return from_events(rows, zone)

    def _read_local(self, zone: Any) -> list[Entry]:
        entries: list[Entry] = []
        reminders = getattr(self.dispatcher, "reminders", None)
        if reminders is not None:
            try:
                entries += from_reminders(reminders.active(), zone)
            except Exception:
                pass
        engine = getattr(self.dispatcher, "goal_tracker", None)
        store = getattr(engine, "store", None)
        if store is not None:
            due: list[tuple[str, str, str]] = []
            try:
                for goal in store.goals(("active",)):
                    if goal.deadline:
                        due.append(("Goal", goal.title, goal.deadline))
                    for milestone in store.milestones(goal.id):
                        if milestone.due and not milestone.done:
                            due.append(("Milestone", milestone.title, milestone.due))
            except Exception:
                pass
            entries += from_due(due)
        return entries

    def _loaded(self, result: tuple[datetime, list[Entry]]) -> None:
        self._loading = False
        now, entries = result
        self.show_entries(entries, now)

    def _load_failed(self, exc: BaseException) -> None:
        self._loading = False
        self.added.setText(f"The calendar could not be read: {str(exc)[:120]}")

    # ── drawing ──────────────────────────────────────────────────────────────

    def show_entries(self, entries: list[Entry], now: datetime) -> None:
        self.entries = list(entries)
        upcoming = next_up(self.entries, now)
        if upcoming is None:
            self.next_title.setText("Nothing coming up")
            self.next_meta.setText("")
        else:
            self.next_title.setText(upcoming.title)
            self.next_meta.setText(meta_line(upcoming, now))
        clear_layout(self.week)
        for day, items in week_of(self.entries, now.date()):
            today = day == now.date()
            column, box = card(name="cardToday" if today else "card")
            box.setContentsMargins(SPACE.MD - SPACE.XS, SPACE.MD, SPACE.MD - SPACE.XS, SPACE.MD)
            head = QHBoxLayout()
            head.addWidget(label("Today" if today else day.strftime("%a"),
                                 "dayNameToday" if today else "dayName"))
            head.addStretch(1)
            head.addWidget(label(str(day.day), "entryTime"))
            box.addLayout(head)
            box.addSpacing(SPACE.XS)
            for entry in items:
                box.addWidget(self._entry(entry))
            if not items:
                box.addWidget(label("Free", "readoutKey"))
            box.addStretch(1)
            column.setAccessibleName(f"{day.strftime('%A %d %B')}, {len(items)} entries")
            self.week.addWidget(column, 1)

    @staticmethod
    def _entry(entry: Entry) -> QFrame:
        frame = QFrame()
        frame.setObjectName(entry.style)
        box = QVBoxLayout(frame)
        box.setContentsMargins(SPACE.SM + 2, SPACE.SM - 2, SPACE.SM, SPACE.SM - 2)
        box.setSpacing(1)
        box.addWidget(label(entry.time_text(), "entryTime"))
        title = label(entry.title, "entryTitle", wrap=True)
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(title)
        frame.setToolTip(entry.source)
        return frame

    # ── adding ───────────────────────────────────────────────────────────────

    def quick_add(self) -> None:
        text = self.quick.text().strip()
        reminders = getattr(self.dispatcher, "reminders", None)
        if not text or self._adding:
            return
        if reminders is None:
            self.added.setText("Reminders are not running.")
            return
        self._adding = True
        self.add_button.setEnabled(False)
        off_thread(self, lambda: reminders.add(phrase=text), done=self._added,
                   failed=self._add_failed, name="orion-calendar-add")

    def _added(self, result: Any) -> None:
        self._adding = False
        self.add_button.setEnabled(True)
        self.added.setText(str(getattr(result, "text", "") or ""))
        if getattr(result, "ok", False):
            self.quick.clear()
            self.load()

    def _add_failed(self, exc: BaseException) -> None:
        self._adding = False
        self.add_button.setEnabled(True)
        self.added.setText(f"That reminder was not set: {str(exc)[:120]}")


__all__ = ["CalendarDeckView", "Entry", "from_due", "from_events", "from_reminders",
           "meta_line", "next_up", "relative", "week_of"]
