"""
One view of the owner's tasks and one rule for "overdue".

25 Oct 2026 (Mark 37.2, the owner's session at 11:52): the briefing read the
Notion board - which is also his calendar - and found five open rows dated in
the past (a finished trip, a house move, a team day, two untitled rows).
ORION said "several overdue tasks require attention". Asked to clear them, he
called awareness(clear_overdue), which looked only at ORION's own task list
(empty), answered "Nothing is overdue", and ORION told the owner nothing had
ever been overdue. The two paths read different stores, and five modules each
had their own overdue rule (a bare date overdue from midnight in some, from the
end of the day in others; a date range judged by its start).

Everything that says a task is open or overdue now reads it here:

* ``hours_until`` / ``is_overdue`` - the one rule. A bare date is due at the
  END of that day; a date range is due when it ends; an aware time is compared
  with an aware "now".
* ``gather`` - ORION's own pending tasks plus the Notion board's open tasks,
  minus everything the owner finished or removed, each marked overdue by that
  rule. A board that could not be read is reported as unread, never as clear.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR

_BARE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
#: The owner's fresh start (config/fresh_start.json {"since": "YYYY-MM-DD"}):
#: a board task that fell due before it is never read out again. 28 Oct 2026:
#: ORION's own tasks were cleared for university, but five overdue rows on the
#: Notion board (due in August and October) kept opening every briefing.
FRESH_START_FILE = "fresh_start.json"
FRESH_START_PATH = Path(CONFIG_DIR) / FRESH_START_FILE


def fresh_start(config_dir: Any = None) -> datetime | None:
    """The start of the owner's fresh start, or None if there is none."""
    path = FRESH_START_PATH if config_dir is None else Path(config_dir) / FRESH_START_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return datetime.fromisoformat(str(data.get("since") or ""))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def before_fresh_start(due: Any, *, end: Any = "", since: datetime | None = None) -> bool:
    """True for a task that fell due before the fresh start."""
    if since is None:
        return False
    moment = due_moment(end) or due_moment(due)
    if moment is None:
        return False
    if moment.tzinfo is not None:
        moment = moment.astimezone().replace(tzinfo=None)
    return moment <= since


def due_moment(raw: Any) -> datetime | None:
    """When *raw* falls due: a bare date at the end of that day, otherwise
    the instant itself (aware when it carries an offset). None if unparseable."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if _BARE_DATE.fullmatch(text):
            return datetime.fromisoformat(text) + timedelta(days=1)
        return datetime.fromisoformat(text.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None


def hours_until(due: Any, *, end: Any = "", now: datetime | None = None) -> float | None:
    """Hours from *now* until the item falls due (negative once it is late).

    A range (start *due*, *end*) falls due when it ends: a two-week trip is
    not "overdue" on its first morning."""
    moment = due_moment(end) or due_moment(due)
    if moment is None:
        return None
    current = now or datetime.now()
    if moment.tzinfo is not None and current.tzinfo is None:
        current = current.astimezone()
    elif moment.tzinfo is None and current.tzinfo is not None:
        current = current.astimezone().replace(tzinfo=None)
    return (moment - current).total_seconds() / 3600.0


def is_overdue(due: Any, *, end: Any = "", now: datetime | None = None) -> bool:
    hours = hours_until(due, end=end, now=now)
    return hours is not None and hours < 0


@dataclass
class TaskView:
    """The owner's open tasks from every store ORION keeps them in."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    board_configured: bool = False
    board_read: bool = False
    board_error: str = ""

    @property
    def overdue(self) -> list[dict[str, Any]]:
        return [row for row in self.rows if row.get("overdue")]

    def counts(self) -> dict[str, int]:
        return {"open": len(self.rows), "overdue": len(self.overdue),
                "board": sum(1 for r in self.rows if r.get("source") == "notion"),
                "orion": sum(1 for r in self.rows if r.get("source") == "orion")}

    def summary_line(self) -> str:
        """One line for a situation report: counts, and what was not read."""
        c = self.counts()
        if not c["open"]:
            text = "Tasks: none open"
        else:
            text = (f"Tasks: {c['open']} open ({c['orion']} on ORION's list, "
                    f"{c['board']} on the Notion board), {c['overdue']} overdue")
            late = [str(r.get("title") or "?")[:60] for r in self.overdue[:5]]
            if late:
                text += ": " + "; ".join(late)
        if self.board_configured and not self.board_read:
            text += f" (the Notion board could not be read: {self.board_error or 'unknown error'})"
        return text + "."


def board_ready(notion: Any) -> bool:
    return bool(notion is not None and getattr(notion, "available", False)
                and getattr(notion, "tasks_db", ""))


async def gather(notion: Any, cognition: Any, *, now: datetime | None = None,
                 since: datetime | None = None, use_fresh_start: bool = True) -> TaskView:
    """ORION's pending tasks + the board's open tasks, finished/removed ones
    and those due before the owner's fresh start excluded, each with
    ``overdue`` and ``source`` ("orion" or "notion")."""
    view = TaskView(board_configured=board_ready(notion))
    if since is None and use_fresh_start:
        since = await asyncio.to_thread(fresh_start)
    seen: set[str] = set()
    if cognition is not None:
        try:
            local = await asyncio.to_thread(cognition.list_tasks)
        except Exception:
            local = []
        for task in local:
            title = str(task.get("title") or "").strip()
            if not title:
                continue
            seen.add(title.casefold())
            view.rows.append({
                "title": title, "due": str(task.get("due") or ""),
                "due_end": str(task.get("due_end") or ""), "status": "",
                "source": "orion", "id": str(task.get("id") or ""),
                "overdue": is_overdue(task.get("due"), end=task.get("due_end"), now=now)})
    if view.board_configured:
        dismissed = getattr(cognition, "is_dismissed", None)
        try:
            board = await notion.open_tasks(limit=None)
            view.board_read = True
        except Exception as exc:
            board = []
            view.board_error = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else type(exc).__name__
        for row in board:
            title = str(row.get("title") or "").strip() or "(untitled)"
            if callable(dismissed):
                try:
                    if dismissed(title):
                        continue
                except Exception:
                    pass
            if title.casefold() in seen and title != "(untitled)":
                continue                     # the same task kept in both stores
            seen.add(title.casefold())
            due = str(row.get("date_raw") or row.get("date") or "")
            end = str(row.get("date_end") or "")
            if before_fresh_start(due, end=end, since=since):
                continue                     # from before the fresh start
            view.rows.append({
                "title": title, "due": str(row.get("date") or due[:16].replace("T", " ")),
                "due_raw": due, "due_end": end, "status": str(row.get("status") or ""),
                "source": "notion", "id": str(row.get("id") or ""),
                "overdue": is_overdue(due, end=end, now=now)})
    # Overdue first, then soonest, then undated - the order everything reads.
    view.rows.sort(key=lambda r: (not r["overdue"],
                                  str(r.get("due_raw") or r.get("due") or "") or "9999"))
    return view


def render(rows: list[dict[str, Any]], *, heading: str = "Open tasks") -> str:
    """The list as one line per task, overdue ones marked, in the same words
    the briefing and the awareness tool both use."""
    if not rows:
        return "No open tasks - ORION's list and the task board are clear."
    late = sum(1 for r in rows if r.get("overdue"))
    lines = [f"{heading} - {len(rows)}" + (f" ({late} overdue):" if late else ":")]
    for row in rows:
        bits = [str(row.get("title") or "?")]
        if row.get("status"):
            bits.append(f"[{row['status']}]")
        if row.get("due"):
            bits.append(f"due {row['due']}" + (" (OVERDUE)" if row.get("overdue") else ""))
        if row.get("source") == "notion":
            bits.append("(Notion)")
        lines.append("- " + "  ".join(bits))
    return "\n".join(lines)


__all__ = ["TaskView", "board_ready", "due_moment", "gather", "hours_until",
           "is_overdue", "render"]
