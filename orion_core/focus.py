"""
Deep Focus (Mark XXV) — protected, measured blocks of concentration.

The study system (Mark XXIV) is what to review; this is the *when you actually
sit and do it*. A focus session is a single timed block with a stated intention,
a planned length and a break to follow, a count of the distractions that broke it,
and an optional energy check-in. Over time that becomes an honest picture of when
the user does their best work — the entrepreneur's and the student's shared need
to protect and understand their attention.

Grounded, not gimmicky:
  * Defaults are 50 min work / 10 min break (the "52-17" / deep-work family) with
    Pomodoro (25/5) a first-class alternative — real intervals, not invented ones.
  * A "streak" is consecutive days with at least one *completed* block, because
    the habit is the point.
  * Everything is local SQLite (config/focus.db); nothing leaves the machine.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from threading import RLock

from .constants import CONFIG_DIR
from .db import apply_pragmas

#: Named cadences the user can ask for by feel rather than by number.
PRESETS: dict[str, tuple[int, int]] = {
    "deep": (50, 10),        # a deep-work block
    "pomodoro": (25, 5),     # the classic
    "sprint": (25, 5),
    "long": (90, 20),        # a full ultradian cycle
    "short": (15, 3),
}
DEFAULT_WORK = 50
DEFAULT_BREAK = 10


def _now() -> datetime:
    from .time_service import TIME
    return TIME.now()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def resolve_cadence(preset: str | None, minutes: int | None = None,
                    break_minutes: int | None = None) -> tuple[int, int]:
    """Turn a spoken cadence ('a pomodoro', 'a deep block', '45 minutes') into a
    concrete (work, break) pair. Explicit minutes always win."""
    work, brk = PRESETS.get((preset or "").strip().lower(), (DEFAULT_WORK, DEFAULT_BREAK))
    if minutes is not None:
        work = int(minutes)
        # A sensible break scales with the block if one wasn't asked for.
        brk = int(break_minutes) if break_minutes is not None else max(3, round(work / 5))
    elif break_minutes is not None:
        brk = int(break_minutes)
    return max(1, work), max(0, brk)


@dataclass
class FocusSession:
    label: str                       # the intention: what this block is for
    planned_minutes: int = DEFAULT_WORK
    break_minutes: int = DEFAULT_BREAK
    kind: str = "deep"               # deep | study | admin | creative | …
    started_at: str = ""
    ended_at: str = ""
    completed: bool = False          # ran to (at least) its planned length
    interruptions: int = 0
    energy_before: int | None = None  # 1–5 self-report, optional
    energy_after: int | None = None
    notes: str = ""
    id: int | None = None

    def elapsed_minutes(self, now: datetime | None = None) -> float:
        if not self.started_at:
            return 0.0
        end = _parse(self.ended_at) if self.ended_at else (now or _now())
        return max(0.0, (end - _parse(self.started_at)).total_seconds() / 60.0)

    def remaining_minutes(self, now: datetime | None = None) -> float:
        return max(0.0, self.planned_minutes - self.elapsed_minutes(now))

    def is_break_due(self, now: datetime | None = None) -> bool:
        """The planned block is up — time to stop and take the break."""
        return not self.ended_at and self.elapsed_minutes(now) >= self.planned_minutes


class FocusStore:
    """Sessions in one small SQLite file."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else (CONFIG_DIR / "focus.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = RLock()
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                planned_minutes INTEGER DEFAULT 50,
                break_minutes INTEGER DEFAULT 10,
                kind TEXT DEFAULT 'deep',
                started_at TEXT DEFAULT '',
                ended_at TEXT DEFAULT '',
                completed INTEGER DEFAULT 0,
                interruptions INTEGER DEFAULT 0,
                energy_before INTEGER,
                energy_after INTEGER,
                notes TEXT DEFAULT ''
            )""")
        # The task being worked on, in small steps (the Focus page's list).
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS plan_steps (
                position INTEGER PRIMARY KEY,
                task TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0
            )""")
        self._db.commit()

    def _row(self, row: sqlite3.Row) -> FocusSession:
        return FocusSession(
            id=row["id"], label=row["label"], planned_minutes=row["planned_minutes"],
            break_minutes=row["break_minutes"], kind=row["kind"],
            started_at=row["started_at"] or "", ended_at=row["ended_at"] or "",
            completed=bool(row["completed"]), interruptions=row["interruptions"],
            energy_before=row["energy_before"], energy_after=row["energy_after"],
            notes=row["notes"] or "")

    def add(self, s: FocusSession) -> FocusSession:
        with self._lock:
            with self._db:
                row_id = self._insert(s)
            s.id = row_id
            return s

    def _insert(self, s: FocusSession) -> int:
        """Insert within the caller's transaction; publish the id after commit."""
        cur = self._db.execute(
            "INSERT INTO sessions (label, planned_minutes, break_minutes, kind, "
            "started_at, ended_at, completed, interruptions, energy_before, "
            "energy_after, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (s.label, s.planned_minutes, s.break_minutes, s.kind, s.started_at,
             s.ended_at, int(s.completed), s.interruptions, s.energy_before,
             s.energy_after, s.notes))
        return int(cur.lastrowid)

    def start_session(self, s: FocusSession) -> FocusSession:
        """Finish the old block and save its replacement in one transaction."""
        with self._lock:
            now = _parse(s.started_at)
            with self._db:
                # Also serialises separate connections to the same SQLite file.
                self._db.execute("BEGIN IMMEDIATE")
                rows = self._db.execute(
                    "SELECT * FROM sessions WHERE started_at<>'' AND ended_at=''").fetchall()
                for row in rows:
                    previous = self._row(row)
                    self._db.execute(
                        "UPDATE sessions SET ended_at=?, completed=? WHERE id=?",
                        (s.started_at, int(previous.is_break_due(now)), previous.id))
                row_id = self._insert(s)
            s.id = row_id
            return s

    def update(self, s: FocusSession) -> None:
        with self._lock:
            if s.id is None:
                return
            with self._db:
                self._db.execute(
                    "UPDATE sessions SET label=?, planned_minutes=?, break_minutes=?, "
                    "kind=?, started_at=?, ended_at=?, completed=?, interruptions=?, "
                    "energy_before=?, energy_after=?, notes=? WHERE id=?",
                    (s.label, s.planned_minutes, s.break_minutes, s.kind, s.started_at,
                     s.ended_at, int(s.completed), s.interruptions, s.energy_before,
                     s.energy_after, s.notes, s.id))

    def active(self) -> FocusSession | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM sessions WHERE started_at<>'' AND ended_at='' "
                "ORDER BY id DESC LIMIT 1").fetchone()
            return self._row(row) if row else None

    def since(self, start: datetime, *, before: datetime | None = None) -> list[FocusSession]:
        with self._lock:
            if before is None:
                rows = self._db.execute(
                    "SELECT * FROM sessions WHERE started_at>=? ORDER BY started_at",
                    (_iso(start),)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM sessions WHERE started_at>=? AND started_at<? ORDER BY started_at",
                    (_iso(start), _iso(before))).fetchall()
            return [self._row(r) for r in rows]

    # -- narrow reporting queries (avoid materialising full sessions) --------

    def completed_days(self, zone: tzinfo | None = None) -> set:
        """Days with a completed block, interpreted in the reporting timezone.

        Keep the compact UTC query for callers requesting UTC; local dates
        require converting real timestamps, including their historical offsets.
        """
        with self._lock:
            utc_days = zone is None or zone == timezone.utc
            expression = "substr(started_at, 1, 10)" if utc_days else "started_at"
            rows = self._db.execute(
                f"SELECT DISTINCT {expression} AS day FROM sessions "
                "WHERE completed = 1 AND started_at <> ''").fetchall()
            days = set()
            for row in rows:
                try:
                    stamp = str(row["day"])
                    days.add(date.fromisoformat(stamp) if utc_days else _parse(stamp).astimezone(zone).date())
                except (TypeError, ValueError):
                    continue
            return days

    def all(self) -> list[FocusSession]:
        with self._lock:
            return [self._row(r) for r in
                    self._db.execute("SELECT * FROM sessions ORDER BY started_at").fetchall()]

    # -- the task in steps ----------------------------------------------------

    def save_plan(self, task: str, steps: list[tuple[str, bool]]) -> None:
        """Replace the saved steps. Kept across restarts so a half-done list
        is still there tomorrow."""
        rows = [(i, str(task or "").strip()[:200], str(text).strip()[:200], int(bool(done)))
                for i, (text, done) in enumerate(steps) if str(text).strip()]
        with self._lock:
            with self._db:
                self._db.execute("DELETE FROM plan_steps")
                self._db.executemany(
                    "INSERT INTO plan_steps (position, task, text, done) VALUES (?,?,?,?)", rows)
                if not rows and str(task or "").strip():
                    self._db.execute("INSERT INTO plan_steps (position, task, text, done) "
                                     "VALUES (-1, ?, '', 0)", (str(task).strip()[:200],))

    def load_plan(self) -> tuple[str, list[tuple[str, bool]]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT task, text, done FROM plan_steps ORDER BY position").fetchall()
        task = next((r["task"] for r in rows if r["task"]), "")
        return task, [(r["text"], bool(r["done"])) for r in rows if r["text"]]

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except Exception:
                pass


class FocusEngine:
    """Start, run, and measure focus blocks. One is active at a time."""

    def __init__(self, store: FocusStore | None = None) -> None:
        self.store = store or FocusStore()

    def start(self, label: str, preset: str | None = None, minutes: int | None = None,
              break_minutes: int | None = None, kind: str = "deep",
              energy: int | None = None, now: datetime | None = None) -> FocusSession:
        """Begin a block. If one is already running it is ended first (completed
        only if it had already reached its planned length), so 'start' is always
        safe to call."""
        now = now or _now()
        # Validate the replacement before touching the currently running block.
        work, brk = resolve_cadence(preset, minutes, break_minutes)
        s = FocusSession(
            label=(label or "focus").strip() or "focus", planned_minutes=work,
            break_minutes=brk, kind=(kind or "deep"), started_at=_iso(now),
            energy_before=_clamp_energy(energy))
        return self.store.start_session(s)

    def active(self, now: datetime | None = None) -> FocusSession | None:
        return self.store.active()

    def interrupt(self, now: datetime | None = None) -> FocusSession | None:
        """Log a distraction against the running block — the honest cost of a
        context switch, so the streak isn't the only thing measured."""
        with self.store._lock:
            s = self.store.active()
            if s is None:
                return None
            s.interruptions += 1
            self.store.update(s)
            return s

    def complete(self, energy: int | None = None, notes: str = "",
                 now: datetime | None = None) -> FocusSession | None:
        """End the running block as a success and start its break."""
        with self.store._lock:
            s = self.store.active()
            if s is None:
                return None
            return self._finish(s, completed=True, energy_after=energy, notes=notes, now=now)

    def cancel(self, now: datetime | None = None) -> FocusSession | None:
        """Abandon the running block; it does not count toward the streak."""
        with self.store._lock:
            s = self.store.active()
            if s is None:
                return None
            return self._finish(s, completed=False, now=now)

    def _finish(self, s: FocusSession, *, completed: bool, energy_after: int | None = None,
                notes: str = "", now: datetime | None = None) -> FocusSession:
        s.ended_at = _iso(now or _now())
        s.completed = completed
        if energy_after is not None:
            s.energy_after = _clamp_energy(energy_after)
        if notes:
            s.notes = notes
        self.store.update(s)
        return s

    # -- reporting --
    def streak(self, now: datetime | None = None) -> int:
        """Consecutive days up to today with at least one COMPLETED block."""
        now = now or _now()
        done_days = self.store.completed_days(now.tzinfo)
        if not done_days:
            return 0
        day = now.date()
        # Allow the streak to stand if today has none yet but yesterday did.
        if day not in done_days:
            day = day - timedelta(days=1)
            if day not in done_days:
                return 0
        count = 0
        while day in done_days:
            count += 1
            day = day - timedelta(days=1)
        return count

    def stats(self, days: int = 7, now: datetime | None = None) -> dict[str, object]:
        now = now or _now()
        window = self.store.since(now - timedelta(days=days))
        completed = [s for s in window if s.completed]
        focus_minutes = sum(s.elapsed_minutes(now) for s in window if s.ended_at)
        interruptions = sum(s.interruptions for s in window)
        rate = round(100.0 * len(completed) / len(window)) if window else 0
        return {
            "days": days,
            "sessions": len(window),
            "completed": len(completed),
            "completion_rate": rate,
            "focus_minutes": round(focus_minutes),
            "interruptions": interruptions,
            "streak": self.streak(now),
        }

    def today(self, now: datetime | None = None) -> dict[str, object]:
        now = now or _now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        todays = self.store.since(start, before=start + timedelta(days=1))
        completed = [s for s in todays if s.completed]
        return {
            "sessions": len(todays),
            "completed": len(completed),
            "focus_minutes": round(sum(s.elapsed_minutes(now) for s in todays if s.ended_at)),
        }


def _clamp_energy(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError, OverflowError):
        return None


_STEP_BREAKS = re.compile(r"\n|;|•|,|\band then\b|\bthen\b|\bafter that\b", re.IGNORECASE)
_STEP_LEAD = re.compile(r"^\s*(?:[-*]\s*|\d+[.)]\s*|and\s+|then\s+)+", re.IGNORECASE)


def chunk_task(text: str, limit: int = 8) -> list[str]:
    """Break a task written as one line into its steps: "outline the essay,
    write the intro then add two sources" becomes three. Only the person's own
    words are used - a task that names one thing stays one step, and the page
    lets them add the rest."""
    steps: list[str] = []
    seen: set[str] = set()
    for part in _STEP_BREAKS.split(str(text or "")):
        pieces = [part]
        # "and" joins two steps only when both sides are a phrase of their
        # own: "write the intro and add sources", not "salt and pepper".
        halves = re.split(r"\s+and\s+", part.strip(), maxsplit=1, flags=re.IGNORECASE)
        if len(halves) == 2 and all(len(h.split()) >= 2 for h in halves):
            pieces = halves
        for piece in pieces:
            step = _STEP_LEAD.sub("", piece).strip(" .")
            if not step or step.casefold() in seen:
                continue
            seen.add(step.casefold())
            steps.append(step[0].upper() + step[1:])
            if len(steps) >= limit:
                return steps
    return steps


def human_minutes(minutes: float) -> str:
    """'50 minutes', '1 h 20 m' — a spoken duration."""
    m = int(round(minutes))
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}"
    h, rem = divmod(m, 60)
    return f"{h} h {rem} m" if rem else f"{h} hour{'s' if h != 1 else ''}"


__all__ = [
    "PRESETS", "DEFAULT_WORK", "DEFAULT_BREAK", "FocusSession", "FocusStore",
    "FocusEngine", "resolve_cadence", "human_minutes", "chunk_task",
]
