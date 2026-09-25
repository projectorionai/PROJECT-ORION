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

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
    return datetime.now(timezone.utc)


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
    if minutes:
        work = int(minutes)
        # A sensible break scales with the block if one wasn't asked for.
        brk = int(break_minutes) if break_minutes else max(3, round(work / 5))
    elif break_minutes:
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
        cur = self._db.execute(
            "INSERT INTO sessions (label, planned_minutes, break_minutes, kind, "
            "started_at, ended_at, completed, interruptions, energy_before, "
            "energy_after, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (s.label, s.planned_minutes, s.break_minutes, s.kind, s.started_at,
             s.ended_at, int(s.completed), s.interruptions, s.energy_before,
             s.energy_after, s.notes))
        self._db.commit()
        s.id = int(cur.lastrowid)
        return s

    def update(self, s: FocusSession) -> None:
        if s.id is None:
            return
        self._db.execute(
            "UPDATE sessions SET label=?, planned_minutes=?, break_minutes=?, "
            "kind=?, started_at=?, ended_at=?, completed=?, interruptions=?, "
            "energy_before=?, energy_after=?, notes=? WHERE id=?",
            (s.label, s.planned_minutes, s.break_minutes, s.kind, s.started_at,
             s.ended_at, int(s.completed), s.interruptions, s.energy_before,
             s.energy_after, s.notes, s.id))
        self._db.commit()

    def active(self) -> FocusSession | None:
        row = self._db.execute(
            "SELECT * FROM sessions WHERE started_at<>'' AND ended_at='' "
            "ORDER BY id DESC LIMIT 1").fetchone()
        return self._row(row) if row else None

    def since(self, start: datetime) -> list[FocusSession]:
        rows = self._db.execute(
            "SELECT * FROM sessions WHERE started_at>=? ORDER BY started_at",
            (_iso(start),)).fetchall()
        return [self._row(r) for r in rows]

    # -- indexed queries (streak/stats were loading every session) --------

    def completed_days(self) -> set:
        """The DATES with at least one completed block — one indexed query
        instead of materialising every session the user has ever run."""
        rows = self._db.execute(
            "SELECT DISTINCT substr(started_at, 1, 10) AS day FROM sessions "
            "WHERE completed = 1 AND started_at <> ''").fetchall()
        days = set()
        for row in rows:
            try:
                days.add(date.fromisoformat(str(row["day"])))
            except (TypeError, ValueError):
                continue
        return days

    def all(self) -> list[FocusSession]:
        return [self._row(r) for r in
                self._db.execute("SELECT * FROM sessions ORDER BY started_at").fetchall()]

    def close(self) -> None:
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
        current = self.store.active()
        if current is not None:
            self._finish(current, completed=current.is_break_due(now), now=now)
        work, brk = resolve_cadence(preset, minutes, break_minutes)
        s = FocusSession(
            label=(label or "focus").strip() or "focus", planned_minutes=work,
            break_minutes=brk, kind=(kind or "deep"), started_at=_iso(now),
            energy_before=_clamp_energy(energy))
        return self.store.add(s)

    def active(self, now: datetime | None = None) -> FocusSession | None:
        return self.store.active()

    def interrupt(self, now: datetime | None = None) -> FocusSession | None:
        """Log a distraction against the running block — the honest cost of a
        context switch, so the streak isn't the only thing measured."""
        s = self.store.active()
        if s is None:
            return None
        s.interruptions += 1
        self.store.update(s)
        return s

    def complete(self, energy: int | None = None, notes: str = "",
                 now: datetime | None = None) -> FocusSession | None:
        """End the running block as a success and start its break."""
        s = self.store.active()
        if s is None:
            return None
        return self._finish(s, completed=True, energy_after=energy, notes=notes, now=now)

    def cancel(self, now: datetime | None = None) -> FocusSession | None:
        """Abandon the running block; it does not count toward the streak."""
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
        done_days = self.store.completed_days()
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
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        todays = self.store.since(start)
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
    except (TypeError, ValueError):
        return None


def human_minutes(minutes: float) -> str:
    """'50 minutes', '1 h 20 m' — a spoken duration."""
    m = int(round(minutes))
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}"
    h, rem = divmod(m, 60)
    return f"{h} h {rem} m" if rem else f"{h} hour{'s' if h != 1 else ''}"


__all__ = [
    "PRESETS", "DEFAULT_WORK", "DEFAULT_BREAK", "FocusSession", "FocusStore",
    "FocusEngine", "resolve_cadence", "human_minutes",
]
