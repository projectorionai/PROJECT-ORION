"""
Workouts, kept locally (config/fitness.db): what was done, for how long and
how far, and the week and the streak that adds up to.

Goals stay in goals.py - a fitness goal is a goal in the "health" area - so
the Fitness page reads those rather than keeping a second kind of goal here.
The one thing stored here besides workouts is the weekly target, which is a
setting of this log rather than a goal with a deadline.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from threading import RLock

from .constants import CONFIG_DIR
from .db import apply_pragmas

#: What can be logged, in the order the page offers them.
KINDS = ("run", "walk", "cycle", "strength", "yoga", "swim", "sport", "other")
#: Kinds where a distance means something.
DISTANCE_KINDS = frozenset({"run", "walk", "cycle", "swim"})
DEFAULT_WEEKLY_WORKOUTS = 3


def _now() -> datetime:
    from .time_service import TIME
    return TIME.now()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def normalise_kind(kind: str) -> str:
    word = str(kind or "").strip().lower()
    aliases = {"running": "run", "jog": "run", "jogging": "run", "walking": "walk",
               "hike": "walk", "cycling": "cycle", "bike": "cycle", "ride": "cycle",
               "gym": "strength", "weights": "strength", "lifting": "strength",
               "swimming": "swim", "stretch": "yoga", "pilates": "yoga"}
    word = aliases.get(word, word)
    return word if word in KINDS else "other"


@dataclass
class Workout:
    kind: str
    minutes: float
    distance_km: float | None = None
    calories: float | None = None
    note: str = ""
    at: str = ""
    id: int | None = None

    def local_day(self, zone: tzinfo | None) -> date:
        stamp = _parse(self.at)
        return (stamp.astimezone(zone) if zone is not None else stamp).date()


class FitnessStore:
    """Workouts and weekly targets in one small SQLite file."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else (CONFIG_DIR / "fitness.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = RLock()
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    minutes REAL NOT NULL,
                    distance_km REAL,
                    calories REAL,
                    note TEXT NOT NULL DEFAULT '',
                    at TEXT NOT NULL
                )""")
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_workouts_at ON workouts(at)")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS targets (metric TEXT PRIMARY KEY, value REAL NOT NULL)")

    @staticmethod
    def _row(row: sqlite3.Row) -> Workout:
        return Workout(kind=row["kind"], minutes=float(row["minutes"]),
                       distance_km=row["distance_km"], calories=row["calories"],
                       note=row["note"] or "", at=row["at"], id=row["id"])

    def log(self, kind: str, minutes: float, distance_km: float | None = None,
            calories: float | None = None, note: str = "",
            now: datetime | None = None) -> Workout:
        minutes = float(minutes)
        if not 0 < minutes <= 24 * 60:
            raise ValueError("A workout lasts between a minute and a day.")
        kind = normalise_kind(kind)
        distance = float(distance_km) if distance_km else None
        if distance is not None and not 0 < distance < 1000:
            raise ValueError("That distance does not look right.")
        workout = Workout(kind=kind, minutes=round(minutes, 1),
                          distance_km=round(distance, 2) if distance else None,
                          calories=float(calories) if calories else None,
                          note=str(note or "").strip()[:200], at=_iso(now or _now()))
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO workouts (kind, minutes, distance_km, calories, note, at) "
                "VALUES (?,?,?,?,?,?)",
                (workout.kind, workout.minutes, workout.distance_km, workout.calories,
                 workout.note, workout.at))
            workout.id = int(cur.lastrowid)
        return workout

    def remove(self, workout_id: int) -> bool:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM workouts WHERE id=?", (int(workout_id),))
            return cur.rowcount > 0

    def recent(self, limit: int = 8) -> list[Workout]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM workouts ORDER BY at DESC, id DESC LIMIT ?",
                                    (max(1, int(limit)),)).fetchall()
        return [self._row(r) for r in rows]

    def between(self, start: datetime, end: datetime) -> list[Workout]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM workouts WHERE at>=? AND at<? ORDER BY at",
                (_iso(start), _iso(end))).fetchall()
        return [self._row(r) for r in rows]

    def active_days(self, zone: tzinfo | None = None) -> set[date]:
        with self._lock:
            rows = self._db.execute("SELECT at FROM workouts").fetchall()
        days = set()
        for row in rows:
            try:
                stamp = _parse(str(row["at"]))
            except (TypeError, ValueError):
                continue
            days.add((stamp.astimezone(zone) if zone is not None else stamp).date())
        return days

    def set_target(self, metric: str, value: float) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT INTO targets (metric, value) VALUES (?, ?) "
                             "ON CONFLICT(metric) DO UPDATE SET value=excluded.value",
                             (str(metric), float(value)))

    def targets(self) -> dict[str, float]:
        with self._lock:
            rows = self._db.execute("SELECT metric, value FROM targets").fetchall()
        return {r["metric"]: float(r["value"]) for r in rows}

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except Exception:
                pass


def streak(days: set[date], today: date) -> int:
    """Consecutive days with a workout, up to today. A day not yet trained
    does not break it while yesterday still counts."""
    day = today if today in days else today - timedelta(days=1)
    count = 0
    while day in days:
        count += 1
        day -= timedelta(days=1)
    return count


def week_summary(store: FitnessStore, now: datetime | None = None) -> dict[str, object]:
    """This week, Monday to Sunday in the owner's own time."""
    now = now or _now()
    zone = now.tzinfo
    monday = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0,
                                                           microsecond=0)
    workouts = store.between(monday, monday + timedelta(days=7))
    per_day = {monday.date() + timedelta(days=i): 0.0 for i in range(7)}
    for w in workouts:
        day = w.local_day(zone)
        if day in per_day:
            per_day[day] += w.minutes
    targets = store.targets()
    return {
        "start": monday.date(),
        "days": list(per_day.items()),
        "workouts": len(workouts),
        "minutes": round(sum(w.minutes for w in workouts)),
        "distance_km": round(sum(w.distance_km or 0.0 for w in workouts), 1),
        "calories": round(sum(w.calories or 0.0 for w in workouts)),
        "target_workouts": int(targets.get("workouts", DEFAULT_WEEKLY_WORKOUTS)),
        "streak": streak(store.active_days(zone), now.date()),
        "today": now.date(),
    }


def clock(minutes: float) -> str:
    """'45m', '1:05' - a week's time at a glance."""
    m = int(round(minutes))
    return f"{m}m" if m < 60 else f"{m // 60}:{m % 60:02d}"


__all__ = [
    "DEFAULT_WEEKLY_WORKOUTS", "DISTANCE_KINDS", "KINDS", "FitnessStore", "Workout",
    "clock", "normalise_kind", "streak", "week_summary",
]
