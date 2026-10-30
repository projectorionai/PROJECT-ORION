"""
Wellbeing & cognitive performance (Mark XXVI, Phase 3) — the other half of the
learning loop.

The study and focus systems measure output; this measures the person producing
it. A check-in records energy, mood, stress and sleep; over time that becomes an
honest picture of when the user works well — and, crucially, it *correlates* with
the focus/study data ORION already holds, so "I focus better on nights I slept
seven hours" stops being a hunch.

Grounded in psychology (self-report on a 1–5 scale, valence for mood), fully
offline SQLite (``config/wellbeing.db``), and cross-linked to focus rather than
siloed. Mirrors the study/focus scaffold.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from threading import RLock

from .constants import CONFIG_DIR
from .db import apply_pragmas

#: Spoken mood labels → a valence in [-2, 2].
MOOD_LABELS: dict[str, int] = {
    "great": 2, "excellent": 2, "good": 1, "fine": 0, "ok": 0, "okay": 0,
    "neutral": 0, "meh": 0, "low": -1, "down": -1, "tired": -1, "bad": -2,
    "awful": -2, "terrible": -2,
}


def _now() -> datetime:
    from .time_service import TIME
    return TIME.now()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _clamp(value, lo, hi, default=None):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def resolve_mood(value) -> int | None:
    """A mood label ('good') or a number → a valence in [-2, 2]."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value.lstrip("+-").isdigit():
            return MOOD_LABELS.get(value.lower())
    return _clamp(value, -2, 2)


def _sleep_hours(value) -> float | None:
    """Invalid or missing self-reports stay missing rather than biasing averages."""
    try:
        hours = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return hours if math.isfinite(hours) and 0 <= hours <= 24 else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length series, or None when it is
    undefined (fewer than two points, or one series is constant)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    if any(not math.isfinite(v) for v in (*xs, *ys)):
        return None
    # Scaling does not change correlation, and avoids overflow when squaring.
    sx, sy = max(abs(x) for x in xs) or 1.0, max(abs(y) for y in ys) or 1.0
    xs, ys = [x / sx for x in xs], [y / sy for y in ys]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return round(cov / math.sqrt(vx * vy), 3)


@dataclass
class Checkin:
    at: str = ""
    energy: int | None = None            # 1–5
    mood_valence: int | None = None      # -2..2
    stress: int | None = None            # 1–5
    sleep_hours: float | None = None
    note: str = ""
    factors: list[str] = field(default_factory=list)
    id: int | None = None

    def day(self, zone: tzinfo | None = None) -> date:
        stamp = _parse(self.at)
        return (stamp.astimezone(zone) if zone is not None else stamp).date()


class WellbeingStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else (CONFIG_DIR / "wellbeing.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = RLock()
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                at TEXT NOT NULL, energy INTEGER, mood_valence INTEGER,
                stress INTEGER, sleep_hours REAL, note TEXT DEFAULT '');
            CREATE TABLE IF NOT EXISTS factors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkin_id INTEGER NOT NULL, tag TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_checkin_at ON checkins(at);
            CREATE INDEX IF NOT EXISTS idx_factors_checkin ON factors(checkin_id);
            """)
        self._db.commit()

    def add(self, c: Checkin) -> Checkin:
        stamp = c.at or _iso(_now())
        # Convert user values before opening a transaction.
        tags = [str(tag) for tag in c.factors]
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO checkins (at, energy, mood_valence, stress, sleep_hours, note) "
                "VALUES (?,?,?,?,?,?)",
                (stamp, c.energy, c.mood_valence, c.stress, c.sleep_hours, c.note))
            row_id = int(cur.lastrowid)
            self._db.executemany("INSERT INTO factors (checkin_id, tag) VALUES (?,?)",
                                 [(row_id, tag) for tag in tags])
        c.id, c.at, c.factors = row_id, stamp, tags
        return c

    @staticmethod
    def _row(r: sqlite3.Row) -> Checkin:
        return Checkin(id=r["id"], at=r["at"], energy=r["energy"],
                       mood_valence=r["mood_valence"], stress=r["stress"],
                       sleep_hours=_sleep_hours(r["sleep_hours"]), note=r["note"] or "")

    def _select(self, where: str = "", params: tuple = ()) -> list[Checkin]:
        """Load check-ins and their factors in one indexed query, in stable order."""
        with self._lock:
            rows = self._db.execute(
                "SELECT c.*, f.tag AS factor_tag FROM checkins c "
                "LEFT JOIN factors f ON f.checkin_id=c.id "
                f"{where} ORDER BY c.at, c.id, f.id", params).fetchall()
        checkins: dict[int, Checkin] = {}
        for row in rows:
            checkin = checkins.get(row["id"])
            if checkin is None:
                checkin = self._row(row)
                checkins[row["id"]] = checkin
            if row["factor_tag"] is not None:
                checkin.factors.append(row["factor_tag"])
        return list(checkins.values())

    def since(self, start: datetime, end: datetime | None = None) -> list[Checkin]:
        if end is None:
            return self._select("WHERE c.at>=?", (_iso(start),))
        return self._select("WHERE c.at>=? AND c.at<=?", (_iso(start), _iso(end)))

    def all(self) -> list[Checkin]:
        return self._select()

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except Exception:
                pass


def _daily_mean(pairs: list[tuple[date, float]]) -> dict[date, float]:
    buckets: dict[date, list[float]] = {}
    for d, v in pairs:
        buckets.setdefault(d, []).append(v)
    return {d: sum(vs) / len(vs) for d, vs in buckets.items()}


class WellbeingEngine:
    def __init__(self, store: WellbeingStore | None = None) -> None:
        self.store = store or WellbeingStore()

    def checkin(self, *, energy=None, mood=None, stress=None, sleep_hours=None,
                note: str = "", factors=None, at: datetime | None = None) -> Checkin:
        sleep = _sleep_hours(sleep_hours)
        tags = [factors] if isinstance(factors, str) else list(factors or [])
        return self.store.add(Checkin(
            at=_iso(at or _now()), energy=_clamp(energy, 1, 5),
            mood_valence=resolve_mood(mood), stress=_clamp(stress, 1, 5),
            sleep_hours=sleep, note=note, factors=tags))

    def today(self, now: datetime | None = None) -> dict[str, object]:
        now = now or _now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0, fold=0)
        todays = self.store.since(start, end=now)
        energies = [c.energy for c in todays if c.energy is not None]
        return {
            "checkins": len(todays),
            "avg_energy": round(sum(energies) / len(energies), 1) if energies else None,
            "logged": bool(todays),
        }

    def trend(self, days: int = 14, now: datetime | None = None) -> dict[str, object]:
        now = now or _now()
        window = self.store.since(now - timedelta(days=days), end=now)

        def _avg(vals):
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            return round(sum(vals) / len(vals), 2) if vals else None

        def _direction(series):
            series = [v for v in series if v is not None]
            if len(series) < 4:
                return "flat"
            mid = len(series) // 2
            first, second = series[:mid], series[mid:]
            a = sum(first) / len(first)
            b = sum(second) / len(second)
            if b - a > 0.3:
                return "up"
            if a - b > 0.3:
                return "down"
            return "flat"

        return {
            "days": days,
            "checkins": len(window),
            "avg_energy": _avg([c.energy for c in window]),
            "avg_mood": _avg([c.mood_valence for c in window]),
            "avg_sleep": _avg([c.sleep_hours for c in window]),
            "avg_stress": _avg([c.stress for c in window]),
            "energy_direction": _direction([c.energy for c in window]),
        }

    def correlate(self, *, focus=None, days: int = 30,
                  now: datetime | None = None) -> list[dict[str, object]]:
        """Correlate wellbeing against the focus record ORION already holds.

        Returns Pearson coefficients over days that have BOTH signals — so
        "energy vs focus minutes" is honest about its sample size. Study
        correlation will follow once the study store exposes reviews-per-day."""
        now = now or _now()
        window = self.store.since(now - timedelta(days=days), end=now)
        zone = now.tzinfo or timezone.utc
        energy_by_day = _daily_mean([(c.day(zone), float(c.energy)) for c in window
                                     if c.energy is not None])
        sleep_by_day = _daily_mean([(c.day(zone), float(c.sleep_hours)) for c in window
                                    if c.sleep_hours is not None
                                    and math.isfinite(c.sleep_hours)
                                    and 0 <= c.sleep_hours <= 24])

        minutes_by_day: dict[date, float] = {}
        completed_by_day: dict[date, float] = {}
        if focus is not None:
            try:
                for s in focus.store.all():
                    if not s.started_at or not s.ended_at:
                        continue
                    d = _parse(s.started_at).astimezone(zone).date()
                    if not 0 <= (now.date() - d).days <= days:
                        continue
                    minutes_by_day[d] = minutes_by_day.get(d, 0.0) + s.elapsed_minutes()
                    completed_by_day[d] = completed_by_day.get(d, 0.0) + (1.0 if s.completed else 0.0)
            except Exception:
                pass

        def _corr(label, a: dict[date, float], b: dict[date, float]):
            common = sorted(set(a) & set(b))
            if len(common) < 2:
                return None
            coef = pearson([a[d] for d in common], [b[d] for d in common])
            if coef is None:
                return None
            return {"pair": label, "coefficient": coef, "n": len(common)}

        results = [
            _corr("energy vs focus minutes", energy_by_day, minutes_by_day),
            _corr("sleep vs focus minutes", sleep_by_day, minutes_by_day),
            _corr("energy vs blocks completed", energy_by_day, completed_by_day),
        ]
        return [r for r in results if r is not None]


__all__ = [
    "MOOD_LABELS", "Checkin", "WellbeingStore", "WellbeingEngine",
    "resolve_mood", "pearson",
]
