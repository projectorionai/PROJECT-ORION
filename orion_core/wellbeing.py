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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .constants import CONFIG_DIR
from .db import apply_pragmas

#: Spoken mood labels → a valence in [-2, 2].
MOOD_LABELS: dict[str, int] = {
    "great": 2, "excellent": 2, "good": 1, "fine": 0, "ok": 0, "okay": 0,
    "neutral": 0, "meh": 0, "low": -1, "down": -1, "tired": -1, "bad": -2,
    "awful": -2, "terrible": -2,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _clamp(value, lo, hi, default=None):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def resolve_mood(value) -> int | None:
    """A mood label ('good') or a number → a valence in [-2, 2]."""
    if value is None or value == "":
        return None
    if isinstance(value, str) and not value.lstrip("-").isdigit():
        return MOOD_LABELS.get(value.strip().lower())
    return _clamp(value, -2, 2)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length series, or None when it is
    undefined (fewer than two points, or one series is constant)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
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

    def day(self) -> date:
        return _parse(self.at).date()


class WellbeingStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else (CONFIG_DIR / "wellbeing.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
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
            """)
        self._db.commit()

    def add(self, c: Checkin) -> Checkin:
        cur = self._db.execute(
            "INSERT INTO checkins (at, energy, mood_valence, stress, sleep_hours, note) "
            "VALUES (?,?,?,?,?,?)",
            (c.at or _iso(_now()), c.energy, c.mood_valence, c.stress, c.sleep_hours, c.note))
        c.id = int(cur.lastrowid)
        for tag in c.factors:
            self._db.execute("INSERT INTO factors (checkin_id, tag) VALUES (?,?)",
                             (c.id, str(tag)))
        self._db.commit()
        return c

    def _row(self, r: sqlite3.Row) -> Checkin:
        factors = [fr["tag"] for fr in self._db.execute(
            "SELECT tag FROM factors WHERE checkin_id=?", (r["id"],)).fetchall()]
        return Checkin(id=r["id"], at=r["at"], energy=r["energy"],
                       mood_valence=r["mood_valence"], stress=r["stress"],
                       sleep_hours=r["sleep_hours"], note=r["note"] or "", factors=factors)

    def since(self, start: datetime) -> list[Checkin]:
        return [self._row(r) for r in self._db.execute(
            "SELECT * FROM checkins WHERE at>=? ORDER BY at", (_iso(start),)).fetchall()]

    def all(self) -> list[Checkin]:
        return [self._row(r) for r in
                self._db.execute("SELECT * FROM checkins ORDER BY at").fetchall()]

    def close(self) -> None:
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
        sleep = None
        try:
            sleep = float(sleep_hours) if sleep_hours is not None else None
        except (TypeError, ValueError):
            sleep = None
        return self.store.add(Checkin(
            at=_iso(at or _now()), energy=_clamp(energy, 1, 5),
            mood_valence=resolve_mood(mood), stress=_clamp(stress, 1, 5),
            sleep_hours=sleep, note=note, factors=list(factors or [])))

    def today(self, now: datetime | None = None) -> dict[str, object]:
        now = now or _now()
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        todays = self.store.since(start)
        energies = [c.energy for c in todays if c.energy is not None]
        return {
            "checkins": len(todays),
            "avg_energy": round(sum(energies) / len(energies), 1) if energies else None,
            "logged": bool(todays),
        }

    def trend(self, days: int = 14, now: datetime | None = None) -> dict[str, object]:
        now = now or _now()
        window = self.store.since(now - timedelta(days=days))

        def _avg(vals):
            vals = [v for v in vals if v is not None]
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
        window = self.store.since(now - timedelta(days=days))
        energy_by_day = _daily_mean([(c.day(), float(c.energy)) for c in window
                                     if c.energy is not None])
        sleep_by_day = _daily_mean([(c.day(), float(c.sleep_hours)) for c in window
                                    if c.sleep_hours is not None])

        minutes_by_day: dict[date, float] = {}
        completed_by_day: dict[date, float] = {}
        if focus is not None:
            try:
                for s in focus.store.all():
                    if not s.started_at or not s.ended_at:
                        continue
                    d = _parse(s.started_at).date()
                    if (now.date() - d).days > days:
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
