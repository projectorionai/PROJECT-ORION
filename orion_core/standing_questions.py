"""
Standing questions (CAP-05) — research that accretes instead of restarting.

    "Register the things I care about, and surface what's new — not the same
     report every morning."

The research director already runs a one-shot agenda: queue a topic, it gets
researched once, it's done. This is the other shape — a question you leave
*standing*. ORION revisits it on a cadence, and each time reports only the
**delta**: the points that weren't in the last look. "Since yesterday: three new
papers on neural interfaces, one worth your time" rather than the whole field
again.

Two pieces make that work:

**A cadence, and a due list.** Each question carries how often it should be
revisited (hourly / daily / weekly, or a number of hours). ``due()`` returns the
ones whose time has come — which is what a standby loop asks for when it's
looking for permitted background work to do.

**A memory of what's already known.** Every check extracts the summary into
discrete points and stores them. The next check compares against that memory and
keeps only what's genuinely new, so the delta is real and not just a reworded
repeat. The comparison is fuzzy enough that "3 new papers" and "three new
papers" count as the same point.

The research itself is injected — ``run_due`` takes an async researcher — so the
store, the cadence and the delta are all tested without running a single real
query. Persistence is a small SQLite file, private and local, like the rest.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Awaitable, Callable

from .constants import CONFIG_DIR
from .utils import utc_stamp
from .db import apply_pragmas

STORE_PATH = CONFIG_DIR / "standing_questions.db"

# Named cadences → hours. A bare number of hours is also accepted.
CADENCES: dict[str, float] = {
    "hourly": 1.0, "daily": 24.0, "twice daily": 12.0, "weekly": 168.0,
    "fortnightly": 336.0, "monthly": 720.0,
}
DEFAULT_CADENCE_HOURS = 24.0


def resolve_cadence(value: Any) -> float:
    """'daily' / 'weekly' / '6' / 6 → hours; defaults to daily."""
    if value is None or value == "":
        return DEFAULT_CADENCE_HOURS
    token = str(value).strip().lower()
    if token in CADENCES:
        return CADENCES[token]
    try:
        hours = float(token)
        return max(0.5, hours)
    except ValueError:
        return DEFAULT_CADENCE_HOURS


def _key_points(summary: str) -> list[str]:
    """Break a summary into discrete, comparable points."""
    points: list[str] = []
    for raw in re.split(r"[\n\r]+|(?<=[.!?])\s+", str(summary or "")):
        line = raw.strip().lstrip("-•*").strip()
        if len(line) >= 12:                 # skip fragments and headers
            points.append(line)
    return points


def _norm(point: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", point.lower())).strip()


@dataclass
class StandingQuestion:
    text: str
    cadence_hours: float = DEFAULT_CADENCE_HOURS
    last_checked: str = ""
    active: bool = True
    created_at: str = field(default_factory=utc_stamp)
    id: int | None = None

    def is_due(self, now: datetime | None = None) -> bool:
        if not self.active:
            return False
        if not self.last_checked:
            return True
        now = now or datetime.now(timezone.utc)
        try:
            last = datetime.fromisoformat(self.last_checked)
        except ValueError:
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return now - last >= timedelta(hours=self.cadence_hours)


@dataclass
class Delta:
    question: StandingQuestion
    summary: str
    new_points: list[str]

    @property
    def has_news(self) -> bool:
        return bool(self.new_points)

    def describe(self) -> str:
        if not self.new_points:
            return f"'{self.question.text}': nothing new since the last look."
        head = f"'{self.question.text}' — {len(self.new_points)} new:"
        return head + "\n" + "\n".join(f"  • {p}" for p in self.new_points)


class StandingQuestions:
    """The store, the cadence, and the delta — behind one object."""

    def __init__(self, path: Any = None) -> None:
        # Resolved when called, not bound as a default when the class was
        # defined — otherwise STORE_PATH cannot be redirected (in tests, or by a
        # relocated config) and every store built with no path went to the
        # real one regardless.
        path = STORE_PATH if path is None else path
        self._lock = RLock()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._conn)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                text          TEXT NOT NULL UNIQUE,
                cadence_hours REAL NOT NULL DEFAULT 24,
                last_checked  TEXT NOT NULL DEFAULT '',
                active        INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS known_points (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                norm        TEXT NOT NULL,
                point       TEXT NOT NULL,
                first_seen  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                summary     TEXT NOT NULL,
                new_count   INTEGER NOT NULL DEFAULT 0,
                at          TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # ── management ────────────────────────────────────────────────────────────

    def add(self, text: str, cadence: Any = None) -> StandingQuestion:
        text = str(text or "").strip()
        if not text:
            raise ValueError("a standing question needs text")
        hours = resolve_cadence(cadence)
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO questions(text, cadence_hours, created_at)
                   VALUES (?,?,?)
                   ON CONFLICT(text) DO UPDATE SET cadence_hours=excluded.cadence_hours,
                       active=1""",
                (text, hours, utc_stamp()))
            self._conn.commit()
            qid = cur.lastrowid or self._id_of(text)
        return self.get(qid)

    def remove(self, question_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE questions SET active=0 WHERE id=?", (question_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list(self, include_inactive: bool = False) -> list[StandingQuestion]:
        with self._lock:
            sql = "SELECT * FROM questions"
            if not include_inactive:
                sql += " WHERE active=1"
            sql += " ORDER BY created_at"
            rows = self._conn.execute(sql).fetchall()
        return [_row_to_q(r) for r in rows]

    def get(self, question_id: int) -> StandingQuestion | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
        return _row_to_q(row) if row else None

    def due(self, now: datetime | None = None) -> list[StandingQuestion]:
        return [q for q in self.list() if q.is_due(now)]

    # ── the delta ─────────────────────────────────────────────────────────────

    def record_check(self, question_id: int, summary: str,
                     now: datetime | None = None) -> Delta:
        """Store a fresh summary, and return only the points not seen before."""
        now = now or datetime.now(timezone.utc)
        question = self.get(question_id)
        if question is None:
            raise ValueError(f"no standing question #{question_id}")
        known = self._known_norms(question_id)
        new_points: list[str] = []
        with self._lock:
            for point in _key_points(summary):
                norm = _norm(point)
                if not norm or self._is_known(norm, known):
                    continue
                known.add(norm)
                new_points.append(point)
                self._conn.execute(
                    "INSERT INTO known_points(question_id, norm, point, first_seen)"
                    " VALUES (?,?,?,?)", (question_id, norm, point, utc_stamp()))
            self._conn.execute(
                "INSERT INTO checks(question_id, summary, new_count, at)"
                " VALUES (?,?,?,?)", (question_id, summary, len(new_points), utc_stamp()))
            self._conn.execute(
                "UPDATE questions SET last_checked=? WHERE id=?",
                (now.isoformat(), question_id))
            self._conn.commit()
        question.last_checked = now.isoformat()
        return Delta(question, summary, new_points)

    async def run_due(
        self,
        researcher: Callable[[str], Awaitable[str]],
        now: datetime | None = None,
    ) -> list[Delta]:
        """Research every due question with the injected *researcher* and return
        the deltas. This is what a standby loop calls when it has permission to
        do background work."""
        deltas: list[Delta] = []
        for question in self.due(now):
            try:
                summary = str(await researcher(question.text) or "")
            except Exception:
                continue                    # a failed query just waits for next due
            if summary.strip():
                deltas.append(self.record_check(question.id, summary, now))
        return deltas

    # ── helpers ───────────────────────────────────────────────────────────────

    def _known_norms(self, question_id: int) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT norm FROM known_points WHERE question_id=?",
                (question_id,)).fetchall()
        return {r[0] for r in rows}

    @staticmethod
    def _is_known(norm: str, known: set[str]) -> bool:
        if norm in known:
            return True
        # fuzzy: near-duplicate wording counts as already-seen
        for prior in known:
            if difflib.SequenceMatcher(None, norm, prior).ratio() >= 0.88:
                return True
        return False

    def _id_of(self, text: str) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM questions WHERE text=?", (text,)).fetchone()
        return int(row[0]) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_q(row: sqlite3.Row) -> StandingQuestion:
    return StandingQuestion(
        id=int(row["id"]), text=row["text"],
        cadence_hours=float(row["cadence_hours"]),
        last_checked=row["last_checked"], active=bool(row["active"]),
        created_at=row["created_at"])


__all__ = [
    "StandingQuestions", "StandingQuestion", "Delta",
    "CADENCES", "resolve_cadence",
]
