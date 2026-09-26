"""
Companion Intelligence — ORION's continuity layer (additive module).

``CompanionEngine`` gives ORION durable awareness of what the user has been
doing across sessions: activities, goals, habits and achievements, all in a
local SQLite ledger, so it can open a session with grounded continuity —

    "You were working on ExampleStore yesterday."
    "You studied neuroscience earlier this week."
    "Your ORION command deck implementation is still incomplete."

``CompanionAgent`` is the conversational skin over the engine: it passively
observes dispatcher traffic for "working on / studying X" signals, and answers
companion queries (brief, goals, habits, achievements).

Design constraints from the execution directive: professional, helpful,
proactive but never intrusive — the companion *volunteers one line* of
continuity at session start and otherwise only speaks when asked.  It
integrates with (never replaces) the existing organs: project/task state lives
in ``CognitiveStateManager``, durable facts in the ``MemoryAgent``, and the
activity/habit/achievement ledger — which no existing module owns — lives here.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from .constants import CONFIG_DIR
from .utils import first_line, utc_stamp
from .db import apply_pragmas

_ACTIVITY_KINDS = {"worked_on", "studied", "researched", "built", "completed",
                   "discussed", "planned"}

# Passive observation: "I'm working on ExampleStore", "studying neuroscience", …
_OBSERVE_RE = re.compile(
    r"(?i)\b(?:i(?:'m| am| was)?|we(?:'re| are| were)?)\s+"
    r"(working on|studying|researching|building|planning)\s+"
    r"(?:the |my |our )?([A-Za-z0-9][\w &'-]{2,60})"
)
_VERB_TO_KIND = {
    "working on": "worked_on", "studying": "studied", "researching": "researched",
    "building": "built", "planning": "planned",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _humanise_when(at_iso: str, now: datetime) -> str:
    """'earlier today' / 'yesterday' / 'on Monday' / 'earlier this week' / date."""
    try:
        at = datetime.strptime(at_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return "recently"
    days = (now.date() - at.date()).days
    if days <= 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return "earlier this week" if days > 3 else f"on {at.strftime('%A')}"
    if days < 14:
        return "last week"
    return f"on {at.strftime('%d %B')}"


@dataclass
class Goal:
    slug: str
    title: str
    progress: int          # 0–100
    target: str            # optional ISO date
    status: str            # open | achieved | dropped
    updated_at: str


class CompanionEngine:
    """SQLite-backed activity / goal / habit / achievement continuity ledger."""

    SCHEMA = "orion.companion.v1"

    def __init__(
        self,
        bus: Any | None = None,
        memory: Any | None = None,
        cognition: Any | None = None,
        telemetry: Any | None = None,
        db_path: Path | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.cognition = cognition
        self.telemetry = telemetry
        self.db_path = db_path or (CONFIG_DIR / "companion.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self.conn)
        self.conn.row_factory = sqlite3.Row
        self._initialise()
        self._session_started = utc_stamp()
        self._session_row_id: int | None = None
        if self.telemetry is not None:
            try:
                self.telemetry.health.register("companion")
                self.telemetry.health.beat("companion", "OK", "continuity ledger ready")
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _initialise(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_activities_at ON activities(at);
                CREATE INDEX IF NOT EXISTS idx_activities_subject ON activities(subject);

                CREATE TABLE IF NOT EXISTS goals (
                    slug TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    target TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'open',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS habits (
                    name TEXT NOT NULL,
                    day TEXT NOT NULL,
                    PRIMARY KEY (name, day)
                );

                CREATE TABLE IF NOT EXISTS achievements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started TEXT NOT NULL,
                    ended TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT ''
                );
                """
            )
            self.conn.commit()

    # ── activity ledger ───────────────────────────────────────────────────────

    def log_activity(self, kind: str, subject: str, detail: str = "") -> dict[str, Any]:
        kind = kind if kind in _ACTIVITY_KINDS else "worked_on"
        subject = first_line(subject, 120).strip()
        if not subject:
            return {}
        at = utc_stamp()
        with self._lock:
            # Collapse same-subject same-kind entries within the day: bump, don't
            # duplicate — the ledger should read like a diary, not a firehose.
            today = at[:10]
            row = self.conn.execute(
                "SELECT id FROM activities WHERE kind=? AND subject=? AND at LIKE ?",
                (kind, subject, f"{today}%"),
            ).fetchone()
            if row is not None:
                self.conn.execute(
                    "UPDATE activities SET at=?, detail=? WHERE id=?",
                    (at, first_line(detail, 240), row["id"]))
            else:
                self.conn.execute(
                    "INSERT INTO activities(kind, subject, detail, at) VALUES (?,?,?,?)",
                    (kind, subject, first_line(detail, 240), at))
            self.conn.commit()
        if self.memory is not None:
            try:
                self.memory.remember(
                    "long_term", f"activity_{kind}_{subject[:40].lower().replace(' ', '_')}",
                    f"{kind.replace('_', ' ')} {subject}" + (f" — {detail}" if detail else ""))
            except Exception:
                pass
        return {"kind": kind, "subject": subject, "at": at}

    def recent_activities(self, days: int = 7, limit: int = 20) -> list[dict[str, Any]]:
        cutoff = (_utc_now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock:
            rows = self.conn.execute(
                "SELECT kind, subject, detail, at FROM activities "
                "WHERE at >= ? ORDER BY at DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── goals ────────────────────────────────────────────────────────────────

    @staticmethod
    def _slug(title: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:64]

    def upsert_goal(self, title: str, target: str = "", progress: int | None = None) -> Goal:
        slug = self._slug(title)
        with self._lock:
            row = self.conn.execute("SELECT * FROM goals WHERE slug=?", (slug,)).fetchone()
            pct = max(0, min(100, int(progress if progress is not None
                                      else (row["progress"] if row else 0))))
            status = "achieved" if pct >= 100 else "open"
            self.conn.execute(
                """
                INSERT INTO goals(slug, title, progress, target, status, updated_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(slug) DO UPDATE SET
                    progress=excluded.progress, status=excluded.status,
                    target=CASE WHEN excluded.target != '' THEN excluded.target ELSE goals.target END,
                    updated_at=excluded.updated_at
                """,
                (slug, first_line(title, 160), pct, str(target or ""), status, utc_stamp()))
            self.conn.commit()
        if status == "achieved" and (row is None or row["status"] != "achieved"):
            self.record_achievement(f"Goal achieved: {title}")
        return Goal(slug, first_line(title, 160), pct, str(target or ""), status, utc_stamp())

    def goals(self, include_achieved: bool = False) -> list[Goal]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM goals ORDER BY updated_at DESC").fetchall()
        out = [Goal(r["slug"], r["title"], r["progress"], r["target"],
                    r["status"], r["updated_at"]) for r in rows]
        if not include_achieved:
            out = [g for g in out if g.status == "open"]
        return out

    # ── habits ───────────────────────────────────────────────────────────────

    def mark_habit(self, name: str, day: str = "") -> dict[str, Any]:
        name = first_line(name, 80).strip().lower()
        if not name:
            return {}
        day = day or _utc_now().strftime("%Y-%m-%d")
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO habits(name, day) VALUES (?,?)", (name, day))
            self.conn.commit()
        return {"name": name, "day": day, "streak": self.habit_streak(name)}

    def habit_streak(self, name: str) -> int:
        """Consecutive days ending today (or yesterday, so a streak isn't lost
        before today's mark)."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT day FROM habits WHERE name=? ORDER BY day DESC LIMIT 366",
                (name.strip().lower(),),
            ).fetchall()
        days = {r["day"] for r in rows}
        if not days:
            return 0
        cursor = _utc_now().date()
        if cursor.strftime("%Y-%m-%d") not in days:
            cursor -= timedelta(days=1)          # today not yet marked
        streak = 0
        while cursor.strftime("%Y-%m-%d") in days:
            streak += 1
            cursor -= timedelta(days=1)
        return streak

    def habits(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT name, COUNT(*) AS marks, MAX(day) AS last FROM habits "
                "GROUP BY name ORDER BY last DESC").fetchall()
        return [{"name": r["name"], "marks": r["marks"], "last": r["last"],
                 "streak": self.habit_streak(r["name"])} for r in rows]

    # ── achievements ─────────────────────────────────────────────────────────

    def record_achievement(self, title: str, detail: str = "") -> dict[str, Any]:
        title = first_line(title, 160).strip()
        if not title:
            return {}
        at = utc_stamp()
        with self._lock:
            self.conn.execute(
                "INSERT INTO achievements(title, detail, at) VALUES (?,?,?)",
                (title, first_line(detail, 240), at))
            self.conn.commit()
        if self.bus is not None:
            try:
                self.bus.log.emit(f"COMPANION: achievement — {title}")
            except Exception:
                pass
        return {"title": title, "at": at}

    def achievements(self, limit: int = 12) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT title, detail, at FROM achievements ORDER BY at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ── session continuity ───────────────────────────────────────────────────

    def begin_session(self) -> None:
        self._session_started = utc_stamp()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO sessions(started) VALUES (?)", (self._session_started,))
            self._session_row_id = cur.lastrowid
            self.conn.commit()

    def end_session(self, summary: str = "") -> None:
        with self._lock:
            row = self.conn.execute(
                "SELECT id FROM sessions WHERE ended='' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                self.conn.execute(
                    "UPDATE sessions SET ended=?, summary=? WHERE id=?",
                    (utc_stamp(), first_line(summary, 300), row["id"]))
                self.conn.commit()

    def last_session(self) -> Optional[dict[str, Any]]:
        """The most recent session other than the current one (id-based, so
        two sessions inside the same clock second still order correctly)."""
        with self._lock:
            row = self.conn.execute(
                "SELECT id, started, ended, summary FROM sessions "
                "WHERE id != COALESCE(?, -1) ORDER BY id DESC LIMIT 1",
                (self._session_row_id,)).fetchone()
        if row is None:
            return None
        return {"started": row["started"], "ended": row["ended"],
                "summary": row["summary"]}

    # ── the continuity brief ─────────────────────────────────────────────────

    def continuity_brief(self, max_lines: int = 3) -> str:
        """Grounded, one-glance continuity: recent work, open thread, streak."""
        now = _utc_now()
        lines: list[str] = []

        acts = self.recent_activities(days=7, limit=6)
        seen_subjects: set[str] = set()
        for act in acts:
            subject = act["subject"]
            if subject.lower() in seen_subjects:
                continue
            seen_subjects.add(subject.lower())
            verb = {"worked_on": "were working on", "studied": "studied",
                    "researched": "were researching", "built": "were building",
                    "completed": "completed", "discussed": "discussed",
                    "planned": "were planning"}.get(act["kind"], "were working on")
            lines.append(f"You {verb} {subject} {_humanise_when(act['at'], now)}.")
            if len(lines) >= max_lines - 1:
                break

        # One open thread from the cognitive state (never more — not intrusive).
        state: dict[str, Any] = {}
        if self.cognition is not None:
            try:
                state = self.cognition.snapshot()
            except Exception:
                state = {}
        pending = [t for t in (state.get("pending_tasks") or {}).values()
                   if str(t.get("status", "pending")) != "completed"]
        if pending:
            top = sorted(pending, key=lambda t: str(t.get("updated_at", "")), reverse=True)[0]
            lines.append(f"Your task '{top.get('title', '')}' is still open.")
        else:
            open_goals = self.goals()
            if open_goals:
                g = open_goals[0]
                lines.append(f"Goal '{g.title}' stands at {g.progress}%.")

        return " ".join(lines[:max_lines]) if lines else ""

    def status(self) -> dict[str, Any]:
        with self._lock:
            acts = self.conn.execute("SELECT COUNT(*) n FROM activities").fetchone()["n"]
            goals = self.conn.execute(
                "SELECT COUNT(*) n FROM goals WHERE status='open'").fetchone()["n"]
            achieved = self.conn.execute("SELECT COUNT(*) n FROM achievements").fetchone()["n"]
            habits = self.conn.execute(
                "SELECT COUNT(DISTINCT name) n FROM habits").fetchone()["n"]
        return {"activities": acts, "open_goals": goals,
                "achievements": achieved, "habits": habits}


class CompanionAgent:
    """Conversational skin: passive observation + companion queries."""

    def __init__(self, engine: CompanionEngine, bus: Any | None = None) -> None:
        self.engine = engine
        self.bus = bus

    # Words that end a subject ("ExampleStore today and…" → "ExampleStore") and
    # pronoun-ish subjects that carry no continuity value on their own.
    _TAIL_RE = re.compile(
        r"\s+(?:today|tonight|yesterday|tomorrow|this|that|right|now|again|"
        r"currently|lately|and|but|so|because|since|while)\b.*$", re.IGNORECASE)
    _JUNK_SUBJECTS = {"it", "this", "that", "them", "stuff", "things", "something"}

    def observe(self, user_text: str) -> Optional[dict[str, Any]]:
        """Silently harvest 'working on X' signals from ordinary conversation."""
        match = _OBSERVE_RE.search(str(user_text or ""))
        if not match:
            return None
        kind = _VERB_TO_KIND.get(match.group(1).lower(), "worked_on")
        subject = self._TAIL_RE.sub("", match.group(2)).strip(" .,!?")
        if len(subject) < 3 or subject.lower() in self._JUNK_SUBJECTS:
            return None
        return self.engine.log_activity(kind, subject)

    def greeting_line(self) -> str:
        """One line of continuity for session start; empty when nothing to say."""
        brief = self.engine.continuity_brief(max_lines=2)
        return brief
