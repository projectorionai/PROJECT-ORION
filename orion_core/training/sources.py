"""
Read-only miners for ORION's own experience.

ORION keeps what it learned from being used in several small SQLite stores in
the config directory. Four of them say something about "the owner asked, ORION
chose a tool":

* ``orion_core.db``      ``episodes`` - the conversation, one row per utterance
                         (roles ``user`` and ``orion``; remote-phone turns carry
                         other role names and are left out).
* ``resolver_shadow.db`` ``observations`` - for each live turn, the first 120
                         characters of what was said and the tool the model chose.
* ``jobs.db``            ``jobs`` - tool calls that became durable jobs, with
                         their arguments, result and outcome.
* ``reflex_learning.db`` ``observations`` - phrases the reflex path has seen
                         work or fail, with the tool and action.

Joining them in time gives real turns: the owner's words, the tool chosen,
sometimes the arguments and result, and what ORION said back.

Everything is opened ``mode=ro`` with ``PRAGMA query_only`` - no store's data
is ever written. A store in WAL mode (ORION's are) gets SQLite's own empty
``-wal`` and ``-shm`` housekeeping files beside it if they are not already
there, as it does for any reader; the database file itself is not touched.
A store that is missing, locked or in an
unexpected shape is reported and skipped; mining never raises because one
file is odd. Not persisted anywhere, and so not mined: the execution
receipts (``execution_receipts.py`` holds a bounded in-memory ledger with no
arguments) and the research-commitment audit (``research_commitments.py``
does no I/O at all).

Nothing read here is returned in a report: counts only.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

STORES = {
    "episodes": "orion_core.db",
    "shadow": "resolver_shadow.db",
    "jobs": "jobs.db",
    "reflex": "reflex_learning.db",
}

#: How close in time a recorded tool choice must be to the utterance it belongs to.
MATCH_WINDOW_S = 120.0
#: A reply this long after the question is not the reply to it.
REPLY_WINDOW_S = 180.0
#: A correction this long after the answer is not about it.
CORRECTION_WINDOW_S = 300.0
#: Characters the shadow log keeps of an utterance.
SHADOW_QUERY_CHARS = 120

CORRECTION = re.compile(
    r"(?i)^(?:no+[,.!]?\s|no$|nope\b|nah\b|wrong\b|that'?s (?:not|wrong)\b|that is not\b|"
    r"that isn'?t\b|that wasn'?t\b|not that\b|not what i\b|i (?:said|meant|asked|wanted)\b|"
    r"what i (?:said|meant|asked)\b|i didn'?t (?:ask|say|mean|want)\b|you (?:misheard|got (?:it|that) wrong)\b|"
    r"try again\b|actually,? i\b|sorry,? i meant\b|no no\b)")


def normalise(text: str) -> str:
    """Lower-case, punctuation-free, single-spaced: for matching, never for output."""
    return " ".join(re.sub(r"[^\w\s']", " ", str(text or "").lower()).split())


def _key(text: str) -> str:
    return normalise(text)[:40]


def is_correction(text: str) -> bool:
    return bool(CORRECTION.match(" ".join(str(text or "").split())))


# ── opening stores ──────────────────────────────────────────────────────────────

def ro_connect(path: Path) -> sqlite3.Connection | None:
    """A read-only connection to *path*, or None if it is not there or not readable."""
    try:
        if not Path(path).is_file():
            return None
        uri = Path(path).resolve().as_uri() + "?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=5.0)
        from ..db import apply_pragmas
        apply_pragmas(con)               # ORION's standard settings, as backup_manager's reader
        con.execute("PRAGMA query_only = ON")
        con.row_factory = sqlite3.Row
        return con
    except (sqlite3.Error, OSError, ValueError):
        return None


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in con.execute(f'PRAGMA table_info("{table}")')}
    except sqlite3.Error:
        return set()


def _epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return 0.0


# ── records ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Episode:
    id: int
    role: str
    content: str
    at: float


@dataclass(frozen=True)
class ShadowRow:
    at: float
    query: str
    tool: str


@dataclass(frozen=True)
class JobRow:
    name: str
    args: dict[str, Any]
    result: str
    ok: bool
    origin: str
    at: float
    kind: str


@dataclass(frozen=True)
class ReflexRow:
    phrase: str
    tool: str
    action: str
    hits: int
    failures: int
    promoted: bool


@dataclass
class ToolEvent:
    name: str
    args: dict[str, Any] | None = None
    result: str | None = None
    ok: bool | None = None
    source: str = ""            # shadow | job


@dataclass
class RealTurn:
    """One owner utterance with whatever ORION recorded about answering it."""

    user_id: int
    user: str
    at: float
    reply: str = ""
    tools: list[ToolEvent] = field(default_factory=list)
    history: tuple[str, str] | None = None     # the exchange just before (user, reply)
    next_user: str = ""
    next_gap_s: float = 1e9


# ── readers ─────────────────────────────────────────────────────────────────────

def read_episodes(config_dir: Path) -> list[Episode]:
    con = ro_connect(Path(config_dir) / STORES["episodes"])
    if con is None:
        return []
    try:
        if not {"id", "role", "content", "created_at"} <= _columns(con, "episodes"):
            return []
        rows = con.execute(
            "SELECT id, role, content, created_at FROM episodes "
            "WHERE role IN ('user', 'orion') ORDER BY id").fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [Episode(int(r["id"]), str(r["role"]), str(r["content"] or ""), _epoch(r["created_at"]))
            for r in rows]


def read_shadow(config_dir: Path) -> list[ShadowRow]:
    con = ro_connect(Path(config_dir) / STORES["shadow"])
    if con is None:
        return []
    try:
        if not {"at", "query", "tool"} <= _columns(con, "observations"):
            return []
        rows = con.execute("SELECT at, query, tool FROM observations ORDER BY at").fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [ShadowRow(float(r["at"] or 0.0), str(r["query"] or ""), str(r["tool"] or "")) for r in rows]


def read_jobs(config_dir: Path) -> list[JobRow]:
    con = ro_connect(Path(config_dir) / STORES["jobs"])
    if con is None:
        return []
    try:
        wanted = {"kind", "name", "state", "origin", "args_json", "result", "ok", "created_at"}
        if not wanted <= _columns(con, "jobs"):
            return []
        rows = con.execute(
            "SELECT kind, name, state, origin, args_json, result, ok, created_at FROM jobs "
            "WHERE state = 'completed' AND kind = 'tool' ORDER BY created_at").fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    jobs = []
    for r in rows:
        try:
            args = json.loads(r["args_json"] or "{}")
        except ValueError:
            continue
        if not isinstance(args, dict) or args.get("_truncated"):
            continue
        jobs.append(JobRow(str(r["name"] or ""), args, str(r["result"] or ""), bool(r["ok"]),
                           str(r["origin"] or ""), float(r["created_at"] or 0.0), str(r["kind"] or "")))
    return jobs


def read_reflex(config_dir: Path) -> list[ReflexRow]:
    con = ro_connect(Path(config_dir) / STORES["reflex"])
    if con is None:
        return []
    try:
        if not {"phrase", "tool", "action", "hits", "failures", "promoted"} <= _columns(con, "observations"):
            return []
        rows = con.execute("SELECT phrase, tool, action, hits, failures, promoted "
                           "FROM observations").fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [ReflexRow(str(r["phrase"] or ""), str(r["tool"] or ""), str(r["action"] or ""),
                      int(r["hits"] or 0), int(r["failures"] or 0), bool(r["promoted"]))
            for r in rows]


def fingerprint(config_dir: Path) -> dict[str, Any]:
    """Which stores were found, with row counts - no content."""
    out: dict[str, Any] = {}
    readers = {"episodes": read_episodes, "shadow": read_shadow, "jobs": read_jobs,
               "reflex": read_reflex}
    for name, filename in STORES.items():
        present = (Path(config_dir) / filename).is_file()
        out[name] = {"file": filename, "found": present,
                     "rows": len(readers[name](Path(config_dir))) if present else 0}
    return out


# ── joining ─────────────────────────────────────────────────────────────────────

def build_turns(episodes: Iterable[Episode], shadow: Iterable[ShadowRow],
                jobs: Iterable[JobRow]) -> list[RealTurn]:
    """Real turns: every owner utterance, with the tools ORION recorded for it
    and ORION's reply. Order is time order."""
    rows = sorted(episodes, key=lambda e: e.id)
    turns: list[RealTurn] = []
    by_key: dict[str, list[RealTurn]] = {}
    previous: tuple[str, str] | None = None
    for index, ep in enumerate(rows):
        if ep.role != "user" or not ep.content.strip():
            continue
        turn = RealTurn(ep.id, ep.content.strip(), ep.at, history=previous)
        nxt = rows[index + 1] if index + 1 < len(rows) else None
        if nxt is not None and nxt.role == "orion" and 0 <= nxt.at - ep.at <= REPLY_WINDOW_S:
            turn.reply = nxt.content.strip()
        previous = (turn.user, turn.reply) if turn.reply else None
        turns.append(turn)
        by_key.setdefault(_key(turn.user), []).append(turn)
    for a, b in zip(turns, turns[1:]):
        a.next_user, a.next_gap_s = b.user, max(0.0, b.at - a.at)

    for row in sorted(shadow, key=lambda s: s.at):
        if not row.query.strip() or not row.tool.strip():
            continue
        best: RealTurn | None = None
        for turn in by_key.get(_key(row.query), ()):
            gap = row.at - turn.at
            if -5.0 <= gap <= MATCH_WINDOW_S and (best is None or abs(gap) < abs(row.at - best.at)):
                best = turn
        if best is not None:
            best.tools.append(ToolEvent(row.tool.strip(), source="shadow"))

    ordered = sorted(turns, key=lambda t: t.at)
    for job in sorted(jobs, key=lambda j: j.at):
        if job.origin.lower() in {"remote", "phone"}:
            continue
        candidates = [t for t in ordered if -5.0 <= job.at - t.at <= MATCH_WINDOW_S]
        if not candidates:
            continue
        turn = candidates[-1]
        existing = next((e for e in turn.tools if e.name == job.name and e.args is None), None)
        if existing is not None:
            existing.args, existing.result, existing.ok = job.args, job.result, job.ok
            existing.source = "job"
        else:
            turn.tools.append(ToolEvent(job.name, job.args, job.result, job.ok, "job"))
    return turns


__all__ = ["CORRECTION", "CORRECTION_WINDOW_S", "Episode", "JobRow", "MATCH_WINDOW_S", "RealTurn",
           "ReflexRow", "STORES", "ShadowRow", "ToolEvent", "build_turns", "fingerprint",
           "is_correction", "normalise", "read_episodes", "read_jobs", "read_reflex", "read_shadow",
           "ro_connect"]
