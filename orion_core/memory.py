"""
Memory subsystem.

Two co-operating layers:

    OrionMemoryMatrix — the persistent SQLite FTS5 store (durable facts in the
        `intelligence` table, episodic conversation history in `episodes`).
        Migrated intact from Mark VII.

    MemoryAgent — the Mark VIII front door.  Combines a volatile *session*
        layer (recent turns, working notes that live only for this run) with
        the persistent matrix, and exposes one `prompt_context()` used by the
        provider router so the model always sees both horizons of memory.

Every other module should depend on MemoryAgent, not the matrix directly;
the agent forwards the full matrix API so existing call sites keep working.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any

from .bus import OrionBus
from .constants import BASE_DIR
from .security import SecuritySanitiser
from .utils import utc_stamp


class MemoryTier(str, Enum):
    """
    The seven memory horizons (Phase 11).

    SHORT_TERM   — the last handful of turns (volatile, RAM).
    SESSION      — pinned working notes for this run only (volatile, RAM).
    CONVERSATION — full episodic history of what was said and when (persistent).
    LONG_TERM    — durable user facts / preferences (persistent).
    KNOWLEDGE    — durable reference knowledge ORION has learned (persistent).
    PROJECT      — facts scoped to a named project (persistent).
    WORKSPACE    — saved workspace snapshots for resume (persistent).
    """

    SHORT_TERM = "short_term"
    SESSION = "session"
    CONVERSATION = "conversation"
    LONG_TERM = "long_term"
    KNOWLEDGE = "knowledge"
    PROJECT = "project"
    WORKSPACE = "workspace"


# ──────────────────────────────────────────────────────────────────────────────
# PERSISTENT LAYER  (SQLite FTS5)
# ──────────────────────────────────────────────────────────────────────────────

class OrionMemoryMatrix:
    _FTS5_STRIP_RE = re.compile(
        r'["\'\[\](){}*?!^~\\]'
        r'|(?<!\w)AND(?!\w)'
        r'|(?<!\w)OR(?!\w)'
        r'|(?<!\w)NOT(?!\w)'
        r'|(?<!\w)NEAR(?!\w)',
        re.IGNORECASE,
    )
    _FTS5_COLLAPSE_RE = re.compile(r'\s{2,}')
    _FTS5_COLSPEC_RE  = re.compile(r'\b\w+\s*:')

    def __init__(self, db_path: Path, config_dir: Path, bus: OrionBus) -> None:
        self.db_path    = db_path
        self.config_dir = config_dir
        self.bus        = bus
        self._lock      = RLock()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialise()
        self._migrate_legacy()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _initialise(self) -> None:
        schema = """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS intelligence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL DEFAULT 'notes',
            key_ref TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_intelligence_key
            ON intelligence(category, key_ref);
        CREATE VIRTUAL TABLE IF NOT EXISTS intelligence_fts USING fts5(
            category, key_ref, value, updated_at,
            content='intelligence', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS intelligence_ai AFTER INSERT ON intelligence BEGIN
            INSERT INTO intelligence_fts(rowid, category, key_ref, value, updated_at)
            VALUES (new.id, new.category, new.key_ref, new.value, new.updated_at);
        END;
        CREATE TRIGGER IF NOT EXISTS intelligence_ad AFTER DELETE ON intelligence BEGIN
            INSERT INTO intelligence_fts(intelligence_fts, rowid, category, key_ref, value, updated_at)
            VALUES('delete', old.id, old.category, old.key_ref, old.value, old.updated_at);
        END;
        CREATE TRIGGER IF NOT EXISTS intelligence_au AFTER UPDATE ON intelligence BEGIN
            INSERT INTO intelligence_fts(intelligence_fts, rowid, category, key_ref, value, updated_at)
            VALUES('delete', old.id, old.category, old.key_ref, old.value, old.updated_at);
            INSERT INTO intelligence_fts(rowid, category, key_ref, value, updated_at)
            VALUES (new.id, new.category, new.key_ref, new.value, new.updated_at);
        END;
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
            role, content, created_at, content='episodes', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
            INSERT INTO episodes_fts(rowid, role, content, created_at)
            VALUES (new.id, new.role, new.content, new.created_at);
        END;
        """
        with self._lock:
            self.conn.executescript(schema)
            self.conn.commit()

    def _migrate_legacy(self) -> None:
        candidates = [
            self.config_dir / "orion_memory.json",
            BASE_DIR / "orion_memory.json",
            BASE_DIR / "memory" / "orion_memory.json",
        ]
        for legacy_path in candidates:
            if not legacy_path.exists():
                continue
            try:
                data = json.loads(legacy_path.read_text(encoding="utf-8"))
                migrated = 0
                for category, key, value, updated in self._flatten_legacy(data):
                    self.save(category, key, value, updated_at=updated, silent=True)
                    migrated += 1
                archive = legacy_path.with_name(
                    f"{legacy_path.stem}.archived-"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{legacy_path.suffix}"
                )
                legacy_path.rename(archive)
                self.bus.log.emit(
                    f"MEM: Migrated {migrated} legacy records into SQLite and archived source."
                )
            except Exception as exc:
                self.bus.log.emit(f"MEM: Legacy migration skipped - {exc}")

    def _flatten_legacy(self, data: Any) -> list[tuple[str, str, str, str]]:
        rows: list[tuple[str, str, str, str]] = []
        if not isinstance(data, dict):
            return rows
        for category, entries in data.items():
            safe_category = self._safe_slug(str(category or "notes"))
            if isinstance(entries, dict):
                for key, value in entries.items():
                    updated = utc_stamp()
                    if isinstance(value, dict):
                        raw = value.get("value", "")
                        updated = str(value.get("updated") or value.get("updated_at") or updated)
                    else:
                        raw = value
                    if raw is not None and str(raw).strip():
                        rows.append((safe_category, self._safe_slug(str(key)), str(raw), updated))
            elif entries is not None and str(entries).strip():
                rows.append(("notes", safe_category, str(entries), utc_stamp()))
        return rows

    def _safe_slug(self, value: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
        return value[:80] or "entry"

    @classmethod
    def _sanitise_fts_query(cls, raw: str) -> str:
        sanitised = cls._FTS5_STRIP_RE.sub(' ', raw)
        sanitised = cls._FTS5_COLSPEC_RE.sub(' ', sanitised)
        sanitised = cls._FTS5_COLLAPSE_RE.sub(' ', sanitised).strip()
        tokens = [t for t in sanitised.split() if len(t) >= 2 and not re.fullmatch(r'[^\w]+', t)]
        return ' '.join(tokens)

    def save(
        self,
        category: str,
        key: str,
        value: str,
        updated_at: str | None = None,
        silent: bool = False,
    ) -> str:
        category  = self._safe_slug(category or "notes")
        key       = self._safe_slug(key or "entry")
        value     = str(value or "").strip()
        if not value:
            return "No intelligence value supplied."
        SecuritySanitiser.guard_text(value, "memory.value")
        timestamp = updated_at or utc_stamp()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO intelligence(category, key_ref, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category, key_ref)
                DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (category, key, value[:1000], timestamp),
            )
            self.conn.commit()
        if not silent:
            self.bus.log.emit(f"MEM: synchronised {category}/{key}")
        return f"Stored intelligence: {category}/{key}."

    def query(self, query: str, limit: int = 8) -> list[dict[str, str]]:
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.query")
        limit = max(1, min(25, int(limit or 8)))
        if not query:
            return []
        sanitised_fts = self._sanitise_fts_query(query)
        with self._lock:
            if sanitised_fts:
                try:
                    rows = self.conn.execute(
                        """
                        SELECT category, key_ref, value, updated_at
                        FROM intelligence_fts
                        WHERE intelligence_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (sanitised_fts, limit),
                    ).fetchall()
                    if rows:
                        return [dict(row) for row in rows]
                except (sqlite3.OperationalError, sqlite3.Error):
                    pass
            like = f"%{query}%"
            rows = self.conn.execute(
                """
                SELECT category, key_ref, value, updated_at
                FROM intelligence
                WHERE value LIKE ? OR key_ref LIKE ? OR category LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (like, like, like, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def forget(self, category: str = "", key_prefix: str = "", contains: str = "") -> int:
        """
        Delete intelligence rows by any combination of category, key prefix and
        value substring; returns how many were removed.  The AFTER DELETE FTS
        trigger keeps the search index in sync.  A no-criteria call is a no-op
        (never wipes everything by accident).
        """
        clauses: list[str] = []
        params: list[Any] = []
        if category:
            clauses.append("category = ?")
            params.append(self._safe_slug(category))
        if key_prefix:
            clauses.append("key_ref LIKE ?")
            params.append(self._safe_slug(key_prefix) + "%")
        if contains:
            SecuritySanitiser.guard_text(contains, "memory.forget")
            clauses.append("value LIKE ?")
            params.append(f"%{contains}%")
        if not clauses:
            return 0
        where = " AND ".join(clauses)
        with self._lock:
            row = self.conn.execute(
                f"SELECT COUNT(*) AS n FROM intelligence WHERE {where}", params
            ).fetchone()
            removed = int(row["n"]) if row else 0
            if removed:
                self.conn.execute(f"DELETE FROM intelligence WHERE {where}", params)
                self.conn.commit()
        if removed:
            self.bus.log.emit(f"MEM: forgot {removed} record(s) [{where}].")
        return removed

    def records(self, query: str = "", limit: int = 100) -> list[dict[str, str]]:
        limit = max(1, min(500, int(limit or 100)))
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.records")
        if query:
            return self.query(query, limit=limit)
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT category, key_ref, value, updated_at
                FROM intelligence
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_episode(self, role: str, content: str) -> None:
        """Episodic conversation memory — timeline recall of what was said and when."""
        content = str(content or "").strip()
        if not content:
            return
        try:
            with self._lock:
                self.conn.execute(
                    "INSERT INTO episodes(role, content, created_at) VALUES (?, ?, ?)",
                    (str(role or "user")[:24], content[:2000], utc_stamp()),
                )
                self.conn.commit()
        except Exception:
            pass  # conversation logging must never break a live turn

    def recall_episodes(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.episodes")
        limit = max(1, min(50, int(limit or 10)))
        with self._lock:
            if query:
                sanitised = self._sanitise_fts_query(query)
                if sanitised:
                    try:
                        rows = self.conn.execute(
                            """
                            SELECT role, content, created_at FROM episodes_fts
                            WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?
                            """,
                            (sanitised, limit),
                        ).fetchall()
                        if rows:
                            return [dict(row) for row in rows]
                    except (sqlite3.OperationalError, sqlite3.Error):
                        pass
                rows = self.conn.execute(
                    """
                    SELECT role, content, created_at FROM episodes
                    WHERE content LIKE ? ORDER BY id DESC LIMIT ?
                    """,
                    (f"%{query}%", limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT role, content, created_at FROM episodes ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def prompt_context(self, limit: int = 18) -> str:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT category, key_ref, value, updated_at
                FROM intelligence
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        if not rows:
            return ""
        lines = ["[LOCAL INTELLIGENCE MATRIX - use naturally, never recite]"]
        for row in rows:
            label = f"{row['category']}/{row['key_ref']}".replace("_", " ")
            lines.append(f"- {label}: {row['value']}")
        return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# MEMORY AGENT  — session + persistent memory behind one interface
# ──────────────────────────────────────────────────────────────────────────────

class MemoryAgent:
    """
    Unified memory front door.

    Session layer (volatile, this run only):
        • a rolling window of recent conversation turns
        • ad-hoc working notes ("session facts") agents can pin mid-task

    Persistent layer:
        • the SQLite FTS5 matrix (facts + full episodic history)

    `prompt_context()` merges both so the model reasons with immediate
    conversational context *and* long-term knowledge.  The full matrix API is
    forwarded, so this object is a drop-in replacement anywhere a matrix was
    previously passed.
    """

    SESSION_TURN_WINDOW = 24        # recent turns kept in RAM
    SESSION_CONTEXT_TURNS = 8       # turns surfaced into the prompt

    def __init__(self, matrix: OrionMemoryMatrix, bus: OrionBus) -> None:
        self.matrix = matrix
        self.bus = bus
        self._session_turns: deque[dict[str, str]] = deque(maxlen=self.SESSION_TURN_WINDOW)
        self._session_facts: dict[str, str] = {}
        self._session_started = time.monotonic()
        # The project ORION is currently working within (Phase 11 resume).
        self._active_project: str = ""
        # Perfect conversation recording: a verbatim, timestamped transcript is
        # written to disk (conversations/<date>_<id>.jsonl) as turns happen, in
        # addition to the searchable SQLite episode log.
        self._transcript_path = self._new_transcript_path()

    def _new_transcript_path(self) -> Path:
        from .constants import BASE_DIR
        conv_dir = BASE_DIR / "conversations"
        try:
            conv_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return conv_dir / "session.jsonl"
        return conv_dir / f"{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"

    # ── session layer ─────────────────────────────────────────────────────────

    def log_episode(self, role: str, content: str) -> None:
        """Record a turn in every horizon: RAM window, durable SQLite log, and
        a verbatim on-disk transcript."""
        content = str(content or "").strip()
        if not content:
            return
        stamp = utc_stamp()
        self._session_turns.append(
            {"role": str(role or "user")[:24], "content": content[:600], "at": stamp}
        )
        self.matrix.log_episode(role, content)
        self._append_transcript(role, content, stamp)

    def _append_transcript(self, role: str, content: str, stamp: str) -> None:
        try:
            with self._transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": stamp, "role": role, "content": content},
                                        ensure_ascii=False) + "\n")
        except Exception:
            pass  # recording must never break a turn

    def transcript_path(self) -> Path:
        return self._transcript_path

    def export_transcript_markdown(self) -> str:
        """Render this session's verbatim transcript as Markdown, return its path."""
        from .constants import BASE_DIR
        lines = [f"# Conversation — {self._transcript_path.stem}", ""]
        try:
            for raw in self._transcript_path.read_text(encoding="utf-8").splitlines():
                turn = json.loads(raw)
                speaker = "You" if str(turn.get("role", "")).startswith("user") else "ORION"
                lines.append(f"**{speaker}** ({turn.get('at', '')[:19]}): {turn.get('content', '')}\n")
        except Exception:
            return ""
        out = BASE_DIR / "conversations" / f"{self._transcript_path.stem}.md"
        try:
            out.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            return ""
        return str(out)

    def note_session_fact(self, key: str, value: str) -> str:
        """Pin a working note for this session only (never written to disk)."""
        key = re.sub(r"[^a-zA-Z0-9_]+", "_", str(key or "note").strip().lower())[:60] or "note"
        value = str(value or "").strip()[:500]
        if not value:
            return "No session note supplied."
        self._session_facts[key] = value
        return f"Session note pinned: {key}."

    def session_facts(self) -> dict[str, str]:
        return dict(self._session_facts)

    def recent_turns(self, limit: int = 8) -> list[dict[str, str]]:
        turns = list(self._session_turns)
        return turns[-max(1, limit):]

    def session_uptime_minutes(self) -> float:
        return (time.monotonic() - self._session_started) / 60.0

    # ── merged context ────────────────────────────────────────────────────────

    def prompt_context(self, limit: int = 18) -> str:
        """Persistent matrix context + a compact session window."""
        parts: list[str] = []
        persistent = self.matrix.prompt_context(limit=limit)
        if persistent:
            parts.append(persistent.rstrip("\n"))
        if self._session_facts:
            fact_lines = ["[SESSION NOTES - current run only]"]
            for key, value in list(self._session_facts.items())[:12]:
                fact_lines.append(f"- {key.replace('_', ' ')}: {value}")
            parts.append("\n".join(fact_lines))
        recent = self.recent_turns(self.SESSION_CONTEXT_TURNS)
        if recent:
            turn_lines = ["[RECENT CONVERSATION - continue naturally, never recite]"]
            for turn in recent:
                turn_lines.append(f"- {turn['role']}: {turn['content'][:220]}")
            parts.append("\n".join(turn_lines))
        return ("\n".join(parts) + "\n") if parts else ""

    # ── persistent layer passthrough (drop-in matrix compatibility) ──────────

    def save(self, category: str, key: str, value: str, **kwargs: Any) -> str:
        return self.matrix.save(category, key, value, **kwargs)

    def query(self, query: str, limit: int = 8) -> list[dict[str, str]]:
        return self.matrix.query(query, limit=limit)

    def records(self, query: str = "", limit: int = 100) -> list[dict[str, str]]:
        return self.matrix.records(query=query, limit=limit)

    def forget(self, category: str = "", key_prefix: str = "", contains: str = "") -> int:
        """Delete persisted records by category / key prefix / value substring."""
        return self.matrix.forget(category=category, key_prefix=key_prefix, contains=contains)

    def recall_episodes(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        return self.matrix.recall_episodes(query, limit=limit)

    # ── tiered memory (Phase 11) ──────────────────────────────────────────────

    @staticmethod
    def _project_slug(name: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", str(name or "").strip().lower()).strip("_")[:48]

    def remember(self, tier: MemoryTier | str, key: str, value: str,
                 project: str = "") -> str:
        """
        Write to a specific memory tier.

        SHORT_TERM / SESSION are volatile (RAM); CONVERSATION appends an
        episode; the remaining tiers persist into the matrix under a
        tier-scoped category so recall can be tier-aware.
        """
        tier = MemoryTier(tier) if not isinstance(tier, MemoryTier) else tier
        if tier is MemoryTier.SHORT_TERM:
            self._session_turns.append(
                {"role": key or "note", "content": str(value)[:600], "at": utc_stamp()}
            )
            return "Noted in short-term memory."
        if tier is MemoryTier.SESSION:
            return self.note_session_fact(key, value)
        if tier is MemoryTier.CONVERSATION:
            self.log_episode(key or "note", value)
            return "Recorded in conversation memory."
        if tier is MemoryTier.PROJECT:
            proj = self._project_slug(project or self._active_project or "general")
            return self.matrix.save(f"project_{proj}", key, value)
        category = {
            MemoryTier.LONG_TERM: "long_term",
            MemoryTier.KNOWLEDGE: "knowledge",
            MemoryTier.WORKSPACE: "workspace",
        }.get(tier, "notes")
        return self.matrix.save(category, key, value)

    def recall(self, tier: MemoryTier | str, query: str = "",
               project: str = "", limit: int = 12) -> list[dict[str, str]]:
        """Read a specific tier (volatile tiers answered from RAM)."""
        tier = MemoryTier(tier) if not isinstance(tier, MemoryTier) else tier
        if tier is MemoryTier.SHORT_TERM:
            return self.recent_turns(limit)
        if tier is MemoryTier.SESSION:
            return [{"key_ref": k, "value": v} for k, v in self._session_facts.items()]
        if tier is MemoryTier.CONVERSATION:
            return self.matrix.recall_episodes(query, limit=limit)
        if tier is MemoryTier.PROJECT:
            proj = self._project_slug(project or self._active_project or "general")
            rows = self.matrix.records(query=query, limit=200)
            return [r for r in rows if r.get("category") == f"project_{proj}"][:limit]
        category = {
            MemoryTier.LONG_TERM: "long_term",
            MemoryTier.KNOWLEDGE: "knowledge",
            MemoryTier.WORKSPACE: "workspace",
        }.get(tier, "notes")
        rows = self.matrix.records(query=query, limit=200)
        return [r for r in rows if r.get("category") == category][:limit]

    # ── project focus + intelligent resume ────────────────────────────────────

    @property
    def active_project(self) -> str:
        return self._active_project

    def set_active_project(self, name: str) -> str:
        self._active_project = self._project_slug(name)
        self.bus.log.emit(f"MEM: active project set to '{self._active_project or 'none'}'.")
        return self._active_project

    def remember_project(self, key: str, value: str, project: str = "") -> str:
        return self.remember(MemoryTier.PROJECT, key, value, project=project)

    def remember_knowledge(self, key: str, value: str) -> str:
        return self.remember(MemoryTier.KNOWLEDGE, key, value)

    def resume_context(self, project: str = "") -> str:
        """
        Assemble a 'resume where we left off' briefing from persistent tiers:
        project facts + the latest workspace snapshot + recent conversation.
        """
        proj = self._project_slug(project or self._active_project)
        parts: list[str] = []
        if proj:
            project_rows = self.recall(MemoryTier.PROJECT, project=proj, limit=10)
            if project_rows:
                lines = [f"[PROJECT MEMORY — {proj}]"]
                for row in project_rows:
                    lines.append(f"- {row.get('key_ref', '')}: {row.get('value', '')}")
                parts.append("\n".join(lines))
        workspace_rows = self.recall(MemoryTier.WORKSPACE, limit=1)
        if workspace_rows:
            parts.append(f"[LAST WORKSPACE]\n{workspace_rows[0].get('value', '')[:600]}")
        recent = self.recall_episodes("", limit=6)
        if recent:
            lines = ["[WHERE WE LEFT OFF]"]
            for row in reversed(recent):
                lines.append(f"- {row['role']}: {row['content'][:180]}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def tiers_snapshot(self) -> dict[str, int]:
        """Row counts per tier for the Command Centre memory panel."""
        rows = self.matrix.records(limit=500)
        counts: dict[str, int] = {
            "short_term": len(self._session_turns),
            "session": len(self._session_facts),
        }
        for row in rows:
            cat = str(row.get("category") or "")
            if cat.startswith("project_"):
                counts["project"] = counts.get("project", 0) + 1
            elif cat in {"long_term", "knowledge", "workspace"}:
                counts[cat] = counts.get(cat, 0) + 1
            else:
                counts["long_term"] = counts.get("long_term", 0) + 1
        return counts

    def close(self) -> None:
        self.matrix.close()
