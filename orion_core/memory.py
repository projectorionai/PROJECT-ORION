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
from .utils import first_line, utc_stamp
from .db import apply_pragmas


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

    #: Words that carry no retrieval signal in a question about stored facts.
    #: Deliberately NOT shared with tool_resolver._STOP: that list drops command
    #: verbs ("run", "set", "show", "make") because a tool query is an
    #: instruction, whereas here those verbs are often the whole point ("what
    #: did I set the rate to"). Two lists, two jobs.
    _FTS_STOPWORDS = frozenset(
        "the a an of to in on at for and or but is are was were be been am do "
        "did does have has had what when where who whom which why how my me i "
        "you your we our it its this that these those there here about with "
        "from into over under as by if then than so any some all can could "
        "would should will shall may might much many anything".split())

    def __init__(self, db_path: Path, config_dir: Path, bus: OrionBus) -> None:
        self.db_path    = db_path
        self.config_dir = config_dir
        self.bus        = bus
        self._lock      = RLock()
        # Optional C2 sync journal, wired post-construction. When set, every
        # persistent write is recorded as an append-only change for desktop⇄cloud
        # reconciliation. None → journalling is simply off (degrades silently).
        self.journal = None
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self.conn)
        self.conn.row_factory = sqlite3.Row
        # Meaning-based recall (semantic.py): source -> (row ids, vectors),
        # rebuilt from semantic_vectors whenever _semantic_version moves.
        self._semantic_cache: dict[str, tuple[int, Any, Any]] = {}
        self._semantic_version = 0
        self._semantic_pending = False
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
            content='intelligence', content_rowid='id',
            tokenize="porter unicode61"
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
            role, content, created_at, content='episodes', content_rowid='id',
            tokenize="porter unicode61"
        );
        CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
            INSERT INTO episodes_fts(rowid, role, content, created_at)
            VALUES (new.id, new.role, new.content, new.created_at);
        END;
        CREATE TABLE IF NOT EXISTS semantic_vectors (
            source TEXT NOT NULL,
            row_id INTEGER NOT NULL,
            digest TEXT NOT NULL,
            vec BLOB NOT NULL,
            PRIMARY KEY (source, row_id)
        );
        """
        with self._lock:
            self.conn.executescript(schema)
            self.conn.commit()
        self._ensure_stemmed_index()

    #: (fts table, content table, column list) for both search indexes.
    _FTS_TABLES = (
        ("intelligence_fts", "intelligence", "category, key_ref, value, updated_at"),
        ("episodes_fts", "episodes", "role, content, created_at"),
    )

    def _ensure_stemmed_index(self) -> None:
        """Rebuild any FTS index still using the default tokenizer.

        ``CREATE VIRTUAL TABLE IF NOT EXISTS`` will not alter a table that
        already exists, so a database created before this change keeps the
        unstemmed tokenizer and cannot match "supervisor" against "supervises".
        Measured, stemming was worth 9 percentage points of recall (83% -> 92%).

        This is safe on real data: both indexes are FTS5 *external content*
        tables, so every byte lives in ``intelligence``/``episodes``. Dropping
        the index and issuing 'rebuild' regenerates it from those; the triggers
        keep their names and bind to the new table. It runs once — afterwards
        the tokenizer is already porter and the loop does nothing.

        A failure here must never stop ORION starting: an index that is merely
        unstemmed still works, it just recalls a little less.
        """
        with self._lock:
            for table, content, columns in self._FTS_TABLES:
                try:
                    row = self.conn.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()
                    if row is None or "porter" in str(row[0] or "").lower():
                        continue
                    self.conn.executescript(
                        f'DROP TABLE {table};'
                        f'CREATE VIRTUAL TABLE {table} USING fts5({columns},'
                        f" content='{content}', content_rowid='id',"
                        f' tokenize="porter unicode61");'
                    )
                    self.conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
                    self.conn.commit()
                    self.bus.log.emit(
                        f"MEMORY: rebuilt {table} with word-stemming search.")
                except Exception as exc:
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass
                    self.bus.log.emit(
                        f"MEMORY: could not upgrade {table} search index "
                        f"({first_line(exc, 90)}); recall is unaffected but "
                        "word endings will not match.")

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

    def _report_fts_fault(self, index: str, exc: Exception) -> None:
        """Say once, per index, that full-text search has stopped working.

        The knowledge graph carried a totally broken FTS query for the entire
        life of its retrieval method because the error went into a bare
        ``except: pass`` and the LIKE fallback made it look merely mediocre.
        Memory degrades the same way, so it says so — once per index per
        session, because a fault that fires on every query would otherwise
        drown the log it is trying to appear in.
        """
        seen = getattr(self, "_fts_faults", None)
        if seen is None:
            seen = set()
            self._fts_faults = seen
        if index in seen:
            return
        seen.add(index)
        try:
            self.bus.log.emit(
                f"MEMORY: {index} full-text search failed, falling back to a "
                f"slower scan - {first_line(exc, 90)}")
        except Exception:
            pass

    @classmethod
    def _like_terms(cls, raw: str) -> list[str]:
        """Meaningful terms for the LIKE fallback.

        The fallback used ``LIKE '%<the entire query>%'``, which asks for the
        whole question to appear verbatim inside one stored value — it could
        essentially never match anything a person actually asked. Matching each
        meaningful term separately is what a degraded path is for.
        """
        tokens = cls._fts_tokens(raw)
        meaningful = [t for t in tokens if t.lower() not in cls._FTS_STOPWORDS]
        return (meaningful or tokens)[:6]

    @classmethod
    def _fts_tokens(cls, raw: str) -> list[str]:
        """Query terms with FTS5 operators and column specs stripped out."""
        sanitised = cls._FTS5_STRIP_RE.sub(' ', raw)
        sanitised = cls._FTS5_COLSPEC_RE.sub(' ', sanitised)
        sanitised = cls._FTS5_COLLAPSE_RE.sub(' ', sanitised).strip()
        return [t for t in sanitised.split()
                if len(t) >= 2 and not re.fullmatch(r'[^\w]+', t)]

    @classmethod
    def _sanitise_fts_query(cls, raw: str) -> str:
        """Turn a natural question into an FTS5 query that can actually match.

        FTS5 treats space-separated terms as an implicit **AND**, so joining
        them with spaces required every word of the question to appear in one
        stored row. Measured against eight ordinary questions about eight stored
        facts, that retrieved **nothing at all** - 0/8. Memory was effectively
        write-only for anything but an exact keyword.

        Joining with OR and letting FTS5's BM25 ``ORDER BY rank`` do the work is
        the ordinary way to run a bag-of-words query: rows matching more, rarer
        terms rank higher. Measured 11/12, with the right fact ranked FIRST in
        all eleven.

        Question and function words are dropped first. They barely change which
        rows match, but they add noise to the ranking - "when is my exam"
        returned exam_date plus two unrelated rows until "is" and "my" were
        removed. If a query is *nothing but* stopwords they are kept: matching
        on a weak term beats returning nothing.
        """
        sanitised = cls._FTS5_STRIP_RE.sub(' ', raw)
        sanitised = cls._FTS5_COLSPEC_RE.sub(' ', sanitised)
        sanitised = cls._FTS5_COLLAPSE_RE.sub(' ', sanitised).strip()
        tokens = [t for t in sanitised.split()
                  if len(t) >= 2 and not re.fullmatch(r'[^\w]+', t)]
        if not tokens:
            return ''
        meaningful = [t for t in tokens if t.lower() not in cls._FTS_STOPWORDS]
        return ' OR '.join(meaningful or tokens)

    def save(
        self,
        category: str,
        key: str,
        value: str,
        updated_at: str | None = None,
        silent: bool = False,
        journal: bool = True,
    ) -> str:
        category  = self._safe_slug(category or "notes")
        key       = self._safe_slug(key or "entry")
        value     = str(value or "").strip()
        if not value:
            return "No intelligence value supplied."
        SecuritySanitiser.guard_text(value, "memory.value")
        timestamp = updated_at or utc_stamp()
        stored = value[:1000]
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO intelligence(category, key_ref, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category, key_ref)
                DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (category, key, stored, timestamp),
            )
            self.conn.commit()
        # C2: record the change for sync — but never when the write *originated*
        # from a sync-apply (journal=False), or reconciliation would loop.
        if journal and self.journal is not None:
            try:
                self.journal.record(tier=category, category=category, key=key, payload=stored)
            except Exception as exc:   # journalling must never break a memory write
                self.bus.log.emit(f"SYNC: journal record skipped - {exc}")
        if not silent:
            self.bus.log.emit(f"MEM: synchronised {category}/{key}")
        self._index_soon()
        return f"Stored intelligence: {category}/{key}."

    def query(self, query: str, limit: int = 8) -> list[dict[str, str]]:
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.query")
        limit = max(1, min(25, int(limit or 8)))
        if not query:
            return []
        rows = self._lexical_intelligence(query, limit)
        fused = self._with_meaning("intelligence", query, rows, limit)
        if fused is not None:
            return fused
        return [{k: row[k] for k in ("category", "key_ref", "value", "updated_at")}
                for row in rows]

    def _lexical_intelligence(self, query: str, limit: int) -> list[sqlite3.Row]:
        """Word matches: FTS5 BM25, else per-term LIKE. Rows carry their id."""
        sanitised_fts = self._sanitise_fts_query(query)
        with self._lock:
            if sanitised_fts:
                try:
                    rows = self.conn.execute(
                        """
                        SELECT rowid AS id, category, key_ref, value, updated_at
                        FROM intelligence_fts
                        WHERE intelligence_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (sanitised_fts, limit),
                    ).fetchall()
                    if rows:
                        return rows
                except (sqlite3.OperationalError, sqlite3.Error) as exc:
                    self._report_fts_fault("intelligence_fts", exc)
            terms = self._like_terms(query) or [query]
            clause = " OR ".join(
                "value LIKE ? OR key_ref LIKE ? OR category LIKE ?" for _ in terms)
            params: list[Any] = []
            for term in terms:
                params.extend([f"%{term}%"] * 3)
            params.append(limit)
            return self.conn.execute(
                f"""
                SELECT id, category, key_ref, value, updated_at
                FROM intelligence
                WHERE {clause}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

    # ── meaning-based recall (semantic.py) ───────────────────────────────────
    #
    # Words alone missed questions phrased differently from what was stored:
    # on ORION's memory benchmark with 106 look-alike distractors, held-out
    # top-1 was 82.4% on words, 100% with meaning fused in. Everything below
    # is inert until semantic.ENCODER is warm, so without the model memory
    # behaves exactly as it did.

    #: (source table, id column, text column, rows to index at most — newest).
    _SEMANTIC_SOURCES = {
        "intelligence": ("id", "value", 50_000),
        "episodes": ("id", "content", 25_000),
    }

    def _with_meaning(self, source: str, query: str, lexical: list[Any],
                      limit: int) -> list[dict[str, str]] | None:
        """Fuse *lexical* rows with meaning matches; None when unavailable."""
        from . import semantic

        if not semantic.ENCODER.ready():
            return None
        try:
            question = semantic.ENCODER.encode_one(query)
            ids, matrix = self._semantic_vectors(source)
        except Exception:
            return None
        if question is None or matrix is None or not len(ids):
            return None
        import numpy as np

        sims = matrix @ question
        top = np.argsort(-sims)[:max(limit, 8)]
        position = {int(row_id): index for index, row_id in enumerate(ids)}
        lexical_ids = [int(row["id"]) for row in lexical]
        similarity = {row_id: float(sims[position[row_id]])
                      for row_id in lexical_ids if row_id in position}
        order = semantic.fuse(lexical_ids,
                              [(int(ids[i]), float(sims[i])) for i in top],
                              similarity=similarity)[:limit]
        if not order:
            return []
        id_column, _text, _cap = self._SEMANTIC_SOURCES[source]
        columns = ("category, key_ref, value, updated_at" if source == "intelligence"
                   else "role, content, created_at")
        marks = ",".join("?" * len(order))
        with self._lock:
            found = {int(row[id_column]): row for row in self.conn.execute(
                f"SELECT {id_column}, {columns} FROM {source} WHERE {id_column} IN ({marks})",
                order).fetchall()}
        keys = [c.strip() for c in columns.split(",")]
        return [{k: found[i][k] for k in keys} for i in order if i in found]

    #: For memories volunteered unasked (relevant()). Measured on 12 requests
    #: that should bring a specific fact and 12 everyday ones that should
    #: bring none, among 122 stored facts:
    #:     0.58  11/12 relevant, 6/12 everyday requests padded
    #:     0.60  10/12 relevant, 1/12 padded   <- chosen
    #:     0.62   8/12 relevant, 0/12 padded
    #: (A z-score against all memories separated them worse at every cut.)
    RELEVANT_FLOOR = 0.60
    #: Health and safety facts clear a lower bar, because the costs are not
    #: symmetric: "I've got a headache, what should I take" scored 0.589
    #: against "allergic to ibuprofen" — missing that is worse than an
    #: unneeded mention.
    SAFETY_FLOOR = 0.55
    _SAFETY_RE = re.compile(
        r"allerg|intoleran|anaphyla|epipen|medicat|prescri|dosage|asthma|"
        r"epilep|seizure|diabet|insulin|pregnan|blood thinner|warfarin|"
        r"do not take|must not|mustn't|never take|coeliac|celiac", re.IGNORECASE)
    #: ...and only when the REQUEST is about health, food or medicine; on its
    #: own the lower bar let "Tom is allergic to cats" ride along with "tell
    #: me a joke".
    _HEALTH_REQUEST_RE = re.compile(
        r"\b(?:ache|aches|headache|migraine|pain|painkiller|hurt|sore|ill|sick|"
        r"fever|flu|cold|cough|symptom|doctor|gp|pharmac|chemist|medicine|"
        r"medic|tablet|pill|dose|take|taking|eat|eating|food|meal|recipe|cook|"
        r"restaurant|takeaway|drink|snack|allerg\w*|diet|vaccin\w*|jab|injection)\b",
        re.IGNORECASE)

    def relevant(self, text: str, limit: int = 3,
                 floor: float | None = None) -> list[dict[str, Any]]:
        """Stored facts that bear on *text* by MEANING, strongest first.

        For volunteering context before a reply — not for answering a recall
        question (query() does that). Empty unless the semantic encoder is
        warm, so it can never pad a prompt with word-match coincidences.
        """
        from . import semantic

        if not semantic.ENCODER.ready() or not str(text or "").strip():
            return []
        floor = self.RELEVANT_FLOOR if floor is None else floor
        try:
            question = semantic.ENCODER.encode_one(str(text)[:1000])
            ids, matrix = self._semantic_vectors("intelligence")
        except Exception:
            return []
        if question is None or matrix is None or not len(ids):
            return []
        import numpy as np

        sims = matrix @ question
        health = bool(self._HEALTH_REQUEST_RE.search(str(text)))
        lowest = min(floor, self.SAFETY_FLOOR) if health else floor
        candidates = [int(i) for i in np.argsort(-sims)[:limit * 4] if sims[int(i)] >= lowest]
        if not candidates:
            return []
        wanted = [int(ids[i]) for i in candidates]
        marks = ",".join("?" * len(wanted))
        with self._lock:
            found = {int(r["id"]): r for r in self.conn.execute(
                f"SELECT id, category, key_ref, value FROM intelligence WHERE id IN ({marks})",
                wanted).fetchall()}
        out: list[dict[str, Any]] = []
        for row_id, index in zip(wanted, candidates):
            row = found.get(row_id)
            if row is None:
                continue
            bar = (self.SAFETY_FLOOR if health and self._SAFETY_RE.search(str(row["value"]))
                   else floor)
            if sims[index] < bar:
                continue
            out.append({"category": row["category"], "key_ref": row["key_ref"],
                        "value": row["value"], "similarity": round(float(sims[index]), 3)})
            if len(out) >= limit:
                break
        return out

    def _semantic_vectors(self, source: str) -> tuple[Any, Any]:
        """(row ids, unit-vector matrix) for *source*, cached per version."""
        cached = self._semantic_cache.get(source)
        if cached is not None and cached[0] == self._semantic_version:
            return cached[1], cached[2]
        from . import semantic
        import numpy as np

        with self._lock:
            rows = self.conn.execute(
                "SELECT row_id, vec FROM semantic_vectors WHERE source = ?", (source,)
            ).fetchall()
        ids, vectors = [], []
        for row in rows:
            vector = semantic.from_blob(row["vec"])
            if vector is not None:
                ids.append(int(row["row_id"]))
                vectors.append(vector)
        matrix = np.vstack(vectors).astype(np.float32) if vectors else None
        id_array = np.asarray(ids, dtype=np.int64)
        self._semantic_cache[source] = (self._semantic_version, id_array, matrix)
        return id_array, matrix

    def index_semantics(self, batch: int = 64, max_rows: int | None = None) -> int:
        """Embed every row not yet embedded (or edited since). Blocking.

        Run from a worker thread: the encoding happens OUTSIDE the lock, so
        recall keeps answering (on words) while this works. Returns how many
        rows were embedded. Vectors of deleted rows are dropped too.
        """
        from . import semantic

        if not semantic.ENCODER.ready():
            return 0
        done = 0
        for source, (id_column, text_column, cap) in self._SEMANTIC_SOURCES.items():
            cap = cap if max_rows is None else min(cap, max_rows)
            with self._lock:
                self.conn.execute(
                    f"DELETE FROM semantic_vectors WHERE source = ? AND row_id NOT IN "
                    f"(SELECT {id_column} FROM {source})", (source,))
                self.conn.commit()
                rows = self.conn.execute(
                    f"""
                    SELECT s.{id_column} AS id, s.{text_column} AS text, v.digest
                    FROM (SELECT {id_column}, {text_column} FROM {source}
                          ORDER BY {id_column} DESC LIMIT ?) AS s
                    LEFT JOIN semantic_vectors AS v
                      ON v.source = ? AND v.row_id = s.{id_column}
                    """,
                    (cap, source),
                ).fetchall()
            todo = [(int(r["id"]), str(r["text"] or ""))
                    for r in rows if r["digest"] != semantic.text_digest(str(r["text"] or ""))]
            # Shortest first across the whole backlog, so every batch holds
            # texts of a similar length (see SemanticEncoder.encode).
            todo.sort(key=lambda item: len(item[1]))
            for start in range(0, len(todo), batch):
                chunk = todo[start:start + batch]
                vectors = semantic.ENCODER.encode([text for _id, text in chunk])
                if vectors is None:
                    return done
                with self._lock:
                    self.conn.executemany(
                        "INSERT OR REPLACE INTO semantic_vectors(source, row_id, digest, vec) "
                        "VALUES (?, ?, ?, ?)",
                        [(source, row_id, semantic.text_digest(text), semantic.to_blob(vec))
                         for (row_id, text), vec in zip(chunk, vectors)])
                    self.conn.commit()
                    self._semantic_version += 1
                done += len(chunk)
        return done

    def _index_soon(self) -> None:
        """Embed a new or changed row shortly, off the caller's thread."""
        from . import semantic

        # Dirty flag first: a save landing while a pass is already running
        # makes that pass go round again rather than being missed.
        self._semantic_dirty = True
        if self._semantic_pending or not semantic.ENCODER.ready():
            return
        self._semantic_pending = True

        def _run() -> None:
            semantic.background_priority()
            try:
                while getattr(self, "_semantic_dirty", False):
                    self._semantic_dirty = False
                    self.index_semantics()
            except Exception:
                pass            # recall still works on words
            finally:
                self._semantic_pending = False

        import threading
        threading.Thread(target=_run, name="orion-memory-embed", daemon=True).start()

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
            return  # conversation logging must never break a live turn
        self._index_soon()

    def recall_episodes(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.episodes")
        limit = max(1, min(50, int(limit or 10)))
        if not query:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT role, content, created_at FROM episodes ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
        rows = self._lexical_episodes(query, limit)
        fused = self._with_meaning("episodes", query, rows, limit)
        if fused is not None:
            return fused
        return [{k: row[k] for k in ("role", "content", "created_at")} for row in rows]

    def _lexical_episodes(self, query: str, limit: int) -> list[sqlite3.Row]:
        with self._lock:
            sanitised = self._sanitise_fts_query(query)
            if sanitised:
                try:
                    rows = self.conn.execute(
                        """
                        SELECT rowid AS id, role, content, created_at FROM episodes_fts
                        WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?
                        """,
                        (sanitised, limit),
                    ).fetchall()
                    if rows:
                        return rows
                except (sqlite3.OperationalError, sqlite3.Error) as exc:
                    self._report_fts_fault("episodes_fts", exc)
            terms = self._like_terms(query) or [query]
            clause = " OR ".join("content LIKE ?" for _ in terms)
            params: list[Any] = [f"%{t}%" for t in terms]
            params.append(limit)
            return self.conn.execute(
                f"""
                SELECT id, role, content, created_at FROM episodes
                WHERE {clause} ORDER BY id DESC LIMIT ?
                """,
                params,
            ).fetchall()

    #: How many omitted keys to name. Keys are a few words each, so this is
    #: cheap compared with values — and it is what turns "I do not know"
    #: into a lookup.
    #: How many passes the user-facing categories get before the bulk ones
    #: are considered at all.
    _CORE_ROUNDS = 2

    INDEX_KEYS = 40

    #: Categories that describe the USER and ORION rather than the world. They
    #: lead the prompt because losing them is what makes an assistant feel like
    #: it has amnesia, no matter how much else it is carrying.
    _CORE_CATEGORIES = ("identity", "personal", "relationship", "long_term",
                        "preferences", "projects", "notes")

    def prompt_context(self, limit: int = 18) -> str:
        """The memory core for the system prompt, plus an index of the rest.

        Pure recency ordering lets frequently updated bulk knowledge crowd out
        identity, relationships, projects and notes. Category weighting keeps
        these less frequently updated facts available to the assistant.

        So categories are INTERLEAVED: every category contributes one row
        before any category contributes a second, with the categories that
        describe the user taking the first pass. A thousand rows of one kind
        can then no longer starve the one row of another.

        The second half matters just as much. A model cannot look something up
        if it does not know the thing exists, so what did not fit is listed by
        KEY — cheap, a few words each — and the model is told it can fetch any
        of them with query_intelligence. Without that index, "who is Sample Relative?"
        gets "I do not know" while the answer sits on disk unread.
        """
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT category, key_ref, value, updated_at
                FROM intelligence
                ORDER BY updated_at DESC
                """
            ).fetchall()
        if not rows:
            return ""

        # Group by category, each still newest-first.
        grouped: dict[str, list] = {}
        for row in rows:
            grouped.setdefault(str(row["category"]), []).append(row)

        ordered = ([c for c in self._CORE_CATEGORIES if c in grouped]
                   + sorted(c for c in grouped if c not in self._CORE_CATEGORIES))

        core_present = [c for c in self._CORE_CATEGORIES if c in grouped]

        chosen: list = []
        omitted: list[tuple[str, str]] = []
        depth = 0
        # Round-robin, but not a flat one. A plain rotation gives `identity` and
        # `pack_tiktok_shop` a slot each, which is equal treatment of things
        # that are not equal: on the measured store the twelve bulk knowledge
        # packs then consumed two thirds of the budget and the user's own facts
        # got one row apiece. So the categories describing the user go round
        # twice before anything else goes round once.
        while len(chosen) < limit:
            pool = core_present if depth < self._CORE_ROUNDS else ordered
            advanced = False
            for category in pool:
                if len(chosen) >= limit:
                    break
                bucket = grouped[category]
                if len(bucket) > depth:
                    chosen.append(bucket[depth])
                    advanced = True
            depth += 1
            if not advanced and depth > self._CORE_ROUNDS:
                break               # every bucket exhausted

        taken = {(str(r["category"]), str(r["key_ref"])) for r in chosen}
        for category in ordered:
            for row in grouped[category]:
                key = (str(row["category"]), str(row["key_ref"]))
                if key not in taken:
                    omitted.append(key)

        # The header says whose facts these are. "identity/jordan waters
        # location" read, to a model, as the user's own identity — and he
        # greeted the user by a friend's name. A named person here is someone
        # else unless the note plainly says it is the user's own name.
        lines = ["[LOCAL INTELLIGENCE MATRIX - use naturally, never recite. "
                 "'identity' and 'personal' notes describe the user; any other "
                 "person named below is someone the user knows, never the user.]"]
        for row in chosen:
            label = f"{row['category']}/{row['key_ref']}".replace("_", " ")
            lines.append(f"- {label}: {row['value']}")

        if omitted:
            # Interleave the index too, for exactly the same reason: sorted by
            # category, a store with four hundred knowledge keys would push the
            # one relationship key off the end of the very list that exists to
            # surface it.
            by_category: dict[str, list[str]] = {}
            for category, key in omitted:
                by_category.setdefault(category, []).append(key)
            index: list[str] = []
            depth = 0
            order = list(by_category)
            while len(index) < self.INDEX_KEYS and any(
                    len(by_category[c]) > depth for c in order):
                for category in order:
                    if len(index) >= self.INDEX_KEYS:
                        break
                    bucket = by_category[category]
                    if len(bucket) > depth:
                        index.append(f"{category}/{bucket[depth]}".replace("_", " "))
                depth += 1
            more = len(omitted) - len(index)
            tail = f" (+{more} more)" if more > 0 else ""
            lines.append(
                "[ALSO REMEMBERED - you hold these but not their contents; "
                "fetch any of them with query_intelligence rather than saying "
                f"you do not know] {', '.join(index)}{tail}")
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
        # Learns lasting facts from what the user says (fact_memory.py); set
        # by app.py once a model router exists. None means it is off.
        self.fact_harvester: Any = None

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
        harvester = self.fact_harvester
        if harvester is not None and str(role or "").lower() == "user":
            try:
                harvester.observe(content)      # a regex gate; never blocks
            except Exception:
                pass

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

    def relevant(self, text: str, limit: int = 3) -> list[dict[str, Any]]:
        """Stored facts that bear on *text* by meaning (see the matrix's)."""
        return self.matrix.relevant(text, limit=limit)

    def relevant_note(self, text: str, limit: int = 3) -> str:
        """The facts above as a note to put beside a request, or "".

        Worded as background the model may use, not as an instruction, so a
        fact that turns out not to matter is simply ignored rather than
        dragged into the reply.
        """
        rows = self.relevant(text, limit=limit)
        if not rows:
            return ""
        facts = "; ".join(str(r["value"]).strip().rstrip(".") for r in rows)
        return (f"[Background from ORION's memory — things already known that may "
                f"bear on this; use only what is relevant, do not recite: {facts}.]")

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
