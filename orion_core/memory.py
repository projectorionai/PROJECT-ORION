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
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

from .bus import OrionBus
from .constants import BASE_DIR
from .security import SecuritySanitiser
from .utils import first_line, utc_stamp
from .db import apply_pragmas


def _top_positions(scores: Any, k: int, pool: Any = None) -> Any:
    """Positions of the *k* highest *scores*, best first, ties by position.

    The same answer as ``np.argsort(-scores, kind="stable")[:k]`` without
    sorting everything: ``np.partition`` finds the k-th value in linear time
    and only the k survivors are sorted. Measured at 50k vectors, the full
    sort plus the per-question id map was 8.6 ms of a 15 ms question; this
    selection is 0.36 ms (tools/bench_retrieval.py, "ranking").

    *pool*, if given, restricts the choice to those positions (ascending, as
    ``np.flatnonzero`` returns them) and the result is in the same space.
    """
    import numpy as np

    if pool is not None:
        return pool[_top_positions(scores[pool], k)]
    n = int(scores.shape[0])
    k = max(0, min(int(k), n))
    if k == 0:
        return np.zeros(0, dtype=np.intp)
    if k == n:
        return np.argsort(-scores, kind="stable")
    kth = np.partition(scores, n - k)[n - k]
    above = np.flatnonzero(scores > kth)
    ties = np.flatnonzero(scores == kth)[:k - above.shape[0]]
    chosen = np.concatenate([above, ties])
    return chosen[np.argsort(-scores[chosen], kind="stable")]


def defragment(text: str, words: Any = None) -> str:
    """*text* with words that a transcript split apart joined back up.

    Until the Live transcript join was fixed (5 Oct 2026) every spoken turn
    was stored in chunks: "I go in to uni versity in neu ro eng ine ering",
    "my bud get ... is 3 , 5 7 3 pou nd s". 700 of the owner's 2,002 stored
    turns read like that, and recall reads them: the word index sees "uni"
    and "versity", the sentence encoder half a sentence. A run of 2-5 pieces
    is joined when the result is a whole English word the encoder knows and
    at least one piece is NOT a word ("uni versity" -> "university", but "a
    cross" and "to day" stay as they are: both halves are words, so joining
    could change what was said). Runs of single digits are joined too
    ("1 2 0" -> "120"). Only ever applied to what is READ - the stored turn
    is never rewritten. Without the dictionary (no encoder installed) the
    text comes back unchanged.
    """
    text = str(text or "")
    if " " not in text:
        return text
    if words is None:
        from .spelling import english_words
        words = english_words()
    if not words:
        return text
    parts = text.split(" ")
    out: list[str] = []
    i = 0
    while i < len(parts):
        piece = parts[i]
        if len(piece) == 1 and piece.isdigit():
            j = i
            while j < len(parts) and len(parts[j]) == 1 and parts[j].isdigit():
                j += 1
            if j - i >= 2:
                out.append("".join(parts[i:j]))
                i = j
                continue
        merged = None
        if piece.isalpha():
            for j in range(min(len(parts), i + 5), i + 1, -1):
                run = parts[i:j]
                if not all(p.isalpha() for p in run):
                    continue
                joined = "".join(run).lower()
                if len(joined) < 4 or joined not in words:
                    continue
                if any(p.lower() not in words for p in run):
                    if all(p.isupper() for p in run):
                        joined = joined.upper()
                    elif run[0][:1].isupper():
                        joined = joined.capitalize()
                    merged = (joined, j)
                    break
        if merged is not None:
            out.append(merged[0])
            i = merged[1]
        else:
            out.append(piece)
            i += 1
    return " ".join(out)


class _VectorSnapshot:
    """One source's vectors as they stood at *generation*. Never mutated.

    Held by reference and replaced whole, so a question that picked this one
    up keeps a consistent view while a newer one is built beside it. ``ids``
    is sorted, which makes it the id -> position map too: a searchsorted on
    the few ids a question needs, instead of a dict over every vector rebuilt
    on every question. ``digests`` say which text each vector was computed
    from, so a vector for words that have since changed is recognisable.
    """

    __slots__ = ("generation", "ids", "matrix", "digests")

    def __init__(self, generation: int, ids: Any, matrix: Any, digests: Any) -> None:
        for array in (ids, matrix, digests):
            array.flags.writeable = False
        self.generation = generation
        self.ids = ids
        self.matrix = matrix
        self.digests = digests

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def positions(self, row_ids: Sequence[int]) -> Any:
        """Where each id's vector is, or -1 where this snapshot has none."""
        import numpy as np

        wanted = np.asarray(list(row_ids), dtype=np.int64)
        if not len(self) or not wanted.size:
            return np.full(wanted.shape, -1, dtype=np.intp)
        at = np.minimum(np.searchsorted(self.ids, wanted), len(self) - 1)
        return np.where(self.ids[at] == wanted, at, -1)


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
    #: FTS5's own keywords: never passed on as terms.
    _FTS5_OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})
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
        # Meaning-based recall (semantic.py). Each source's vectors are one
        # immutable _VectorSnapshot, built off the question's path and
        # published by swapping a single reference. _vector_generation counts
        # committed changes to semantic_vectors (bumped under _lock, after the
        # commit); a snapshot records the count it was read at, so one that
        # is behind is rebuilt rather than mistaken for current.
        #
        # This replaced a cache rebuilt INLINE by whichever question first
        # noticed a change: at 50k rows that question paid 334 ms, and during
        # a backfill every indexed batch cost the next question a rebuild
        # (p95 419 ms). relevant_note gives up after 500 ms.
        self._snapshots: dict[str, _VectorSnapshot] = {}
        # Typing slips in a question put right towards the words of what is
        # remembered ("where does my fathdr live", "hannha's birthday"):
        # measured, one or two slips took recall top-1 from 90.9% to 47.0%.
        from .spelling import CachedSpeller
        self._speller = CachedSpeller("memory", self._fact_words, self._fact_words_key,
                                     english_candidates=True)
        self._vector_generation: dict[str, int] = {}
        # What changed since the published snapshot, so a refresh can apply it
        # (_bump_vectors); the floor is the first generation the log cannot
        # reproduce. Both guarded by _lock.
        self._vector_log: dict[str, list[tuple[int, tuple, tuple]]] = {}
        self._vector_log_floor: dict[str, int] = {}
        # Rows written since the last pass, embedded by id instead of scanning
        # the whole table for them (guarded by _state).
        self._dirty_rows: dict[str, set[int]] = {}
        self._full_pass_wanted = False
        self._snapshot_build_lock = threading.Lock()
        # The connection big background reads use (_read_aside), opened once:
        # a fresh one paid ~10 ms parsing the schema before its first row.
        self._reader: sqlite3.Connection | None = None
        self._reader_lock = threading.Lock()
        # One embedding pass at a time: two overlapping passes could each read
        # a row, and the slower one write a vector for text already replaced.
        self._index_lock = threading.Lock()
        # Guards the indexer/refresher flags below and wakes wait_semantic_idle.
        self._state = threading.Condition()
        self._semantic_pending = False
        self._semantic_dirty = False
        self._refreshing: set[str] = set()
        self._closed = False
        self._initialise()
        self._migrate_legacy()

    def close(self) -> None:
        with self._state:
            self._closed = True
            self._state.notify_all()
        with self._reader_lock:
            if self._reader is not None:
                self._reader.close()
                self._reader = None
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
        CREATE INDEX IF NOT EXISTS idx_intelligence_prompt
            ON intelligence(category, updated_at DESC, id ASC);
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
        CREATE TABLE IF NOT EXISTS scene_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL, epoch REAL NOT NULL,
            source TEXT NOT NULL, location TEXT NOT NULL,
            summary TEXT NOT NULL, searchable TEXT NOT NULL,
            record TEXT NOT NULL, status TEXT NOT NULL, vector BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_scene_time ON scene_memories(epoch DESC);
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
            self.conn.execute("DELETE FROM scene_memories WHERE epoch < ?", (time.time()-30*86400,))
            self.conn.commit()
        self._ensure_stemmed_index()

    #: (fts table, content table, column list) for both search indexes.
    _FTS_TABLES = (
        ("intelligence_fts", "intelligence", "category, key_ref, value, updated_at"),
        ("episodes_fts", "episodes", "role, content, created_at"),
    )

    def save_scene(self, *, captured_at: str, epoch: float, source: str = "camera",
                   location: str = "", summary: str = "", searchable: str = "",
                   record: str = "{}", status: str = "", vector: bytes | None = None) -> int:
        """Keep bounded scene descriptions; pictures never enter this table."""
        import math
        if not isinstance(record, str) or len(record) > 60000:
            raise ValueError("scene memory accepts bounded descriptions, not images")
        data = json.loads(record)
        if not isinstance(data, dict) or not math.isfinite(float(epoch)):
            raise ValueError("a scene requires a text record and a finite capture time")
        def text_only(value):
            if isinstance(value, dict):
                return all(str(k).lower() not in {
                    "image", "frame", "pixels", "jpeg", "png", "image_data", "base64"
                } and text_only(v) for k, v in value.items())
            if isinstance(value, list):
                return all(text_only(v) for v in value)
            if isinstance(value, str):
                return not value.lower().startswith("data:image/")
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                return math.isfinite(value)
            return value is None or isinstance(value, bool)
        if not text_only(data) or len(record) > 60000:
            raise ValueError("scene memory accepts bounded descriptions, not images")
        if vector is not None and (not isinstance(vector, bytes) or len(vector) > 16384):
            raise ValueError("invalid scene embedding")
        with self._lock:
            cursor = self.conn.execute(
                "INSERT INTO scene_memories(captured_at,epoch,source,location,summary,"
                "searchable,record,status,vector) VALUES(?,?,?,?,?,?,?,?,?)",
                (str(captured_at)[:80], float(epoch), str(source)[:100], str(location)[:300],
                 str(summary)[:2000], str(searchable)[:16000], record, str(status)[:40], vector))
            row_id = int(cursor.lastrowid)
            self.conn.execute("DELETE FROM scene_memories WHERE epoch < ?", (time.time()-30*86400,))
            self.conn.execute("DELETE FROM scene_memories WHERE id NOT IN "
                              "(SELECT id FROM scene_memories ORDER BY epoch DESC,id DESC LIMIT 2000)")
            self.conn.commit()
        return row_id

    def scene_memories(self, query: str = "", *, since: float | None = None,
                       until: float | None = None, limit: int = 8, source: str | None = None,
                       oldest_first: bool = False) -> list[dict[str, Any]]:
        """Recall descriptions by time, words, and an already warm encoder."""
        clauses, params = ["epoch >= ?"], [time.time()-30*86400]
        for bound, op in ((since, ">="), (until, "<=")):
            if bound is not None:
                clauses.append(f"epoch {op} ?")
                params.append(float(bound))
        if source is not None:
            clauses.append("source = ?")
            params.append(str(source))
        terms = self._like_terms(query)
        count = max(1, min(100, int(limit)))
        where = " WHERE " + " AND ".join(clauses)
        with self._lock:
            expired = self.conn.execute("DELETE FROM scene_memories WHERE epoch < ?",
                                        (time.time()-30*86400,))
            if expired.rowcount:
                self.conn.commit()
            if not terms:
                order = "ASC" if oldest_first else "DESC"
                return [dict(r) for r in self.conn.execute(
                    "SELECT id,captured_at,epoch,source,location,summary,searchable,record,status "
                    "FROM scene_memories" + where + f" ORDER BY epoch {order},id {order} LIMIT ?",
                    [*params, count]).fetchall()]
            ranking = " + ".join("(instr(lower(searchable || ' ' || location), ?) > 0)" for _ in terms)
            rows = [dict(r) for r in self.conn.execute(
                "SELECT id,epoch,vector,(" + ranking + ") AS lexical FROM scene_memories"
                + where + " ORDER BY epoch DESC,id DESC LIMIT 2000", [*terms, *params]).fetchall()]
        if terms:
            scores = {row['id']: float(row['lexical']) for row in rows if row['lexical']}
            try:
                from . import semantic
                if semantic.ENCODER.ready():
                    encoded = semantic.ENCODER.encode([str(query)[:2000]])
                    if encoded is not None and len(encoded):
                        import numpy as np
                        wanted = encoded[0]
                        norm = float(np.linalg.norm(wanted))
                        for row in rows:
                            if row['vector'] and norm:
                                vec = np.frombuffer(row['vector'], dtype=np.float32)
                                if vec.shape == wanted.shape:
                                    denom = float(np.linalg.norm(vec))*norm
                                    similarity = float(np.dot(vec, wanted)/denom) if denom else 0.0
                                    if similarity >= 0.45:
                                        scores[row['id']] = scores.get(row['id'], 0.0)+similarity
            except Exception:
                pass
            rows = [r for r in rows if r['id'] in scores]
            rows.sort(key=lambda r: (scores[r['id']], r['epoch'], r['id']), reverse=True)
        ids = [row['id'] for row in rows[:count]]
        if not ids:
            return []
        with self._lock:
            chosen = {row['id']: dict(row) for row in self.conn.execute(
                "SELECT id,captured_at,epoch,source,location,summary,searchable,record,status "
                "FROM scene_memories WHERE id IN (" + ','.join('?' for _ in ids) + ')', ids)}
        return [chosen[i] for i in ids if i in chosen]

    def forget_scenes(self, *, since: float | None = None, until: float | None = None) -> int:
        clauses, params = [], []
        for bound, op in ((since, ">="), (until, "<=")):
            if bound is not None:
                clauses.append(f"epoch {op} ?")
                params.append(float(bound))
        with self._lock:
            cursor = self.conn.execute("DELETE FROM scene_memories"
                                       + (" WHERE " + " AND ".join(clauses) if clauses else ""), params)
            self.conn.commit()
            return cursor.rowcount

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
        # Only runs of word characters reach FTS5, as barewords. Punctuation
        # used to survive inside a token ("hello," "2,5" "node.js,") and FTS5
        # rejects a bare comma: "fts5: syntax error near ','" on the owner's
        # ordinary questions (26 Oct 2026), every one falling back to a scan.
        # Operator words are dropped in any case; the index folds case anyway.
        tokens = [t.lower() for t in re.findall(r'\w+', str(raw or ''))
                  if len(t) >= 2 and t.upper() not in cls._FTS5_OPERATORS]
        if not tokens:
            return ''
        tokens = list(dict.fromkeys(tokens))
        meaningful = [t for t in tokens if t not in cls._FTS_STOPWORDS]
        return ' OR '.join(meaningful or tokens)

    def category_counts(self) -> dict[str, int]:
        """How many recallable memories each category holds (the Memory
        page's constellation) - the same rows recall can answer from."""
        recallable = self._recallable("intelligence")
        with self._lock:
            return {str(row["category"]): int(row["n"]) for row in self.conn.execute(
                "SELECT category, COUNT(*) AS n FROM intelligence "
                f"WHERE {recallable} GROUP BY category")}

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
            saved = self.conn.execute(
                "SELECT id FROM intelligence WHERE category = ? AND key_ref = ?",
                (category, key)).fetchone()
        # C2: record the change for sync — but never when the write *originated*
        # from a sync-apply (journal=False), or reconciliation would loop.
        if journal and self.journal is not None:
            try:
                self.journal.record(tier=category, category=category, key=key, payload=stored)
            except Exception as exc:   # journalling must never break a memory write
                self.bus.log.emit(f"SYNC: journal record skipped - {exc}")
        if not silent and not self._is_record_key(key):
            # A record that recall never returns (an event mirror, a request
            # pattern, ORION's bookkeeping) is not announced as a memory: the
            # activity feed turns this line into "Noted something about your
            # graph event 5983116a..." - which the owner read out on 9 Oct as
            # proof ORION had noted something it then could not recall.
            self.bus.log.emit(f"MEM: synchronised {category}/{key}")
        self._index_soon("intelligence", int(saved[0]) if saved else None)
        return f"Stored intelligence: {category}/{key}."

    def query(self, query: str, limit: int = 8) -> list[dict[str, str]]:
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.query")
        limit = max(1, min(25, int(limit or 8)))
        if not query:
            return []
        query = self._speller.correct(query)
        rows = self._lexical_intelligence(query, limit)
        fused = self._with_meaning("intelligence", query, rows, limit)
        if fused is not None:
            return fused
        return [{k: row[k] for k in ("category", "key_ref", "value", "updated_at")}
                for row in rows]

    def _fact_words(self) -> tuple[dict[str, int], dict[str, int]]:
        """Every word of what recall can return, counted (for spelling.py)."""
        from collections import Counter
        words: Counter[str] = Counter()
        for row in self._read_aside(f"SELECT key_ref, value FROM intelligence "
                                    f"WHERE {self._recallable('intelligence')}"):
            words.update(re.findall(r"[a-z]+", f"{str(row[0]).replace('_', ' ')} {row[1]}".lower()))
        return words, {}

    def _fact_words_key(self) -> tuple[Any, ...]:
        """Changes whenever a recallable fact is added, edited or removed."""
        with self._lock:
            return tuple(self.conn.execute(
                f"SELECT count(*), max(updated_at), max(id) FROM intelligence "
                f"WHERE {self._recallable('intelligence')}").fetchone())

    def _lexical_intelligence(self, query: str, limit: int, *, category: str = "") -> list[sqlite3.Row]:
        """Word matches: FTS5 BM25, else per-term LIKE. Rows carry their id."""
        sanitised_fts = self._sanitise_fts_query(query)
        scope = " AND category = ?" if category else ""
        scope_values = (category,) if category else ()
        with self._lock:
            if sanitised_fts:
                try:
                    rows = self.conn.execute(
                        f"""
                        SELECT rowid AS id, category, key_ref, value, updated_at
                        FROM intelligence_fts
                        WHERE intelligence_fts MATCH ? AND {self._recallable("intelligence")}{scope}
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (sanitised_fts, *scope_values, limit),
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
            params.extend(scope_values)
            params.append(limit)
            return self.conn.execute(
                f"""
                SELECT id, category, key_ref, value, updated_at
                FROM intelligence
                WHERE ({clause}) AND {self._recallable("intelligence")}{scope}
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
    #: Rows that are RECORDS, not facts, and so are never recall answers:
    #: the knowledge graph mirrors every event into memory as
    #: "conversation: <title> (3 linked entities)", and the intent tracker
    #: files "Repeated request pattern: ..." rows. On the live store they
    #: were ~1,200 of 1,794 rows; on a benchmark shaped like it (128 facts
    #: among event and pattern rows on the same topics) they took dev top-1
    #: from 80% to 55% by outranking the facts. Both have their own readers
    #: (the knowledge graph's index, the intent tracker's state), so they
    #: stay stored - they are just neither ranked nor embedded here.
    #: 17 Oct 2026, the same for two of ORION's own bookkeeping rows, measured
    #: on a copy of the owner's store: notes/unresolved_query is the model's
    #: last FAILED lookup ("chess learning"), filed by the dispatcher, and it
    #: was the answer offered to "what language am I learning"; long_term/
    #: last_cognitive_resume is a status line rewritten every launch, and it
    #: held one of the sixteen fact slots in every system prompt. Neither is
    #: anything the user said.
    _NOT_RECALLED = {
        "intelligence": "(substr(key_ref, 1, 12) = 'graph_event_' OR substr(key_ref, 1, 7) = 'intent_'"
                        " OR key_ref IN ('unresolved_query', 'last_cognitive_resume'))",
    }

    #: What is EMBEDDED for a row, as SQL. A fact is stored with a label the
    #: model chose ("salary", "wake_up_time", "move_out_date") and the label
    #: is the strongest single clue to what the fact answers - yet only the
    #: sentence was embedded, so "what's my salary" lost to "gets paid on the
    #: last working day" by meaning. Embedding "salary: earns 31,000 a year"
    #: took held-out top-1 from 80.0% to 90.8% (hit@3 87.7 -> 95.4) on 128
    #: facts among labelled look-alikes and record noise. _embed_text is the
    #: same thing in Python; the two must agree, or every vector reads stale.
    _EMBED_TEXT_SQL = {
        "intelligence": "replace(key_ref, '_', ' ') || ': ' || value",
        "episodes": "content",
    }

    @classmethod
    def _embed_text(cls, source: str, row: Any) -> str:
        if source == "intelligence":
            return f"{str(row['key_ref']).replace('_', ' ')}: {row['value']}"
        return cls._prepare(source, str(row[cls._SEMANTIC_SOURCES[source][1]] or ""))

    #: Conversation turns are embedded with split words joined (defragment):
    #: the encoder read "uni versity" and "neu ro eng ine ering" as noise.
    #: The digest is of the joined text, so on the first pass after this
    #: change only the ~700 fragmented turns are embedded again (once, in the
    #: background indexer); clean turns join to themselves and keep theirs.
    DEFRAGMENT_EPISODES = True

    @classmethod
    def _prepare(cls, source: str, text: str) -> str:
        """The text that is embedded for *text* as the SQL selects it."""
        if source == "episodes" and cls.DEFRAGMENT_EPISODES:
            return defragment(text)
        return text

    #: _NOT_RECALLED["intelligence"] in Python, for one key (the two agree).
    _RECORD_PREFIXES = ("graph_event_", "intent_")
    _RECORD_KEYS = frozenset({"unresolved_query", "last_cognitive_resume"})

    @classmethod
    def _is_record_key(cls, key: str) -> bool:
        key = str(key or "")
        return key.startswith(cls._RECORD_PREFIXES) or key in cls._RECORD_KEYS

    @classmethod
    def _recallable(cls, source: str) -> str:
        """SQL condition for the rows of *source* that recall may return."""
        excluded = cls._NOT_RECALLED.get(source)
        return f"NOT {excluded}" if excluded else "1"

    #: What a recall answer carries, per source.
    _SOURCE_COLUMNS = {
        "intelligence": "category, key_ref, value, updated_at",
        "episodes": "role, content, created_at",
    }
    #: Sources whose snapshot the indexer builds as soon as it has vectors.
    #: relevant() reads intelligence on every typed request, so its first
    #: snapshot must not wait for a question; episodes are built on first use.
    _EAGER_SOURCES = ("intelligence",)
    #: During a long backfill, publish a fresh snapshot after this many new
    #: vectors (or an eighth of the current snapshot, whichever is larger), so
    #: new rows become findable by meaning without a rebuild per batch.
    PUBLISH_EVERY = 256

    def _with_meaning(self, source: str, query: str, lexical: list[Any],
                      limit: int) -> list[dict[str, str]] | None:
        """Fuse *lexical* rows with meaning matches; None when unavailable."""
        from . import semantic

        if not semantic.ENCODER.ready():
            return None
        started = time.perf_counter()
        try:
            question = semantic.ENCODER.encode_one(query)
            snapshot = self._snapshot(source)
            if question is None or snapshot is None or not len(snapshot):
                return None
            sims = snapshot.matrix @ question
        except Exception:
            return None
        lexical_ids = [int(row["id"]) for row in lexical]
        fresh, rows = self._meaning_candidates(source, snapshot, sims, max(limit, 8),
                                               extra_ids=lexical_ids)
        _id, text_column, _cap = self._SEMANTIC_SOURCES[source]
        similarity: dict[int, float] = {}
        for row_id, position in zip(lexical_ids, snapshot.positions(lexical_ids).tolist()):
            row = rows.get(row_id)
            if position >= 0 and row is not None and self._still_embedded(
                    snapshot, position, self._embed_text(source, row)):
                similarity[row_id] = float(sims[position])
        # A word match deleted since the word search ran is not an answer.
        live = [row_id for row_id in lexical_ids if row_id in rows]
        adjust = (self._subject_adjust(query, rows, text_column)
                  if source == "intelligence" else None)
        order = semantic.fuse(live, fresh, similarity=similarity, adjust=adjust)[:limit]
        self._trace_recall("query", source, snapshot, started)
        keys = [c.strip() for c in self._SOURCE_COLUMNS[source].split(",")]
        return [{k: rows[i][k] for k in keys} for i in order if i in rows]

    # ── who a fact is about ──────────────────────────────────────────────────
    #
    # "Am I allergic to anything?" is about the user, and "Priya is allergic
    # to peanuts" scores almost as well on meaning as "The user is allergic
    # to ibuprofen". A question asked in the first person nudges facts written
    # about the user ("the user", "my ...", "your ...") up. Nothing is pushed
    # down: ORION does not know the owner's name, so "Sam is ..." could be the
    # owner and a penalty for named subjects cost the original benchmark a
    # held-out case. Measured on 128 facts among look-alikes and record noise:
    # held-out top-1 78.5 -> 80.0, hit@3 86.2 -> 87.7; stable from 0.01 to 0.06.
    SUBJECT_BONUS = 0.03
    _FIRST_PERSON = re.compile(r"\b(?:i|i'm|im|i've|ive|i'd|my|me|mine|myself)\b", re.IGNORECASE)
    _ABOUT_USER = re.compile(
        r"\b(?:the user|user's)\b|^(?:i|i'm|i've|my|you|your|you're)\b", re.IGNORECASE)

    def _subject_adjust(self, query: str, rows: dict[int, Any],
                        text_column: str) -> dict[int, float] | None:
        if not self.SUBJECT_BONUS or not self._FIRST_PERSON.search(str(query or "")):
            return None
        return {row_id: self.SUBJECT_BONUS for row_id, row in rows.items()
                if self._ABOUT_USER.search(str(row[text_column] or "").strip())}

    def _trace_recall(self, kind: str, source: str, snapshot: _VectorSnapshot,
                      started: float) -> None:
        """Put the meaning half of a recall on the request being served, if any."""
        from .request_trace import mark

        mark("memory_meaning", kind=kind, source=source, vectors=len(snapshot),
             behind=snapshot.generation < self._vector_generation.get(source, 0),
             ms=(time.perf_counter() - started) * 1000.0)

    @staticmethod
    def _still_embedded(snapshot: _VectorSnapshot, position: int, text: Any) -> bool:
        """Whether the vector at *position* was computed from *text* as it is now."""
        from . import semantic

        digest = semantic.text_digest(str(text or "")).encode("ascii")
        return bool(snapshot.digests[position] == digest)

    def _meaning_candidates(self, source: str, snapshot: _VectorSnapshot, sims: Any,
                            wanted: int, extra_ids: Sequence[int] = (),
                            pool: Any = None) -> tuple[list[tuple[int, float]], dict[int, Any]]:
        """The best *wanted* meaning matches that are still true, and their rows.

        A snapshot can be behind the store: a fact edited or forgotten since it
        was built still has its old vector here. So each candidate is checked
        against its row as it stands NOW — a row that is gone is dropped, and
        one whose text no longer matches the vector's digest is dropped too,
        because that cosine describes words that are not there any more (the
        indexer re-embeds it; until then it is found by words, like any row not
        yet embedded). Dropped candidates are replaced from further down the
        ranking, so staleness costs recall nothing it can avoid.

        Returns best-first (id, cosine) pairs and {id: row} for every
        candidate still present plus *extra_ids* — the one SELECT the answer
        needed anyway, now also the check.
        """
        id_column, text_column, _cap = self._SEMANTIC_SOURCES[source]
        columns = self._SOURCE_COLUMNS[source]
        available = int(sims.shape[0] if pool is None else pool.shape[0])
        rows: dict[int, Any] = {}
        looked: set[int] = set()
        need = {int(i) for i in extra_ids}
        fresh: list[tuple[int, float]] = []
        taken = 0
        fetch = min(max(0, wanted), available)
        while True:
            positions = _top_positions(sims, fetch, pool)[taken:]
            candidate_ids = [int(i) for i in snapshot.ids[positions]]
            need.update(candidate_ids)
            missing = [i for i in need if i not in looked]
            if missing:
                marks = ",".join("?" * len(missing))
                with self._lock:
                    for row in self.conn.execute(
                            f"SELECT {id_column}, {columns} FROM {source} "
                            f"WHERE {id_column} IN ({marks})", missing).fetchall():
                        rows[int(row[id_column])] = row
                looked.update(missing)
            for position, row_id in zip(positions.tolist(), candidate_ids):
                row = rows.get(row_id)
                if row is not None and self._still_embedded(snapshot, position,
                                                            self._embed_text(source, row)):
                    fresh.append((row_id, float(sims[position])))
            taken += len(candidate_ids)
            if len(fresh) >= wanted or taken >= available:
                return fresh[:max(0, wanted)], rows
            fetch = min(available, taken + 2 * max(wanted - len(fresh), 4))

    #: For memories volunteered unasked (relevant()). Measured on 12 requests
    #: that should bring a specific fact and 12 everyday ones that should
    #: bring none, among 122 stored facts:
    #:     0.58  11/12 relevant, 6/12 everyday requests padded
    #:     0.60  10/12 relevant, 1/12 padded   <- chosen
    #:     0.62   8/12 relevant, 0/12 padded
    #: (A z-score against all memories separated them worse at every cut.)
    #: Re-measured 6 Oct 2026 once facts were embedded with their labels
    #: ("allergy: ..."), which lifts every score: on 15 should-bring requests
    #: and 20 everyday ones among 128 facts plus look-alikes, 0.60 padded
    #: 11/20 everyday requests (sentence-only vectors padded 11/20 there too -
    #: a fact-rich store is harsher than the original 122). At no padding,
    #: label vectors find 8/15 at 0.70, as sentence vectors did at 0.68.
    RELEVANT_FLOOR = 0.70
    #: Health and safety facts clear a lower bar, because the costs are not
    #: symmetric: "I've got a headache, what should I take" scored 0.589
    #: against "allergic to ibuprofen" — missing that is worse than an
    #: unneeded mention. Kept the same 0.05 below the floor.
    SAFETY_FLOOR = 0.65
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
    #: Below RELEVANT_FLOOR a fact may still be volunteered, from this bar up,
    #: when the request NAMES what the fact is about - a meaningful word of
    #: its label ("remind me to pay the RENT" / "rent: ..."): meaning and
    #: words agreeing. The label, not the whole sentence: matching any word
    #: of the fact let "minimise all the windows" bring "prefers a window
    #: seat". Measured 6 Oct 2026, 0.62 chosen on the calibration lines (15
    #: should-bring, 20 everyday) then checked once on 30 lines written
    #: beforehand: should-bring found 18/30 -> 24/30, everyday requests given
    #: a note 0/35 -> 2/35 ("search the web for laptops" -> the work laptop,
    #: "open youtube" -> the user's channel). None turns it off.
    WORD_SUPPORTED_FLOOR: float | None = 0.62
    #: Words too common in requests and facts alike to count as agreement.
    _PLAIN_WORDS = frozenset(
        "the user users and for you your are was were his her him she they them their "
        "there then than that this with from into out over some any all not but has had "
        "have does did doing done going thing something orion just really much many more "
        "most one two new old good best about what when where which who why how will "
        "would could should can cant need want help please make made get got put tell "
        "know like let give take today tonight tomorrow morning mornings evening "
        "evenings night nights week weekend weekday day days month year time times next "
        "last every other usual usually again also still yes now soon later me my our "
        "its it's set start stop open close find search show check turn play look "
        "send call run see go".split())

    @classmethod
    def _content_stems(cls, text: str) -> set[str]:
        """The meaningful words of *text*, crudely stemmed ("running" -> "run")."""
        stems = set()
        for word in re.findall(r"[a-z]+", str(text).lower()):
            if len(word) < 3 or word in cls._PLAIN_WORDS:
                continue
            for end in ("ing", "es", "ed", "s"):
                if len(word) - len(end) >= 3 and word.endswith(end):
                    word = word[:-len(end)]
                    if len(word) > 3 and word[-1] == word[-2]:
                        word = word[:-1]              # "runn" -> "run"
                    break
            stems.add(word)
        return stems

    def relevant(self, text: str, limit: int = 3,
                 floor: float | None = None) -> list[dict[str, Any]]:
        """Stored facts that bear on *text* by MEANING, strongest first.

        For volunteering context before a reply — not for answering a recall
        question (query() does that). Meaning-based once the semantic encoder
        is warm; until then (about 25 s after launch) only the strict label
        test of _relevant_by_label, so it never pads a prompt with word-match
        coincidences.
        """
        from . import semantic

        if not str(text or "").strip():
            return []
        if not semantic.ENCODER.ready():
            return self._relevant_by_label(str(text)[:1000], limit)
        text = self._speller.correct(str(text)[:1000])
        floor = self.RELEVANT_FLOOR if floor is None else floor
        started = time.perf_counter()
        try:
            question = semantic.ENCODER.encode_one(text)
            snapshot = self._snapshot("intelligence")
            if question is None or snapshot is None or not len(snapshot):
                return []
            sims = snapshot.matrix @ question
        except Exception:
            return []
        import numpy as np

        health = bool(self._HEALTH_REQUEST_RE.search(str(text)))
        lowest = min(floor, self.SAFETY_FLOOR) if health else floor
        supported = self.WORD_SUPPORTED_FLOOR
        if supported is not None:
            lowest = min(lowest, supported)
            request_words = self._content_stems(text)
        # Only rows over the lowest bar can be volunteered, and on a real
        # request almost none are, so the selection runs over those alone.
        pool = np.flatnonzero(sims >= lowest)
        if not pool.size:
            self._trace_recall("relevant", "intelligence", snapshot, started)
            return []
        candidates, found = self._meaning_candidates(
            "intelligence", snapshot, sims, limit * 4, pool=pool)
        self._trace_recall("relevant", "intelligence", snapshot, started)
        out: list[dict[str, Any]] = []
        for row_id, similarity in candidates:
            row = found[row_id]
            bar = (self.SAFETY_FLOOR if health and self._SAFETY_RE.search(str(row["value"]))
                   else floor)
            if similarity < bar and not (
                    supported is not None and similarity >= supported
                    and request_words & self._content_stems(str(row["key_ref"]).replace("_", " "))):
                continue
            out.append({"category": row["category"], "key_ref": row["key_ref"],
                        "value": row["value"], "similarity": round(similarity, 3)})
            if len(out) >= limit:
                break
        return out

    def _relevant_by_label(self, text: str, limit: int = 3) -> list[dict[str, Any]]:
        """relevant() while the encoder is still cold: facts the request NAMES.

        Until the encoder warms, requests - spoken ones included - met no
        stored fact at all. Words alone are exactly what relevant() refuses
        to pad a prompt with, so this is the strictest word test that still
        finds anything: EVERY meaningful word of a fact's label must be in
        the request ("pay the RENT" -> rent; "my exam" does not bring
        exam_date, which also needs "date") - the label rule
        WORD_SUPPORTED_FLOOR already applies beside meaning, without the
        meaning. Labels whose words are all under four letters never match
        on their own. Never raises."""
        request = self._content_stems(text)
        if not request:
            return []
        stems = sorted(request, key=len, reverse=True)[:12]
        clause = " OR ".join("key_ref LIKE ?" for _ in stems)
        try:
            rows = self._read_aside(
                f"SELECT category, key_ref, value FROM intelligence "
                f"WHERE ({clause}) AND {self._recallable('intelligence')} "
                f"ORDER BY updated_at DESC LIMIT 200",
                [f"%{stem}%" for stem in stems])
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            label = self._content_stems(str(row["key_ref"]).replace("_", " "))
            if not label or not label <= request or max(len(s) for s in label) < 4:
                continue
            out.append({"category": row["category"], "key_ref": row["key_ref"],
                        "value": row["value"], "similarity": None})
            if len(out) >= limit:
                break
        return out

    # ── vector snapshots ─────────────────────────────────────────────────────

    def _snapshot(self, source: str) -> _VectorSnapshot | None:
        """The published snapshot for *source*; never rebuilt on the caller's time.

        One that is behind the store is still served — every match it yields
        is checked against the row as it is now (_meaning_candidates) — and a
        rebuild is started in the background. Only a source with no snapshot
        at all is built here, once, because there is nothing to serve from.
        """
        current = self._snapshots.get(source)
        if current is None:
            return self.refresh_semantic_snapshot(source)
        if current.generation < self._vector_generation.get(source, 0):
            self._refresh_in_background(source)
        return current

    def _semantic_vectors(self, source: str) -> tuple[Any, Any]:
        """(row ids, unit-vector matrix) of the published snapshot, or (ids, None)."""
        import numpy as np

        snapshot = self._snapshot(source)
        if snapshot is None or not len(snapshot):
            return np.zeros(0, dtype=np.int64), None
        return snapshot.ids, snapshot.matrix

    #: Vector rows the change log may hold before it gives up and the next
    #: refresh reads the whole table instead (about 6 MB of float16 blobs).
    DELTA_LOG_ROWS = 8192

    def _bump_vectors(self, source: str, upserts: Sequence[tuple[int, str, bytes]] = (),
                      deletes: Sequence[int] = ()) -> None:
        """Record a committed change to *source*'s vectors. Call holding _lock.

        The change itself is logged too, while a snapshot is out, so the next
        refresh can apply it to that snapshot instead of re-reading every
        vector: at 50k rows a full read is ~258 ms of mostly GIL-holding work,
        which one saved fact used to cost. Anything the log cannot cover (it
        overflowed, or nothing was published yet) moves the floor, and a
        refresh from below the floor reads the table in full. So does a bump
        that names no rows: a change nobody described cannot be replayed.
        """
        generation = self._vector_generation.get(source, 0) + 1
        self._vector_generation[source] = generation
        log = self._vector_log.setdefault(source, [])
        rows = sum(len(u) + len(d) for _g, u, d in log) + len(upserts) + len(deletes)
        described = bool(upserts or deletes)
        if described and source in self._snapshots and rows <= self.DELTA_LOG_ROWS:
            log.append((generation, tuple(upserts), tuple(int(i) for i in deletes)))
        else:
            log.clear()
            self._vector_log_floor[source] = generation

    def _pending_changes(self, source: str, since: int, target: int) -> list | None:
        """The logged changes taking a snapshot at *since* to *target*, or None
        if any are missing. Call holding _lock."""
        if since < self._vector_log_floor.get(source, 0):
            return None
        changes = [entry for entry in self._vector_log.get(source, ())
                   if since < entry[0] <= target]
        return changes if len(changes) == target - since else None

    @staticmethod
    def _apply_changes(base: _VectorSnapshot, changes: list, generation: int) -> _VectorSnapshot:
        """*base* with *changes* applied: the same arrays a full read would give.

        Each change sets absolute values (this id now has this vector, or
        none), so applying one the base already reflects is harmless — a full
        read labelled g may include writes after g. New vectors are decoded
        exactly as a full read decodes them, so the two agree bit for bit.
        Costs one or two copies of the matrix rather than a read of the table.
        """
        import numpy as np
        from . import semantic

        dim = semantic.MODEL.dim
        final: dict[int, tuple[bytes, bytes] | None] = {}
        for _generation, upserts, deletes in changes:
            for row_id in deletes:
                final[int(row_id)] = None
            for row_id, digest, blob in upserts:
                ok = len(blob or b"") == dim * 2
                final[int(row_id)] = (str(digest or "").encode("ascii", "replace"), blob) if ok else None
        if not final:
            return _VectorSnapshot(generation, base.ids, base.matrix, base.digests)
        touched = np.fromiter(sorted(final), dtype=np.int64, count=len(final))
        at = base.positions(touched)
        live = [(row_id, value) for row_id, value in zip(touched.tolist(), (final[i] for i in touched.tolist()))
                if value is not None]
        width = max([base.digests.dtype.itemsize, 1] + [len(v[0]) for _i, v in live])
        new_ids = np.fromiter((i for i, _v in live), dtype=np.int64, count=len(live))
        new_rows = (np.frombuffer(b"".join(v[1] for _i, v in live), dtype=np.float16)
                    .reshape(len(live), dim).astype(np.float32) if live
                    else np.zeros((0, dim), dtype=np.float32))
        new_digests = np.array([v[0] for _i, v in live], dtype=f"S{width}")
        digests = base.digests.astype(f"S{width}", copy=False)
        if len(live) == len(final) and bool((at >= 0).all()):
            # Edits only: every id is already here, nothing leaves.
            matrix, marks = base.matrix.copy(), digests.copy()
            matrix[at] = new_rows
            marks[at] = new_digests
            return _VectorSnapshot(generation, base.ids.copy(), matrix, marks)
        present = at[at >= 0]
        if present.size:
            keep = np.ones(len(base), dtype=bool)
            keep[present] = False
            ids, matrix, digests = base.ids[keep], base.matrix[keep], digests[keep]
        else:
            ids, matrix = base.ids, base.matrix
        spot = np.searchsorted(ids, new_ids)
        return _VectorSnapshot(generation, np.insert(ids, spot, new_ids),
                               np.insert(matrix, spot, new_rows, axis=0),
                               np.insert(digests, spot, new_digests))

    def _build_snapshot(self, source: str) -> _VectorSnapshot:
        """Read *source*'s vectors into a new snapshot. Does not publish it.

        The generation is read under _lock BEFORE the rows, and every writer
        commits before bumping it under the same lock, so the rows include at
        least everything that generation promises. A write that lands during
        the read leaves the generation ahead of the snapshot, and it is simply
        rebuilt again — an insert, edit or delete is never lost.

        The rows come through a second, read-only connection where the store
        is a file: WAL lets it read beside the writer, so a rebuild never holds
        _lock (and every question's SQL) for the length of a 50k-row read.
        """
        import numpy as np
        from . import semantic

        dim = semantic.MODEL.dim
        with self._lock:
            generation = self._vector_generation.get(source, 0)
        rows = self._read_aside(
            "SELECT row_id, digest, vec FROM semantic_vectors WHERE source = ? ORDER BY row_id",
            (source,))
        width = dim * 2                                   # float16, see semantic.to_blob
        rows = [row for row in rows if len(row[2] or b"") == width]
        ids = np.fromiter((int(row[0]) for row in rows), dtype=np.int64, count=len(rows))
        if rows:
            matrix = np.frombuffer(b"".join(row[2] for row in rows),
                                   dtype=np.float16).reshape(len(rows), dim).astype(np.float32)
            digests = np.array([str(row[1] or "").encode("ascii", "replace") for row in rows])
        else:
            matrix = np.zeros((0, dim), dtype=np.float32)
            digests = np.zeros(0, dtype="S16")
        return _VectorSnapshot(generation, ids, matrix, digests)

    def _read_aside(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        """Run a big read on its own connection instead of under _lock.

        WAL lets a second connection read beside the writer, so scanning every
        vector or every row (tens of thousands) no longer holds the lock that
        every memory write and question needs. Rows come back as sqlite3.Row.
        An in-memory store has no second connection to open; it reads under
        the lock as before.
        """
        if str(self.db_path) == ":memory:":
            with self._lock:
                return self.conn.execute(sql, tuple(params)).fetchall()
        with self._reader_lock:
            if self._closed:
                raise sqlite3.ProgrammingError("the memory store is closed")
            if self._reader is None:
                reader = sqlite3.connect(str(self.db_path), timeout=5.0,
                                         check_same_thread=False)
                reader.row_factory = sqlite3.Row
                reader.execute("PRAGMA query_only=1")
                self._reader = reader
            try:
                # Autocommit: the read transaction ends with the statement, so
                # this connection never holds back a checkpoint between reads.
                return self._reader.execute(sql, tuple(params)).fetchall()
            except sqlite3.Error:
                self._reader.close()
                self._reader = None
                raise

    def _publish_snapshot(self, source: str, snapshot: _VectorSnapshot) -> bool:
        """Make *snapshot* the one questions use, unless a newer one is already out."""
        with self._lock:
            current = self._snapshots.get(source)
            if current is not None and current.generation >= snapshot.generation:
                return False
            self._snapshots[source] = snapshot
            log = self._vector_log.get(source)
            if log:
                log[:] = [entry for entry in log if entry[0] > snapshot.generation]
            return True

    def refresh_semantic_snapshot(self, source: str) -> _VectorSnapshot | None:
        """Bring *source*'s published snapshot up to the store, and return it.

        From the change log when it covers the gap (a copy of the matrix),
        otherwise by reading the table. Blocking; the indexer and the
        background refresher call it, and a question only for a source that
        has never had one. Bounded: under a steady stream of writes it stops
        after a few rounds and the next question or indexing pass does the rest.
        """
        with self._snapshot_build_lock:
            for _round in range(4):
                current = self._snapshots.get(source)
                with self._lock:
                    target = self._vector_generation.get(source, 0)
                    if current is not None and current.generation >= target:
                        return current
                    changes = (None if current is None else
                               self._pending_changes(source, current.generation, target))
                try:
                    fresh = (self._build_snapshot(source) if changes is None
                             else self._apply_changes(current, changes, target))
                    self._publish_snapshot(source, fresh)
                except Exception:
                    return self._snapshots.get(source)
            return self._snapshots.get(source)

    def _refresh_in_background(self, source: str) -> None:
        with self._state:
            if self._closed or source in self._refreshing:
                return
            self._refreshing.add(source)

        def _run() -> None:
            from . import semantic

            semantic.background_priority()
            fruitless = 0
            while True:
                before = self._snapshots.get(source)
                try:
                    self.refresh_semantic_snapshot(source)
                except Exception:
                    pass        # the stale snapshot keeps answering, checked row by row
                # Decide to stop under the same lock a new request checks, so a
                # change that arrives while this one is finishing either sees
                # it still running (and is caught by this re-check) or starts
                # a fresh refresher. Otherwise it would wait for a question.
                with self._state:
                    current = self._snapshots.get(source)
                    behind = (current is not None and
                              current.generation < self._vector_generation.get(source, 0))
                    advanced = current is not None and (
                        before is None or current.generation > before.generation)
                    fruitless = 0 if advanced else fruitless + 1
                    # A store that will not read (closed, locked, damaged)
                    # must not spin this thread; three empty rounds and it rests.
                    if self._closed or not behind or fruitless >= 3:
                        self._refreshing.discard(source)
                        self._state.notify_all()
                        return

        threading.Thread(target=_run, name="orion-memory-snapshot", daemon=True).start()

    def wait_semantic_idle(self, timeout: float = 30.0) -> bool:
        """Block until no embedding pass or snapshot refresh is running.

        For tests, benchmarks and an orderly shutdown; nothing on a user's
        path waits on this. True when idle, False on timeout.
        """
        with self._state:
            return self._state.wait_for(
                lambda: not (self._semantic_pending or self._refreshing), timeout)

    def index_semantics(self, batch: int = 64, max_rows: int | None = None) -> int:
        """Embed every row not yet embedded (or edited since). Blocking.

        Run from a worker thread: the encoding happens OUTSIDE the lock, so
        recall keeps answering (on words) while this works. Returns how many
        rows were embedded. Vectors of deleted rows are dropped too, and each
        source's snapshot is rebuilt here, in this thread, rather than by the
        next question.
        """
        from . import semantic

        if not semantic.ENCODER.ready():
            return 0
        try:
            self._speller.get()          # built here, off the question's path
        except Exception:
            pass
        done = 0
        with self._index_lock:
            for source, (id_column, text_column, cap) in self._SEMANTIC_SOURCES.items():
                cap = cap if max_rows is None else min(cap, max_rows)
                # Both scans read aside (see _read_aside): at 50k rows the
                # second held _lock for ~250 ms. Reading beside the writer is
                # safe here — ids are never reused (AUTOINCREMENT), so an orphan
                # stays an orphan, and _store_vectors re-checks every row's
                # text under the lock before it writes.
                # An orphan is a vector whose row is gone - or is no longer
                # something recall returns (_NOT_RECALLED), so old event
                # mirrors drop out of the meaning index on the next pass.
                recallable = self._recallable(source)
                orphans = [int(r[0]) for r in self._read_aside(
                    f"SELECT row_id FROM semantic_vectors WHERE source = ? AND row_id NOT IN "
                    f"(SELECT {id_column} FROM {source} WHERE {recallable})", (source,))]
                if orphans:
                    with self._lock:
                        self.conn.executemany(
                            "DELETE FROM semantic_vectors WHERE source = ? AND row_id = ?",
                            [(source, row_id) for row_id in orphans])
                        self.conn.commit()
                        self._bump_vectors(source, deletes=orphans)
                rows = self._read_aside(
                    f"""
                    SELECT s.{id_column} AS id, s.embed_text AS text, v.digest
                    FROM (SELECT {id_column}, {self._EMBED_TEXT_SQL[source]} AS embed_text
                          FROM {source} WHERE {recallable}
                          ORDER BY {id_column} DESC LIMIT ?) AS s
                    LEFT JOIN semantic_vectors AS v
                      ON v.source = ? AND v.row_id = s.{id_column}
                    """,
                    (cap, source),
                )
                written, finished = self._embed(source, rows, batch)
                done += written
                if not finished:
                    return done
        return done

    def index_rows(self, rows: dict[str, Sequence[int]], batch: int = 64) -> int:
        """Embed just these rows (source -> ids) if their vectors are missing
        or stale. Blocking; the background indexer's usual path.

        The full pass above reads every row and hashes every text to find the
        one that changed: at 50k facts that held _lock — and so every memory
        write and question, including each conversation turn's episode log —
        for ~250 ms, after every single save. A write already knows its id.
        """
        from . import semantic

        if not semantic.ENCODER.ready():
            return 0
        done = 0
        with self._index_lock:
            for source, wanted in rows.items():
                if source not in self._SEMANTIC_SOURCES or not wanted:
                    continue
                id_column, text_column, _cap = self._SEMANTIC_SOURCES[source]
                ids = sorted({int(i) for i in wanted})
                found: list[Any] = []
                for start in range(0, len(ids), 500):
                    part = ids[start:start + 500]
                    marks = ",".join("?" * len(part))
                    with self._lock:
                        found.extend(self.conn.execute(
                            f"""
                            SELECT s.{id_column} AS id, {self._EMBED_TEXT_SQL[source]} AS text,
                                   v.digest
                            FROM {source} AS s
                            LEFT JOIN semantic_vectors AS v
                              ON v.source = ? AND v.row_id = s.{id_column}
                            WHERE s.{id_column} IN ({marks}) AND {self._recallable(source)}
                            """, [source, *part]).fetchall())
                written, finished = self._embed(source, found, batch)
                done += written
                if not finished:
                    return done
        return done

    def _embed(self, source: str, rows: Sequence[Any], batch: int) -> tuple[int, bool]:
        """Encode the rows whose vector is missing or stale, store, publish.

        Returns (vectors written, whether the encoder kept going). Call
        holding _index_lock.
        """
        from . import semantic

        prepared = ((int(r["id"]), self._prepare(source, str(r["text"] or "")), r["digest"])
                    for r in rows)
        todo = [(row_id, text) for row_id, text, digest in prepared
                if digest != semantic.text_digest(text)]
        # Shortest first across the whole backlog, so every batch holds
        # texts of a similar length (see SemanticEncoder.encode).
        todo.sort(key=lambda item: len(item[1]))
        done = unpublished = 0
        # Only a real backlog may build the GPU session (semantic.py,
        # GPU_BACKLOG_ROWS): a boot top-up did, beside the face, every launch.
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            vectors = semantic.encode_for_index([text for _id, text in chunk], backlog=len(todo))
            if vectors is None:
                self._publish_after_indexing(source)
                return done, False
            written = self._store_vectors(source, chunk, vectors)
            done += written
            unpublished += written
            current = self._snapshots.get(source)
            if unpublished >= max(self.PUBLISH_EVERY, len(current or ()) // 8):
                self.refresh_semantic_snapshot(source)
                unpublished = 0
        self._publish_after_indexing(source)
        return done, True

    def _store_vectors(self, source: str, chunk: list[tuple[int, str]], vectors: Any) -> int:
        """Write the vectors whose rows still hold the text that was encoded.

        Encoding runs outside the lock, so a row can be edited or forgotten
        while its old text is being embedded. Writing that vector anyway would
        file old words under the new text's id (or resurrect a vector for a
        deleted row); the row is re-read here, under the lock, and skipped.
        A skipped edit has already asked for another pass (_index_soon).
        """
        from . import semantic

        id_column, text_column, _cap = self._SEMANTIC_SOURCES[source]
        marks = ",".join("?" * len(chunk))
        with self._lock:
            now = {int(r[0]): self._prepare(source, str(r[1] or "")) for r in self.conn.execute(
                f"SELECT {id_column}, {self._EMBED_TEXT_SQL[source]} FROM {source} "
                f"WHERE {id_column} IN ({marks})", [row_id for row_id, _t in chunk])}
            keep = [(source, row_id, semantic.text_digest(text), semantic.to_blob(vec))
                    for (row_id, text), vec in zip(chunk, vectors) if now.get(row_id) == text]
            if keep:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO semantic_vectors(source, row_id, digest, vec) "
                    "VALUES (?, ?, ?, ?)", keep)
                self.conn.commit()
                self._bump_vectors(source, upserts=[(row_id, digest, blob)
                                                    for _s, row_id, digest, blob in keep])
        return len(keep)

    def _publish_after_indexing(self, source: str) -> None:
        """Bring *source*'s snapshot up to date at the end of a pass.

        Only when one is already out (or the source is eager): a snapshot
        nobody has asked for would hold every conversation turn's vector in
        memory for nothing.
        """
        if source in self._snapshots or source in self._EAGER_SOURCES:
            self.refresh_semantic_snapshot(source)

    def _index_soon(self, source: str | None = None, row_id: int | None = None) -> None:
        """Embed a new or changed row shortly, off the caller's thread.

        With *source* and *row_id* only that row is looked at; without, the
        next pass scans everything (the start-up backfill does that anyway).
        """
        from . import semantic

        # Dirty state first, and all of it under one lock: a save landing just
        # as a pass decides it has finished either makes that pass go round
        # again or starts a new one. Unlocked, a write in that gap waited for
        # the NEXT save to be embedded.
        with self._state:
            self._semantic_dirty = True
            if source in self._SEMANTIC_SOURCES and row_id is not None:
                self._dirty_rows.setdefault(source, set()).add(int(row_id))
            else:
                self._full_pass_wanted = True
            if self._closed or self._semantic_pending or not semantic.ENCODER.ready():
                return
            self._semantic_pending = True

        def _run() -> None:
            semantic.background_priority()
            while True:
                with self._state:
                    if self._closed or not self._semantic_dirty:
                        self._semantic_pending = False
                        self._state.notify_all()
                        return
                    self._semantic_dirty = False
                    rows, self._dirty_rows = self._dirty_rows, {}
                    full, self._full_pass_wanted = self._full_pass_wanted, False
                try:
                    if full:
                        self.index_semantics()
                    elif rows:
                        self.index_rows(rows)
                except Exception:
                    pass        # recall still works on words

        threading.Thread(target=_run, name="orion-memory-embed", daemon=True).start()

    def forget(self, category: str = "", key_prefix: str = "", contains: str = "",
               *, key: str = "") -> int:
        """
        Delete intelligence rows by any combination of category, key prefix,
        exact *key* and value substring; returns how many were removed.  The
        AFTER DELETE FTS trigger keeps the search index in sync.  A no-criteria
        call is a no-op (never wipes everything by accident).
        """
        clauses: list[str] = []
        params: list[Any] = []
        def literal_like(value: str) -> str:
            return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        if category:
            clauses.append("category = ?")
            params.append(self._safe_slug(category))
        if key:
            # Exactly one key. A prefix is right for "forget my project_
            # notes" and wrong for one row: the memory viewer's Forget beside
            # evolution/change_1 also removed change_10 to change_18.
            exact = self._safe_slug(key)
            clauses.append("key_ref = ?")
            params.append(exact)
        if key_prefix:
            # Stored keys are slugs, but trailing underscores still belong
            # to a prefix ('project_' must not delete 'projectxone').
            prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", key_prefix.strip().lower()).lstrip("_")[:80]
            if not prefix:
                return 0
            clauses.append("key_ref LIKE ? ESCAPE '\\'")
            params.append(literal_like(prefix) + "%")
        if contains:
            SecuritySanitiser.guard_text(contains, "memory.forget")
            clauses.append("value LIKE ? ESCAPE '\\'")
            params.append(f"%{literal_like(contains)}%")
        if not clauses:
            return 0
        where = " AND ".join(clauses)
        vectors_dropped = 0
        with self._lock:
            row = self.conn.execute(
                f"SELECT COUNT(*) AS n FROM intelligence WHERE {where}", params
            ).fetchone()
            removed = int(row["n"]) if row else 0
            if removed:
                # The meaning of a forgotten fact goes with it, in the same
                # transaction — not whenever the next indexing pass happens to
                # sweep orphans, which without the encoder is never.
                dropped = [int(r[0]) for r in self.conn.execute(
                    "SELECT row_id FROM semantic_vectors WHERE source = 'intelligence' AND row_id IN "
                    f"(SELECT id FROM intelligence WHERE {where})", params)]
                if dropped:
                    self.conn.executemany(
                        "DELETE FROM semantic_vectors WHERE source = 'intelligence' AND row_id = ?",
                        [(row_id,) for row_id in dropped])
                self.conn.execute(f"DELETE FROM intelligence WHERE {where}", params)
                self.conn.commit()
                if dropped:
                    self._bump_vectors("intelligence", deletes=dropped)
                vectors_dropped = len(dropped)
        if vectors_dropped and "intelligence" in self._snapshots:
            self._refresh_in_background("intelligence")
        if removed:
            self.bus.log.emit(f"MEM: forgot {removed} record(s) [{where}].")
        return removed

    def records(self, query: str = "", limit: int = 100, *, category: str = "",
                about_user_first: bool = False) -> list[dict[str, str]]:
        """Inventory records, optionally scoped before any candidate limit.

        A scoped text lookup uses the lexical index within that category;
        unrelated categories cannot consume its result budget.

        An unscoped listing holds what recall can return - not the
        knowledge-graph mirrors and request patterns (_NOT_RECALLED). On the
        owner's store the newest 40 rows, which is what the OPS memory panel
        showed as "memories", were ALL mirrors of his own requests
        ("conversation: Give me the latest on your own updates.") - rows
        recall had excluded since 6 Oct, so the panel showed things ORION
        would never bring up when asked. *about_user_first* puts the
        categories that describe the user ahead of everything else.
        """
        limit = max(1, min(500, int(limit or 100)))
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.records")
        category = SecuritySanitiser.guard_text(str(category or "").strip(), "memory.category")
        if query:
            if category:
                return [{k: row[k] for k in ("category", "key_ref", "value", "updated_at")}
                        for row in self._lexical_intelligence(query, limit, category=category)]
            return self.query(query, limit=limit)
        if category:
            where, params = "WHERE category = ?", (category, limit)
        else:
            where, params = f"WHERE {self._recallable('intelligence')}", (limit,)
        order = "updated_at DESC"
        if about_user_first and not category:
            core = ", ".join(f"'{c}'" for c in self._CORE_CATEGORIES)
            order = f"(category IN ({core})) DESC, updated_at DESC"
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT category, key_ref, value, updated_at
                FROM intelligence
                {where}
                ORDER BY {order}
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def log_episode(self, role: str, content: str) -> None:
        """Episodic conversation memory — timeline recall of what was said and when."""
        content = str(content or "").strip()
        if not content:
            return
        try:
            with self._lock:
                row_id = self.conn.execute(
                    "INSERT INTO episodes(role, content, created_at) VALUES (?, ?, ?)",
                    (str(role or "user")[:24], content[:2000], utc_stamp()),
                ).lastrowid
                self.conn.commit()
        except Exception:
            return  # conversation logging must never break a live turn
        self._index_soon("episodes", row_id)

    def recent_replies(self, limit: int = 60) -> list[str]:
        """ORION's own latest replies, newest first (for voice_variety)."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT content FROM episodes WHERE role LIKE 'orion%' OR role = 'assistant' "
                "ORDER BY id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
        return [str(row[0] or "") for row in rows]

    def user_turns_since(self, since: str, limit: int = 5000) -> list[tuple[str, str]]:
        """The user's own turns (text, created_at) from *since* (ISO UTC) on,
        oldest first - for mood_insights."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT content, created_at FROM episodes WHERE role LIKE 'user%' "
                "AND created_at >= ? ORDER BY id DESC LIMIT ?",
                (str(since), max(1, min(20000, int(limit))))).fetchall()
        return [(str(row[0] or ""), str(row[1] or "")) for row in reversed(rows)]

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

    def _lexical_episodes(self, query: str, limit: int, *, user_only: bool = False) -> list[sqlite3.Row]:
        speaker = " AND role LIKE 'user%'" if user_only else ""
        with self._lock:
            sanitised = self._sanitise_fts_query(query)
            if sanitised:
                try:
                    rows = self.conn.execute(
                        f"""
                        SELECT rowid AS id, role, content, created_at FROM episodes_fts
                        WHERE episodes_fts MATCH ?{speaker} ORDER BY rank LIMIT ?
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
                WHERE ({clause}){speaker} ORDER BY id DESC LIMIT ?
                """,
                params,
            ).fetchall()

    # ── what the user said, as recall ────────────────────────────────────────
    #
    # Measured 17 Oct 2026 on a copy of the owner's store, 38 questions about
    # small details he really gave ORION: 19 of them were never turned into a
    # fact at all - his name, his course, his work days, his budget, a trip -
    # and lived only in the conversation log, which query_intelligence (the
    # tool the system prompt names for remembered facts) never searched. So
    # the user's own turns are recall too: what he said, in his words, dated.

    #: A turn that only ASKS ("what's my dog called?") states nothing to
    #: remember, and the user's past questions are, by meaning, the closest
    #: rows to his new one - they took the slots the answers needed.
    _ASKS = re.compile(
        r"\?\s*$|^\W*(?:(?:so|and|but|ok|okay|right|alright|all right|hey|hi|hello|orion|ryan|"
        r"um|uh|oh|well|yeah|yes|no|sir)\W+)*"
        r"(?:what|what's|whats|who|who's|whos|where|where's|when|when's|why|how|how's|which|"
        r"whose|do|does|did|is|are|am|was|were|can|could|would|will|should|have|has|tell|give|"
        r"show)\b", re.IGNORECASE)
    #: ...unless it also states something ("can you remember that my dog is
    #: called Lucy?").
    _STATES = re.compile(
        r"\b(?:remember|note|know)\s+(?:that|this|of)\b|\bby the way\b|"
        r"\bi(?:'m|\s+am|'ve|\s+have|\s+was|'ll|\s+will|\s+go|\s+work|\s+live|\s+like|"
        r"\s+love|\s+hate|\s+got)\b", re.IGNORECASE)
    #: A turn found by meaning alone must clear this (the fact floor is
    #: semantic.SEMANTIC_FLOOR, 0.58). Turns are long and chatty, so they sit
    #: closer to everything; tuned on the dev half of the owner benchmark.
    SAID_FLOOR = 0.62

    @classmethod
    def states_something(cls, text: str) -> bool:
        """Whether a user turn says something, rather than only asking."""
        text = str(text or "").strip()
        return bool(text) and (not cls._ASKS.search(text) or bool(cls._STATES.search(text)))

    def _user_turn_pool(self, snapshot: _VectorSnapshot) -> Any:
        """Snapshot positions of the user's own turns, ascending (cached)."""
        import numpy as np

        key = (snapshot.generation, len(snapshot), id(snapshot))
        cached = getattr(self, "_user_pool", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        ids = [int(row[0]) for row in self._read_aside(
            "SELECT id FROM episodes WHERE role LIKE 'user%' ORDER BY id")]
        positions = snapshot.positions(ids)
        pool = np.sort(positions[positions >= 0])
        self._user_pool = (key, pool)
        return pool

    def recall_said(self, query: str, limit: int = 4) -> list[dict[str, str]]:
        """The user's own past statements that answer *query*, best first.

        Word matches (FTS) and meaning matches over the user's turns only,
        fused like a fact query; pure questions dropped. Each row's content
        is returned with transcript-split words joined (defragment) - the
        same words, readable. Blocking: call it off the event loop.
        """
        from . import semantic

        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.said")
        limit = max(1, min(20, int(limit or 4)))
        if not query:
            return []
        query = self._speller.correct(query)
        wanted = limit * 4
        lexical = [row for row in self._lexical_episodes(query, wanted, user_only=True)
                   if self.states_something(row["content"])]
        order: list[int] = [int(row["id"]) for row in lexical]
        rows: dict[int, Any] = {int(row["id"]): row for row in lexical}
        if semantic.ENCODER.ready():
            try:
                question = semantic.ENCODER.encode_one(query)
                snapshot = self._snapshot("episodes")
            except Exception:
                question = snapshot = None
            if question is not None and snapshot is not None and len(snapshot):
                pool = self._user_turn_pool(snapshot)
                if pool.size:
                    sims = snapshot.matrix @ question
                    fresh, found = self._meaning_candidates(
                        "episodes", snapshot, sims, wanted, extra_ids=order, pool=pool)
                    similarity: dict[int, float] = {}
                    lexical_ids = [i for i in order if i in found]
                    for row_id, position in zip(lexical_ids,
                                                snapshot.positions(lexical_ids).tolist()):
                        if position >= 0 and self._still_embedded(
                                snapshot, position, self._embed_text("episodes", found[row_id])):
                            similarity[row_id] = float(sims[position])
                    meaning = [(i, c) for i, c in fresh
                               if str(found[i]["role"]).lower().startswith("user")
                               and self.states_something(found[i]["content"])]
                    order = semantic.fuse(lexical_ids, meaning, similarity=similarity,
                                          floor=self.SAID_FLOOR)
                    rows = found
        out = []
        for row_id in order[:limit]:
            row = rows.get(row_id)
            if row is None:
                continue
            out.append({"role": str(row["role"]), "content": defragment(str(row["content"])),
                        "created_at": str(row["created_at"])})
        return out

    #: How many passes the user-facing categories get before the bulk ones
    #: are considered at all. None: as many as they need. Two passes measured
    #: badly on a copy of the owner's store (17 Oct 2026): ~19 facts about him,
    #: 16 slots, and after two passes every pack, founder note and forge log
    #: took one each - his dog's name, wake-up time and university start date
    #: were only listed by key, so a spoken "what's my dog called?" had to
    #: think of a tool call to answer from his own memory.
    _CORE_ROUNDS: int | None = None

    #: How many omitted keys to name. Keys are a few words each, so this is
    #: cheap compared with values — and it is what turns "I do not know"
    #: into a lookup.
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
        # Only the chosen rows need their values. Reading every value made
        # each turn copy the whole knowledge store before discarding almost
        # all of it. Counts and bounded, indexed slices retain the same core
        # and recall index without materialising those unused values.
        limit = max(0, int(limit))
        # Records that are never a recall answer (knowledge-graph event
        # mirrors, request patterns, ORION's own bookkeeping) are not carried
        # either: on the owner's store they were ~1,600 of 1,960 rows, held
        # fact slots ("Repeated request pattern: . . . .") and filled the index.
        recallable = self._recallable("intelligence")
        with self._lock:
            # Counts and slices must describe one SQLite snapshot even if
            # another process adds a fact between queries. A savepoint also
            # nests safely inside an existing caller-owned transaction.
            self.conn.execute("SAVEPOINT prompt_context_snapshot")
            try:
                counts = {str(row["category"]): int(row["n"])
                          for row in self.conn.execute(
                              "SELECT category, COUNT(*) AS n FROM intelligence "
                              f"WHERE {recallable} GROUP BY category")}
                if not counts:
                    return ""
                ordered = ([c for c in self._CORE_CATEGORIES if c in counts]
                           + sorted(c for c in counts if c not in self._CORE_CATEGORIES))
                core_present = [c for c in self._CORE_CATEGORIES if c in counts]
                used = {c: 0 for c in ordered}
                selection: list[tuple[str, int]] = []

                def take_round(pool: list[str]) -> bool:
                    advanced = False
                    for category in pool:
                        if len(selection) >= limit:
                            break
                        offset = used[category]
                        if offset < counts[category]:
                            selection.append((category, offset))
                            used[category] += 1
                            advanced = True
                    return advanced

                rounds = 0
                while (self._CORE_ROUNDS is None or rounds < self._CORE_ROUNDS) \
                        and len(selection) < limit and take_round(core_present):
                    rounds += 1
                while len(selection) < limit and take_round(ordered):
                    pass
                # Each category starts at its own consumed offset. The old
                # shared depth skipped the two newest bulk rows altogether.
                buckets = {
                    category: self.conn.execute(
                        "SELECT category, key_ref, value FROM intelligence "
                        f"WHERE category=? AND {recallable} "
                        "ORDER BY updated_at DESC, id ASC LIMIT ?",
                        (category, count)).fetchall()
                    for category, count in used.items() if count
                }
                chosen = [buckets[category][offset] for category, offset in selection]

                remaining = {c: counts[c] - used[c] for c in ordered if counts[c] > used[c]}
                index_selection: list[tuple[str, int]] = []
                index_used = {c: 0 for c in remaining}
                while len(index_selection) < self.INDEX_KEYS:
                    advanced = False
                    for category, count in remaining.items():
                        if len(index_selection) >= self.INDEX_KEYS:
                            break
                        offset = index_used[category]
                        if offset < count:
                            index_selection.append((category, offset))
                            index_used[category] += 1
                            advanced = True
                    if not advanced:
                        break
                key_buckets = {
                    category: self.conn.execute(
                        f"SELECT key_ref FROM intelligence WHERE category=? AND {recallable} "
                        "ORDER BY updated_at DESC, id ASC LIMIT ? OFFSET ?",
                        (category, count, used[category])).fetchall()
                    for category, count in index_used.items() if count
                }
                index = [f"{category}/{key_buckets[category][offset]['key_ref']}".replace("_", " ")
                         for category, offset in index_selection]
                omitted_count = sum(remaining.values())
            finally:
                self.conn.execute("RELEASE prompt_context_snapshot")

        # The header says whose facts these are. "identity/jordan waters
        # location" read, to a model, as the user's own identity — and he
        # greeted the user by a friend's name. A named person here is someone
        # else unless the note plainly says it is the user's own name.
        # The second sentence is the Live voice path's only route to the rest:
        # a spoken question gets no per-request memory note (that is typed
        # turns only), so the model must be told plainly that one tool call
        # searches every store before it says it does not know.
        lines = ["[LOCAL INTELLIGENCE MATRIX - use naturally, never recite. "
                 "'identity' and 'personal' notes describe the user; any other "
                 "person named below is someone the user knows, never the user. "
                 "For anything else about the user - a name, a date, a plan, a "
                 "record, something they told you - call query_intelligence with "
                 "their question before answering: it searches these facts, their "
                 "records and everything they have said to you.]"]
        for row in chosen:
            label = f"{row['category']}/{row['key_ref']}".replace("_", " ")
            lines.append(f"- {label}: {row['value']}")

        if omitted_count:
            more = omitted_count - len(index)
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

    def category_counts(self) -> dict[str, int]:
        return self.matrix.category_counts()

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
        notes: list[str] = []
        if rows:
            facts = "; ".join(str(r["value"]).strip().rstrip(".") for r in rows)
            notes.append(f"[Background from ORION's memory — things already known that may "
                         f"bear on this; use only what is relevant, do not recite: {facts}.]")
        past_of = getattr(self, "past_note", None)
        past = past_of(text) if callable(past_of) else ""
        if past:
            notes.append(past)
        return "\n".join(notes)

    #: A request that leans on an earlier conversation ("like last time", "the
    #: file I mentioned", "what I told you about...").
    PAST_REFERENCE_RE = re.compile(
        r"(?i)\b(?:last time|the other day|(?:like|as|same as) before|(?:do|did) (?:it|that) again|"
        r"same again|previous(?:ly)?|"
        r"like i said|as i said|i told you|i mentioned|i said|what i said|you said|"
        r"remember (?:when|what|that|the)|we (?:talked|spoke|discussed|did)|"
        r"(?:that|the same|my usual) (?:file|document|essay|project|thing|place|song|game|"
        r"plan|one|app|folder|recipe|workout))\b")

    def past_note(self, text: str, limit: int = 2) -> str:
        """What the user said before that this request leans on, as a note -
        his memory of past conversations attached to the request, so "open the
        file I mentioned" or "like last time" needs no repeating. Only for a
        request that refers back (PAST_REFERENCE_RE); "" otherwise, or when
        nothing earlier matches. Blocking: called off the event loop."""
        said = str(text or "")
        if not self.PAST_REFERENCE_RE.search(said):
            return ""
        try:
            rows = self.recall_said(said, limit=limit + 2)
        except Exception:
            return ""
        lines: list[str] = []
        for row in rows:
            content = " ".join(str(row.get("content") or "").split())
            if not content or content.casefold() == said.strip().casefold():
                continue                       # the request itself, just logged
            when = str(row.get("created_at") or "")[:10]
            lines.append(f'on {when} they said "{content[:220]}"' if when else f'they said "{content[:220]}"')
            if len(lines) >= limit:
                break
        if not lines:
            return ""
        return ("[From your past conversations with the user — what they said before that "
                "this may refer to; use it instead of asking them to repeat it: "
                + "; ".join(lines) + ".]")

    def records(self, query: str = "", limit: int = 100) -> list[dict[str, str]]:
        return self.matrix.records(query=query, limit=limit)

    def forget(self, category: str = "", key_prefix: str = "", contains: str = "") -> int:
        """Delete persisted records by category / key prefix / value substring."""
        return self.matrix.forget(category=category, key_prefix=key_prefix, contains=contains)

    def recall_episodes(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        return self.matrix.recall_episodes(query, limit=limit)

    def recall_said(self, query: str, limit: int = 4) -> list[dict[str, str]]:
        """The user's own past statements that answer *query* (see the
        matrix's). Blocking: query_intelligence calls it on a worker thread."""
        return self.matrix.recall_said(query, limit=limit)

    def save_scene(self, **kwargs: Any) -> int:
        return self.matrix.save_scene(**kwargs)

    def scene_memories(self, query: str = "", **kwargs: Any) -> list[dict[str, Any]]:
        return self.matrix.scene_memories(query, **kwargs)

    def forget_scenes(self, **kwargs: Any) -> int:
        return self.matrix.forget_scenes(**kwargs)

    def recent_replies(self, limit: int = 60) -> list[str]:
        return self.matrix.recent_replies(limit)

    def user_turns_since(self, since: str, limit: int = 5000) -> list[tuple[str, str]]:
        return self.matrix.user_turns_since(since, limit)

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
        limit = max(0, min(500, int(limit)))
        if limit == 0:
            return []
        terms = str(query or "").casefold().split()
        if tier is MemoryTier.SHORT_TERM:
            rows = self.recent_turns(self.SESSION_TURN_WINDOW)
            return [row for row in rows if all(term in str(row).casefold() for term in terms)][-limit:]
        if tier is MemoryTier.SESSION:
            return [{"key_ref": k, "value": v} for k, v in self._session_facts.items()
                    if all(term in f"{k} {v}".casefold() for term in terms)][:limit]
        if tier is MemoryTier.CONVERSATION:
            return self.matrix.recall_episodes(query, limit=limit)
        if tier is MemoryTier.PROJECT:
            proj = self._project_slug(project or self._active_project or "general")
            return self.matrix.records(query=query, limit=limit, category=f"project_{proj}")
        category = {
            MemoryTier.LONG_TERM: "long_term",
            MemoryTier.KNOWLEDGE: "knowledge",
            MemoryTier.WORKSPACE: "workspace",
        }.get(tier, "notes")
        return self.matrix.records(query=query, limit=limit, category=category)

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
