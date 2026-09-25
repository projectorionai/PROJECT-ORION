"""
Local knowledge graph memory engine for ORION Mark XI.

The engine builds deterministic, offline relationships between conversations,
files, projects, suppliers, products, emails, research, campaigns, repositories
and meetings.  It uses SQLite plus lexical scoring, and — once ORION's LOCAL
sentence encoder is warm (semantic.py; never a cloud call) — meaning as well,
so historical reasoning remains available in MODE B.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Optional

from .bus import OrionBus
from .constants import CONFIG_DIR
from .memory import MemoryAgent, MemoryTier
from .security import SecuritySanitiser
from .utils import first_line, utc_stamp
from .db import apply_pragmas


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _entity_id(kind: str, name: str) -> str:
    digest = hashlib.sha1(f"{kind}:{_normalise(name)}".encode("utf-8")).hexdigest()[:16]
    return f"{kind}_{digest}"


# A deliberately simple contradiction heuristic (Mark XX design-spec §3):
# two relationships between the same entity pair "disagree" when one's
# evidence carries a negation cue and the other's doesn't. Not NLP-grade
# entailment — a cheap, explainable signal that two things ORION was told
# about the same pair don't obviously agree, worth a human glance.
_NEGATION_RE = re.compile(
    r"\b(not|no longer|isn't|wasn't|aren't|weren't|never|contrary|reversed|"
    r"however|instead|rather than|but not)\b", re.IGNORECASE)


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(str(text or "")))


def _tokens(value: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about",
        "after", "before", "what", "which", "were", "been", "have", "has",
        "our", "your", "orion", "sir",
    }
    return {
        t for t in re.findall(r"[a-z0-9]{3,}", str(value or "").lower())
        if t not in stop
    }


def _connected_components(node_ids: list[str], edges: list[dict[str, Any]]) -> dict[str, int]:
    """Cluster id per node — plain union-find over the given edge list.

    Mark XX design-spec §3: computed fresh from whatever slice
    graph_snapshot() is rendering, not stored — a node's cluster only ever
    needs to make sense relative to the nodes on screen with it. Two
    isolated nodes with no edge between them get two different cluster ids
    even if they'd be connected through a node outside this slice; that's
    the correct behaviour for a visualiser that only ever shows a slice.
    """
    parent = {node_id: node_id for node_id in node_ids}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        s, t = edge.get("source"), edge.get("target")
        if s in parent and t in parent:
            _union(s, t)

    roots = sorted({_find(n) for n in node_ids})
    root_to_cluster = {root: i for i, root in enumerate(roots)}
    return {node_id: root_to_cluster[_find(node_id)] for node_id in node_ids}


@dataclass
class GraphEntity:
    id: str
    name: str
    kind: str = "concept"
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    mentions: int = 0
    created_at: str = field(default_factory=utc_stamp)
    updated_at: str = field(default_factory=utc_stamp)


@dataclass
class GraphEvent:
    id: str
    source_type: str
    title: str
    text: str
    at: str
    entity_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


#: Meaning-only matches below this are not trusted; word matches whose meaning
#: is below the veto are coincidences of spelling (see semantic.py).
_SEMANTIC_FLOOR = 0.58
_SEMANTIC_VETO = 0.45
_LEXICAL_BONUS = 0.03


class KnowledgeGraphEngine:
    """SQLite-backed semantic relationship and timeline engine."""

    SCHEMA = "orion.mark_xi.knowledge_graph.v1"
    ENTITY_KINDS = {
        "conversation", "file", "project", "supplier", "product", "email",
        "research", "marketing_campaign", "code_repository", "meeting",
        "person", "brand", "concept", "citation",
    }

    def __init__(
        self,
        bus: OrionBus,
        memory: MemoryAgent,
        db_path: Path | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.telemetry = telemetry
        # Optional C2 sync journal (wired post-construction; see sync/). The
        # graph is knowledge-tier, so entity writes merge additively across nodes.
        self.journal = None
        self.db_path = db_path or (CONFIG_DIR / "knowledge_graph.db")
        self._lock = RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self.conn)
        self.conn.row_factory = sqlite3.Row
        self._initialise()
        if self.telemetry is not None:
            self.telemetry.health.register("knowledge_graph")
            self.telemetry.health.beat("knowledge_graph", "OK", "graph ready")

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _initialise(self) -> None:
        schema = """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            mentions INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
        CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
            name, kind, aliases, content=''
        );
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            evidence TEXT NOT NULL DEFAULT '',
            at TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            contradicts INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS event_vectors (
            event_id TEXT PRIMARY KEY,
            digest TEXT NOT NULL,
            vec BLOB NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
            title, text, source_type, at, content='events', content_rowid='rowid',
            tokenize="porter unicode61"
        );
        CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
            INSERT INTO events_fts(rowid, title, text, source_type, at)
            VALUES (new.rowid, new.title, new.text, new.source_type, new.at);
        END;
        CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
            INSERT INTO events_fts(events_fts, rowid, title, text, source_type, at)
            VALUES('delete', old.rowid, old.title, old.text, old.source_type, old.at);
        END;
        CREATE TABLE IF NOT EXISTS event_entities (
            event_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY(event_id, entity_id)
        );
        """
        with self._lock:
            self.conn.executescript(schema)
            # Mark XX design-spec §3/§9: a database created before this pass
            # won't have the confidence/contradicts columns even though
            # CREATE TABLE IF NOT EXISTS leaves an existing table alone —
            # migrate it in place. SQLite has no "ADD COLUMN IF NOT EXISTS",
            # so the duplicate-column error is the signal the migration
            # already ran.
            for ddl in (
                "ALTER TABLE relationships ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
                "ALTER TABLE relationships ADD COLUMN contradicts INTEGER NOT NULL DEFAULT 0",
            ):
                try:
                    self.conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass   # column already exists
            self.conn.commit()
        self._ensure_stemmed_events_index()

    def _ensure_stemmed_events_index(self) -> None:
        """Rebuild events_fts if it predates word-stemming.

        Same reasoning as OrionMemoryMatrix._ensure_stemmed_index: CREATE
        VIRTUAL TABLE IF NOT EXISTS leaves an existing table alone, so a graph
        built before this change cannot match "supervisor" against
        "supervises". events_fts is an FTS5 external-content table, so every
        byte lives in `events` and 'rebuild' regenerates the index from it —
        no data is at risk.

        A failure here is not worth refusing to start over: an unstemmed index
        still searches, just slightly less well.
        """
        with self._lock:
            try:
                row = self.conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='events_fts'"
                ).fetchone()
                if row is None or "porter" in str(row[0] or "").lower():
                    return
                self.conn.executescript(
                    "DROP TABLE events_fts;"
                    "CREATE VIRTUAL TABLE events_fts USING fts5("
                    "title, text, source_type, at, content='events',"
                    " content_rowid='rowid', tokenize=\"porter unicode61\");"
                )
                self.conn.execute("INSERT INTO events_fts(events_fts) VALUES('rebuild')")
                self.conn.commit()
            except Exception:
                try:
                    self.conn.rollback()
                except Exception:
                    pass

    def upsert_entity(
        self,
        name: str,
        kind: str = "concept",
        aliases: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GraphEntity:
        clean_name = SecuritySanitiser.guard_text(name, "graph.entity")[:180]
        if not clean_name:
            raise ValueError("entity name is required")
        clean_kind = kind if kind in self.ENTITY_KINDS else "concept"
        entity = GraphEntity(
            id=_entity_id(clean_kind, clean_name),
            name=clean_name,
            kind=clean_kind,
            aliases=[SecuritySanitiser.guard_text(str(a), "graph.alias")[:140] for a in (aliases or []) if str(a).strip()],
            metadata=metadata or {},
        )
        with self._lock:
            row = self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity.id,)).fetchone()
            if row:
                old_aliases = set(json.loads(row["aliases_json"] or "[]"))
                entity.aliases = sorted(old_aliases | set(entity.aliases))
                entity.mentions = int(row["mentions"] or 0) + 1
                entity.created_at = str(row["created_at"])
            else:
                entity.mentions = 1
            entity.updated_at = utc_stamp()
            self.conn.execute(
                """
                INSERT INTO entities(id, name, kind, aliases_json, metadata_json, mentions, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    kind=excluded.kind,
                    aliases_json=excluded.aliases_json,
                    metadata_json=excluded.metadata_json,
                    mentions=excluded.mentions,
                    updated_at=excluded.updated_at
                """,
                (
                    entity.id,
                    entity.name,
                    entity.kind,
                    json.dumps(entity.aliases, ensure_ascii=False),
                    json.dumps(entity.metadata, ensure_ascii=False),
                    entity.mentions,
                    entity.created_at,
                    entity.updated_at,
                ),
            )
            self.conn.execute(
                "INSERT INTO entities_fts(rowid, name, kind, aliases) VALUES (?, ?, ?, ?)",
                (
                    abs(hash(entity.id)) % (2**31),
                    entity.name,
                    entity.kind,
                    " ".join(entity.aliases),
                ),
            )
            self.conn.commit()
        # C2: journal the entity write for desktop⇄cloud sync (knowledge tier →
        # additive merge). Guarded: journalling must never break a graph write.
        if self.journal is not None:
            try:
                payload = json.dumps({
                    "name": entity.name, "kind": entity.kind,
                    "aliases": entity.aliases, "metadata": entity.metadata,
                }, ensure_ascii=False, sort_keys=True)
                self.journal.record(tier="knowledge", category="graph_entity",
                                    key=entity.id, payload=payload)
            except Exception as exc:
                self.bus.log.emit(f"SYNC: graph journal record skipped - {exc}")
        return entity

    def link_entities(
        self,
        source_id: str,
        target_id: str,
        kind: str = "related_to",
        weight: float = 1.0,
        evidence: str = "",
        confidence: float = 1.0,
    ) -> bool:
        """Record a relationship. Returns True when it was flagged as
        contradicting an existing one (Mark XX design-spec §3): when a new
        relationship shares the same (source, target, kind) as an existing
        one but its evidence text's negation ("not", "no longer", "however"…)
        disagrees with the existing evidence's, both the new row and every
        matching existing row are marked contradicts=1 rather than the new
        fact silently overwriting or invisibly coexisting with the old one.
        A real, if intentionally simple, heuristic — not NLP-grade
        entailment, but enough to surface "these two things ORION was told
        disagree" instead of nothing at all."""
        if not source_id or not target_id or source_id == target_id:
            return False
        clean_kind = _normalise(kind).replace(" ", "_")[:60] or "related_to"
        clean_evidence = SecuritySanitiser.guard_text(evidence, "graph.evidence")[:600]
        clean_confidence = max(0.0, min(1.0, float(confidence)))
        new_negated = _has_negation(clean_evidence)
        contradicted = False
        with self._lock:
            existing = self.conn.execute(
                """
                SELECT id, evidence FROM relationships
                WHERE source_id = ? AND target_id = ? AND kind = ?
                """,
                (source_id, target_id, clean_kind),
            ).fetchall()
            conflicting_ids = [
                row["id"] for row in existing
                if row["evidence"] and _has_negation(row["evidence"]) != new_negated
            ]
            if conflicting_ids:
                contradicted = True
                placeholders = ",".join("?" for _ in conflicting_ids)
                self.conn.execute(
                    f"UPDATE relationships SET contradicts = 1 WHERE id IN ({placeholders})",
                    conflicting_ids,
                )
            self.conn.execute(
                """
                INSERT INTO relationships(source_id, target_id, kind, weight, evidence, at,
                                          confidence, contradicts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, target_id, clean_kind, max(0.0, min(10.0, float(weight))),
                 clean_evidence, utc_stamp(), clean_confidence, 1 if contradicted else 0),
            )
            self.conn.commit()
        return contradicted

    def ingest_record(
        self,
        source_type: str,
        title: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        at: str = "",
    ) -> GraphEvent:
        clean_source = source_type if source_type in self.ENTITY_KINDS else _normalise(source_type).replace(" ", "_")[:60] or "conversation"
        clean_title = SecuritySanitiser.guard_text(title, "graph.title")[:240] or clean_source
        clean_text = SecuritySanitiser.guard_text(text, "graph.text")[:6000]
        if not clean_text:
            raise ValueError("record text is required")
        metadata = metadata or {}
        event_id = hashlib.sha1(
            f"{clean_source}:{clean_title}:{clean_text[:240]}:{at}".encode("utf-8")
        ).hexdigest()[:24]
        entities = self.extract_entities(clean_text, metadata=metadata, source_type=clean_source)
        entity_ids = [e.id for e in entities]
        event = GraphEvent(
            id=event_id,
            source_type=clean_source,
            title=clean_title,
            text=clean_text,
            at=at or utc_stamp(),
            entity_ids=entity_ids,
            metadata=metadata,
        )
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO events(id, source_type, title, text, at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_type=excluded.source_type,
                    title=excluded.title,
                    text=excluded.text,
                    at=excluded.at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    event.id,
                    event.source_type,
                    event.title,
                    event.text,
                    event.at,
                    json.dumps(event.metadata, ensure_ascii=False),
                ),
            )
            for entity_id in entity_ids:
                self.conn.execute(
                    "INSERT OR IGNORE INTO event_entities(event_id, entity_id) VALUES (?, ?)",
                    (event.id, entity_id),
                )
            self.conn.commit()
        for left_index, left in enumerate(entity_ids):
            for right in entity_ids[left_index + 1:left_index + 5]:
                self.link_entities(left, right, "co_mentioned", 1.0, clean_title)
        self.memory.remember(
            MemoryTier.KNOWLEDGE,
            f"graph_event_{event.id}",
            f"{event.source_type}: {event.title} ({len(entity_ids)} linked entities)",
        )
        self._index_soon()              # embed by meaning, off this thread
        if self.telemetry is not None:
            self.telemetry.metrics.incr("knowledge_graph.events")
        self.bus.dashboard_event.emit("knowledge_graph", self.stats())
        return event

    def extract_entities(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        source_type: str = "conversation",
    ) -> list[GraphEntity]:
        metadata = metadata or {}
        candidates: list[tuple[str, str]] = []
        for key, kind in (
            ("supplier", "supplier"),
            ("product", "product"),
            ("project", "project"),
            ("campaign", "marketing_campaign"),
            ("repository", "code_repository"),
            ("repo", "code_repository"),
            ("email", "email"),
            ("meeting", "meeting"),
            ("brand", "brand"),
        ):
            value = metadata.get(key)
            if value:
                candidates.append((str(value), kind))
        if source_type in self.ENTITY_KINDS:
            title = metadata.get("title") or metadata.get("subject")
            if title:
                candidates.append((str(title), source_type))
        patterns = [
            (r"\b(?:supplier|vendor)\s+([A-Z][A-Za-z0-9 &'-]{2,60})", "supplier"),
            (r"\b(?:product|sku)\s+([A-Z][A-Za-z0-9 &'-]{2,60})", "product"),
            (r"\b(?:project)\s+([A-Z][A-Za-z0-9 &'-]{2,60})", "project"),
            (r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,4})\b", "concept"),
        ]
        for pattern, kind in patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip(" .,:;")
                if len(name) >= 3 and not name.lower().startswith(("the ", "and ")):
                    candidates.append((name, kind))
        seen: set[str] = set()
        entities: list[GraphEntity] = []
        for name, kind in candidates[:40]:
            key = f"{kind}:{_normalise(name)}"
            if key in seen:
                continue
            seen.add(key)
            try:
                entities.append(self.upsert_entity(name, kind=kind, metadata={"source": source_type}))
            except Exception:
                continue
        return entities[:24]

    def semantic_retrieve(self, query: str, limit: int = 10) -> list[GraphEvent]:
        clean = SecuritySanitiser.guard_text(query, "graph.query")[:500]
        limit = max(1, min(50, int(limit or 10)))
        query_tokens = _tokens(clean)
        rows: list[sqlite3.Row] = []
        with self._lock:
            if clean:
                # OR, not the implicit AND that " ".join produces. FTS5 treats
                # space-separated terms as AND, so "who is my supervisor"
                # required the word "who" to appear in the stored event —
                # measured 1/6 on six ordinary questions, 6/6 with OR plus
                # stemming. BM25 (ORDER BY rank) then does the ranking, and the
                # token-overlap score below re-ranks what comes back.
                fts = " OR ".join(sorted(query_tokens))
                if fts:
                    try:
                        rows = self.conn.execute(
                            """
                            SELECT events.id, events.source_type, events.title,
                                   events.text, events.at, events.metadata_json
                            FROM events_fts JOIN events ON events_fts.rowid = events.rowid
                            WHERE events_fts MATCH ?
                            ORDER BY rank LIMIT ?
                            """,
                            (fts, limit),
                        ).fetchall()
                    except sqlite3.Error as exc:
                        # This used to be a bare `rows = []`, and it hid a real
                        # defect for the entire life of the method: the columns
                        # were unqualified, and source_type/title/text/at exist
                        # in BOTH events_fts and events, so every query raised
                        # "ambiguous column name" and silently fell through to
                        # the LIKE fallback. The index was never once used.
                        # Qualified now — and the failure is reported, because a
                        # search index that quietly stops working is worse than
                        # one that is loudly broken.
                        rows = []
                        self.bus.log.emit(
                            f"GRAPH: full-text search failed, falling back to a "
                            f"slower scan - {exc}")
            if not rows:
                # Per-term, not the whole question: LIKE '%<entire query>%'
                # could essentially never match anything a person asked.
                terms = sorted(query_tokens)[:6] or [clean]
                clause = " OR ".join(
                    "title LIKE ? OR text LIKE ? OR source_type LIKE ?"
                    for _ in terms)
                params: list[Any] = []
                for term in terms:
                    params.extend([f"%{term}%"] * 3)
                params.append(limit * 3)
                rows = self.conn.execute(
                    f"""
                    SELECT id, source_type, title, text, at, metadata_json
                    FROM events
                    WHERE {clause}
                    ORDER BY at DESC LIMIT ?
                    """,
                    params,
                ).fetchall()
        meaning = self._meaning_for(clean, limit)
        word_rank = {str(row["id"]): rank for rank, row in enumerate(rows)}
        if meaning:
            have = {str(row["id"]) for row in rows}
            extra = [event_id for event_id, cosine in meaning["top"]
                     if event_id not in have and cosine >= _SEMANTIC_FLOOR]
            if extra:
                marks = ",".join("?" * len(extra))
                with self._lock:
                    rows = list(rows) + self.conn.execute(
                        f"SELECT id, source_type, title, text, at, metadata_json "
                        f"FROM events WHERE id IN ({marks})", extra).fetchall()
        events = [self._event_from_row(row) for row in rows]
        for event in events:
            event_tokens = _tokens(event.title + " " + event.text + " " + event.source_type)
            overlap = (len(query_tokens & event_tokens) / max(1, len(query_tokens))
                       if query_tokens else 0.0)
            cosine = meaning["all"].get(event.id) if meaning else None
            # With meaning: the cosine is the score, plus the same small bonus
            # for the word-search RANK that memory was tuned on (semantic.
            # LEXICAL_BONUS). Word overlap as the tie-breaker lost near-ties:
            # it is a fraction every "…is allergic to…" note shares equally.
            # Without meaning: overlap, exactly as before.
            if cosine is not None:
                rank = word_rank.get(event.id)
                bonus = _LEXICAL_BONUS / (rank + 1) if rank is not None else 0.0
                event.score = round(cosine + bonus, 4)
            else:
                event.score = round(overlap, 3)
        if meaning:
            events = [e for e in events
                      if e.id not in meaning["all"] or meaning["all"][e.id] >= _SEMANTIC_VETO]
        events.sort(key=lambda e: (e.score, e.at), reverse=True)
        return events[:limit]

    # -- meaning (semantic.py) ------------------------------------------------

    def _meaning_for(self, query: str, limit: int) -> dict[str, Any] | None:
        """{"all": id -> cosine, "top": [(id, cosine)]}, or None when cold."""
        from . import semantic

        if not query or not semantic.ENCODER.ready():
            return None
        try:
            ids, matrix = self._event_matrix()
            question = semantic.ENCODER.encode_one(query)
        except Exception:
            return None
        if matrix is None or question is None:
            return None
        sims = matrix @ question
        order = sorted(range(len(ids)), key=lambda i: -float(sims[i]))[:max(limit * 2, 8)]
        return {"all": {ids[i]: float(sims[i]) for i in range(len(ids))},
                "top": [(ids[i], float(sims[i])) for i in order]}

    def _event_matrix(self) -> tuple[list[str], Any]:
        cached = getattr(self, "_semantic_cache", None)
        version = getattr(self, "_semantic_version", 0)
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]
        from . import semantic
        import numpy as np

        with self._lock:
            rows = self.conn.execute("SELECT event_id, vec FROM event_vectors").fetchall()
        ids, vectors = [], []
        for row in rows:
            vector = semantic.from_blob(row["vec"])
            if vector is not None:
                ids.append(str(row["event_id"]))
                vectors.append(vector)
        matrix = np.vstack(vectors).astype(np.float32) if vectors else None
        self._semantic_cache = (version, ids, matrix)
        return ids, matrix

    def _index_soon(self) -> None:
        """Embed new events shortly, on a worker thread (inert while cold)."""
        from . import semantic

        self._semantic_dirty = True
        if getattr(self, "_semantic_pending", False) or not semantic.ENCODER.ready():
            return
        self._semantic_pending = True

        def _run() -> None:
            semantic.background_priority()
            try:
                while getattr(self, "_semantic_dirty", False):
                    self._semantic_dirty = False
                    self.index_semantics()
            except Exception:
                pass
            finally:
                self._semantic_pending = False

        import threading
        threading.Thread(target=_run, name="orion-graph-embed", daemon=True).start()

    def index_semantics(self, batch: int = 16) -> int:
        """Embed every event not yet embedded (or edited since). Blocking."""
        from . import semantic

        if not semantic.ENCODER.ready():
            return 0
        with self._lock:
            self.conn.execute(
                "DELETE FROM event_vectors WHERE event_id NOT IN (SELECT id FROM events)")
            self.conn.commit()
            rows = self.conn.execute(
                """
                SELECT e.id AS id, e.title AS title, e.text AS text, v.digest AS digest
                FROM events e LEFT JOIN event_vectors v ON v.event_id = e.id
                """
            ).fetchall()
        todo = []
        for row in rows:
            text = f"{row['title']}: {row['text']}"[:2000]
            if row["digest"] != semantic.text_digest(text):
                todo.append((str(row["id"]), text))
        todo.sort(key=lambda item: len(item[1]))
        done = 0
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            vectors = semantic.ENCODER.encode([text for _id, text in chunk])
            if vectors is None:
                break
            with self._lock:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO event_vectors(event_id, digest, vec) VALUES (?, ?, ?)",
                    [(event_id, semantic.text_digest(text), semantic.to_blob(vec))
                     for (event_id, text), vec in zip(chunk, vectors)])
                self.conn.commit()
            self._semantic_version = getattr(self, "_semantic_version", 0) + 1
            done += len(chunk)
        return done

    def timeline_reconstruction(
        self,
        query: str = "",
        after: str = "",
        before: str = "",
        limit: int = 20,
    ) -> list[GraphEvent]:
        limit = max(1, min(100, int(limit or 20)))
        if query:
            events = self.semantic_retrieve(query, limit=limit * 2)
        else:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT id, source_type, title, text, at, metadata_json FROM events ORDER BY at DESC LIMIT ?",
                    (limit * 2,),
                ).fetchall()
            events = [self._event_from_row(row) for row in rows]
        if after:
            events = [e for e in events if e.at > after]
        if before:
            events = [e for e in events if e.at < before]
        events.sort(key=lambda e: e.at)
        return events[:limit]

    def context_reconstruction(self, query: str, limit: int = 8) -> str:
        events = self.semantic_retrieve(query, limit=limit)
        if not events:
            return "No graph context found for that query."
        lines = [f"Knowledge graph context for '{query}':"]
        for event in events:
            lines.append(f"- {event.at[:19]} [{event.source_type}] {event.title}: {event.text[:220]}")
        return "\n".join(lines)

    def answer_offline(self, query: str) -> str:
        clean = SecuritySanitiser.guard_text(query, "graph.answer")[:500]
        lowered = clean.lower()
        if "what happened after" in lowered:
            anchor = lowered.split("what happened after", 1)[1].strip(" ?")
            timeline = self.timeline_reconstruction(anchor, limit=12)
            if timeline:
                pivot = timeline[0]
                later = self.timeline_reconstruction(after=pivot.at, limit=6)
                if later:
                    return "After that point:\n" + "\n".join(
                        f"- {e.at[:19]} [{e.source_type}] {e.title}: {e.text[:180]}" for e in later
                    )
        if "reject" in lowered or "rejected" in lowered:
            events = [
                e for e in self.semantic_retrieve("reject rejected passed avoid category product", limit=20)
                if re.search(r"\b(reject|rejected|avoid|pass|passed)\b", e.text, re.I)
            ]
            if events:
                return "Rejected or avoided categories/products found:\n" + "\n".join(
                    f"- {e.at[:19]} {e.title}: {e.text[:180]}" for e in events[:8]
                )
        if "marketing" in lowered and ("failed" in lowered or "fail" in lowered):
            events = [
                e for e in self.semantic_retrieve("marketing campaign failed test experiment no traction", limit=20)
                if re.search(r"\b(fail|failed|loss|poor|no traction|underperformed)\b", e.text, re.I)
            ]
            if events:
                return "Failed marketing experiments found:\n" + "\n".join(
                    f"- {e.at[:19]} {e.title}: {e.text[:180]}" for e in events[:8]
                )
        return self.context_reconstruction(clean, limit=8)

    def entity_neighbourhood(self, name: str, limit: int = 20) -> list[dict[str, Any]]:
        matches = self.search_entities(name, limit=5)
        if not matches:
            return []
        ids = [m.id for m in matches]
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT r.source_id, r.target_id, r.kind, r.weight, r.evidence, r.at,
                       s.name AS source_name, s.kind AS source_kind,
                       t.name AS target_name, t.kind AS target_kind
                FROM relationships r
                JOIN entities s ON s.id = r.source_id
                JOIN entities t ON t.id = r.target_id
                WHERE r.source_id IN ({}) OR r.target_id IN ({})
                ORDER BY r.weight DESC, r.at DESC
                LIMIT ?
                """.format(",".join("?" for _ in ids), ",".join("?" for _ in ids)),
                (*ids, *ids, max(1, min(80, int(limit or 20)))),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_entities(self, query: str, limit: int = 12) -> list[GraphEntity]:
        clean = SecuritySanitiser.guard_text(query, "graph.entity_query")[:240]
        limit = max(1, min(50, int(limit or 12)))
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM entities
                WHERE name LIKE ? OR kind LIKE ? OR aliases_json LIKE ?
                ORDER BY mentions DESC, updated_at DESC
                LIMIT ?
                """,
                (f"%{clean}%", f"%{clean}%", f"%{clean}%", limit),
            ).fetchall()
        return [self._entity_from_row(row) for row in rows]

    def ingest_memory_snapshot(self, query: str = "", limit: int = 80) -> int:
        count = 0
        for row in self.memory.records(query=query, limit=limit):
            try:
                self.ingest_record(
                    source_type=str(row.get("category") or "memory"),
                    title=str(row.get("key_ref") or "memory"),
                    text=str(row.get("value") or ""),
                    metadata={"memory_category": row.get("category"), "title": row.get("key_ref")},
                    at=str(row.get("updated_at") or utc_stamp()),
                )
                count += 1
            except Exception as exc:
                self.bus.log.emit(f"GRAPH: memory row skipped - {first_line(exc)}")
        return count

    def stats(self) -> dict[str, Any]:
        with self._lock:
            entities = self.conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"]
            events = self.conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
            relationships = self.conn.execute("SELECT COUNT(*) AS n FROM relationships").fetchone()["n"]
        return {"entities": int(entities), "events": int(events), "relationships": int(relationships)}

    def graph_snapshot(self, limit: int = 42) -> dict[str, Any]:
        """A renderable slice of the graph for the live visualiser: the most
        connected entities and the relationships among them.

        This is *persistent* — it reflects whatever ORION has actually learned
        so far and grows as ingestion adds entities and links, so the on-screen
        graph correlates with his knowledge and activity rather than being a
        static illustration.

        Mark XX design-spec §3: edges now carry confidence and a contradicts
        flag (set by link_entities' negation heuristic), and nodes carry a
        cluster id — connected components computed here at render time, not
        stored, since it only ever needs to reflect the current slice."""
        limit = max(4, min(120, int(limit or 42)))
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT e.id, e.name, e.kind, e.mentions,
                       (SELECT COUNT(*) FROM relationships r
                        WHERE r.source_id = e.id OR r.target_id = e.id) AS degree
                FROM entities e
                ORDER BY degree DESC, e.mentions DESC, e.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            nodes = [
                {"id": r["id"], "name": r["name"], "kind": r["kind"],
                 "mentions": int(r["mentions"] or 0), "degree": int(r["degree"] or 0)}
                for r in rows
            ]
            ids = [n["id"] for n in nodes]
            edges: list[dict[str, Any]] = []
            if ids:
                ph = ",".join("?" for _ in ids)
                erows = self.conn.execute(
                    f"""
                    SELECT source_id, target_id, kind, weight, confidence, contradicts
                    FROM relationships
                    WHERE source_id IN ({ph}) AND target_id IN ({ph})
                    ORDER BY weight DESC
                    LIMIT 500
                    """,
                    (*ids, *ids),
                ).fetchall()
                idset = set(ids)
                for r in erows:
                    s, t = r["source_id"], r["target_id"]
                    if s != t and s in idset and t in idset:
                        edges.append({
                            "source": s, "target": t, "kind": r["kind"],
                            "weight": float(r["weight"] or 1.0),
                            "confidence": float(r["confidence"] if r["confidence"] is not None else 1.0),
                            "contradicts": bool(r["contradicts"]),
                        })
        clusters = _connected_components(ids, edges)
        for node in nodes:
            node["cluster"] = clusters.get(node["id"], -1)
        return {"nodes": nodes, "edges": edges, "totals": self.stats()}

    def entity_name(self, entity_id: str) -> str:
        """Resolve an entity id to its readable name (empty string if unknown)."""
        with self._lock:
            row = self.conn.execute(
                "SELECT name FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
        return str(row["name"]) if row else ""

    def _event_from_row(self, row: sqlite3.Row) -> GraphEvent:
        with self._lock:
            entity_rows = self.conn.execute(
                "SELECT entity_id FROM event_entities WHERE event_id = ?",
                (row["id"],),
            ).fetchall()
        return GraphEvent(
            id=str(row["id"]),
            source_type=str(row["source_type"]),
            title=str(row["title"]),
            text=str(row["text"]),
            at=str(row["at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
            entity_ids=[str(r["entity_id"]) for r in entity_rows],
        )

    def _entity_from_row(self, row: sqlite3.Row) -> GraphEntity:
        return GraphEntity(
            id=str(row["id"]),
            name=str(row["name"]),
            kind=str(row["kind"]),
            aliases=json.loads(row["aliases_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
            mentions=int(row["mentions"] or 0),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
