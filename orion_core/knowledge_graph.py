"""
Local knowledge graph memory engine for ORION Mark XI.

The engine builds deterministic, offline relationships between conversations,
files, projects, suppliers, products, emails, research, campaigns, repositories
and meetings.  It uses SQLite plus simple lexical scoring rather than cloud
embeddings, so historical reasoning remains available in MODE B.
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


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _entity_id(kind: str, name: str) -> str:
    digest = hashlib.sha1(f"{kind}:{_normalise(name)}".encode("utf-8")).hexdigest()[:16]
    return f"{kind}_{digest}"


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


class KnowledgeGraphEngine:
    """SQLite-backed semantic relationship and timeline engine."""

    SCHEMA = "orion.mark_xi.knowledge_graph.v1"
    ENTITY_KINDS = {
        "conversation", "file", "project", "supplier", "product", "email",
        "research", "marketing_campaign", "code_repository", "meeting",
        "person", "brand", "concept",
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
        self.db_path = db_path or (CONFIG_DIR / "knowledge_graph.db")
        self._lock = RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
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
            at TEXT NOT NULL
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
        CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
            title, text, source_type, at, content='events', content_rowid='rowid'
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
            self.conn.commit()

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
        return entity

    def link_entities(
        self,
        source_id: str,
        target_id: str,
        kind: str = "related_to",
        weight: float = 1.0,
        evidence: str = "",
    ) -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        clean_kind = _normalise(kind).replace(" ", "_")[:60] or "related_to"
        clean_evidence = SecuritySanitiser.guard_text(evidence, "graph.evidence")[:600]
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO relationships(source_id, target_id, kind, weight, evidence, at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_id, target_id, clean_kind, max(0.0, min(10.0, float(weight))), clean_evidence, utc_stamp()),
            )
            self.conn.commit()

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
                fts = " ".join(sorted(query_tokens))
                if fts:
                    try:
                        rows = self.conn.execute(
                            """
                            SELECT id, source_type, title, text, at, metadata_json
                            FROM events_fts JOIN events ON events_fts.rowid = events.rowid
                            WHERE events_fts MATCH ?
                            ORDER BY rank LIMIT ?
                            """,
                            (fts, limit),
                        ).fetchall()
                    except sqlite3.Error:
                        rows = []
            if not rows:
                like = f"%{clean}%"
                rows = self.conn.execute(
                    """
                    SELECT id, source_type, title, text, at, metadata_json
                    FROM events
                    WHERE title LIKE ? OR text LIKE ? OR source_type LIKE ?
                    ORDER BY at DESC LIMIT ?
                    """,
                    (like, like, like, limit * 3),
                ).fetchall()
        events = [self._event_from_row(row) for row in rows]
        for event in events:
            event_tokens = _tokens(event.title + " " + event.text + " " + event.source_type)
            event.score = round(len(query_tokens & event_tokens) / max(1, len(query_tokens)), 3) if query_tokens else 0.0
        events.sort(key=lambda e: (e.score, e.at), reverse=True)
        return events[:limit]

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
