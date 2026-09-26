"""
Unified file & folder ingestion engine (additive module).

ORION ingests arbitrary documents — code, prose, spreadsheets, archives, images
— exactly once and scales to large libraries by *never* reprocessing content it
has already seen.  The engine is deliberately offline-first: text extraction,
summarisation, embedding and search all run locally with no provider call, so
ingestion works in MODE B with the network down.

Four cooperating services, each named in the ORION execution directive:

    • DocumentFingerprintService  — SHA256 + size + mtime + version per path,
      so an unchanged file is skipped and a changed file is re-indexed.
    • EmbeddingCache              — deterministic local embeddings keyed by
      content hash, so identical chunks (across files) embed once.
    • KnowledgeDeduplicationService — a content-hash ledger, so an identical
      chunk is never written to memory / the knowledge graph twice.
    • IncrementalIndexer          — decides skip / re-index / resume for a path
      and drives chunk-level idempotent storage.

``IngestionEngine`` ties them together and integrates with the *existing*
architecture rather than duplicating it: extracted text is fed to the
``KnowledgeGraphEngine`` (entity extraction, cross-referencing, relationship
mapping) and a per-document summary is written to the ``MemoryAgent`` KNOWLEDGE
tier so ordinary recall and the offline LocalBrain surface ingested material.

All heavy parsing degrades gracefully: an unavailable optional parser (pypdf,
python-docx, openpyxl, an OCR engine) yields a clear per-file note, never an
exception that aborts a folder import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Optional

from .constants import BASE_DIR, CONFIG_DIR
from .utils import first_line, utc_stamp
from .db import apply_pragmas

# ── format taxonomy ───────────────────────────────────────────────────────────
# Suffix → coarse category.  The category drives tagging, the extractor and how
# aggressively the text is treated as prose vs. structured/code.

_CODE_SUFFIXES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".cs": "csharp", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".c": "c", ".h": "c", ".hpp": "cpp", ".sql": "sql",
    ".rb": "ruby", ".go": "go", ".rs": "rust", ".php": "php",
    ".sh": "shell", ".ps1": "powershell",
}
_DATA_SUFFIXES = {".json": "json", ".xml": "xml", ".yaml": "yaml",
                  ".yml": "yaml", ".csv": "csv", ".tsv": "tsv", ".toml": "toml"}
_PROSE_SUFFIXES = {".txt": "text", ".md": "markdown", ".markdown": "markdown",
                   ".rst": "text", ".log": "log"}
_MARKUP_SUFFIXES = {".html": "html", ".htm": "html", ".rtf": "rtf"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
_OFFICE_SUFFIXES = {".docx": "docx", ".doc": "doc", ".xlsx": "xlsx",
                    ".xls": "xls", ".ods": "ods"}

# Everything the engine will attempt (images/zip/pdf handled specially).
SUPPORTED_SUFFIXES = (
    set(_CODE_SUFFIXES) | set(_DATA_SUFFIXES) | set(_PROSE_SUFFIXES)
    | set(_MARKUP_SUFFIXES) | set(_OFFICE_SUFFIXES) | _IMAGE_SUFFIXES
    | {".pdf", ".zip"}
)

_EMBED_DIM = 128            # hashing-trick embedding dimensionality
#: Lexical overlap's share beside a meaning score (cosine, ~0.3-0.8). Small:
#: meaning does the ranking, exact words break near-ties.
SEMANTIC_LEXICAL_WEIGHT = 0.15
_CHUNK_CHARS = 1600         # target characters per chunk
_MAX_CHUNKS = 400           # hard cap so a giant file cannot exhaust storage
_MAX_ZIP_MEMBERS = 200
_SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?](?:\s|$)")
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
# Normative / imperative language that marks a sentence as an instruction —
# used to distil a file's *directive* (what it tells ORION to do / know).
_DIRECTIVE_MARKERS = re.compile(
    r"\b(must|should|shall|do not|don't|never|always|ensure|require[sd]?|"
    r"needs? to|has to|have to|make sure|avoid|prohibit(?:ed)?|mandatory|"
    r"forbidden|important|note that|remember to)\b",
    re.IGNORECASE,
)
_IMPERATIVE_START = re.compile(
    r"^\s*(please\s+)?(use|add|remove|create|delete|update|set|configure|install|"
    r"enable|disable|run|build|write|read|follow|implement|check|verify|ensure|"
    r"keep|make|call|import|export|store|remember|ignore|handle|do|use)\b",
    re.IGNORECASE,
)
_DECL_RE = re.compile(
    r"(?m)^\s*(?:async\s+)?(?:def|class|func|function|type|interface|struct)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)")
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "are", "was",
    "were", "has", "have", "not", "but", "you", "your", "our", "its", "his",
    "her", "their", "which", "while", "then", "than", "them", "these", "those",
    "will", "would", "can", "could", "should", "may", "also", "such", "each",
}


@dataclass
class IngestionResult:
    """Outcome of ingesting one file."""

    path: str
    status: str                     # indexed | updated | skipped | unsupported | error | empty
    doc_id: str = ""
    category: str = ""
    chunks: int = 0
    words: int = 0
    version: int = 1
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    directive: str = ""
    entities: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"indexed", "updated", "skipped"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "status": self.status, "doc_id": self.doc_id,
            "category": self.category, "chunks": self.chunks, "words": self.words,
            "version": self.version, "tags": self.tags, "summary": self.summary,
            "directive": self.directive,
            "entities": self.entities[:12], "note": self.note,
        }


@dataclass
class BatchResult:
    """Outcome of ingesting a folder."""

    root: str
    results: list[IngestionResult] = field(default_factory=list)

    def _count(self, *statuses: str) -> int:
        return sum(1 for r in self.results if r.status in statuses)

    @property
    def indexed(self) -> int: return self._count("indexed", "updated")
    @property
    def skipped(self) -> int: return self._count("skipped")
    @property
    def failed(self) -> int: return self._count("error", "unsupported", "empty")

    def summary(self) -> str:
        return (
            f"{len(self.results)} file(s): {self.indexed} indexed, "
            f"{self.skipped} unchanged (skipped), {self.failed} skipped/failed."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root, "total": len(self.results),
            "indexed": self.indexed, "skipped": self.skipped, "failed": self.failed,
            "results": [r.to_dict() for r in self.results[:200]],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Local embedding (offline, deterministic hashing trick)
# ──────────────────────────────────────────────────────────────────────────────

def _hash_embed(text: str, dim: int = _EMBED_DIM) -> list[float]:
    """A deterministic bag-of-tokens embedding, L2-normalised.

    No model, no network: each token is hashed into a bucket with a signed
    weight.  Good enough for near-duplicate detection and lexical-semantic
    similarity search entirely offline.
    """
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOP:
            continue
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        bucket = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))     # both are unit vectors


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


# ──────────────────────────────────────────────────────────────────────────────
# Persistence layer — one SQLite database, several logical services over it
# ──────────────────────────────────────────────────────────────────────────────

class _Store:
    """Shared SQLite connection + schema for every ingestion service."""

    SCHEMA_VERSION = "orion.ingestion.v1"

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self.conn)
        self.conn.row_factory = sqlite3.Row
        self._initialise()

    def _initialise(self) -> None:
        with self.lock:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL DEFAULT 0,
                    mtime REAL NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'pending',
                    category TEXT NOT NULL DEFAULT '',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    word_count INTEGER NOT NULL DEFAULT 0,
                    summary TEXT NOT NULL DEFAULT '',
                    directive TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    indexed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
                CREATE INDEX IF NOT EXISTS idx_documents_cat ON documents(category);

                CREATE TABLE IF NOT EXISTS chunks (
                    doc_id TEXT NOT NULL,
                    ord INTEGER NOT NULL,
                    hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY (doc_id, ord)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);

                CREATE TABLE IF NOT EXISTS embeddings (
                    hash TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS semantic_chunks (
                    hash TEXT PRIMARY KEY,
                    vec BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunk_ledger (
                    hash TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    at TEXT NOT NULL
                );
                """
            )
            # Additive migration (#13): the directive column on stores created
            # before file-directive memory existed.  SQLite has no
            # ADD COLUMN IF NOT EXISTS, so guard on the current schema.
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(documents)")}
            if "directive" not in cols:
                self.conn.execute(
                    "ALTER TABLE documents ADD COLUMN directive TEXT NOT NULL DEFAULT ''")
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()


class DocumentFingerprintService:
    """Tracks SHA256 / size / mtime / version per path to gate reprocessing."""

    def __init__(self, store: _Store) -> None:
        self._s = store

    @staticmethod
    def fingerprint(path: Path) -> tuple[str, int, float]:
        data = path.read_bytes()
        return _sha256(data), len(data), path.stat().st_mtime

    def lookup(self, doc_id: str) -> Optional[sqlite3.Row]:
        with self._s.lock:
            cur = self._s.conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            )
            return cur.fetchone()

    def decide(self, doc_id: str, sha: str) -> str:
        """skip (unchanged+complete) | resume (partial) | update (changed) | new."""
        row = self.lookup(doc_id)
        if row is None:
            return "new"
        if row["sha256"] == sha:
            return "skip" if row["status"] == "complete" else "resume"
        return "update"

    def upsert(self, doc_id: str, path: str, sha: str, size: int, mtime: float,
               *, version: int, status: str, category: str, chunk_count: int,
               word_count: int, summary: str, tags: list[str],
               metadata: dict[str, Any], directive: str = "") -> None:
        now = utc_stamp()
        with self._s.lock:
            self._s.conn.execute(
                """
                INSERT INTO documents(doc_id, path, sha256, size, mtime, version,
                    status, category, chunk_count, word_count, summary, directive,
                    tags_json, metadata_json, indexed_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    path=excluded.path, sha256=excluded.sha256, size=excluded.size,
                    mtime=excluded.mtime, version=excluded.version,
                    status=excluded.status, category=excluded.category,
                    chunk_count=excluded.chunk_count, word_count=excluded.word_count,
                    summary=excluded.summary, directive=excluded.directive,
                    tags_json=excluded.tags_json,
                    metadata_json=excluded.metadata_json, updated_at=excluded.updated_at
                """,
                (doc_id, path, sha, size, mtime, version, status, category,
                 chunk_count, word_count, summary, directive, json.dumps(tags),
                 json.dumps(metadata, ensure_ascii=False), now, now),
            )
            self._s.conn.commit()

    def set_status(self, doc_id: str, status: str) -> None:
        with self._s.lock:
            self._s.conn.execute(
                "UPDATE documents SET status=?, updated_at=? WHERE doc_id=?",
                (status, utc_stamp(), doc_id),
            )
            self._s.conn.commit()


class EmbeddingCache:
    """Content-hash → embedding, so identical chunks embed exactly once."""

    def __init__(self, store: _Store) -> None:
        self._s = store
        self.hits = 0
        self.misses = 0

    def embed(self, text: str, content_hash: str) -> list[float]:
        with self._s.lock:
            row = self._s.conn.execute(
                "SELECT vector_json FROM embeddings WHERE hash=?", (content_hash,)
            ).fetchone()
            if row is not None:
                self.hits += 1
                return json.loads(row["vector_json"])
            self.misses += 1
            vec = _hash_embed(text)
            self._s.conn.execute(
                "INSERT OR REPLACE INTO embeddings(hash, vector_json) VALUES (?, ?)",
                (content_hash, json.dumps(vec)),
            )
            self._s.conn.commit()
            return vec


class KnowledgeDeduplicationService:
    """A ledger of chunk content-hashes already committed to knowledge."""

    def __init__(self, store: _Store) -> None:
        self._s = store

    def is_new(self, content_hash: str) -> bool:
        with self._s.lock:
            row = self._s.conn.execute(
                "SELECT 1 FROM chunk_ledger WHERE hash=?", (content_hash,)
            ).fetchone()
            return row is None

    def mark(self, content_hash: str, doc_id: str) -> None:
        with self._s.lock:
            self._s.conn.execute(
                "INSERT OR IGNORE INTO chunk_ledger(hash, doc_id, at) VALUES (?,?,?)",
                (content_hash, doc_id, utc_stamp()),
            )
            self._s.conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# The engine
# ──────────────────────────────────────────────────────────────────────────────

class IngestionEngine:
    """Parse → index → summarise → embed → tag → store → cross-reference."""

    def __init__(
        self,
        bus: Any | None = None,
        memory: Any | None = None,
        knowledge_graph: Any | None = None,
        telemetry: Any | None = None,
        ocr: Any | None = None,
        db_path: Path | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.graph = knowledge_graph
        self.telemetry = telemetry
        self.ocr = ocr
        self.store = _Store(db_path or (CONFIG_DIR / "ingestion.db"))
        self.fingerprints = DocumentFingerprintService(self.store)
        self.embeddings = EmbeddingCache(self.store)
        self.dedup = KnowledgeDeduplicationService(self.store)
        # Meaning-based search (semantic.py): chunk hash -> vector, loaded as
        # one matrix and rebuilt when _semantic_version moves.
        self._semantic_cache: tuple[int, list[str], Any] | None = None
        self._semantic_version = 0
        self._semantic_pending = False
        self._semantic_dirty = False
        if self.telemetry is not None:
            try:
                self.telemetry.health.register("ingestion")
                self.telemetry.health.beat("ingestion", "OK", "engine ready")
            except Exception:
                pass

    def close(self) -> None:
        self.store.close()

    # ── public API ────────────────────────────────────────────────────────────

    def ingest_file(self, path: str | os.PathLike[str],
                    tags: Iterable[str] | None = None) -> IngestionResult:
        source = self._resolve(path)
        rel = str(source)
        if source is None or not source.is_file():
            return IngestionResult(path=str(path), status="error",
                                   note="file not found")
        suffix = source.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            return IngestionResult(path=rel, status="unsupported",
                                   note=f"unsupported type '{suffix}'")
        if suffix == ".zip":
            # Archives fan out into a batch; represent as an error-free note here.
            batch = self.ingest_zip(source, tags=tags)
            return IngestionResult(path=rel, status="indexed", category="zip",
                                   chunks=batch.indexed,
                                   note=f"archive → {batch.summary()}")
        try:
            sha, size, mtime = self.fingerprints.fingerprint(source)
        except OSError as exc:
            return IngestionResult(path=rel, status="error", note=first_line(exc, 100))

        doc_id = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:20]
        decision = self.fingerprints.decide(doc_id, sha)
        prior = self.fingerprints.lookup(doc_id)
        if decision == "skip":
            return IngestionResult(
                path=rel, status="skipped", doc_id=doc_id,
                category=(prior["category"] if prior else ""),
                chunks=(prior["chunk_count"] if prior else 0),
                words=(prior["word_count"] if prior else 0),
                version=(prior["version"] if prior else 1),
                summary=(prior["summary"] if prior else ""),
                directive=(prior["directive"] if prior else ""),
                note="unchanged (SHA256 match)")

        text, note = self._extract(source, suffix)
        if not text.strip():
            self.fingerprints.upsert(
                doc_id, str(source), sha, size, mtime, version=1,
                status="empty", category=self._category(suffix), chunk_count=0,
                word_count=0, summary="", tags=list(tags or []), metadata={})
            return IngestionResult(path=rel, status="empty", doc_id=doc_id,
                                   category=self._category(suffix),
                                   note=note or "no extractable text")

        version = (prior["version"] + 1) if (prior and decision == "update") else 1
        return self._index(source, doc_id, sha, size, mtime, suffix, text,
                           tags=list(tags or []), version=version,
                           updating=(decision in {"update", "resume"}),
                           extract_note=note)

    def ingest_folder(self, path: str | os.PathLike[str], *, recursive: bool = True,
                      tags: Iterable[str] | None = None,
                      progress: Callable[[IngestionResult], None] | None = None,
                      ) -> BatchResult:
        root = self._resolve(path)
        batch = BatchResult(root=str(root) if root else str(path))
        if root is None or not root.is_dir():
            return batch
        walker = root.rglob("*") if recursive else root.glob("*")
        for item in sorted(walker):
            if not item.is_file():
                continue
            if item.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if self._is_noise(item):
                continue
            result = self.ingest_file(item, tags=tags)
            batch.results.append(result)
            if progress is not None:
                try:
                    progress(result)
                except Exception:
                    pass
        if self.bus is not None:
            try:
                self.bus.log.emit(f"INGEST: {root.name} — {batch.summary()}")
                self.bus.dashboard_event.emit("ingestion", batch.to_dict())
            except Exception:
                pass
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("ingestion.batches")
            except Exception:
                pass
        return batch

    def ingest_zip(self, path: str | os.PathLike[str],
                   tags: Iterable[str] | None = None) -> BatchResult:
        source = self._resolve(path)
        batch = BatchResult(root=str(source) if source else str(path))
        if source is None or not source.is_file():
            return batch
        extract_root = CONFIG_DIR / "ingest_cache" / source.stem
        try:
            with zipfile.ZipFile(source) as zf:
                members = [m for m in zf.namelist()
                           if not m.endswith("/")][:_MAX_ZIP_MEMBERS]
                for member in members:
                    if Path(member).suffix.lower() not in SUPPORTED_SUFFIXES:
                        continue
                    if "/../" in member or member.startswith("/"):
                        continue                     # zip-slip guard
                    try:
                        target = Path(zf.extract(member, extract_root))
                    except Exception:
                        continue
                    batch.results.append(
                        self.ingest_file(target, tags=list(tags or []) + ["from-zip"]))
        except zipfile.BadZipFile:
            batch.results.append(IngestionResult(
                path=str(source), status="error", note="corrupt archive"))
        return batch

    # ── search ──────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Rank stored chunks by meaning when the encoder is warm and every
        chunk is embedded, else by the offline hashed embedding — each blended
        with lexical overlap."""
        if not query.strip():
            return []
        qvec = _hash_embed(query)
        qtok = set(_tokens(query))
        with self.store.lock:
            rows = self.store.conn.execute(
                """
                SELECT c.doc_id, c.ord, c.text, c.hash, d.path, d.category, d.tags_json
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                """
            ).fetchall()
        meaning = self._semantic_similarities(query, {row["hash"] for row in rows})
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            lexical = len(qtok & set(_tokens(row["text"]))) / (len(qtok) or 1)
            if meaning is not None:
                score = meaning[row["hash"]] + SEMANTIC_LEXICAL_WEIGHT * lexical
            else:
                vec = self.embeddings.embed(row["text"], row["hash"])
                sim = _cosine(qvec, vec)
                score = 0.7 * sim + 0.3 * lexical
            if score <= 0:
                continue
            scored.append((score, {
                "doc_id": row["doc_id"], "path": row["path"],
                "category": row["category"], "ord": row["ord"],
                "snippet": first_line(row["text"], 220),
                "score": round(score, 4),
                "tags": json.loads(row["tags_json"] or "[]"),
            }))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    # ── meaning-based search (semantic.py) ─────────────────────────────────

    def _semantic_similarities(self, query: str, hashes: set[str]) -> dict[str, float] | None:
        """hash -> cosine for every chunk, or None to use the hashed path.

        None unless the encoder is warm AND every chunk has a vector: the two
        paths score on different scales, so a half-indexed library is searched
        the old way until the background pass finishes.
        """
        from . import semantic

        if not hashes or not semantic.ENCODER.ready():
            return None
        try:
            known, matrix = self._semantic_matrix()
        except Exception:
            return None
        if matrix is None or not hashes.issubset(known):
            self._index_soon()
            return None
        question = semantic.ENCODER.encode_one(query)
        if question is None:
            return None
        sims = matrix @ question
        order = self._semantic_cache[1] if self._semantic_cache else []
        return {h: float(sims[i]) for i, h in enumerate(order)}

    def _semantic_matrix(self) -> tuple[set[str], Any]:
        cached = self._semantic_cache
        if cached is not None and cached[0] == self._semantic_version:
            return set(cached[1]), cached[2]
        from . import semantic
        import numpy as np

        with self.store.lock:
            rows = self.store.conn.execute("SELECT hash, vec FROM semantic_chunks").fetchall()
        hashes, vectors = [], []
        for row in rows:
            vector = semantic.from_blob(row["vec"])
            if vector is not None:
                hashes.append(row["hash"])
                vectors.append(vector)
        matrix = np.vstack(vectors).astype(np.float32) if vectors else None
        self._semantic_cache = (self._semantic_version, hashes, matrix)
        return set(hashes), matrix

    def index_semantics(self, batch: int = 16) -> int:
        """Embed every chunk not yet embedded. Blocking - run it off the loop.

        Keyed by content hash, like the hashed cache, so a chunk shared by two
        files (or re-ingested unchanged) is embedded once.
        """
        from . import semantic

        if not semantic.ENCODER.ready():
            return 0
        with self.store.lock:
            rows = self.store.conn.execute(
                """
                SELECT c.hash AS hash, MIN(c.text) AS text FROM chunks c
                LEFT JOIN semantic_chunks s ON s.hash = c.hash
                WHERE s.hash IS NULL GROUP BY c.hash
                """
            ).fetchall()
        todo = sorted(((r["hash"], str(r["text"] or "")) for r in rows),
                      key=lambda item: len(item[1]))
        done = 0
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            vectors = semantic.ENCODER.encode([text for _h, text in chunk])
            if vectors is None:
                break
            with self.store.lock:
                self.store.conn.executemany(
                    "INSERT OR REPLACE INTO semantic_chunks(hash, vec) VALUES (?, ?)",
                    [(h, semantic.to_blob(v)) for (h, _t), v in zip(chunk, vectors)])
                self.store.conn.commit()
            self._semantic_version += 1
            done += len(chunk)
        return done

    def _index_soon(self) -> None:
        """Embed new chunks shortly, on a worker thread."""
        from . import semantic

        self._semantic_dirty = True
        if self._semantic_pending or not semantic.ENCODER.ready():
            return
        self._semantic_pending = True

        def _run() -> None:
            semantic.background_priority()
            try:
                while self._semantic_dirty:
                    self._semantic_dirty = False
                    self.index_semantics()
            except Exception:
                pass
            finally:
                self._semantic_pending = False

        import threading
        threading.Thread(target=_run, name="orion-library-embed", daemon=True).start()

    def library(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.store.lock:
            rows = self.store.conn.execute(
                """
                SELECT doc_id, path, category, chunk_count, word_count, version,
                       status, summary, directive, tags_json, updated_at
                FROM documents ORDER BY updated_at DESC LIMIT ?
                """, (limit,),
            ).fetchall()
        return [{
            "doc_id": r["doc_id"], "path": r["path"], "category": r["category"],
            "chunks": r["chunk_count"], "words": r["word_count"],
            "version": r["version"], "status": r["status"], "summary": r["summary"],
            "directive": r["directive"], "tags": json.loads(r["tags_json"] or "[]"),
            "updated_at": r["updated_at"],
        } for r in rows]

    def catalog(self, limit: int = 50) -> list[dict[str, Any]]:
        """Per-file directive manifest: what each ingested file *is for* (#13).

        This is the explicit record the user asked ORION to keep — every
        ingested file with a clear statement of its directive.
        """
        with self.store.lock:
            rows = self.store.conn.execute(
                """
                SELECT path, category, directive, summary, updated_at
                FROM documents WHERE status != 'empty'
                ORDER BY updated_at DESC LIMIT ?
                """, (limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "filename": Path(r["path"]).name,
                "path": r["path"],
                "category": r["category"],
                "directive": r["directive"] or r["summary"] or "(no directive derived)",
                "updated_at": r["updated_at"],
            })
        return out

    def stats(self) -> dict[str, Any]:
        with self.store.lock:
            docs = self.store.conn.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
            chunks = self.store.conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
            uniq = self.store.conn.execute("SELECT COUNT(*) n FROM chunk_ledger").fetchone()["n"]
        return {
            "documents": docs, "chunks": chunks, "unique_chunks": uniq,
            "embed_hits": self.embeddings.hits, "embed_misses": self.embeddings.misses,
        }

    # ── indexing pipeline ─────────────────────────────────────────────────────

    def _index(self, source: Path, doc_id: str, sha: str, size: int, mtime: float,
               suffix: str, text: str, *, tags: list[str], version: int,
               updating: bool, extract_note: str) -> IngestionResult:
        category = self._category(suffix)
        # Fresh (re)index replaces prior chunks for this document.
        with self.store.lock:
            self.store.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.store.conn.commit()

        chunks = self._chunk(text)
        title = self._title(source, text)
        summary = self._summarise(text, category)
        directive = self._derive_directive(text, category, title, summary, source, suffix)
        derived_tags = self._auto_tags(text, category, suffix)
        all_tags = sorted(set(tags) | derived_tags)
        word_count = len(text.split())

        stored = 0
        new_chunks = 0
        for ord_, chunk in enumerate(chunks):
            chash = _sha256(chunk.encode("utf-8"))
            with self.store.lock:
                self.store.conn.execute(
                    "INSERT OR REPLACE INTO chunks(doc_id, ord, hash, text) VALUES (?,?,?,?)",
                    (doc_id, ord_, chash, chunk),
                )
                self.store.conn.commit()
            self.embeddings.embed(chunk, chash)      # populate cache
            stored += 1
            if self.dedup.is_new(chash):
                self.dedup.mark(chash, doc_id)
                new_chunks += 1
            # Checkpoint so an interruption leaves a resumable 'partial' row.
            self.fingerprints.upsert(
                doc_id, str(source), sha, size, mtime, version=version,
                status="partial", category=category, chunk_count=stored,
                word_count=word_count, summary=summary, tags=all_tags,
                metadata=self._metadata(source, suffix, text), directive=directive)

        metadata = self._metadata(source, suffix, text)
        self.fingerprints.upsert(
            doc_id, str(source), sha, size, mtime, version=version,
            status="complete", category=category, chunk_count=stored,
            word_count=word_count, summary=summary, tags=all_tags,
            metadata=metadata, directive=directive)

        entities = self._cross_reference(title, text, category, metadata)
        self._remember(doc_id, title, summary, all_tags, category, source, directive)
        if new_chunks:
            self._index_soon()          # embed by meaning, off this thread

        if self.bus is not None:
            try:
                self.bus.log.emit(
                    f"INGEST: {source.name} — {stored} chunk(s), "
                    f"{new_chunks} new, {len(entities)} entities.")
                self.bus.dashboard_event.emit("ingestion_file", {
                    "path": str(source), "category": category, "chunks": stored,
                    "tags": all_tags, "entities": entities[:8]})
            except Exception:
                pass
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("ingestion.files")
                self.telemetry.metrics.gauge("ingestion.new_chunks", float(new_chunks))
            except Exception:
                pass

        return IngestionResult(
            path=str(source), status=("updated" if updating else "indexed"),
            doc_id=doc_id, category=category, chunks=stored, words=word_count,
            version=version, tags=all_tags, summary=summary, directive=directive,
            entities=entities, note=extract_note)

    # ── extraction dispatch ───────────────────────────────────────────────────

    def _extract(self, source: Path, suffix: str) -> tuple[str, str]:
        try:
            if suffix == ".pdf":
                return self._extract_pdf(source)
            if suffix in {".docx"}:
                return self._extract_docx(source)
            if suffix in {".xlsx"}:
                return self._extract_xlsx(source)
            if suffix in _IMAGE_SUFFIXES:
                return self._extract_image(source)
            if suffix in _MARKUP_SUFFIXES:
                raw = source.read_text(encoding="utf-8", errors="replace")
                if suffix == ".rtf":
                    return self._strip_rtf(raw), ""
                return re.sub(r"<[^>]+>", " ", raw), ""
            if suffix in _DATA_SUFFIXES or suffix in _CODE_SUFFIXES or suffix in _PROSE_SUFFIXES:
                return source.read_text(encoding="utf-8", errors="replace"), ""
            # Fallback: best-effort text.
            return source.read_text(encoding="utf-8", errors="replace"), ""
        except OSError as exc:
            return "", first_line(exc, 100)

    def _extract_pdf(self, source: Path) -> tuple[str, str]:
        reader_cls: Any = None
        try:
            from pypdf import PdfReader as reader_cls  # type: ignore
        except Exception:
            try:
                from PyPDF2 import PdfReader as reader_cls  # type: ignore
            except Exception:
                reader_cls = None
        if reader_cls is None:
            return "", "pypdf not installed; PDF text disabled (pip install pypdf)"
        try:
            reader = reader_cls(str(source))
            chunks = []
            for page in reader.pages[:120]:
                try:
                    chunks.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(chunks), ""
        except Exception as exc:
            return "", f"PDF parse failed: {first_line(exc, 80)}"

    def _extract_docx(self, source: Path) -> tuple[str, str]:
        try:
            import docx  # type: ignore
            document = docx.Document(str(source))
            paras = [p.text for p in document.paragraphs]
            for table in getattr(document, "tables", []):
                for r in table.rows:
                    paras.append(" | ".join(c.text for c in r.cells))
            return "\n".join(paras), ""
        except Exception:
            # Degrade: a .docx is a zip of XML; pull document.xml text nodes.
            try:
                with zipfile.ZipFile(source) as zf:
                    xml = zf.read("word/document.xml").decode("utf-8", "replace")
                text = re.sub(r"<w:p[ >]", "\n", xml)
                return re.sub(r"<[^>]+>", " ", text), "python-docx not installed; used raw XML"
            except Exception as exc:
                return "", f"DOCX parse failed: {first_line(exc, 80)}"

    def _extract_xlsx(self, source: Path) -> tuple[str, str]:
        try:
            import openpyxl  # type: ignore
            wb = openpyxl.load_workbook(str(source), read_only=True, data_only=True)
            lines = []
            for ws in wb.worksheets:
                lines.append(f"# sheet: {ws.title}")
                for row in ws.iter_rows(values_only=True):
                    cells = [str(c) for c in row if c is not None]
                    if cells:
                        lines.append(" | ".join(cells))
            return "\n".join(lines), ""
        except Exception as exc:
            return "", f"openpyxl not installed or parse failed: {first_line(exc, 80)}"

    def _extract_image(self, source: Path) -> tuple[str, str]:
        if self.ocr is None or not getattr(self.ocr, "available", lambda: False)():
            return f"[image: {source.name}]", "no OCR engine available"
        try:
            from PIL import Image  # type: ignore
            with Image.open(source) as img:
                text = self.ocr.image_to_text(img)
            return (text or f"[image: {source.name}]"), ("" if text else "OCR found no text")
        except Exception as exc:
            return f"[image: {source.name}]", f"OCR failed: {first_line(exc, 80)}"

    @staticmethod
    def _strip_rtf(raw: str) -> str:
        text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)
        text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
        text = re.sub(r"[{}]", " ", text)
        return text

    # ── analysis helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _category(suffix: str) -> str:
        if suffix in _CODE_SUFFIXES:
            return "code"
        if suffix in _DATA_SUFFIXES:
            return "data"
        if suffix in _MARKUP_SUFFIXES:
            return "markup"
        if suffix in _OFFICE_SUFFIXES:
            return "document"
        if suffix in _IMAGE_SUFFIXES:
            return "image"
        if suffix == ".pdf":
            return "pdf"
        return "text"

    def _chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        # Prefer paragraph boundaries; pack up to ~_CHUNK_CHARS per chunk.
        paras = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        buf = ""
        for para in paras:
            para = para.strip()
            if not para:
                continue
            if len(buf) + len(para) + 1 > _CHUNK_CHARS and buf:
                chunks.append(buf)
                buf = para
            else:
                buf = f"{buf}\n{para}" if buf else para
            if len(buf) >= _CHUNK_CHARS:
                chunks.append(buf)
                buf = ""
            if len(chunks) >= _MAX_CHUNKS:
                break
        if buf and len(chunks) < _MAX_CHUNKS:
            chunks.append(buf)
        # Hard-split any oversized single paragraph.
        out: list[str] = []
        for c in chunks:
            if len(c) <= _CHUNK_CHARS * 2:
                out.append(c)
            else:
                for i in range(0, len(c), _CHUNK_CHARS):
                    out.append(c[i:i + _CHUNK_CHARS])
        return out[:_MAX_CHUNKS]

    def _summarise(self, text: str, category: str) -> str:
        """Offline extractive summary: keyword-scored top sentences."""
        if category == "code":
            # Summarise code by its declarations, not prose sentences.
            decls = re.findall(
                r"(?m)^\s*(?:def|class|function|public|private|func|fn|type|interface)\b[^\n{:(]{0,80}",
                text)
            head = "; ".join(d.strip() for d in decls[:6])
            return first_line(head or text, 300)
        sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if len(s.strip()) > 30]
        if not sentences:
            return first_line(text, 300)
        freq: dict[str, int] = {}
        for tok in _tokens(text):
            freq[tok] = freq.get(tok, 0) + 1
        def score(sent: str) -> float:
            toks = _tokens(sent)
            return sum(freq.get(t, 0) for t in toks) / (len(toks) or 1)
        ranked = sorted(sentences[:200], key=score, reverse=True)[:3]
        # Restore original order for readability.
        ordered = [s for s in sentences if s in ranked][:3]
        return first_line(" ".join(ordered), 500)

    def _derive_directive(self, text: str, category: str, title: str,
                          summary: str, source: Path, suffix: str) -> str:
        """Distil *what this file is for* — its directive — in one line (#13).

        The goal is that ORION remembers each ingested file with an explicit
        understanding of its purpose: for instructional prose that means the
        actual imperatives it carries; for code, what it defines; for data,
        what it configures; otherwise, what it is a reference on.
        """
        name = source.name
        if category == "code":
            decls: list[str] = []
            for match in _DECL_RE.finditer(text):
                if match.group(1) not in decls:
                    decls.append(match.group(1))
                if len(decls) >= 6:
                    break
            lang = _CODE_SUFFIXES.get(suffix, "code")
            if decls:
                return f"{lang.title()} module '{name}': defines {', '.join(decls)}."
            return f"{lang.title()} source '{name}' — code ORION can reference."

        if category == "data":
            hint = ""
            if suffix == ".json":
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict) and obj:
                        hint = ", ".join(list(obj.keys())[:8])
                except Exception:
                    hint = ""
            detail = f"keys: {hint}" if hint else first_line(summary or title, 160)
            return f"Data/config '{name}': {detail}."

        if category == "image":
            snippet = first_line(text.replace("[image:", "").strip(" ]"), 120)
            return f"Visual reference '{name}'" + (f": {snippet}." if snippet else ".")

        # Prose / markup / pdf / document — surface real instructions if present.
        sentences = [s.strip() for s in _SENTENCE_RE.findall(text)
                     if 12 <= len(s.strip()) <= 240]
        directives: list[str] = []
        for sent in sentences:
            if _DIRECTIVE_MARKERS.search(sent) or _IMPERATIVE_START.match(sent):
                directives.append(sent)
            if len(directives) >= 3:
                break
        if directives:
            return f"Instructions in '{name}': " + first_line(" ".join(directives), 320)
        topic = title if (title and title.lower() != name.lower()) else summary
        return f"Reference '{name}': " + first_line(topic or first_line(text, 160), 200)

    def _auto_tags(self, text: str, category: str, suffix: str) -> set[str]:
        tags = {category, suffix.lstrip(".")}
        lang = _CODE_SUFFIXES.get(suffix) or _DATA_SUFFIXES.get(suffix)
        if lang:
            tags.add(lang)
        freq: dict[str, int] = {}
        for tok in _tokens(text):
            freq[tok] = freq.get(tok, 0) + 1
        for tok, _n in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:6]:
            tags.add(tok)
        return {t for t in tags if t}

    def _title(self, source: Path, text: str) -> str:
        for line in text.splitlines():
            line = line.strip(" #\t")
            if 8 <= len(line) <= 140:
                return line
        return source.stem.replace("_", " ").title()

    def _metadata(self, source: Path, suffix: str, text: str) -> dict[str, Any]:
        try:
            st = source.stat()
        except OSError:
            st = None
        return {
            "filename": source.name,
            "suffix": suffix,
            "category": self._category(suffix),
            "size": (st.st_size if st else 0),
            "modified": (st.st_mtime if st else 0),
            "lines": text.count("\n") + 1,
            "words": len(text.split()),
            "title": self._title(source, text),
        }

    def _cross_reference(self, title: str, text: str, category: str,
                         metadata: dict[str, Any]) -> list[str]:
        """Hand text to the knowledge graph for entity + relationship mapping."""
        if self.graph is None:
            return []
        try:
            event = self.graph.ingest_record(
                source_type="file", title=title, text=text[:6000],
                metadata={"category": category, **metadata})
        except Exception as exc:
            if self.bus is not None:
                try:
                    self.bus.log.emit(f"INGEST: graph link skipped — {first_line(exc, 80)}")
                except Exception:
                    pass
            return []
        # Resolve entity ids to readable names when the graph exposes them.
        names: list[str] = []
        getter = getattr(self.graph, "entity_name", None)
        for eid in getattr(event, "entity_ids", [])[:12]:
            if callable(getter):
                try:
                    names.append(getter(eid) or eid)
                    continue
                except Exception:
                    pass
            names.append(eid)
        return names

    def _remember(self, doc_id: str, title: str, summary: str,
                  tags: list[str], category: str, source: Path,
                  directive: str = "") -> None:
        if self.memory is None:
            return
        try:
            from .memory import MemoryTier
            tier: Any = MemoryTier.KNOWLEDGE
        except Exception:
            tier = "knowledge"
        # Explicitly record the file *and* its directive, so recalling this
        # document always surfaces what ORION is meant to do with it (#13).
        value = (f"Ingested file '{source.name}' [{category}]. "
                 f"Directive: {directive or summary} "
                 f"Summary: {summary} (tags: {', '.join(tags[:6])})")
        try:
            self.memory.remember(tier, f"doc_{doc_id}", value)
        except Exception:
            pass

    # ── path + noise helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve(path: str | os.PathLike[str]) -> Optional[Path]:
        try:
            p = Path(os.path.expandvars(os.path.expanduser(str(path))))
        except Exception:
            return None
        if not p.is_absolute():
            p = BASE_DIR / p
        return p

    @staticmethod
    def _is_noise(item: Path) -> bool:
        parts = {p.lower() for p in item.parts}
        noise_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv",
                      ".mypy_cache", ".pytest_cache", "dist", "build", ".idea"}
        return bool(parts & noise_dirs)
