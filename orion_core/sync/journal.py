"""
Append-only sync journal (Cloud Roadmap C2, step 2).

Every persistent memory/graph write is recorded here as an immutable entry:

    (node_id, lamport_ts, tier, category, key, value_hash, payload)

The journal is the shippable unit of synchronisation — segments are exchanged
over ``/v1/sync/journal`` and applied idempotently on the far side. A Lamport
clock provides a causal order without wall-clock trust; ``(lamport_ts, node_id)``
is a total order every node computes identically, which is what lets the
conflict rule converge.

Idempotency: an entry is uniquely identified by ``(node_id, lamport_ts)``, so
re-applying a segment (retries, overlapping pulls) is a no-op.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, Optional

from ..constants import CONFIG_DIR
from ..utils import utc_stamp
from ..db import apply_pragmas


def value_hash(payload: str) -> str:
    return hashlib.sha256(str(payload or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JournalEntry:
    node_id: str
    lamport_ts: int
    tier: str
    category: str
    key: str
    value_hash: str
    payload: str
    seq: int = 0          # local autoincrement id (a pull cursor); 0 = unsaved

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "lamport_ts": self.lamport_ts,
            "tier": self.tier,
            "category": self.category,
            "key": self.key,
            "value_hash": self.value_hash,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JournalEntry":
        return cls(
            node_id=str(data["node_id"]),
            lamport_ts=int(data["lamport_ts"]),
            tier=str(data.get("tier", "")),
            category=str(data.get("category", "")),
            key=str(data.get("key", "")),
            value_hash=str(data.get("value_hash") or value_hash(data.get("payload", ""))),
            payload=str(data.get("payload", "")),
        )


class SyncJournal:
    """The append-only change log for one node."""

    def __init__(self, db_path: Path | None = None, *, node_id: str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else (CONFIG_DIR / "sync_journal.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        # An append-only journal is the write-heaviest store there is; the
        # default rollback journal cost 2.81 ms per commit on the qasync
        # thread. See orion_core/db.py.
        apply_pragmas(self.conn)
        self.conn.row_factory = sqlite3.Row
        self._initialise()
        self.node_id = node_id or self._ensure_node_id()

    # ── schema + metadata ────────────────────────────────────────────────────────

    def _initialise(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS journal (
                    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id    TEXT NOT NULL,
                    lamport_ts INTEGER NOT NULL,
                    tier       TEXT NOT NULL,
                    category   TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value_hash TEXT NOT NULL,
                    payload    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_journal_entry
                    ON journal(node_id, lamport_ts);
                CREATE INDEX IF NOT EXISTS idx_journal_key
                    ON journal(tier, category, key);
                CREATE TABLE IF NOT EXISTS journal_meta (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                );
                """
            )
            self.conn.commit()

    def _meta_get(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT v FROM journal_meta WHERE k=?", (key,)).fetchone()
        return row["v"] if row else None

    def _meta_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO journal_meta(k, v) VALUES (?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (key, str(value)),
        )

    def _ensure_node_id(self) -> str:
        with self._lock:
            existing = self._meta_get("node_id")
            if existing:
                return existing
            node_id = uuid.uuid4().hex
            self._meta_set("node_id", node_id)
            self.conn.commit()
            return node_id

    # ── Lamport clock ────────────────────────────────────────────────────────────

    def _current_lamport(self) -> int:
        return int(self._meta_get("lamport") or 0)

    def _tick(self) -> int:
        nxt = self._current_lamport() + 1
        self._meta_set("lamport", str(nxt))
        return nxt

    def _observe(self, remote_ts: int) -> None:
        """Lamport merge rule on receiving a remote timestamp."""
        merged = max(self._current_lamport(), int(remote_ts))
        self._meta_set("lamport", str(merged))

    # ── writing ──────────────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def record(self, tier: str, category: str, key: str, payload: str) -> JournalEntry:
        """Append a local change and return the entry."""
        payload = str(payload or "")
        with self._lock:
            ts = self._tick()
            entry = JournalEntry(
                node_id=self.node_id, lamport_ts=ts, tier=str(tier),
                category=str(category), key=str(key),
                value_hash=value_hash(payload), payload=payload,
            )
            cur = self.conn.execute(
                "INSERT INTO journal(node_id, lamport_ts, tier, category, key, "
                "value_hash, payload, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (entry.node_id, entry.lamport_ts, entry.tier, entry.category,
                 entry.key, entry.value_hash, entry.payload, utc_stamp()),
            )
            self.conn.commit()
            return JournalEntry(**{**entry.__dict__, "seq": int(cur.lastrowid)})

    def apply_remote(self, entries: Iterable) -> int:
        """Idempotently insert remote entries; advance the Lamport clock. Returns
        the number of genuinely new entries applied."""
        applied = 0
        with self._lock:
            for raw in entries:
                entry = raw if isinstance(raw, JournalEntry) else JournalEntry.from_dict(raw)
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO journal(node_id, lamport_ts, tier, "
                    "category, key, value_hash, payload, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (entry.node_id, entry.lamport_ts, entry.tier, entry.category,
                     entry.key, entry.value_hash, entry.payload, utc_stamp()),
                )
                if cur.rowcount:
                    applied += 1
                self._observe(entry.lamport_ts)
            self.conn.commit()
        return applied

    # ── reading ──────────────────────────────────────────────────────────────────

    def _row_to_entry(self, row: sqlite3.Row) -> JournalEntry:
        return JournalEntry(
            node_id=row["node_id"], lamport_ts=int(row["lamport_ts"]),
            tier=row["tier"], category=row["category"], key=row["key"],
            value_hash=row["value_hash"], payload=row["payload"], seq=int(row["seq"]),
        )

    def all_entries(self) -> list[JournalEntry]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM journal ORDER BY seq").fetchall()
        return [self._row_to_entry(r) for r in rows]

    def segments_since(self, cursor_seq: int = 0, limit: int = 1000) -> list[JournalEntry]:
        """Entries newer than *cursor_seq* (this node's local id), for a pull."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM journal WHERE seq > ? ORDER BY seq LIMIT ?",
                (int(cursor_seq), int(limit)),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def latest_seq(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT MAX(seq) AS m FROM journal").fetchone()
        return int(row["m"] or 0)

    def compact(self) -> int:
        """Bound journal growth without changing what ``reconcile`` produces.

        Keeps, per (tier, category, key), only the entries the merge rule needs —
        the single winner for last-writer-wins tiers, and the latest entry per
        distinct ``value_hash`` for additive tiers — and drops the rest. A later
        node still materialises to the identical state; re-received old entries
        are harmless (they lose the merge again). Returns entries removed.
        """
        from .conflict import is_additive
        with self._lock:
            rows = self.conn.execute("SELECT * FROM journal ORDER BY seq").fetchall()
            by_key: dict[tuple[str, str, str], list] = {}
            for r in rows:
                by_key.setdefault((r["tier"], r["category"], r["key"]), []).append(r)
            keep: set[int] = set()
            rank = lambda r: (int(r["lamport_ts"]), str(r["node_id"]))  # noqa: E731
            for (tier, _cat, _key), group in by_key.items():
                if is_additive(tier):
                    latest_by_hash: dict[str, int] = {}
                    for r in sorted(group, key=rank):
                        latest_by_hash[r["value_hash"]] = int(r["seq"])
                    keep.update(latest_by_hash.values())
                else:
                    keep.add(int(max(group, key=rank)["seq"]))
            to_delete = [int(r["seq"]) for r in rows if int(r["seq"]) not in keep]
            if to_delete:
                self.conn.executemany(
                    "DELETE FROM journal WHERE seq = ?", [(s,) for s in to_delete])
                self.conn.commit()
            return len(to_delete)
