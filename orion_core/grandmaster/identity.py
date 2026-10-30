"""SQLite authorisation; all calls from asyncio are offloaded by the owner."""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..db import apply_pragmas


def normalise(handle: str) -> str:
    name = handle.strip().removeprefix("@").lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,29}", name):
        raise ValueError("Invalid Lichess username")
    return name


@dataclass(frozen=True)
class Identity:
    handle: str
    entity: str
    elo: int = 2604
    commentary: bool = True


class IdentityStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5)
        try:
            apply_pragmas(db)
            # Authorisation changes, including revocations, must survive a
            # power failure. Keep this store's original durability guarantee.
            db.execute("PRAGMA synchronous=FULL")
            db.execute("""CREATE TABLE IF NOT EXISTS friends (
                handle TEXT PRIMARY KEY COLLATE NOCASE,
                entity TEXT NOT NULL, elo INTEGER NOT NULL CHECK(elo BETWEEN 400 AND 3000),
                commentary INTEGER NOT NULL CHECK(commentary IN (0,1)))""")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            with db:
                yield db
        finally:
            db.close()

    def migrate(self, handles: list[str]) -> None:
        """One-time migration. Never resurrect a revoked identity on restart."""
        with self.connect() as db:
            if db.execute("SELECT 1 FROM metadata WHERE key='migrated'").fetchone():
                return
            for handle in handles:
                name = normalise(handle)
                db.execute("INSERT OR IGNORE INTO friends VALUES (?, ?, 2604, 1)", (name, name))
            db.execute("INSERT INTO metadata VALUES ('migrated', '1')")

    def put(self, identity: Identity) -> None:
        name = normalise(identity.handle)
        if not identity.entity.strip() or len(identity.entity) > 200:
            raise ValueError("A memory entity must contain 1–200 characters")
        if not 400 <= identity.elo <= 3000:
            raise ValueError("Handicap must be between 400 and 3000")
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO friends VALUES (?, ?, ?, ?)",
                       (name, identity.entity, identity.elo, int(identity.commentary)))

    def remove(self, handle: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM friends WHERE handle=?", (normalise(handle),))

    def get(self, handle: str) -> Identity | None:
        try:
            name = normalise(handle)
        except ValueError:
            return None
        with self.connect() as db:
            row = db.execute("SELECT handle, entity, elo, commentary FROM friends WHERE handle=?",
                             (name,)).fetchone()
        return Identity(row[0], row[1], row[2], bool(row[3])) if row else None

    def all(self) -> list[Identity]:
        with self.connect() as db:
            rows = db.execute("SELECT handle, entity, elo, commentary FROM friends ORDER BY handle").fetchall()
        return [Identity(r[0], r[1], r[2], bool(r[3])) for r in rows]
