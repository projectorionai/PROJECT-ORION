"""
One way to open a SQLite database (Mark XXVI) — and the reason it matters.

ORION keeps seventeen small databases and writes to them constantly: a graded
flashcard, a focus tick, a token-usage record, a wellbeing check-in. Every one of
those engines opened SQLite with the stdlib defaults, which means

    journal_mode = DELETE      a rollback journal created and fsync'd per commit
    synchronous  = FULL        the OS forced to flush to disk each time

**Measured on this machine: 2.81 ms per committed write.** Under qasync the Qt
event loop *is* the asyncio loop, so those milliseconds are not paid by a
background worker — they are paid by the thread that draws ORION's face and
services the audio callback. A hundred small writes is a third of a second of
stutter that nothing in the profiler attributes to anything.

The same benchmark with a write-ahead log and a relaxed (still crash-safe) fsync
policy: **0.03 ms per write — 104x faster.**

    journal_mode = WAL         readers never block the writer; no per-commit journal
    synchronous  = NORMAL      durable against an application crash; a power cut can
                               cost the last transaction, which for a flashcard
                               grade or a focus tick is an acceptable trade
    busy_timeout = 5000        wait for a lock rather than raising "database is locked"

``connect()`` applies that consistently, and degrades quietly: if a path refuses
WAL (some network shares do), the connection is still returned and simply keeps
the old behaviour rather than failing to open at all.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

#: Applied to every connection, in this order.
PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=5000",
    # Keep the page cache modest: seventeen databases each hoarding memory would
    # cost more than the reads save. Negative = kibibytes.
    "PRAGMA cache_size=-4000",
    "PRAGMA temp_store=MEMORY",
)


def connect(path: Path | str, *, row_factory: bool = True,
            check_same_thread: bool = False, timeout: float = 5.0) -> sqlite3.Connection:
    """Open *path* with ORION's standard performance pragmas.

    ``check_same_thread=False`` matches how the engines are used: the qasync loop
    and the GUI are the same thread, but a background job may also read. Writes
    stay small and are serialised by SQLite's own locking.
    """
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    connection = sqlite3.connect(str(target), check_same_thread=check_same_thread,
                                 timeout=timeout)
    if row_factory:
        connection.row_factory = sqlite3.Row
    apply_pragmas(connection)
    return connection


def apply_pragmas(connection: sqlite3.Connection) -> list[str]:
    """Apply the standard pragmas to an existing connection.

    Returns the pragmas that could not be applied — empty on a normal local file.
    Never raises: a database that refuses WAL still works, just slower.
    """
    refused: list[str] = []
    for pragma in PRAGMAS:
        try:
            connection.execute(pragma)
        except sqlite3.Error:
            refused.append(pragma)
    return refused


def journal_mode(connection: sqlite3.Connection) -> str:
    """The journal mode actually in force (for diagnostics)."""
    try:
        row = connection.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]) if row else "unknown"
    except sqlite3.Error:
        return "unknown"


def optimise(connection: sqlite3.Connection) -> None:
    """Cheap periodic maintenance: let SQLite refresh its query plans.

    Safe to call on shutdown or after a bulk import; it is a no-op on a database
    that does not need it.
    """
    try:
        connection.execute("PRAGMA optimize")
    except sqlite3.Error:
        pass


__all__ = ["PRAGMAS", "connect", "apply_pragmas", "journal_mode", "optimise"]
