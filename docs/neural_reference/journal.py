from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path


class Journal:
    """Durable job snapshots and small telemetry events; local disk only."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
                ts REAL, job TEXT, event TEXT, tool TEXT, ok INTEGER, seconds REAL);
        """)

    def _record(self, snapshot, event, tool, ok, seconds):
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?, ?)",
                (snapshot["id"], json.dumps(snapshot)),
            )
            self.db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), snapshot["id"], event, tool, ok, seconds),
            )

    async def record(self, snapshot, event, tool="", ok=None, seconds=0.0):
        await asyncio.to_thread(self._record, snapshot, event, tool, ok, seconds)

    def load(self):
        with self.lock:
            return [
                json.loads(row[0])
                for row in self.db.execute("SELECT snapshot FROM jobs")
            ]

    def metrics(self):
        with self.lock:
            return [
                dict(zip(("tool", "attempts", "success_rate", "mean_seconds"), row))
                for row in self.db.execute(
                    "SELECT tool, COUNT(*), AVG(ok), AVG(seconds) FROM events "
                    "WHERE event = 'finished' GROUP BY tool"
                )
            ]

    def close(self):
        with self.lock:
            self.db.close()
