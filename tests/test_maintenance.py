"""Tests for DatabaseHousekeeper (Mark X.12 §5.2)."""

from __future__ import annotations

import asyncio
import sqlite3

from orion_core.bus import OrionBus
from orion_core.maintenance import DatabaseHousekeeper


def _make_bloated_db(path, rows: int = 500) -> None:
    """Create a WAL-mode store, fill it, then delete most rows so it carries a
    populated WAL and free pages — the exact condition housekeeping compacts."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE blob (id INTEGER PRIMARY KEY, payload TEXT)")
    conn.executemany(
        "INSERT INTO blob (payload) VALUES (?)",
        [("x" * 512,) for _ in range(rows)],
    )
    conn.commit()
    conn.execute("DELETE FROM blob WHERE id > 10")  # leave free pages behind
    conn.commit()
    conn.close()


def test_sweep_discovers_and_compacts_every_store(tmp_path):
    for name in ("alpha.db", "beta.db"):
        _make_bloated_db(tmp_path / name)
    # A non-.db sibling must be ignored by discovery.
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    keeper = DatabaseHousekeeper(OrionBus(), config_dir=tmp_path)
    report = keeper.sweep()

    assert set(report.swept) == {"alpha.db", "beta.db"}
    assert not report.skipped
    # VACUUM should reclaim the free pages we deliberately created.
    assert report.reclaimed > 0


def test_sweep_preserves_data_and_leaves_a_valid_store(tmp_path):
    # (SQLite checkpoints and removes the WAL on the last clean close, so a
    # populated WAL can't be asserted after _make_bloated_db returns; what
    # matters is that the swept store stays valid with its surviving rows.)
    db = tmp_path / "store.db"
    _make_bloated_db(db)

    report = DatabaseHousekeeper(OrionBus(), config_dir=tmp_path).sweep()
    assert "store.db" in report.swept

    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM blob").fetchone()[0] == 10
    finally:
        conn.close()


def test_corrupt_store_is_skipped_not_fatal(tmp_path):
    _make_bloated_db(tmp_path / "good.db")
    (tmp_path / "broken.db").write_bytes(b"this is not a sqlite database at all")

    report = DatabaseHousekeeper(OrionBus(), config_dir=tmp_path).sweep()

    assert "good.db" in report.swept
    assert "broken.db" in report.skipped   # logged + skipped, never raised


def test_run_loop_stops_cleanly(tmp_path):
    _make_bloated_db(tmp_path / "store.db")
    keeper = DatabaseHousekeeper(OrionBus(), config_dir=tmp_path, interval_seconds=0.02)

    async def _drive() -> int:
        task = asyncio.create_task(keeper.run())
        await asyncio.sleep(0.05)   # let at least one interval elapse + sweep
        keeper.stop()
        await asyncio.wait_for(task, timeout=1.0)
        return 1

    assert asyncio.run(_drive()) == 1
