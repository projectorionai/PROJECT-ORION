"""Grounded prompts read bounded values and retain the newest facts."""

from __future__ import annotations

import sqlite3
import threading
from types import SimpleNamespace

from orion_core.memory import OrionMemoryMatrix


def matrix(rows, path=":memory:"):
    store = OrionMemoryMatrix.__new__(OrionMemoryMatrix)
    store._lock = threading.RLock()
    store.conn = sqlite3.connect(path, check_same_thread=False)
    store.conn.row_factory = sqlite3.Row
    store.conn.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE intelligence (id INTEGER PRIMARY KEY, category TEXT,
                                  key_ref TEXT, value TEXT, updated_at TEXT);
        CREATE INDEX idx_intelligence_prompt
            ON intelligence(category, updated_at DESC, id ASC);
    """)
    store.conn.executemany(
        "INSERT INTO intelligence(category,key_ref,value,updated_at) VALUES (?,?,?,?)", rows)
    store.conn.commit()
    return store


def test_bulk_categories_start_with_the_newest_rows_after_core_rounds():
    store = matrix([
        ("identity", "name", "Your name", "1"),
        ("identity", "role", "Your role", "2"),
        ("knowledge", "oldest", "Old reference", "1"),
        ("knowledge", "middle", "Previous reference", "2"),
        ("knowledge", "newest", "Current reference", "3"),
    ])
    try:
        text = store.prompt_context(limit=3)
        body = [line for line in text.splitlines() if line.startswith("- ")]
        assert body == ["- identity/role: Your role", "- identity/name: Your name",
                        "- knowledge/newest: Current reference"]
        index = next(line for line in text.splitlines() if line.startswith("[ALSO REMEMBERED"))
        assert "knowledge/middle, knowledge/oldest" in index
        assert "knowledge/newest" not in index
    finally:
        store.conn.close()


def test_single_bulk_category_retains_its_two_newest_values():
    store = matrix([("knowledge", f"k{i}", f"fact{i}", str(i)) for i in range(6)])
    try:
        body = [line for line in store.prompt_context(limit=2).splitlines() if line.startswith("- ")]
        assert body == ["- knowledge/k5: fact5", "- knowledge/k4: fact4"]
    finally:
        store.conn.close()


def test_large_values_are_not_transferred_for_omitted_facts():
    store = matrix([
        ("identity" if i % 7 == 0 else "knowledge", f"k{i}", "large value " * 400, f"{i:05}")
        for i in range(15_000)
    ])
    queries = []
    fetched = []
    original = store.conn

    class Observed:
        def execute(self, query, parameters=()):
            cursor = original.execute(query, parameters)
            if "SELECT category, key_ref, value" in query:
                queries.append((query, parameters))

                class Rows:
                    def fetchall(self):
                        rows = cursor.fetchall()
                        fetched.extend(rows)
                        return rows

                return Rows()
            return cursor

    store.conn = Observed()
    try:
        text = store.prompt_context(limit=18)
        assert len(fetched) == 18
        assert queries and all("LIMIT ?" in query for query, _ in queries)
        assert "(+14942 more)" in text
        assert "query_intelligence" in text
    finally:
        original.close()


def test_prompt_index_and_values_use_one_snapshot_during_an_external_write(tmp_path):
    path = tmp_path / "facts.db"
    store = matrix([("identity", "name", "old fact", "1")], str(path))
    writer = sqlite3.connect(path)
    inserted = False

    def concurrent_write(query):
        nonlocal inserted
        if "SELECT category, key_ref, value" in query and not inserted:
            inserted = True
            writer.execute("INSERT INTO intelligence(category,key_ref,value,updated_at) "
                           "VALUES ('identity','new','new fact','2')")
            writer.commit()

    store.conn.set_trace_callback(concurrent_write)
    try:
        first = store.prompt_context(limit=2)
        assert inserted and "old fact" in first and "new fact" not in first
        assert "ALSO REMEMBERED" not in first
        store.conn.set_trace_callback(None)
        second = store.prompt_context(limit=2)
        assert "old fact" in second and "new fact" in second
    finally:
        writer.close()
        store.conn.close()


def test_constructor_adds_the_index_for_existing_stores(tmp_path):
    path = tmp_path / "existing.db"
    store = matrix([("notes", "kept", "original", "1")], str(path))
    store.conn.execute("DROP INDEX idx_intelligence_prompt")
    store.conn.close()
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    migrated = OrionMemoryMatrix(path, tmp_path / "config", bus)
    try:
        plan = migrated.conn.execute(
            "EXPLAIN QUERY PLAN SELECT category,key_ref,value FROM intelligence "
            "WHERE category=? ORDER BY updated_at DESC,id ASC LIMIT ?", ("notes", 18)).fetchall()
        assert any("idx_intelligence_prompt" in str(row[3]) for row in plan)
        assert "original" in migrated.prompt_context()
    finally:
        migrated.close()
