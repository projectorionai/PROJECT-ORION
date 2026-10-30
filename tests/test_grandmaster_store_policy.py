"""Exercise live SQLite settings, including the authorization durability exception."""
import sqlite3

from orion_core.grandmaster import book
from orion_core.grandmaster.dataset import Dataset
from orion_core.grandmaster.identity import Identity, IdentityStore


def settings(db):
    return {name: db.execute(f"PRAGMA {name}").fetchone()[0] for name in
            ("journal_mode", "synchronous", "busy_timeout", "cache_size", "temp_store")}


def test_dataset_uses_fast_writes_without_losing_its_resource_limits(tmp_path):
    store = Dataset(tmp_path / "training.db")
    try:
        assert settings(store.db) == dict(journal_mode="wal", synchronous=1,
                                         busy_timeout=10000, cache_size=-4096, temp_store=2)
        assert store.db.execute("PRAGMA journal_size_limit").fetchone()[0] == 8388608
        size = store.db.execute("PRAGMA page_size").fetchone()[0]
        assert store.db.execute("PRAGMA max_page_count").fetchone()[0] * size == 2 * 1024**3
    finally:
        store.close()


def test_book_compiler_uses_the_shared_policy(tmp_path, monkeypatch):
    original_connect = sqlite3.connect
    observed = []

    class Connection(sqlite3.Connection):
        def close(self):
            observed.append(settings(self))
            super().close()

    monkeypatch.setattr(book.sqlite3, "connect", lambda *a, **kw:
                        original_connect(*a, **kw, factory=Connection))
    book.compile_book([], tmp_path)
    assert observed == [dict(journal_mode="wal", synchronous=1, busy_timeout=10000,
                             cache_size=-4096, temp_store=2)]


def test_identity_keeps_fully_durable_authorization_writes(tmp_path):
    store = IdentityStore(tmp_path / "identities.db")
    store.put(Identity("example_user", "Example User"))
    with store.connect() as db:
        assert settings(db) == dict(journal_mode="wal", synchronous=2,
                                   busy_timeout=5000, cache_size=-4000, temp_store=2)
    store.remove("example_user")
    assert IdentityStore(store.path).get("example_user") is None
