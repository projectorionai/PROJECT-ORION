"""Change evidence must survive contention, restart and an unavailable RAG index."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from orion_core.evolution import EvolutionStore, bump


def record(store, number=1, **extra):
    args = dict(module="demo.py", before="old\n", after=f"new {number}\n",
                reasoning="fix concurrent telemetry collection", status="applied")
    args.update(extra)
    return store.record(**args)


def test_versions_and_diffs_survive_restart(tmp_path):
    path = tmp_path / "history.sqlite3"
    store = EvolutionStore(path)
    record(store)
    record(store, 2, status="failed", verification={"error": "memory exhaustion"})
    record(store, 3, impact="minor")
    rows = EvolutionStore(path).search("", 10)
    assert [r["version"] for r in rows] == ["0.1.0", None, "0.0.1"]
    assert "-old" in rows[-1]["diff"] and "+new 1" in rows[-1]["diff"]
    assert rows[0]["timestamp"].endswith("+00:00")
    assert rows[0]["before_hash"] != rows[0]["after_hash"]


def test_concurrent_writers_allocate_distinct_versions(tmp_path):
    path = tmp_path / "history.sqlite3"
    stores = [EvolutionStore(path), EvolutionStore(path)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: record(stores[n % 2], n), range(24)))
    rows = stores[0].search("", 50)
    assert len(rows) == 24
    assert {r["version"] for r in rows} == {f"0.0.{n}" for n in range(1, 25)}


def test_failed_rag_write_remains_pending_and_retries(tmp_path):
    class Memory:
        down = True
        saved = {}
        def save(self, category, key, value, **kwargs):
            if self.down:
                raise RuntimeError("index temporarily unavailable")
            self.saved[key] = (category, value)
        def query(self, query, limit):
            return [{"category": "evolution", "key_ref": key} for key in self.saved]
    memory = Memory()
    store = EvolutionStore(tmp_path / "history.sqlite3", memory)
    record(store)
    assert not memory.saved
    memory.down = False
    assert store.index_pending() == 1
    assert store.index_pending() == 0
    assert store.search("paraphrase with zero lexical overlap")[0]["id"] == 1
    assert "new 1" not in memory.saved["change_1"][1]  # source stays out of RAG


def test_manual_edits_additions_and_removals_are_recorded(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    first = root / "demo.py"
    first.write_text("x = 1\n")
    store = EvolutionStore(tmp_path / "history.sqlite3")
    assert store.observe_tree(root) == 0
    first.write_text("x = 2\n")
    assert store.observe_tree(root) == 1
    first.unlink()
    (root / "added.py").write_text("pass\n")
    assert store.observe_tree(root) == 2
    assert store.observe_tree(root) == 0
    rows = store.search("", 10)
    assert len(rows) == 3
    assert all(r["status"] == "observed" and "unverified" in r["verification"] for r in rows)
    assert all("not supplied" in r["reasoning"] for r in rows)


def test_unreadable_file_is_not_a_deletion(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "demo.py"
    path.write_text("pass\n")
    store = EvolutionStore(tmp_path / "history.sqlite3")
    store.observe_tree(root)
    original = type(path).read_text
    def unreadable(self, *args, **kwargs):
        if self == path:
            raise PermissionError("locked")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(type(path), "read_text", unreadable)
    assert store.observe_tree(root) == 0
    assert not store.search("")


def test_versions_and_statuses_are_validated_before_writing(tmp_path):
    store = EvolutionStore(tmp_path / "history.sqlite3")
    for value in ["1.2", "01.2.3", "-1.0.0", "1.0.0rc1"]:
        with pytest.raises(ValueError):
            bump(value)
    with pytest.raises(ValueError):
        record(store, status="probably_worked")
    assert not store.path.exists()
    assert bump("1.2.3", "major") == "2.0.0"


def test_history_uses_common_cache_policy_and_durable_commits(tmp_path):
    from contextlib import closing
    store = EvolutionStore(tmp_path / "history.sqlite3")
    with closing(store._connect()) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
        assert conn.execute("PRAGMA cache_size").fetchone()[0] == -4000


def test_search_treats_query_as_data(tmp_path):
    store = EvolutionStore(tmp_path / "history.sqlite3")
    record(store)
    store.search("'; DROP TABLE changes; --")
    assert len(store.search("")) == 1
    assert store.search("", 0) == []


def test_source_tracker_attaches_history_without_changing_legacy_report(tmp_path):
    from orion_core.change_awareness import SourceChangeTracker
    root = tmp_path / "source"
    root.mkdir()
    path = root / "demo.py"
    path.write_text("x = 1")
    history = EvolutionStore(tmp_path / "history.sqlite3")
    tracker = SourceChangeTracker(root, tmp_path / "source.json", history=history)
    assert tracker.detect().first_snapshot
    path.write_text("x = 222")
    assert tracker.detect().modified == ["demo.py"]
    assert history.search("")[0]["module"] == "source/demo.py"
