"""Backups must be complete snapshots and restore only the selected archive."""
from datetime import datetime
import sqlite3
from types import SimpleNamespace
import zipfile

import pytest

import orion_core.backup_manager as backups


@pytest.fixture
def manager(tmp_path, monkeypatch):
    base = tmp_path / "app"
    config = base / "config"
    config.mkdir(parents=True)
    monkeypatch.setattr(backups, "BASE_DIR", base)
    monkeypatch.setattr(backups, "CONFIG_DIR", config)
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None),
                          dashboard_event=SimpleNamespace(emit=lambda *_: None))
    return backups.BackupManager(bus, destination=tmp_path / "backups")


def test_same_second_backups_do_not_overwrite_one_another(manager, monkeypatch):
    class Clock:
        @staticmethod
        def now():
            return datetime(2026, 9, 26, 12, 0, 0)
    monkeypatch.setattr(backups, "datetime", Clock)
    assert manager._backup_sync("first").ok
    assert manager._backup_sync("second").ok
    archives = list(manager.destination.glob("orion_backup_*.zip"))
    assert len(archives) == 2
    notes = set()
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            notes.add(zf.read("BACKUP_NOTE.txt"))
    assert notes == {b"first", b"second"}


def test_failed_backup_does_not_publish_a_partial_archive(manager, monkeypatch):
    def fail(*_, **__):
        raise OSError("disk full")
    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail)
    result = manager._backup_sync("note")
    assert not result.ok
    assert list(manager.destination.glob("orion_backup_*.zip")) == []


def test_wal_database_is_a_standalone_consistent_snapshot(manager, tmp_path):
    database = backups.CONFIG_DIR / "memory.sqlite3"
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE records(value TEXT)")
        conn.execute("INSERT INTO records VALUES ('committed in WAL')")
        conn.commit()
        assert manager._backup_sync("").ok
        archive = next(manager.destination.glob("orion_backup_*.zip"))
        with zipfile.ZipFile(archive) as zf:
            snapshot = tmp_path / "snapshot.sqlite3"
            snapshot.write_bytes(zf.read("config/memory.sqlite3"))
            assert not any(name.endswith(("-wal", "-shm")) for name in zf.namelist())
        with sqlite3.connect(snapshot) as saved:
            assert saved.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert saved.execute("SELECT value FROM records").fetchall() == [("committed in WAL",)]


def test_unknown_restore_name_never_silently_restores_latest(manager):
    assert manager._backup_sync("current").ok
    result = manager._restore_sync("does-not-exist.zip")
    assert not result.ok
    assert not (backups.BASE_DIR / "restore").exists()


def test_repeat_restore_preserves_previously_reviewed_files(manager):
    settings = backups.CONFIG_DIR / "settings.json"
    settings.write_text("original", encoding="utf-8")
    assert manager._backup_sync("").ok
    assert manager._restore_sync("").ok
    restored = next((backups.BASE_DIR / "restore").rglob("settings.json"))
    restored.write_text("reviewed edit", encoding="utf-8")
    assert manager._restore_sync("").ok
    assert restored.read_text(encoding="utf-8") == "reviewed edit"


def test_external_configuration_directory_keeps_canonical_archive_paths(manager, tmp_path, monkeypatch):
    external = tmp_path / "external-config"
    external.mkdir()
    (external / "settings.json").write_text("external settings", encoding="utf-8")
    monkeypatch.setattr(backups, "CONFIG_DIR", external)
    assert manager._backup_sync("").ok
    with zipfile.ZipFile(next(manager.destination.glob("orion_backup_*.zip"))) as zf:
        assert zf.read("config/settings.json") == b"external settings"


def test_partial_file_read_failure_keeps_previous_backup(manager, monkeypatch):
    assert manager._backup_sync("stable").ok
    original = next(manager.destination.glob("orion_backup_*.zip"))
    before = original.read_bytes()
    (backups.CONFIG_DIR / "settings.json").write_text("settings", encoding="utf-8")
    def fail(*_, **__):
        raise OSError("file unavailable")
    monkeypatch.setattr(zipfile.ZipFile, "write", fail)
    assert not manager._backup_sync("").ok
    assert list(manager.destination.glob("orion_backup_*.zip")) == [original]
    assert original.read_bytes() == before


def test_ambiguous_restore_name_requires_unique_selection(manager):
    assert manager._backup_sync("first").ok
    assert manager._backup_sync("second").ok
    assert not manager._restore_sync("orion_backup_").ok


def test_backup_cannot_recursively_include_its_own_archive(manager):
    manager.destination = backups.CONFIG_DIR / "backups"
    manager.destination.mkdir()
    assert not manager._backup_sync("").ok
    assert list(manager.destination.iterdir()) == []


def test_snapshot_uses_common_policy_without_changing_source_journal_mode(manager, tmp_path, monkeypatch):
    database = backups.CONFIG_DIR / "memory.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE records(value)")
    observed = []
    original = backups.apply_pragmas
    def inspect(connection):
        refused = original(connection)
        observed.append(connection.execute("PRAGMA cache_size").fetchone()[0])
        return refused
    monkeypatch.setattr(backups, "apply_pragmas", inspect)
    backups.BackupManager._snapshot_sqlite(database, tmp_path / "snapshot.db")
    assert observed == [-4000, -4000]
    with sqlite3.connect(database) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
