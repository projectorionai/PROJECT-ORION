"""
BackupManager (audit 2026-10-03).

Pins the four properties the audit found missing:

* a backup works when the config directory is NOT inside the project — the
  recommended setup after ``tools/move_config.py`` — where every backup used to
  fail with "is not in the subpath of";
* a failed backup leaves no truncated archive for list/restore to offer;
* SQLite stores are archived as consistent snapshots (the backup API), never as
  raw .db/-wal/-shm files a running ORION is writing to;
* credentials and logged-in sessions stay out of an archive whose default home
  is a cloud-synced folder, unless ORION_BACKUP_SECRETS=1 says otherwise.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import backup_manager as bm  # noqa: E402
from orion_core.backup_manager import BackupManager, classify  # noqa: E402


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _Bus:
    def __init__(self):
        self.dashboard_event = _Signal()
        self.log = _Signal()


def _config(tmp_path: Path) -> Path:
    """A config directory OUTSIDE the project, as move_config.py leaves it."""
    config = tmp_path / "elsewhere" / "orion-data"
    config.mkdir(parents=True)
    (config / "missions.json").write_text('{"missions": []}', encoding="utf-8")
    (config / "api_keys.json").write_text('{"gemini": "AIza-secret"}', encoding="utf-8")
    (config / "remote_secret.key").write_text("ab" * 32, encoding="utf-8")
    (config / "mcp_servers.json").write_text('{"github": {"env": {"T": "x"}}}', encoding="utf-8")
    (config / "remote_tls").mkdir()
    (config / "remote_tls" / "key.pem").write_text("-----BEGIN PRIVATE KEY-----", encoding="utf-8")
    (config / "browser_profile").mkdir()
    (config / "browser_profile" / "Cookies").write_text("session", encoding="utf-8")
    (config / "tts_cache").mkdir()
    (config / "tts_cache" / "clip.pcm").write_bytes(b"\0" * 64)
    db = sqlite3.connect(config / "orion_core.db")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE facts (v TEXT)")
    db.executemany("INSERT INTO facts VALUES (?)", [(f"fact {i}",) for i in range(50)])
    db.commit()
    db.close()
    return config


def _manager(tmp_path: Path, config: Path) -> BackupManager:
    return BackupManager(_Bus(), destination=tmp_path / "dest", config_dir=config)


def _names(archive: Path) -> set[str]:
    with zipfile.ZipFile(archive) as zf:
        return set(zf.namelist())


def test_backup_works_with_a_config_directory_outside_the_project(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_BACKUP_SECRETS", raising=False)
    config = _config(tmp_path)
    manager = _manager(tmp_path, config)

    result = asyncio.run(manager.backup("before the move"))

    assert result.ok, result.text
    archives = list((tmp_path / "dest").glob("orion_backup_*.zip"))
    assert len(archives) == 1
    names = _names(archives[0])
    assert "config/missions.json" in names
    assert "config/orion_core.db" in names
    assert "BACKUP_NOTE.txt" in names


def test_credentials_and_sessions_are_withheld_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_BACKUP_SECRETS", raising=False)
    config = _config(tmp_path)
    result = asyncio.run(_manager(tmp_path, config).backup())

    archive = next((tmp_path / "dest").glob("orion_backup_*.zip"))
    names = _names(archive)
    for secret in ("config/api_keys.json", "config/remote_secret.key",
                   "config/mcp_servers.json", "config/remote_tls/key.pem",
                   "config/browser_profile/Cookies"):
        assert secret not in names, secret
    with zipfile.ZipFile(archive) as zf:
        manifest = zf.read("WITHHELD_SECRETS.txt").decode("utf-8")
        assert "api_keys.json" in manifest
        # No credential VALUE travels in the archive at all.
        for name in zf.namelist():
            assert b"AIza-secret" not in zf.read(name)
    assert "left out" in result.text


def test_secrets_are_included_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_BACKUP_SECRETS", "1")
    config = _config(tmp_path)
    asyncio.run(_manager(tmp_path, config).backup())
    names = _names(next((tmp_path / "dest").glob("orion_backup_*.zip")))
    assert "config/api_keys.json" in names
    assert "WITHHELD_SECRETS.txt" not in names


def test_caches_and_sqlite_sidecars_are_not_archived(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_BACKUP_SECRETS", raising=False)
    config = _config(tmp_path)
    # Hold a writer open so the -wal and -shm siblings exist during the backup.
    live = sqlite3.connect(config / "orion_core.db")
    live.execute("INSERT INTO facts VALUES ('written while backing up')")
    live.commit()
    try:
        assert (config / "orion_core.db-wal").exists()
        asyncio.run(_manager(tmp_path, config).backup())
    finally:
        live.close()
    names = _names(next((tmp_path / "dest").glob("orion_backup_*.zip")))
    assert not any(n.endswith(("-wal", "-shm")) for n in names)
    assert not any("tts_cache" in n for n in names)


def test_the_archived_database_is_a_consistent_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_BACKUP_SECRETS", raising=False)
    config = _config(tmp_path)
    live = sqlite3.connect(config / "orion_core.db")
    live.execute("INSERT INTO facts VALUES ('only in the WAL so far')")
    live.commit()                       # committed to the WAL, not checkpointed
    try:
        asyncio.run(_manager(tmp_path, config).backup())
    finally:
        live.close()
    archive = next((tmp_path / "dest").glob("orion_backup_*.zip"))
    out = tmp_path / "restored.db"
    with zipfile.ZipFile(archive) as zf:
        out.write_bytes(zf.read("config/orion_core.db"))
    restored = sqlite3.connect(out)
    try:
        assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        rows = restored.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        # The raw .db alone would hold 50; the WAL row arrives only through
        # the backup API.
        assert rows == 51
    finally:
        restored.close()


def test_a_failed_backup_leaves_no_partial_archive(tmp_path, monkeypatch):
    config = _config(tmp_path)
    manager = _manager(tmp_path, config)

    def fail_rename(*_a, **_k):
        raise OSError("destination vanished")

    monkeypatch.setattr(bm.os, "replace", fail_rename)
    result = asyncio.run(manager.backup())

    assert not result.ok
    leftovers = list((tmp_path / "dest").iterdir())
    assert leftovers == [], leftovers


def test_a_backup_that_could_read_nothing_is_a_failure(tmp_path, monkeypatch):
    config = _config(tmp_path)
    manager = _manager(tmp_path, config)

    def unreadable(*_a, **_k):
        raise OSError("locked")

    monkeypatch.setattr(bm, "_snapshot_sqlite", unreadable)
    monkeypatch.setattr(bm.zipfile.ZipFile, "write", unreadable)
    result = asyncio.run(manager.backup())

    assert not result.ok
    assert "could be read" in result.text
    assert list((tmp_path / "dest").iterdir()) == []


@pytest.mark.parametrize("relative, expected", [
    ("missions.json", "keep"),
    ("orion_core.db", "keep"),
    ("orion_core.db-wal", "skip"),
    ("orion_core.db-shm", "skip"),
    ("api_keys.json", "secret"),
    ("telephony.json", "secret"),
    ("messaging.json", "secret"),
    ("plugin_vault.json", "secret"),
    ("plugin_vault.key", "secret"),
    ("remote_tls/key.pem", "secret"),
    ("social_profile/Default/Cookies", "secret"),
    ("models/bge-small-en-v1.5/model.onnx", "skip"),
    ("knowledge_corpus/shard_001.md", "skip"),
    ("knowledge_corpus/manifest.json", "keep"),
])
def test_classification(relative, expected):
    assert classify(Path(relative)) == expected
