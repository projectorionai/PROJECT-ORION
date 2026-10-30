"""A complete backup receipt must not hide enumeration or diagnostics failures."""
import os
from types import SimpleNamespace

import pytest

import orion_core.backup_manager as backups


@pytest.fixture
def manager(tmp_path, monkeypatch):
    base = tmp_path / "app"
    config = base / "config"
    config.mkdir(parents=True)
    monkeypatch.setattr(backups, "BASE_DIR", base)
    monkeypatch.setattr(backups, "CONFIG_DIR", config)
    bus = SimpleNamespace(dashboard_event=SimpleNamespace(emit=lambda *_: None))
    return backups.BackupManager(bus, destination=tmp_path / "archives")


def test_missing_config_is_a_failed_backup(manager):
    backups.CONFIG_DIR.rmdir()
    assert not manager._backup_sync("").ok
    assert not list(manager.destination.glob("*.zip"))


def test_unreadable_nested_folder_does_not_publish_an_incomplete_backup(manager, monkeypatch):
    assert manager._backup_sync("previous").ok
    previous = list(manager.destination.glob("*.zip"))
    hidden = backups.CONFIG_DIR / "nested"
    hidden.mkdir()
    (hidden / "important.json").write_text("important", encoding="utf-8")
    original = os.scandir
    def fail(path):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(hidden)):
            raise PermissionError("synthetic unreadable subfolder")
        return original(path)
    monkeypatch.setattr(os, "scandir", fail)
    result = manager._backup_sync("")
    assert not result.ok
    assert list(manager.destination.glob("*.zip")) == previous


def test_diagnostics_failure_does_not_turn_a_completed_backup_into_an_exception(manager):
    def fail(*_):
        raise RuntimeError("synthetic metrics failure")
    manager.telemetry = SimpleNamespace(metrics=SimpleNamespace(incr=fail, gauge=fail))
    assert manager._backup_sync("complete").ok
    assert len(list(manager.destination.glob("*.zip"))) == 1
