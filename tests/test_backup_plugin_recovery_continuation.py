"""Isolated recovery boundaries, using only synthetic files and observers."""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import orion_core.backup_manager as backups
from orion_core.data import ToolResult
from orion_core.plugin_manifest import discover_manifests
from orion_core.plugin_runner import PluginScheduler


@pytest.mark.parametrize("bad_json", ["[" * 10000 + "0" + "]" * 10000,
                                      '{"number":' + "9" * 5000 + "}"],
                         ids=["nesting_limit", "integer_limit"])
def test_invalid_manifest_parser_limits_do_not_hide_valid_neighbours(tmp_path, bad_json):
    (tmp_path / "broken.plugin.json").write_text(bad_json, encoding="utf-8")
    (tmp_path / "working.plugin.json").write_text(
        json.dumps({"name": "working", "module": "working_tool.py"}), encoding="utf-8")
    manifests, errors = discover_manifests(tmp_path)
    assert [manifest.name for manifest in manifests] == ["working"]
    assert "broken.plugin.json" in errors


@pytest.fixture
def manager(tmp_path, monkeypatch):
    base = tmp_path / "app"
    config = base / "config"
    config.mkdir(parents=True)
    monkeypatch.setattr(backups, "BASE_DIR", base)
    monkeypatch.setattr(backups, "CONFIG_DIR", config)
    bus = SimpleNamespace(dashboard_event=SimpleNamespace(emit=lambda *_: None))
    return backups.BackupManager(bus, destination=tmp_path / "archives")


def test_published_backup_keeps_its_receipt_when_dashboard_observer_fails(manager):
    def fail(*_):
        raise RuntimeError("synthetic dashboard observer failure")
    manager.bus.dashboard_event.emit = fail
    result = manager._backup_sync("complete")
    assert result.ok
    assert len(list(manager.destination.glob("*.zip"))) == 1
    assert result.evidence["archive"]


def test_failed_retention_keeps_the_published_archive_and_reports_the_limitation(manager, monkeypatch):
    assert manager._backup_sync("previous").ok
    previous = next(manager.destination.glob("*.zip"))
    before = previous.read_bytes()
    def fail():
        raise PermissionError("synthetic unavailable retention listing")
    monkeypatch.setattr(manager, "_prune", fail)
    result = manager._backup_sync("new")
    assert result.ok and "retention" in result.text.lower()
    assert result.evidence["retention_error"]
    assert previous.read_bytes() == before
    assert len(list(manager.destination.glob("*.zip"))) == 2


def test_failed_dashboard_observer_is_logged_once_without_chat_spam(manager, caplog):
    def fail(*_):
        raise RuntimeError("synthetic dashboard observer failure")
    manager.bus.dashboard_event.emit = fail
    assert manager._backup_sync("first").ok
    assert manager._backup_sync("second").ok
    assert sum("dashboard could not finish" in record.message for record in caplog.records) == 1


def test_existing_empty_backup_location_is_distinct_from_unavailable(manager):
    assert manager.list_backups().ok
    assert "No backups yet" in manager.list_backups().text
    assert not manager._restore_sync("").ok
    assert "no backups to restore" in manager._restore_sync("").text.lower()


@pytest.mark.parametrize("operation", ["list", "restore"])
def test_unreadable_backup_directory_is_not_reported_as_empty(manager, monkeypatch, operation):
    assert manager._backup_sync("saved").ok
    before = next(manager.destination.glob("*.zip")).read_bytes()
    original = os.scandir
    def fail(path):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(manager.destination)):
            raise PermissionError("synthetic unavailable backup location")
        return original(path)
    monkeypatch.setattr(os, "scandir", fail)
    result = manager.list_backups() if operation == "list" else manager._restore_sync("")
    assert not result.ok
    assert "no backups" not in result.text.lower()
    assert "no backups to restore" not in result.text.lower()
    monkeypatch.setattr(os, "scandir", original)
    assert next(manager.destination.glob("*.zip")).read_bytes() == before


async def test_duplicate_supervisor_start_cannot_run_the_same_plugin_twice():
    started = asyncio.Event()
    release = asyncio.Event()
    starts = []
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    async def dispatch(name, arguments):
        starts.append(name)
        started.set()
        await release.wait()
        return ToolResult("done")
    schedule = SimpleNamespace(next_due=lambda _: now + timedelta(minutes=1))
    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), clock=lambda: now)
    # The production entry type and run loop are unchanged by this transport.
    from orion_core.plugin_runner import Entry
    scheduler.entries["working"] = Entry("working", schedule, next_due=now)
    first = asyncio.create_task(scheduler.run())
    duplicate = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        duplicate = asyncio.create_task(scheduler.run())
        # Let the second run reach its first await without sleeping on wall time.
        for _ in range(5):
            await asyncio.sleep(0)
        assert starts == ["working"]
        assert scheduler.running
    finally:
        scheduler.stop()
        release.set()
        tasks = [first] + ([duplicate] if duplicate is not None else [])
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

