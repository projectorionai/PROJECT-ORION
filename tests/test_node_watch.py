"""
Unattended-node care (brief §4.2): an external heartbeat and scheduled backups.

Neither existed: nothing watched a headless node's /api/health, and nothing
backed up its SQLite stores.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult  # noqa: E402
from orion_core.node_watch import HealthPinger, ScheduledBackup  # noqa: E402


class _Log:
    def __init__(self):
        self.lines = []

    def emit(self, line, *rest):
        self.lines.append(line)


class _Bus:
    def __init__(self):
        self.log = _Log()
        self.dashboard_event = _Log()


def _run(coro):
    return asyncio.run(coro)


def test_pinging_is_off_unless_configured(monkeypatch):
    monkeypatch.delenv("ORION_HEALTHCHECK_URL", raising=False)
    assert HealthPinger.from_env(_Bus()) is None
    monkeypatch.setenv("ORION_HEALTHCHECK_URL", "file:///etc/passwd")
    assert HealthPinger.from_env(_Bus()) is None          # only http(s)


def test_the_interval_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ORION_HEALTHCHECK_URL", "https://hc-ping.com/abc")
    monkeypatch.setenv("ORION_HEALTHCHECK_MINUTES", "2")
    assert HealthPinger.from_env(_Bus()).interval == 120.0
    monkeypatch.setenv("ORION_HEALTHCHECK_MINUTES", "0.01")
    assert HealthPinger.from_env(_Bus()).interval == 60.0  # floor of one minute


def test_a_failing_monitor_is_reported_once_and_its_recovery_once():
    bus = _Bus()
    answers = iter([200, 500, 500, "boom", 200, 200])

    async def fetch(_url):
        answer = next(answers)
        if answer == "boom":
            raise OSError("connection refused")
        return answer

    pinger = HealthPinger("https://hc-ping.com/abc", bus, fetch=fetch)
    for _ in range(6):
        _run(pinger.step())
    failures = [line for line in bus.log.lines if "failed" in line]
    recoveries = [line for line in bus.log.lines if "again" in line]
    assert len(failures) == 1 and "HTTP 500" in failures[0]
    assert len(recoveries) == 1
    assert pinger.pings == 3


def test_the_loop_pings_until_stopped():
    calls = []

    async def fetch(url):
        calls.append(url)
        return 200

    async def flow():
        pinger = HealthPinger("https://hc-ping.com/abc", _Bus(), interval=0.01, fetch=fetch)
        task = asyncio.create_task(pinger.run())
        await asyncio.sleep(0.08)
        pinger.stop()
        await asyncio.wait_for(task, 1)

    _run(flow())
    assert len(calls) >= 3


def test_scheduled_backups_are_off_unless_configured(monkeypatch):
    monkeypatch.delenv("ORION_NODE_BACKUP_DIR", raising=False)
    assert ScheduledBackup.from_env(_Bus()) is None


def test_scheduled_backups_use_the_configured_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("ORION_NODE_BACKUP_DIR", str(tmp_path / "node-backups"))
    monkeypatch.setenv("ORION_NODE_BACKUP_HOURS", "6")
    watcher = ScheduledBackup.from_env(_Bus())
    assert watcher.interval == 6 * 3600
    assert watcher.manager.destination == tmp_path / "node-backups"


def test_a_backup_failure_is_logged_once_until_it_recovers():
    bus = _Bus()
    outcomes = iter([ToolResult("Backup failed: disk full", ok=False),
                     ToolResult("Backup failed: disk full", ok=False),
                     ToolResult("Backup complete — 12 file(s)")])

    class _Manager:
        async def backup(self, note=""):
            return next(outcomes)

    watcher = ScheduledBackup(_Manager(), bus, interval=3600, first_delay=0)
    for _ in range(3):
        _run(watcher.step())
    assert sum("failed" in line for line in bus.log.lines) == 1
    assert any("working again" in line for line in bus.log.lines)
    assert watcher.made == 1


def test_a_real_backup_lands_in_the_folder(monkeypatch, tmp_path):
    from orion_core.backup_manager import BackupManager

    config = tmp_path / "config"
    config.mkdir()
    (config / "missions.json").write_text("{}", encoding="utf-8")
    manager = BackupManager(_Bus(), destination=tmp_path / "out", config_dir=config)
    watcher = ScheduledBackup(manager, _Bus(), interval=3600, first_delay=0)
    _run(watcher.step())
    assert len(list((tmp_path / "out").glob("orion_backup_*.zip"))) == 1
