"""Workspace restoration must retain actual launcher outcomes and scope."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.execution_policy import background_scope, current_scope
from orion_core.workspace import WorkspaceManager, WorkspaceSnapshot, WindowState


def _manager(outcomes, *, windows=None):
    snapshot = WorkspaceSnapshot(
        at="now", active_window="", active_project="saved-project",
        windows=windows if windows is not None else [WindowState("main.py - Code", 0, 0, 400, 300)],
    )
    changed = []
    calls = []
    memory = SimpleNamespace(
        active_project="current-project",
        recall=lambda *args: [{"key_ref": "coding", "value": snapshot.to_json()}],
        set_active_project=changed.append,
    )
    def open_app(name):
        calls.append((name, current_scope()))
        outcome = outcomes[name]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *args: None))
    return WorkspaceManager(bus, memory, SimpleNamespace(open_app=open_app)), calls, changed


async def test_background_restore_defers_before_launch_or_project_change():
    manager, calls, changed = _manager({"code": ToolResult("Opened Code")})
    with background_scope("test"):
        result = await manager.restore_workspace_state("coding")
    assert not result.ok
    assert result.receipt["status"] == "requires_foreground"
    assert result.receipt["effect"] == "workspace_restore"
    assert calls == changed == []


@pytest.mark.parametrize("outcome", [
    ToolResult("Application unavailable", ok=False),
    ToolResult("Needs foreground", ok=False, receipt={"status": "requires_foreground"}),
    RuntimeError("launch failed"),
    None,
])
async def test_failed_launch_is_never_reported_as_reopened(outcome):
    manager, calls, changed = _manager({"code": outcome})
    result = await manager.restore_workspace_state("coding")
    assert not result.ok
    assert result.receipt["status"] == "failed"
    assert result.receipt["reopened"] == []
    assert result.receipt["failures"][0]["app"] == "code"
    assert "Relaunched: code" not in result.text
    assert [name for name, _ in calls] == ["code"]
    assert changed == []


async def test_partial_restore_reports_only_successful_unique_apps():
    manager, calls, changed = _manager(
        {"code": ToolResult("Opened Code"), "word": ToolResult("Word unavailable", ok=False)},
        windows=[WindowState(title, 0, 0, 400, 300) for title in
                 ("main.py - Code", "other.py - Code", "report.docx - Word")],
    )
    result = await manager.restore_workspace_state("coding")
    assert not result.ok
    assert result.receipt["status"] == "partial"
    assert result.receipt["reopened"] == ["code"]
    assert [name for name, _ in calls] == ["code", "word"]
    assert "Word unavailable" in result.text
    assert changed == []


async def test_successful_restore_updates_project_and_does_not_block_event_loop():
    manager, calls, changed = _manager({"code": ToolResult("Opened Code")})
    owner = threading.get_ident()
    threads = []
    original = manager.desktop.open_app
    def open_app(name):
        threads.append(threading.get_ident())
        return original(name)
    manager.desktop.open_app = open_app
    result = await manager.restore_workspace_state("coding")
    assert result.ok
    assert result.receipt["reopened"] == ["code"]
    assert changed == ["saved-project"]
    assert len(threads) == 1 and threads[0] != owner


async def test_snapshot_without_launchable_apps_still_restores_project_in_background():
    manager, calls, changed = _manager({}, windows=[])
    with background_scope("test"):
        result = await manager.restore_workspace_state("coding")
    assert result.ok
    assert result.receipt["reopened"] == []
    assert calls == []
    assert changed == ["saved-project"]
