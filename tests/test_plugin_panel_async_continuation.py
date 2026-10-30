"""The Plugins page's blocking registry work must leave the GUI loop."""
import asyncio
import threading
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.gui.plugin_deck import PluginDeckView


@pytest.mark.parametrize("operation", ["refresh", "toggle", "doctor", "backfill"])
async def test_panel_registry_io_does_not_run_on_the_gui_thread(operation):
    app = QApplication.instance() or QApplication([])
    owner = threading.get_ident()
    calls = []

    def snapshot():
        calls.append(threading.get_ident())
        return {"total": 0, "enabled": 0, "healthy": 0, "plugins": [], "state_error": ""}

    def action(*_):
        calls.append(threading.get_ident())
        return ToolResult("synthetic receipt")

    def manifests():
        calls.append(threading.get_ident())
        return []

    registry = SimpleNamespace(snapshot=snapshot, set_enabled=action, doctor=action,
                               backfill_manifests=manifests, manifestless=manifests)
    view = PluginDeckView(OrionBus(), registry)
    try:
        await asyncio.sleep(0)
        calls.clear()
        if operation == "toggle":
            view._toggle("demo", False)
        else:
            getattr(view, "_" + operation)()
        await asyncio.sleep(0)
        for attribute in ("_action_task", "_refresh_task"):
            task = getattr(view, attribute, None)
            if task is not None:
                await task
        assert calls and owner not in calls
    finally:
        for attribute in ("_action_task", "_refresh_task"):
            task = getattr(view, attribute, None)
            if task is not None:
                await task
        view._timer.stop()
        view.deleteLater()
    assert app is not None


@pytest.mark.parametrize("delete_before_finish", [False, True])
async def test_pending_panel_action_cannot_duplicate_or_touch_a_deleted_widget(delete_before_finish):
    from PyQt6 import sip
    app = QApplication.instance() or QApplication([])
    entered, release = threading.Event(), threading.Event()
    calls = []

    def action(*_):
        calls.append("save")
        entered.set()
        assert release.wait(3)
        return ToolResult("Synthetic saved <resource> & result")

    registry = SimpleNamespace(snapshot=lambda: {"total": 0, "plugins": [], "state_error": ""}, set_enabled=action)
    view = PluginDeckView(OrionBus(), registry)
    await view._refresh_task
    try:
        view._toggle("demo", False)
        assert await asyncio.to_thread(entered.wait, 2)
        view._toggle("demo", False)
        assert not any(button.isEnabled() for button in view._controls)
        if delete_before_finish:
            sip.delete(view)
    finally:
        release.set()
        await view._action_task
        if not sip.isdeleted(view):
            if view._refresh_task is not None:
                await view._refresh_task
            assert view.report.text() == "Synthetic saved <resource> & result"
            from PyQt6.QtCore import Qt
            assert view.report.textFormat() == Qt.TextFormat.PlainText
            view._timer.stop()
            view.deleteLater()
    assert calls == ["save"]
    assert app is not None


async def test_a_late_snapshot_cannot_repaint_the_previous_registry():
    app = QApplication.instance() or QApplication([])
    entered, release = threading.Event(), threading.Event()

    def old_snapshot():
        entered.set()
        assert release.wait(3)
        return {"total": 99, "plugins": [], "state_error": ""}

    view = PluginDeckView(OrionBus(), SimpleNamespace(snapshot=old_snapshot))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        view.attach_registry(SimpleNamespace(snapshot=lambda: {"total": 2, "plugins": [], "state_error": ""}))
    finally:
        release.set()
        await view._refresh_task
        assert "2 installed" in view.summary.text()
        view._timer.stop()
        view.deleteLater()
    assert app is not None
