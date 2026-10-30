"""A saved disable must stop future work and a failed save must retain hooks."""
from datetime import datetime, timezone
import asyncio
import threading
from types import SimpleNamespace

import pytest

from orion_core import plugin_registry as module
from orion_core.data import ToolResult
from orion_core.dispatch_productivity import ProductivityDispatchMixin
from orion_core.plugin_runner import PluginScheduler, Entry
from orion_core.dispatcher import OrionDispatcher
from orion_core.plugin_events import PluginEventBridge
from orion_core.bus import OrionBus


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    value = module.PluginRegistry(directory=tmp_path / "tools")
    assert value.create("demo", "Synthetic scheduled plugin", "confirm").ok
    return value


async def test_disabling_a_plugin_stops_future_scheduled_actions(registry):
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    calls = []
    async def dispatch(name, args):
        calls.append(name)
        return ToolResult("synthetic action")
    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), clock=lambda: now)
    scheduler.entries["demo"] = Entry("demo", SimpleNamespace(next_due=lambda _: now), next_due=now)
    context = SimpleNamespace(plugins=registry, plugin_scheduler=scheduler, plugin_events=None)
    result = await ProductivityDispatchMixin.plugin_tool(context, {"action": "disable", "name": "demo"})
    assert result.ok and not registry.is_enabled("demo")
    await scheduler._tick()
    assert calls == []


async def test_failed_disable_save_does_not_disconnect_a_working_hook(registry, monkeypatch):
    forgotten = []
    monkeypatch.setattr(registry, "_save_state", lambda: False)
    context = SimpleNamespace(plugins=registry, plugin_events=SimpleNamespace(forget=forgotten.append))
    result = await ProductivityDispatchMixin.plugin_tool(context, {"action": "disable", "name": "demo"})
    assert not result.ok and registry.is_enabled("demo")
    assert forgotten == []


@pytest.mark.parametrize("confirm_mode", ["1", "0"])
def test_loaded_disabled_plugin_cannot_pass_the_dispatch_gate(registry, monkeypatch, confirm_mode):
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", confirm_mode)
    assert registry.set_enabled("demo", False).ok
    context = SimpleNamespace(plugins=registry)
    blocked = OrionDispatcher._plugin_gate(context, "demo", {})
    assert blocked is not None and not blocked.ok


async def test_bound_schedule_resumes_once_after_reenable(registry):
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    calls = []
    async def dispatch(name, _):
        calls.append(name)
        return ToolResult("synthetic")
    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), clock=lambda: now, registry=registry)
    scheduler.entries["demo"] = Entry("demo", SimpleNamespace(next_due=lambda _: None), next_due=now)
    assert registry.set_enabled("demo", False).ok
    await scheduler._tick()
    assert calls == []
    assert registry.set_enabled("demo", True).ok
    await scheduler._tick()
    await scheduler._tick()
    assert calls == ["demo"]


def test_shared_registry_blocks_and_resumes_real_bus_delivery(registry):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    bus = OrionBus()
    delivered = []
    bridge = PluginEventBridge(bus=bus, registry=registry)
    bridge.subscribe("demo", SimpleNamespace(on_event=lambda *args: delivered.append(args)), ["state"])
    bus.state.emit("before")
    app.processEvents()
    assert registry.set_enabled("demo", False).ok
    bus.state.emit("held")
    app.processEvents()
    assert registry.set_enabled("demo", True).ok
    bus.state.emit("after")
    app.processEvents()
    assert delivered == [("state", "before"), ("state", "after")]
    bridge.forget("demo")
    assert app is not None


def test_panel_disable_uses_the_same_live_event_preference_gate(registry):
    from PyQt6.QtWidgets import QApplication
    from orion_core.gui.plugin_deck import PluginDeckView
    app = QApplication.instance() or QApplication([])
    bus = OrionBus()
    delivered = []
    bridge = PluginEventBridge(bus=bus, registry=registry)
    bridge.subscribe("demo", SimpleNamespace(on_event=lambda *args: delivered.append(args)), ["state"])
    view = PluginDeckView(bus, registry)
    try:
        view._toggle("demo", False)
        bus.state.emit("held")
        app.processEvents()
        view._toggle("demo", True)
        bus.state.emit("resumed")
        app.processEvents()
        assert delivered == [("state", "resumed")]
    finally:
        bridge.forget("demo")
        view._timer.stop()
        view.deleteLater()
    assert app is not None


async def test_pending_failed_save_does_not_publish_a_disable_to_live_readers(registry, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def save():
        entered.set()
        assert release.wait(timeout=3)
        return False
    monkeypatch.setattr(registry, "_save_state", save)
    task = asyncio.create_task(asyncio.to_thread(registry.set_enabled, "demo", False))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert registry.is_enabled("demo")
        assert OrionDispatcher._plugin_gate(SimpleNamespace(plugins=registry), "demo", {}) is None
    finally:
        release.set()
        assert not (await task).ok
    assert registry.is_enabled("demo")


async def test_unavailable_saved_preferences_hold_event_and_schedule_work(registry):
    module.PLUGIN_STATE_PATH.write_text("broken", encoding="utf-8")
    registry._load_state()
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    calls = []
    async def dispatch(name, _):
        calls.append(name)
        return ToolResult("synthetic")
    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), clock=lambda: now, registry=registry)
    scheduler.entries["demo"] = Entry("demo", SimpleNamespace(next_due=lambda _: now), next_due=now)
    bridge = PluginEventBridge(registry=registry)
    bridge._handlers["demo"] = lambda *_: calls.append("hook")
    assert not bridge.dispatch("demo", "state", "held")
    await scheduler._tick()
    assert calls == []


async def test_hot_reload_passes_the_live_scheduler(registry, monkeypatch):
    import orion_core.plugin_manifest as manifests
    received = []
    async def load(*args, **kwargs):
        received.append(kwargs)
        return manifests.LoadPlan()
    monkeypatch.setattr(manifests, "load_plugins", load)
    scheduler = object()
    context = SimpleNamespace(plugins=registry, forge=SimpleNamespace(loader=object()),
                              bus=SimpleNamespace(), plugin_events=None, plugin_scheduler=scheduler)
    result = await ProductivityDispatchMixin._reload_plugins(context)
    assert result.ok
    assert received[0]["scheduler"] is scheduler


async def test_disable_during_an_earlier_run_holds_the_queued_action(registry):
    assert registry.create("second", "Synthetic second plugin", "confirm").ok
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    calls = []

    async def dispatch(name, _):
        calls.append(name)
        assert registry.set_enabled("second", False).ok
        await asyncio.sleep(0)
        return ToolResult("synthetic")

    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), clock=lambda: now, registry=registry)
    schedule = SimpleNamespace(next_due=lambda _: None)
    for name in ("demo", "second"):
        scheduler.entries[name] = Entry(name, schedule, next_due=now)
    await scheduler._tick()
    assert calls == ["demo"]
    assert scheduler.entries["second"].runs == 0


async def test_preference_reader_failure_holds_all_future_work(monkeypatch):
    def unavailable(_):
        raise OSError("synthetic preference reader failure")

    registry = SimpleNamespace(is_enabled=unavailable)
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    calls = []

    async def dispatch(name, _):
        calls.append(name)
        return ToolResult("synthetic")

    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), clock=lambda: now, registry=registry)
    scheduler.entries["demo"] = Entry("demo", SimpleNamespace(next_due=lambda _: now), next_due=now)
    bridge = PluginEventBridge(registry=registry)
    bridge._handlers["demo"] = lambda *_: calls.append("hook")
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", "0")
    assert not bridge.dispatch("demo", "state", "held")
    await scheduler._tick()
    result = OrionDispatcher._plugin_gate(SimpleNamespace(plugins=registry), "demo", {})
    assert result is not None and not result.ok
    assert calls == []
