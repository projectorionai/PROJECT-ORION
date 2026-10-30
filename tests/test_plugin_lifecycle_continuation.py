"""Synthetic reproduction of uninstall receipts and removed automatic hooks."""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import plugin_registry as module
from orion_core.data import ToolResult
from orion_core.dispatch_productivity import ProductivityDispatchMixin
from orion_core.dispatcher import OrionDispatcher
from orion_core.plugin_events import PluginEventBridge
from orion_core.plugin_manifest import load_plugins
from orion_core.plugin_runner import Entry, PluginScheduler


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    registry = module.PluginRegistry(directory=tmp_path / "tools")
    assert registry.create("demo", "Synthetic lifecycle plugin", "confirm").ok
    return registry


def _contents(registry):
    return {path.name: path.read_bytes() for path in registry.directory.iterdir() if path.is_file()}


def test_a_second_removal_failure_preserves_the_whole_working_plugin(registry, monkeypatch):
    before = _contents(registry)
    state = module.PLUGIN_STATE_PATH.read_bytes()
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == registry.directory / "demo.plugin.json":
            raise PermissionError("synthetic locked manifest")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    result = registry.remove("demo")
    assert not result.ok
    assert _contents(registry) == before
    assert module.PLUGIN_STATE_PATH.read_bytes() == state


def test_failed_removal_preference_save_preserves_the_working_plugin(registry, monkeypatch):
    before = _contents(registry)
    state = module.PLUGIN_STATE_PATH.read_bytes()
    monkeypatch.setattr(registry, "_save_state", lambda: False)
    result = registry.remove("demo")
    assert not result.ok
    assert _contents(registry) == before
    assert module.PLUGIN_STATE_PATH.read_bytes() == state
    assert registry.is_enabled("demo")


def test_successfully_removed_plugin_cannot_run_a_cached_handler(registry, monkeypatch):
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", "0")
    assert registry.remove("demo").ok
    result = OrionDispatcher._plugin_gate(SimpleNamespace(plugins=registry), "demo", {})
    assert result is not None and not result.ok


async def test_failed_uninstall_keeps_the_working_event_hook(registry):
    module.PLUGIN_STATE_PATH.write_text("broken", encoding="utf-8")
    registry._load_state()
    forgotten = []
    context = SimpleNamespace(plugins=registry, plugin_events=SimpleNamespace(forget=forgotten.append))
    result = await ProductivityDispatchMixin.plugin_tool(context, {"action": "remove", "name": "demo"})
    assert not result.ok
    assert forgotten == []


def test_failed_save_readback_does_not_leave_opposite_restart_preferences(registry, monkeypatch):
    state = module.PLUGIN_STATE_PATH
    before = state.read_bytes()
    original = Path.read_text
    refused = False

    def read(path, *args, **kwargs):
        nonlocal refused
        value = original(path, *args, **kwargs)
        if path == state and '"demo"' in value and json.loads(value).get("disabled") == ["demo"] and not refused:
            refused = True
            raise PermissionError("synthetic readback lock after successful replacement")
        return value

    monkeypatch.setattr(Path, "read_text", read)
    result = registry.set_enabled("demo", False)
    assert not result.ok
    assert state.read_bytes() == before
    assert registry.is_enabled("demo")
    assert module.PluginRegistry(directory=registry.directory).is_enabled("demo")


def test_failed_preference_recovery_retains_prior_bytes_and_holds_calls(registry, monkeypatch):
    state = module.PLUGIN_STATE_PATH
    before = state.read_bytes()
    original = Path.read_text
    def read(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == state and json.loads(value).get("disabled") == ["demo"]:
            raise PermissionError("synthetic readback lock")
        return value
    monkeypatch.setattr(Path, "read_text", read)
    def locked(*_):
        raise PermissionError("synthetic recovery lock")
    monkeypatch.setattr(module, "atomic_write_bytes", locked)
    result = registry.set_enabled("demo", False)
    assert not result.ok and registry.state_error
    assert not registry.is_enabled("demo")
    backups = list(state.parent.glob("*.recovery.tmp"))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    assert str(backups[0]) in result.text


def test_failed_removal_rollback_keeps_original_files_and_holds_plugin(registry, monkeypatch):
    before = _contents(registry)
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path == registry.directory / "demo.plugin.json":
            raise PermissionError("synthetic manifest lock")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    def locked(*_):
        raise PermissionError("synthetic module restore lock")
    monkeypatch.setattr(module, "_replace", locked)
    result = registry.remove("demo")
    assert not result.ok and "rollback could not finish" in result.text.lower()
    recovery = next(registry.directory.glob(".remove-recovery-*"))
    assert (recovery / "demo_tool.py").read_bytes() == before["demo_tool.py"]
    assert str(recovery) in result.text
    assert not registry.is_enabled("demo")
    assert not registry.set_enabled("demo", True).ok


async def test_successful_uninstall_retires_future_runtime_registrations(registry):
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    delivered = []
    bridge = PluginEventBridge(registry=registry)
    bridge._handlers["demo"] = lambda *_: delivered.append("event")
    scheduler = PluginScheduler(registry=registry, clock=lambda: now)
    scheduler.entries["demo"] = Entry("demo", SimpleNamespace(next_due=lambda _: None), next_due=now)
    context = SimpleNamespace(plugins=registry, plugin_events=bridge, plugin_scheduler=scheduler,
                              _tool_handlers={"demo": lambda: "synthetic"},
                              _forge_rollbacks={"demo": object()},
                              TOOL_DECLARATIONS=[{"name": "demo"}, {"name": "native"}])
    result = await ProductivityDispatchMixin.plugin_tool(context, {"action": "remove", "name": "demo"})
    assert result.ok
    assert not bridge.dispatch("demo", "state", "held")
    assert "demo" not in scheduler.entries and "demo" not in context._tool_handlers
    assert "demo" not in context._forge_rollbacks
    assert context.TOOL_DECLARATIONS == [{"name": "native"}]
    assert delivered == []
    assert registry.create("demo", "Synthetic reinstall", "confirm").ok
    assert not registry.is_enabled("demo")
    assert registry.set_enabled("demo", True).ok


async def test_install_and_remove_cannot_interleave_a_publication(registry, tmp_path, monkeypatch):
    import asyncio
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "demo_tool.py").write_bytes((registry.directory / "demo_tool.py").read_bytes())
    entered, release, removal_started = threading.Event(), threading.Event(), threading.Event()
    original = module._replace
    def replace(source_path, target):
        if Path(target) == registry.directory.resolve() / "demo.plugin.json" and Path(source_path).parent.name.startswith(".install-stage-"):
            entered.set()
            assert release.wait(3)
        return original(source_path, target)
    monkeypatch.setattr(module, "_replace", replace)
    installing = asyncio.create_task(asyncio.to_thread(registry.install, str(source / "demo_tool.py")))
    removing = None
    def remove():
        removal_started.set()
        return registry.remove("demo")
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        removing = asyncio.create_task(asyncio.to_thread(remove))
        assert await asyncio.to_thread(removal_started.wait, 2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(removing), .25)
    finally:
        release.set()
        assert (await installing).ok
        if removing is not None:
            assert (await removing).ok
    assert registry.scan() == []


@pytest.mark.parametrize("automatic", ["event", "schedule"])
async def test_reloaded_passive_manifest_drops_the_previous_automatic_work(registry, automatic):
    manifest_path = registry.directory / "demo.plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events"] = []
    manifest["interval_seconds"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    delivered = []
    bridge = PluginEventBridge(registry=registry)
    bridge._handlers["demo"] = lambda *_: delivered.append("event")

    async def dispatch(*_):
        delivered.append("schedule")
        return ToolResult("synthetic")

    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), clock=lambda: now, registry=registry)
    scheduler.entries["demo"] = Entry("demo", SimpleNamespace(next_due=lambda _: None), next_due=now)
    outcome = SimpleNamespace(succeeded=True, module_name="demo_tool", tool_schema={"name": "demo"})

    async def load(*_):
        return outcome

    async def activate(*_):
        return ToolResult("active")

    loader = SimpleNamespace(load_and_register=load, activate_tool=activate)
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    plan = await load_plugins(loader, None, registry.directory, bus, registry=registry,
                              events=bridge, scheduler=scheduler)
    assert [item.name for item in plan.loadable] == ["demo"]
    if automatic == "event":
        bridge.dispatch("demo", "state", "synthetic")
    else:
        await scheduler._tick()
    assert delivered == []


@pytest.mark.parametrize("preceding_failures", [0, 2])
async def test_scheduler_cancellation_preserves_capability_health_and_partial_history(preceding_failures):
    import asyncio
    from orion_core.plugin_runner import MUTE_AFTER

    started = asyncio.Event()

    async def dispatch(*_):
        started.set()
        await asyncio.Event().wait()

    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch))
    entry = Entry("demo", SimpleNamespace(), failures=preceding_failures)
    for _ in range(MUTE_AFTER):
        started.clear()
        task = asyncio.create_task(scheduler._execute(entry))
        try:
            await asyncio.wait_for(started.wait(), 1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)

        assert entry.failures == preceding_failures
        assert not entry.muted
        assert entry.last.ok is False
        assert entry.last.finished_at >= entry.last.started_at
        receipt = entry.last.as_dict()
        assert receipt["status"] == "cancelled"
        assert receipt["ok"] is False
        assert "unverified" in receipt["detail"].lower()

    assert entry.runs == MUTE_AFTER
    assert len(scheduler.history) == MUTE_AFTER
    assert all(run.as_dict()["status"] == "cancelled" for run in scheduler.history)


async def test_cancelled_scheduled_run_revokes_its_inherited_secret_scope():
    import asyncio
    from orion_core.plugin_runner import scoped_plugin_secret

    started, released = asyncio.Event(), asyncio.Event()
    children = []

    async def inherited_scope():
        await released.wait()
        return scoped_plugin_secret("demo", "TOKEN")

    async def dispatch(*_):
        assert scoped_plugin_secret("demo", "TOKEN") == "synthetic-token"
        children.append(asyncio.create_task(inherited_scope()))
        started.set()
        await asyncio.Event().wait()

    vault = SimpleNamespace(secrets_for=lambda *_: {"TOKEN": "synthetic-token"})
    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), vault=vault)
    entry = Entry("demo", SimpleNamespace(), secrets=("TOKEN",))
    task = asyncio.create_task(scheduler._execute(entry))
    try:
        await asyncio.wait_for(started.wait(), 1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        released.set()
        inherited = await asyncio.gather(*children)

    assert inherited == [""]
    assert scoped_plugin_secret("demo", "TOKEN") is None
    assert entry.last.as_dict()["status"] == "cancelled"
    assert entry.failures == 0
