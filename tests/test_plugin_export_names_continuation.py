"""Archive publication and consistent logical-name preferences/policy."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from orion_core import plugin_registry as module
from orion_core.dispatcher import OrionDispatcher
from orion_core.plugin_manifest import plan_load


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    registry = module.PluginRegistry(directory=tmp_path / "tools")
    assert registry.create("demo", "Synthetic export/name plugin", "confirm").ok
    return registry


@pytest.mark.parametrize("destination_kind", ["previous_bundle", "own_module"])
def test_export_failure_preserves_previous_destination_and_plugin(registry, tmp_path, monkeypatch, destination_kind):
    path = (registry.directory / "demo_tool.py" if destination_kind == "own_module"
            else tmp_path / "previous.orionplugin")
    if destination_kind == "previous_bundle":
        path.write_bytes(b"synthetic prior bundle bytes")
    before = path.read_bytes()
    def refuse(*_, **__):
        raise OSError("synthetic archive member write failure")
    monkeypatch.setattr(zipfile.ZipFile, "write", refuse)
    result = registry.export_bundle("demo", str(path))
    assert not result.ok
    assert path.read_bytes() == before


@pytest.mark.parametrize("logical_name", ["Demo", "demo-tool"])
def test_a_valid_logical_name_can_be_managed_without_rewriting_it(registry, logical_name):
    path = registry.directory / "demo.plugin.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["name"] = logical_name
    path.write_text(json.dumps(manifest), encoding="utf-8")
    record = registry.get(logical_name)
    assert record is not None and record.name == logical_name
    assert registry.set_enabled(logical_name, False).ok
    assert not registry.is_enabled(logical_name)


@pytest.mark.parametrize("logical_name", ["Demo", "demo-tool"])
def test_saved_actual_logical_name_cannot_bypass_disabled_preferences(registry, logical_name):
    module.PLUGIN_STATE_PATH.write_text(json.dumps({"disabled": [logical_name], "meta": {}}), encoding="utf-8")
    registry._load_state()
    assert not registry.is_enabled(logical_name)


@pytest.mark.parametrize("logical_name", ["Demo", "demo-tool"])
def test_network_confirmation_applies_to_the_actual_logical_name(registry, monkeypatch, logical_name):
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", "1")
    path = registry.directory / "demo.plugin.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(name=logical_name, permissions=["network"])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = OrionDispatcher._plugin_gate(SimpleNamespace(plugins=registry), logical_name, {})
    assert result is not None and not result.ok


def test_export_preserves_the_actual_mapped_manifest(registry, tmp_path):
    module_path = registry.directory / "demo_tool.py"
    module_path.rename(registry.directory / "implementation_tool.py")
    old = registry.directory / "demo.plugin.json"
    manifest = json.loads(old.read_text(encoding="utf-8"))
    manifest.update(module="implementation_tool.py", events=["state"], interval_seconds=30,
                    secrets=["SYNTHETIC_KEY"])
    old.unlink()
    (registry.directory / "metadata.plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    target = tmp_path / "review.orionplugin"
    assert registry.export_bundle("demo", str(target)).ok
    with zipfile.ZipFile(target) as bundle:
        metadata = next(name for name in bundle.namelist() if name.endswith(".plugin.json"))
        assert json.loads(bundle.read(metadata)) == manifest


def test_normalization_collisions_are_held_before_import(registry):
    path = registry.directory / "demo.plugin.json"
    first = json.loads(path.read_text(encoding="utf-8"))
    first["name"] = "demo-tool"
    path.write_text(json.dumps(first), encoding="utf-8")
    second = {**first, "name": "demo_tool", "module": "second_tool.py"}
    (registry.directory / "second.plugin.json").write_text(json.dumps(second), encoding="utf-8")
    (registry.directory / "second_tool.py").write_bytes((registry.directory / "demo_tool.py").read_bytes())
    plan = plan_load(registry.directory)
    assert plan.loadable == []
    assert len(plan.skipped) == 2
    records = registry.scan()
    assert len(records) == 1 and "Duplicate manifest identity" in records[0].problems[0]
    assert not registry.is_enabled("demo-tool")
    assert not registry.set_enabled("demo_tool", True).ok
    assert not registry.remove("demo-tool").ok


@pytest.mark.parametrize("logical_name", ["Demo", "demo-tool"])
def test_raw_saved_disable_can_be_reenabled_and_survives_a_restart(registry, logical_name):
    path = registry.directory / "demo.plugin.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["name"] = logical_name
    path.write_text(json.dumps(payload), encoding="utf-8")
    module.PLUGIN_STATE_PATH.write_text(json.dumps({"disabled": [logical_name], "meta": {}}), encoding="utf-8")
    restarted = module.PluginRegistry(directory=registry.directory)
    assert not restarted.get(logical_name).enabled
    assert restarted.set_enabled(logical_name, True).ok
    restarted = module.PluginRegistry(directory=registry.directory)
    assert restarted.get(logical_name).name == logical_name
    assert restarted.is_enabled(logical_name)
    assert restarted.set_enabled(logical_name, False).ok
    restarted = module.PluginRegistry(directory=registry.directory)
    assert not restarted.is_enabled(logical_name)


@pytest.mark.parametrize("logical_name,alias", [("Demo", "demo"), ("demo-tool", "demo_tool")])
async def test_alias_removal_retires_actual_runtime_keys(registry, logical_name, alias):
    from datetime import datetime, timezone
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    from orion_core.plugin_events import PluginEventBridge
    from orion_core.plugin_runner import Entry, PluginScheduler
    path = registry.directory / "demo.plugin.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["name"] = logical_name
    path.write_text(json.dumps(payload), encoding="utf-8")
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    bridge = PluginEventBridge(registry=registry)
    bridge._handlers[logical_name] = lambda *_: None
    scheduler = PluginScheduler(registry=registry, clock=lambda: now)
    scheduler.entries[logical_name] = Entry(logical_name, SimpleNamespace(next_due=lambda _: None), next_due=now)
    context = SimpleNamespace(plugins=registry, plugin_events=bridge, plugin_scheduler=scheduler,
                              _tool_handlers={logical_name: lambda: "synthetic"},
                              _forge_rollbacks={logical_name: object()},
                              TOOL_DECLARATIONS=[{"name": logical_name}, {"name": "native"}])
    receipt = await ProductivityDispatchMixin.plugin_tool(context, {"action": "remove", "name": alias})
    assert receipt.ok and receipt.evidence[0]["name"] == logical_name
    assert not context._tool_handlers and not context._forge_rollbacks
    assert not scheduler.entries and not bridge._handlers
    assert context.TOOL_DECLARATIONS == [{"name": "native"}]
    assert not registry.is_enabled(logical_name)
    assert registry.scan() == []


def test_refused_export_publication_preserves_prior_bundle_and_cleans_staging(registry, tmp_path, monkeypatch):
    target = tmp_path / "saved.orionplugin"
    before = b"synthetic previous portable bundle"
    target.write_bytes(before)
    def locked(*_):
        raise PermissionError("synthetic destination lock")
    monkeypatch.setattr(module, "_replace", locked)
    assert not registry.export_bundle("demo", str(target)).ok
    assert target.read_bytes() == before
    assert not list(target.parent.glob("*.export.tmp"))
    assert registry.get("demo").healthy


@pytest.mark.parametrize("source_name", ["demo.plugin.json", "demo_tool.py"])
def test_successful_archive_writer_cannot_replace_its_own_sources(registry, source_name):
    target = registry.directory / source_name
    before = target.read_bytes()
    assert not registry.export_bundle("demo", str(target)).ok
    assert target.read_bytes() == before
    assert not list(target.parent.glob("*.export.tmp"))


def test_export_replaces_a_previous_bundle_only_after_complete_generation(registry, tmp_path):
    target = tmp_path / "saved.orionplugin"
    target.write_bytes(b"old synthetic destination")
    assert registry.export_bundle("demo", str(target)).ok
    with zipfile.ZipFile(target) as bundle:
        assert bundle.testzip() is None
        assert bundle.read("demo_tool.py") == (registry.directory / "demo_tool.py").read_bytes()
        assert bundle.read("demo.plugin.json") == (registry.directory / "demo.plugin.json").read_bytes()
    assert not list(target.parent.glob("*.export.tmp"))
