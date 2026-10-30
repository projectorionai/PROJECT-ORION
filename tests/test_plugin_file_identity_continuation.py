"""Manifest ownership is independent of its module and metadata basenames."""
import json
from types import SimpleNamespace

import pytest

from orion_core import forge as forge_module
from orion_core import plugin_registry as module
from orion_core.data import ToolResult
from orion_core.plugin_manifest import load_plugins


@pytest.fixture
def mapped(tmp_path, monkeypatch):
    directory = tmp_path / "config" / "custom_tools"
    monkeypatch.setattr(module, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    monkeypatch.setattr(forge_module, "CONFIG_DIR", directory.parent)
    registry = module.PluginRegistry(directory=directory)
    assert registry.create("demo", "Synthetic mapped module", "confirm").ok
    (directory / "demo_tool.py").rename(directory / "implementation_tool.py")
    old = directory / "demo.plugin.json"
    payload = json.loads(old.read_text(encoding="utf-8"))
    payload["module"] = "implementation_tool.py"
    old.unlink()
    (directory / "metadata.plugin.json").write_text(json.dumps(payload), encoding="utf-8")
    return registry


def test_a_manifest_owned_module_is_not_a_second_orphan(mapped):
    assert [record.name for record in mapped.scan()] == ["demo"]


def test_remove_uses_the_actual_manifest_owned_files(mapped):
    result = mapped.remove("demo")
    assert result.ok
    assert not (mapped.directory / "implementation_tool.py").exists()
    assert not (mapped.directory / "metadata.plugin.json").exists()


async def test_forge_does_not_revive_a_disabled_mapped_manifest_module(mapped):
    assert mapped.set_enabled("demo", False).ok
    imports = []

    async def load(path):
        imports.append(path)
        return SimpleNamespace(succeeded=True)

    async def activate(*_):
        return ToolResult("active")

    manager = object.__new__(forge_module.ForgeOrchestrationManager)
    manager.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    manager.history = None
    manager.dispatcher = SimpleNamespace(plugins=mapped)
    manager.loader = SimpleNamespace(load_and_register=load, activate_tool=activate)
    assert await manager.reload_persisted_tools() == 0
    assert imports == []


async def test_duplicate_manifest_identities_are_held_before_import(mapped):
    first = mapped.directory / "metadata.plugin.json"
    second = mapped.directory / "duplicate.plugin.json"
    second.write_bytes(first.read_bytes())
    imports = []

    async def load(path):
        imports.append(path)
        return SimpleNamespace(succeeded=True, module_name="implementation_tool", tool_schema={"name": "demo"})

    async def activate(*_):
        return ToolResult("active")

    loader = SimpleNamespace(load_and_register=load, activate_tool=activate)
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    plan = await load_plugins(loader, None, mapped.directory, bus, registry=mapped)
    assert plan.loadable == []
    assert imports == []
    assert len(plan.skipped) == 2
