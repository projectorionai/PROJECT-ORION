"""Saved disable choices must not disappear when their file is unavailable."""
import json
from types import SimpleNamespace

import pytest

from orion_core import plugin_registry as registry_module
from orion_core.plugin_manifest import load_plugins


@pytest.fixture
def configured(tmp_path, monkeypatch):
    state = tmp_path / "plugins.json"
    monkeypatch.setattr(registry_module, "PLUGIN_STATE_PATH", state)
    directory = tmp_path / "tools"
    directory.mkdir()
    (directory / "demo_tool.py").write_text(
        "def get_tool_schema():\n    return {'name':'demo','parameters':{}}\n"
        "def run(**kwargs):\n    return 'done'\n", encoding="utf-8")
    (directory / "demo.plugin.json").write_text(
        json.dumps({"name": "demo", "module": "demo_tool.py"}), encoding="utf-8")
    state.write_text(json.dumps({"disabled": ["demo"], "meta": {}}), encoding="utf-8")
    return directory, state


@pytest.mark.parametrize("raw", ["{broken", "[]", '{"disabled":"demo"}',
                                 '{"disabled":[{}]}'],
                         ids=["broken_json", "wrong_root", "wrong_disabled_list", "wrong_name_type"])
def test_invalid_state_keeps_plugins_held_and_preserves_file(configured, raw):
    directory, state = configured
    state.write_text(raw, encoding="utf-8")
    registry = registry_module.PluginRegistry(directory=directory)
    assert not registry.is_enabled("demo")
    assert not registry.list_plugins().ok
    assert not registry.doctor().ok
    assert not registry.set_enabled("demo", True).ok
    assert state.read_text(encoding="utf-8") == raw


async def test_unreadable_disable_state_does_not_load_plugin(configured, monkeypatch):
    directory, state = configured
    original = type(state).read_text
    def read(path, *args, **kwargs):
        if path == state:
            raise PermissionError("synthetic unavailable saved preferences")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(type(state), "read_text", read)
    registry = registry_module.PluginRegistry(directory=directory)
    async def must_not_import(*_):
        raise AssertionError("held plugin must not be imported")
    loader = SimpleNamespace(load_and_register=must_not_import)
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    plan = await load_plugins(loader, None, directory, bus, registry=registry)
    assert plan.loadable == []
    assert "demo" in plan.skipped
    assert "unavailable" in plan.skipped["demo"].lower()


@pytest.mark.parametrize("operation", ["create", "install", "install_bundle", "remove"])
def test_unavailable_state_blocks_mutation_and_keeps_existing_tool(configured, operation):
    directory, state = configured
    state.write_text("broken", encoding="utf-8")
    registry = registry_module.PluginRegistry(directory=directory)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    result = getattr(registry, operation)("new" if operation == "create" else "demo")
    assert not result.ok
    assert state.read_text(encoding="utf-8") == "broken"
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original


def test_unreadable_state_without_exception_text_still_holds_plugins(configured, monkeypatch):
    directory, state = configured
    original = type(state).read_text
    def read(path, *args, **kwargs):
        if path == state:
            raise PermissionError()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(type(state), "read_text", read)
    registry = registry_module.PluginRegistry(directory=directory)
    assert not registry.is_enabled("demo")
    assert registry.state_error


def test_an_absent_state_file_preserves_first_run_enabled_default(configured):
    directory, state = configured
    state.unlink()
    registry = registry_module.PluginRegistry(directory=directory)
    assert registry.is_enabled("demo")


def test_valid_state_can_be_reloaded_after_the_user_repairs_it(configured):
    directory, state = configured
    state.write_text("broken", encoding="utf-8")
    registry = registry_module.PluginRegistry(directory=directory)
    assert not registry.is_enabled("demo")
    state.write_text(json.dumps({"disabled": ["demo"], "meta": {}}), encoding="utf-8")
    registry._load_state()
    assert not registry.is_enabled("demo")
    assert registry.set_enabled("demo", True).ok
    assert registry.is_enabled("demo")
