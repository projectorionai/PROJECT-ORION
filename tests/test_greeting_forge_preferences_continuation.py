"""Greeting observers and persisted Forge tools must respect user preferences."""
import json
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core import temporal
from orion_core import forge as forge_module
from orion_core import plugin_registry as registry_module


def _fail(*_):
    raise RuntimeError("synthetic health observer failure")


def test_temporal_can_start_without_its_optional_health_observer(tmp_path, monkeypatch):
    monkeypatch.setattr(temporal, "STATE_PATH", tmp_path / "temporal.json")
    telemetry = SimpleNamespace(health=SimpleNamespace(register=_fail))
    presence = temporal.TemporalPresence(SimpleNamespace(), telemetry=telemetry)
    assert presence is not None


async def test_health_observer_failure_does_not_discard_a_composed_greeting(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(temporal, "STATE_PATH", tmp_path / "temporal.json")
    presence = temporal.TemporalPresence(SimpleNamespace())
    presence.telemetry = SimpleNamespace(health=SimpleNamespace(beat=_fail))
    async def no_feeds(*_, **__):
        return []
    monkeypatch.setattr(presence, "_enrich", no_feeds)
    first = await presence.compose_greeting()
    second = await presence.compose_greeting()
    assert first and second and first != second
    assert presence.greeting_instruction(second)
    assert sum("health beat unavailable" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("state_content,expected", [
    (json.dumps({"disabled": ["demo"], "meta": {}}), 0),
    ("broken", 0), (json.dumps({"disabled": [], "meta": {}}), 1)],
    ids=["disabled", "unavailable_preferences", "enabled"])
async def test_manifestless_tool_reload_honours_saved_preferences(tmp_path, monkeypatch, state_content, expected):
    directory = tmp_path / "config" / "custom_tools"
    directory.mkdir(parents=True)
    monkeypatch.setattr(forge_module, "CONFIG_DIR", directory.parent)
    state = tmp_path / "plugins.json"
    monkeypatch.setattr(registry_module, "PLUGIN_STATE_PATH", state)
    state.write_text(state_content, encoding="utf-8")
    (directory / "demo_tool.py").write_text("# never execute this tool\n", encoding="utf-8")
    registry = registry_module.PluginRegistry(directory=directory)
    imports = []
    async def load(path):
        imports.append(path)
        return SimpleNamespace(succeeded=True)
    async def activate(*_):
        return ToolResult("active")
    # Only transports are fake; the production reload loop performs the decision.
    manager = object.__new__(forge_module.ForgeOrchestrationManager)
    manager.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    manager.history = None
    manager.dispatcher = SimpleNamespace(plugins=registry)
    manager.loader = SimpleNamespace(load_and_register=load, activate_tool=activate)
    assert await manager.reload_persisted_tools() == expected
    assert len(imports) == expected

