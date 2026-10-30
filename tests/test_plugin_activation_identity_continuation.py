"""The actual callable name must match the declared tier and schedule name."""
import json
import asyncio
import threading
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.plugin_manifest import load_plugins
from orion_core import remote_capability
from orion_core.dynamic_loader import ModuleLoadOutcome, ReflectiveModuleLoader


def _activation_fixture():
    loader = ReflectiveModuleLoader.__new__(ReflectiveModuleLoader)
    loader.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    old = lambda: "previous generation"
    schema = {"name": "demo", "description": "Return a synthetic value",
              "parameters": {"type": "object", "properties": {}, "required": []}}
    dispatcher = SimpleNamespace(handler_table=lambda: {}, _mcp_routes={},
                                 _tool_handlers={"demo": old}, TOOL_DECLARATIONS=[schema.copy()], memory=None)
    outcome = ModuleLoadOutcome(True, "demo_tool", tool_schema=schema, handler_fn=lambda: "candidate")
    return loader, dispatcher, outcome, old


@pytest.mark.parametrize("tool_name,expected_active", [("different", False), ("demo", True)])
async def test_plugin_activation_requires_matching_declared_identity(tmp_path, monkeypatch, tool_name, expected_active):
    monkeypatch.setattr(remote_capability, "_PLUGIN_TIERS", {})
    (tmp_path / "demo_tool.py").write_text("# synthetic transport; no import\n", encoding="utf-8")
    (tmp_path / "demo.plugin.json").write_text(json.dumps(
        {"name": "demo", "module": "demo_tool.py", "tier": "forbid", "interval_seconds": 30}),
        encoding="utf-8")
    activated = []
    scheduled = []
    outcome = SimpleNamespace(succeeded=True, module_name="demo_tool", tool_schema={"name": tool_name})
    async def load(*_):
        return outcome
    async def activate(*_):
        activated.append(tool_name)
        return ToolResult("active")
    loader = SimpleNamespace(load_and_register=load, activate_tool=activate)
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    scheduler = SimpleNamespace(add=lambda manifest: scheduled.append(manifest.name))
    plan = await load_plugins(loader, None, tmp_path, bus, scheduler=scheduler)
    assert bool(plan.loadable) is expected_active
    assert activated == (["demo"] if expected_active else [])
    assert scheduled == (["demo"] if expected_active else [])
    if not expected_active:
        assert "demo" in plan.skipped
        assert remote_capability._PLUGIN_TIERS == {}


@pytest.mark.parametrize("mutation", ["missing_description", "bad_parameters", "bad_handler", "truthy_success"])
async def test_activation_rechecks_mutable_load_outcomes(mutation):
    loader, dispatcher, outcome, old = _activation_fixture()
    if mutation == "missing_description":
        del outcome.tool_schema["description"]
    elif mutation == "bad_parameters":
        outcome.tool_schema["parameters"] = []
    elif mutation == "bad_handler":
        outcome.handler_fn = lambda missing_argument: None
    else:
        outcome.succeeded = "false"
    declarations = list(dispatcher.TOOL_DECLARATIONS)
    result = await loader.activate_tool(outcome, dispatcher)
    assert not result.ok
    assert dispatcher._tool_handlers["demo"] is old
    assert dispatcher.TOOL_DECLARATIONS == declarations


async def test_activation_publishes_a_detached_schema_and_a_defined_receipt():
    loader, dispatcher, outcome, _old = _activation_fixture()
    result = await loader.activate_tool(outcome, dispatcher)
    outcome.tool_schema["description"] = "changed after admission"
    outcome.tool_schema["parameters"]["properties"]["unaccepted"] = {"type": "string"}
    assert result.ok
    assert dispatcher.TOOL_DECLARATIONS[0]["description"] == "Return a synthetic value"
    assert dispatcher.TOOL_DECLARATIONS[0]["parameters"]["properties"] == {}
    assert result.receipt["state"] == "activated"
    assert result.receipt["tool"] == "demo"
    assert result.receipt["schema_sha256"]


async def test_activation_uses_one_snapshot_when_optional_indexing_mutates_outcome(monkeypatch):
    loader, dispatcher, outcome, _old = _activation_fixture()
    expected_handler = outcome.handler_fn
    dispatcher.memory = object()
    def mutate(*_):
        outcome.tool_schema["name"] = "different"
        outcome.handler_fn = lambda: "different"
    monkeypatch.setattr(loader, "_register_in_knowledge", mutate)
    result = await loader.activate_tool(outcome, dispatcher)
    assert result.ok
    assert dispatcher._tool_handlers == {"demo": expected_handler}
    assert dispatcher.TOOL_DECLARATIONS[0]["name"] == "demo"
    assert result.receipt["tool"] == "demo"


async def test_cancelled_optional_indexing_cannot_leave_a_published_half_activation(monkeypatch):
    loader, dispatcher, outcome, old = _activation_fixture()
    dispatcher.memory = object()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()
    def wait(*_):
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(5)
        finally:
            finished.set()
    monkeypatch.setattr(loader, "_register_in_knowledge", wait)
    task = asyncio.create_task(loader.activate_tool(outcome, dispatcher))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)
    assert dispatcher._tool_handlers["demo"] is old
    assert dispatcher.TOOL_DECLARATIONS[0]["name"] == "demo"


async def test_optional_activation_observers_cannot_erase_a_committed_result():
    loader, dispatcher, outcome, _old = _activation_fixture()
    def broken(*_):
        raise RuntimeError("synthetic log observer unavailable")
    loader.bus.log.emit = broken
    result = await loader.activate_tool(outcome, dispatcher)
    assert result.ok
    assert dispatcher._tool_handlers["demo"]() == "candidate"
    assert result.receipt["state"] == "activated"


@pytest.mark.parametrize("same_name", [True, False])
async def test_slow_activation_preserves_newer_registration_without_blocking_other_tools(monkeypatch, same_name):
    loader, dispatcher, first, _old = _activation_fixture()
    dispatcher.memory = object()
    first.tool_schema["description"] = "Slow first candidate"
    entered = asyncio.Event()
    release, finished = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    def index(_memory, _name, schema):
        if schema["description"] == "Slow first candidate":
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(5)
            finally:
                finished.set()
    monkeypatch.setattr(loader, "_register_in_knowledge", index)
    pending = asyncio.create_task(loader.activate_tool(first, dispatcher))
    other_name = "demo" if same_name else "other"
    second_schema = {"name": other_name, "description": "Newer completed candidate",
                     "parameters": {"type": "object", "properties": {}, "required": []}}
    second_handler = lambda: "newer"
    second = ModuleLoadOutcome(True, "second_tool", tool_schema=second_schema, handler_fn=second_handler)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert (await loader.activate_tool(second, dispatcher)).ok
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)
    result = await pending
    assert dispatcher._tool_handlers[other_name] is second_handler
    if same_name:
        assert not result.ok and result.receipt["state"] == "superseded"
        assert dispatcher.TOOL_DECLARATIONS[0]["description"] == "Newer completed candidate"
    else:
        assert result.ok
        assert dispatcher._tool_handlers["demo"] is first.handler_fn
