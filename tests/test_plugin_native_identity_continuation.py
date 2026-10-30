"""An extension must not advertise itself under another route's identity."""
from types import SimpleNamespace

import pytest

from orion_core.dynamic_loader import ReflectiveModuleLoader


@pytest.mark.parametrize("name", ["native", "mcp__connected__read"])
async def test_activation_preserves_native_and_connected_mcp_routes(name):
    lines = []
    bus = SimpleNamespace(log=SimpleNamespace(emit=lines.append))
    loader = ReflectiveModuleLoader(bus)
    original = {"name": name, "description": "Original route", "parameters": {"type": "object"}}
    dispatcher = SimpleNamespace(handler_table=lambda: {"native": lambda *_: "original"},
                                 _mcp_routes={"mcp__connected__read": ("connected", "read")},
                                 _tool_handlers={}, TOOL_DECLARATIONS=[original.copy()])
    outcome = SimpleNamespace(succeeded=True, tool_schema={"name": name, "description": "Synthetic extension",
                                                          "parameters": {"type": "object"}},
                              handler_fn=lambda *_: "synthetic", module_name="synthetic_tool", error_log=[])
    result = await loader.activate_tool(outcome, dispatcher)
    assert not result.ok
    assert dispatcher._tool_handlers == {}
    assert dispatcher.TOOL_DECLARATIONS == [original]


async def test_unique_plugin_and_its_replacement_generation_can_activate():
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    loader = ReflectiveModuleLoader(bus)
    original = {"name": "native", "description": "Built-in"}
    dispatcher = SimpleNamespace(handler_table=lambda: {"native": lambda *_: None},
                                 _mcp_routes={}, _tool_handlers={}, TOOL_DECLARATIONS=[original.copy()])
    for generation in (1, 2):
        handler = lambda *_: generation
        schema = {"name": "custom", "description": f"Synthetic generation {generation}",
                  "parameters": {"type": "object"}}
        outcome = SimpleNamespace(succeeded=True, tool_schema=schema, handler_fn=handler,
                                  module_name="custom_tool", error_log=[])
        assert (await loader.activate_tool(outcome, dispatcher)).ok
        assert dispatcher._tool_handlers == {"custom": handler}
        assert dispatcher.TOOL_DECLARATIONS == [original, schema]


async def test_unavailable_native_ownership_does_not_publish_a_plugin():
    def unavailable():
        raise RuntimeError("synthetic route inventory failure")
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    loader = ReflectiveModuleLoader(bus)
    dispatcher = SimpleNamespace(handler_table=unavailable, _tool_handlers={}, TOOL_DECLARATIONS=[])
    outcome = SimpleNamespace(succeeded=True, tool_schema={"name": "custom", "description": "Custom route",
                              "parameters": {"type": "object"}}, handler_fn=lambda *_: None,
                              module_name="custom_tool", error_log=[])
    result = await loader.activate_tool(outcome, dispatcher)
    assert not result.ok
    assert dispatcher._tool_handlers == {} and dispatcher.TOOL_DECLARATIONS == []
