"""Retirement removes stale alias hooks and preserves a connected route schema."""
from types import SimpleNamespace

from orion_core.plugin_events import PluginEventBridge
from orion_core.plugin_registry import detach_runtime


def test_forgetting_a_plugin_retires_previous_spelling_generations():
    delivered = []
    bridge = PluginEventBridge()
    for name in ("Demo-tool", "demo_tool"):
        bridge._handlers[name] = lambda *_: delivered.append("stale")
        bridge._generation[name] = 1
    assert bridge.forget("DEMO TOOL")
    assert bridge._handlers == {}
    for name in ("Demo-tool", "demo_tool"):
        assert bridge._generation[name] == 2
        assert not bridge.dispatch(name, "state", "after removal")
    assert delivered == []


def test_removing_a_legacy_shadow_keeps_connected_mcp_declaration():
    name = "mcp__connected__read"
    declaration = {"name": name, "description": "Connected route"}
    dispatcher = SimpleNamespace(handler_table=lambda: {}, _mcp_routes={name: ("connected", "read")},
                                 _tool_handlers={name: lambda *_: None}, TOOL_DECLARATIONS=[declaration])
    detach_runtime(dispatcher, name)
    assert dispatcher._tool_handlers == {}
    assert dispatcher.TOOL_DECLARATIONS == [declaration]
