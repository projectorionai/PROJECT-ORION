"""Gateway wrappers must not skip remote capability decisions."""

import asyncio
import sys
from types import SimpleNamespace

from orion_core.data import ToolResult
from orion_core.remote_capability import RemoteConfirmationRegistry, RemoteToolGate


class _Dispatcher:
    TOOL_DECLARATIONS = [
        {"name": "use_tool", "parameters": {"type": "OBJECT"}},
        {"name": "forge", "parameters": {"type": "OBJECT"}},
        {"name": "web_search", "parameters": {"type": "OBJECT"}},
        {"name": "file_controller", "parameters": {"type": "OBJECT"}},
    ]

    def __init__(self):
        self.calls = []

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        return ToolResult("done")


def _gate():
    dispatcher = _Dispatcher()
    registry = RemoteConfirmationRegistry()
    return RemoteToolGate(dispatcher, registry), dispatcher, registry


def test_gateway_cannot_wrap_forbidden_tool_even_with_approval():
    gate, dispatcher, registry = _gate()

    async def scenario():
        with gate.for_device("paired"):
            denied = await gate.dispatch("use_tool", {
                "name": "forge", "arguments": '{"action":"status"}',
            })
        assert not denied.ok
        assert registry.pending_for("paired") == []

        # A pending wrapper saved by an older version must also be refused.
        old = registry.create("use_tool", {
            "name": "forge", "arguments": {"action": "status"},
        }, "paired", "use a tool")
        assert registry.resolve(old.id, old.token, True, device_id="paired") is old
        result = await gate.run_approved(old)
        assert not result.ok

    asyncio.run(scenario())
    assert dispatcher.calls == []


def test_gateway_approval_is_bound_to_underlying_tool_and_arguments():
    gate, dispatcher, registry = _gate()

    async def scenario():
        with gate.for_device("paired"):
            pending = await gate.dispatch("use_tool", {
                "name": "file_controller", "arguments": '{"action":"write","path":"notes.txt"}',
            })
        assert pending.ok
        conf = registry.get(pending.media["confirm"]["id"])
        assert conf.tool == "file_controller"
        assert conf.args == {"action": "write", "path": "notes.txt"}
        assert registry.resolve(conf.id, conf.token, True, device_id="paired") is conf
        assert (await gate.run_approved(conf)).ok

    asyncio.run(scenario())
    assert dispatcher.calls == [("file_controller", {"action": "write", "path": "notes.txt"})]


def test_plan_gateway_target_is_checked_before_any_step_runs(monkeypatch):
    gate, dispatcher, _ = _gate()
    # The plan catalogue imports the desktop Qt bus; this policy test only
    # needs its argument-checking interface.
    monkeypatch.setitem(sys.modules, "orion_core.plan_executor", SimpleNamespace(
        ToolCatalogue=lambda _declarations: SimpleNamespace(check_args=lambda _name, _args: None),
    ))

    async def scenario():
        with gate.for_device("paired"):
            result = await gate.dispatch_plan([
                {"tool": "web_search", "args": {"query": "weather"}},
                {"tool": "use_tool", "args": {"name": "forge", "arguments": {}}},
            ])
        assert not result.ok
        assert "forge" in result.text

    asyncio.run(scenario())
    assert dispatcher.calls == []


def test_allowed_gateway_target_runs_as_the_target():
    gate, dispatcher, registry = _gate()

    async def scenario():
        with gate.for_device("paired"):
            result = await gate.dispatch("use_tool", {
                "name": "web_search", "arguments": '{"query":"weather"}',
            })
        assert result.ok
        assert registry.pending_for("paired") == []

    asyncio.run(scenario())
    assert dispatcher.calls == [("web_search", {"query": "weather"})]


def test_approved_plan_executes_gateway_step_as_reviewed_target():
    gate, dispatcher, registry = _gate()

    async def scenario():
        with gate.for_device("paired"):
            pending = await gate.dispatch_plan([
                {"tool": "use_tool", "args": {
                    "name": "file_controller", "arguments": {"action": "write", "path": "notes.txt"},
                }},
            ])
        assert pending.ok
        conf = registry.get(pending.media["confirm"]["id"])
        assert "file" in conf.summary
        assert registry.resolve(conf.id, conf.token, True, device_id="paired") is conf
        assert (await gate.run_approved(conf)).ok

    asyncio.run(scenario())
    assert dispatcher.calls == [("file_controller", {"action": "write", "path": "notes.txt"})]
