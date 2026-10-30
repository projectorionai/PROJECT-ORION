"""An MCP call owns its server while argument validation is offloaded."""

import asyncio
import threading
import time
from types import SimpleNamespace

from orion_core import mcp_policy
from orion_core.mcp_host import MCPHost, MCPServerConn


def test_idle_parking_does_not_stop_call_during_argument_validation(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original = mcp_policy.validate_arguments

    def slow_validation(spec, arguments):
        entered.set()
        assert release.wait(2)
        return original(spec, arguments)

    monkeypatch.setattr(mcp_policy, "validate_arguments", slow_validation)
    conn = MCPServerConn("local", {}, None)
    conn.ready = True
    conn.proc = SimpleNamespace(returncode=None)
    conn.last_used = time.monotonic() - 100

    async def request(_method, _params):
        return {"content": [{"type": "text", "text": "done"}]}

    monkeypatch.setattr(conn, "_request", request)
    host = MCPHost(None)
    host.servers["local"] = conn
    host.idle_park_s = 1

    async def scenario():
        task = asyncio.create_task(conn.call_tool("check", {}))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            # Validation can outlive a short memory-pressure parking budget.
            assert conn.busy()
            assert await host.park_idle(now=time.monotonic() + 100) == []
        finally:
            release.set()
        assert str(await task) == "done"
        assert "local" not in host.parked

    asyncio.run(scenario())


def test_validation_failure_releases_the_connection_and_updates_idle_time(monkeypatch):
    from orion_core.mcp_host import MCPReply

    def rejection(_spec, _arguments):
        raise ValueError("invalid arguments")

    monkeypatch.setattr(mcp_policy, "validate_arguments", rejection)
    conn = MCPServerConn("local", {}, None)
    conn.tools = [{"name": "check"}]
    host = MCPHost(None)
    host.servers["local"] = conn

    async def scenario():
        started = time.monotonic()
        reply = await host.call("local", "check", {})
        assert isinstance(reply, MCPReply) and reply.kind == "protocol"
        assert not conn.busy()
        assert conn.last_used >= started

    asyncio.run(scenario())
