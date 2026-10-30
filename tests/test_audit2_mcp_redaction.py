"""Protocol errors must have the same credential boundary as tool results."""
from types import SimpleNamespace
import sys

import orion_core.mcp_host as mcp
from orion_core.mcp_host import MCPHost, MCPProtocolError, MCPServerConn


async def test_protocol_rejection_redacts_configured_credentials(monkeypatch):
    credential = "synthetic-" + "credential-12345"
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    conn = MCPServerConn("probe", {"command": "unused", "env": {"API_TOKEN": credential}}, bus)
    conn.tools = [{"name": "read"}]
    async def reject(*_):
        raise MCPProtocolError({"code": -32602, "message": "Invalid params", "data": credential})
    monkeypatch.setattr(conn, "_request", reject)
    host = MCPHost(bus)
    host.servers["probe"] = conn
    reply = await host.call("probe", "read", {})
    assert reply.kind == "protocol" and "Invalid params" in reply
    assert credential not in reply
    assert "[redacted]" in reply


async def test_explicitly_inherited_credentials_are_redacted_on_launch_failure(monkeypatch):
    credential = "inherited-" + "credential-67890"
    monkeypatch.setenv("AUDIT_API_TOKEN", credential)
    monkeypatch.setattr(mcp, "resolve_command", lambda _: sys.executable)
    monkeypatch.setattr(mcp, "_tool_dirs", lambda: [])
    async def fail(*_, **kwargs):
        assert kwargs["env"]["AUDIT_API_TOKEN"] == credential
        raise OSError(f"Launch rejected: {credential}")
    monkeypatch.setattr(mcp.asyncio, "create_subprocess_exec", fail)
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    conn = MCPServerConn("probe", {"command": "python", "inherit_env": ["AUDIT_API_TOKEN"]}, bus)
    assert not await conn.start()
    assert credential not in conn.last_error
    assert "[redacted]" in conn.last_error
