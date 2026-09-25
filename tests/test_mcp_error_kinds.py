"""
MCP replies carry their outcome, and protocol rejections are told apart from
tool failures.

A real stdio JSON-RPC server (a small Python script) is spawned, so this runs
the actual transport: handshake, tool calls, a JSON-RPC error reply, a tool
result with ``isError``, and a server that writes non-protocol noise to
stdout and a flood to stderr before answering.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dispatch_web import mcp_call_failed
from orion_core.mcp_host import MCPHost, MCPProtocolError, MCPReply, MCPServerConn

_SERVER = textwrap.dedent('''
    import json, sys
    TOOLS = [{"name": n, "inputSchema": {"type": "object"}} for n in
             ("echo", "fail", "reject", "noisy", "slow")]
    for line in sys.stdin:
        msg = json.loads(line)
        if "id" not in msg:
            continue
        method, rid = msg.get("method"), msg["id"]
        params = msg.get("params") or {}
        reply = {"jsonrpc": "2.0", "id": rid}
        if method == "initialize":
            reply["result"] = {"protocolVersion": "2024-11-05", "capabilities": {},
                               "serverInfo": {"name": "fake"}}
        elif method == "tools/list":
            reply["result"] = {"tools": TOOLS}
        elif method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name == "echo":
                reply["result"] = {"content": [{"type": "text", "text": str(args.get("text"))}]}
            elif name == "fail":
                reply["result"] = {"isError": True,
                                   "content": [{"type": "text", "text": "path must be absolute"}]}
            elif name == "reject":
                reply["error"] = {"code": -32602, "message": "Invalid params",
                                  "data": "text: required property missing"}
            elif name == "noisy":
                print("npm WARN deprecated something")      # non-JSON on stdout
                sys.stderr.write("x" * 200000 + "\\n")        # a stderr flood
                sys.stderr.flush()
                reply["result"] = {"content": [{"type": "text", "text": "still fine"}]}
            elif name == "slow":
                import time; time.sleep(1.0)
                reply["result"] = {"content": [{"type": "text", "text": "late"}]}
        else:
            reply["error"] = {"code": -32601, "message": "Method not found"}
        print(json.dumps(reply), flush=True)
''')


class _Signal:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def _run_with_server(tmp_path, body):
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(_SERVER, encoding="utf-8")

    async def flow():
        host = MCPHost(_Bus(), telemetry=None)
        conn = MCPServerConn("fake", {"command": sys.executable, "args": [str(script)]}, _Bus())
        assert await conn.start(), conn.last_error
        host.servers["fake"] = conn
        try:
            return await body(host)
        finally:
            await conn.stop()

    return asyncio.run(flow())


def test_success_is_a_reply_without_an_error(tmp_path):
    async def body(host):
        return await host.call("fake", "echo", {"text": "hello"})
    reply = _run_with_server(tmp_path, body)
    assert isinstance(reply, MCPReply) and reply == "hello"
    assert reply.kind == "" and not reply.is_error
    assert mcp_call_failed(reply) is False


def test_a_tool_that_reports_isError_is_a_tool_failure(tmp_path):
    async def body(host):
        return await host.call("fake", "fail", {})
    reply = _run_with_server(tmp_path, body)
    assert reply.kind == "tool"
    assert "path must be absolute" in reply        # the model's cue to self-correct
    assert mcp_call_failed(reply) is True


def test_a_json_rpc_error_is_a_protocol_rejection_with_an_actionable_message(tmp_path):
    async def body(host):
        return await host.call("fake", "reject", {})
    reply = _run_with_server(tmp_path, body)
    assert reply.kind == "protocol"
    assert "Invalid params (JSON-RPC -32602)" in reply
    assert "required property missing" in reply
    assert "input schema" in reply
    assert mcp_call_failed(reply) is True


def test_stdout_noise_and_a_stderr_flood_do_not_break_the_session(tmp_path):
    async def body(host):
        first = await host.call("fake", "noisy", {})
        second = await host.call("fake", "echo", {"text": "after"})
        return first, second
    first, second = _run_with_server(tmp_path, body)
    assert first == "still fine" and not first.is_error
    assert second == "after"


def test_a_timeout_is_reported_and_the_late_reply_is_not_misattributed(tmp_path, monkeypatch):
    import orion_core.mcp_host as mcp_host
    monkeypatch.setattr(mcp_host, "_CALL_TIMEOUT", 0.5)

    async def body(host):
        late = await host.call("fake", "slow", {})
        monkeypatch.setattr(mcp_host, "_CALL_TIMEOUT", 10.0)
        # The slow reply arrives while this call waits; it must be skipped by
        # id, not handed back as the answer to a different request.
        after = await host.call("fake", "echo", {"text": "mine"})
        return late, after
    late, after = _run_with_server(tmp_path, body)
    assert late.kind == "timeout"
    assert after == "mine"


def test_unknown_tool_and_unknown_server_are_typed(tmp_path):
    async def body(host):
        return (await host.call("fake", "nope", {}),
                await host.call("missing", "echo", {}))
    no_tool, no_server = _run_with_server(tmp_path, body)
    assert no_tool.kind == "protocol" and "Its tools:" in no_tool
    assert no_server.kind == "unavailable"


def test_a_successful_reply_that_looks_like_an_error_is_not_a_failure():
    """The whole point of carrying the outcome: wording is not evidence."""
    reply = MCPReply("(tool error) is a phrase this file happens to start with")
    assert mcp_call_failed(reply) is False
    # A plain string still gets the wording heuristic.
    assert mcp_call_failed("(tool error) something broke") is True


def test_protocol_error_formats_non_dict_errors():
    assert str(MCPProtocolError("boom")) == "boom"
    err = MCPProtocolError({"code": -32601, "message": "Method not found"})
    assert err.code == -32601 and "Method not found (JSON-RPC -32601)" == str(err)
