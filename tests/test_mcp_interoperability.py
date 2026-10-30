"""Real stdio fixtures for both MCP protocol eras and failure recovery.

The 2026 fixture follows the published server/discover and per-request _meta
wire contract. It does not depend on (or claim to be) an installed MCP SDK.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest

from orion_core import mcp_host as mh


@pytest.mark.parametrize("version", ["2025-06-18", mh._MODERN_PROTOCOL_VERSION])
async def test_paginated_catalogue_keeps_tools_from_every_page(version):
    conn = mh.MCPServerConn("fixture", {}, None)
    conn.protocol_version = version
    requests = []

    async def request(method, params):
        requests.append((method, params))
        result = {"tools": [{"name": "second" if params else "first"}]}
        if version == mh._MODERN_PROTOCOL_VERSION:
            result["resultType"] = "complete"
        if not params:
            result["nextCursor"] = "opaque/page/2"
        return result

    conn._request = request
    await conn._list_tools()
    assert [tool["name"] for tool in conn.tools] == ["first", "second"]
    assert requests == [("tools/list", {}), ("tools/list", {"cursor": "opaque/page/2"})]


@pytest.mark.parametrize("fault", ["cycle", "invalid_cursor", "duplicate", "invalid_page"])
async def test_paginated_catalogue_rejects_broken_pages_without_publishing_partial_tools(fault):
    conn = mh.MCPServerConn("fixture", {}, None)
    conn.protocol_version = "2025-06-18"
    conn.tools = [{"name": "previous"}]
    calls = 0

    async def request(_method, _params):
        nonlocal calls
        calls += 1
        assert calls < 4, "pagination did not stop at the malformed cursor"
        if calls == 1:
            return {"tools": [{"name": "first"}], "nextCursor": "page2"}
        if fault == "cycle":
            return {"tools": [], "nextCursor": "page2"}
        if fault == "invalid_cursor":
            return {"tools": [], "nextCursor": 7}
        if fault == "duplicate":
            return {"tools": [{"name": "first"}]}
        return {"tools": "bad"}

    conn._request = request
    with pytest.raises(mh.MCPProtocolError):
        await conn._list_tools()
    assert calls == 2
    assert conn.tools == [{"name": "previous"}]


async def test_paginated_catalogue_has_a_finite_page_budget(monkeypatch):
    conn = mh.MCPServerConn("fixture", {}, None)
    conn.protocol_version = "2025-06-18"
    monkeypatch.setattr(mh, "_CATALOGUE_MAX_PAGES", 3, raising=False)
    calls = 0

    async def request(_method, _params):
        nonlocal calls
        calls += 1
        assert calls <= 3
        return {"tools": [{"name": f"tool{calls}"}], "nextCursor": f"page{calls + 1}"}

    conn._request = request
    with pytest.raises(mh.MCPProtocolError, match="page limit"):
        await conn._list_tools()
    assert calls == 3 and conn.tools == []


async def test_json_rpc_boolean_id_cannot_answer_integer_request():
    conn = mh.MCPServerConn("fixture", {}, None)
    reader = asyncio.StreamReader()
    reader.feed_data((json.dumps({"jsonrpc": "2.0", "id": True, "result": "forged"}) + "\n"
                      + json.dumps({"jsonrpc": "2.0", "id": 1, "result": "correct"}) + "\n").encode())
    reader.feed_eof()
    from types import SimpleNamespace
    conn.proc = SimpleNamespace(stdout=reader)

    async def send(_message):
        pass

    conn._send = send
    assert await conn._request("tools/list", {}) == "correct"


class _CatalogueWire:
    """A fully in-memory stdio server; no subprocess or socket is opened."""

    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.stdin = self
        self.stderr = None
        self.returncode = None
        self.tools = [{"name": "old", "inputSchema": {"type": "object"}}]
        self.requests = []
        self.held = None
        self.hold_call = False
        self.hold_list = False
        self.hold_cursor = None
        self.pages = {}
        self.modern = False
        self.closed = False

    def send(self, message):
        self.stdout.feed_data((json.dumps(message) + "\n").encode())

    def changed(self):
        self.send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})

    def answer(self, request, result):
        self.send({"jsonrpc": "2.0", "id": request["id"], "result": result})

    def write(self, data):
        request = json.loads(data)
        self.requests.append(request)
        if request.get("method") == "tools/list":
            cursor = request.get("params", {}).get("cursor", "")
            if self.hold_list or cursor == self.hold_cursor:
                self.held = request
            else:
                result = self.pages.get(cursor, {"tools": self.tools})
                if self.modern:
                    result = dict(result, resultType="complete")
                self.answer(request, result)
        elif request.get("method") == "tools/call":
            if self.hold_call:
                self.held = request
            else:
                self.answer(request, {"content": [{"type": "text", "text": "done"}]})

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait(self):
        self.returncode = 0
        self.stdout.feed_eof()
        return 0


async def _wire_host(monkeypatch, modern=False):
    wire = _CatalogueWire()
    wire.modern = modern
    host = mh.MCPHost(_Bus())
    monkeypatch.setattr(mh, "load_config", lambda: {
        "servers": {"fixture": {"enabled": True, "command": "unused", "env": {}}}})
    monkeypatch.setattr(mh, "_load_catalogue", lambda: {})
    monkeypatch.setattr(host, "_remember_tools", lambda *_args: None)

    async def start(conn):
        conn.proc = wire
        conn.protocol_version = mh._MODERN_PROTOCOL_VERSION if modern else "2025-11-25"
        await conn._list_tools()
        conn.ready = True
        return True

    monkeypatch.setattr(mh.MCPServerConn, "start", start)
    await host.connect_all()
    publications = []
    host._on_reconnect = lambda _name, conn: publications.append(
        list(getattr(conn, "tools", conn)))
    return host, wire, publications


async def _eventually(predicate):
    async with asyncio.timeout(0.4):
        while not predicate():
            await asyncio.sleep(0.002)


@pytest.mark.parametrize("terminal_output", ["eof", "malformed"])
async def test_start_cannot_revive_receiver_failure_after_final_catalogue_reply(monkeypatch, terminal_output):
    wire = _CatalogueWire()
    conn = mh.MCPServerConn("fixture", {"command": sys.executable}, _Bus())
    write = wire.write

    def finish_then_fail(data):
        write(data)
        # Both messages are already buffered when the sole reader wakes.
        # It resolves tools/list and then observes the terminal failure before
        # the startup coroutine gets a chance to publish readiness.
        if terminal_output == "eof":
            wire.stdout.feed_eof()
        else:
            wire.stdout.feed_data(b"invalid JSON-RPC output\n" * 8)

    async def spawn(*_args, **_kwargs):
        return wire

    monkeypatch.setattr(wire, "write", finish_then_fail)
    monkeypatch.setattr(mh, "resolve_command", lambda command: command)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(conn, "_handshake", conn._list_tools)
    try:
        assert await conn.start() is False
        assert not conn.ready and conn.proc is None
        assert conn._receive_error is not None
        assert "handshake failed" in conn.last_error
        assert wire.closed
    finally:
        await conn.stop()


async def test_dynamic_catalogue_idle_notification_replaces_routes_and_schema(monkeypatch):
    from orion_core.dispatcher import OrionDispatcher

    host, wire, publications = await _wire_host(monkeypatch)
    dispatcher = OrionDispatcher.__new__(OrionDispatcher)
    dispatcher.register_mcp_tools("fixture", host.servers["fixture"])
    record_publication = host._on_reconnect

    def publish(name, conn):
        record_publication(name, conn)
        dispatcher.register_mcp_tools(name, conn)

    host._on_reconnect = publish
    try:
        wire.tools = [{"name": "new", "inputSchema": {"type": "object", "required": ["query"]}}]
        wire.changed()
        await _eventually(lambda: bool(publications))
        assert host.servers["fixture"].tools == wire.tools
        assert publications == [wire.tools]
        assert "mcp__fixture__old" not in dispatcher._mcp_routes
        assert dispatcher._mcp_routes["mcp__fixture__new"] == ("fixture", "new")
        assert dispatcher._mcp_schemas["mcp__fixture__new"]["required"] == ["query"]
        assert (await host.call("fixture", "old", {})).kind == "protocol"
        assert await host.call("fixture", "new", {"query": "synthetic"}) == "done"
    finally:
        await host.close()


async def test_dynamic_catalogue_refresh_preserves_an_in_flight_call(monkeypatch):
    host, wire, publications = await _wire_host(monkeypatch)
    wire.hold_call = True
    pending = asyncio.create_task(host.call("fixture", "old", {}))
    try:
        await _eventually(lambda: wire.held is not None)
        held = wire.held
        wire.tools = [{"name": "new", "inputSchema": {"type": "object"}}]
        for _ in range(20):
            wire.changed()
        await asyncio.sleep(0.01)
        assert not pending.done()
        wire.answer(held, {"content": [{"type": "text", "text": "original call"}]})
        assert await pending == "original call"
        await _eventually(lambda: bool(publications))
        assert publications == [wire.tools]
        assert len([r for r in wire.requests if r.get("method") == "tools/list"]) == 2
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await host.close()


async def test_dynamic_catalogue_close_retires_pending_refresh_and_reader(monkeypatch):
    host, wire, publications = await _wire_host(monkeypatch)
    conn = host.servers["fixture"]
    wire.hold_list = True
    wire.changed()
    try:
        await _eventually(lambda: wire.held is not None)
        assert conn.busy()
        await asyncio.wait_for(host.close(), timeout=0.4)
        assert publications == [[]]
        assert wire.closed and conn.proc is None
        assert not [t for t in asyncio.all_tasks() if not t.done()
                    and t.get_name().startswith("orion-mcp-")]
    finally:
        await host.close()


@pytest.mark.parametrize("schema", [{"properties": []}, {"required": [1]}, {"type": "nonsense"}])
async def test_dynamic_catalogue_invalid_schema_never_publishes(monkeypatch, schema):
    conn = mh.MCPServerConn("fixture", {}, None)
    conn.tools = [{"name": "previous"}]

    async def request(*_args):
        return {"tools": [{"name": "bad", "inputSchema": schema}]}

    conn._request = request
    with pytest.raises(mh.MCPProtocolError, match="catalogue"):
        await conn._list_tools()
    assert conn.tools == [{"name": "previous"}]


@pytest.mark.parametrize("error", [False, True])
async def test_mcp_receipt_preserves_structured_media_resource_provenance_and_redaction(error):
    secret = "synthetic-private-token"
    conn = mh.MCPServerConn("fixture", {"env": {"API_KEY": secret}}, None)
    result = {"content": [
        {"type": "text", "text": "summary " + secret},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png",
         "annotations": {"audience": ["user"]}},
        {"type": "resource_link", "uri": "https://example.test/report", "name": "report"},
        {"type": "resource", "resource": {"uri": "data://synthetic/report",
                                             "text": "detail " + secret}},
    ], "structuredContent": {"measure": 12, "token": secret},
        "_meta": {"source": "synthetic " + secret}, "isError": error}

    async def request(*_args):
        return result

    conn._request = request
    receipt = await conn.call_tool("read", {})
    assert receipt.kind == ("tool" if error else "")
    assert receipt.structured_content == {"measure": 12, "token": "[redacted]"}
    assert receipt.content[1] == result["content"][1]
    assert receipt.content[2]["uri"] == "https://example.test/report"
    assert receipt.content[3]["resource"]["text"] == "detail [redacted]"
    assert receipt.meta == {"source": "synthetic [redacted]"}
    assert receipt.provenance["server"] == "fixture" and receipt.provenance["tool"] == "read"
    receipt.content[1]["data"] = "changed"
    assert result["content"][1]["data"] == "aGVsbG8="
    assert secret not in str(receipt)


@pytest.mark.parametrize("result", [
    {"content": [42]}, {"content": [{"type": "text", "text": 42}]},
    {"content": [], "structuredContent": []},
    {"content": [{"type": "unsupported_extension", "data": "unknown"}]},
])
def test_mcp_receipt_preserves_failure_for_malformed_or_unsupported_content(result):
    assert mh.MCPServerConn._render(result).kind == "protocol"


@pytest.mark.parametrize("modern", [False, True])
async def test_dynamic_catalogue_changed_pages_publish_together_after_the_final_page(monkeypatch, modern):
    host, wire, publications = await _wire_host(monkeypatch, modern=modern)
    conn = host.servers["fixture"]
    old_tools = conn.tools
    wire.pages = {"": {"tools": [{"name": "first"}], "nextCursor": "page2"}}
    wire.hold_cursor = "page2"
    try:
        wire.changed()
        await _eventually(lambda: wire.held is not None)
        assert conn.tools == old_tools and publications == []
        final = {"tools": [{"name": "second", "inputSchema": {"type": "object"}}]}
        if modern:
            final["resultType"] = "complete"
        wire.answer(wire.held, final)
        await _eventually(lambda: bool(publications))
        assert [t["name"] for t in publications[0]] == ["first", "second"]
        assert not any(t["name"] == "old" for t in conn.tools)
        if modern:
            assert all(r["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"]
                       == mh._MODERN_PROTOCOL_VERSION for r in wire.requests)
    finally:
        await host.close()


async def test_dynamic_catalogue_notifications_during_refresh_request_one_followup(monkeypatch):
    host, wire, publications = await _wire_host(monkeypatch)
    wire.hold_list = True
    try:
        wire.changed()
        await _eventually(lambda: wire.held is not None)
        held = wire.held
        for _ in range(20):
            wire.changed()
        await asyncio.sleep(0.01)
        wire.hold_list = False
        wire.tools = [{"name": "latest"}]
        wire.answer(held, {"tools": [{"name": "intermediate"}]})
        await _eventually(lambda: len(publications) == 2)
        assert publications == [[{"name": "intermediate"}], [{"name": "latest"}]]
        assert len([r for r in wire.requests if r.get("method") == "tools/list"]) == 3
    finally:
        await host.close()


async def test_dynamic_catalogue_invalid_final_page_retires_routes_without_partial_publication(monkeypatch):
    host, wire, publications = await _wire_host(monkeypatch)
    conn = host.servers["fixture"]
    old_tools = conn.tools
    wire.pages = {"": {"tools": [{"name": "unpublished"}], "nextCursor": "page2"},
                  "page2": {"tools": [{"name": "bad", "inputSchema": {"required": 7}}]}}
    try:
        wire.changed()
        await _eventually(lambda: bool(publications))
        assert publications == [[]]
        assert conn.tools == old_tools and not conn.ready
        assert host.health_snapshot() == {"fixture": "DOWN"}
        assert (await host.call("fixture", "old", {})).kind == "unavailable"
    finally:
        await host.close()


async def test_dynamic_catalogue_empty_refresh_replaces_persisted_cache(monkeypatch, tmp_path):
    host, wire, publications = await _wire_host(monkeypatch)
    cache = tmp_path / "catalogue.json"
    monkeypatch.setattr(mh, "MCP_CATALOGUE_PATH", cache)
    monkeypatch.setattr(host, "_remember_tools", mh.MCPHost._remember_tools.__get__(host))
    try:
        wire.tools = []
        wire.changed()
        await _eventually(lambda: bool(publications))
        assert publications == [[]]
        assert host.servers["fixture"].ready
        assert json.loads(cache.read_text(encoding="utf-8"))["fixture"]["tools"] == []
    finally:
        await host.close()


async def test_dynamic_catalogue_invalid_refresh_cannot_revive_stale_tools_at_next_boot(
        monkeypatch, tmp_path):
    # Use actual cache I/O: the live route controls deliberately stub it, so
    # they cannot observe a retired tool being offered by the next lazy boot.
    load_catalogue = mh._load_catalogue
    host, wire, publications = await _wire_host(monkeypatch)
    cache = tmp_path / "catalogue.json"
    monkeypatch.setattr(mh, "MCP_CATALOGUE_PATH", cache)
    monkeypatch.setattr(mh, "_load_catalogue", load_catalogue)
    monkeypatch.setattr(host, "_remember_tools", mh.MCPHost._remember_tools.__get__(host))
    conn = host.servers["fixture"]
    host._remember_tools("fixture", conn.spec, conn)
    saved = load_catalogue()
    other = {"fingerprint": "unrelated-server", "tools": [{"name": "keep"}]}
    saved["other"] = other
    cache.write_text(json.dumps(saved), encoding="utf-8")
    restarted = mh.MCPHost(_Bus())
    new_wire = _CatalogueWire()
    new_wire.tools = [{"name": "freshly_discovered"}]
    starts = []

    async def start(replacement):
        starts.append(replacement.name)
        replacement.proc = new_wire
        replacement.protocol_version = "2025-11-25"
        await replacement._list_tools()
        replacement.ready = True
        return True

    try:
        assert saved["fixture"]["tools"] == conn.tools
        wire.tools = [{"name": "invalid", "inputSchema": {"required": 7}}]
        wire.changed()
        await _eventually(lambda: bool(publications))
        assert publications == [[]] and not conn.ready
        assert (await host.call("fixture", "old", {})).kind == "unavailable"
        await host.close()
        monkeypatch.setattr(mh.MCPServerConn, "start", start)
        await restarted.connect_all()
        assert starts == ["fixture"], "retired cache bypassed fresh discovery"
        assert "fixture" not in restarted.parked
        assert restarted.servers["fixture"].tools == new_wire.tools
        assert load_catalogue()["other"] == other
    finally:
        await host.close()
        await restarted.close()


async def test_dynamic_catalogue_disable_retires_refresh_and_ignores_old_callback(monkeypatch):
    host, wire, publications = await _wire_host(monkeypatch)
    conn = host.servers["fixture"]
    monkeypatch.setattr(host, "_write_config", lambda _config: True)
    wire.hold_list = True
    try:
        wire.changed()
        await _eventually(lambda: wire.held is not None)
        assert "disabled" in await host.set_enabled("fixture", False)
        conn._on_catalogue_changed(conn)
        assert publications == [[]] and host.servers == {}
        assert not [t for t in asyncio.all_tasks() if not t.done()
                    and t.get_name().startswith("orion-mcp-")]
    finally:
        await host.close()


async def test_dynamic_catalogue_reconnect_owns_new_reader_and_ignores_old_callback(monkeypatch):
    host, old_wire, publications = await _wire_host(monkeypatch)
    old_conn = host.servers["fixture"]
    new_wire = _CatalogueWire()
    new_wire.tools = [{"name": "replacement"}]

    async def start(conn):
        conn.proc = new_wire
        conn.protocol_version = "2025-11-25"
        await conn._list_tools()
        conn.ready = True
        return True

    monkeypatch.setattr(mh.MCPServerConn, "start", start)
    old_wire.hold_list = True
    try:
        old_wire.changed()
        await _eventually(lambda: old_wire.held is not None)
        assert await host.reconnect("fixture")
        old_conn._on_catalogue_changed(old_conn)
        assert publications == [[{"name": "replacement"}]]
        assert old_wire.closed
        assert len([t for t in asyncio.all_tasks() if not t.done()
                    and t.get_name().startswith("orion-mcp-receive-")]) == 1
        new_wire.tools = [{"name": "replacement_changed"}]
        new_wire.changed()
        await _eventually(lambda: len(publications) == 2)
        assert publications[-1] == new_wire.tools
    finally:
        await host.close()


async def test_dynamic_catalogue_idle_pipe_failure_retires_routes(monkeypatch):
    host, wire, publications = await _wire_host(monkeypatch)
    try:
        wire.stdout.feed_eof()
        await _eventually(lambda: bool(publications))
        assert publications == [[]]
        assert host.health_snapshot() == {"fixture": "DOWN"}
        assert (await host.call("fixture", "old", {})).kind == "unavailable"
    finally:
        await host.close()


async def test_dynamic_catalogue_cancelled_call_late_reply_cannot_answer_next_id(monkeypatch):
    host, wire, _publications = await _wire_host(monkeypatch)
    wire.hold_call = True
    cancelled = asyncio.create_task(host.call("fixture", "old", {}))
    try:
        await _eventually(lambda: wire.held is not None)
        held = wire.held
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        wire.hold_call = False
        wire.answer(held, {"content": [{"type": "text", "text": "stale"}]})
        assert await host.call("fixture", "old", {}) == "done"
        assert not host.servers["fixture"].busy()
    finally:
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        await host.close()


async def test_dynamic_catalogue_deadline_starts_after_an_admitted_call_releases_the_wire(monkeypatch):
    host, wire, publications = await _wire_host(monkeypatch)
    monkeypatch.setattr(mh, "_DISCOVER_TIMEOUT", 0.03)
    wire.hold_call = True
    pending = asyncio.create_task(host.call("fixture", "old", {}))
    try:
        await _eventually(lambda: wire.held is not None)
        held = wire.held
        wire.tools = [{"name": "after_long_call"}]
        wire.changed()
        await asyncio.sleep(0.06)
        assert host.servers["fixture"].ready and publications == []
        wire.answer(held, {"content": [{"type": "text", "text": "admitted result"}]})
        assert await pending == "admitted result"
        await _eventually(lambda: bool(publications))
        assert publications == [[{"name": "after_long_call"}]]
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await host.close()


async def test_dynamic_catalogue_refresh_owns_wire_across_pages_until_publication(monkeypatch):
    host, wire, publications = await _wire_host(monkeypatch)
    wire.hold_list = True
    wire.hold_cursor = "page2"
    pending = None
    try:
        wire.changed()
        await _eventually(lambda: wire.held is not None)
        first = wire.held
        pending = asyncio.create_task(host.call("fixture", "old", {}))
        await asyncio.sleep(0.01)
        wire.hold_list = False
        wire.answer(first, {"tools": [{"name": "first"}], "nextCursor": "page2"})
        await _eventually(lambda: wire.held.get("params", {}).get("cursor") == "page2")
        assert publications == [] and not pending.done()
        assert not any(r.get("method") == "tools/call" for r in wire.requests)
        wire.answer(wire.held, {"tools": [{"name": "second"}]})
        assert await pending == "done"
        await _eventually(lambda: bool(publications))
        assert [tool["name"] for tool in publications[0]] == ["first", "second"]
        assert [r["method"] for r in wire.requests] == [
            "tools/list", "tools/list", "tools/list", "tools/call"]
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await host.close()


async def test_dynamic_catalogue_refresh_deadline_still_retires_unresponsive_discovery(monkeypatch):
    host, wire, publications = await _wire_host(monkeypatch)
    monkeypatch.setattr(mh, "_DISCOVER_TIMEOUT", 0.03)
    wire.hold_list = True
    try:
        wire.changed()
        await _eventually(lambda: wire.held is not None)
        await _eventually(lambda: bool(publications))
        conn = host.servers["fixture"]
        assert publications == [[]] and not conn.ready
        assert conn._pending == {} and not conn.busy()
        assert host.health_snapshot() == {"fixture": "DOWN"}
    finally:
        await host.close()


async def test_dynamic_catalogue_old_failed_callback_cannot_retire_replacement_cache(monkeypatch, tmp_path):
    load_catalogue = mh._load_catalogue
    host, _wire, _publications = await _wire_host(monkeypatch)
    cache = tmp_path / "catalogue.json"
    monkeypatch.setattr(mh, "MCP_CATALOGUE_PATH", cache)
    monkeypatch.setattr(mh, "_load_catalogue", load_catalogue)
    monkeypatch.setattr(host, "_remember_tools", mh.MCPHost._remember_tools.__get__(host))
    old = host.servers["fixture"]
    replacement = mh.MCPServerConn("fixture", old.spec, host.bus)
    replacement.ready = True
    replacement.tools = [{"name": "replacement"}]
    host.servers["fixture"] = replacement
    host._remember_tools("fixture", replacement.spec, replacement)
    try:
        old.ready = False
        old._catalogue_refresh_failed = True
        old._on_catalogue_changed(old)
        assert load_catalogue()["fixture"]["tools"] == replacement.tools
    finally:
        await old.stop()
        await host.close()


def test_mcp_receipt_preserves_audio_binary_resource_and_structured_only_results():
    blocks = [{"type": "audio", "mimeType": "audio/wav", "data": "aGVsbG8="},
              {"type": "resource", "resource": {"uri": "data://fixture/blob", "blob": "aGVsbG8=",
                                                   "mimeType": "application/octet-stream"}}]
    reply = mh.MCPServerConn._render({"content": blocks})
    assert not reply.is_error and reply.content == blocks
    structured = mh.MCPServerConn._render({"content": [], "structuredContent": {"count": 4}})
    assert not structured.is_error and structured.structured_content == {"count": 4}
    assert '"count": 4' in structured


@pytest.mark.parametrize("block", [
    {"type": []}, {"type": "image", "data": "bytes"},
    {"type": "resource", "resource": {"uri": "data://fixture", "text": "a", "blob": "b"}},
])
def test_mcp_receipt_preserves_protocol_failure_for_invalid_media(block):
    assert mh.MCPServerConn._render({"content": [block]}).kind == "protocol"

SERVER = textwrap.dedent(r'''
    import json
    import sys
    from pathlib import Path

    mode = sys.argv[1]
    state = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    modern = mode in {"modern", "modern_bad_version", "modern_bad_tool_result", "modern_slow_probe"}
    probes = 0

    def send(request, *, result=None, error=None):
        reply = {"jsonrpc": "2.0", "id": request["id"]}
        if error is not None:
            reply["error"] = error
        else:
            reply["result"] = result
        print(json.dumps(reply), flush=True)

    def tools():
        names = ["echo", "write_note"]
        if mode == "schema":
            names = [state.read_text(encoding="utf-8").strip()]
        return [{"name": name, "description": name,
                 "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}
                for name in names]

    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request.get("method")
        params = request.get("params") or {}
        if method == "server/discover":
            probes += 1
            if mode == "modern_slow_probe" and probes == 1:
                import time
                time.sleep(0.2)
            if mode == "legacy_closes_probe":
                sys.exit(0)
            if mode == "bad_json":
                print("not a JSON response", flush=True)
                sys.exit(0)
            if mode == "bad_rpc":
                print(json.dumps({"id": request["id"], "result": {}}), flush=True)
                sys.exit(0)
            if modern:
                send(request, result={"resultType": "complete",
                      "supportedVersions": (["2099-01-01"] if mode == "modern_bad_version"
                                            else ["2026-07-28"]),
                      "capabilities": {"tools": {}},
                      "cacheScope": "private", "ttlMs": 0})
            else:
                send(request, error={"code": -32601, "message": "Method not found"})
            continue
        if method == "initialize":
            if modern:
                send(request, error={"code": -32022, "message": "No legacy handshake",
                                     "data": {"supported": ["2026-07-28"]}})
            else:
                version = "2099-01-01" if mode == "bad_version" else "2024-11-05"
                send(request, result={"protocolVersion": version,
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}})
            continue
        if modern:
            meta = params.get("_meta") or {}
            if (meta.get("io.modelcontextprotocol/protocolVersion") != "2026-07-28"
                    or not isinstance(meta.get("io.modelcontextprotocol/clientCapabilities"), dict)):
                send(request, error={"code": -32602, "message": "missing 2026 request metadata"})
                continue
        if method == "tools/list":
            result = {"tools": tools()}
            if modern:
                result["resultType"] = "complete"
            send(request, result=result)
        elif method == "tools/call":
            name = params.get("name")
            if mode == "drop_once" and not state.exists():
                state.write_text("dropped", encoding="utf-8")
                sys.exit(0)
            if mode == "bad_tool_rpc":
                print(json.dumps({"id": request["id"], "result": {"content": []}}), flush=True)
                continue
            if mode == "bad_tool_shape":
                send(request, result={"content": "not a list"})
                continue
            if mode == "slow_call" and not state.exists():
                import time
                state.write_text("slow request started", encoding="utf-8")
                time.sleep(1)
            if name == "write_note":
                state.write_text(str((params.get("arguments") or {}).get("text")), encoding="utf-8")
            text = "called " + str(name)
            if name == "echo" and (params.get("arguments") or {}).get("text"):
                text += ":" + str(params["arguments"]["text"])
            result = {"content": [{"type": "text", "text": text}]}
            if modern and mode != "modern_bad_tool_result":
                result["resultType"] = "complete"
            send(request, result=result)
        else:
            send(request, error={"code": -32601, "message": "Method not found"})
''')


class _Signal:
    def __init__(self) -> None:
        self.events = []

    def emit(self, *args) -> None:
        self.events.append(args)


class _Bus:
    def __init__(self) -> None:
        self.log = _Signal()
        self.confirm_action = _Signal()


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "server.py"
    path.write_text(SERVER, encoding="utf-8")
    return path


def _spec(script: Path, mode: str, state: Path | None = None, **extra):
    args = [str(script), mode]
    if state is not None:
        args.append(str(state))
    return {"enabled": True, "command": sys.executable, "args": args,
            "env": {}, **extra}


@pytest.mark.parametrize("mode", ["legacy", "legacy_closes_probe", "modern"])
async def test_stdio_negotiates_both_eras_and_calls_with_their_wire_contract(script, mode):
    conn = mh.MCPServerConn("fixture", _spec(script, mode), _Bus())
    try:
        assert await conn.start(), conn.last_error
        assert conn.protocol_version == ("2026-07-28" if mode == "modern" else "2024-11-05")
        assert {t["name"] for t in conn.tools} == {"echo", "write_note"}
        reply = await conn.call_tool("echo", {"text": "hello"})
        assert reply == "called echo:hello" and not reply.is_error
    finally:
        await conn.stop()


async def test_slow_modern_probe_is_recovered_without_an_incompatible_handshake(script, monkeypatch):
    monkeypatch.setattr(mh, "_DISCOVER_TIMEOUT", 0.08)
    conn = mh.MCPServerConn("fixture", _spec(script, "modern_slow_probe"), _Bus())
    try:
        assert await conn.start(), conn.last_error
        assert conn.protocol_version == "2026-07-28"
        assert not (await conn.call_tool("echo", {"text": "quiet startup"})).is_error
    finally:
        await conn.stop()


@pytest.mark.parametrize("mode,expected", [
    ("bad_version", "unsupported initialize protocolVersion"),
    ("modern_bad_version", "invalid or unsupported server/discover result"),
    ("bad_json", "invalid JSON-RPC output"),
    ("bad_rpc", "invalid JSON-RPC response"),
])
async def test_invalid_handshakes_fail_closed_and_reap_the_process(script, mode, expected):
    conn = mh.MCPServerConn("fixture", _spec(script, mode), _Bus())
    assert not await conn.start()
    assert expected in conn.last_error
    assert conn.proc is None and not conn.ready


@pytest.mark.parametrize("mode", ["bad_tool_rpc", "bad_tool_shape",
                                  "modern_bad_tool_result"])
async def test_invalid_tool_replies_are_typed_and_mark_health_down(script, tmp_path, monkeypatch, mode):
    config = {"servers": {"fixture": _spec(script, mode)}}
    monkeypatch.setattr(mh, "load_config", lambda: config)
    monkeypatch.setattr(mh, "MCP_CATALOGUE_PATH", tmp_path / "catalogue.json")
    host = mh.MCPHost(_Bus())
    try:
        await host.connect_all()
        reply = await host.call("fixture", "echo", {})
        assert reply.kind == "protocol" and reply.is_error
        assert host.health_snapshot() == {"fixture": "DOWN"}
    finally:
        await host.close()


async def test_cancelling_a_tool_call_does_not_misattribute_its_late_reply(script, tmp_path):
    marker = tmp_path / "slow.txt"
    conn = mh.MCPServerConn("fixture", _spec(script, "slow_call", marker), _Bus())
    try:
        assert await conn.start(), conn.last_error
        task = asyncio.create_task(conn.call_tool("echo", {"text": "cancelled"}))
        for _ in range(100):
            if conn.busy():
                break
            await asyncio.sleep(0.01)
        assert conn.busy()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not conn.busy()
        assert await conn.call_tool("echo", {"text": "mine"}) == "called echo:mine"
    finally:
        await conn.stop()


def _mcp_sdk_server_is_installed():
    # A reports/mcp directory on the runner's path is an implicit namespace,
    # not an installed SDK. Only a real server module admits the SDK smoke.
    return (importlib.util.find_spec("mcp") is not None
            and importlib.util.find_spec("mcp.server") is not None)


@pytest.mark.parametrize("parent,server,expected", [
    (None, None, False), (object(), None, False), (object(), object(), True),
])
def test_mcp_sdk_gate_requires_server_module_not_just_a_namespace(monkeypatch, parent, server, expected):
    lookups = []

    def find_spec(name):
        lookups.append(name)
        return {"mcp": parent, "mcp.server": server}[name]

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert _mcp_sdk_server_is_installed() is expected
    assert lookups == (["mcp"] if parent is None else ["mcp", "mcp.server"])


async def test_installed_current_python_sdk_stdio_when_available(tmp_path):
    """Optional real-SDK gate; the contract fixture above runs in every CI."""
    if not _mcp_sdk_server_is_installed():
        pytest.skip("MCP Python SDK v2 is not installed in this environment")
    from mcp import server as sdk_server
    if not hasattr(sdk_server, "MCPServer"):
        pytest.skip("installed MCP Python SDK predates v2")
    script = tmp_path / "sdk_server.py"
    script.write_text(textwrap.dedent('''
        from mcp.server import MCPServer
        mcp = MCPServer("ORION compatibility fixture")
        @mcp.tool()
        def echo(text: str) -> str:
            """Echo a small text value."""
            return "sdk:" + text
        if __name__ == "__main__":
            mcp.run(transport="stdio")
    '''), encoding="utf-8")
    conn = mh.MCPServerConn("sdk", _spec(script, ""), _Bus())
    conn.args = [str(script)]
    try:
        assert await conn.start(), conn.last_error
        assert conn.protocol_version == "2026-07-28"
        assert "echo" in {tool["name"] for tool in conn.tools}
        assert "sdk:hello" in await conn.call_tool("echo", {"text": "hello"})
    finally:
        await conn.stop()


async def test_dropped_server_has_down_health_then_reconnects(script, tmp_path, monkeypatch):
    marker = tmp_path / "dropped.txt"
    config = {"servers": {"fixture": _spec(script, "drop_once", marker)}}
    monkeypatch.setattr(mh, "load_config", lambda: config)
    monkeypatch.setattr(mh, "MCP_CATALOGUE_PATH", tmp_path / "catalogue.json")
    host = mh.MCPHost(_Bus())
    try:
        await host.connect_all()
        assert host.health_snapshot() == {"fixture": "OK"}
        failed = await host.call("fixture", "echo", {})
        assert failed.kind == "unavailable"
        assert host.health_snapshot() == {"fixture": "DOWN"}
        assert "connection dropped" in host.list_servers()
        assert await host.reconnect("fixture")
        assert host.health_snapshot() == {"fixture": "OK"}
        assert await host.call("fixture", "echo", {}) == "called echo"
    finally:
        await host.close()


async def test_schema_reconnect_replaces_routes_and_disabling_removes_them(
        script, tmp_path, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher

    schema = tmp_path / "schema.txt"
    schema.write_text("old_tool", encoding="utf-8")
    config = {"servers": {"fixture": _spec(script, "schema", schema)}}
    monkeypatch.setattr(mh, "load_config", lambda: config)
    monkeypatch.setattr(mh, "MCP_CATALOGUE_PATH", tmp_path / "catalogue.json")
    host = mh.MCPHost(_Bus())
    dispatcher = OrionDispatcher.__new__(OrionDispatcher)
    try:
        await host.connect_all()
        dispatcher.register_mcp_tools("fixture", host.servers["fixture"].tools)
        host._on_reconnect = dispatcher.register_mcp_tools
        assert "mcp__fixture__old_tool" in dispatcher._mcp_routes
        schema.write_text("new_tool", encoding="utf-8")
        assert await host.reconnect("fixture")
        assert "mcp__fixture__old_tool" not in dispatcher._mcp_routes
        assert "mcp__fixture__new_tool" in dispatcher._mcp_routes
        config["servers"]["fixture"]["enabled"] = False
        assert not await host.reconnect("fixture")
        assert "mcp__fixture__new_tool" not in dispatcher._mcp_routes
    finally:
        await host.close()


async def test_write_tool_requires_local_approval_on_both_dispatch_routes(
        script, tmp_path, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.system_guard import SystemActionGuard

    written = tmp_path / "written.txt"
    config = {"servers": {"fixture": _spec(script, "legacy", written,
                                           confirm=True, confirm_reason="writes a note")}}
    monkeypatch.setattr(mh, "load_config", lambda: config)
    monkeypatch.setattr(mh, "MCP_CATALOGUE_PATH", tmp_path / "catalogue.json")
    bus = _Bus()
    host = mh.MCPHost(bus)
    dispatcher = OrionDispatcher.__new__(OrionDispatcher)
    dispatcher.mcp_host = host
    dispatcher.system_guard = SystemActionGuard()
    dispatcher.bus = bus
    try:
        await host.connect_all()
        dispatcher.register_mcp_tools("fixture", host.servers["fixture"].tools)
        generic = await dispatcher.mcp_tool({"action": "call", "server": "fixture",
                                             "tool": "write_note", "arguments": {"text": "secret"}})
        direct = await dispatcher.dispatch("mcp__fixture__write_note", {"text": "secret"})
        assert not generic.ok and not direct.ok and not written.exists()
        pending = await dispatcher.mcp_tool({"action": "call", "server": "fixture",
                                             "tool": "write_note", "arguments": {"text": "secret"},
                                             "confirm": True})
        assert not pending.ok and bus.confirm_action.events and not written.exists()
    finally:
        await host.close()


# ── ORION's own servers in the packaged app (owner's report, 7 Oct 2026) ─────

def test_frozen_bundled_servers_run_through_the_executable(monkeypatch):
    """Frozen, a bundled server used to be `python <_internal>/mcp_world.py`:
    any Python on the PATH, on a loose file that cannot import the rest of
    ORION (world_feeds lives in the exe's archive)."""
    from orion_core import mcp_bundled
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Apps\ORION\ORION.exe")
    for name in mcp_bundled.BUNDLED:
        assert mcp_bundled.launch_spec(name) == (r"C:\Apps\ORION\ORION.exe",
                                                 ["--mcp-server", name])
    saved = {"servers": {
        "orion_world": {"command": "python",
                        "args": [r"C:\Apps\ORION\_internal\orion_core\mcp_world.py"]},
        "somebody_elses": {"command": "python", "args": [r"C:\tools\mcp_world.py"]}}}
    servers = mh._host_runtime_servers(saved)["servers"]
    assert servers["orion_world"]["args"] == ["--mcp-server", "orion_world"]
    assert servers["somebody_elses"]["args"] == [r"C:\tools\mcp_world.py"]


def test_an_unknown_bundled_server_name_is_refused():
    from orion_core import mcp_bundled
    assert mcp_bundled.run(["not_a_server"]) == 2
    assert mcp_bundled.run([]) == 2


def test_the_entry_point_serves_a_bundled_server_over_stdio():
    """`orion.py --mcp-server` is the path the exe takes: before the
    single-instance lock and Qt, straight into the server."""
    import json
    import subprocess
    root = Path(__file__).resolve().parents[1]
    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                          "clientInfo": {"name": "test", "version": "1"}}}
    proc = subprocess.run([sys.executable, str(root / "orion.py"), "--mcp-server", "orion_telemetry"],
                          input=json.dumps(request) + "\n", capture_output=True, text=True,
                          cwd=root, timeout=60)
    reply = json.loads(proc.stdout.splitlines()[0])
    assert reply["id"] == 1 and "result" in reply, proc.stderr[-800:]
