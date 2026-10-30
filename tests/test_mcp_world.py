"""The local World MCP is read-only, bounded and honest about availability."""

from __future__ import annotations

import pytest

from orion_core import mcp_host, mcp_world


async def test_handshake_and_discovery_need_no_network():
    reply = await mcp_world.respond({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                     "params": {"protocolVersion": "2025-06-18"}})
    assert reply["result"]["protocolVersion"] == "2025-06-18"
    reply = await mcp_world.respond({"id": 2, "method": "tools/list"})
    tools = reply["result"]["tools"]
    assert len(tools) == 6          # five world layers (incl. NASA EONET events) + status
    assert all(tool["annotations"]["readOnlyHint"] for tool in tools)
    assert all(not tool["annotations"]["destructiveHint"] for tool in tools)


@pytest.mark.parametrize("arguments", [{"url": "http://127.0.0.1/private"}, {"limit": True},
                                        {"limit": 201}, {"limit": 0}, {"limit": "2"}, None])
async def test_tools_refuse_arbitrary_urls_and_bad_limits(arguments):
    reply = await mcp_world.respond({"id": 1, "method": "tools/call", "params": {
        "name": "world_earthquakes", "arguments": arguments}})
    assert reply["result"]["isError"]


async def test_structured_result_is_bounded_and_unavailable_is_an_error(monkeypatch):
    class Service:
        async def snapshot(self, layers, **_kwargs):
            return {"generated_at": "2026-09-30T12:00:00Z", "layers": {
                layers[0]: {"items": [{"id": "1"}, {"id": "2"}], "status": "unavailable",
                            "error": "provider unavailable", "stale": True}}}

    monkeypatch.setattr(mcp_world, "get_world_feeds", Service)
    reply = await mcp_world.respond({"id": 1, "method": "tools/call", "params": {
        "name": "world_weather", "arguments": {"limit": 1}}})
    result = reply["result"]
    data = result["structuredContent"]
    assert data["record_count"] == 2 and data["truncated"]
    assert data["items"] == [{"id": "1"}]
    assert result["isError"]


async def test_status_only_reads_cache(monkeypatch):
    class Service:
        async def snapshot(self, *, refresh):
            assert refresh is False
            return {"generated_at": "fixture", "layers": {"weather": {
                "items": [], "status": "unavailable", "error": None}}}

    monkeypatch.setattr(mcp_world, "get_world_feeds", Service)
    data = await mcp_world.sample("world_feed_status", {})
    assert data["layers"]["weather"]["record_count"] == 0
    assert "items" not in data["layers"]["weather"]


async def test_unknown_method_notification_and_params_validation():
    assert await mcp_world.respond({"method": "notifications/initialized"}) is None
    assert (await mcp_world.respond({"id": 2, "method": "not-supported"}))["error"]["code"] == -32601
    assert (await mcp_world.respond({"id": 2, "method": "tools/list", "params": []}))["error"]["code"] == -32602


def test_migration_enables_new_local_world_server_and_preserves_disabled_choice():
    data = {"_revision": "mark32", "servers": {}}
    assert mcp_host.migrate_config(data)
    assert data["servers"]["orion_world"]["enabled"]
    assert data["servers"]["orion_world"]["args"][0].endswith("mcp_world.py")
    data["servers"]["orion_world"]["enabled"] = False
    data["_revision"] = "older"
    mcp_host.migrate_config(data)
    assert not data["servers"]["orion_world"]["enabled"]
