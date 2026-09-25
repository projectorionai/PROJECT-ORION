"""A web URL is read by the fetch MCP, while local files stay local."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from orion_core.dispatch_schema import TOOL_DECLARATIONS
from orion_core.dispatch_web import WebDispatchMixin, mcp_call_failed
from orion_core.tool_gateway import LIVE_CORE_TOOLS, find_tool


class _Host:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, str, dict]] = []

    async def call(self, server: str, tool: str, arguments: dict) -> str:
        self.calls.append((server, tool, arguments))
        return self.result


async def _fetch_url_uses_the_mcp_and_can_resume_a_long_page():
    host = _Host("Contents of https://example.com/report:\n# Report")
    target = SimpleNamespace(mcp_host=host)
    result = await WebDispatchMixin.fetch_url(target, {
        "url": "https://example.com/report", "start_index": 8000,
        "max_length": 6000})
    assert result.ok and "# Report" in result.text
    assert host.calls == [("fetch", "fetch", {
        "url": "https://example.com/report", "start_index": 8000,
        "max_length": 6000})]


async def _fetch_url_rejects_local_paths_and_reports_mcp_failure():
    host = _Host("MCP server 'fetch' was paused to save memory and could not restart: failed")
    target = SimpleNamespace(mcp_host=host)
    local = await WebDispatchMixin.fetch_url(target, {"url": "C:/Documents/report.pdf"})
    assert not local.ok and not host.calls
    remote = await WebDispatchMixin.fetch_url(target, {"url": "https://example.com"})
    assert not remote.ok
    assert mcp_call_failed(host.result)


def test_fetch_url_uses_the_mcp_and_can_resume_a_long_page():
    asyncio.run(_fetch_url_uses_the_mcp_and_can_resume_a_long_page())


def test_fetch_url_rejects_local_paths_and_reports_mcp_failure():
    asyncio.run(_fetch_url_rejects_local_paths_and_reports_mcp_failure())


def test_fetch_is_directly_available_to_live_and_discoverable():
    assert "fetch_url" in LIVE_CORE_TOOLS
    declarations = {item["name"]: item for item in TOOL_DECLARATIONS}
    assert "fetch_url" in declarations
    found = find_tool(SimpleNamespace(TOOL_DECLARATIONS=TOOL_DECLARATIONS), {
        "need": "fetch the content of this https web URL"})
    assert "fetch_url" in found.text
