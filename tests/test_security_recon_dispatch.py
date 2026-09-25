"""
Tests for OrionDispatcher.security_recon_tool — the routing layer between
the model-facing 'security_recon' tool and SecurityReconService, plus the
write_tool action's delegation to the existing Forge pipeline (writing a
security tool touches no target, so it's always permitted — routed straight
through self.forge_tool rather than SecurityReconService, which only owns
network-facing actions).
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.dispatcher import OrionDispatcher


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubRecon:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def authorize_target(self, target):
        self.calls.append(("authorize_target", (target,)))
        return ToolResult(f"authorized {target}")

    def list_authorized(self):
        self.calls.append(("list_authorized", ()))
        return ToolResult("none")

    async def scan_host(self, target, ports):
        self.calls.append(("scan_host", (target, ports)))
        return ToolResult(f"scanned {target}")

    async def packet_capture(self, target, count, timeout):
        self.calls.append(("packet_capture", (target, count, timeout)))
        return ToolResult("captured")

    async def craft_packet(self, target):
        self.calls.append(("craft_packet", (target,)))
        return ToolResult("pinged")

    async def cve_lookup(self, query):
        self.calls.append(("cve_lookup", (query,)))
        return ToolResult("cve info")


def _dispatcher(recon=None) -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _StubBus()
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)
    d.security_recon = recon
    d.forge_calls: list[dict] = []

    async def _fake_forge_tool(args):
        d.forge_calls.append(args)
        return ToolResult(f"forged {args.get('tool_name')}")

    d.forge_tool = _fake_forge_tool
    return d


def test_unavailable_when_service_not_wired():
    d = _dispatcher(recon=None)
    result = asyncio.run(d.security_recon_tool({"action": "list_authorized"}))
    assert not result.ok


def test_routes_authorize_target():
    recon = _StubRecon()
    d = _dispatcher(recon)
    result = asyncio.run(d.security_recon_tool({"action": "authorize_target", "target": "10.0.0.5"}))
    assert result.text == "authorized 10.0.0.5"
    assert recon.calls == [("authorize_target", ("10.0.0.5",))]


def test_routes_scan_host_with_ports():
    recon = _StubRecon()
    d = _dispatcher(recon)
    asyncio.run(d.security_recon_tool({"action": "scan_host", "target": "10.0.0.5", "ports": "1-100"}))
    assert recon.calls == [("scan_host", ("10.0.0.5", "1-100"))]


def test_routes_packet_capture_with_defaults():
    recon = _StubRecon()
    d = _dispatcher(recon)
    asyncio.run(d.security_recon_tool({"action": "packet_capture", "target": "10.0.0.5"}))
    assert recon.calls == [("packet_capture", ("10.0.0.5", 10, 10.0))]


def test_routes_cve_lookup():
    recon = _StubRecon()
    d = _dispatcher(recon)
    asyncio.run(d.security_recon_tool({"action": "cve_lookup", "query": "log4j"}))
    assert recon.calls == [("cve_lookup", ("log4j",))]


def test_write_tool_delegates_to_forge_not_recon():
    recon = _StubRecon()
    d = _dispatcher(recon)
    result = asyncio.run(d.security_recon_tool({
        "action": "write_tool", "description": "a simple port scanner",
        "tool_name": "my_scanner",
    }))
    assert recon.calls == []  # never touched SecurityReconService
    assert len(d.forge_calls) == 1
    call = d.forge_calls[0]
    assert call["tool_name"] == "my_scanner"
    assert "port scanner" in call["tool_plan"]
    assert result.text == "forged my_scanner"


def test_write_tool_requires_a_description():
    d = _dispatcher(_StubRecon())
    result = asyncio.run(d.security_recon_tool({"action": "write_tool"}))
    assert not result.ok
    assert d.forge_calls == []


def test_unknown_action_reports_cleanly():
    d = _dispatcher(_StubRecon())
    result = asyncio.run(d.security_recon_tool({"action": "not_a_real_action"}))
    assert not result.ok
    assert "security_recon actions" in result.text
