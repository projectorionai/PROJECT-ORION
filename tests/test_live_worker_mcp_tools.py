"""
Tests for _build_config reading the dispatcher's per-instance
TOOL_DECLARATIONS (Mark XXI, Track E1) — so a connected MCP server's tools
(and any freshly forged tool) actually reach the live Gemini session's
function-calling schema, not only the static module-level list frozen at
import time.
"""

from __future__ import annotations

import sys
import types as pytypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.live_worker as lw
from orion_core.live_worker import GenAILiveWorker, TOOL_DECLARATIONS


def _worker(dispatcher_declarations=None) -> pytypes.SimpleNamespace:
    w = pytypes.SimpleNamespace()
    w.router = pytypes.SimpleNamespace(system_instruction=lambda: "persona")
    w._search_tool_enabled = False
    w._resumption_handle = None
    if dispatcher_declarations is None:
        w.dispatcher = pytypes.SimpleNamespace()   # no TOOL_DECLARATIONS attribute at all
    else:
        w.dispatcher = pytypes.SimpleNamespace(TOOL_DECLARATIONS=dispatcher_declarations)
    return w


def _force_dict_fallback(monkeypatch) -> None:
    # _build_config prefers the real google-genai LiveConnectConfig (a
    # Pydantic-style model, not subscriptable) when the SDK is installed;
    # this test only cares about the plain "tools" list *passed in* to
    # that constructor, so force the method's own except-path, which
    # builds the equivalent plain dict this test can inspect directly.
    def _boom(*a, **k):
        raise RuntimeError("forced for test")
    monkeypatch.setattr(lw.types, "LiveConnectConfig", _boom)
    # These tests are about WHICH source list is used; the core/gateway filter
    # (tool_gateway) is tested separately below.
    monkeypatch.setenv("ORION_LIVE_TOOLS", "full")


def test_falls_back_to_the_static_list_when_dispatcher_has_none(monkeypatch):
    _force_dict_fallback(monkeypatch)
    w = _worker(dispatcher_declarations=None)
    config = GenAILiveWorker._build_config(w)
    assert config["tools"][0]["function_declarations"] == TOOL_DECLARATIONS


def test_uses_the_dispatchers_per_instance_list_when_present(monkeypatch):
    _force_dict_fallback(monkeypatch)
    widened = list(TOOL_DECLARATIONS) + [{"name": "mcp__gmail__send_email", "description": "x",
                                          "parameters": {"type": "OBJECT", "properties": {}}}]
    w = _worker(dispatcher_declarations=widened)
    config = GenAILiveWorker._build_config(w)
    names = {t["name"] for t in config["tools"][0]["function_declarations"]}
    assert "mcp__gmail__send_email" in names


def test_falls_back_when_dispatcher_declarations_is_falsy(monkeypatch):
    _force_dict_fallback(monkeypatch)
    # An empty list is falsy — must not silently produce a schema with zero
    # tools; fall back to the real static list instead.
    w = _worker(dispatcher_declarations=[])
    config = GenAILiveWorker._build_config(w)
    assert config["tools"][0]["function_declarations"] == TOOL_DECLARATIONS


# ── the Live tool gateway (core tools + find_tool/use_tool) ─────────────────

def test_the_live_session_carries_the_core_set_and_the_gateway(monkeypatch):
    monkeypatch.setattr(lw.types, "LiveConnectConfig",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("dict path")))
    monkeypatch.delenv("ORION_LIVE_TOOLS", raising=False)
    widened = list(TOOL_DECLARATIONS) + [{"name": "mcp__notion__search", "description": "x",
                                          "parameters": {"type": "OBJECT", "properties": {}}}]
    w = _worker(dispatcher_declarations=widened)
    config = GenAILiveWorker._build_config(w)
    names = {t["name"] for t in config["tools"][0]["function_declarations"]}
    from orion_core.tool_gateway import LIVE_CORE_TOOLS
    assert {"find_tool", "use_tool"} <= names
    assert names <= LIVE_CORE_TOOLS
    assert "mcp__notion__search" not in names and "chess" not in names
    assert "find_tool" in config["system_instruction"]


async def test_a_gated_tool_is_still_reachable_through_use_tool():
    from orion_core.data import ToolResult
    from orion_core import tool_gateway
    calls = []

    class _D:
        TOOL_DECLARATIONS = list(TOOL_DECLARATIONS) + [
            {"name": "mcp__notion__search", "description": "Search Notion.",
             "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}}}]

        async def dispatch_chain(self, name, args):
            calls.append((name, args))
            return ToolResult("found 3 pages")

    found = tool_gateway.find_tool(_D(), {"need": "search my notion pages"})
    assert "mcp__notion__search" in found.text
    result = await tool_gateway.use_tool(_D(), {"name": "mcp__notion__search",
                                                "arguments": '{"query": "budget"}'})
    assert result.ok and calls == [("mcp__notion__search", {"query": "budget"})]
    refused = await tool_gateway.use_tool(_D(), {"name": "use_tool", "arguments": "{}"})
    assert not refused.ok
