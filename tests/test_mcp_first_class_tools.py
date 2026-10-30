"""
Tests for MCP tools becoming first-class, namespaced dispatcher tools
(Mark XXI, Track E1) — previously an MCP tool was only reachable through
the generic `mcp` list/call indirection; register_mcp_tools() appends each
connected server's tools to dispatcher.TOOL_DECLARATIONS as
"mcp__<server>__<tool>", and dispatch() routes calls to that name straight
to mcp_host.call(...).

This also covers a real, previously-dead mechanism this pass fixed as a
side effect: dynamic_loader.py's activate_tool() has always tried to
append a forged tool's schema to `dispatcher.TOOL_DECLARATIONS`, but no
per-instance attribute of that name ever existed (hasattr() was always
False), so that half of tool activation silently no-op'd. OrionDispatcher
now sets one in __init__.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.dispatcher import OrionDispatcher, TOOL_DECLARATIONS


def _dispatcher() -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.mcp_host = None
    return d


class _StubMCPHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self._response = "ok from server"

    async def call(self, server, tool, arguments):
        self.calls.append((server, tool, arguments))
        return self._response


# ── per-instance TOOL_DECLARATIONS (the underlying fix) ─────────────────────

def test_a_real_dispatcher_gets_its_own_copy_of_tool_declarations():
    from orion_core.dispatcher import OrionDispatcher as _OD
    import inspect
    # Constructing a full OrionDispatcher needs many collaborators; instead
    # verify __init__'s source actually assigns the per-instance list, so
    # this test fails loudly if that line is ever removed/renamed.
    source = inspect.getsource(_OD.__init__)
    assert "self.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)" in source


def test_per_instance_list_starts_as_a_copy_not_the_same_object():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    assert d.TOOL_DECLARATIONS == TOOL_DECLARATIONS
    assert d.TOOL_DECLARATIONS is not TOOL_DECLARATIONS
    d.TOOL_DECLARATIONS.append({"name": "test_only"})
    assert not any(t.get("name") == "test_only" for t in TOOL_DECLARATIONS)


# ── register_mcp_tools() ─────────────────────────────────────────────────────

def test_register_mcp_tools_appends_namespaced_declarations():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    count = d.register_mcp_tools("gmail", [
        {"name": "send_email", "description": "Send an email.",
         "inputSchema": {"type": "object", "properties": {
             "to": {"type": "string"}, "subject": {"type": "string"}}}},
        {"name": "list_emails", "description": "List recent emails.",
         "inputSchema": {"type": "object", "properties": {}}},
    ])
    assert count == 2
    names = {t["name"] for t in d.TOOL_DECLARATIONS}
    assert "mcp__gmail__send_email" in names
    assert "mcp__gmail__list_emails" in names


def test_register_mcp_tools_translates_json_schema_types_to_gemini_style():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    d.register_mcp_tools("gmail", [
        {"name": "send_email", "description": "Send.",
         "inputSchema": {"type": "object", "properties": {
             "to": {"type": "string"}, "cc": {"type": "array", "items": {"type": "string"}},
             "urgent": {"type": "boolean"}}}},
    ])
    decl = next(t for t in d.TOOL_DECLARATIONS if t["name"] == "mcp__gmail__send_email")
    props = decl["parameters"]["properties"]
    assert decl["parameters"]["type"] == "OBJECT"
    assert props["to"]["type"] == "STRING"
    assert props["cc"]["type"] == "ARRAY"
    assert props["cc"]["items"]["type"] == "STRING"
    assert props["urgent"]["type"] == "BOOLEAN"


def test_register_mcp_tools_defaults_to_empty_object_params_when_schema_missing():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    d.register_mcp_tools("filesystem", [{"name": "list_dir", "description": "List a dir."}])
    decl = next(t for t in d.TOOL_DECLARATIONS if t["name"] == "mcp__filesystem__list_dir")
    assert decl["parameters"] == {"type": "OBJECT", "properties": {}}


def test_register_mcp_tools_is_idempotent_on_reconnect():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    d.register_mcp_tools("gmail", [{"name": "send_email", "description": "v1"}])
    d.register_mcp_tools("gmail", [{"name": "send_email", "description": "v2"},
                                   {"name": "list_emails", "description": "new"}])
    matches = [t for t in d.TOOL_DECLARATIONS if t["name"] == "mcp__gmail__send_email"]
    assert len(matches) == 1
    assert matches[0]["description"] == "v2"
    assert any(t["name"] == "mcp__gmail__list_emails" for t in d.TOOL_DECLARATIONS)


def test_reconnect_removes_stale_tools_for_a_sanitised_server_name():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    d.register_mcp_tools("my server", [{"name": "old_tool", "description": "old"}])
    d.register_mcp_tools("my server", [{"name": "new_tool", "description": "new"}])

    names = {t["name"] for t in d.TOOL_DECLARATIONS}
    assert "mcp__my_server__old_tool" not in names
    assert "mcp__my_server__old_tool" not in d._mcp_routes
    assert "mcp__my_server__new_tool" in names


def test_register_mcp_tools_for_one_server_does_not_touch_another():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    d.register_mcp_tools("gmail", [{"name": "send_email", "description": "x"}])
    d.register_mcp_tools("filesystem", [{"name": "list_dir", "description": "y"}])
    names = {t["name"] for t in d.TOOL_DECLARATIONS}
    assert "mcp__gmail__send_email" in names
    assert "mcp__filesystem__list_dir" in names


def test_register_mcp_tools_skips_a_tool_with_no_name():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    count = d.register_mcp_tools("gmail", [{"description": "no name field"}])
    assert count == 0


def test_register_mcp_tools_lazily_initialises_on_a_bare_dispatcher():
    d = _dispatcher()   # no TOOL_DECLARATIONS/_mcp_routes set at all
    count = d.register_mcp_tools("gmail", [{"name": "send_email", "description": "x"}])
    assert count == 1
    assert any(t["name"] == "mcp__gmail__send_email" for t in d.TOOL_DECLARATIONS)


# ── dispatch() routing ──────────────────────────────────────────────────────

async def test_dispatch_routes_an_mcp_tool_by_its_namespaced_name():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    d.mcp_host = _StubMCPHost()
    d.register_mcp_tools("gmail", [{"name": "send_email", "description": "x"}])
    result = await d.dispatch("mcp__gmail__send_email", {"to": "user@example.com"})
    assert isinstance(result, ToolResult)
    assert result.ok
    assert result.text == "ok from server"
    assert d.mcp_host.calls == [("gmail", "send_email", {"to": "user@example.com"})]


async def test_dispatch_falls_through_cleanly_when_mcp_host_is_absent():
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    d.mcp_host = None
    d.register_mcp_tools("gmail", [{"name": "send_email", "description": "x"}])
    result = await d.dispatch("mcp__gmail__send_email", {})
    assert not result.ok
    assert "Unknown dispatch target" in result.text


async def test_dispatch_still_reports_unknown_for_a_genuinely_unknown_tool():
    d = _dispatcher()
    d.mcp_host = None
    result = await d.dispatch("definitely_not_a_real_tool", {})
    assert not result.ok


async def test_native_tools_are_never_shadowed_by_mcp_routing():
    # A native handler_table entry must win even if somehow an MCP route
    # shared the same name (defence in depth — MCP names are namespaced
    # with "mcp__" precisely so this collision cannot happen in practice).
    from collections import deque

    d = _dispatcher()
    d.mcp_host = _StubMCPHost()
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)

    async def _native(args):
        return ToolResult("native handler won")

    d.handler_table = lambda: {"open_app": _native}
    result = await d.dispatch("open_app", {})
    assert result.text == "native handler won"
    assert d.mcp_host.calls == []


def test_reregistering_from_a_connection_object_keeps_the_tools():
    """MCPHost.supervise calls on_reconnect(name, CONNECTION). Iterating the
    connection raised after the old tools were removed, so every supervised
    reconnect deleted that server's tools for the rest of the session."""
    import types as t
    d = _dispatcher()
    d.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    d.register_mcp_tools("gmail", [{"name": "send_email", "description": "x"}])
    conn = t.SimpleNamespace(tools=[{"name": "send_email", "description": "x"},
                                    {"name": "list_emails", "description": "y"}])
    count = d.register_mcp_tools("gmail", conn)
    names = {x["name"] for x in d.TOOL_DECLARATIONS}
    assert count == 2
    assert {"mcp__gmail__send_email", "mcp__gmail__list_emails"} <= names


# ── per-tool confirmation + the shipped server set (Sep 25 2026) ────────────
# "confirm": true gated EVERY tool on a server, so the only way to make git's
# commit ask was to make its status ask too. confirm_tools / confirm_except
# gate the tools that act and leave the ones that read alone.

def _host_with(monkeypatch, servers):
    from orion_core import mcp_host as mh
    monkeypatch.setattr(mh, "load_config", lambda: {"servers": servers})
    return mh.MCPHost.__new__(mh.MCPHost)


def test_confirm_tools_gates_only_the_listed_tools(monkeypatch):
    host = _host_with(monkeypatch, {"git": {
        "confirm_tools": ["git_commit", "git_check*"],
        "confirm_reason": "changes the repository"}})
    assert host.requires_confirmation("git", "git_commit") == \
        "git.git_commit changes the repository"
    assert host.requires_confirmation("git", "GIT_CHECKOUT")      # wildcard, any case
    assert host.requires_confirmation("git", "git_status") == ""
    assert host.requires_confirmation("git", "git_log") == ""


def test_confirm_except_frees_only_the_reading_tools(monkeypatch):
    host = _host_with(monkeypatch, {"home_assistant": {
        "confirm": True, "confirm_except": ["GetLiveContext", "GetDateTime"]}})
    assert host.requires_confirmation("home_assistant", "GetLiveContext") == ""
    assert host.requires_confirmation("home_assistant", "getdatetime") == ""
    assert host.requires_confirmation("home_assistant", "HassTurnOn")
    assert host.requires_confirmation("home_assistant", "script_unlock_door")


def test_confirm_except_cannot_free_a_server_that_spends_money(monkeypatch):
    host = _host_with(monkeypatch, {"twilio": {
        "confirm": True, "confirm_except": ["send_sms"]}})
    assert "costs money" in host.requires_confirmation("twilio", "send_sms")


def test_the_template_ships_no_retired_package_and_no_whole_home_scope():
    import json
    import os
    from orion_core import mcp_host as mh
    servers = mh._default_config()["servers"]
    text = json.dumps(servers)
    for retired in ("@modelcontextprotocol/server-github", "@modelcontextprotocol/server-postgres",
                    "@modelcontextprotocol/server-gdrive", "homeassistant-mcp",
                    "mcp-server-sqlite-npx"):
        assert retired not in text, retired
    assert os.path.expanduser("~") not in servers["filesystem"]["args"]
    assert servers["filesystem"]["args"][2:]                  # named folders instead
    for name, spec in servers.items():
        if name not in mh._RECOMMENDED:
            assert spec["enabled"] is False, f"{name} must ship off"
    for name in ("home_assistant", "git"):
        assert servers[name]["confirm"] is True, name
    assert "git_status" in servers["git"]["confirm_except"]
    assert "git_commit" not in servers["git"]["confirm_except"]
    for name in ("github", "playwright"):
        assert servers[name]["confirm_tools"], name
    # mcp-proxy at INFO logs each request's headers — the bearer token — into
    # the stderr tail a failed handshake quotes.
    ha_args = servers["home_assistant"]["args"]
    assert ha_args[ha_args.index("--log-level") + 1] == "WARNING"


def test_the_repo_config_holds_no_credential():
    """The project's config/mcp_servers.json sits in the synced project folder
    and travels with exports of it: every credential-named env value must be
    empty (a path or a flag is not a credential). Keys belong in the runtime
    config outside the project."""
    import json
    import pytest
    from orion_core import mcp_host as mh
    path = Path(__file__).resolve().parents[1] / "config" / "mcp_servers.json"
    if not path.is_file():
        pytest.skip("no project MCP config in this checkout (it is gitignored)")
    raw = path.read_text(encoding="utf-8")
    for name, spec in json.loads(raw)["servers"].items():
        for key, value in (spec.get("env") or {}).items():
            value = str(value or "").strip()
            if not mh._SECRET_RE.search(key) or not value:
                continue
            is_path = ":\\" in value or value.startswith(("/", "~"))
            assert is_path or value.lower() in {"0", "1", "true", "false"}, \
                f"{name}.{key} holds a value in a tracked file"


# Structured MCP evidence must survive actual dispatcher-to-model serialization.
import pytest as _receipt_pytest


@_receipt_pytest.mark.parametrize("kind", ["", "tool", "protocol", "timeout", "unavailable"])
def test_mcp_reply_retains_typed_receipt_through_first_class_dispatch(kind):
    import asyncio
    import json
    from orion_core.mcp_host import MCPReply

    content = [
        {"type": "text", "text": "Measured result"},
        {"type": "image", "data": "aW1hZ2U=", "mimeType": "image/png"},
        {"type": "audio", "data": "YXVkaW8=", "mimeType": "audio/wav"},
        {"type": "resource_link", "uri": "fixture://report", "name": "report"},
    ]
    reply = MCPReply("Measured result", kind, content=content,
                     structured_content={"count": 3}, meta={"fixture": "metadata"},
                     provenance={"server": "fixture", "tool": "read", "untrusted": True})
    host = _StubMCPHost()
    host._response = reply
    dispatcher = _dispatcher()
    dispatcher.mcp_host = host
    dispatcher.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    dispatcher.register_mcp_tools("fixture", [{"name": "read"}])
    result = asyncio.run(dispatcher._dispatch_mcp("mcp__fixture__read", {}))
    payload = json.loads(json.dumps(result.response_payload()))
    assert payload["ok"] is (not bool(kind))
    assert payload["result"] == "Measured result"
    assert payload["receipt"] == {
        "type": "mcp", "content": content, "structuredContent": {"count": 3},
        "_meta": {"fixture": "metadata"},
        "provenance": {"server": "fixture", "tool": "read", "untrusted": True},
        "isError": bool(kind), "errorKind": kind, "untrusted": True,
    }
    assert result.media is None and result.evidence is None
    assert host.calls == [("fixture", "read", {})]


def test_mcp_receipt_is_snapshot_at_result_creation_and_each_payload():
    from orion_core.mcp_host import MCPReply
    reply = MCPReply("summary", content=[{"type": "text", "text": "original"}],
                     structured_content={"rows": [{"value": 1}]}, provenance={"server": "fixture"})
    result = ToolResult(reply)
    reply.content[0]["text"] = "caller mutation"
    reply.structured_content["rows"][0]["value"] = 99
    payload = result.response_payload()
    assert payload["receipt"]["content"][0]["text"] == "original"
    assert payload["receipt"]["structuredContent"]["rows"][0]["value"] == 1
    payload["receipt"]["provenance"]["server"] = "changed returned payload"
    assert result.response_payload()["receipt"]["provenance"]["server"] == "fixture"


def test_explicit_receipt_does_not_replace_gemini_media_or_human_evidence():
    receipt = {"type": "fixture", "rows": [{"value": 1}]}
    media = {"data": b"raw", "mime_type": "audio/wav"}
    evidence = [{"source": "human-facing evidence"}]
    result = ToolResult("plain", media=media, evidence=evidence, receipt=receipt)
    receipt["rows"][0]["value"] = 99
    assert result.response_payload()["receipt"]["rows"][0]["value"] == 1
    assert result.media is media and result.evidence is evidence
    assert "media" not in result.response_payload() and "evidence" not in result.response_payload()


@_receipt_pytest.mark.parametrize("path", ["chain", "gateway"])
@_receipt_pytest.mark.parametrize("kind", ["", "tool", "protocol", "timeout", "unavailable"])
def test_xxxv_typed_mcp_receipt_survives_real_chain_and_gateway(monkeypatch, path, kind):
    import asyncio
    import json
    from types import SimpleNamespace
    from orion_core.mcp_host import MCPReply
    import orion_core.dispatcher as dispatcher_module
    from orion_core.tool_gateway import use_tool

    monkeypatch.setattr(dispatcher_module, "_action_guard_on", lambda: False)
    monkeypatch.setattr(dispatcher_module, "_note_gap", lambda *_args: None)
    reply = MCPReply("measured", kind,
        content=[{"type": "text", "text": "measured"},
                 {"type": "resource_link", "uri": "fixture://report", "name": "report"}],
        structured_content={"rows": [1]}, meta={"version": 1},
        provenance={"server": "fixture", "tool": "read", "untrusted": True})
    host = _StubMCPHost()
    host._response = reply
    dispatcher = _dispatcher()
    dispatcher.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_args: None))
    dispatcher.mcp_host = host
    dispatcher._mcp_spend_gate = lambda *_args, **_kwargs: None
    dispatcher.telemetry = None
    dispatcher.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    dispatcher.register_mcp_tools("fixture", [{"name": "read"}])
    name = "mcp__fixture__read"
    if path == "gateway":
        result = asyncio.run(use_tool(dispatcher, {"name": name, "arguments": "{}"}))
    else:
        result = asyncio.run(dispatcher.dispatch_chain(name, {}))
    payload = json.loads(json.dumps(result.response_payload()))
    assert payload["ok"] is (not bool(kind))
    assert payload["receipt"] == reply.as_tool_receipt()
    assert result.evidence is None and result.media is None
    reply.structured_content["rows"].append(99)
    assert result.response_payload()["receipt"]["structuredContent"]["rows"] == [1]
    payload["receipt"]["provenance"]["server"] = "changed"
    assert result.response_payload()["receipt"]["provenance"]["server"] == "fixture"
    assert host.calls == [("fixture", "read", {})]


def test_ordinary_tool_result_keeps_existing_response_contract():
    assert ToolResult("ordinary", ok=False).response_payload() == {"ok": False, "result": "ordinary"}
