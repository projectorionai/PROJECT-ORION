"""ORION asks before spending your money.

The Twilio entry in the server template carried the warning from the day it was
added — "calls dial immediately and cost money — confirm before calling" — and
nothing enforced it. `mcp` action='call' invoked any tool on any connected
server with no gate whatsoever, so enabling the server was enough for a model
that had misread a sentence to place a real phone call to a real number.

CallAuthority does not cover this. That guards INBOUND callers, deciding what a
voice on the telephone may make ORION do. This is the other direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import mcp_host as mh  # noqa: E402
from orion_core.dispatcher import OrionDispatcher  # noqa: E402


@pytest.fixture()
def host(monkeypatch):
    """A host with no config file behind it, so only the built-ins apply."""
    monkeypatch.setattr(mh, "load_config", lambda: {"servers": {}})
    return mh.MCPHost.__new__(mh.MCPHost)


# ── what needs confirming ────────────────────────────────────────────────────

@pytest.mark.parametrize("tool", ["create_call", "make_outbound_call", "dial",
                                  "send_sms", "messages_create"])
def test_a_billable_telephony_tool_needs_confirmation(host, tool):
    assert host.requires_confirmation("twilio", tool)


def test_the_reason_says_what_will_happen(host):
    """"This needs confirmation" does not tell someone what they are being
    asked to approve."""
    reason = host.requires_confirmation("twilio", "create_call")
    assert "phone call" in reason and "costs money" in reason


def test_reading_things_does_not_need_confirmation(host):
    for server, tool in (("gmail", "search_emails"),
                         ("gmail", "read_email"),
                         ("google_calendar", "list-events"),
                         ("google_calendar", "get-freebusy"),
                         ("filesystem", "read_file"),
                         ("brave_search", "search")):
        assert host.requires_confirmation(server, tool) == ""


@pytest.mark.parametrize("server,tool", [
    ("gmail", "send_email"),
    ("gmail", "delete_email"),
    ("gmail", "get_or_create_label"),
    ("gmail", "download_attachment"),
    ("google_calendar", "create-event"),
    ("google_calendar", "update-event"),
    ("google_calendar", "delete-event"),
    ("google_calendar", "respond-to-event"),
    ("google_calendar", "manage-accounts"),
])
def test_email_and_calendar_changes_need_confirmation_even_with_old_config(host, server, tool):
    assert host.requires_confirmation(server, tool)


def test_any_server_can_be_marked_in_its_own_config(monkeypatch):
    """Declarative, so a server the user adds is protected the same way
    without this file knowing it exists."""
    monkeypatch.setattr(mh, "load_config", lambda: {
        "servers": {"my_bank": {"enabled": True, "confirm": True}}})
    host = mh.MCPHost.__new__(mh.MCPHost)
    assert host.requires_confirmation("my_bank", "anything")


def test_the_shipped_twilio_entry_is_marked():
    """A config written before this change is covered by CONFIRM_SERVERS; a
    config written after it should carry the flag."""
    spec = mh._default_config()["servers"]["twilio"]
    assert spec.get("confirm") is True
    assert spec.get("enabled") is False


# ── the gate is actually applied ─────────────────────────────────────────────

class _FakeHost:
    def __init__(self, reason=""):
        self.reason = reason
        self.calls = []

    def requires_confirmation(self, server, tool):
        return self.reason

    async def call(self, server, tool, arguments):
        self.calls.append((server, tool, arguments))
        return "done"


def _dispatcher(host):
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.mcp_host = host
    return d


async def test_a_billable_call_without_confirmation_is_refused():
    host = _FakeHost("twilio.create_call places a real phone call")
    result = await _dispatcher(host).mcp_tool(
        {"action": "call", "server": "twilio", "tool": "create_call",
         "arguments": {"to": "+44..."}})
    assert result.ok is False
    assert "real phone call" in result.text
    assert host.calls == [], "the call was placed anyway"


async def test_an_ordinary_tool_is_not_made_to_ask():
    """A gate on everything is a gate nobody reads."""
    host = _FakeHost("")
    result = await _dispatcher(host).mcp_tool(
        {"action": "call", "server": "gmail", "tool": "search_emails",
         "arguments": {}})
    assert result.ok
    assert len(host.calls) == 1


async def test_a_host_that_cannot_answer_does_not_block_ordinary_tools():
    """The gate must not become a way for every MCP call to fail."""
    class Broken(_FakeHost):
        def requires_confirmation(self, server, tool):
            raise RuntimeError("no config")

    host = Broken()
    result = await _dispatcher(host).mcp_tool(
        {"action": "call", "server": "gmail", "tool": "search_emails",
         "arguments": {}})
    assert result.ok


def test_the_model_is_told_the_gate_exists():
    """A parameter the model does not know about is a parameter it never
    passes, which turns the gate into a wall."""
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    mcp = next(t for t in TOOL_DECLARATIONS if t["name"] == "mcp")
    assert "confirm" in mcp["parameters"]["properties"]
    assert "confirm=true" in mcp["description"]


# ── the second door ─────────────────────────────────────────────────────────
#
# The telephony tool goes behind a human-issued token. `mcp` action=call
# reaches the SAME Twilio tools directly, and was guarded only by confirm=true
# — a parameter the model fills in for itself. Two doors to one bill, one of
# them bolted.

class _Guarded(_FakeHost):
    pass


def _guarded_dispatcher(reason="twilio.create_call spends real money"):
    from orion_core.system_guard import SystemActionGuard

    host = _FakeHost(reason)
    d = _dispatcher(host)
    d.system_guard = SystemActionGuard()

    class _Signal:
        def emit(self, *a, **k):
            pass

    class _Bus:
        def __getattr__(self, name):
            return _Signal()

    d.bus = _Bus()
    return d, host


async def test_a_billable_mcp_call_waits_for_a_human_even_with_confirm():
    d, host = _guarded_dispatcher()
    result = await d.mcp_tool(
        {"action": "call", "server": "twilio", "tool": "create_call",
         "arguments": {"to": "+44"}, "confirm": True})
    assert result.ok is False
    assert "approve" in result.text.lower()
    assert host.calls == [], "it spent money on the model's say-so"


async def test_an_ordinary_mcp_call_is_not_made_to_wait():
    """A gate on everything is a gate nobody reads."""
    d, host = _guarded_dispatcher(reason="")
    result = await d.mcp_tool(
        {"action": "call", "server": "gmail", "tool": "search_emails",
         "arguments": {}})
    assert result.ok
    assert len(host.calls) == 1


async def test_the_armed_token_carries_which_tool_it_was_for():
    d, host = _guarded_dispatcher()
    await d.mcp_tool(
        {"action": "call", "server": "twilio", "tool": "create_call",
         "arguments": {"to": "+44"}, "confirm": True})
    pending = list(d.system_guard._pending.values())
    assert len(pending) == 1
    assert pending[0].payload["server"] == "twilio"
    assert pending[0].payload["tool"] == "create_call"


async def test_without_a_guard_a_confirmed_mcp_call_is_refused():
    """Model-supplied confirm=true cannot replace human approval."""
    host = _FakeHost("twilio spends money")
    d = _dispatcher(host)
    result = await d.mcp_tool(
        {"action": "call", "server": "twilio", "tool": "create_call",
         "arguments": {}, "confirm": True})
    assert result.ok is False
    assert host.calls == []


# ── the third door: first-class mcp__<server>__<tool> names ─────────────────
#
# register_mcp_tools exposes every server tool directly (Mark XXI E1). Those
# names went straight to the host with no gate at all, so a spending tool
# reached by its own name skipped even the confirm flag.

async def test_a_first_class_billable_tool_waits_for_a_human():
    d, host = _guarded_dispatcher()
    d.register_mcp_tools("twilio", [{"name": "create_call", "description": "Call.",
                                     "inputSchema": {"type": "object", "properties": {
                                         "to": {"type": "string"}}}}])
    refused = await d.dispatch("mcp__twilio__create_call", {"to": "+44"})
    assert refused.ok is False and host.calls == []
    armed = await d.dispatch("mcp__twilio__create_call", {"to": "+44", "confirm": True})
    assert armed.ok is False and "approve" in armed.text.lower()
    assert host.calls == [], "a first-class name bypassed the spend gate"


async def test_a_failed_mcp_call_is_not_reported_as_success():
    class _Failing(_FakeHost):
        async def call(self, server, tool, arguments):
            self.calls.append((server, tool, arguments))
            return f"MCP tool '{server}.{tool}' timed out."

    d = _dispatcher(_Failing(""))
    result = await d.mcp_tool({"action": "call", "server": "fetch", "tool": "fetch",
                               "arguments": {}})
    assert result.ok is False
