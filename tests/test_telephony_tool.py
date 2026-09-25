"""The tool that lets ORION be asked to ring you.

The gateway underneath it was written, tested and never constructed: nothing
outside the test suite ever made a TelephonyGateway, so "call my phone" had
nowhere to go however Twilio was configured. And when it was finally called,
it spoke a contract MCPHost does not implement — see test_telephony.py.

This covers the reachable surface, and the three guards that stand between a
sentence and a real phone ringing.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.dispatcher import OrionDispatcher  # noqa: E402
from orion_core.telephony import ContactBook, TelephonyGateway  # noqa: E402


class _Signal:
    def emit(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, name):
        return _Signal()


class _Host:
    """MCPHost's real shape: call(server, tool, args) -> str."""

    def __init__(self, reply="CA" + "0" * 32):
        self.reply = reply
        self.calls = []
        self.servers = {"twilio": object()}

    async def call(self, server, tool, arguments):
        self.calls.append((server, tool, arguments))
        return self.reply


class _Guard:
    def scan(self, text):
        return type("Clean", (), {"safe": True, "text": text,
                                  "removed": [], "reason": ""})()


@pytest.fixture()
def dispatcher(tmp_path):
    path = tmp_path / "telephony_contacts.json"
    path.write_text(json.dumps({"contacts": [
        {"name": "me", "number": "07700 900123"}]}), encoding="utf-8")
    host = _Host()
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _Bus()
    d.telephony = TelephonyGateway(mcp=host, bus=_Bus(),
                                   contacts=ContactBook(path), guard=_Guard())
    d.host = host
    return d


# ── the guards ───────────────────────────────────────────────────────────────

async def test_a_call_without_confirmation_is_refused(dispatcher):
    """Real money, real phone, immediate. There is no draft state and no tap
    to confirm the way there is on the phone hand-off."""
    result = await dispatcher.telephony_tool(
        {"action": "call", "to": "me", "message": "Time to leave."})
    assert result.ok is False
    assert "confirm" in result.text.lower()
    assert dispatcher.host.calls == [], "it dialled anyway"


async def test_with_confirmation_it_rings(dispatcher):
    result = await dispatcher.telephony_tool(
        {"action": "call", "to": "me", "message": "Time to leave.",
         "confirm": True})
    assert result.ok, result.text
    assert len(dispatcher.host.calls) == 1
    server, tool, arguments = dispatcher.host.calls[0]
    assert server == "twilio" and "call" in tool
    assert arguments["to"] == "+447700900123"


async def test_a_number_that_is_not_in_the_book_cannot_be_dialled(dispatcher):
    """Default-deny. ORION must not be able to ring a number somebody said
    out loud, or one that was sitting in a web page he read."""
    result = await dispatcher.telephony_tool(
        {"action": "call", "to": "+44 7700 900999", "message": "hello",
         "confirm": True})
    assert result.ok is False
    assert dispatcher.host.calls == []


async def test_nothing_to_say_is_refused_before_dialling(dispatcher):
    result = await dispatcher.telephony_tool(
        {"action": "call", "to": "me", "message": "", "confirm": True})
    assert result.ok is False
    assert dispatcher.host.calls == []


async def test_a_failure_from_the_server_is_reported_as_one(dispatcher):
    """The gateway used to read MCPHost's failure STRINGS as success, so
    ORION would announce a call he never placed."""
    dispatcher.host.reply = "MCP tool 'twilio.create_call' failed: 401"
    result = await dispatcher.telephony_tool(
        {"action": "call", "to": "me", "message": "hello", "confirm": True})
    assert result.ok is False


# ── texting shares the guards ────────────────────────────────────────────────

async def test_a_text_also_needs_confirmation(dispatcher):
    result = await dispatcher.telephony_tool(
        {"action": "text", "to": "me", "message": "on my way"})
    assert result.ok is False
    assert dispatcher.host.calls == []


async def test_a_confirmed_text_is_sent(dispatcher):
    result = await dispatcher.telephony_tool(
        {"action": "text", "to": "me", "message": "on my way", "confirm": True})
    assert result.ok, result.text
    assert len(dispatcher.host.calls) == 1


# ── the read-only actions ────────────────────────────────────────────────────

async def test_contacts_lists_who_may_be_rung(dispatcher):
    result = await dispatcher.telephony_tool({"action": "contacts"})
    assert result.ok and "me" in result.text


async def test_status_says_which_route_is_live(dispatcher):
    """Which ROUTE, not just "connected": there are two now, and on a machine
    without Node the MCP one cannot start at all."""
    result = await dispatcher.telephony_tool({"action": "status"})
    assert result.ok
    assert "ready" in result.text.lower()
    assert "telephony server" in result.text.lower()


async def test_without_a_gateway_it_says_so(dispatcher):
    dispatcher.telephony = None
    result = await dispatcher.telephony_tool({"action": "status"})
    assert result.ok is False


# ── it is reachable at all ───────────────────────────────────────────────────

def test_the_tool_is_registered():
    d = OrionDispatcher.__new__(OrionDispatcher)
    assert d.handler_table()["telephony"] == d.telephony_tool


def test_the_gateway_is_built_by_the_application():
    """It was written, tested, and constructed nowhere but the tests."""
    app = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "TelephonyGateway(" in app, (
        "nothing builds a gateway, so the capability is unreachable")
    assert "dispatcher.telephony" in app


def test_the_model_is_told_confirmation_is_required():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "telephony")
    assert "confirm" in tool["parameters"]["properties"]
    assert "confirm=true" in tool["description"]
    assert "phone_action" in tool["description"], (
        "nothing distinguishes this from the phone hand-off, which is the "
        "tool that does NOT cost money")


# ── confirm=true is a convention, not a gate ─────────────────────────────────
#
# process_governor worked this out first and its comment is the argument:
#
#     This used to read "repeat the request with confirm=true and I shall stop
#     it", checking a parameter the MODEL fills in — which is a convention,
#     not a gate: nothing stopped the model sending confirm=true on the first
#     call, and the message explained how.
#
# Spending money is in the same category as terminating a process, so it goes
# behind the same human-issued, single-use, expiring token, with the number
# bound to it — the one approved on screen is necessarily the one dialled.

@pytest.fixture()
def guarded(dispatcher):
    """A dispatcher with the real system guard attached."""
    from orion_core.system_guard import SystemActionGuard

    dispatcher.system_guard = SystemActionGuard()
    return dispatcher


async def test_a_confirmed_call_still_waits_for_a_human(guarded):
    """confirm=true gets it as far as the prompt and no further."""
    result = await guarded.telephony_tool(
        {"action": "call", "to": "me", "message": "Time to leave.",
         "confirm": True})
    assert result.ok is False
    assert "approve" in result.text.lower()
    assert guarded.host.calls == [], "it dialled on the model's say-so"


async def test_the_approved_call_uses_the_number_that_was_approved(guarded):
    """The payload comes back bound to the token, so a model that changes its
    mind between arming and approval cannot redirect the call."""
    from orion_core.system_guard import ActionIntent

    await guarded.telephony_tool(
        {"action": "call", "to": "me", "message": "Time to leave.",
         "confirm": True})
    pending = list(guarded.system_guard._pending.values())
    assert len(pending) == 1
    assert pending[0].payload["to"] == "me"
    assert pending[0].intent is ActionIntent.PLACE_CALL


async def test_approving_the_token_actually_places_the_call(guarded):
    """Arming without an executor is the worst of both worlds: the user
    approves, nothing happens, and the approval succeeding makes it look as
    though the call went out."""
    await guarded.telephony_tool(
        {"action": "call", "to": "me", "message": "Time to leave.",
         "confirm": True})
    token = next(iter(guarded.system_guard._pending))
    result = guarded.confirm_system_action(token)
    assert result.ok, result.text
    await asyncio.sleep(0.05)
    assert len(guarded.host.calls) == 1, "approved, and nothing was dialled"


async def test_a_token_cannot_be_used_twice(guarded):
    await guarded.telephony_tool(
        {"action": "text", "to": "me", "message": "on my way", "confirm": True})
    token = next(iter(guarded.system_guard._pending))
    assert guarded.confirm_system_action(token).ok
    await asyncio.sleep(0.05)
    assert guarded.confirm_system_action(token).ok is False, (
        "a replayed token placed a second call")


async def test_an_unknown_token_places_nothing(guarded):
    assert guarded.confirm_system_action("not-a-real-token").ok is False
    assert guarded.host.calls == []


def test_spending_is_classed_with_the_destructive_actions():
    from orion_core.system_guard import ActionIntent, DESTRUCTIVE_INTENTS

    assert ActionIntent.PLACE_CALL in DESTRUCTIVE_INTENTS
    assert ActionIntent.SEND_MESSAGE in DESTRUCTIVE_INTENTS


def test_the_prompt_says_it_costs_money():
    """The user is reading one line before approving. It has to say what it
    will do to them."""
    from orion_core.system_guard import ActionIntent, _INTENT_PHRASE

    phrase = _INTENT_PHRASE[ActionIntent.PLACE_CALL]
    assert "real" in phrase and "phone call" in phrase


def test_the_prompt_uses_the_key_the_dialog_actually_reads():
    """Caught in review: the emit used "detail" and the dialog reads
    "message", so the user would have been shown the default "Confirm this
    action?" — asked to approve a phone call without being told which one."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(
        encoding="utf-8")
    handler = source[source.index("def _on_confirm_action"):]
    handler = handler[:handler.index("\n    def ", 10)]
    assert 'payload.get("message")' in handler

    web = (ROOT / "orion_core" / "dispatch_web.py").read_text(encoding="utf-8")
    emits = web.count("confirm_action.emit(")
    assert emits >= 2
    block = web[web.index("confirm_action.emit("):]
    assert '"message"' in block[:600], (
        "an armed confirmation carries no message, so the dialog cannot say "
        "what it is asking about")
