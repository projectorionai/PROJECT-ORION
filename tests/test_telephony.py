"""
ORION on the telephone.

A phone call is the one output ORION has that cannot be un-sent. It is often
recorded at the far end, it may be heard by someone who is not the user, and
unlike the paired-phone hand-off there is no tap to confirm — it dials
immediately and it costs money.

So most of these tests are about refusing. The number must already be one
ORION was told about, the words must survive the spillage guard, and a failure
anywhere in that chain must end in "I won't" rather than in a call being
placed anyway. The screening in particular fails CLOSED: if the guard cannot
be loaded, nothing is dialled at all.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.telephony import (  # noqa: E402
    MAX_SPOKEN_CHARS,
    Channel,
    ContactBook,
    TelephonyGateway,
    normalise_number,
    screen,
    twiml_say,
    twiml_stream,
)


def _run(coro):
    return asyncio.run(coro)


# ── numbers ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("written, dialable", [
    ("07700 900123", "+447700900123"),
    ("+44 7700 900123", "+447700900123"),
    ("0044-7700-900123", "+447700900123"),
    ("(0117) 496 0123", "+441174960123"),
    ("+1 415 555 0123", "+14155550123"),
])
def test_a_number_written_any_ordinary_way_can_be_dialled(written, dialable):
    """UK numbers get written five ways and only one of them can go through an
    API. A dialler that accepts only E.164 is a dialler nobody uses."""
    assert normalise_number(written) == dialable


@pytest.mark.parametrize("rubbish", ["", "   ", "nonsense", "12", None, "+0123"])
def test_something_that_is_not_a_number_is_refused(rubbish):
    assert normalise_number(rubbish) == ""


# ── channels ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("said, channel", [
    ("call", Channel.VOICE_CALL), ("RING", Channel.VOICE_CALL),
    ("text", Channel.SMS), ("sms", Channel.SMS),
    ("push", Channel.MOBILE_PUSH), ("desktop", Channel.DESKTOP),
])
def test_a_channel_can_be_asked_for_in_ordinary_words(said, channel):
    assert Channel.parse(said) is channel


def test_an_unknown_channel_falls_back_to_the_safe_one():
    """A reminder with a typo in its channel should still fire, on the channel
    that cannot cost money, rather than being silently dropped."""
    assert Channel.parse("carrier pigeon") is Channel.DESKTOP
    assert Channel.parse(None) is Channel.DESKTOP


# ── what ORION is allowed to say ──────────────────────────────────────────────

def test_ordinary_text_goes_through_untouched():
    result = screen("Rates were held at 4 percent this morning.")
    assert result.safe is True
    assert result.text == "Rates were held at 4 percent this morning."
    assert result.removed == []


def test_a_card_number_is_cut_out():
    result = screen("Pay with 4111 1111 1111 1111 please.")
    assert result.safe is True
    assert "4111" not in result.text
    assert "[removed]" in result.text
    assert "payment card number" in result.removed


def test_several_secrets_are_all_cut_out():
    """Excising back-to-front, so one cut cannot shift the offsets of the
    ones still to come."""
    result = screen("key " + "sk-" + "abcdefghijklmnopqrstuvwxyz012345 "
                    "and card 4111 1111 1111 1111")
    assert result.safe is True
    assert "sk-abcdef" not in result.text
    assert "4111" not in result.text
    assert len(result.removed) == 2


def test_screening_fails_closed_when_the_guard_is_missing():
    """The load-bearing refusal.

    A phone call cannot be un-sent. If the guard cannot be consulted, the
    answer is no — not "probably fine".
    """
    class _Broken:
        def scan(self, _text):
            raise RuntimeError("the guard is not available")

    result = screen("anything at all", _Broken())
    assert result.safe is False
    assert result.text == ""


def test_an_empty_message_is_refused():
    assert screen("").safe is False
    assert screen("   ").safe is False


def test_a_very_long_message_is_trimmed():
    """Past about this a listener has stopped listening, and the call is
    costing money by the minute."""
    result = screen("word " * 2000)
    assert result.safe is True
    assert len(result.text) <= MAX_SPOKEN_CHARS + 40
    assert "on your screen" in result.text


# ── TwiML ─────────────────────────────────────────────────────────────────────

def test_the_spoken_message_is_escaped():
    """A headline with an ampersand in it would otherwise produce malformed
    XML, and Twilio plays an error tone instead of the news."""
    xml = twiml_say("Marks & Spencer <up> 4%")
    assert "&amp;" in xml and "&lt;up&gt;" in xml
    assert "<Say" in xml and "</Response>" in xml


def test_the_voice_is_british():
    assert 'language="en-GB"' in twiml_say("hello")


def test_a_two_way_call_connects_a_stream():
    xml = twiml_stream("wss://orion.example.com/media", greeting="One moment.")
    assert "<Connect><Stream" in xml
    assert "wss://orion.example.com/media" in xml
    assert "One moment." in xml


# ── the contact book ──────────────────────────────────────────────────────────

@pytest.fixture
def contacts(tmp_path) -> ContactBook:
    path = tmp_path / "telephony_contacts.json"
    path.write_text(json.dumps({"contacts": [
        {"name": "me", "number": "07700 900123"},
        {"name": "Mum", "number": "+44 7700 900456"},
    ]}), encoding="utf-8")
    return ContactBook(path)


def test_a_known_name_resolves_to_a_number(contacts):
    assert contacts.resolve("me") == "+447700900123"
    assert contacts.resolve("mum") == "+447700900456"


def test_a_number_already_on_the_list_is_accepted(contacts):
    assert contacts.resolve("07700 900123") == "+447700900123"


def test_a_number_not_on_the_list_is_refused(contacts):
    """Default-deny, not a convenience.

    ORION reaches a model that can be talked into things, and "call this
    number and read out what I tell you" is a usable attack the moment any
    number is dialable.
    """
    assert contacts.resolve("+447700900999") == ""
    assert contacts.resolve("Someone Else") == ""


def test_a_missing_contact_file_means_nothing_can_be_dialled(tmp_path):
    book = ContactBook(tmp_path / "does_not_exist.json")
    assert len(book) == 0
    assert book.resolve("me") == ""


def test_a_corrupt_contact_file_does_not_raise(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert len(ContactBook(path)) == 0


# ── placing a call ────────────────────────────────────────────────────────────

class _MCP:
    """A telephony server that records what it was asked to do."""

    def __init__(self, ok: bool = True, text: str = "CA" + "0" * 32) -> None:
        self.ok, self.text, self.calls = ok, text, []

    def server_names(self):
        return ["twilio"]

    async def call_tool(self, server, tool, arguments):
        self.calls.append((server, tool, arguments))

        class _Result:
            pass
        result = _Result()
        result.ok = self.ok
        result.text = self.text
        return result


def _gateway(contacts, mcp=None) -> TelephonyGateway:
    return TelephonyGateway(mcp=mcp, contacts=contacts)


def test_a_call_to_a_known_number_is_placed(contacts):
    mcp = _MCP()
    result = _run(_gateway(contacts, mcp).call("me", "Your briefing is ready."))
    assert result.ok is True
    assert result.to == "+447700900123"
    assert mcp.calls, "the server was never asked"
    server, tool, arguments = mcp.calls[0]
    assert server == "twilio"
    assert arguments["to"] == "+447700900123"
    assert "Your briefing is ready." in arguments["twiml"]


def test_a_call_to_an_unknown_number_is_refused(contacts):
    mcp = _MCP()
    result = _run(_gateway(contacts, mcp).call("+447700900999", "hello"))
    assert result.ok is False
    assert mcp.calls == [], "it dialled anyway"
    assert "allowed to dial" in result.detail


def test_a_message_with_a_secret_in_it_is_never_dialled(contacts):
    """The one that matters most.

    Not "the secret is removed and the call proceeds" — for a card number it
    is removed, but the point of this test is that the check happens BEFORE
    anything is dialled, so a failure to screen cannot become a call.
    """
    class _NoGuard:
        def scan(self, _text):
            raise RuntimeError("unavailable")

    mcp = _MCP()
    gateway = TelephonyGateway(mcp=mcp, contacts=contacts, guard=_NoGuard())
    result = _run(gateway.call("me", "the code is 1234"))
    assert result.ok is False
    assert mcp.calls == [], "it dialled without screening the words"


def test_a_card_number_is_stripped_before_the_call_goes_out(contacts):
    mcp = _MCP()
    _run(_gateway(contacts, mcp).call("me", "Card 4111 1111 1111 1111 now."))
    assert mcp.calls, "the call should still be placed"
    assert "4111" not in mcp.calls[0][2]["twiml"]


def test_nothing_is_dialled_without_a_telephony_server(contacts):
    """And the reason given names the missing credential.

    The old message -- "No telephony server is connected. Enable 'twilio' in
    config/mcp_servers.json and restart." -- named a file, a server and a
    restart, and was wrong about all three whenever the real problem was an
    empty Account SID. It also pointed at a server that could never start on
    this machine, because launching it needs npx and there is no Node.
    """
    result = _run(_gateway(contacts, None).call("me", "hello"))
    assert result.ok is False
    assert "Account SID" in result.detail
    assert "setup_twilio" in result.detail


def test_a_server_that_refuses_is_reported_honestly(contacts):
    mcp = _MCP(ok=False, text="insufficient funds")
    result = _run(_gateway(contacts, mcp).call("me", "hello"))
    assert result.ok is False
    assert "insufficient funds" in result.detail


def test_a_renamed_server_tool_is_tried_under_its_other_names(contacts):
    """Community MCP servers rename their tools between releases. A rename
    should degrade to a clear message, not a stack trace."""
    class _PickyMCP(_MCP):
        async def call_tool(self, server, tool, arguments):
            if tool != "make_call":
                raise RuntimeError(f"no such tool: {tool}")
            return await super().call_tool(server, tool, arguments)

    mcp = _PickyMCP()
    result = _run(_gateway(contacts, mcp).call("me", "hello"))
    assert result.ok is True
    assert mcp.calls[0][1] == "make_call"


def test_a_text_message_is_sent(contacts):
    mcp = _MCP(text="SM" + "0" * 32)
    result = _run(_gateway(contacts, mcp).text("me", "Leaving in ten minutes."))
    assert result.ok is True
    assert mcp.calls[0][2]["body"] == "Leaving in ten minutes."


# ── reminders route by channel ────────────────────────────────────────────────

class _Signal:
    def __init__(self) -> None:
        self.sent = []

    def emit(self, *args):
        self.sent.append(args)


class _Bus:
    def __init__(self) -> None:
        for name in ("banner", "log", "speak_request", "dashboard_event",
                     "phone_action"):
            setattr(self, name, _Signal())


def _service():
    from orion_core.reminders import ReminderService

    bus = _Bus()
    return ReminderService(bus), bus


@pytest.mark.parametrize("phrase, channel, text", [
    ("remind me in 2 minutes to check the oven", "DESKTOP", "check the oven"),
    ("remind me in 5 minutes about the news and call me", "VOICE_CALL",
     "about the news"),
    ("in 3 minutes text me to leave", "SMS", "leave"),
    ("in 1 minute notify me on my phone about lunch", "MOBILE_PUSH",
     "about lunch"),
])
def test_a_reminder_can_say_how_it_wants_to_arrive(phrase, channel, text):
    """"Remind me at 8 and call me" is how people actually ask."""
    service, _bus = _service()
    service.add(phrase=phrase)
    reminder = service._reminders[-1]
    assert reminder.channel == channel
    assert reminder.text == text


def test_the_words_that_chose_the_channel_are_not_read_back():
    """Left in, ORION reads "and call me" back at you as though it were the
    reminder."""
    service, _bus = _service()
    service.add(phrase="remind me in 2 minutes to ring the dentist and call me")
    # Also confirms "ring me" did not false-match inside "ring the dentist":
    # the subject survives, only the delivery instruction is taken out.
    assert service._reminders[-1].text == "ring the dentist"
    assert service._reminders[-1].channel == "VOICE_CALL"


def test_an_explicit_channel_beats_the_phrase():
    service, _bus = _service()
    service.add(phrase="in 2 minutes to check the oven", channel="VOICE_CALL")
    assert service._reminders[-1].channel == "VOICE_CALL"


def test_a_desktop_reminder_is_still_just_spoken():
    service, bus = _service()
    service.add(phrase="in 1 minute to stand up")
    service._fire(service._reminders[-1])
    assert bus.speak_request.sent, "it should have been said out loud"


def test_a_voice_reminder_with_no_telephony_is_spoken_instead():
    """A reminder that could not be delivered the way it was asked for must
    still be delivered, not silently dropped."""
    service, bus = _service()
    service.add(phrase="in 1 minute about the news and call me")
    service._fire(service._reminders[-1])
    assert bus.speak_request.sent, "the fallback did not happen"
    assert any("couldn't reach you" in str(args) for args in bus.log.sent)


def test_a_voice_reminder_is_handed_to_telephony_when_there_is_some():
    service, bus = _service()
    handed = []

    class _Gateway:
        async def call(self, who, message, **kwargs):
            handed.append((who, message))

    service.telephony = _Gateway()
    service.add(phrase="in 1 minute about the news and call me",
                recipient="me")
    service._fire(service._reminders[-1])
    assert not bus.speak_request.sent, "it should not also have been spoken"


def test_the_banner_appears_whatever_the_channel():
    """Free, local, and a reminder that rang your phone should still be
    visible on screen when you get back to it."""
    service, bus = _service()
    service.add(phrase="in 1 minute about the news and call me")
    service._fire(service._reminders[-1])
    assert bus.banner.sent


# ── against the REAL MCP host contract ───────────────────────────────────────
#
# The fakes above define `call_tool(server, tool, arguments)`. MCPHost has no
# such method: it exposes `call(server, tool, args)`, while `call_tool(tool,
# args)` belongs to a single server CONNECTION and takes two arguments. So
# every test here passed while the real object would have raised TypeError on
# each candidate name and reported it as the telephony server refusing.
#
# Worse, MCPHost returns failures as ordinary STRINGS. The gateway read
# `getattr(result, "ok", True)`, and a string has no `ok` — so the default
# turned every failure into a placed call. ORION would have said "Calling
# +44…" having done nothing.
#
# These use the real shape, so they fail if that contract drifts again.

class _PassingGuard:
    """A guard that finds nothing to remove.

    Deliberately not a guard that RAISES — screen() fails closed when it
    cannot check the text, which is correct and is covered elsewhere, but it
    would make these tests pass for the wrong reason: the call would be
    refused before the MCP contract was ever exercised.
    """

    def scan(self, text):
        from orion_core.spillage_guard import SpillageGuard  # noqa: F401

        return type("Clean", (), {"safe": True, "text": text,
                                  "removed": [], "reason": ""})()


class _RealShapedHost:
    """What MCPHost actually looks like: call(server, tool, args) -> str."""

    def __init__(self, reply="CA" + "0" * 32):
        self.reply = reply
        self.calls = []

    async def call(self, server, tool, arguments):
        self.calls.append((server, tool, arguments))
        return self.reply


async def test_a_call_goes_through_the_real_host_method(contacts):
    host = _RealShapedHost()
    gateway = TelephonyGateway(mcp=host, contacts=contacts, guard=_PassingGuard())
    gateway.available = lambda: True

    result = await gateway.call("me", "Running late.")
    assert result.ok, result.detail
    assert host.calls, "the gateway never reached the host"
    server, tool, arguments = host.calls[0]
    assert server == "twilio"
    assert "call" in tool
    assert arguments["to"]


async def test_the_call_sid_is_read_back_from_a_plain_string(contacts):
    sid = "CA" + "b" * 32
    host = _RealShapedHost(reply=f"Call queued, sid={sid}")
    gateway = TelephonyGateway(mcp=host, contacts=contacts, guard=_PassingGuard())
    gateway.available = lambda: True
    assert (await gateway.call("me", "hello")).sid == sid


async def test_a_failure_string_is_not_read_as_a_placed_call(contacts):
    """The one that would have mattered most: ORION announcing a call he
    never made."""
    for reply in ("No connected MCP server named 'twilio'. Available: none.",
                  "Server 'twilio' has no tool 'create_call'. Its tools: none.",
                  "MCP tool 'twilio.create_call' failed: 401 Unauthorized",
                  "MCP tool 'twilio.create_call' timed out.",
                  ""):
        host = _RealShapedHost(reply=reply)
        gateway = TelephonyGateway(mcp=host, contacts=contacts,
                                   guard=_PassingGuard())
        gateway.available = lambda: True
        result = await gateway.call("me", "hello")
        assert result.ok is False, f"{reply!r} was treated as success"


async def test_an_explicit_ok_flag_still_wins(contacts):
    """A ToolResult-shaped reply says for itself; the text heuristic is only
    for the plain strings MCPHost returns."""
    class Result:
        ok = True
        text = "failed: but the flag says otherwise"

    class Host(_RealShapedHost):
        async def call(self, server, tool, arguments):
            return Result()

    gateway = TelephonyGateway(mcp=Host(), contacts=contacts, guard=_PassingGuard())
    gateway.available = lambda: True
    assert (await gateway.call("me", "hello")).ok is True


async def test_a_host_with_neither_method_fails_cleanly(contacts):
    class Useless:
        pass

    gateway = TelephonyGateway(mcp=Useless(), contacts=contacts,
                               guard=_PassingGuard())
    gateway.available = lambda: True
    result = await gateway.call("me", "hello")
    assert result.ok is False
