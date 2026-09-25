"""
The process Twilio talks to, and what a voice on the line may make ORION do.

Two things are load-bearing here.

**The signature must be verified against the PUBLIC url.** Behind a reverse
proxy the request arrives as `http://127.0.0.1:8790/...`, but Twilio signed
`https://orion.example.com/...`. Verifying against what aiohttp sees fails
every single time — a total outage that looks exactly like a wrong credential,
and one of the easiest things in this whole system to get subtly wrong.

**A telephone caller is unauthenticated.** Caller ID is trivially spoofed and
voice is not a secret — a recording of someone saying "yes" is easy to obtain
and no voice print survives a good one. So voice identification is a
convenience, never authority: a call may ask questions freely, and may not run
anything that changes something, spends money or leaves the machine without a
code delivered somewhere the caller does not control.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.telephony_server import (  # noqa: E402
    CONFIRM_DIGITS,
    SAFE_OVER_THE_PHONE,
    BridgeConfig,
    CallAuthority,
    TelephonyServer,
)


# ── configuration ─────────────────────────────────────────────────────────────

def test_a_bridge_with_nothing_configured_is_not_usable():
    """Neither has a sensible default: no token means every request is
    refused, and no public host means Twilio has nowhere to fetch from."""
    config = BridgeConfig()
    assert config.usable is False
    assert len(config.problems()) == 2


def test_both_halves_are_needed():
    assert BridgeConfig(auth_token="t").usable is False
    assert BridgeConfig(public_host="h").usable is False
    assert BridgeConfig(auth_token="t", public_host="h").usable is True


def test_the_problems_say_what_to_do():
    problems = " ".join(BridgeConfig().problems())
    assert "TWILIO_AUTH_TOKEN" in problems
    assert "ORION_PUBLIC_HOST" in problems


def test_configuration_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
    monkeypatch.setenv("ORION_PUBLIC_HOST", "orion.example.com")
    monkeypatch.setenv("ORION_TELEPHONY_PORT", "9999")
    config = BridgeConfig.from_env()
    assert config.auth_token == "tok"
    assert config.port == 9999
    assert config.usable is True


# ── what a caller may do ──────────────────────────────────────────────────────

@pytest.mark.parametrize("tool", [
    "briefing", "catch_up", "flight_search", "recall_conversation",
    "resource_status",
])
def test_reading_things_out_is_allowed(tool):
    """A call may ask questions and hear answers."""
    assert CallAuthority().permits(tool) is True


@pytest.mark.parametrize("tool", [
    "messaging", "process_control", "shutdown_orion", "desktop_control",
    "file_controller", "phone_action", "forge", "self_repair", "backup",
    "web_automation", "commerce_hub", "security_recon",
])
def test_anything_that_changes_something_is_gated(tool):
    """The list is default-deny: a tool is allowed over the phone only by
    being named, so a new tool is gated until somebody thinks about it."""
    assert CallAuthority().permits(tool) is False


def test_an_unknown_tool_is_gated():
    assert CallAuthority().permits("something_invented_later") is False
    assert CallAuthority().permits("") is False


def test_the_safe_list_contains_nothing_that_writes():
    """A guard against the list quietly growing a tool that acts.

    Checked by name rather than by inspecting each handler, because the point
    is that adding one here should require noticing this test.
    """
    dangerous = {"control", "kill", "delete", "shutdown", "restart", "send",
                 "write", "install", "execute", "forge", "repair", "buy",
                 "order", "pay", "post", "publish"}
    for tool in SAFE_OVER_THE_PHONE:
        assert not any(word in tool for word in dangerous), (
            f"{tool} looks like it acts, and it is on the phone-safe list")


# ── the second factor ─────────────────────────────────────────────────────────

def test_a_challenge_is_a_short_numeric_code():
    """Short enough to say down a bad line, long enough that guessing inside
    the window is hopeless."""
    code = CallAuthority().challenge("CA1", "messaging")
    assert len(code) == CONFIRM_DIGITS
    assert code.isdigit()


def test_the_right_code_authorises_the_tool_it_was_issued_for():
    authority = CallAuthority()
    code = authority.challenge("CA1", "messaging")
    ok, tool = authority.confirm("CA1", code)
    assert ok is True and tool == "messaging"


def test_a_code_works_once():
    """Otherwise an overheard code is a standing authorisation."""
    authority = CallAuthority()
    code = authority.challenge("CA1", "messaging")
    authority.confirm("CA1", code)
    assert authority.confirm("CA1", code)[0] is False


def test_a_wrong_code_is_refused():
    authority = CallAuthority()
    authority.challenge("CA1", "messaging")
    assert authority.confirm("CA1", "000000")[0] is False


def test_a_code_expires(monkeypatch):
    import orion_core.telephony_server as module

    authority = CallAuthority()
    code = authority.challenge("CA1", "messaging")
    # Anchored to the real clock, not a fixed number: monotonic() counts from
    # an arbitrary origin and on a machine that has been up a while it is
    # already far past any constant one might pick.
    later = module.time.monotonic() + module.CONFIRM_WINDOW_S + 1
    monkeypatch.setattr(module.time, "monotonic", lambda: later)
    ok, reason = authority.confirm("CA1", code)
    assert ok is False and "expired" in reason


def test_a_code_said_aloud_is_understood():
    """Speech-to-text will not agree with itself about spacing, and "four one
    nine" and "419" are the same answer."""
    authority = CallAuthority()
    code = authority.challenge("CA1", "messaging")
    assert authority.confirm("CA1", " ".join(code))[0] is True


def test_confirming_when_nothing_was_asked_is_refused():
    assert CallAuthority().confirm("CA9", "123456")[0] is False


def test_a_code_for_one_call_does_not_unlock_another():
    authority = CallAuthority()
    code = authority.challenge("CA1", "messaging")
    assert authority.confirm("CA2", code)[0] is False


def test_the_code_goes_to_the_phone_not_down_the_line():
    """A code read to the caller proves nothing — the caller is who we are
    unsure about."""
    sent = []

    class _Signal:
        def emit(self, *args):
            sent.append(args)

    class _Bus:
        banner = _Signal()
        phone_action = _Signal()

    CallAuthority(_Bus()).challenge("CA1", "messaging")
    assert sent, "nothing was delivered out of band"


def test_the_refusal_explains_what_to_do():
    wording = CallAuthority().refusal("messaging")
    assert "code" in wording and "messaging" in wording


# ── the server ────────────────────────────────────────────────────────────────

def _server(**kwargs) -> TelephonyServer:
    return TelephonyServer(
        config=BridgeConfig(auth_token="tok", public_host="orion.example.com"),
        **kwargs)


def test_the_expected_endpoints_are_served():
    routes = {route.resource.canonical
              for route in _server().build_app().router.routes()}
    assert {"/twilio/voice", "/media", "/health"} <= routes


def test_media_websocket_requires_twilio_signature():
    from aiohttp import WSServerHandshakeError
    from aiohttp.test_utils import TestClient, TestServer

    from orion_core.telephony_bridge import twilio_signature

    async def scenario():
        client = TestClient(TestServer(_server().build_app()))
        await client.start_server()
        try:
            try:
                await client.ws_connect("/media")
                raise AssertionError("unsigned media socket was accepted")
            except WSServerHandshakeError as exc:
                assert exc.status == 403
            signature = twilio_signature("tok", "wss://orion.example.com/media")
            socket = await client.ws_connect(
                "/media", headers={"X-Twilio-Signature": signature})
            try:
                await client.ws_connect(
                    "/media", headers={"X-Twilio-Signature": signature})
                raise AssertionError("a second media call replaced the first")
            except WSServerHandshakeError as exc:
                assert exc.status == 409
            await socket.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_the_signature_is_checked_against_the_public_url():
    """The subtle one.

    Behind Caddy the request arrives as http://127.0.0.1:8790/..., but Twilio
    signed https://orion.example.com/... Verifying against what aiohttp sees
    fails every time and looks exactly like a wrong credential.
    """
    class _Request:
        method = "POST"
        path = "/twilio/voice"
        path_qs = "/twilio/voice"
        host = "127.0.0.1:8790"
        headers: dict = {}

    assert _server()._public_url(_Request()) == \
        "https://orion.example.com/twilio/voice"


def test_the_query_string_is_part_of_the_signed_url():
    """Twilio signs the URL it requested, query string included."""
    class _Request:
        method = "GET"
        path = "/twilio/voice"
        path_qs = "/twilio/voice?CallSid=CA1"
        host = "127.0.0.1:8790"
        headers: dict = {}

    assert _server()._public_url(_Request()).endswith("?CallSid=CA1")


def test_an_unknown_caller_is_refused_when_there_is_a_contact_book():
    class _Contacts:
        def allows(self, number):
            return number == "+447700900123"

    server = _server(contacts=_Contacts())
    assert server._caller_allowed("+447700900123") is True
    assert server._caller_allowed("+447700900999") is False


def test_no_contact_book_means_nobody_is_allowed():
    assert _server()._caller_allowed("+447700900123") is False


def test_a_contact_book_that_raises_denies_rather_than_allows():
    class _Hostile:
        def allows(self, number):
            raise RuntimeError("the file is gone")

    assert _server(contacts=_Hostile())._caller_allowed("+44770090123") is False


# ── it is actually what the unit runs ─────────────────────────────────────────

def test_the_systemd_unit_starts_this_module():
    """The bridge has no entry point of its own — a unit pointing at it would
    start and immediately exit, which is how this was caught."""
    unit = (ROOT / "deploy" / "orion-telephony.service").read_text(
        encoding="utf-8")
    assert "orion_core.telephony_server" in unit


def test_this_module_can_be_run():
    import orion_core.telephony_server as module

    assert callable(module.main)
    assert callable(module.serve)


# ── the gate is enforced, not merely present ──────────────────────────────────

class _Inner:
    """The real dispatcher, standing in."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict]] = []
        self.unrelated = "still reachable"

    async def dispatch(self, name, args):
        self.ran.append((name, dict(args or {})))

        class _Result:
            ok = True
            text = f"{name} ran"
        return _Result()


def _gate(authority=None):
    from orion_core.telephony_server import GatedDispatcher

    inner = _Inner()
    authority = authority or CallAuthority()
    return inner, authority, GatedDispatcher(inner, authority, "CA1")


def test_a_read_only_tool_goes_straight_through():
    import asyncio

    inner, _authority, gate = _gate()
    result = asyncio.run(gate.dispatch("briefing", {}))
    assert result.ok is True
    assert [name for name, _ in inner.ran] == ["briefing"]


def test_a_tool_that_acts_does_not_reach_the_dispatcher():
    """The whole point. Not "it is refused afterwards" — it never runs."""
    import asyncio

    inner, _authority, gate = _gate()
    result = asyncio.run(gate.dispatch("messaging", {"to": "mum"}))
    assert result.ok is False
    assert inner.ran == [], "the tool ran despite being gated"


def test_the_right_code_releases_the_tool_with_its_original_arguments():
    """The caller does not repeat the request after confirming — the code
    authorises what was already asked."""
    import asyncio

    inner, authority, gate = _gate()
    asyncio.run(gate.dispatch("messaging", {"to": "mum", "body": "hi"}))
    code = authority._pending["CA1"][0]
    result = asyncio.run(gate.dispatch("confirm", {"code": code}))
    assert result.ok is True
    assert inner.ran[-1] == ("messaging", {"to": "mum", "body": "hi"})


def test_a_code_authorises_one_action_not_a_session():
    import asyncio

    inner, authority, gate = _gate()
    asyncio.run(gate.dispatch("messaging", {}))
    code = authority._pending["CA1"][0]
    asyncio.run(gate.dispatch("confirm", {"code": code}))
    before = len(inner.ran)
    asyncio.run(gate.dispatch("shutdown_orion", {}))
    assert len(inner.ran) == before, "the session stayed unlocked"


def test_a_wrong_code_releases_nothing():
    import asyncio

    inner, _authority, gate = _gate()
    asyncio.run(gate.dispatch("messaging", {}))
    result = asyncio.run(gate.dispatch("confirm", {"code": "000000"}))
    assert result.ok is False
    assert inner.ran == []


def test_the_gate_keeps_an_audit_of_what_it_refused():
    import asyncio

    _inner, _authority, gate = _gate()
    asyncio.run(gate.dispatch("briefing", {}))
    asyncio.run(gate.dispatch("shutdown_orion", {}))
    assert gate.allowed == ["briefing"]
    assert gate.refused == ["shutdown_orion"]


def test_everything_else_reaches_the_real_dispatcher():
    """A pass-through rather than a whitelist: this class gates `dispatch`,
    and hiding unrelated attributes breaks the engine in ways that look like
    a different bug."""
    _inner, _authority, gate = _gate()
    assert gate.unrelated == "still reachable"


def test_the_gate_is_attached_and_removed_by_the_server():
    """Built-but-unwired is the failure this whole class exists to prevent,
    so the wiring is asserted too."""
    source = (ROOT / "orion_core" / "telephony_server.py").read_text(
        encoding="utf-8")
    assert "GatedDispatcher(dispatcher, self.authority)" in source
    assert "self.engine.dispatcher = gate.inner" in source, (
        "the gate must be removed when the call ends, or telephone rules "
        "silently apply to the desktop")
