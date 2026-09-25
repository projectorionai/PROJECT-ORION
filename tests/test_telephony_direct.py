"""Outbound calling could never have worked, and it was not the credentials.

ORION's telephony went through Twilio's MCP server, launched with
``npx -y @twilio-alpha/mcp``. This machine has no Node, so npx is not on PATH
and that server could never start. Perfect credentials would have changed
nothing, and the only thing the user ever saw was "No telephony server is
connected. Enable 'twilio' in config/mcp_servers.json and restart." -- which
names a file, a server and a restart, and is wrong about all three.

Placing a call needs none of that. Twilio's REST API is one HTTPS POST with
basic auth that takes the TwiML inline, so there is no webhook to host and
nothing public to expose. The MCP server stays preferred when it IS connected,
because it does more than dial and it is the audit trail the model's own tool
calls go through. This is the fallback that makes dialling work on a machine
with nothing installed.

What is still not possible is calling "without it being set up": Twilio bills a
real account and will not place a call for one that does not exist. These tests
pin the next best thing -- that the setup is one step, that ORION says which of
the three values is missing, and that no route around the spend gate was opened
while adding a second transport.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import telephony, telephony_direct  # noqa: E402

GATEWAY_SOURCE = (ROOT / "orion_core" / "telephony.py").read_text(
    encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No ambient credentials, from any of the three layers."""
    monkeypatch.setattr(telephony_direct, "TELEPHONY_PATH",
                        tmp_path / "telephony.json")
    monkeypatch.setattr(telephony_direct, "_from_mcp_config", lambda: {})
    for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_API_KEY",
                 "TWILIO_API_SECRET", "TWILIO_FROM_NUMBER"):
        monkeypatch.delenv(name, raising=False)


def _complete(monkeypatch, **overrides):
    values = {"TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
              "TWILIO_AUTH_TOKEN": "s3cret-token",
              "TWILIO_FROM_NUMBER": "+441234567890"}
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


# -- what is missing, and saying so ------------------------------------------

def test_nothing_set_names_all_three():
    gaps = telephony_direct.missing()
    assert len(gaps) == 3
    assert any("Account SID" in g for g in gaps)
    assert any("Auth Token" in g for g in gaps)
    assert any("call FROM" in g for g in gaps)


def test_a_partial_setup_names_only_what_is_left(monkeypatch):
    """The whole point: say which value, not "something is wrong"."""
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC" + "0" * 32)
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "s3cret-token")
    gaps = telephony_direct.missing()
    assert len(gaps) == 1 and "call FROM" in gaps[0]


def test_a_complete_setup_is_complete(monkeypatch):
    _complete(monkeypatch)
    assert telephony_direct.read_credentials().complete


def test_the_secret_is_never_in_the_diagnostic(monkeypatch):
    """It is a bearer credential for an account that can be charged."""
    _complete(monkeypatch)
    assert "s3cret-token" not in telephony_direct.describe()
    assert "s3cret-token" not in " ".join(telephony_direct.missing())
    assert "s3cret-token" not in telephony_direct.read_credentials().source


def test_the_diagnostic_points_at_the_one_step(monkeypatch):
    assert "setup_twilio" in telephony_direct.describe()


# -- where credentials come from ---------------------------------------------

def test_the_environment_wins(monkeypatch):
    telephony_direct.save_credentials("AC" + "1" * 32, "stored", "+440000000")
    _complete(monkeypatch)
    assert telephony_direct.read_credentials().password == "s3cret-token"


def test_orions_own_file_is_read(monkeypatch):
    telephony_direct.save_credentials("AC" + "1" * 32, "stored", "+441111111")
    creds = telephony_direct.read_credentials()
    assert creds.complete and creds.password == "stored"


def test_credentials_already_in_the_mcp_block_are_reused(monkeypatch):
    """Somebody who filled in mcp_servers.json should not type it all again."""
    monkeypatch.setattr(telephony_direct, "_from_mcp_config", lambda: {
        "TWILIO_ACCOUNT_SID": "AC" + "2" * 32,
        "TWILIO_API_KEY": "SK" + "3" * 32,
        "TWILIO_API_SECRET": "api-secret",
        "TWILIO_FROM_NUMBER": "+442222222",
    })
    creds = telephony_direct.read_credentials()
    assert creds.complete
    # An API key is a separate identity acting ON the account, so it is the
    # username while the account SID stays in the URL path. The other way
    # round is a 401 that reads like bad credentials.
    assert creds.username.startswith("SK")
    assert creds.account_sid.startswith("AC")


def test_an_auth_token_authenticates_as_the_account(monkeypatch):
    _complete(monkeypatch)
    creds = telephony_direct.read_credentials()
    assert creds.username == creds.account_sid
    assert creds.password == "s3cret-token"


def test_the_secret_is_written_but_not_the_empty_keys(tmp_path):
    path = telephony_direct.save_credentials("AC" + "4" * 32, "tok", "+4433")
    written = json.loads(Path(path).read_text(encoding="utf-8"))
    assert written["TWILIO_AUTH_TOKEN"] == "tok"
    assert "TWILIO_API_KEY" not in written, "an empty key would shadow nothing"


# -- placing the call --------------------------------------------------------

class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def _fake_post(monkeypatch, capture: dict, response: _Response):
    async def post(path, data, credentials):
        capture["path"] = path
        capture["data"] = data
        capture["credentials"] = credentials
        if response.status_code in (200, 201):
            return True, "", str(response.json().get("sid", ""))
        return False, str(response.json().get("message", "")), ""

    monkeypatch.setattr(telephony_direct, "_post", post)
    return capture


def test_a_call_sends_the_twiml_inline(monkeypatch):
    _complete(monkeypatch)
    seen = _fake_post(monkeypatch, {}, _Response(201, {"sid": "CA" + "a" * 32}))
    ok, detail, sid = asyncio.run(
        telephony_direct.place_call("+447700900000", "<Response/>"))
    assert ok and sid.startswith("CA"), detail
    assert seen["path"] == "Calls.json"
    # Inline TwiML is what removes the need to host a webhook.
    assert seen["data"]["Twiml"] == "<Response/>"
    assert seen["data"]["To"] == "+447700900000"
    assert seen["data"]["From"] == "+441234567890"


def test_a_call_without_credentials_says_what_is_missing(monkeypatch):
    ok, detail, _sid = asyncio.run(
        telephony_direct.place_call("+447700900000", "<Response/>"))
    assert not ok
    assert "Account SID" in detail


def test_twilios_own_refusal_reaches_the_user(monkeypatch):
    """"Unverified number" is what a trial account actually says, and it is
    far more use than anything generic."""
    _complete(monkeypatch)
    _fake_post(monkeypatch, {}, _Response(
        400, {"message": "The number +447700900000 is unverified",
              "code": 21219}))
    ok, detail, _sid = asyncio.run(
        telephony_direct.place_call("+447700900000", "<Response/>"))
    assert not ok and "unverified" in detail


def test_an_sms_goes_to_the_messages_resource(monkeypatch):
    _complete(monkeypatch)
    seen = _fake_post(monkeypatch, {}, _Response(201, {"sid": "SM" + "b" * 32}))
    ok, _detail, sid = asyncio.run(
        telephony_direct.send_sms("+447700900000", "on my way"))
    assert ok and sid.startswith("SM")
    assert seen["path"] == "Messages.json"
    assert seen["data"]["Body"] == "on my way"


# -- the gateway chooses a route ---------------------------------------------

class _Contacts:
    def resolve(self, who):
        return "+447700900000"

    def names(self):
        return ["friend"]


def _gateway(monkeypatch, **kwargs):
    gateway = telephony.TelephonyGateway(contacts=_Contacts(), **kwargs)
    return gateway


def test_credentials_alone_make_calling_available(monkeypatch):
    """No MCP server, no Node, no npx -- and ORION can still dial."""
    _complete(monkeypatch)
    gateway = _gateway(monkeypatch)
    assert gateway.available()
    assert not gateway._mcp_connected()
    assert gateway._direct_ready()


def test_without_either_route_it_is_unavailable(monkeypatch):
    gateway = _gateway(monkeypatch)
    assert not gateway.available()


def test_the_reason_given_is_the_missing_credential(monkeypatch):
    """Not "enable 'twilio' in config/mcp_servers.json and restart", which
    was wrong about the file, the server and the restart."""
    gateway = _gateway(monkeypatch)
    why = gateway.why_unavailable()
    assert "Account SID" in why
    assert "restart" not in why.lower()


def test_a_connected_server_is_still_preferred(monkeypatch):
    """It does more than dial, and it is the audit trail the model's own
    tool calls already go through."""
    _complete(monkeypatch)
    used = []

    class _Host:
        def server_names(self):
            return ["twilio"]

    gateway = _gateway(monkeypatch, mcp=_Host())

    async def _invoke(tool, arguments, number, channel):
        used.append(tool)
        return telephony.Dispatch(True, channel, "ok", to=number)

    monkeypatch.setattr(gateway, "_invoke", _invoke)

    async def _direct(*a, **k):
        used.append("direct")
        return telephony.Dispatch(True, telephony.Channel.VOICE_CALL, "ok")

    monkeypatch.setattr(gateway, "_direct", _direct)

    result = asyncio.run(gateway.call("friend", "hello"))
    assert result.ok
    assert used == ["create_call"], "the direct route jumped the MCP server"


def test_the_direct_route_is_used_when_no_server_is_connected(monkeypatch):
    _complete(monkeypatch)
    gateway = _gateway(monkeypatch)
    used = []

    async def _direct(channel, number, payload):
        used.append((channel, number, payload))
        return telephony.Dispatch(True, channel, "Calling.", to=number)

    monkeypatch.setattr(gateway, "_direct", _direct)
    result = asyncio.run(gateway.call("friend", "hello"))
    assert result.ok and used, "the direct route was never reached"
    assert "<Response>" in used[0][2] or "Say" in used[0][2]


# -- the gate is untouched ---------------------------------------------------

def test_adding_a_transport_opened_no_route_around_the_spend_gate():
    """A second way to dial must not be a second way to dial UNAPPROVED.

    The token lives above the gateway, in the tool that calls it, so the
    direct route inherits it -- but only as long as nothing here reaches for
    a transport of its own.
    """
    web = (ROOT / "orion_core" / "dispatch_web.py").read_text(encoding="utf-8")
    assert "ActionIntent.PLACE_CALL" in web
    assert "request_confirmation" in web
    # The gateway must not be able to place a call except through the two
    # named routes, both of which sit below that gate.
    assert "httpx" not in GATEWAY_SOURCE, (
        "telephony.py reached for HTTP directly, below its own routing")


def test_the_direct_transport_says_it_is_behind_the_gate():
    assert "_direct" in GATEWAY_SOURCE
    doc = inspect.getdoc(telephony.TelephonyGateway._direct) or ""
    assert "guard" in doc.lower() or "token" in doc.lower()


def test_the_setup_tool_verifies_before_it_claims_success():
    """A setup tool that reports success on credentials that do not work is
    worse than no setup tool."""
    source = (ROOT / "tools" / "setup_twilio.py").read_text(encoding="utf-8")
    assert source.index("_verify(candidate)") < source.index(
        "save_credentials")
    assert "Nothing was written." in source
