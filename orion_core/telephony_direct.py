"""
Place a real phone call without node, npx, or an MCP server.

Why this exists
---------------
ORION's telephony went through Twilio's MCP server, launched with
``npx -y @twilio-alpha/mcp``. On this machine — and on any Windows install
without Node — **npx is not on PATH**, so that server could never start. No
amount of correct credentials would have helped: outbound calling was
structurally impossible, and the only thing the user ever saw was "No
telephony server is connected", which points at the wrong problem entirely.

Placing a call does not need any of that machinery. Twilio's REST API is one
HTTPS POST with HTTP basic auth, and it accepts the TwiML inline, so there is
no webhook to host and nothing public to expose. That removes Node, npx, a
subprocess, a JSON-RPC transport and a community package's release-to-release
tool renaming from the path between "ring my friend" and a ringing phone.

The MCP server stays supported and stays first when it is connected: it can do
more than dial (lookups, recordings, account queries) and it is the one audit
trail the model's own tool calls already go through. This is the fallback that
makes the common case work on a machine that has nothing installed.

What it still cannot do
-----------------------
Call "without it being set up", which is what was asked for. A phone call is
billed to somebody, and Twilio will not place one for an account that does not
exist. Three values are needed — an Account SID, a token, and a number to call
FROM — and no code can invent them.

What it can do is make that the *only* step, stop pretending the problem is
elsewhere, and say precisely which of the three is missing and where to put
it. ``tools/setup_twilio.py`` writes all three in one go.

Security
--------
This is a transport. It is called from ``TelephonyAgent``, which already sits
behind the single-use human-issued token that ``ActionIntent.PLACE_CALL``
requires — spending money is not gated by a ``confirm`` flag the model fills
in for itself. Nothing here weakens that, and nothing here is reachable
without passing it first.

The token is never logged. It is a bearer credential for an account that can
be charged, so it is read, used as a basic-auth password, and not put anywhere
it could be read back — not in a log line, not in an error message, not in the
diagnostic that reports what is missing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

#: Where credentials are written by ``tools/setup_twilio.py``.
TELEPHONY_PATH = CONFIG_DIR / "telephony.json"

#: Twilio's REST base. Versioned by Twilio, not by us.
API_ROOT = "https://api.twilio.com/2010-04-01"

#: A call that has not connected within this many seconds has failed for a
#: reason no retry will fix.
TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class Credentials:
    """What Twilio needs to bill somebody for a call.

    ``username``/``password`` are the HTTP basic pair. Twilio accepts two
    kinds and both are supported, because the two places ORION already
    documented credentials ask for different ones: the MCP server block wants
    an API key and secret, while ``telephony_server.py`` reads an auth token.
    Either is valid for the REST API, so the one that is present wins rather
    than the user being told to go and find the other.
    """

    account_sid: str
    username: str
    password: str
    from_number: str
    #: Where these came from, for the diagnostic. Never includes the secret.
    source: str = ""

    @property
    def complete(self) -> bool:
        return bool(self.account_sid and self.username
                    and self.password and self.from_number)


def _stored() -> dict[str, Any]:
    try:
        data = json.loads(TELEPHONY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _from_mcp_config() -> dict[str, str]:
    """The twilio server's env block, so one set of credentials serves both.

    Somebody who has already filled in ``config/mcp_servers.json`` should not
    have to type the same three values again to use a different transport.
    """
    try:
        from .mcp_host import MCP_CONFIG_PATH

        data = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
        env = data.get("servers", {}).get("twilio", {}).get("env", {})
        return {str(k): str(v) for k, v in env.items() if v}
    except Exception:
        return {}


def read_credentials() -> Credentials:
    """Whatever credentials can be found, from wherever they are.

    Environment first (a one-off, or a machine that injects secrets), then
    ORION's own file, then the MCP server block. An incomplete result is
    returned rather than None so ``missing()`` can say which part is absent.
    """
    layers: list[tuple[str, dict[str, str]]] = [
        ("the environment", {k: v for k, v in os.environ.items()
                             if k.startswith("TWILIO_") and v.strip()}),
        (f"config/{TELEPHONY_PATH.name}",
         {str(k): str(v) for k, v in _stored().items() if v}),
        ("config/mcp_servers.json", _from_mcp_config()),
    ]

    def pick(*names: str) -> tuple[str, str]:
        for source, values in layers:
            for name in names:
                value = str(values.get(name, "") or "").strip()
                if value:
                    return value, source
        return "", ""

    account_sid, origin = pick("TWILIO_ACCOUNT_SID", "account_sid")
    token, _ = pick("TWILIO_AUTH_TOKEN", "auth_token")
    key, _ = pick("TWILIO_API_KEY", "api_key")
    secret, _ = pick("TWILIO_API_SECRET", "api_secret")
    from_number, _ = pick("TWILIO_FROM_NUMBER", "from_number", "from")

    # An auth token authenticates AS the account, so the account SID is both
    # the username and the path. An API key is a separate identity that acts
    # on the account, so it goes in the username and the account SID stays in
    # the path. Getting this pair the wrong way round is a 401 that reads
    # like bad credentials.
    if token:
        username, password = account_sid, token
    else:
        username, password = key, secret

    return Credentials(account_sid=account_sid, username=username,
                       password=password, from_number=from_number,
                       source=origin or "nowhere")


def missing(credentials: Credentials | None = None) -> list[str]:
    """Exactly which pieces are absent, in words that say what to do.

    The old message — "No telephony server is connected. Enable 'twilio' in
    config/mcp_servers.json and restart." — named a file, a server and a
    restart, and was wrong about all three whenever the real problem was an
    empty credential. Saying which value is missing is the difference between
    a user fixing it and a user concluding the feature is broken.
    """
    creds = credentials or read_credentials()
    gaps: list[str] = []
    if not creds.account_sid:
        gaps.append("the Account SID (it starts with AC, on the Twilio "
                    "console home page)")
    if not creds.password:
        gaps.append("an Auth Token (on the same page) or an API key and "
                    "secret")
    if not creds.from_number:
        gaps.append("a Twilio phone number to call FROM, in +44... form")
    return gaps


def describe() -> str:
    """One line for the user, and for ORION to say out loud."""
    creds = read_credentials()
    if creds.complete:
        return (f"I can dial directly through Twilio from "
                f"{creds.from_number} (credentials from {creds.source}).")
    gaps = missing(creds)
    return ("I cannot place calls yet — I still need " + ", and ".join(gaps)
            + ". Run  python tools/setup_twilio.py  to set all of it in one "
              "step.")


async def _post(path: str, data: dict[str, str],
                credentials: Credentials) -> tuple[bool, str, str]:
    """POST to Twilio. Returns (ok, human detail, resource SID)."""
    import httpx

    url = f"{API_ROOT}/Accounts/{credentials.account_sid}/{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                url, data=data,
                auth=(credentials.username, credentials.password))
    except Exception as exc:
        # The token could not appear here - it is never put in a URL - but the
        # message is still kept to the exception type and text rather than
        # echoing the request.
        return False, f"Twilio was unreachable ({type(exc).__name__}).", ""

    try:
        body = response.json()
    except Exception:
        body = {}

    if response.status_code in (200, 201):
        return True, "", str(body.get("sid") or "")

    # Twilio's own message is far better than anything generic: it says
    # "unverified number", "insufficient funds" or "not a valid phone number"
    # in plain words, which is what the user needs to hear.
    detail = str(body.get("message") or f"HTTP {response.status_code}")
    code = body.get("code")
    if code:
        detail = f"{detail} (Twilio code {code})"
    return False, detail, ""


async def place_call(to: str, twiml: str,
                     credentials: Credentials | None = None
                     ) -> tuple[bool, str, str]:
    """Ring *to* and speak *twiml*. Returns (ok, detail, call SID).

    The TwiML travels inline, so there is no webhook to host and nothing of
    ORION's has to be reachable from the internet.
    """
    creds = credentials or read_credentials()
    if not creds.complete:
        return False, "I still need " + ", and ".join(missing(creds)) + ".", ""
    return await _post("Calls.json",
                       {"To": to, "From": creds.from_number, "Twiml": twiml},
                       creds)


async def send_sms(to: str, body: str,
                   credentials: Credentials | None = None
                   ) -> tuple[bool, str, str]:
    """Send an SMS. Returns (ok, detail, message SID)."""
    creds = credentials or read_credentials()
    if not creds.complete:
        return False, "I still need " + ", and ".join(missing(creds)) + ".", ""
    return await _post("Messages.json",
                       {"To": to, "From": creds.from_number, "Body": body},
                       creds)


def save_credentials(account_sid: str, token: str, from_number: str,
                     api_key: str = "", api_secret: str = "") -> Any:
    """Write credentials to ORION's own file. Returns the path."""
    payload = {
        "schema": "orion.telephony.v1",
        "TWILIO_ACCOUNT_SID": account_sid.strip(),
        "TWILIO_FROM_NUMBER": from_number.strip(),
    }
    if token.strip():
        payload["TWILIO_AUTH_TOKEN"] = token.strip()
    if api_key.strip():
        payload["TWILIO_API_KEY"] = api_key.strip()
    if api_secret.strip():
        payload["TWILIO_API_SECRET"] = api_secret.strip()

    TELEPHONY_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(TELEPHONY_PATH, json.dumps(payload, indent=2), encoding="utf-8")
    try:
        # Best effort: on Windows this is advisory, but leaving a bearer
        # credential world-readable when it costs nothing to narrow it would
        # be careless.
        os.chmod(TELEPHONY_PATH, 0o600)
    except Exception:
        pass
    return TELEPHONY_PATH


__all__ = [
    "API_ROOT", "TELEPHONY_PATH", "TIMEOUT_SECONDS", "Credentials",
    "describe", "missing", "place_call", "read_credentials",
    "save_credentials", "send_sms",
]
