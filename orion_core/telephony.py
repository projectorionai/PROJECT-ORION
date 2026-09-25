"""
ORION on the telephone.

He could already hand a `call` intent to the paired Android app, which opens
the dialler pre-filled and waits for a tap. That is the safe path and it stays
the default — but it cannot ring you, which is the whole point of a reminder
that has to reach someone who is not at their desk.

This is the other path: ORION places the call himself, through Twilio, and
speaks. One-way for a briefing or an alert; two-way, over a media stream, when
there is a conversation to have.

Delivery is a property of the reminder, not of this module
----------------------------------------------------------
``ReminderService`` gained channels — DESKTOP, MOBILE_PUSH, SMS, VOICE_CALL —
and this is what the last two resolve to. A reminder that says how it wants to
arrive is far more useful than a reminder that always arrives the same way: an
8 a.m. briefing should ring you, a "stand up" nudge should not.

What it refuses to say
----------------------
Anything that looks like a credential, a card number or a high-entropy token
is removed before it is spoken or texted, by the same
:class:`~orion_core.spillage_guard.SpillageGuard` that already watches ORION's
other outbound channels. This matters more here than anywhere else in ORION:
a phone call is the one output that cannot be un-sent, is often recorded by
the far end, and may be heard by someone who is not the user. The guard fails
CLOSED — if it cannot be loaded, nothing is dialled.

Real calls cost money and dial immediately
------------------------------------------
Unlike the phone hand-off there is no tap to confirm. So every outbound call
is gated: the number must be one ORION has been told about, the text must pass
the guard, and a call placed by a voice command that arrived *over the
telephone* is refused outright — otherwise anyone who can reach the number can
make ORION call anyone else.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape as xml_escape

#: The longest thing ORION will say down a phone line in one go. Past about
#: this, a listener has stopped listening and the call is costing money by the
#: minute; the briefing gets trimmed with a pointer to the full version.
MAX_SPOKEN_CHARS = 1400

#: Twilio's neural voices. British by default, because ORION is.
DEFAULT_VOICE = "Polly.Amy-Neural"

#: Where a number has to appear before ORION will dial it.
CONTACTS_PATH = "telephony_contacts.json"


class Channel(str, Enum):
    """How a reminder or alert should reach the user."""

    DESKTOP = "DESKTOP"            # speak it here, the way it always worked
    MOBILE_PUSH = "MOBILE_PUSH"    # the paired app's notification
    SMS = "SMS"                    # a text, via Twilio
    VOICE_CALL = "VOICE_CALL"      # ORION rings and speaks

    @classmethod
    def parse(cls, value: Any, default: "Channel" = None) -> "Channel":
        """Read a channel from whatever the caller had.

        Unknown values fall back to DESKTOP rather than raising: a reminder
        with a typo in its channel should still fire, on the safest channel,
        rather than being silently dropped.
        """
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().upper().replace("-", "_")
        aliases = {
            "CALL": cls.VOICE_CALL, "PHONE": cls.VOICE_CALL,
            "VOICE": cls.VOICE_CALL, "RING": cls.VOICE_CALL,
            "TEXT": cls.SMS, "MESSAGE": cls.SMS,
            "PUSH": cls.MOBILE_PUSH, "MOBILE": cls.MOBILE_PUSH,
            "LOCAL": cls.DESKTOP, "SPEAK": cls.DESKTOP, "HERE": cls.DESKTOP,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError:
            return default or cls.DESKTOP


#: The channels that leave this machine and cost money.
OUTBOUND = frozenset({Channel.SMS, Channel.VOICE_CALL})


# ── numbers ───────────────────────────────────────────────────────────────────

_E164 = re.compile(r"\A\+[1-9]\d{6,14}\Z")


def normalise_number(raw: str, default_country: str = "44") -> str:
    """A phone number in E.164, or "" if it is not one.

    UK numbers get written five different ways and only one of them can be
    dialled through an API. "07700 900123", "+44 7700 900123" and
    "0044-7700-900123" are the same number; a dialler that accepts only the
    third is a dialler nobody uses.
    """
    text = re.sub(r"[\s()\-.]", "", str(raw or ""))
    if not text:
        return ""
    if text.startswith("+"):
        return text if _E164.match(text) else ""
    if text.startswith("00"):
        text = "+" + text[2:]
    elif text.startswith("0"):
        text = f"+{default_country}{text[1:]}"
    elif text.isdigit():
        text = f"+{text}"
    return text if _E164.match(text) else ""


@dataclass
class Contact:
    """Somewhere ORION is permitted to dial."""

    name: str
    number: str
    note: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "number": self.number, "note": self.note}


class ContactBook:
    """The numbers ORION may call, and nothing else.

    A default-deny list rather than a convenience. ORION reaches a model that
    can be talked into things, and "call this number and read out what I tell
    you" is a usable attack the moment any number is dialable. The user adds
    numbers deliberately; the model cannot add one on its own behalf.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self._contacts: dict[str, Contact] = {}
        self.reload()

    def reload(self) -> None:
        self._contacts.clear()
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for entry in (raw.get("contacts") or []):
            if not isinstance(entry, dict):
                continue
            number = normalise_number(entry.get("number", ""))
            name = str(entry.get("name") or "").strip()
            if number and name:
                self._contacts[number] = Contact(name, number,
                                                 str(entry.get("note") or ""))

    def allows(self, number: str) -> bool:
        return normalise_number(number) in self._contacts

    def resolve(self, who: str) -> str:
        """A name or a number in, a dialable number out, or "".

        Accepts a raw number too — but only if it is already on the list.
        """
        direct = normalise_number(who)
        if direct and direct in self._contacts:
            return direct
        wanted = str(who or "").strip().casefold()
        for contact in self._contacts.values():
            if contact.name.casefold() == wanted:
                return contact.number
        return ""

    def names(self) -> list[str]:
        return sorted(contact.name for contact in self._contacts.values())

    def __len__(self) -> int:
        return len(self._contacts)


# ── what ORION is allowed to say ──────────────────────────────────────────────

@dataclass
class Screened:
    """The result of checking something before it goes down the line."""

    text: str
    safe: bool
    reason: str = ""
    removed: list[str] = field(default_factory=list)


def screen(text: str, guard: Any = None) -> Screened:
    """Strip anything that must not be spoken aloud.

    Fails CLOSED. If the guard cannot be loaded this returns unsafe rather
    than passing the text through unchecked — a phone call cannot be un-sent,
    is often recorded at the far end, and may be heard by someone who is not
    the user.
    """
    body = str(text or "").strip()
    if not body:
        return Screened("", False, "there was nothing to say")

    if guard is None:
        try:
            from .spillage_guard import SpillageGuard

            guard = SpillageGuard()
        except Exception as exc:
            return Screened("", False,
                            f"the spillage guard is unavailable ({exc})")

    try:
        report = guard.scan(body)
    except Exception as exc:
        return Screened("", False, f"the text could not be screened ({exc})")

    removed: list[str] = []
    if not getattr(report, "safe", True):
        # Cut by SPAN, not by matching the secret's text — a Finding
        # deliberately never carries the raw value, only a redacted preview
        # and where it was. Applied back-to-front so each excision cannot
        # shift the offsets of the ones still to come.
        spans: list[tuple[int, int, str]] = []
        for finding in getattr(report, "findings", []) or []:
            span = getattr(finding, "span", None)
            kind = str(getattr(finding, "kind", "something sensitive"))
            if (isinstance(span, (tuple, list)) and len(span) == 2
                    and 0 <= int(span[0]) < int(span[1]) <= len(body)):
                spans.append((int(span[0]), int(span[1]), kind))
        for start, end, kind in sorted(spans, reverse=True):
            body = body[:start] + "[removed]" + body[end:]
            removed.append(kind)
        if not removed:
            # Something was found but could not be located precisely enough to
            # cut out. Refusing is the only safe answer.
            return Screened("", False, getattr(report, "describe", lambda: "")()
                            or "it contained something sensitive")
        # Re-scan what is left. One secret can hide another, and the excision
        # itself can join two fragments into a new match.
        try:
            if not guard.scan(body).safe:
                return Screened("", False,
                                "it still contained something sensitive after "
                                "redaction")
        except Exception:
            return Screened("", False, "the redacted text could not be re-checked")

    if len(body) > MAX_SPOKEN_CHARS:
        body = body[:MAX_SPOKEN_CHARS].rsplit(" ", 1)[0] + \
            "… the rest is on your screen."
    return Screened(body, True, removed=removed)


# ── TwiML ─────────────────────────────────────────────────────────────────────

def twiml_say(text: str, voice: str = DEFAULT_VOICE,
              language: str = "en-GB") -> str:
    """A one-way spoken message, as TwiML.

    Escaped properly: a briefing headline containing an ampersand would
    otherwise produce malformed XML and Twilio would play an error tone
    instead of the news.
    """
    body = xml_escape(str(text or "").strip())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Say voice="{xml_escape(voice)}" language="{xml_escape(language)}">'
        f"{body}</Say>"
        "<Pause length=\"1\"/>"
        "</Response>"
    )


def twiml_stream(websocket_url: str, greeting: str = "",
                 voice: str = DEFAULT_VOICE, language: str = "en-GB") -> str:
    """A two-way call, bridged to ORION's conversational engine.

    ``<Connect><Stream>`` hands the call's audio to a WebSocket in both
    directions — 8 kHz G.711 μ-law, base64 in JSON frames. The greeting is
    spoken first so the far end hears something immediately rather than
    silence while the socket comes up.
    """
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]
    if greeting:
        parts.append(
            f'<Say voice="{xml_escape(voice)}" language="{xml_escape(language)}">'
            f"{xml_escape(greeting)}</Say>")
    parts.append(f'<Connect><Stream url="{xml_escape(websocket_url)}"/></Connect>')
    parts.append("</Response>")
    return "".join(parts)


# ── placing the call ──────────────────────────────────────────────────────────

@dataclass
class Dispatch:
    """What ORION did, or why he did not."""

    ok: bool
    channel: Channel
    detail: str
    to: str = ""
    sid: str = ""

    def __str__(self) -> str:
        return self.detail


class TelephonyGateway:
    """Places calls and sends texts through the Twilio MCP server.

    Goes through the MCP host rather than importing a Twilio SDK: the server
    is already declared in ``config/mcp_servers.json``, the credentials
    already live there, and routing through MCP means the same tool the model
    can call is the one this uses — so there is one audit trail, not two.
    """

    def __init__(self, mcp: Any = None, bus: Any = None,
                 contacts: ContactBook | None = None,
                 guard: Any = None) -> None:
        self.mcp = mcp
        self.bus = bus
        self.contacts = contacts or ContactBook()
        self.guard = guard

    # ── availability ─────────────────────────────────────────────────────────

    def _mcp_connected(self) -> bool:
        """Whether a telephony MCP server is actually connected."""
        host = self.mcp
        if host is None:
            return False
        try:
            for name in getattr(host, "server_names", lambda: [])():
                if "twilio" in str(name).lower():
                    return True
        except Exception:
            pass
        try:
            return bool(getattr(host, "servers", {}).get("twilio"))
        except Exception:
            return False

    def _direct_ready(self) -> bool:
        """Whether ORION can reach Twilio's REST API on his own.

        The MCP server needs ``npx``, which a Windows machine without Node
        does not have — and did not have here, so outbound calling could
        never have worked however good the credentials were. The REST API
        needs nothing but the credentials.
        """
        try:
            from . import telephony_direct

            return telephony_direct.read_credentials().complete
        except Exception:
            return False

    def available(self) -> bool:
        """Whether ORION can place a call at all, by either route."""
        return self._mcp_connected() or self._direct_ready()

    def why_unavailable(self) -> str:
        """What is actually missing, in words that say what to do.

        The old message named a file, a server and a restart, and was wrong
        about all three whenever the real problem was an empty credential.
        """
        try:
            from . import telephony_direct

            return telephony_direct.describe()
        except Exception:
            return ("I cannot place calls: no telephony server is connected "
                    "and I have no Twilio credentials.")

    def _prefer_server(self) -> bool:
        """Whether to go through the MCP server rather than straight to Twilio.

        The server wins whenever it is there: it does more than dial, and it
        is the audit trail the model's own tool calls already go through. It
        is skipped only when it is known NOT to be connected and the direct
        route is ready — no point paying for a round trip to something that
        cannot answer when the alternative works.

        ``self.mcp is not None`` is checked separately from connectedness on
        purpose. Detection reads ``server_names``/``servers``, and a host that
        exposes neither is not necessarily absent; falling straight through to
        the direct route on a detection miss would quietly stop using a server
        that was working perfectly well.
        """
        if self.mcp is None:
            return False
        return self._mcp_connected() or not self._direct_ready()

    def _log(self, line: str) -> None:
        if self.bus is None:
            return
        try:
            self.bus.log.emit(f"[Phone] {line}")
        except Exception:
            pass

    # ── the two outbound actions ─────────────────────────────────────────────

    async def call(self, to: str, message: str, *,
                   twiml: str = "", reason: str = "") -> Dispatch:
        """Ring *to* and speak *message*.

        Dials immediately and costs money — there is no tap-to-confirm the way
        there is on the phone hand-off. Hence the number must already be in
        the contact book, and the message must survive screening.
        """
        number = self.contacts.resolve(to)
        if not number:
            return Dispatch(
                False, Channel.VOICE_CALL,
                f"I don't have {to!r} in the numbers I'm allowed to dial. "
                f"Add it to config/{CONTACTS_PATH} first.")

        screened = screen(message, self.guard)
        if not screened.safe:
            return Dispatch(False, Channel.VOICE_CALL,
                            f"I won't say that down a phone line — "
                            f"{screened.reason}.", to=number)
        if screened.removed:
            self._log(f"removed {', '.join(screened.removed)} before dialling")

        if not self.available():
            return Dispatch(False, Channel.VOICE_CALL,
                            self.why_unavailable(), to=number)

        markup = twiml or twiml_say(screened.text)
        self._log(f"calling {number}"
                  f"{' — ' + reason if reason else ''}")
        if self._prefer_server():
            result = await self._invoke("create_call",
                                        {"to": number, "twiml": markup},
                                        number, Channel.VOICE_CALL)
            if result.ok or not self._direct_ready():
                return result
            self._log("the telephony server would not place it — going "
                      "straight to Twilio")
        return await self._direct(Channel.VOICE_CALL, number, markup)

    async def text(self, to: str, message: str) -> Dispatch:
        """Send an SMS. Cheaper, quieter, and it leaves a record they can read."""
        number = self.contacts.resolve(to)
        if not number:
            return Dispatch(
                False, Channel.SMS,
                f"I don't have {to!r} in the numbers I'm allowed to message.")
        screened = screen(message, self.guard)
        if not screened.safe:
            return Dispatch(False, Channel.SMS,
                            f"I won't send that — {screened.reason}.", to=number)
        if not self.available():
            return Dispatch(False, Channel.SMS, self.why_unavailable(),
                            to=number)
        if self._prefer_server():
            result = await self._invoke("send_message",
                                        {"to": number, "body": screened.text},
                                        number, Channel.SMS)
            if result.ok or not self._direct_ready():
                return result
            self._log("the telephony server would not send it — going "
                      "straight to Twilio")
        return await self._direct(Channel.SMS, number, screened.text)

    async def _direct(self, channel: Channel, number: str,
                      payload: str) -> Dispatch:
        """Reach Twilio's REST API without an MCP server in the way.

        Used when no telephony server is connected but credentials exist,
        which on a machine without Node is every time — ``npx`` is what
        launches the MCP server, and a Windows install without Node has none.

        The guard is unchanged: this is a transport, called from the same
        place the MCP route is, and everything above it still requires the
        single-use human-issued token that spending money needs.
        """
        from . import telephony_direct

        try:
            if channel is Channel.VOICE_CALL:
                ok, detail, sid = await telephony_direct.place_call(
                    number, payload)
            else:
                ok, detail, sid = await telephony_direct.send_sms(
                    number, payload)
        except Exception as exc:
            return Dispatch(False, channel,
                            f"I could not reach Twilio — "
                            f"{type(exc).__name__}.", to=number)
        if ok:
            verb = "Calling" if channel is Channel.VOICE_CALL else "Texting"
            return Dispatch(True, channel, f"{verb} {number}.",
                            to=number, sid=sid)
        return Dispatch(False, channel, detail or "Twilio refused that.",
                        to=number)

    async def _invoke(self, tool: str, arguments: dict, number: str,
                      channel: Channel) -> Dispatch:
        """Call the MCP tool, tolerating the server naming things differently.

        Community MCP servers rename their tools between releases, so the
        candidates are tried in order rather than one name being assumed. A
        rename should degrade to a clear message, not a stack trace.
        """
        candidates = {
            "create_call": ("create_call", "make_call", "call",
                            "twilio_create_call"),
            "send_message": ("send_message", "send_sms", "create_message",
                             "twilio_send_message"),
        }.get(tool, (tool,))

        last = ""
        for name in candidates:
            try:
                result = await self._mcp_call(name, arguments)
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                continue
            text = str(getattr(result, "text", result) or "")
            if self._succeeded(result, text):
                sid = ""
                match = re.search(r"\b(CA|SM|MM)[0-9a-f]{32}\b", text)
                if match:
                    sid = match.group(0)
                verb = "Calling" if channel is Channel.VOICE_CALL else "Texting"
                return Dispatch(True, channel, f"{verb} {number}.",
                                to=number, sid=sid)
            last = text
        return Dispatch(False, channel,
                        f"The telephony server would not do that — "
                        f"{last or 'no reason given'}.", to=number)

    async def _mcp_call(self, tool: str, arguments: dict) -> Any:
        """Invoke *tool* on the telephony server, whatever we were handed.

        MCPHost exposes ``call(server, tool, args)``. A single server
        CONNECTION exposes ``call_tool(tool, args)`` — two arguments, no
        server name. This used to call the second signature on the first
        object, which raised TypeError every time and reported it as the
        telephony server refusing.
        """
        host = self.mcp
        caller = getattr(host, "call", None)
        if callable(caller):
            return await caller("twilio", tool, arguments)
        caller = getattr(host, "call_tool", None)
        if callable(caller):
            try:
                return await caller(tool, arguments)
            except TypeError:
                # A test double, or a host that wants the server named too.
                return await caller("twilio", tool, arguments)
        raise AttributeError(
            "the MCP host exposes neither call() nor call_tool()")

    #: How a string result announces that nothing happened. MCPHost reports
    #: failures as ordinary text, so there is no status field to read.
    _FAILURE_MARKERS = (
        "no connected mcp server", "has no tool", "timed out", "failed:",
        "error", "not connected", "unauthor", "forbidden", "invalid",
    )

    def _succeeded(self, result: Any, text: str) -> bool:
        """Whether the server actually did it.

        A ToolResult-shaped object says so itself. A plain string does not,
        and ``getattr(result, "ok", True)`` therefore treated every failure
        message as a placed call — ORION would have said "Calling +44…"
        having done nothing at all.
        """
        flag = getattr(result, "ok", None)
        if flag is not None:
            return bool(flag)
        lowered = text.lower()
        if not lowered.strip():
            return False        # silence is not confirmation
        return not any(marker in lowered for marker in self._FAILURE_MARKERS)


__all__ = [
    "CONTACTS_PATH", "DEFAULT_VOICE", "MAX_SPOKEN_CHARS", "OUTBOUND",
    "Channel", "Contact", "ContactBook", "Dispatch", "Screened",
    "TelephonyGateway", "normalise_number", "screen", "twiml_say",
    "twiml_stream",
]
