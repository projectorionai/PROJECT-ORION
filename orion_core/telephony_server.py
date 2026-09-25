"""
The process Twilio actually talks to.

``telephony_bridge`` knows how to convert audio and speak Twilio's protocol.
This is the thing that listens: an aiohttp application serving the TwiML
endpoints over HTTP and the Media Streams socket over WebSocket, started by
the ``orion-telephony`` systemd unit.

They are separate on purpose. The bridge is pure logic and is tested without a
socket; this is the plumbing, and plumbing that cannot be tested without a
network is plumbing that should contain no decisions.

Everything here is refused by default
-------------------------------------
This is ORION's only listening port that faces the internet, so every entry
point starts from no:

  * **Every request is signature-checked** against Twilio's auth token before
    it is acted on. Caddy cannot add a bearer token to a request Twilio
    composes, so this check is the entire authentication. No token configured
    means every request is refused, not accepted.
  * **Inbound callers must be in the contact book.** A public phone number
    reaches a machine that can run tools.
  * **A caller cannot make ORION do anything.** A voice on a telephone is
    unauthenticated by nature — anyone can spoof caller ID, and a recording of
    the user's voice is trivial to obtain. So a call may ask questions and
    hear answers, and may not run a tool that changes anything without a
    second factor the telephone cannot supply. See :class:`CallAuthority`.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

#: What a caller is allowed to do without proving anything beyond being on the
#: contact list. Read-only, and nothing that leaves the machine.
SAFE_OVER_THE_PHONE: frozenset[str] = frozenset({
    "briefing", "catch_up", "situation_report", "recall_conversation",
    "query_intelligence", "second_brain", "awareness", "resource_status",
    "capabilities", "token_usage", "study", "decision", "geo",
    "programming_knowledge", "cyber_knowledge", "neuro_knowledge",
    "founder_knowledge", "flight_search", "research",
})

#: How long a spoken confirmation code stays valid. Long enough to read it off
#: a phone screen and say it; short enough that an overheard one is stale.
CONFIRM_WINDOW_S = 180.0

#: Digits in a confirmation code. Six is a compromise: short enough to say
#: down a bad line, long enough that guessing inside the window is hopeless.
CONFIRM_DIGITS = 6


class CallAuthority:
    """Decides what a voice on the telephone is allowed to make ORION do.

    The premise is that a telephone caller is **unauthenticated**. Caller ID
    is trivially spoofed, and voice is not a secret — a recording of someone
    saying "yes" is easy to obtain and a voice print does not survive a good
    one. So voice identification is treated as a convenience, never as
    authority.

    Read-only questions are answered freely. Anything that changes something,
    spends money or leaves the machine needs a code that is delivered to the
    paired phone or the desktop — a channel the caller does not control just
    by being on the line.
    """

    def __init__(self, bus: Any = None,
                 allowed: frozenset[str] | None = None) -> None:
        self.bus = bus
        self.allowed = allowed if allowed is not None else SAFE_OVER_THE_PHONE
        self._pending: dict[str, tuple[str, str, float]] = {}

    def permits(self, tool: str) -> bool:
        """Whether *tool* may run from a phone call with no second factor."""
        return str(tool or "").strip() in self.allowed

    def challenge(self, call_sid: str, tool: str) -> str:
        """Issue a code for *tool*, delivered somewhere the caller is not.

        Returned so it can be shown on the desktop and pushed to the paired
        phone. It is deliberately NOT spoken down the telephone: a code read
        to the caller proves nothing, because the caller is who we are unsure
        about.
        """
        code = "".join(secrets.choice("0123456789") for _ in range(CONFIRM_DIGITS))
        self._pending[str(call_sid)] = (code, str(tool), time.monotonic())
        if self.bus is not None:
            try:
                self.bus.banner.emit(
                    f"PHONE ASKED FOR '{tool}' — code {code}", 5)
                self.bus.phone_action.emit({
                    "kind": "share",
                    "text": f"ORION: a caller asked to run '{tool}'. "
                            f"If that was you, the code is {code}."})
            except Exception:
                pass
        return code

    def confirm(self, call_sid: str, spoken: str) -> tuple[bool, str]:
        """Check a code the caller read back. Single use, and it expires."""
        entry = self._pending.pop(str(call_sid), None)
        if entry is None:
            return False, "There's nothing waiting to be confirmed."
        code, tool, issued = entry
        if time.monotonic() - issued > CONFIRM_WINDOW_S:
            return False, "That code has expired. Ask again."
        # Digits only: "four one nine" and "419" are the same answer, and a
        # speech-to-text transcript will not agree with itself about spacing.
        said = "".join(character for character in str(spoken or "")
                       if character.isdigit())
        if not secrets.compare_digest(said, code):
            return False, "That code isn't right."
        return True, tool

    def refusal(self, tool: str) -> str:
        """What ORION says when a caller asks for something gated."""
        return (f"I can't do that over the phone without confirming it's you. "
                f"I've sent a code to your phone — read it back and I'll "
                f"run {tool}.")


class GatedDispatcher:
    """The dispatcher a telephone caller is talking through.

    This is where :class:`CallAuthority` stops being a policy and becomes an
    enforcement. Without it the authority is a class nobody asks — which is
    exactly the failure it is supposed to prevent.

    Every tool call made during a call passes through here. Read-only ones go
    straight on. Anything else is refused with a spoken explanation, and a
    code goes to the paired phone; the caller reads it back, and only then
    does the tool run.
    """

    #: What the caller says to answer a challenge. Matched loosely because a
    #: speech transcript of a six-digit code is never clean.
    CONFIRM_TOOLS = frozenset({"confirm", "confirm_action", "verify"})

    def __init__(self, inner: Any, authority: "CallAuthority",
                 call_sid: str = "") -> None:
        self.inner = inner
        self.authority = authority
        self.call_sid = str(call_sid or "")
        self.refused: list[str] = []
        self.allowed: list[str] = []
        self._pending_args: dict[str, Any] = {}

    async def dispatch(self, name: str, args: dict[str, Any] | None = None) -> Any:
        from .data import ToolResult

        tool = str(name or "").strip()
        payload = dict(args or {})

        if tool in self.CONFIRM_TOOLS:
            return await self._confirm(payload)

        if self.authority.permits(tool):
            self.allowed.append(tool)
            return await self.inner.dispatch(tool, payload)

        # Gated. The code is issued to the phone, never spoken down the line.
        self.refused.append(tool)
        self._pending_args = payload
        self.authority.challenge(self.call_sid, tool)
        return ToolResult(self.authority.refusal(tool), ok=False)

    async def _confirm(self, payload: dict[str, Any]) -> Any:
        from .data import ToolResult

        spoken = str(payload.get("code") or payload.get("text")
                     or payload.get("query") or "")
        ok, outcome = self.authority.confirm(self.call_sid, spoken)
        if not ok:
            return ToolResult(outcome, ok=False)
        # `outcome` is the tool the code was issued for — a code authorises
        # one specific action, not a session.
        self.allowed.append(outcome)
        return await self.inner.dispatch(outcome, dict(self._pending_args))

    def __getattr__(self, item: str) -> Any:
        """Anything else the engine wants, from the real dispatcher.

        Deliberately a pass-through rather than a whitelist: this class exists
        to gate `dispatch`, and hiding unrelated attributes would break the
        engine in ways that look like a different bug.
        """
        return getattr(self.inner, item)


@dataclass
class BridgeConfig:
    """How the bridge is reachable, and what it trusts."""

    host: str = "127.0.0.1"
    port: int = 8790
    auth_token: str = ""
    public_host: str = ""

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        return cls(
            host=os.getenv("ORION_TELEPHONY_HOST", "127.0.0.1").strip(),
            port=int(os.getenv("ORION_TELEPHONY_PORT", "8790") or 8790),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
            public_host=os.getenv("ORION_PUBLIC_HOST", "").strip(),
        )

    @property
    def usable(self) -> bool:
        """Whether this configuration can actually serve calls.

        Both are required and neither has a sensible default: without a token
        every request is refused, and without a public host Twilio has nowhere
        to fetch TwiML from.
        """
        return bool(self.auth_token and self.public_host)

    def problems(self) -> list[str]:
        missing = []
        if not self.auth_token:
            missing.append("TWILIO_AUTH_TOKEN is not set — every request will "
                           "be refused, which is the safe default but means "
                           "no calls work")
        if not self.public_host:
            missing.append("ORION_PUBLIC_HOST is not set — Twilio has no "
                           "address to fetch TwiML from")
        return missing


class TelephonyServer:
    """The aiohttp application serving Twilio's endpoints.

    Holds no decisions of its own: it authenticates, then hands off to
    ``telephony_bridge``. Everything interesting is tested there, without a
    socket.
    """

    def __init__(self, config: BridgeConfig | None = None, engine: Any = None,
                 bus: Any = None, contacts: Any = None,
                 authority: CallAuthority | None = None) -> None:
        self.config = config or BridgeConfig.from_env()
        self.engine = engine
        self.bus = bus
        self.contacts = contacts
        self.authority = authority or CallAuthority(bus)
        self._runner: Any = None
        self.calls = 0
        self.refused = 0
        self._media_lock = asyncio.Lock()

    # ── the application ──────────────────────────────────────────────────────

    def build_app(self) -> Any:
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/twilio/voice", self.handle_voice)
        app.router.add_get("/twilio/voice", self.handle_voice)
        app.router.add_post("/twilio/status", self.handle_status)
        app.router.add_get("/media", self.handle_media)
        app.router.add_get("/health", self.handle_health)
        return app

    async def start(self) -> None:
        from aiohttp import web

        for problem in self.config.problems():
            self._log(problem)

        app = self.build_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await site.start()
        self._log(f"listening on {self.config.host}:{self.config.port}"
                  f"{' (public: ' + self.config.public_host + ')' if self.config.public_host else ''}")

    async def stop(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            await runner.cleanup()

    # ── authentication ───────────────────────────────────────────────────────

    async def _authentic(self, request: Any) -> bool:
        """Whether this request is really from Twilio."""
        from .telephony_bridge import verify_twilio

        signature = request.headers.get("X-Twilio-Signature", "")
        params: dict[str, Any] = {}
        if request.method == "POST":
            try:
                params = dict(await request.post())
            except Exception:
                params = {}
        url = self._public_url(request)
        if verify_twilio(self.config.auth_token, signature, url, params):
            return True
        self.refused += 1
        self._log(f"refused an unsigned request to {request.path}")
        return False

    def _public_url(self, request: Any) -> str:
        """The URL as Twilio saw it, not as Caddy forwarded it.

        The signature is computed over the public HTTPS URL. Behind a reverse
        proxy the request arrives as http://127.0.0.1:8790/..., so verifying
        against what aiohttp sees fails every time — a subtle and total
        outage that looks like a credential problem.
        """
        host = self.config.public_host or request.host
        host = host.replace("https://", "").replace("http://", "").strip("/")
        return f"https://{host}{request.path_qs}"

    # ── the endpoints ────────────────────────────────────────────────────────

    async def handle_voice(self, request: Any) -> Any:
        """Twilio asks what to do with a call; the answer is TwiML."""
        from aiohttp import web

        from .telephony import twiml_say
        from .telephony_bridge import answer_twiml

        if not await self._authentic(request):
            return web.Response(status=403, text="forbidden")

        params = dict(await request.post()) if request.method == "POST" else {}
        caller = str(params.get("From") or "")
        direction = str(params.get("Direction") or "")
        inbound = "inbound" in direction.lower()

        if inbound and not self._caller_allowed(caller):
            self._log(f"refused an inbound call from {caller or 'a withheld number'}")
            return web.Response(
                content_type="text/xml",
                text=twiml_say("I'm sorry, I can't take calls from this "
                               "number."))

        self.calls += 1
        self._log(f"answering a call from {caller or 'unknown'}")
        return web.Response(content_type="text/xml", text=answer_twiml())

    async def handle_status(self, request: Any) -> Any:
        """Twilio's call-progress callbacks. Logged, never acted on."""
        from aiohttp import web

        if not await self._authentic(request):
            return web.Response(status=403, text="forbidden")
        params = dict(await request.post())
        self._log(f"call {params.get('CallSid', '?')} is "
                  f"{params.get('CallStatus', 'in an unknown state')}")
        return web.Response(text="")

    async def handle_media(self, request: Any) -> Any:
        """The Media Streams WebSocket: audio both ways, for one call."""
        from aiohttp import WSMsgType, web

        from .telephony_bridge import MAX_CALL_SECONDS, MediaStreamHandler, verify_twilio

        # Twilio signs the WSS upgrade as well as the HTTP webhooks. Validate
        # before accepting the socket so an arbitrary client cannot feed audio.
        signature = request.headers.get("X-Twilio-Signature", "")
        # Which scheme Twilio signs the upgrade under is not something the
        # offline tests can settle, and guessing wrong refuses every real
        # call. Both forms are equally unforgeable without the auth token.
        https_url = self._public_url(request)
        wss_url = https_url.replace("https://", "wss://", 1)
        if not any(verify_twilio(self.config.auth_token, signature, candidate)
                   for candidate in (wss_url, wss_url + "/", https_url, https_url + "/")):
            self.refused += 1
            return web.Response(status=403, text="forbidden")
        # The engine's dispatcher is scoped by this call's capability gate.
        # Only one call may own it at once; a second must never swap the gate.
        if self._media_lock.locked():
            return web.Response(status=409, text="another call is active")
        await self._media_lock.acquire()

        socket = web.WebSocketResponse(heartbeat=20.0)
        try:
            await socket.prepare(request)
        except BaseException:
            self._media_lock.release()
            raise

        handler = None
        started = time.monotonic()
        try:
            handler = MediaStreamHandler(self.engine, self.bus, self.contacts)
            # Whatever the engine reaches for during this call goes through the
            # gate. Attached per call, because a confirmation code authorises one
            # action on one call and nothing else.
            dispatcher = getattr(self.engine, "dispatcher", None)
            if dispatcher is not None and not isinstance(dispatcher, GatedDispatcher):
                handler.gate = GatedDispatcher(dispatcher, self.authority)
                try:
                    self.engine.dispatcher = handler.gate
                except Exception:
                    pass
            async for message in socket:
                if message.type is not WSMsgType.TEXT:
                    continue
                for frame in await handler.handle_message(message.data):
                    await socket.send_json(frame)
                if handler.session.ended:
                    break
                if time.monotonic() - started > MAX_CALL_SECONDS:
                    # A far end that never hangs up bills by the minute until
                    # someone notices.
                    self._log(f"{handler.session.describe()} hit the time "
                              f"limit and was closed")
                    break
        except Exception as exc:
            self._log(f"the media stream faulted - {exc}")
        finally:
            gate = getattr(handler, "gate", None)
            if gate is not None:
                # Restored even if the call faulted: leaving the gate in place
                # would silently apply telephone rules to the desktop.
                try:
                    self.engine.dispatcher = gate.inner
                except Exception:
                    pass
                if gate.refused:
                    self._log(f"refused over the phone: "
                              f"{', '.join(sorted(set(gate.refused)))}")
            try:
                await socket.close()
            finally:
                self._media_lock.release()
        return socket

    async def handle_health(self, request: Any) -> Any:
        """For the VPS checklist. Says nothing a stranger could use."""
        from aiohttp import web

        return web.json_response({
            "ok": True,
            "configured": self.config.usable,
            "calls": self.calls,
            "refused": self.refused,
        })

    def _caller_allowed(self, number: str) -> bool:
        if self.contacts is None:
            return False
        try:
            return bool(self.contacts.allows(number))
        except Exception:
            return False

    def _log(self, line: str) -> None:
        from .startup_report import say

        say(f"[Telephony] {line}", self.bus)


async def serve(config: BridgeConfig | None = None) -> None:
    """Run the bridge until it is stopped."""
    from .telephony import CONTACTS_PATH, ContactBook

    try:
        from .constants import CONFIG_DIR

        contacts = ContactBook(CONFIG_DIR / CONTACTS_PATH)
    except Exception:
        contacts = ContactBook()

    server = TelephonyServer(config=config, contacts=contacts)
    await server.start()
    try:
        await asyncio.Event().wait()          # until the unit stops us
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


def main() -> None:
    """``python -m orion_core.telephony_server``."""
    from .startup_report import say

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        say("[Telephony] stopped.")


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIRM_DIGITS", "CONFIRM_WINDOW_S", "SAFE_OVER_THE_PHONE",
    "BridgeConfig", "CallAuthority", "GatedDispatcher",
    "TelephonyServer", "main", "serve",
]
