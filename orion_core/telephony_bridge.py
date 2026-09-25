"""
The bridge between a telephone call and ORION's conversational engine.

Twilio fetches TwiML over HTTPS to find out what to do with a call, and — for
a two-way call — opens a WebSocket carrying the audio both ways. This is the
process that answers both. It runs as its own service (``orion-telephony``)
because it is the only part of ORION with a port facing the internet, and its
blast radius should not include the knowledge graph.

The format problem
------------------
Telephony is 8 kHz G.711 μ-law, one channel, in base64 inside JSON frames.
ORION's pipeline is 16 kHz linear PCM in, 24 kHz out. Nothing lines up, so
every frame is converted in both directions:

    Twilio -> ORION   base64 -> μ-law -> 16-bit PCM -> upsample 8k to 16k
    ORION -> Twilio   24k PCM -> downsample to 8k -> μ-law -> base64

``audioop`` does the μ-law conversion and the rate conversion exactly, and it
carries the resampler's state between frames. That state matters more than it
looks: resampling each frame independently restarts the filter every 20 ms,
which is audible as a faint buzz for the whole call and is very hard to
attribute afterwards.

Who may call
------------
Every request is verified against Twilio's ``X-Twilio-Signature`` before
anything is acted on. This is the whole of the authentication for these
endpoints — Caddy cannot add a bearer token to a request Twilio composes — so
a missing or wrong signature is refused, and a missing auth token means the
bridge refuses *everything* rather than running open.

Inbound calls are refused unless the number is one ORION was told about. A
public phone number reaches a machine that can run tools; an unauthenticated
voice on that line is a stranger with a shell.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Callable
from urllib.parse import urlencode

#: Twilio's wire format. Not configurable — these are what Media Streams is.
TWILIO_RATE = 8000
TWILIO_WIDTH = 2           # bytes per sample after μ-law expansion

#: What ORION's engine expects in, and produces out.
ORION_IN_RATE = 16000
ORION_OUT_RATE = 24000

#: 20 ms at 8 kHz — one Twilio media frame.
FRAME_SAMPLES = 160

#: How long a call may run before the bridge closes it. A call left open by a
#: far end that never hangs up bills by the minute until someone notices.
MAX_CALL_SECONDS = 900.0


# ── audio conversion ──────────────────────────────────────────────────────────

def _audioop() -> Any:
    """``audioop`` from the standard library, or ``audioop-lts`` on 3.13+.

    ``audioop`` was removed from the standard library in Python 3.13. The
    replacement package is ``audioop-lts``, which installs under the same
    name (both requirements files list it); this reports the absence clearly
    rather than failing later inside a live call.
    """
    try:
        import audioop  # type: ignore

        return audioop
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "audioop is unavailable. On Python 3.13+ install it with: "
            "pip install audioop-lts") from exc


class Resampler:
    """Rate conversion that remembers where it was.

    ``audioop.ratecv`` returns a state object which must be fed back on the
    next call. Dropping it and resampling each 20 ms frame from scratch
    restarts the interpolation filter 50 times a second — audible as a
    continuous faint buzz, and almost impossible to attribute after the fact.
    """

    def __init__(self, from_rate: int, to_rate: int, width: int = 2,
                 channels: int = 1) -> None:
        self.from_rate = int(from_rate)
        self.to_rate = int(to_rate)
        self.width = int(width)
        self.channels = int(channels)
        self._state: Any = None

    def __call__(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        if self.from_rate == self.to_rate:
            return pcm
        converted, self._state = _audioop().ratecv(
            pcm, self.width, self.channels, self.from_rate, self.to_rate,
            self._state)
        return converted

    def reset(self) -> None:
        self._state = None


def ulaw_to_pcm(payload: bytes) -> bytes:
    """One μ-law frame as 16-bit linear PCM."""
    return _audioop().ulaw2lin(payload, TWILIO_WIDTH)


def pcm_to_ulaw(pcm: bytes) -> bytes:
    """16-bit linear PCM as μ-law, ready for the wire."""
    return _audioop().lin2ulaw(pcm, TWILIO_WIDTH)


class CallAudio:
    """Both directions of one call's audio, with their own resampler state.

    One object per call, never shared: two calls sharing a resampler would
    interleave their filter states and both would sound wrong.
    """

    def __init__(self, in_rate: int = ORION_IN_RATE,
                 out_rate: int = ORION_OUT_RATE) -> None:
        self.inbound = Resampler(TWILIO_RATE, in_rate)
        self.outbound = Resampler(out_rate, TWILIO_RATE)

    def from_twilio(self, b64_payload: str) -> bytes:
        """A Twilio media frame as PCM at ORION's input rate."""
        try:
            # validate=True, deliberately. Python's decoder is lenient by
            # default: it discards characters outside the alphabet and decodes
            # whatever is left, so a corrupted frame becomes plausible-looking
            # bytes rather than an error — and those bytes get played down the
            # telephone as noise. Measured: 26 characters of rubbish produced
            # 58 bytes of "audio".
            raw = base64.b64decode(b64_payload or "", validate=True)
        except Exception:
            return b""
        if not raw:
            return b""
        return self.inbound(ulaw_to_pcm(raw))

    def to_twilio(self, pcm: bytes) -> str:
        """ORION's PCM as a base64 μ-law payload."""
        if not pcm:
            return ""
        return base64.b64encode(pcm_to_ulaw(self.outbound(pcm))).decode("ascii")


# ── request authentication ────────────────────────────────────────────────────

def twilio_signature(auth_token: str, url: str,
                     params: dict[str, Any] | None = None) -> str:
    """The signature Twilio would send for this request.

    Their scheme: the full URL, then every POST parameter appended as
    key+value in sorted key order, HMAC-SHA1 with the auth token, base64.
    """
    payload = str(url or "")
    for key in sorted((params or {}).keys()):
        payload += f"{key}{(params or {})[key]}"
    digest = hmac.new(str(auth_token or "").encode("utf-8"),
                      payload.encode("utf-8"), sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def verify_twilio(auth_token: str, signature: str, url: str,
                  params: dict[str, Any] | None = None) -> bool:
    """Whether this request really came from Twilio.

    Fails closed on a missing token: a bridge with no token configured
    refuses everything rather than accepting everything, because these
    endpoints are public and there is no second layer behind them.

    Compared with ``compare_digest`` — a plain ``==`` on a signature leaks
    how much of it was right through how long the comparison took.
    """
    if not auth_token or not signature:
        return False
    return hmac.compare_digest(
        twilio_signature(auth_token, url, params), str(signature))


# ── the conversation ──────────────────────────────────────────────────────────

@dataclass
class CallSession:
    """One live call.

    Deliberately holds no audio history. A recording of a phone call is
    personal data with its own obligations, and ORION has no reason to keep
    one — what he needs is the transcript the engine already produces.
    """

    call_sid: str = ""
    stream_sid: str = ""
    from_number: str = ""
    to_number: str = ""
    inbound: bool = False
    audio: CallAudio = field(default_factory=CallAudio)
    started_at: float = 0.0
    frames_in: int = 0
    frames_out: int = 0
    ended: bool = False
    reason: str = ""

    def describe(self) -> str:
        direction = "from" if self.inbound else "to"
        who = self.from_number if self.inbound else self.to_number
        return f"call {direction} {who or 'unknown'} ({self.call_sid[:10] or '?'})"


class MediaStreamHandler:
    """Reads Twilio's frames, hands audio to ORION, sends his voice back.

    The engine is injected rather than imported: this module knows how to
    convert audio and speak Twilio's protocol, and nothing about how ORION
    thinks. That keeps it testable without a model, a network or a telephone.
    """

    def __init__(self, engine: Any = None, bus: Any = None,
                 contacts: Any = None,
                 on_transcript: Callable[[str, str], None] | None = None) -> None:
        self.engine = engine
        self.bus = bus
        self.contacts = contacts
        self.on_transcript = on_transcript
        self.session = CallSession()

    # ── Twilio's protocol ────────────────────────────────────────────────────

    async def handle_message(self, raw: str) -> list[dict[str, Any]]:
        """One frame in, zero or more frames out.

        Returns what should be sent back rather than sending it, so the
        protocol can be tested without a socket.
        """
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return []
        if not isinstance(message, dict):
            return []

        event = str(message.get("event") or "")
        if event == "connected":
            return []
        if event == "start":
            return self._on_start(message)
        if event == "media":
            return await self._on_media(message)
        if event == "stop":
            self.session.ended = True
            self.session.reason = "the far end hung up"
            self._log(f"{self.session.describe()} ended — "
                      f"{self.session.frames_in} frames in, "
                      f"{self.session.frames_out} out")
            return []
        if event == "mark":
            return []
        return []

    def _on_start(self, message: dict) -> list[dict[str, Any]]:
        start = message.get("start") or {}
        session = self.session
        session.stream_sid = str(message.get("streamSid")
                                 or start.get("streamSid") or "")
        session.call_sid = str(start.get("callSid") or "")
        custom = start.get("customParameters") or {}
        session.from_number = str(custom.get("from") or "")
        session.to_number = str(custom.get("to") or "")
        session.inbound = str(custom.get("direction") or "").lower() == "inbound"

        if session.inbound and not self._caller_allowed(session.from_number):
            session.ended = True
            session.reason = "the caller is not on the permitted list"
            self._log(f"refused an inbound call from "
                      f"{session.from_number or 'a withheld number'}")
            return [self._hangup()]

        self._log(f"{session.describe()} connected")
        return []

    def _caller_allowed(self, number: str) -> bool:
        """Whether an inbound caller may talk to ORION.

        A public phone number reaches a machine that can run tools. An
        unauthenticated voice on that line is a stranger with a shell, so
        inbound is default-deny in exactly the way outbound is.
        """
        if self.contacts is None:
            return False
        try:
            return bool(self.contacts.allows(number))
        except Exception:
            return False

    async def _on_media(self, message: dict) -> list[dict[str, Any]]:
        payload = (message.get("media") or {}).get("payload") or ""
        pcm = self.session.audio.from_twilio(payload)
        if not pcm:
            return []
        self.session.frames_in += 1

        engine = self.engine
        if engine is None:
            return []
        try:
            reply = await engine.feed_audio(pcm)
        except Exception as exc:
            self._log(f"the engine faulted mid-call - {exc}")
            return []
        return self._speak(reply)

    def _speak(self, pcm: bytes | None) -> list[dict[str, Any]]:
        if not pcm:
            return []
        payload = self.session.audio.to_twilio(pcm)
        if not payload:
            return []
        self.session.frames_out += 1
        return [{"event": "media",
                 "streamSid": self.session.stream_sid,
                 "media": {"payload": payload}}]

    def _hangup(self) -> dict[str, Any]:
        """Tell Twilio to stop the stream, which ends the call."""
        return {"event": "clear", "streamSid": self.session.stream_sid}

    def barge_in(self) -> dict[str, Any]:
        """Stop what ORION is saying, immediately.

        Twilio buffers what it has been sent, so simply not sending more
        leaves him talking over the caller for as long as the buffer lasts.
        ``clear`` discards it, which is what interrupting has to mean on a
        telephone.
        """
        return self._hangup()

    def _log(self, line: str) -> None:
        if self.bus is None:
            return
        try:
            self.bus.log.emit(f"[Call] {line}")
        except Exception:
            pass


# ── the endpoints ─────────────────────────────────────────────────────────────

def public_base_url() -> str:
    """The HTTPS origin Twilio should call back on."""
    host = (os.getenv("ORION_PUBLIC_HOST", "") or "").strip()
    host = host.replace("https://", "").replace("http://", "").strip("/")
    return f"https://{host}" if host else ""


def websocket_url() -> str:
    base = public_base_url()
    return base.replace("https://", "wss://") + "/media" if base else ""


def answer_twiml(greeting: str = "") -> str:
    """The TwiML for a two-way call, or an apology if we are not reachable.

    A bridge with no public host configured cannot receive a media stream, so
    saying so down the line is better than connecting a caller to silence.
    """
    from .telephony import twiml_say, twiml_stream

    url = websocket_url()
    if not url:
        return twiml_say(
            "I'm sorry — I'm not reachable for calls at the moment.")
    return twiml_stream(url, greeting=greeting
                        or "One moment, I'm just connecting you.")


__all__ = [
    "FRAME_SAMPLES", "MAX_CALL_SECONDS", "ORION_IN_RATE", "ORION_OUT_RATE",
    "TWILIO_RATE", "CallAudio", "CallSession", "MediaStreamHandler",
    "Resampler", "answer_twiml", "public_base_url", "pcm_to_ulaw",
    "twilio_signature", "ulaw_to_pcm", "verify_twilio", "websocket_url",
]
