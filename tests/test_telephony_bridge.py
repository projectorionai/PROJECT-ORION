"""
The bridge between a phone call and ORION's conversational engine.

Two things here are easy to get wrong in ways that are very hard to diagnose
afterwards, so both are tested directly.

The resampler carries state between frames. Dropped, the interpolation filter
restarts every 20 ms — fifty times a second — which is audible as a continuous
faint buzz for the whole call and is almost impossible to attribute to its
cause once the call has ended.

The signature check is the *entire* authentication for these endpoints. Caddy
cannot add a bearer token to a request Twilio composes, so if this is wrong,
a public URL reaches a machine that can run tools. It fails closed on a
missing token, and it compares in constant time.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.telephony_bridge import (  # noqa: E402
    ORION_IN_RATE,
    ORION_OUT_RATE,
    TWILIO_RATE,
    CallAudio,
    MediaStreamHandler,
    Resampler,
    answer_twiml,
    pcm_to_ulaw,
    public_base_url,
    twilio_signature,
    ulaw_to_pcm,
    verify_twilio,
    websocket_url,
)


def _tone(samples: int = 160, hz: int = 440, rate: int = TWILIO_RATE) -> bytes:
    """A pure tone as 16-bit PCM — something with structure to compare."""
    return b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * hz * n / rate)))
        for n in range(samples))


def _samples(pcm: bytes) -> list[int]:
    return [struct.unpack("<h", pcm[i:i + 2])[0] for i in range(0, len(pcm), 2)]


def _run(coro):
    return asyncio.run(coro)


# ── audio conversion ──────────────────────────────────────────────────────────

def test_a_frame_survives_the_round_trip():
    """μ-law is lossy by design — roughly eight bits of resolution — so this
    checks the shape came back, not that the bytes did."""
    original = _tone()
    recovered = ulaw_to_pcm(pcm_to_ulaw(original))
    assert len(recovered) == len(original)
    worst = max(abs(a - b) for a, b in
                zip(_samples(original), _samples(recovered)))
    assert worst < 600, f"μ-law lost too much: worst sample error {worst}"


def test_mu_law_halves_the_size():
    """160 samples of 16-bit PCM is 320 bytes, and one Twilio frame is 160."""
    assert len(pcm_to_ulaw(_tone())) == 160


def test_the_resampler_remembers_where_it_was():
    """The bug this prevents is a continuous faint buzz for a whole call."""
    resampler = Resampler(TWILIO_RATE, ORION_IN_RATE)
    resampler(_tone())
    assert resampler._state is not None, "state was dropped between frames"


def test_a_reset_resampler_starts_again():
    resampler = Resampler(TWILIO_RATE, ORION_IN_RATE)
    resampler(_tone())
    resampler.reset()
    assert resampler._state is None


def test_upsampling_doubles_the_sample_count():
    """8 kHz in, 16 kHz out. The first frame is one sample short while the
    filter fills, which is correct and not worth hiding."""
    audio = CallAudio()
    first = audio.from_twilio(base64.b64encode(pcm_to_ulaw(_tone())).decode())
    second = audio.from_twilio(base64.b64encode(pcm_to_ulaw(_tone())).decode())
    assert len(second) // 2 == 320
    assert abs(len(first) // 2 - 320) <= 1


def test_downsampling_thirds_the_sample_count():
    """24 kHz out of ORION, 8 kHz onto the wire."""
    audio = CallAudio()
    payload = audio.to_twilio(b"\x00\x10" * 480)
    assert len(base64.b64decode(payload)) == 160


def test_an_empty_or_broken_payload_is_survivable():
    audio = CallAudio()
    assert audio.from_twilio("") == b""
    assert audio.from_twilio("this is not base64 at all!!") == b""
    assert audio.to_twilio(b"") == ""


def test_a_rate_that_does_not_change_is_passed_through():
    resampler = Resampler(TWILIO_RATE, TWILIO_RATE)
    tone = _tone()
    assert resampler(tone) == tone


# ── authentication ────────────────────────────────────────────────────────────

URL = "https://orion.example.com/twilio/voice"
PARAMS = {"CallSid": "CA123", "From": "+447700900123"}
TOKEN = "test-auth-token"


def test_a_genuine_twilio_request_verifies():
    signature = twilio_signature(TOKEN, URL, PARAMS)
    assert verify_twilio(TOKEN, signature, URL, PARAMS) is True


def test_a_wrong_signature_is_refused():
    assert verify_twilio(TOKEN, "not-the-signature", URL, PARAMS) is False


def test_tampering_with_a_parameter_invalidates_it():
    """The signature covers the parameters, not just the URL — otherwise the
    caller's number could be swapped after signing."""
    signature = twilio_signature(TOKEN, URL, PARAMS)
    tampered = {**PARAMS, "From": "+15550000000"}
    assert verify_twilio(TOKEN, signature, URL, tampered) is False


def test_tampering_with_the_url_invalidates_it():
    signature = twilio_signature(TOKEN, URL, PARAMS)
    assert verify_twilio(TOKEN, signature, URL + "/elsewhere", PARAMS) is False


def test_no_auth_token_refuses_everything():
    """The load-bearing refusal.

    These endpoints are public and there is nothing behind them. A bridge with
    no token configured must refuse everything rather than accept everything.
    """
    signature = twilio_signature(TOKEN, URL, PARAMS)
    assert verify_twilio("", signature, URL, PARAMS) is False
    assert verify_twilio(None, signature, URL, PARAMS) is False


def test_an_empty_signature_is_refused():
    assert verify_twilio(TOKEN, "", URL, PARAMS) is False


def test_parameter_order_does_not_matter():
    """Twilio sorts by key; a dict in another order is the same request."""
    forward = twilio_signature(TOKEN, URL, {"a": "1", "b": "2"})
    backward = twilio_signature(TOKEN, URL, {"b": "2", "a": "1"})
    assert forward == backward


# ── the stream protocol ───────────────────────────────────────────────────────

class _Engine:
    def __init__(self, reply: bytes | None = b"\x00\x10" * 480) -> None:
        self.reply = reply
        self.fed = 0

    async def feed_audio(self, pcm: bytes):
        self.fed += 1
        return self.reply


class _Contacts:
    def __init__(self, allowed=("+447700900123",)) -> None:
        self._allowed = set(allowed)

    def allows(self, number):
        return number in self._allowed


class _Signal:
    def __init__(self):
        self.lines = []

    def emit(self, line):
        self.lines.append(line)


class _Bus:
    def __init__(self):
        self.log = _Signal()


def _start(number="+447700900123", direction="inbound") -> str:
    return json.dumps({
        "event": "start", "streamSid": "MZ1",
        "start": {"callSid": "CA1",
                  "customParameters": {"from": number, "direction": direction}},
    })


def _media() -> str:
    return json.dumps({"event": "media", "media": {
        "payload": base64.b64encode(pcm_to_ulaw(_tone())).decode()}})


_UNSET = object()


def _handler(contacts=_UNSET, engine=None) -> MediaStreamHandler:
    """A handler with a contact book by default, or none if asked explicitly.

    The sentinel matters: `contacts=None` is a test case in its own right —
    "there is no contact book" — and must not quietly get the default one.
    """
    return MediaStreamHandler(engine or _Engine(), _Bus(),
                              _Contacts() if contacts is _UNSET else contacts)


def test_a_known_caller_is_connected():
    handler = _handler()
    assert _run(handler.handle_message(_start())) == []
    assert handler.session.ended is False
    assert handler.session.call_sid == "CA1"


def test_an_unknown_caller_is_hung_up_on():
    """A public phone number reaches a machine that can run tools. An
    unauthenticated voice on that line is a stranger with a shell."""
    handler = _handler()
    replies = _run(handler.handle_message(_start("+447700900999")))
    assert handler.session.ended is True
    assert replies and replies[0]["event"] == "clear"


def test_no_contact_book_means_nobody_gets_through():
    handler = _handler(contacts=None)
    _run(handler.handle_message(_start()))
    assert handler.session.ended is True


def test_audio_is_fed_to_the_engine_and_the_reply_is_returned():
    engine = _Engine()
    handler = _handler(engine=engine)
    _run(handler.handle_message(_start()))
    replies = _run(handler.handle_message(_media()))
    assert engine.fed == 1
    assert replies[0]["event"] == "media"
    assert replies[0]["streamSid"] == "MZ1"
    assert replies[0]["media"]["payload"]


def test_an_engine_with_nothing_to_say_sends_nothing():
    handler = _handler(engine=_Engine(reply=None))
    _run(handler.handle_message(_start()))
    assert _run(handler.handle_message(_media())) == []


def test_an_engine_that_faults_does_not_drop_the_call():
    class _Broken:
        async def feed_audio(self, pcm):
            raise RuntimeError("the model went away")

    handler = _handler(engine=_Broken())
    _run(handler.handle_message(_start()))
    assert _run(handler.handle_message(_media())) == []
    assert handler.session.ended is False, "the call should still be up"


def test_a_stop_event_ends_the_session():
    handler = _handler()
    _run(handler.handle_message(_start()))
    _run(handler.handle_message('{"event": "stop"}'))
    assert handler.session.ended is True


def test_a_malformed_frame_is_ignored():
    handler = _handler()
    assert _run(handler.handle_message("{not json")) == []
    assert _run(handler.handle_message("[]")) == []
    assert _run(handler.handle_message("")) == []


def test_interrupting_clears_what_is_buffered():
    """Twilio buffers what it has been sent, so simply not sending more leaves
    ORION talking over the caller for as long as the buffer lasts."""
    handler = _handler()
    _run(handler.handle_message(_start()))
    assert handler.barge_in()["event"] == "clear"


def test_a_call_holds_no_recording():
    """A recording of a phone call is personal data with its own obligations,
    and ORION has no reason to keep one."""
    handler = _handler()
    _run(handler.handle_message(_start()))
    _run(handler.handle_message(_media()))
    stored = [value for value in vars(handler.session).values()
              if isinstance(value, (bytes, bytearray)) and len(value) > 64]
    assert stored == [], "the session is holding audio"


# ── the callback URLs ─────────────────────────────────────────────────────────

def test_the_callback_urls_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ORION_PUBLIC_HOST", "orion.example.com")
    assert public_base_url() == "https://orion.example.com"
    assert websocket_url() == "wss://orion.example.com/media"


def test_a_scheme_in_the_host_is_tolerated(monkeypatch):
    monkeypatch.setenv("ORION_PUBLIC_HOST", "https://orion.example.com/")
    assert public_base_url() == "https://orion.example.com"


def test_without_a_public_host_a_caller_is_told_rather_than_connected(monkeypatch):
    """Connecting someone to a media stream that cannot exist gives them
    silence; saying so gives them something to act on."""
    monkeypatch.delenv("ORION_PUBLIC_HOST", raising=False)
    xml = answer_twiml()
    assert "<Stream" not in xml
    assert "not reachable" in xml


def test_with_a_public_host_the_call_is_bridged(monkeypatch):
    monkeypatch.setenv("ORION_PUBLIC_HOST", "orion.example.com")
    xml = answer_twiml(greeting="Hello.")
    assert "wss://orion.example.com/media" in xml
    assert "Hello." in xml
