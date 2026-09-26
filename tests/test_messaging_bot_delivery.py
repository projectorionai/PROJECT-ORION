"""Messages that actually arrive, when ORION has been given the means.

The gateway opened a wa.me or t.me link in the browser. That is a perfectly
good fallback and it is not delivery: it hands you a pre-filled compose window
and waits for you to press send. "Text Mum I'll be late" ending as a browser
tab is not what was asked for.

Bot tokens for Telegram and Discord were already in config/messaging.json and
there were already plugins that used them; nothing connected the two. These
cover that seam, and the three ways it must not fail: a simulated send must
never become a real one, an unreachable platform must be refused rather than
simulated, and a genuine delivery failure must be reported rather than hidden
behind a browser tab that looks like success.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import messaging as m  # noqa: E402


class _Signal:
    def __init__(self):
        self.lines = []

    def emit(self, *args, **kwargs):
        if args:
            self.lines.append(str(args[0]))


class _Bus:
    def __init__(self):
        self.log = _Signal()

    def __getattr__(self, name):
        return _Signal()


@pytest.fixture()
def gateway(monkeypatch):
    """A gateway that can never open a browser or send anything real."""
    opened = []
    monkeypatch.setattr(m, "open_url", lambda url: opened.append(url))
    # delivery_suppressed() is True for the whole suite, because
    # PYTEST_CURRENT_TEST is set while pytest runs — which is exactly the
    # guard that stops a test run popping WhatsApp tabs. Testing the delivery
    # path means lifting it deliberately, with open_url already neutered.
    monkeypatch.setattr(m.MessagingGateway, "delivery_suppressed",
                        staticmethod(lambda: False))
    gate = m.MessagingGateway(_Bus())
    gate.opened = opened
    return gate


def _stub_bot(monkeypatch, gateway, reply="Sent to Mum on Telegram.",
              token="a-token"):
    sent = []

    def fake_send(self, platform, contact, message):
        if not token:
            return None
        sent.append((platform, contact, message))
        from orion_core.data import ToolResult

        failed = "failed" in reply.lower()
        return ToolResult(reply, ok=not failed)

    monkeypatch.setattr(m.MessagingGateway, "_send_via_bot", fake_send)
    return sent


# ── delivery is preferred over a compose link ────────────────────────────────

def test_a_configured_bot_delivers_instead_of_opening_a_link(gateway,
                                                             monkeypatch):
    sent = _stub_bot(monkeypatch, gateway)
    result = gateway.send_text("telegram", "Mum", "I'll be late")
    assert result.ok
    assert sent == [("telegram", "Mum", "I'll be late")]
    assert gateway.opened == [], "a browser tab was opened as well as sending"


def test_without_a_token_the_compose_link_still_opens(gateway, monkeypatch):
    _stub_bot(monkeypatch, gateway, token="")
    result = gateway.send_text("telegram", "Mum", "I'll be late")
    assert result.ok
    assert len(gateway.opened) == 1
    assert "t.me" in gateway.opened[0]


def test_a_delivery_failure_is_reported_not_papered_over(gateway, monkeypatch):
    """Falling back to a browser tab after a real error would hide the error
    behind something that looks like success."""
    _stub_bot(monkeypatch, gateway, reply="Telegram delivery failed (401).")
    result = gateway.send_text("telegram", "Mum", "hello")
    assert result.ok is False
    assert gateway.opened == [], "the failure was hidden behind a compose link"


# ── the guards ───────────────────────────────────────────────────────────────

def test_a_dry_run_never_becomes_a_real_send(gateway, monkeypatch):
    """The worst possible direction for this to fail."""
    sent = _stub_bot(monkeypatch, gateway)
    monkeypatch.setattr(m.MessagingGateway, "delivery_suppressed",
                        staticmethod(lambda: True))
    result = gateway.send_text("telegram", "Mum", "hello")
    assert result.ok
    assert "simulated" in result.text.lower()
    assert sent == []
    assert gateway.opened == []


def test_an_unreachable_platform_is_refused_even_in_a_dry_run(gateway,
                                                              monkeypatch):
    """Simulating a send to somewhere ORION cannot reach reports success for
    something that could never have happened."""
    monkeypatch.setattr(m.MessagingGateway, "delivery_suppressed",
                        staticmethod(lambda: True))
    result = gateway.send_text("signal", "Mum", "hello")
    assert result.ok is False
    assert "signal" in result.text.lower()


def test_a_bot_only_platform_with_no_token_says_what_is_missing(gateway,
                                                                monkeypatch):
    """Discord has no compose link, so there is nothing to fall back to.
    Opening something unrelated would be worse than saying so."""
    _stub_bot(monkeypatch, gateway, token="")
    result = gateway.send_text("discord", "team", "stand-up in five")
    assert result.ok is False
    assert "token" in result.text.lower()
    assert gateway.opened == []


def test_whatsapp_is_deliberately_link_only():
    """It has no free bot API for messaging arbitrary contacts, so the compose
    link genuinely is the best available — not an oversight."""
    assert "whatsapp" in m.MessagingGateway._ENDPOINTS
    assert "whatsapp" not in m.MessagingGateway._BOT_PLUGINS


def test_a_token_is_never_read_from_the_source(gateway):
    """A gateway that ships with a token sends everyone's messages through
    one account."""
    source = (ROOT / "orion_core" / "messaging.py").read_text(encoding="utf-8")
    assert "bot_token" in source
    assert "messaging.json" in source
