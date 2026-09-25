"""ORION can email somebody, and says truthfully which of two things he did.

His only mail path was a COM bridge to the local Outlook client, so when
Outlook misbehaves he cannot email anybody at all. Two routes replace that, in
order of how much they actually accomplish: SMTP when an account is
configured, and otherwise a Gmail compose window with everything already typed.

The distinction lives in the wording, because "I have emailed them" and "I
have written it and it is waiting for you" are very different claims to make
to somebody who then walks away from the computer.

Offline: no mail is sent and no browser opens.
"""

from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import orion_core.messaging as messaging  # noqa: E402
from orion_core.messaging import MessagingGateway  # noqa: E402


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, line: str) -> None:
        self.lines.append(line)


class _Bus:
    def __init__(self) -> None:
        self.log = _Log()


@pytest.fixture
def gateway(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(messaging, "open_url", lambda url: opened.append(url))
    gate = MessagingGateway(_Bus())
    monkeypatch.setattr(gate, "delivery_suppressed", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_email_account", staticmethod(dict))
    gate._opened = opened            # type: ignore[attr-defined]
    return gate


def test_an_unconfigured_send_opens_a_prefilled_draft(gateway):
    result = gateway.send_email("friend@example.com", "Lunch?", "Free at 1?")
    assert result.ok
    assert gateway._opened, "nothing was opened"
    query = urllib.parse.parse_qs(urllib.parse.urlparse(gateway._opened[0]).query)
    assert query["to"] == ["friend@example.com"]
    assert query["su"] == ["Lunch?"]
    assert query["body"] == ["Free at 1?"]
    assert query["view"] == ["cm"], "not a compose window"


def test_a_draft_is_never_described_as_sent(gateway):
    """Somebody who is told it was sent will walk away from the computer."""
    text = gateway.send_email("a@b.com", "Subject", "Body").text.lower()
    assert "waiting for you" in text or "press send" in text
    assert "email sent" not in text


def test_a_multiline_body_survives_the_url(gateway):
    gateway.send_email("a@b.com", "s", "line one\nline two")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(gateway._opened[0]).query)
    assert query["body"] == ["line one\nline two"]


def test_a_configured_account_sends_for_real(monkeypatch):
    sent: list = []

    class _SMTP:
        def __init__(self, host, port, timeout=0):
            sent.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def starttls(self):
            sent.append(("starttls",))

        def login(self, address, password):
            sent.append(("login", address, password))

        def send_message(self, message):
            sent.append(("send", message["To"], message["Subject"]))

    monkeypatch.setattr("smtplib.SMTP", _SMTP)
    monkeypatch.setattr(messaging, "open_url",
                        lambda _u: pytest.fail("a browser opened despite an account"))
    gate = MessagingGateway(_Bus())
    monkeypatch.setattr(gate, "delivery_suppressed", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_email_account", staticmethod(
        lambda: {"address": "me@gmail.com", "password": "app-password"}))

    result = gate.send_email("them@example.com", "Hello", "Body")
    assert result.ok and "sent" in result.text.lower()
    assert ("login", "me@gmail.com", "app-password") in sent
    assert ("send", "them@example.com", "Hello") in sent
    assert ("starttls",) in sent, "credentials went over an unencrypted link"


def test_a_delivery_failure_is_not_hidden_behind_a_draft(monkeypatch):
    """Falling back to a browser after a real failure would look like
    progress, and the user would believe the mail had gone."""
    def _boom(*_a, **_k):
        raise OSError("mailbox unavailable")

    monkeypatch.setattr("smtplib.SMTP", _boom)
    monkeypatch.setattr(messaging, "open_url",
                        lambda _u: pytest.fail("a draft opened after a real failure"))
    gate = MessagingGateway(_Bus())
    monkeypatch.setattr(gate, "delivery_suppressed", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_email_account", staticmethod(
        lambda: {"address": "me@gmail.com", "password": "pw"}))

    result = gate.send_email("them@example.com", "Hello", "Body")
    assert result.ok is False
    assert "mailbox unavailable" in result.text


@pytest.mark.parametrize("to,subject,body", [("", "s", "b"), ("  ", "s", "b")])
def test_an_email_needs_a_recipient(gateway, to, subject, body):
    assert gateway.send_email(to, subject, body).ok is False


def test_the_suite_can_never_send_or_open_anything():
    """PYTEST_CURRENT_TEST alone must suppress delivery, so a test that
    forgets to patch cannot email a real person."""
    gate = MessagingGateway(_Bus())
    assert gate.delivery_suppressed() is True
    result = gate.send_email("real@person.com", "oops", "body")
    assert result.ok and "dry-run" in result.text


def test_the_tool_routes_email(monkeypatch):
    from orion_core.dispatch_web import WebDispatchMixin

    class _D(WebDispatchMixin):
        def __init__(self):
            self.bus = _Bus()
            self.messaging = MessagingGateway(self.bus)

    dispatcher = _D()
    for args in ({"action": "email", "to": "a@b.com", "subject": "s", "body": "b"},
                 {"action": "send", "platform": "gmail", "to": "a@b.com", "message": "b"}):
        result = dispatcher.messaging_tool(args)
        assert result.ok and "a@b.com" in result.text


def test_the_model_is_told_not_to_overclaim():
    schema = (ROOT / "orion_core" / "dispatch_schema.py").read_text(encoding="utf-8")
    assert "do not claim the mail was sent" in schema
