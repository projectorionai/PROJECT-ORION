"""
Ensures the messaging gateway never opens a real WhatsApp/Telegram browser
window during tests or self-test suites, while genuine user sends still deliver.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.messaging as m


class _Sig:
    def emit(self, *a):
        pass


class _Bus:
    def __getattr__(self, n):
        s = _Sig()
        object.__setattr__(self, n, s)
        return s


def test_dry_run_under_pytest_does_not_open_browser(monkeypatch):
    # Patches open_url, the seam messaging actually uses. Patching
    # webbrowser.open here would intercept nothing and this safety check would
    # pass vacuously — while a real send during the suite opened a real window.
    opened = []
    monkeypatch.setattr(m, "open_url", lambda url: opened.append(url) or True)
    # PYTEST_CURRENT_TEST is set automatically by pytest → suppressed.
    gw = m.MessagingGateway(_Bus())
    res = gw.send_text("whatsapp", "441234567890", "system alert fired")
    assert res.ok
    assert "dry-run" in res.text.lower()
    assert opened == [], "no browser window may open during a test"


def test_unsuppressed_background_send_never_opens_default_browser(monkeypatch):
    """The seam is ``open_url``, not ``webbrowser.open``.

    Every link ORION opens now goes through orion_core.browser so that one
    place decides which browser is used — the user's own default, which on this
    machine is Edge. Patching webbrowser.open here would no longer intercept
    anything, and the test would pass a real URL to a real browser during the
    suite.
    """
    opened = []
    monkeypatch.setattr(m, "open_url", lambda url: opened.append(url) or True)
    monkeypatch.setattr(m.MessagingGateway, "delivery_suppressed", staticmethod(lambda: False))
    gw = m.MessagingGateway(_Bus())
    monkeypatch.setattr(gw, "_bot_token", lambda *_: "")
    res = gw.send_text("telegram", "example_contact", "hello there friend")
    assert not res.ok and res.receipt["status"] == "requires_foreground"
    assert opened == []


def test_explicit_foreground_draft_preserves_exact_encoded_body(monkeypatch):
    from orion_core.execution_policy import foreground_scope
    from urllib.parse import parse_qs, urlparse
    opened = []
    monkeypatch.setattr(m, "open_url", lambda url: opened.append(url) or True)
    gate = m.MessagingGateway(_Bus())
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    monkeypatch.setattr(gate, "_bot_token", lambda *_: "")
    body = "  exact message\nUnicode 🙂  "
    with foreground_scope("compose-test:user-authorized"):
        result = gate.send_text("telegram", "example_contact", body)
    assert result.ok and result.receipt["status"] == "draft" and result.receipt["sent"] is False
    assert len(opened) == 1 and parse_qs(urlparse(opened[0]).query)["text"] == [body]


def test_explicit_dryrun_env_forces_suppression(monkeypatch):
    monkeypatch.setenv("ORION_MESSAGING_DRYRUN", "1")
    assert m.MessagingGateway.delivery_suppressed() is True


def test_unsupported_platform_is_rejected():
    gw = m.MessagingGateway(_Bus())
    res = gw.send_text("signal", "x", "y")
    assert not res.ok
    assert "unsupported" in res.text.lower()
