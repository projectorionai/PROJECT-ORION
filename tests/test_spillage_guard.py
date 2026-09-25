"""
The spillage guard (CAP-08) — catch a credential before it leaves.

    "private data spillage ... must be voiced to me directly, overriding
     standby."

Test secrets are synthetic — deliberately shaped like the real thing, but not
real keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.spillage_guard import SpillageGuard, Severity  # noqa: E402


guard = SpillageGuard()
# Assemble synthetic credentials so repository scanners need no test exemption.
FAKE_API_KEY = "sk-" + "abcdef012345678901234567890"
FAKE_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"


# ── detection ────────────────────────────────────────────────────────────────

def test_openai_style_key_is_critical():
    r = guard.scan(f"here it is {FAKE_API_KEY} ok")
    assert not r.safe
    assert any("OpenAI" in f.kind for f in r.findings)
    assert r.worst is Severity.CRITICAL


def test_aws_access_key_is_caught():
    r = guard.scan(f"{FAKE_AWS_KEY} is the id")
    assert any("AWS" in f.kind for f in r.findings)


def test_private_key_block_is_critical():
    text = ("-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n"
            "-----END RSA PRIVATE KEY-----")
    r = guard.scan(text)
    assert r.worst is Severity.CRITICAL
    assert any("private key" in f.kind for f in r.findings)


def test_password_assignment_is_high():
    r = guard.scan("db config: password = hunter2secret")
    assert not r.safe
    assert any("credential assignment" in f.kind for f in r.findings)


def test_luhn_valid_card_is_flagged():
    r = guard.scan("card 4242 4242 4242 4242 expires soon")
    assert any("card" in f.kind for f in r.findings)


def test_a_random_16_digit_non_luhn_is_not_a_card():
    r = guard.scan("order number 1234 5678 9012 3457 shipped")   # fails Luhn
    assert not any("card" in f.kind for f in r.findings)


def test_plain_text_is_safe():
    r = guard.scan("Let's meet at three to talk about the Mars report.")
    assert r.safe
    assert r.findings == []


def test_a_uuid_is_not_flagged():
    r = guard.scan("run id 550e8400-e29b-41d4-a716-446655440000 finished")
    assert r.safe


def test_a_git_hash_is_not_flagged():
    r = guard.scan("commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0 landed")
    assert r.safe


def test_high_entropy_token_is_medium_but_not_blocking():
    # random-looking, not hex, not a known prefix
    tok = "Zk9Q2mVx7Lp4Rt8Wc1Nb6Yd3Fg5Hj0Ss"
    r = guard.scan(f"the value is {tok} apparently")
    assert any(f.severity is Severity.MEDIUM for f in r.findings)
    assert r.safe          # MEDIUM alone doesn't block sending


# ── known secrets ────────────────────────────────────────────────────────────

def test_orions_own_secret_echoed_back_is_critical():
    g = SpillageGuard(known_secrets=["my-super-secret-value-123"])
    r = g.scan("the api key configured is my-super-secret-value-123 fyi")
    assert r.worst is Severity.CRITICAL
    assert any("own credential" in f.kind for f in r.findings)


def test_short_known_values_are_ignored():
    g = SpillageGuard(known_secrets=["abc"])       # too short to be meaningful
    assert g.scan("abc def ghi").safe


# ── redaction ────────────────────────────────────────────────────────────────

def test_redaction_removes_the_secret():
    text = f"key {FAKE_API_KEY} done"
    red = guard.redact(text)
    assert FAKE_API_KEY not in red
    assert "[REDACTED:" in red


def test_preview_never_contains_the_raw_secret():
    raw = FAKE_API_KEY
    r = guard.scan(f"x {raw} y")
    for f in r.findings:
        assert raw not in f.preview


# ── the alarm ────────────────────────────────────────────────────────────────

class _Signal:
    def __init__(self):
        self.messages = []
    def emit(self, msg):
        self.messages.append(msg)


class _Bus:
    def __init__(self):
        self.safety_alert = _Signal()
        self.log = _Signal()


def test_guard_raises_on_the_safety_channel_for_critical():
    bus = _Bus()
    guard.guard(f"leaking {FAKE_API_KEY} now", bus=bus)
    assert bus.safety_alert.messages
    assert "credential" in bus.safety_alert.messages[0].lower()


def test_guard_stays_quiet_for_clean_text():
    bus = _Bus()
    guard.guard("nothing sensitive here at all", bus=bus)
    assert not bus.safety_alert.messages


def test_guard_does_not_need_a_bus():
    # must not raise when no bus is provided
    r = guard.guard("password = topsecret123")
    assert not r.safe


# ── the tool wiring ──────────────────────────────────────────────────────────

def test_tool_and_schema_present():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"privacy_guard"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "privacy_guard")
    assert "action" in tool["parameters"]["properties"]
    assert "text" in tool["parameters"]["properties"]
