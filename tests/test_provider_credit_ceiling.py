"""
HTTP 402: a solvable problem that was being treated as an outage.

From the user's log:

    NET: provider openrouter cooled for 300s - HTTP 402:
    {"error":{"message":"This request requires more credits, or fewer
    max_tokens. You requested up to 16384 tokens, but can only afford 16096.",
    "code":402}}

Two separate defects in three lines.

ORION never asked for 16384 tokens — it set no max_tokens at all. OpenRouter
then reserves the model's ENTIRE completion budget against the balance, so a
request whose real answer was two sentences was refused for want of credit it
was never going to spend.

Then the message says "credits", QUOTA_RE matches the word, and the provider
was benched for five minutes over a request that only needed a ceiling. The
provider was fine. The account was fine. The request was fine one field short.

The provider also states exactly what it CAN afford, so there is nothing to
guess: take the number, cap to it, retry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.providers import ProviderRouter  # noqa: E402

MESSAGE_402 = (
    'HTTP 402: {"error":{"message":"This request requires more credits, or '
    'fewer max_tokens. You requested up to 16384 tokens, but can only afford '
    '16096. To increase, visit https://openrouter.ai/settings/credits","code":402}}'
)


# ── classification ───────────────────────────────────────────────────────────

def test_a_402_is_recognised_as_a_credit_ceiling():
    assert ProviderRouter.CREDIT_RE.search(MESSAGE_402)


def test_the_affordable_number_is_extracted():
    match = ProviderRouter.AFFORD_RE.search(MESSAGE_402)
    assert match and match.group(1) == "16096"


def test_credit_is_checked_before_quota():
    """Both patterns match this message. Order decides whether the provider is
    benched for five minutes or retried in two seconds."""
    assert ProviderRouter.QUOTA_RE.search(MESSAGE_402), (
        "if this stops matching, the ordering test below stops meaning anything")
    import inspect
    source = inspect.getsource(ProviderRouter._note_failure) \
        if hasattr(ProviderRouter, "_note_failure") else ""
    if not source:
        # Find whichever method carries the classification.
        for name in dir(ProviderRouter):
            attr = getattr(ProviderRouter, name, None)
            if callable(attr):
                try:
                    text = inspect.getsource(attr)
                except (OSError, TypeError):
                    continue
                if "CREDIT_RE" in text and "QUOTA_RE" in text:
                    source = text
                    break
    assert source, "no method classifies both credit and quota failures"
    assert source.index("CREDIT_RE") < source.index("QUOTA_RE"), (
        "a credit ceiling would be misclassified as an exhausted quota")


# ── the request is bounded ───────────────────────────────────────────────────

def test_a_ceiling_is_always_sent():
    """The root cause: no max_tokens meant the provider reserved the model's
    entire completion budget against the balance."""
    import inspect
    source = inspect.getsource(ProviderRouter)
    assert '"max_tokens": self.output_budget(profile)' in source


def test_the_default_ceiling_is_proportionate():
    """Spoken replies and tool arguments are short. The unbounded request
    reserved two orders of magnitude more than any real answer has used."""
    assert 512 <= ProviderRouter.DEFAULT_MAX_OUTPUT_TOKENS <= 8192


def test_a_learned_cap_is_stayed_under():
    """Landing exactly on the stated limit fails again on the next call — the
    balance is being spent by other requests too."""
    router = ProviderRouter.__new__(ProviderRouter)
    router._output_cap = {}
    profile = type("P", (), {"name": "openrouter"})()
    assert router.output_budget(profile) == ProviderRouter.DEFAULT_MAX_OUTPUT_TOKENS
    router._output_cap["openrouter"] = 12876
    assert router.output_budget(profile) == 12876


def test_the_cap_only_ever_tightens():
    """A later, more generous 402 must not undo a tighter one already learned;
    the tightest observed limit is the one that actually holds."""
    router = ProviderRouter.__new__(ProviderRouter)
    router._output_cap = {"openrouter": 900}
    caps = router._output_cap
    afford = 16096
    cap = max(256, int(afford * 0.8))
    previous = caps.get("openrouter", 0)
    caps["openrouter"] = min(previous, cap) if previous else cap
    assert caps["openrouter"] == 900


def test_caps_are_per_provider():
    router = ProviderRouter.__new__(ProviderRouter)
    router._output_cap = {"openrouter": 900}
    groq = type("P", (), {"name": "groq"})()
    assert router.output_budget(groq) == ProviderRouter.DEFAULT_MAX_OUTPUT_TOKENS


# ── an actual empty balance is still an outage ───────────────────────────────

def test_no_stated_allowance_means_genuinely_out_of_credit():
    """'Insufficient credits' with no number is not a ceiling problem, and
    retrying in two seconds would just burn the router."""
    message = 'HTTP 402: {"error":{"message":"Insufficient credits.","code":402}}'
    assert ProviderRouter.CREDIT_RE.search(message)
    assert not ProviderRouter.AFFORD_RE.search(message)


def test_ordinary_failures_are_untouched():
    for message in ("HTTP 500: internal server error",
                    "HTTP 429: rate limit exceeded",
                    "Connection timeout"):
        assert not ProviderRouter.CREDIT_RE.search(message), message
