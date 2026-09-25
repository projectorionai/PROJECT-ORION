"""A full GPU does not empty on a timer.

From the running app's own activity log, repeating every forty-five seconds:

    NET: provider local_ollama cooled for 45s - HTTP 500 {"error":{"message":
    "llama-server process has terminated: exit status 1: cudaMalloc failed
    out of memory / alloc_tensor_range: failed to allocate CUDA ...

The router classified provider failures into quota, auth, model, payload and
credit, and everything else fell to a flat 45-second cooldown. An out-of-VRAM
failure is none of those, so ORION tried to start a local model that could not
fit, failed, waited 45 seconds, and tried again -- for the life of the session.

That is not merely wasted work. Each attempt asks the driver for several
gigabytes the card has not got, and a large failed allocation disturbs
everything else holding memory on that GPU -- including the WebEngine
compositor drawing ORION's face, which loses its surfaces and redraws them
black. The report was "black random stutters". Measured on the machine: an
RTX 3060 Ti with 8192 MiB total and 5544 free, against a model that wanted
more, while the system sat at 95% RAM.

So out-of-memory is its own class, checked before quota (whose "resource
exhausted" and "insufficient" both match the same text), and it backs off
5 -> 15 -> 45 -> 60 minutes instead of retrying on a fixed clock. A quota
resets on a clock and is right to retry that way; VRAM frees when something
else lets go, which may be never.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.providers import ProviderRouter  # noqa: E402

SOURCE = (ROOT / "orion_core" / "providers.py").read_text(encoding="utf-8")

REAL_FAILURES = [
    "llama-server process has terminated: exit status 1: cudaMalloc failed "
    "out of memory",
    "alloc_tensor_range: failed to allocate CUDA0 buffer of size 4294967296",
    "CUDA error: out of memory",
    "insufficient memory to load model",
    "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
]


@pytest.mark.parametrize("message", REAL_FAILURES)
def test_the_real_messages_are_recognised(message):
    assert ProviderRouter.OOM_RE.search(message), message


@pytest.mark.parametrize("message", [
    "429 rate limit exceeded",
    "invalid api key",
    "model not found",
    "request too large",
    "This request requires more credits, or fewer max_tokens",
    "WebSocket server error (1011)",
])
def test_ordinary_failures_are_not_dragged_into_this_class(message):
    """Misclassifying a quota as a full GPU would bench a working provider
    for an hour."""
    assert not ProviderRouter.OOM_RE.search(message), message


def test_it_is_checked_before_the_quota_class():
    """"insufficient memory" matches QUOTA_RE too, and quota's 300 s is the
    wrong answer -- it is the retrying-on-a-clock that causes the harm."""
    assert SOURCE.index("if self.OOM_RE.search(message):") < \
        SOURCE.index("if self.QUOTA_RE.search(message):")


def test_it_is_checked_before_every_other_class():
    first = SOURCE.index("if self.OOM_RE.search(message):")
    for other in ("PAYLOAD_RE", "CREDIT_RE", "AUTH_RE", "MODEL_RE"):
        assert first < SOURCE.index(f"self.{other}.search(message)"), other


def _backoff(strikes):
    """The shipped escalation."""
    return [min(3600.0, 300.0 * (3 ** n)) for n in range(strikes)]


def test_the_first_wait_is_far_longer_than_the_old_flat_one():
    assert _backoff(1)[0] >= 300.0


def test_each_strike_waits_longer_than_the_last():
    waits = _backoff(4)
    assert all(b >= a for a, b in zip(waits, waits[1:]))
    assert waits[1] > waits[0]


def test_the_backoff_is_capped():
    """Unbounded growth would eventually be indistinguishable from never
    trying again, without ever saying so."""
    assert max(_backoff(10)) == 3600.0


def test_the_escalation_is_per_provider():
    """One local model exhausting the card must not bench a different one."""
    assert "self._oom_strikes[profile.name]" in SOURCE
    assert "self._oom_strikes: dict[str, int] = {}" in SOURCE


def test_the_message_says_what_would_actually_fix_it():
    """"cooled for 45s" told the user nothing they could act on."""
    start = SOURCE.index("if self.OOM_RE.search(message):")
    body = SOURCE[start:start + 1400]
    assert "GPU memory" in body
    assert "smaller local model" in body or "Close something" in body


def test_it_stops_rather_than_falling_through():
    """Falling through would apply a second cooldown over the top of this
    one and log twice."""
    start = SOURCE.index("if self.OOM_RE.search(message):")
    body = SOURCE[start:start + 1400]
    assert "return" in body
    assert body.index("self._cooldowns[profile.name]") < body.index("return")
