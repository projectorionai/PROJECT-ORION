"""
One slow model can no longer spend a request's whole budget.

26 Oct 2026, the owner's log: a specialist agent's question went to
gemini_reasoning, whose own timeout equals the 90 s interactive budget. It
was still answering when the budget ran out, and the request failed - "all
text providers failed: all text providers failed: gemini_reasoning was still
answering when the time budget ran out" - with no other provider asked. Now,
while another provider waits behind it, an attempt may use at most the budget
less a reserve (FALLBACK_RESERVE_S, never more than a third of what is
left), and an attempt cut short for that reason is an ordinary failure, so
the next provider answers.

Real loopback HTTP, no accounts.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from orion_core.providers import ProviderDeadlineError, _CallScope
from test_local_provider_memory import _cloud, _router


def _app(hits):
    async def slow(_request):
        hits.append("slow")
        await asyncio.sleep(30)
        return web.json_response({"choices": [{"message": {"content": "too late"},
                                               "finish_reason": "stop"}]})

    async def fast(_request):
        hits.append("fast")
        return web.json_response({"choices": [{"message": {"content": "Here is the answer."},
                                               "finish_reason": "stop"}]})

    app = web.Application()
    app.router.add_post("/slow/v1/chat/completions", slow)
    app.router.add_post("/fast/v1/chat/completions", fast)
    return app


async def test_a_slow_first_model_leaves_time_for_the_next_one():
    hits = []
    async with TestServer(_app(hits)) as server:
        router = _router(_cloud("reasoning", str(server.make_url("/slow/v1")), timeout_s=90.0),
                         _cloud("flash", str(server.make_url("/fast/v1"))))
        router.MIN_ATTEMPT_S = 0.2
        try:
            started = time.monotonic()
            profile, answer = await router.generate_text("Explain engineering functions.",
                                                         deadline_s=3.0)
            elapsed = time.monotonic() - started
        finally:
            await router.aclose()
    assert profile.name == "flash" and answer == "Here is the answer."
    assert hits == ["slow", "fast"]
    assert elapsed < 3.0, "the fallback answered inside the original budget"


async def test_with_no_other_provider_the_only_one_keeps_the_whole_budget():
    hits = []
    async with TestServer(_app(hits)) as server:
        router = _router(_cloud("reasoning", str(server.make_url("/slow/v1")), timeout_s=90.0))
        router.MIN_ATTEMPT_S = 0.2
        try:
            started = time.monotonic()
            with pytest.raises(ProviderDeadlineError):
                await router.generate_text("Explain engineering functions.", deadline_s=1.2)
            assert time.monotonic() - started >= 1.0, "cut short with nobody to fall back on"
        finally:
            await router.aclose()
    assert hits == ["slow"]


def test_the_reserve_is_at_most_a_third_and_never_without_a_deadline():
    router = _router(_cloud("one", "https://unused.invalid/v1"))
    assert router._fallback_reserve_s(None) == 0.0
    assert router._fallback_reserve_s(90.0) == pytest.approx(30.0)
    assert router._fallback_reserve_s(9.0) == pytest.approx(3.0)
    scope = _CallScope(deadline=time.monotonic() + 10.0, reserve_s=4.0)
    assert 5.5 < scope.attempt_remaining() <= 6.0


def test_a_specialist_says_plainly_that_no_model_answered():
    from orion_core.agents import _provider_failure
    from orion_core.providers import AllTextProvidersFailedError
    late = ProviderDeadlineError(RuntimeError("gemini_reasoning was still answering"),
                                 attempts=["gemini_reasoning"])
    said = _provider_failure("AI / Machine-Learning Agent", late)
    assert said == ("The AI / Machine-Learning Agent could not reach a provider in time: "
                    "every model it asked was too slow to answer.")
    failed = _provider_failure("Coding Agent", AllTextProvidersFailedError(
        RuntimeError("HTTP 503"), attempts=["gemini_reasoning", "gemini_text"]))
    assert failed == ("The Coding Agent could not reach a provider: every model it asked "
                      "failed (gemini_reasoning, gemini_text).")
    assert "all text providers failed" not in said + failed
