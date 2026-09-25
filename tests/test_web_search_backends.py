"""Finding pages, by whichever route is actually working.

Written after watching a single-backend version fail in the way that matters.
DuckDuckGo's HTML endpoint answered the first query with forty results and
every query after that with **202 Accepted and an empty page**. Not an error,
not a 429 - a success-shaped response containing nothing. A research engine
sitting on that alone reports "I could not find anything about X" and sounds
like it looked.

Wikipedia then refused too, with a 403 and a link to its robot policy, because
"ORION-Research/1.0" is not an agent it will serve. Both failures were silent
and both produced an empty list, which is why the distinction these tests
enforce - between "nothing found" and "nothing could be asked" - is the whole
point of the module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import web_search_backends as wsb  # noqa: E402

#: A DuckDuckGo page with one real result in it.
GOOD_HTML = '''
<div><a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">
Example page</a>
<a class="result__snippet">Some description of it.</a></div>
'''

#: What being throttled looks like: right shape, no results.
THROTTLED_HTML = "<html><body><div>Try again later</div></body></html>"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    wsb.reset()
    # These exercise the keyless ladder. Without this the Google backend would
    # read the developer's real Gemini key and search the live web mid-test.
    monkeypatch.setattr(wsb, "gemini_key", lambda: "")
    yield
    wsb.reset()


async def test_google_leads_when_orion_has_a_gemini_key(monkeypatch):
    async def fake_google(query, limit):
        return [wsb.Hit("G", "https://g.example", "", "google")]
    monkeypatch.setattr(wsb, "BACKENDS", (("google", fake_google),) + tuple(
        b for b in wsb.BACKENDS if b[0] != "google"))
    hits = await wsb.search("anything")
    assert hits[0].backend == "google"


def _responder(monkeypatch, replies):
    """Stub the HTTP layer with (status, body) per URL fragment."""
    calls = []

    async def fake_get(url, headers, params=None):
        calls.append((url, headers, params))
        for fragment, reply in replies.items():
            if fragment in url:
                return reply
        return (404, "")

    monkeypatch.setattr(wsb, "_get", fake_get)
    return calls


# -- the 202 trap ------------------------------------------------------------

async def test_a_throttled_duckduckgo_is_not_read_as_nothing_found(monkeypatch):
    """The fault that started all this: 202 with an empty page is a refusal,
    not an answer, and returning [] makes it indistinguishable from an obscure
    topic."""
    _responder(monkeypatch, {"duckduckgo": (202, THROTTLED_HTML),
                             "wikipedia": (500, "")})
    monkeypatch.setattr(wsb, "brave_key", lambda: "")
    with pytest.raises(wsb.SearchUnavailable):
        await wsb.search("anything")


async def test_a_200_with_no_results_also_counts_as_throttled(monkeypatch):
    """It sometimes answers 200 with the results stripped out, which is the
    same refusal wearing a better status code."""
    _responder(monkeypatch, {"duckduckgo": (200, THROTTLED_HTML),
                             "wikipedia": (500, "")})
    monkeypatch.setattr(wsb, "brave_key", lambda: "")
    with pytest.raises(wsb.SearchUnavailable):
        await wsb.search("anything")
    assert "duckduckgo" in wsb.status()["standing_down"]


async def test_a_throttled_backend_is_left_alone_afterwards(monkeypatch):
    """Hammering it makes the window longer, so it is stood down rather than
    retried into the ground."""
    calls = _responder(monkeypatch, {"duckduckgo": (202, THROTTLED_HTML),
                                     "wikipedia": (500, "")})
    monkeypatch.setattr(wsb, "brave_key", lambda: "")
    for _ in range(3):
        with pytest.raises(wsb.SearchUnavailable):
            await wsb.search("anything")
    hits = [c for c in calls if "duckduckgo" in c[0]]
    assert len(hits) == 1, f"asked a stood-down backend {len(hits)} times"


# -- falling through ---------------------------------------------------------

async def test_it_falls_through_to_a_backend_that_answers(monkeypatch):
    _responder(monkeypatch, {
        "duckduckgo": (202, THROTTLED_HTML),
        "api.php": (200, '{"query":{"search":[{"title":"Spacing effect",'
                         '"snippet":"a <b>thing</b>"}]}}')})
    monkeypatch.setattr(wsb, "brave_key", lambda: "")
    hits = await wsb.search("spacing", limit=3)
    assert hits and hits[0].backend == "wikipedia"
    assert "Spacing effect" in hits[0].title
    assert "<b>" not in hits[0].snippet


async def test_brave_is_preferred_when_a_key_exists(monkeypatch):
    _responder(monkeypatch, {
        "brave": (200, '{"web":{"results":[{"title":"T","url":"https://e.com",'
                       '"description":"d"}]}}')})
    monkeypatch.setattr(wsb, "brave_key", lambda: "a-key")
    hits = await wsb.search("anything")
    assert hits[0].backend == "brave"


async def test_no_brave_key_means_brave_is_simply_skipped(monkeypatch):
    calls = _responder(monkeypatch, {
        "duckduckgo": (200, GOOD_HTML), "api.php": (500, "")})
    monkeypatch.setattr(wsb, "brave_key", lambda: "")
    hits = await wsb.search("anything")
    assert hits[0].backend == "duckduckgo"
    assert not [c for c in calls if "brave" in c[0]]


# -- parsing -----------------------------------------------------------------

async def test_the_redirect_wrapper_is_unwrapped(monkeypatch):
    """DuckDuckGo wraps every result in its own redirect; the real URL is
    inside, and following the wrapper would make every source read as
    duckduckgo.com."""
    _responder(monkeypatch, {"duckduckgo": (200, GOOD_HTML)})
    monkeypatch.setattr(wsb, "brave_key", lambda: "")
    hits = await wsb.search("anything")
    assert hits[0].url == "https://example.com/page"
    assert "uddg" not in hits[0].url


async def test_snippets_are_stripped_of_markup(monkeypatch):
    _responder(monkeypatch, {"duckduckgo": (200, GOOD_HTML)})
    monkeypatch.setattr(wsb, "brave_key", lambda: "")
    hits = await wsb.search("anything")
    assert hits[0].snippet == "Some description of it."


# -- the agent Wikimedia insists on ------------------------------------------

def test_wikipedia_identifies_itself_as_the_policy_requires():
    """"ORION-Research/1.0" earned a 403 and a link to the robot policy, which
    is a polite way of saying identify yourself."""
    assert "ORION" in wsb._WIKI_AGENT
    assert "http" in wsb._WIKI_AGENT, "no contact URL, which is what it wants"


def test_the_agent_does_not_leak_a_personal_address():
    assert "@" not in wsb._WIKI_AGENT.replace("://", "")


async def test_wikipedia_is_asked_with_that_agent(monkeypatch):
    calls = _responder(monkeypatch, {"api.php": (200, '{"query":{"search":[]}}')})
    await wsb.search_wikipedia("anything", 3)
    assert calls[0][1]["User-Agent"] == wsb._WIKI_AGENT


# -- reporting ---------------------------------------------------------------

async def test_the_caller_is_told_which_backend_answered(monkeypatch):
    _responder(monkeypatch, {"duckduckgo": (200, GOOD_HTML)})
    monkeypatch.setattr(wsb, "brave_key", lambda: "")
    said = []
    await wsb.search("anything", say=lambda kind, message: said.append(message))
    assert any("duckduckgo" in m for m in said)
    assert any("BRAVE_API_KEY" in m or "API key" in m for m in said), (
        "nothing points at the fix for the unreliable free backend")


async def test_total_failure_says_what_was_tried(monkeypatch):
    _responder(monkeypatch, {})
    monkeypatch.setattr(wsb, "brave_key", lambda: "")
    said = []
    with pytest.raises(wsb.SearchUnavailable):
        await wsb.search("x", say=lambda kind, message: said.append(message))
    assert any("no search backend would answer" in m for m in said)


def test_status_reports_what_is_usable():
    state = wsb.status()
    assert "brave_key" in state and "order" in state
    assert state["order"][-1] == "wikipedia", (
        "Wikipedia is the last resort; anything after it would never run")
