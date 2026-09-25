"""Research that reads, rather than research that writes.

What ORION did before was fetch a Wikipedia summary and a few headlines, hand
those to the model as context, and ask it to write. The prose came out fluent
and fast and almost entirely from the model's own memory - a good way to
produce an essay and a poor way to find anything out. It cannot tell you what
a page actually said, it cannot notice two sources disagreeing, and its
citations point at things it never opened.

These cover the loop that does the other thing, and the two failures that made
the first version of it useless: a search backend that answers "nothing" when
it means "not you", and a page fetcher that sent the wrong User-Agent.

No network here. The fetch is stubbed, because a test that depends on the live
web fails for reasons that have nothing to do with this code.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import live_research as lr  # noqa: E402
from orion_core.live_research import Finding, Investigation, LiveResearcher  # noqa: E402
from orion_core.dispatch_knowledge import KnowledgeDispatchMixin  # noqa: E402
from orion_core.data import ToolResult  # noqa: E402


def test_browse_research_releases_voice_turn_and_announces_completion():
    spoken = []

    async def browse(_topic, _args):
        await asyncio.sleep(0)
        return ToolResult("Research complete from the sources")

    shim = SimpleNamespace(
        research=object(),
        _browse_research=browse,
        bus=SimpleNamespace(speak_request=SimpleNamespace(emit=spoken.append)),
    )

    async def scenario():
        result = await KnowledgeDispatchMixin.research_tool(
            shim, {"action": "browse", "topic": "speech reliability"})
        assert "I'm researching" in result.text
        await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert spoken == ["Research complete from the sources"]

PAGE = ("<html><head><title>Spacing effect</title></head><body>"
        "<script>ignore()</script><p>" + ("Real readable prose. " * 60) +
        "</p></body></html>")


async def _ask(prompt, persona="x"):
    if persona == "research planner":
        if "still missing" in prompt:
            return "NOTHING"
        return "What does the evidence say?\nHow large is the effect?"
    if persona == "research reader":
        return "The page reports a measurable effect."
    return "Summary from the notes [1]."


@pytest.fixture()
def researcher(monkeypatch, tmp_path):
    """A researcher whose web is a dictionary."""
    r = LiveResearcher(ask=_ask, root=tmp_path)
    r.events = []
    r.on_event = r.events.append

    async def fake_fetch(url, timeout):
        return PAGE if "good" in url else ""

    monkeypatch.setattr(r, "_fetch", fake_fetch)

    async def fake_search(query, limit=8, say=None):
        from orion_core.web_search_backends import Hit

        return [Hit(title=f"Good {query[:12]}", url="https://good.example/a",
                    backend="stub"),
                Hit(title="Bad", url="https://bad.example/b", backend="stub")]

    monkeypatch.setattr("orion_core.web_search_backends.search", fake_search)
    return r


# -- reading a page ----------------------------------------------------------

def test_a_page_is_stripped_to_readable_text():
    text = lr._clean(PAGE)
    assert "Real readable prose" in text
    assert "<p>" not in text and "ignore()" not in text


async def test_a_page_that_yields_almost_nothing_is_not_noted(researcher):
    """A cookie wall, a login page or a JavaScript shell. Taking a note from
    one produces a confident sentence about nothing."""
    assert await researcher.read(Finding("Bad", "https://bad.example/b")) == ""


async def test_a_read_page_is_capped(monkeypatch, researcher):
    """One enormous article must not swallow the whole context budget."""
    async def huge(url, timeout):
        return "<html><body>" + ("word " * 200_000) + "</body></html>"

    monkeypatch.setattr(researcher, "_fetch", huge)
    text = await researcher.read(Finding("Huge", "https://good.example/a"))
    assert 0 < len(text) <= lr.MAX_PAGE_CHARS


def test_wikimedia_gets_the_agent_its_policy_demands():
    """Wikimedia REFUSES a generic agent, and a browser string earned a 403 on
    every article - which presented as "every page is unreadable". Most other
    sites do the opposite and serve a stub to anything script-shaped, so one
    agent for both was never going to work."""
    source = (ROOT / "orion_core" / "live_research.py").read_text(
        encoding="utf-8")
    assert "_WIKI_AGENT" in source and "wikipedia.org" in source


def test_the_wikipedia_agent_does_not_carry_a_personal_address():
    """Wikimedia asks for somewhere to complain to. That is the project, not
    the person running it."""
    from orion_core.web_search_backends import _WIKI_AGENT

    assert "@" not in _WIKI_AGENT.replace("://", "")
    assert "ORION" in _WIKI_AGENT


# -- the loop ----------------------------------------------------------------

async def test_an_investigation_reads_pages_and_notes_them(researcher):
    run = await researcher.investigate("spacing", questions=2,
                                       per_question=2, rounds=1)
    assert run.notes, "nothing was noted"
    assert run.read, "nothing was read"
    assert all(n.url for n in run.notes), "a note with no source"


async def test_every_note_carries_the_page_it_came_from(researcher):
    run = await researcher.investigate("spacing", questions=1,
                                       per_question=1, rounds=1)
    for note in run.notes:
        assert note.url.startswith("http")
        assert note.question


async def test_the_same_page_is_not_read_twice(researcher):
    run = await researcher.investigate("spacing", questions=3,
                                       per_question=2, rounds=1)
    assert len(run.read) == len(set(run.read))


async def test_it_narrates_before_it_acts(researcher):
    """A long job with no visible progress is indistinguishable from a hung
    one, and the interesting moment in "reading nature.com/..." is while it is
    happening."""
    await researcher.investigate("spacing", questions=1, per_question=1,
                                 rounds=1)
    kinds = [e["kind"] for e in researcher.events]
    assert kinds.index("search") < kinds.index("reading") < kinds.index("note")
    assert "start" in kinds and "done" in kinds


async def test_what_it_finds_decides_what_it_looks_at_next(researcher):
    """A fixed list answered once is a survey. This is the difference."""
    async def ask(prompt, persona="x"):
        if persona == "research planner":
            if "still missing" in prompt:
                return "What about the follow-up question?"
            return "First question here?"
        if persona == "research reader":
            return "A note."
        return "Summary."

    researcher.ask = ask
    run = await researcher.investigate("spacing", questions=1,
                                       per_question=1, rounds=2)
    assert any("follow-up" in q for q in run.questions), (
        "reading changed nothing about what was looked up next")


async def test_an_unreadable_page_is_recorded_not_silently_dropped(researcher):
    run = await researcher.investigate("spacing", questions=1,
                                       per_question=2, rounds=1)
    assert run.failed, "the page that would not open left no trace"


async def test_a_deadline_is_respected(researcher):
    run = await researcher.investigate("spacing", questions=6, per_question=3,
                                       rounds=3, deadline_s=0.001)
    assert run.seconds < 30


# -- writing it up -----------------------------------------------------------

async def test_the_summary_is_built_from_the_notes(researcher):
    run = await researcher.investigate("spacing", questions=1,
                                       per_question=1, rounds=1)
    seen = {}

    async def capture(prompt, persona="x"):
        seen[persona] = prompt
        return "Summary."

    researcher.ask = capture
    await researcher.summarise(run)
    prompt = seen.get("research writer", "")
    assert "notes I took from pages I actually read" in prompt
    assert "rather than filling the gap from your own knowledge" in prompt


async def test_nothing_read_says_so_rather_than_inventing(researcher):
    empty = Investigation(topic="obscure")
    assert "could not read anything" in await researcher.summarise(empty)


async def test_notes_are_filed_one_per_source(researcher, tmp_path):
    """A single transcript is something nobody opens twice."""
    run = await researcher.investigate("spacing", questions=2,
                                       per_question=1, rounds=1)
    folder = researcher.write_folder(run, "The summary.")
    assert folder is not None
    files = list((folder / "notes").glob("*.md"))
    assert len(files) == len(run.notes)
    assert (folder / "SUMMARY.md").exists()
    assert (folder / "run.json").exists()
    assert "Source: http" in files[0].read_text(encoding="utf-8")


# -- it is reachable ---------------------------------------------------------

def test_the_research_tool_has_a_browse_mode():
    source = (ROOT / "orion_core" / "dispatch_knowledge.py").read_text(
        encoding="utf-8")
    assert "_browse_research" in source
    assert '"browse"' in source


def test_the_model_is_told_which_mode_actually_reads():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "research")
    assert "browse" in tool["description"]
    assert "genuinely opened" in tool["description"]
