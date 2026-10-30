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


async def test_one_failing_question_does_not_sink_the_run(researcher, monkeypatch):
    """A search fault on one question is recorded; the others still finish,
    and the run is written up rather than raising."""
    from orion_core.web_search_backends import Hit

    async def flaky_search(query, limit=8, say=None):
        if "effect" in query:
            raise RuntimeError("backend exploded")
        return [Hit(title="Good page", url="https://good.example/a", backend="stub")]

    monkeypatch.setattr("orion_core.web_search_backends.search", flaky_search)
    run = await researcher.investigate("spacing", questions=2, per_question=1, rounds=1)
    assert any("backend exploded" in reason for _q, reason in run.failed)
    assert run.notes, "the healthy question should still have produced a note"


def test_a_page_is_capped_and_cleaned_off_the_event_loop():
    # Mark XXXX: a 5 MB page took 117 ms to clean, on the loop beside his voice.
    import inspect
    from orion_core import live_research as lr
    big = "<title>Big page</title>" + "<p>word</p>" * (lr.MAX_PAGE_BYTES // 10)
    title, text = lr._digest(big)
    assert title == "Big page" and text.startswith("Big page")
    assert len(text) < lr.MAX_PAGE_BYTES
    assert "asyncio.to_thread(_digest, body)" in inspect.getsource(lr.LiveResearcher.read)


def test_a_page_sent_in_chunks_is_read_to_the_end(tmp_path):
    """Mark XXXX.4: content.read(n) returned only the first chunk to arrive,
    usually the <head>, so ordinary pages were dropped as thin."""
    import asyncio
    from aiohttp import web
    from orion_core import live_research as lr

    page = ("<html><head><title>T</title></head><body>" + "x" * 200_000
            + "<p>THE END</p></body></html>").encode()

    async def handler(request):
        response = web.StreamResponse(headers={"Content-Type": "text/html"})
        await response.prepare(request)
        for start in range(0, len(page), 20_000):
            await response.write(page[start:start + 20_000])
            await asyncio.sleep(0.005)
        return response

    async def run():
        app = web.Application()
        app.router.add_get("/", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            researcher = lr.LiveResearcher(ask=_ask, root=tmp_path)
            return await researcher._fetch(f"http://127.0.0.1:{port}/", 10.0)
        finally:
            await runner.cleanup()

    body = asyncio.run(run())
    assert len(body) == len(page) and body.endswith("</html>")



def test_a_second_run_on_the_research_already_running_is_not_started():
    """30 Oct 2026: a dossier on "circadian rhythm and sleep strategies" started
    90 s into a run on the same question and spent the quota twice."""
    from types import SimpleNamespace
    from orion_core.data import ToolResult
    from orion_core.dispatch_knowledge import _overlapping_research

    class Running:
        def done(self):
            return False

    research = SimpleNamespace(
        _active={"task": Running(), "topic": "circadian rhythm phase delay and practical "
                                             "strategies for severe sleep schedule disruption"},
        status=lambda: ToolResult("Researching it - 2.0 of 30 minutes elapsed."))
    said = _overlapping_research(research, "circadian rhythm and sleep strategies")
    assert said.startswith("That is already being researched.")
    assert _overlapping_research(research, "the history of the printing press") == ""


# -- accuracy (Mark XXXX.9, owner 30 Oct 2026: "if ORION's researching it must
# be more accurate") ----------------------------------------------------------

def test_a_claim_with_no_source_is_marked_and_an_invented_citation_removed():
    """A note that cites a number it was never given, or states something no
    source says, must not read as sourced."""
    from orion_core.live_research import UNSOURCED, check_citations

    text = ("Bright light after waking advanced the sleep phase by an hour [1].\n"
            "Sources disagree: [1] says one hour while [2] reports no change at all.\n"
            "Consistent wake times are widely recommended by most sleep clinicians.\n"
            "Shift workers report similar improvements within a fortnight [9].")
    checked, check = check_citations(text, {1: "page", 2: "snippet"})
    lines = checked.splitlines()
    assert lines[0].endswith("[1].") and UNSOURCED not in lines[0]
    assert UNSOURCED not in lines[1], "a labelled disagreement is not a bare claim"
    assert lines[2].endswith(f"clinicians {UNSOURCED}.")
    assert "[9]" not in checked and lines[3].endswith(f"fortnight {UNSOURCED}.")
    assert (check.claims, check.cited, check.on_pages, check.unsourced, check.removed,
            check.disagreements) == (3, 1, 1, 2, 1, 1)
    assert "1 of 3 claims cite a source" in check.line()


def test_the_passage_that_answers_the_question_reaches_the_model():
    """An excerpt used to be the first few hundred characters - menus and an
    introduction - so a fact further down the page never reached the note."""
    from orion_core.live_research import relevant_passages

    filler = "\n".join("Readers often write in to ask about naps, caffeine and the best pillow to buy."
                       for _ in range(60))
    fact = "In a controlled trial, 30 minutes of bright light advanced the sleep phase by about an hour."
    page = f"Home | News | Sleep\n{filler}\n{fact}\n{filler}"
    passage = relevant_passages(page, "How does bright light shift a delayed sleep phase?", limit=300)
    assert fact in passage and len(passage) <= 300
    assert "Home | News" not in passage


def test_what_the_sources_really_were_is_said_plainly():
    from orion_core.live_research import describe_evidence

    assert describe_evidence([]) == ("none", "no sources")
    level, phrase = describe_evidence([{"title": "Wikipedia: Sleep", "snippet": "x"}])
    assert level == "encyclopaedia_only" and "no web page was opened" in phrase
    level, phrase = describe_evidence([{"title": "Hit", "url": "https://x", "evidence": "snippet"},
                                       {"title": "News", "url": "https://y", "evidence": "headline"}])
    assert level == "snippets_only" and "no web page was opened" in phrase
    level, phrase = describe_evidence([{"title": "Hit", "url": "https://x", "evidence": "page"},
                                       {"title": "News", "url": "https://y", "evidence": "headline"}])
    assert level == "pages" and phrase.startswith("1 page read and 1 ")


async def test_a_page_that_does_not_address_the_question_is_not_a_source(researcher):
    """Its "note" was kept, numbered and listed under Sources, so an off-topic
    page made the investigation look better sourced than it was."""
    async def ask(prompt, persona="x"):
        if persona == "research planner":
            return "NOTHING" if "still missing" in prompt else "What does the evidence say?"
        if persona == "research reader":
            assert "NOT RELEVANT" in prompt
            return "NOT RELEVANT"
        return "Summary."

    researcher.ask = ask
    run = await researcher.investigate("spacing", questions=1, per_question=1, rounds=1)
    assert run.notes == []
    assert any(reason == "did not address the question" for _url, reason in run.failed)


async def test_the_write_up_cannot_cite_a_note_that_does_not_exist(researcher):
    run = await researcher.investigate("spacing", questions=1, per_question=1, rounds=1)
    assert len(run.notes) == 1

    async def writer(prompt, persona="x"):
        return ("The page reports a measurable effect of spacing on recall [1]. "
                "A second study found the same effect in older adults [4].")

    researcher.ask = writer
    summary = await researcher.summarise(run)
    assert "[1]" in summary and "[4]" not in summary
    assert summary.endswith("older adults [unsourced].")
    folder = researcher.write_folder(run, summary)
    assert "[1] " in (folder / "SUMMARY.md").read_text(encoding="utf-8").split("## Sources")[1]


async def test_a_reading_run_reports_its_start_and_one_finish(researcher):
    """The BRAIN and NEURAL pages show research only while it really runs."""
    from orion_core import research_runs

    seen = []
    researcher.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None),
                                     dashboard_event=SimpleNamespace(emit=lambda *a: seen.append(a)))
    await researcher.investigate("spacing", questions=1, per_question=1, rounds=1)
    research = [payload for channel, payload in seen if channel == "research"]
    assert research[0]["status"] == "running" and research[0]["engine"] == "browse"
    assert any(p.get("question") for p in research[1:-1]), "the current question was reported"
    assert [p["status"] for p in research].count("done") == 1 and research[-1]["status"] == "done"
    assert all(p["run"] == research[0]["run"] for p in research)
    assert not [row for row in research_runs.live_activity() if row["run"] == research[0]["run"]]
