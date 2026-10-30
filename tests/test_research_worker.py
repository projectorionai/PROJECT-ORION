"""
Investigations that run while nobody is watching.

The engine — decomposition, synthesis, section writing — is ``ResearchAgent``
and is called rather than reimplemented. What is tested here is everything
that only matters when the run is unattended: that sources are read from
whatever shape the search server returned them in, that citations are
deduplicated, that a dossier with no sources says so, and that finishing is
announced — because a report nobody knows exists is a report nobody reads.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from orion_core.data import ToolResult

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.research_worker import (  # noqa: E402
    Dossier,
    ResearchWorker,
    Source,
    dedupe,
    slugify,
)
from orion_core.research_worker import _parse_results  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


# ── naming and deduplication ──────────────────────────────────────────────────

def test_a_dossier_filename_still_looks_like_its_topic():
    assert slugify("UK Interest Rates & the Pound!") == "uk-interest-rates-the-pound"
    assert slugify("") == "untitled"
    assert len(slugify("x" * 200)) <= 60


def test_the_same_page_found_twice_is_cited_once():
    """Several searches on related questions return the same page repeatedly.
    Left alone the citation list is mostly the same three links, which makes a
    thin dossier look well-sourced."""
    kept = dedupe([
        Source("A", "https://x/1"), Source("A again", "https://x/1"),
        Source("B", "https://x/2/"), Source("B again", "https://x/2"),
    ])
    assert [s.title for s in kept] == ["A", "B"]


def test_a_source_with_no_url_is_deduplicated_by_title():
    """An encyclopaedia summary has no URL but is still the same summary."""
    kept = dedupe([Source("Overview"), Source("Overview"), Source("Other")])
    assert len(kept) == 2


def test_the_first_title_seen_is_the_one_kept():
    kept = dedupe([Source("Proper title", "https://x"), Source("SEO junk", "https://x")])
    assert kept[0].title == "Proper title"


# ── reading whatever the server returned ──────────────────────────────────────

@pytest.mark.parametrize("payload, title, url", [
    ('[{"title":"T1","url":"https://a","content":"c"}]', "T1", "https://a"),
    ('{"web":{"results":[{"title":"T2","url":"https://b","description":"d"}]}}',
     "T2", "https://b"),
    ('{"results":[{"name":"T3","link":"https://c","snippet":"s"}]}',
     "T3", "https://c"),
    ("I found [Bank rate held](https://d) today.", "Bank rate held", "https://d"),
])
def test_results_are_read_in_every_shape_a_server_returns(payload, title, url):
    """MCP servers return a JSON array, an object with a results key, or prose
    with links, depending on the server and the version. Assuming one is how a
    working search server produces an empty dossier."""
    found = _parse_results(payload, "test")
    assert found and found[0].title == title and found[0].url == url


@pytest.mark.parametrize("nothing", ["", "   ", "no results at all"])
def test_a_reply_with_nothing_in_it_yields_nothing(nothing):
    assert _parse_results(nothing) == []


# ── a whole investigation ─────────────────────────────────────────────────────

class _Agent:
    """Stands in for ResearchAgent."""

    def __init__(self, sources=None) -> None:
        self._sources = sources if sources is not None else [
            {"title": "A source", "url": "https://x/1"}]

    async def _decompose(self, topic):
        return [f"Q1 about {topic}", f"Q2 about {topic}"]

    async def gather_sources(self, question):
        return list(self._sources)

    async def _synthesise(self, topic, question, rows):
        return f"Body for {question} from {len(rows)} sources."

    async def _final_summary(self, topic, questions):
        return f"Summary of {topic}."

    async def _ask(self, prompt, persona):
        return "- First thing that matters\n- Second thing that matters"


class _Signal:
    def __init__(self):
        self.sent = []

    def emit(self, *args):
        self.sent.append(args)


class _Bus:
    def __init__(self):
        self.log = _Signal()
        self.speak_request = _Signal()


@pytest.fixture
def worker(tmp_path):
    bus = _Bus()
    return ResearchWorker(agent=_Agent(), bus=bus, root=tmp_path), bus


def test_an_investigation_produces_a_filed_dossier(worker):
    research, _bus = worker
    dossier = _run(research.investigate("UK interest rates", questions=2))
    assert dossier.path is not None and dossier.path.exists()
    assert dossier.sources


def test_the_dossier_has_the_sections_it_promises(worker):
    research, _bus = worker
    dossier = _run(research.investigate("UK interest rates", questions=2))
    text = dossier.path.read_text(encoding="utf-8")
    for heading in ("# UK interest rates", "## Executive summary",
                    "## Key takeaways", "## Sources"):
        assert heading in text, f"missing {heading}"


def test_finishing_is_announced(worker):
    """A report nobody knows exists is a report nobody reads — and this runs
    unattended, so there is no moment the user would notice by themselves."""
    research, bus = worker
    _run(research.investigate("anything", questions=1))
    assert bus.speak_request.sent


def test_a_reminder_service_is_preferred_for_the_announcement(tmp_path):
    told = []

    class _Reminders:
        def add(self, **kwargs):
            told.append(kwargs)
            return ToolResult("Reminder set")

    bus = _Bus()
    research = ResearchWorker(agent=_Agent(), bus=bus, reminders=_Reminders(),
                              root=tmp_path)
    _run(research.investigate("anything", questions=1))
    assert told, "the reminder service should have been used"
    assert not bus.speak_request.sent, "it should not also have been spoken"


def test_a_dossier_with_no_sources_admits_it(tmp_path):
    """Otherwise it reads as researched when it was written from the model's
    own memory, which is the one thing a reader must not be wrong about."""
    research = ResearchWorker(agent=_Agent(sources=[]), bus=_Bus(),
                              root=tmp_path)
    dossier = _run(research.investigate("obscure", questions=1))
    assert "model's own knowledge" in dossier.path.read_text(encoding="utf-8")


def test_an_empty_topic_is_refused_quietly(tmp_path):
    research = ResearchWorker(agent=_Agent(), bus=_Bus(), root=tmp_path)
    dossier = _run(research.investigate("   "))
    assert dossier.path is None
    assert "No topic" in dossier.summary


def test_the_dossier_names_which_gatherer_found_the_sources(worker):
    """"No sources found" and "no search server configured" are very different
    problems and should not look alike."""
    research, _bus = worker
    dossier = _run(research.investigate("anything", questions=1))
    assert dossier.gatherer
    assert dossier.gatherer in dossier.as_markdown()


# ── the search server ─────────────────────────────────────────────────────────

class _MCP:
    def __init__(self, names, payload='[{"title":"T","url":"https://a"}]',
                 ok=True):
        self._names, self._payload, self._ok = names, payload, ok
        self.calls = []

    def server_names(self):
        return list(self._names)

    async def call_tool(self, server, tool, arguments):
        self.calls.append((server, tool))
        if tool not in {"search", "tavily_search", "brave_web_search",
                        "web_search"}:
            raise RuntimeError("no such tool")

        class _Result:
            pass
        result = _Result()
        result.ok = self._ok
        result.text = self._payload
        return result


def test_a_search_server_is_preferred_when_one_is_connected(tmp_path):
    mcp = _MCP(["tavily"])
    research = ResearchWorker(agent=_Agent(), mcp=mcp, bus=_Bus(), root=tmp_path)
    assert research.search_server() == "tavily"
    found, gatherer = _run(research.gather("what happened?"))
    assert gatherer == "tavily"
    assert found[0].url == "https://a"


def test_brave_is_used_when_tavily_is_not_there(tmp_path):
    research = ResearchWorker(agent=_Agent(), mcp=_MCP(["brave-search"]),
                              bus=_Bus(), root=tmp_path)
    assert "brave" in research.search_server()


def test_without_a_search_server_the_builtin_path_runs(tmp_path):
    research = ResearchWorker(agent=_Agent(), mcp=_MCP(["gmail"]), bus=_Bus(),
                              root=tmp_path)
    assert research.search_server() == ""
    _found, gatherer = _run(research.gather("what happened?"))
    assert "built-in" in gatherer


def test_a_search_server_returning_nothing_falls_back_and_says_so(tmp_path):
    bus = _Bus()
    research = ResearchWorker(agent=_Agent(), mcp=_MCP(["tavily"], payload="[]"),
                              bus=bus, root=tmp_path)
    found, _gatherer = _run(research.gather("what happened?"))
    assert found, "it should have fallen back to the built-in lookup"
    assert any("returned nothing" in str(line) for line in bus.log.sent)


def test_a_gatherer_that_raises_does_not_end_the_investigation(tmp_path):
    class _Broken(_Agent):
        async def gather_sources(self, question):
            raise RuntimeError("the network is gone")

    research = ResearchWorker(agent=_Broken(), bus=_Bus(), root=tmp_path)
    dossier = _run(research.investigate("anything", questions=1))
    assert dossier.path is not None, "it should still have filed something"


def test_a_dossier_can_be_rendered_without_ever_running_anything():
    """The markdown is a pure function of the dossier, so it can be checked
    without a model, a network or a filesystem."""
    dossier = Dossier(topic="Topic", started_at=0.0, summary="S",
                      takeaways=["One", "Two"],
                      sections=[("Q", "A")],
                      sources=[Source("T", "https://u")], gatherer="test")
    text = dossier.as_markdown()
    assert "1. One" in text and "[T](https://u)" in text and "## Q" in text


def test_failed_filing_is_not_announced_as_saved_in_memory(tmp_path):
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("synthetic marker", encoding="utf-8")
    bus = _Bus()
    research = ResearchWorker(agent=_Agent(), bus=bus, root=blocked)
    dossier = _run(research.investigate("solar energy", questions=1))
    assert dossier.path is None and dossier.filing_error
    assert str(blocked) in dossier.filing_error
    assert dossier.summary and dossier.sections
    announcement = " ".join(str(item) for item in bus.speak_request.sent)
    assert "no report was saved" in announcement
    assert "filed as memory" not in announcement
    assert blocked.read_text(encoding="utf-8") == "synthetic marker"


def test_same_topic_and_timestamp_do_not_overwrite_an_earlier_dossier(tmp_path):
    research = ResearchWorker(root=tmp_path)
    first = Dossier("same topic", 1.0, summary="First result")
    second = Dossier("same topic", 1.0, summary="Second result")
    first_path = research.file(first)
    second_path = research.file(second)
    assert first_path != second_path
    assert "First result" in first_path.read_text(encoding="utf-8")
    assert "Second result" in second_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".dossier-*.tmp"))


def test_failed_atomic_publication_leaves_existing_reports_and_no_partial_file(tmp_path, monkeypatch):
    import orion_core.research_worker as module
    older = tmp_path / "older-report.md"
    older.write_bytes(b"older verified report")
    def denied(_source, _destination):
        raise PermissionError("synthetic access denied")
    monkeypatch.setattr(module.os, "replace", denied)
    dossier = Dossier("current topic", 1.0, summary="New result")
    assert ResearchWorker(root=tmp_path).file(dossier) is None
    assert "synthetic access denied" in dossier.filing_error
    assert list(tmp_path.iterdir()) == [older]
    assert older.read_bytes() == b"older verified report"


def test_default_location_is_independent_of_the_launcher_working_directory(tmp_path, monkeypatch):
    import orion_core.research_worker as module
    configured = tmp_path / "private" / "research" / "dossiers"
    monkeypatch.setattr(module, "DOSSIER_DIR", configured)
    unrelated = tmp_path / "launcher-cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    path = ResearchWorker().file(Dossier("default location", 1.0, summary="Result"))
    assert path.parent == configured
    assert not (unrelated / "research").exists()


def test_investigation_files_off_the_event_loop(tmp_path):
    import threading
    owner = threading.get_ident()
    threads = []
    class Recorded(ResearchWorker):
        def file(self, dossier):
            threads.append(threading.get_ident())
            return super().file(dossier)
    dossier = _run(Recorded(agent=_Agent(), root=tmp_path).investigate("topic", questions=1))
    assert dossier.path is not None, dossier.filing_error
    assert threads and owner not in threads


def test_dispatch_delivers_research_content_with_failed_filing_status(monkeypatch):
    from types import SimpleNamespace
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    dossier = Dossier("solar energy", 1.0, summary="Useful synthetic summary",
                      takeaways=["Useful takeaway"], filing_error="synthetic access denied")
    async def completed(self, topic, questions):
        return dossier
    monkeypatch.setattr(ResearchWorker, "investigate", completed)
    dispatch = SimpleNamespace(research=None, bus=None, reminders=None)
    result = _run(KnowledgeDispatchMixin._dossier(dispatch, "solar energy", {"questions": 1}))
    assert not result.ok
    assert "synthetic access denied" in result.text
    assert "Useful synthetic summary" in result.text and "Useful takeaway" in result.text


# ── accuracy (Mark XXXX.9, owner 30 Oct 2026) ─────────────────────────────────

class _ReadingAgent(_Agent):
    """Gathers like ResearchAgent: pages read when asked, snippets otherwise."""

    def __init__(self, read=True):
        super().__init__()
        self.read = read
        self.asked_to_read = []
        self.rows = []

    async def gather_sources(self, question, *, read_pages=0):
        self.asked_to_read.append(read_pages)
        return [{"title": "Read page", "url": "https://x/read",
                 "snippet": "Evidence from the page.", "evidence": "page" if self.read else "snippet"},
                {"title": "Headline", "url": "https://x/news", "snippet": "A headline.",
                 "evidence": "headline"}]

    async def _synthesise(self, topic, question, rows):
        self.rows.append(rows)
        return f"Body for {question} [{rows[0]['n']}]."


def test_a_dossier_asks_for_pages_numbers_its_citations_and_says_what_was_read(tmp_path):
    """"11 sources via the built-in encyclopaedia and news lookup" was eleven
    snippets and headlines, not one page read."""
    agent = _ReadingAgent()
    dossier = _run(ResearchWorker(agent=agent, bus=_Bus(), root=tmp_path).investigate("Topic", questions=2))
    assert agent.asked_to_read and all(n > 0 for n in agent.asked_to_read)
    # One numbering for the whole dossier: both questions cite the read page as [1].
    assert [rows[0]["n"] for rows in agent.rows] == [1, 1]
    assert dossier.evidence_level == "pages" and "1 page read" in dossier.gatherer
    text = dossier.path.read_text(encoding="utf-8")
    assert "1. [Read page](https://x/read) — page read" in text
    assert "2. [Headline](https://x/news) — headline only" in text


def test_a_dossier_that_read_no_page_says_so(tmp_path):
    bus = _Bus()
    dossier = _run(ResearchWorker(agent=_ReadingAgent(read=False), bus=bus, root=tmp_path)
                   .investigate("Topic", questions=1))
    assert dossier.evidence_level == "snippets_only"
    assert "no web page was opened" in dossier.gatherer
    assert "No web page could be read" in str(bus.speak_request.sent)


def test_a_dossier_reports_its_start_and_one_finish_even_when_cancelled(tmp_path):
    from orion_core import research_runs

    class _Hanging(_Agent):
        async def _synthesise(self, topic, question, rows):
            await asyncio.sleep(30)

    bus = _Bus()
    bus.dashboard_event = _Signal()

    async def scenario():
        task = asyncio.create_task(ResearchWorker(agent=_Hanging(), bus=bus, root=tmp_path)
                                   .investigate("Hanging topic", questions=1))
        await asyncio.sleep(0.05)
        assert any(row["topic"] == "Hanging topic" for row in research_runs.live_activity())
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    _run(scenario())
    events = [payload for channel, payload in bus.dashboard_event.sent if channel == "research"]
    assert events[0]["status"] == "running" and events[-1]["status"] == "cancelled"
    assert [e["status"] for e in events].count("cancelled") == 1
    assert not any(row["topic"] == "Hanging topic" for row in research_runs.live_activity())
