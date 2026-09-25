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
