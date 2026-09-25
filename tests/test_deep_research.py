"""
Research at the length the work actually needs.

  "they can't just be short pieces of research and short paragraphs, I'm
   talking PAGES and PAGES like up to 50 pages in research. Research when done
   must be explicitly announced."

The old paper was six fixed sections whose prompts asked for "150-300 words"
each — about 1,500 words, three pages. The gap to fifty pages is a hundredfold,
so raising the numbers was never the fix: a model asked for 25,000 words in one
call produces something stretched rather than developed.

Length comes from DECOMPOSITION. A real outline is planned for the specific
topic, and each section is written as its own substantial piece. That is also
what makes it faster rather than slower — sections do not depend on each other,
so they are written concurrently and wall-clock time is set by the slowest
section rather than the sum.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import research_docx  # noqa: E402
from orion_core.deep_research import (  # noqa: E402
    DEPTHS, MAX_PARALLEL_SECTIONS, STYLES, WORDS_PER_PAGE, DeepResearchEngine,
    Depth, Paper, Section, Style, resolve_depth, resolve_style,
)


class _Agent:
    """A stub model, so these tests exercise the engine and not a provider."""

    bus = None

    def __init__(self, delay: float = 0.0, fail_section: int | None = None):
        self.delay = delay
        self.fail_section = fail_section
        self.prompts: list[str] = []

    async def _gather_sources(self, topic):
        return [{"title": f"Source {n}", "url": f"https://example.org/{n}",
                 "snippet": "evidence"} for n in range(1, 6)]

    async def _ask(self, prompt, persona):
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if persona == "research planner":
            import re
            wanted = int(re.search(r"EXACTLY (\d+) sections", prompt).group(1))
            return "\n".join(f"Topic {n} :: establish point {n}"
                             for n in range(1, wanted + 1))
        if self.fail_section is not None and f"section {self.fail_section} " in prompt:
            raise RuntimeError("provider unavailable")
        return "word " * 1300


# ── the length targets are internally consistent ─────────────────────────────

@pytest.mark.parametrize("depth", list(Depth))
def test_the_page_target_matches_the_word_budget(depth):
    """The first version promised 50 pages and would have produced 98:
    38 sections x 1,300 words is 49,400 words, not 25,000. ORION would have
    announced a page count he was nowhere near. A target nobody checks is a
    wish."""
    pages, sections, words = DEPTHS[depth]
    implied = sections * words / WORDS_PER_PAGE
    assert abs(implied - pages) <= pages * 0.12, (
        f"{depth.value}: {sections}x{words} words is ~{implied:.0f} pages, "
        f"but it claims {pages}")


@pytest.mark.parametrize("depth", list(Depth))
def test_sections_are_long_enough_to_develop_an_argument(depth):
    """Too short and each section is a summary; too long and the model pads."""
    assert 800 <= DEPTHS[depth][2] <= 1400


def test_fifty_pages_is_reachable():
    pages, sections, words = DEPTHS[resolve_depth("50 pages")]
    assert pages >= 50 or sections * words >= 24_000


# ── the request is understood ────────────────────────────────────────────────

@pytest.mark.parametrize("request_text,expected", [
    ("50 pages", Depth.EXHAUSTIVE),
    ("35 pages", Depth.THESIS),
    ("5 pages", Depth.BRIEF),
    ("thesis", Depth.THESIS),
    ("", Depth.STANDARD),
])
def test_depth_is_resolved_from_how_it_was_asked(request_text, expected):
    assert resolve_depth(request_text) is expected


def test_a_page_count_rounds_UP_to_the_nearest_tier():
    """'40 pages' must not quietly become a 15-page report."""
    assert resolve_depth("40 pages") is Depth.EXHAUSTIVE
    assert DEPTHS[resolve_depth("40 pages")][0] >= 40


def test_an_absurd_page_count_is_capped_not_refused():
    assert resolve_depth("500 pages") is Depth.EXHAUSTIVE


@pytest.mark.parametrize("request_text,expected", [
    ("literature review", Style.LITERATURE_REVIEW),
    ("deep dive", Style.TECHNICAL),
    ("compare", Style.COMPARATIVE),
    ("teach", Style.PRIMER),
    ("critique", Style.CRITICAL),
])
def test_style_is_resolved_from_how_it_was_asked(request_text, expected):
    assert resolve_style(request_text) is expected


def test_an_unknown_style_falls_back_rather_than_failing():
    assert resolve_style("something else entirely") is Style.SYSTEMATIC


def test_every_style_is_a_genuinely_different_brief():
    """'Multiple styles' means different KINDS of thinking, not different
    adjectives — a literature review and a technical deep-dive muddled together
    are useless as either."""
    outlines = {spec.outline_brief for spec in STYLES.values()}
    voices = {spec.section_voice for spec in STYLES.values()}
    assert len(outlines) == len(STYLES)
    assert len(voices) == len(STYLES)


def test_every_style_has_an_evidence_standard():
    for style, spec in STYLES.items():
        assert len(spec.evidence) > 30, style


# ── planning ─────────────────────────────────────────────────────────────────

def test_the_outline_is_planned_for_the_topic():
    engine = DeepResearchEngine(_Agent())
    sections = asyncio.run(engine.plan("brain-computer interfaces",
                                       Style.TECHNICAL, Depth.STANDARD))
    assert len(sections) == DEPTHS[Depth.STANDARD][1]
    assert all(s.title for s in sections)
    assert all(s.brief for s in sections)


def test_list_markers_and_numbering_are_stripped():
    """Models add '1.' and '- ' despite being told the format."""
    engine = DeepResearchEngine(_Agent())
    parsed = engine._parse_outline(
        "1. First topic :: do this\n- Second topic :: do that\n"
        "  * Third topic :: and this", 3)
    assert [s.title for s in parsed] == ["First topic", "Second topic", "Third topic"]


def test_an_unparseable_outline_falls_back_rather_than_aborting():
    engine = DeepResearchEngine(_Agent())
    assert engine._parse_outline("the model ignored the format entirely", 6) == []
    fallback = engine._fallback_outline("a topic", STYLES[Style.SYSTEMATIC], 6)
    assert len(fallback) == 6


# ── writing ──────────────────────────────────────────────────────────────────

def test_a_paper_reaches_the_requested_length():
    engine = DeepResearchEngine(_Agent())
    paper = asyncio.run(engine.write("neural interfaces", "technical", "50 pages"))
    assert paper.words >= 20_000, f"only produced {paper.words} words"
    assert paper.pages >= 40


def test_a_short_request_stays_short():
    """Depth must actually mean something in both directions."""
    engine = DeepResearchEngine(_Agent())
    paper = asyncio.run(engine.write("a small question", "primer", "5 pages"))
    assert paper.pages <= 8


def test_sections_are_written_concurrently():
    """This is what makes deeper research FASTER, not slower. Sections do not
    depend on each other, so wall-clock is the slowest section plus planning —
    not the sum of them."""
    engine = DeepResearchEngine(_Agent(delay=0.05))

    async def _run():
        loop = asyncio.get_running_loop()
        started = loop.time()
        paper = await engine.write("a topic", "systematic", "standard")
        return paper, loop.time() - started

    paper, elapsed = asyncio.run(_run())
    sequential = 0.05 * (len(paper.sections) + 1)
    assert elapsed < sequential * 0.7, (
        f"took {elapsed:.2f}s; sequential would be {sequential:.2f}s")


def test_concurrency_is_bounded():
    """Unbounded parallel model calls trip provider rate limits and end up
    slower, which is how this project has been bitten by 429s before."""
    assert 2 <= MAX_PARALLEL_SECTIONS <= 12


def test_one_failed_section_does_not_lose_the_paper():
    engine = DeepResearchEngine(_Agent(fail_section=3))
    paper = asyncio.run(engine.write("a topic", "systematic", "standard"))
    assert len(paper.failed) == 1
    assert paper.words > 0, "the whole run was lost over one section"


def test_a_failed_section_is_named_not_hidden():
    engine = DeepResearchEngine(_Agent(fail_section=2))
    paper = asyncio.run(engine.write("a topic", "systematic", "standard"))
    assert paper.failed[0].error
    assert "section(s) came back empty" in paper.announcement()


def test_sections_are_told_what_the_others_cover():
    """Otherwise twenty independent sections repeat each other."""
    agent = _Agent()
    engine = DeepResearchEngine(agent)
    asyncio.run(engine.write("a topic", "systematic", "brief"))
    writing = [p for p in agent.prompts if "THE OTHER SECTIONS" in p]
    assert writing, "no section was told about the others"
    assert "do not duplicate" in writing[0].lower()


# ── announcing ───────────────────────────────────────────────────────────────

def test_the_announcement_says_what_was_produced():
    engine = DeepResearchEngine(_Agent())
    paper = asyncio.run(engine.write("neural interfaces", "technical", "brief"))
    said = paper.announcement()
    assert "Research finished" in said
    assert "neural interfaces" in said
    # "a 49-page report" — singular in the compound adjective, which is correct
    # English and was worth keeping over a test that wanted "pages".
    assert "-page" in said and "words" in said
    assert "section" in said


def test_the_announcement_says_where_it_went():
    """A fifty-page document that appears silently in a folder may as well not
    exist."""
    engine = DeepResearchEngine(_Agent())
    paper = asyncio.run(engine.write("a topic", "systematic", "brief"))
    with tempfile.TemporaryDirectory() as tmp:
        paper.docx_path, _ = research_docx.write_paper(paper, Path(tmp))
        assert paper.docx_path is not None
        assert paper.docx_path.name in paper.announcement()


def test_the_dispatcher_announces_out_loud():
    import inspect

    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    source = inspect.getsource(KnowledgeDispatchMixin._deep_research)
    assert "speak_request.emit" in source


def test_the_dispatcher_files_it_into_memory():
    """'everything he writes he must be able to remember'."""
    import inspect

    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    source = inspect.getsource(KnowledgeDispatchMixin._file_research)
    assert "ingest_record" in source or "remember" in source


def test_it_is_filed_section_by_section():
    """A single 25,000-word record can only be retrieved whole, which is the
    same as not being retrievable."""
    import inspect

    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    source = inspect.getsource(KnowledgeDispatchMixin._file_research)
    assert "for section in paper.sections" in source


# ── Word output ──────────────────────────────────────────────────────────────

def _paper() -> Paper:
    paper = Paper(topic="A Test Topic", style=Style.TECHNICAL, depth=Depth.BRIEF)
    paper.sections = [
        Section(1, "First", "b", "## A subheading\n\nSome prose here.\n\nMore prose.", 6),
        Section(2, "Second", "b", "- one\n- two", 2),
    ]
    paper.sources = [{"title": "A source", "url": "https://example.org"}]
    return paper


@pytest.mark.skipif(not research_docx.available(), reason="python-docx absent")
def test_a_real_word_document_is_written():
    with tempfile.TemporaryDirectory() as tmp:
        path, note = research_docx.write_paper(_paper(), Path(tmp))
        assert path is not None and path.suffix == ".docx"
        assert path.stat().st_size > 5000
        assert note == ""


@pytest.mark.skipif(not research_docx.available(), reason="python-docx absent")
def test_the_document_uses_real_heading_styles():
    """Fifty pages is only navigable with real Heading styles — that is what
    drives Word's navigation pane. Bold text that merely looks like a heading
    does not."""
    import docx

    with tempfile.TemporaryDirectory() as tmp:
        path, _ = research_docx.write_paper(_paper(), Path(tmp))
        document = docx.Document(str(path))
        styles = {p.style.name for p in document.paragraphs}
        assert any(name.startswith("Heading") for name in styles)


@pytest.mark.skipif(not research_docx.available(), reason="python-docx absent")
def test_markdown_emphasis_does_not_leak_into_the_document():
    import docx

    paper = _paper()
    paper.sections[0].body = "This has **bold** and *italic* and `code` in it."
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = research_docx.write_paper(paper, Path(tmp))
        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
        assert "**" not in text and "`" not in text


@pytest.mark.skipif(not research_docx.available(), reason="python-docx absent")
def test_a_failed_section_is_marked_in_the_document():
    import docx

    paper = _paper()
    paper.sections[1].body = ""
    paper.sections[1].error = "the provider was unavailable"
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = research_docx.write_paper(paper, Path(tmp))
        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
        assert "could not be written" in text


def test_a_missing_library_saves_the_work_anyway(monkeypatch):
    """Losing a fifty-page run because a formatting library is absent would be
    an absurd way to fail."""
    monkeypatch.setattr(research_docx, "available", lambda: False)
    with tempfile.TemporaryDirectory() as tmp:
        path, note = research_docx.write_paper(_paper(), Path(tmp))
        assert path is not None and path.suffix == ".md"
        assert "python-docx" in note
        assert "A Test Topic" in path.read_text(encoding="utf-8")


def test_the_filename_is_safe_and_dated():
    with tempfile.TemporaryDirectory() as tmp:
        paper = _paper()
        paper.topic = 'Weird: "topic" / with \\ characters?'
        path, _ = research_docx.write_paper(paper, Path(tmp))
        assert path is not None
        for bad in '<>:"/\\|?*':
            assert bad not in path.name


# ── the PhD frame: abstract, contributions, citations ────────────────────────
#
# "make his research a lot better, more efficient, detailed and PhD level."
# The abstract is written LAST, from the finished sections, because that is the
# only point at which there is a paper to summarise — an abstract written first
# is a guess. One extra call against N section calls, so it barely moves the
# clock: more PhD, still efficient.

class _FramingAgent(_Agent):
    async def _ask(self, prompt, persona):
        self.prompts.append(prompt)
        if persona == "research planner":
            import re
            wanted = int(re.search(r"EXACTLY (\d+) sections", prompt).group(1))
            return "\n".join(f"Topic {n} :: establish point {n}"
                             for n in range(1, wanted + 1))
        if persona == "research editor":
            return ("ABSTRACT:\nThis paper examines the question and finds a clear "
                    "answer, which matters.\n\nCONTRIBUTIONS:\n- Establishes one\n"
                    "- Argues two\n- Shows three")
        return "word " * 1300


def test_the_abstract_is_written_from_the_finished_sections():
    engine = DeepResearchEngine(_FramingAgent())
    paper = asyncio.run(engine.write("a topic", "systematic", "brief"))
    assert paper.abstract
    assert "examines" in paper.abstract


def test_the_abstract_pass_runs_after_the_sections():
    """It must see finished bodies — an editor prompt with the section digest,
    sent after the writers."""
    agent = _FramingAgent()
    asyncio.run(DeepResearchEngine(agent).write("a topic", "systematic", "brief"))
    editor = [p for p in agent.prompts if "ABSTRACT:" in p and "THE PAPER:" in p]
    assert editor, "no abstract/editor pass ran"


def test_key_contributions_are_extracted():
    engine = DeepResearchEngine(_FramingAgent())
    paper = asyncio.run(engine.write("a topic", "systematic", "brief"))
    assert len(paper.contributions) == 3
    assert paper.contributions[0] == "Establishes one"


def test_the_frame_split_is_robust_to_formatting():
    from orion_core.deep_research import DeepResearchEngine as E
    abstract, contributions = E._split_frame(
        "ABSTRACT:\nA paragraph here.\n\nCONTRIBUTIONS:\n* one\n- two\n\u2022 three")
    assert abstract == "A paragraph here."
    assert contributions == ["one", "two", "three"]


def test_the_frame_pass_failing_does_not_lose_the_paper():
    class _NoEditor(_Agent):
        async def _ask(self, prompt, persona):
            if persona == "research editor":
                raise RuntimeError("provider down")
            return await super()._ask(prompt, persona)

    paper = asyncio.run(DeepResearchEngine(_NoEditor()).write("t", "systematic", "brief"))
    assert paper.words > 0          # the body survives an abstract failure
    assert paper.abstract == ""


def test_sections_are_told_to_cite_numbered_sources():
    agent = _Agent()
    asyncio.run(DeepResearchEngine(agent).write("a topic", "systematic", "brief"))
    writing = [p for p in agent.prompts if "THE OTHER SECTIONS" in p]
    assert any("[1], [2]" in p for p in writing)
    assert any("SOURCES (numbered)" in p for p in writing)


@pytest.mark.skipif(not research_docx.available(), reason="python-docx absent")
def test_the_abstract_and_contributions_render_in_the_document():
    import docx

    paper = _paper()
    paper.abstract = "A real abstract paragraph summarising the work."
    paper.contributions = ["First contribution", "Second contribution"]
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = research_docx.write_paper(paper, Path(tmp))
        document = docx.Document(str(path))
        headings = [p.text for p in document.paragraphs
                    if p.style.name.startswith("Heading")]
        assert "Abstract" in headings
        assert "Key contributions" in headings
        text = "\n".join(p.text for p in document.paragraphs)
        assert "A real abstract paragraph" in text
        assert "First contribution" in text
