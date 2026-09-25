"""
Research at the length the work actually needs.

  "His research skills must be vastly improved and faster, the documents he
   makes must be strictly in word only ... Make his analysis, summaries and
   overall research like a PhD style - there should be multiple styles of
   research that he does but they all need to be detailed, comprehensive and be
   useful - they can't just be short pieces of research and short paragraphs,
   I'm talking PAGES and PAGES like up to 50 pages in research. Research when
   done must be explicitly announced."

The gap was not architectural. ``research.py`` already gathered sources, wrote
sections and saved a paper — it was simply *instructed* to be short:

    research.py:231   "concise ... 150-250 words"
    research.py:241   "150-300 words; British English"
    research.py:248   "executive briefing (200 words)"

Six sections at that length is roughly 1,500 words — three pages. Fifty pages
is about 25,000 words, so raising the numbers alone was never going to work:
a model asked for 25,000 words in one call produces something thin and
repetitive that has been stretched rather than developed.

How length is actually reached
------------------------------
By decomposition, not by asking harder. A real outline is planned first — the
sections a specialist would actually use for THIS topic, not a fixed
Abstract/Introduction/Conclusion skeleton — and each section is then written as
its own substantial piece with its own sources and its own brief. Twenty
sections of 1,300 words each is 26,000 words, and every one of them is about
something specific.

That decomposition is also what makes it FASTER rather than slower, which is
the part that looks contradictory. Sections do not depend on each other, so
they are written concurrently; wall-clock time is set by the slowest section
plus the planning pass, not by the sum of them. A 40-section paper written
eight-at-a-time finishes in roughly the time nine sequential sections would.

Styles
------
"Multiple styles" are not cosmetic variations. Each carries a different
outline shape, a different voice, and a different standard of evidence, because
a literature review and a technical deep-dive are different *kinds of thinking*
and a paper that muddles them is useless as either.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .constants import BASE_DIR
from .data import ToolResult
from .utils import first_line, utc_stamp

#: Words per page of a typical Word document at 12pt, 1.15 spacing. Used only
#: to translate the user's "pages" into a target the engine can aim at.
WORDS_PER_PAGE = 500

#: How many sections are written at once. Bounded because each is a model call:
#: unbounded concurrency here would trip provider rate limits (the 429s that
#: have bitten this project before) and make the whole run slower, not faster.
MAX_PARALLEL_SECTIONS = 6
#: How long one research call may wait for a rate-limited provider to recover
#: (ProviderRouter.generate_text patience) before the section is given up.
RESEARCH_PATIENCE_S = 90.0


class Depth(str, Enum):
    """How long the finished document should be."""

    BRIEF = "brief"            # ~5 pages   — a considered answer
    STANDARD = "standard"      # ~15 pages  — a proper report
    THESIS = "thesis"          # ~35 pages  — a dissertation chapter
    EXHAUSTIVE = "exhaustive"  # ~50 pages  — everything he can establish


#: (target pages, section count, words per section).
#:
#: sections x words must land within about 10% of pages x WORDS_PER_PAGE, and
#: there is a test asserting exactly that. The first version did not: 38
#: sections of 1,300 words is 49,400 words, which is ninety-eight pages, not
#: the fifty it promised — ORION would have announced "a 50-page report" and
#: handed over something twice that. A target nobody checks is a wish.
#:
#: Words-per-section stays in the 850-1,350 band: long enough to develop an
#: argument, short enough that a model stays specific instead of padding.
DEPTHS: dict[Depth, tuple[int, int, int]] = {
    Depth.BRIEF:      (5,   3,  850),    #  2,550 words ~  5 pages
    Depth.STANDARD:   (15,  6, 1250),    #  7,500 words ~ 15 pages
    Depth.THESIS:     (35, 14, 1250),    # 17,500 words ~ 35 pages
    Depth.EXHAUSTIVE: (50, 19, 1300),    # 24,700 words ~ 49 pages
}


class Style(str, Enum):
    """What KIND of research this is. Each is a different way of thinking."""

    LITERATURE_REVIEW = "literature_review"
    TECHNICAL = "technical"
    COMPARATIVE = "comparative"
    CRITICAL = "critical"
    SYSTEMATIC = "systematic"
    PRIMER = "primer"


@dataclass(frozen=True)
class StyleSpec:
    label: str
    outline_brief: str
    section_voice: str
    evidence: str


STYLES: dict[Style, StyleSpec] = {
    Style.LITERATURE_REVIEW: StyleSpec(
        "Literature review",
        "Organise by THEME and by school of thought, not chronologically and "
        "not one-source-per-section. Sections should name the debates, who "
        "holds which position, and where the field genuinely disagrees.",
        "Survey what has been argued, attribute positions to their proponents, "
        "and make the disagreements explicit rather than averaging them away.",
        "Attribute every claim. Where sources conflict, say so and say why.",
    ),
    Style.TECHNICAL: StyleSpec(
        "Technical deep-dive",
        "Organise by MECHANISM: how the thing actually works, layer by layer, "
        "from the substrate upward. Include failure modes, constraints and "
        "trade-offs as their own sections.",
        "Explain mechanism precisely. Prefer concrete numbers, equations and "
        "worked examples over description. State assumptions explicitly.",
        "Quantify wherever a quantity exists. Name the limits of each claim.",
    ),
    Style.COMPARATIVE: StyleSpec(
        "Comparative study",
        "Organise by DIMENSION OF COMPARISON, not by subject — one section per "
        "axis along which the alternatives genuinely differ, so like is "
        "compared with like.",
        "Compare on the stated axis only, hold everything else constant, and "
        "reach a judgement rather than listing properties.",
        "Every comparison needs a criterion stated before the verdict.",
    ),
    Style.CRITICAL: StyleSpec(
        "Critical appraisal",
        "Organise by CLAIM: take the strongest version of each position, then "
        "test it. Include a section on what would have to be true for the "
        "opposing view to be right.",
        "Steelman before you critique. Separate what is established from what "
        "is inferred from what is asserted.",
        "Distinguish evidence from argument from opinion, explicitly.",
    ),
    Style.SYSTEMATIC: StyleSpec(
        "Systematic analysis",
        "Organise by a stated method: scope, inclusion criteria, evidence "
        "gathered, synthesis, limitations, conclusions. The method section is "
        "not a formality — state what would have changed the conclusion.",
        "Be methodical and explicit about procedure. Report negative findings "
        "as carefully as positive ones.",
        "State the evidence base and its gaps before drawing conclusions.",
    ),
    Style.PRIMER: StyleSpec(
        "Teaching primer",
        "Organise by LEARNING ORDER: each section should only depend on what "
        "came before it. Build from first principles to working competence, "
        "with worked examples and common misconceptions called out.",
        "Teach. Define terms on first use, work examples in full, and name the "
        "mistakes a learner actually makes at this point.",
        "Every abstraction gets a concrete example immediately after it.",
    ),
}


@dataclass
class Section:
    number: int
    title: str
    brief: str
    body: str = ""
    words: int = 0
    error: str = ""


@dataclass
class Paper:
    topic: str
    style: Style
    depth: Depth
    sections: list[Section] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    abstract: str = ""
    contributions: list[str] = field(default_factory=list)
    docx_path: Path | None = None
    seconds: float = 0.0

    @property
    def words(self) -> int:
        return sum(section.words for section in self.sections)

    @property
    def pages(self) -> int:
        return max(1, round(self.words / WORDS_PER_PAGE))

    @property
    def failed(self) -> list[Section]:
        return [s for s in self.sections if s.error]

    def announcement(self) -> str:
        """What ORION says out loud when it is done.

        Explicitly announced, because a fifty-page document that appears
        silently in a folder may as well not exist. Says what it IS and where
        it went — the two things needed to act on it.
        """
        where = f" It's saved as {self.docx_path.name}." if self.docx_path else ""
        note = ""
        if self.failed:
            note = (f" {len(self.failed)} section(s) came back empty and are "
                    "marked in the document.")
        if not self.sources:
            # Said plainly, because it changes what the document is: without a
            # single page read, it is the model's general knowledge in a
            # paper's clothing, and nothing in it has been checked.
            note += (" I couldn't read any sources this time — every search "
                     "route was unavailable — so it's written from general "
                     "knowledge and should be treated as unverified.")
        return (f"Research finished on {self.topic}. "
                f"I've written a {self.pages}-page "
                f"{STYLES[self.style].label.lower()} — {len(self.sections)} "
                f"sections, {self.words:,} words, {len(self.sources)} "
                f"source(s).{where}{note}")


def resolve_depth(value: Any) -> Depth:
    """Accept a Depth, a name, or a page count ('50 pages', 50)."""
    if isinstance(value, Depth):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return Depth.STANDARD
    for depth in Depth:
        if depth.value == text:
            return depth
    match = re.search(r"(\d+)", text)
    if match:
        pages = int(match.group(1))
        # Nearest tier at or above the request, so "40 pages" does not quietly
        # become a 15-page report.
        for depth in (Depth.BRIEF, Depth.STANDARD, Depth.THESIS, Depth.EXHAUSTIVE):
            if DEPTHS[depth][0] >= pages:
                return depth
        return Depth.EXHAUSTIVE
    return Depth.STANDARD


def resolve_style(value: Any) -> Style:
    if isinstance(value, Style):
        return value
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    for style in Style:
        if style.value == text:
            return style
    aliases = {
        "review": Style.LITERATURE_REVIEW, "literature": Style.LITERATURE_REVIEW,
        "deep_dive": Style.TECHNICAL, "deepdive": Style.TECHNICAL,
        "engineering": Style.TECHNICAL,
        "compare": Style.COMPARATIVE, "comparison": Style.COMPARATIVE,
        "critique": Style.CRITICAL, "appraisal": Style.CRITICAL,
        "method": Style.SYSTEMATIC, "meta_analysis": Style.SYSTEMATIC,
        "teach": Style.PRIMER, "tutorial": Style.PRIMER, "learn": Style.PRIMER,
        "explainer": Style.PRIMER,
    }
    return aliases.get(text, Style.SYSTEMATIC)


#: The system instruction for every research model call. Deliberately NOT the
#: conversational persona: that carries the whole tool map and a "sir" habit,
#: which is how outlines came back as "Certainly, sir. Here are six sections".
RESEARCH_INSTRUCTION = (
    "You are ORION's research engine: a meticulous postgraduate researcher "
    "writing in British English. Follow the requested output format exactly. "
    "Never address the reader, never add preamble or sign-off, and never "
    "invent citations — cite only the numbered sources you are given.")

#: How much reading each depth gets before a word is written:
#: (questions, pages per question, rounds of follow-up, minutes allowed).
READING: dict[Depth, tuple[int, int, int, float]] = {
    Depth.BRIEF:      (3, 2, 1, 2.0),
    Depth.STANDARD:   (4, 3, 2, 4.0),
    Depth.THESIS:     (6, 3, 2, 6.0),
    Depth.EXHAUSTIVE: (8, 3, 2, 8.0),
}

#: Output ceilings. A section of ~1,300 words is ~1,800 tokens; asking for the
#: conversational default (2,048) left no headroom and clipped long sections.
SECTION_TOKENS = 3600
PLAN_TOKENS = 1500


#: Sources each section is given when they can be chosen by meaning.
SOURCES_PER_SECTION = 8


def _source_line(number: int, source: dict[str, str]) -> str:
    return (f"[{number}] {source.get('title', '')} ({source.get('url', '')}): "
            f"{str(source.get('snippet', ''))[:600]}")


def _sources_per_section(sections: list["Section"],
                         sources: list[dict[str, str]]) -> dict[int, str]:
    """section number -> the source block for that section; {} to use all.

    Picks the SOURCES_PER_SECTION sources whose notes are closest in meaning
    to the section's title and brief, keeping their paper-wide numbers (in
    numeric order) so an inline [n] still points at the right reference.
    Empty when the encoder is cold or there are too few sources to choose.
    """
    try:
        from . import semantic

        if len(sources) <= SOURCES_PER_SECTION or not semantic.ENCODER.ready():
            return {}
        source_vecs = semantic.ENCODER.encode(
            [f"{s.get('title', '')}. {str(s.get('snippet', ''))[:600]}" for s in sources])
        section_vecs = semantic.ENCODER.encode(
            [f"{s.title}. {s.brief}" for s in sections])
        if source_vecs is None or section_vecs is None:
            return {}
        out: dict[int, str] = {}
        for index, section in enumerate(sections):
            sims = source_vecs @ section_vecs[index]
            chosen = sorted(sorted(range(len(sources)), key=lambda i: -float(sims[i]))
                            [:SOURCES_PER_SECTION])
            out[section.number] = "\n".join(_source_line(i + 1, sources[i]) for i in chosen)
        return out
    except Exception:
        return {}


class DeepResearchEngine:
    """Plans a real outline, then writes every section properly and in parallel.

    Composes with the existing ResearchAgent rather than replacing it: the
    model router and the memory hook are already solved there. Evidence now
    comes from LiveResearcher — pages actually searched for, opened and read,
    each step announced — rather than an encyclopaedia summary and headlines.
    """

    def __init__(self, agent: Any, bus: Any = None, router: Any = None,
                 on_event: Callable[[dict], None] | None = None,
                 live_reading: bool | None = None) -> None:
        self.agent = agent
        self.bus = bus if bus is not None else getattr(agent, "bus", None)
        self.router = router if router is not None else getattr(agent, "router", None)
        self.on_event = on_event
        # Reading the live web needs a real model router behind it (every
        # note is a model call). A stub agent without one — the test suite —
        # uses the agent's own gathering instead of touching the network.
        self.live_reading = (self.router is not None) if live_reading is None else live_reading
        self.last: Paper | None = None

    # ── logging ──────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(message)
            except Exception:
                pass

    def _step(self, kind: str, message: str, **extra: Any) -> None:
        """Announce a stage on the research console as well as the log."""
        self._log(f"RESEARCH: {message}")
        payload = {"kind": kind, "message": message, "at": time.time(), **extra}
        if self.bus is not None:
            try:
                self.bus.dashboard_event.emit("research_step", payload)
            except Exception:
                pass
        if self.on_event is not None:
            try:
                self.on_event(payload)
            except Exception:
                pass

    # ── the outline ──────────────────────────────────────────────────────────

    async def plan(self, topic: str, style: Style, depth: Depth,
                   source_context: str = "") -> list[Section]:
        """Ask for an outline built for THIS topic, then parse it strictly.

        A fixed skeleton is what kept the old papers short and generic: six
        headings that fit anything fit nothing well. The outline is where the
        length actually comes from, so it is planned before a word is written.
        """
        _pages, wanted, words = DEPTHS[depth]
        spec = STYLES[style]
        prompt = (
            f"Plan the section structure of a {spec.label.lower()} on: {topic}\n\n"
            f"{spec.outline_brief}\n\n"
            f"Produce EXACTLY {wanted} sections. Each will be written as a "
            f"substantial {words}-word piece, so each must be narrow enough to "
            f"sustain that length without padding, and must not overlap with "
            f"the others.\n"
            "The method above describes HOW to think, not what to call the "
            "sections. Every title must name this topic's actual subject matter "
            "(e.g. 'Dendrite growth at the lithium-metal interface'), never a "
            "generic label such as 'Scope', 'Background', 'Inclusion criteria' "
            "or 'Evidence gathered'. With fewer sections than the method has "
            "stages, merge stages — the last section must still carry the "
            "findings and conclusions.\n"
            f"{('Context from sources:' + chr(10) + source_context) if source_context else ''}\n\n"
            "Return one section per line, in this exact format and nothing else "
            "(no numbering, no bold, no preamble):\n"
            "Section title :: a one-sentence brief saying specifically what this "
            "section must establish\n"
        )
        self._step("planning", f"planning the outline — {wanted} sections")
        sections: list[Section] = []
        for attempt in range(2):
            try:
                raw = await self._ask(prompt, "research planner", max_tokens=PLAN_TOKENS)
            except Exception as exc:
                self._log(f"RESEARCH: the planning call failed - {first_line(exc, 120)}")
                continue
            parsed = self._parse_outline(raw, wanted)
            if len(parsed) >= max(2, wanted // 2):
                sections = parsed
                generic = [s.title for s in parsed if self._is_generic_title(s.title)]
                if attempt or len(generic) * 2 <= len(parsed):
                    break
                # A method template copied as headings ("Scope", "Inclusion
                # criteria", "Evidence gathered") — a brief systematic paper
                # came out exactly like that, with no findings section at all.
                prompt += ("\nThese titles are generic method labels, not this "
                           f"topic's subject matter: {'; '.join(generic)}. Name "
                           "what each section is actually about.\n")
                continue
            prompt += ("\nYour previous answer did not follow the format. Use "
                       "exactly one line per section: Title :: brief\n")
        if not sections:
            # A refusal to guess. Inventing a generic outline here is how the
            # old version produced papers that were the same shape regardless
            # of subject.
            self._log("RESEARCH: the outline could not be parsed — using a "
                      "structural fallback so the run still produces something.")
            sections = self._fallback_outline(topic, spec, wanted)
        for section in sections:
            self._step("outline", f"section {section.number}: {section.title}",
                       number=section.number, title=section.title)
        return sections

    #: Words a heading made only of says nothing about its subject.
    _GENERIC_WORDS = frozenset({
        "scope", "background", "introduction", "inclusion", "exclusion",
        "criteria", "evidence", "gathered", "synthesis", "limitations",
        "limitation", "conclusion", "conclusions", "method", "methods",
        "methodology", "overview", "discussion", "results", "findings",
        "analysis", "definitions", "context", "implications", "summary",
        "and", "of", "the", "current", "understanding", "key", "further",
    })

    @classmethod
    def _is_generic_title(cls, title: str) -> bool:
        words = re.findall(r"[a-z]+", title.lower())
        return bool(words) and len(words) <= 4 and all(w in cls._GENERIC_WORDS for w in words)

    #: A leading label the model sometimes copies from the format line.
    _LABEL = re.compile(r"^(?:section\s*(?:title)?|title)\s*(?:\d+)?\s*(?:::|:|-)\s*",
                        re.IGNORECASE)

    @classmethod
    def _parse_outline(cls, raw: str, wanted: int) -> list[Section]:
        """Title/brief pairs, whatever list dressing the model wrapped them in.

        Tolerates "1. **Title ::** brief", "TITLE :: Title :: brief", and the
        "Title: brief" / "Title — brief" forms models fall back to.
        """
        sections: list[Section] = []
        for line in str(raw or "").splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\s\-\*•\d\.\)#]+", "", line).replace("**", "").strip()
            line = cls._LABEL.sub("", line)
            if "::" in line:
                title, _, brief = line.partition("::")
            else:
                match = re.match(r"^(.{3,120}?)\s*(?:—|–|:| - )\s+(.{12,})$", line)
                if not match:
                    continue
                title, brief = match.group(1), match.group(2)
            title = title.strip(" #*:_\"'")
            brief = brief.strip(" *:_").strip()
            if not title or len(title) > 160 or title.lower().startswith(
                    ("here are", "certainly", "sure", "below")):
                continue
            sections.append(Section(len(sections) + 1, title[:160], brief[:400]))
            if len(sections) >= wanted:
                break
        return sections

    @staticmethod
    def _fallback_outline(topic: str, spec: StyleSpec, wanted: int) -> list[Section]:
        base = ["Scope and definitions", "Background", "Current understanding",
                "Mechanisms and evidence", "Points of disagreement",
                "Practical implications", "Limitations", "Conclusions"]
        out: list[Section] = []
        for index in range(wanted):
            title = base[index] if index < len(base) else f"Further analysis {index - len(base) + 2}"
            out.append(Section(index + 1, title,
                               f"Cover {title.lower()} for {topic}."))
        return out

    # ── writing ──────────────────────────────────────────────────────────────

    async def _write_section(self, topic: str, section: Section, style: Style,
                             words: int, source_context: str,
                             outline_context: str) -> Section:
        spec = STYLES[style]
        prompt = (
            f"Write section {section.number} of a {spec.label.lower()} on: {topic}\n\n"
            f"SECTION: {section.title}\n"
            f"THIS SECTION MUST ESTABLISH: {section.brief}\n\n"
            f"VOICE: {spec.section_voice}\n"
            f"EVIDENCE: {spec.evidence}\n\n"
            f"Write approximately {words} words of finished prose in British "
            f"English, at postgraduate level. Develop the argument in depth — "
            f"do not summarise, do not write an overview, and do not repeat the "
            f"section title back as an opening sentence.\n"
            f"Use subheadings where the material genuinely divides. Do not "
            f"write a conclusion for the whole paper; this is one section of "
            f"many.\n\n"
            f"THE OTHER SECTIONS (do not duplicate their material):\n"
            f"{outline_context}\n\n"
            + ("Cite the numbered sources inline as [1], [2] where they support "
               "a specific claim — sparingly and only where they genuinely do, "
               "never decoratively.\n" if source_context else "")
            + (("SOURCES (numbered):\n" + source_context) if source_context else "")
        )
        self._step("writing", f"writing section {section.number}: {section.title}",
                   number=section.number, title=section.title)
        try:
            body = await self._ask(prompt, "research writer", max_tokens=SECTION_TOKENS)
        except Exception as exc:
            section.error = first_line(exc, 120)
            self._log(f"RESEARCH: section {section.number} failed - {section.error}")
            return section
        body = str(body or "").strip()
        if not body:
            section.error = "the model returned nothing"
            return section
        section.body = body
        section.words = len(body.split())
        return section

    async def write(self, topic: str, style: Any = Style.SYSTEMATIC,
                    depth: Any = Depth.STANDARD,
                    on_progress: Callable[[int, int], None] | None = None) -> Paper:
        """Plan, then write every section concurrently, then assemble."""
        style = resolve_style(style)
        depth = resolve_depth(depth)
        pages, count, words = DEPTHS[depth]
        started = time.monotonic()
        self._step("paper_start", f"starting a {STYLES[style].label.lower()} on "
                   f"'{topic}' — targeting ~{pages} pages across {count} sections.",
                   topic=topic, pages=pages, sections=count)

        sources = await self._sources(topic, depth)
        if not sources:
            self._step("unsourced", "no page could be read — every search route was "
                                    "unavailable; writing from general knowledge, "
                                    "which the paper will say is unverified")
        # Numbered so an inline [1]/[2] in a section maps to the reference list.
        # Notes from pages actually read are long enough to write from; a bare
        # search snippet is not, so each entry keeps up to ~600 characters.
        source_context = "\n".join(
            f"[{i}] {s.get('title','')} ({s.get('url','')}): "
            f"{str(s.get('snippet',''))[:600]}"
            for i, s in enumerate(sources[:24], 1))

        sections = await self.plan(topic, style, depth, source_context)
        outline_context = "\n".join(f"{s.number}. {s.title}" for s in sections)
        # Each section sees the sources that bear on IT (by meaning, keeping
        # the paper-wide numbers so [n] still maps to the reference list)
        # rather than all two dozen: focused citations, a third of the tokens.
        focused = await asyncio.to_thread(_sources_per_section, sections, sources[:24])
        self._step("planned", f"outline planned — {len(sections)} sections.",
                   sections=len(sections))

        # Concurrent, but bounded: each section is a model call, and unbounded
        # parallelism trips provider rate limits and ends up SLOWER.
        gate = asyncio.Semaphore(MAX_PARALLEL_SECTIONS)
        done = 0

        async def _one(section: Section) -> Section:
            nonlocal done
            async with gate:
                result = await self._write_section(
                    topic, section, style, words,
                    focused.get(section.number, source_context), outline_context)
            done += 1
            if on_progress is not None:
                try:
                    on_progress(done, len(sections))
                except Exception:
                    pass
            self._step("section_done",
                       f"{done}/{len(sections)} sections written "
                       f"({sum(s.words for s in sections):,} words so far).",
                       done=done, total=len(sections))
            return result

        written = await asyncio.gather(*(_one(s) for s in sections),
                                       return_exceptions=False)
        paper = Paper(topic=topic, style=style, depth=depth,
                      sections=list(written), sources=sources,
                      seconds=time.monotonic() - started)
        # The scholarly frame, written LAST — because that is the only point at
        # which there is a finished paper to summarise. A PhD abstract written
        # first is a guess; written from the actual sections it is a real
        # précis. One extra call against N section calls, so it barely moves the
        # clock — "more PhD, still efficient".
        self._step("framing", "writing the abstract and key contributions")
        await self._frame(paper)
        self.last = paper
        self._step("finished", f"finished '{topic}' — {paper.words:,} words, "
                   f"~{paper.pages} pages, {len(sources)} source(s), in "
                   f"{paper.seconds:.0f}s.", words=paper.words, pages=paper.pages)
        return paper

    async def _frame(self, paper: Paper) -> None:
        """Add the abstract and key contributions from the finished sections."""
        digest = "\n".join(
            f"{s.number}. {s.title}: {s.body[:260]}"
            for s in paper.sections if s.body)[:6000]
        if not digest:
            return
        prompt = (
            f"Below is a finished {STYLES[paper.style].label.lower()} on "
            f"'{paper.topic}', section by section. Write, in British English:\n\n"
            "ABSTRACT: a single 180-220 word paragraph that states the question, "
            "the approach, the substantive findings and their significance — a "
            "real postgraduate abstract, not a table of contents in prose.\n"
            "CONTRIBUTIONS: 3-6 bullet points, each a specific thing this paper "
            "establishes or argues, not a description of what it 'discusses'.\n\n"
            "Return exactly:\n"
            "ABSTRACT:\n<the paragraph>\n\nCONTRIBUTIONS:\n- <point>\n- <point>\n\n"
            f"THE PAPER:\n{digest}"
        )
        try:
            raw = await self._ask(prompt, "research editor")
        except Exception as exc:
            self._log(f"RESEARCH: abstract pass skipped - {first_line(exc, 100)}")
            return
        paper.abstract, paper.contributions = self._split_frame(str(raw or ""))

    @staticmethod
    def _split_frame(raw: str) -> tuple[str, list[str]]:
        abstract, contributions = "", []
        upper = raw.upper()
        if "CONTRIBUTIONS:" in upper:
            cut = upper.index("CONTRIBUTIONS:")
            head, tail = raw[:cut], raw[cut + len("CONTRIBUTIONS:"):]
        else:
            head, tail = raw, ""
        abstract = head.split(":", 1)[-1].strip() if head.upper().startswith("ABSTRACT") \
            else head.strip()
        abstract = abstract.lstrip(": ").strip()
        for line in tail.splitlines():
            line = line.strip().lstrip("-*•").strip()
            if line:          # any non-empty bullet; a short one is still valid
                contributions.append(line)
        return abstract, contributions[:6]

    # ── collaborators ────────────────────────────────────────────────────────

    async def _sources(self, topic: str, depth: Depth = Depth.STANDARD) -> list[dict[str, str]]:
        """Evidence from pages actually read, each with the URL it came from.

        LiveResearcher searches, opens every result, reads it and takes a note
        of what THAT page says — announcing each step on the log and the
        research console as it goes, so the user can watch the work rather
        than wait for a document. Falls back to the agent's quick gathering
        (encyclopaedia summary + headlines) only when nothing could be read.
        """
        questions, per_question, rounds, minutes = READING[depth]
        sources: list[dict[str, str]] = []
        try:
            if not self.live_reading:
                raise LookupError("live reading is off")
            from .live_research import LiveResearcher

            researcher = LiveResearcher(
                ask=lambda prompt, persona="research reader":
                    self._ask(prompt, persona, max_tokens=900),
                bus=self.bus, on_event=self.on_event)
            run = await researcher.investigate(
                topic, questions=questions, per_question=per_question,
                rounds=rounds, deadline_s=minutes * 60.0)
            for note in run.notes:
                if "does not really address" in note.note.lower():
                    continue
                sources.append({"title": note.title or note.url, "url": note.url,
                                "snippet": note.note})
        except LookupError:
            pass
        except Exception as exc:
            self._log(f"RESEARCH: live reading failed - {first_line(exc, 120)}")
        if sources:
            return sources
        getter = getattr(self.agent, "_gather_sources", None) or \
            getattr(self.agent, "gather_sources", None)
        if getter is None:
            return []
        try:
            return list(await getter(topic) or [])
        except Exception as exc:
            self._log(f"RESEARCH: source gathering failed - {first_line(exc, 100)}")
            return []

    async def _ask(self, prompt: str, persona: str, max_tokens: int | None = None) -> str:
        """One model call that FAILS when the model does.

        The research agent's own _ask turns every failure into placeholder
        prose ("[Model unavailable — …]"), which is right for a chat reply and
        disastrous here: the placeholder was parsed as an outline and written
        into sections as if it were findings.
        """
        router = self.router
        if router is not None and hasattr(router, "generate_text"):
            kwargs = dict(instruction=f"{RESEARCH_INSTRUCTION}\nRole: {persona}.",
                          task="research", max_tokens=max_tokens)
            try:
                # Research runs in the background: waiting a minute for a
                # rate-limited provider beats failing every section at once.
                _profile, text = await router.generate_text(
                    prompt, patience_s=RESEARCH_PATIENCE_S, **kwargs)
            except TypeError:
                _profile, text = await router.generate_text(prompt, **kwargs)
            return str(text or "")
        ask = getattr(self.agent, "_ask", None)
        if ask is None:
            raise RuntimeError("no model access on the research agent")
        text = str(await ask(prompt, persona) or "")
        if text.startswith("[Model unavailable"):
            raise RuntimeError(text.strip("[]")[:160])
        return text


__all__ = [
    "DEPTHS", "MAX_PARALLEL_SECTIONS", "STYLES", "WORDS_PER_PAGE",
    "DeepResearchEngine", "Depth", "Paper", "Section", "Style", "StyleSpec",
    "resolve_depth", "resolve_style",
]
