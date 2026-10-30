"""
ResearchAgent — autonomous, time-boxed research and research-paper authoring.

Two capabilities:

    conduct(topic, minutes)  — ORION researches a topic on his own for a set
                               period (e.g. "research this for 30 minutes while
                               I'm away").  He decomposes the topic into
                               sub-questions, gathers sources (online) or draws
                               on local knowledge (offline), synthesises a note
                               per sub-question, and writes everything into an
                               organised, dated project folder.  Progress is
                               reported on the bus so the dashboard shows it.

    write_paper(topic)       — produces a structured research paper (title,
                               abstract, sections, conclusion, references) into
                               the same organised folder layout.

Folder layout (under <workspace>/research/):

    research/
      2026-07-03_dropshipping-trends/
        README.md            ← overview + status
        notes/               ← one markdown note per sub-question
        paper/paper.md       ← the assembled paper (when requested)
        sources.md           ← gathered sources with links

Works in both modes: online it fetches Wikipedia + news sources; offline it
synthesises from the local model, memory and knowledge packs.  All model calls
go through the ProviderRouter, so cloud or local is chosen automatically.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, quote_plus

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from .lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")

from .bus import OrionBus
from .constants import BASE_DIR
from .data import ToolResult
from .security import SecuritySanitiser
from .utils import first_line, utc_stamp


#: Headlines whose article text is fetched for the snippet, the bound on that
#: excerpt, and how long one article read may take (they run concurrently).
NEWS_EXCERPTS = 3
NEWS_EXCERPT_CHARS = 700
NEWS_EXCERPT_TIMEOUT_S = 8.0

#: Search results opened and read for each question (Mark XXXX.9). Until
#: 30 Oct 2026 the timed run and the dossier never opened one: "11 sources via
#: the built-in encyclopaedia and news lookup" in the owner's log were search
#: snippets and Google News headlines, so a note could only restate a snippet
#: or the model's own memory. Page reads are network, not model calls - the
#: free-tier quota is untouched.
READ_PAGES = 3
#: Candidates tried for those reads: a refused page hands its slot on.
READ_CANDIDATES = 6
READ_TIMEOUT_S = 10.0
#: Sources put in front of the writer for one note, pages first.
NOTE_SOURCES = 7
#: The whole executive-briefing prompt, bounded (notes are cut to fit).
SUMMARY_PROMPT_CHARS = 12_500
#: Hosts whose links are redirects to a script page, never the article.
_UNREADABLE_HOSTS = ("news.google.com",)


def _numbered(sources: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """(number, source): the number the caller assigned ("n"), else 1..k."""
    out = []
    for index, source in enumerate(sources, 1):
        try:
            number = int(source.get("n") or index)
        except (TypeError, ValueError):
            number = index
        out.append((number, source))
    return out


def _source_line(number: int, source: dict[str, Any]) -> str:
    from .live_research import EVIDENCE_LABELS, evidence_kind
    label = EVIDENCE_LABELS[evidence_kind(source)]
    where = f" — {source['url']}" if source.get("url") else ""
    return f"[{number}] ({label}) {source.get('title') or 'Untitled'}{where}"


def _accepts(function: Any, count: int = 0, keyword: str = "") -> bool:
    """Whether *function* takes *count* positional arguments / *keyword*.
    Callers stub these methods with narrower signatures than the real ones."""
    import inspect
    try:
        params = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_KEYWORD for p in params) and keyword:
        return True
    if keyword:
        return any(p.name == keyword for p in params)
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    return len([p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= count


def _plain(markup: str) -> str:
    """RSS description HTML as plain text."""
    import html
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(markup or "")))
    return " ".join(html.unescape(text).split())


def _shown(folder: Path) -> str:
    """The folder as the owner hears it: relative to ORION's own folder when it
    is inside it, otherwise in full. A research root set elsewhere (another
    drive, the owner's documents, a test's temporary folder) made
    ``relative_to`` raise AFTER the research task had started, so the owner was
    told it failed while it ran on."""
    try:
        return str(folder.relative_to(BASE_DIR))
    except ValueError:
        return str(folder)


class ResearchAgent:
    RESEARCH_ROOT = BASE_DIR / "research"

    def __init__(self, bus: OrionBus, router: Any, memory: Any,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.router = router
        self.memory = memory
        self.telemetry = telemetry
        self._active: dict[str, Any] = {}
        # Phase 3: awaited by the ResearchDirector when a run finishes, so the
        # persistent agenda can harvest findings and advance to the next topic.
        self.on_complete: Any = None  # async (topic: str, folder: Path) -> None
        self.on_settled: Any = None  # async (topic, folder, actual ToolResult), after task.done()

    # ── public API ────────────────────────────────────────────────────────────

    def start_research(self, topic: str, minutes: float = 30.0, *,
                       _accepted: asyncio.Event | None = None) -> ToolResult:
        """Kick off a background research run and return immediately."""
        topic = SecuritySanitiser.guard_text(str(topic or "").strip(), "research.topic")
        if not topic:
            return ToolResult("What should I research?", ok=False)
        if self._active.get("task") and not self._active["task"].done():
            return ToolResult(
                f"I'm already researching '{self._active.get('topic')}'. "
                "Let me finish that first, or say 'stop research'.",
                ok=False,
            )
        minutes = max(1.0, min(180.0, float(minutes)))
        folder = self._make_folder(topic)
        from . import background
        async def run() -> ToolResult:
            if _accepted is not None:
                await _accepted.wait()
            return await self._run(topic, minutes, folder)

        task = background.spawn(run(), name="orion-timed-research")
        if task is None:
            return ToolResult("Research could not start because the event loop is closing.", ok=False)
        self._active = {"topic": topic, "folder": folder, "task": task,
                        "started": time.monotonic(), "minutes": minutes,
                        "state": "running", "result": None}

        def settled(done: asyncio.Task) -> None:
            async def report() -> None:
                if done.cancelled():
                    result = ToolResult(f"Research on '{topic}' was cancelled.", ok=False,
                                        receipt={"research_state": "cancelled", "file_paths": [str(folder)]})
                else:
                    try:
                        result = done.result()
                    except Exception as exc:
                        result = ToolResult(f"Research on '{topic}' failed: {first_line(exc, 160)}", ok=False,
                                            receipt={"research_state": "failed"})
                    if not isinstance(result, ToolResult):
                        result = ToolResult("Research ended without a recorded outcome.", ok=False)
                if callable(self.on_settled):
                    await self.on_settled(topic, folder, result)
                elif result.ok and callable(self.on_complete):
                    await self.on_complete(topic, folder)

            background.spawn(report(), name="orion-research-settled")

        task.add_done_callback(settled)
        return ToolResult(
            f"Very good. I'll research '{topic}' for about {int(minutes)} minutes "
            f"and organise my findings in {_shown(folder)}. "
            "Go ahead — I'll have it ready when you return.",
            receipt={"research_state": "running", "topic": topic, "file_paths": []},
        )

    def stop_research(self) -> ToolResult:
        task = self._active.get("task")
        if task and not task.done():
            task.cancel()
            return ToolResult(f"Stopping research on '{self._active.get('topic')}'.")
        return ToolResult("No research is currently running.")

    def status(self) -> ToolResult:
        task = self._active.get("task")
        if not task:
            return ToolResult("No research has been run this session.")
        if not task.done():
            elapsed = (time.monotonic() - self._active["started"]) / 60.0
            return ToolResult(
                f"Researching '{self._active['topic']}' — {elapsed:.1f} of "
                f"{self._active['minutes']:.0f} minutes elapsed. Notes in "
                f"{_shown(self._active['folder'])}."
            )
        if task.cancelled():
            return ToolResult(f"Research on '{self._active['topic']}' was cancelled before completion.", ok=False)
        error = task.exception()
        if error is not None:
            return ToolResult(f"Research on '{self._active['topic']}' failed: {first_line(error, 160)}", ok=False)
        result = self._active.get("result") or task.result()
        if isinstance(result, ToolResult):
            return result
        return ToolResult(f"Research on '{self._active['topic']}' ended without a verified report.", ok=False)

    # ── the autonomous run ────────────────────────────────────────────────────

    async def _run(self, topic: str, minutes: float, folder: Path) -> ToolResult:
        from .live_research import describe_evidence, evidence_kind
        from .research_runs import ActivitySignal, publish
        from .research_docx import write_text_artifact
        deadline = time.monotonic() + minutes * 60.0
        notes_dir = folder / "notes"
        # The run's real start, current question and single finish, for the
        # BRAIN and NEURAL pages (research_runs.ActivitySignal).
        activity = ActivitySignal(self.bus, topic, "timed")
        try:
            await asyncio.to_thread(notes_dir.mkdir, parents=True, exist_ok=True)
            self.bus.log.emit(f"RESEARCH: started '{topic}' → {folder.name}")
            activity.start(folder=str(folder))
            questions = await self._decompose(topic)
            await asyncio.to_thread((folder / "README.md").write_text,
                self._readme(topic, questions), encoding="utf-8"
            )
            # One numbering for the whole run (Mark XXXX.9): a note's [3] is
            # sources.md's [3] and the summary's [3], so every citation can be
            # followed to the page it came from.
            registry: dict[str, dict[str, Any]] = {}
            completed_notes: list[tuple[str, str]] = []
            done = 0
            for index, question in enumerate(questions, 1):
                if time.monotonic() > deadline:
                    break
                self.bus.log.emit(f"RESEARCH: investigating — {question}")
                activity.progress(question, index=index, total=len(questions))
                sources = self._register(registry, await self._gather_sources(question))
                note = await self._synthesise(topic, question, sources)
                note_path = notes_dir / f"{index:02d}_{self._slug(question)[:40]}.md"
                await publish(write_text_artifact, note_path, f"# {question}\n\n_{utc_stamp()}_\n\n{note}\n",
                              artifact_paths=lambda path: [path])
                done += 1
                evidence = "\n".join(
                    f"Source [{s['n']}] ({'page read' if evidence_kind(s) == 'page' else 'not opened'}): "
                    f"{s.get('url') or s.get('title')}" for s in sources[:8])
                completed_notes.append((question, f"{evidence}\n{note}"))
                if self.telemetry is not None:
                    self.telemetry.metrics.incr("research.notes")
                # Pace so the run genuinely spans the requested window rather
                # than finishing in seconds and idling. Divided by the questions
                # left PLUS one: dividing by those left alone slept the whole
                # remaining window after the second-to-last question, so in any
                # window under a minute or so the last question was never asked.
                remaining = deadline - time.monotonic()
                gap = remaining / (len(questions) - index + 1) if index < len(questions) else 0
                await asyncio.sleep(max(0.0, min(gap, 20.0)))
            sources_all = sorted(registry.values(), key=lambda s: s["n"])
            # Final synthesis + sources index.
            await publish(self._write_sources, folder, sources_all,
                          artifact_paths=lambda _result: [folder / "sources.md"])
            summary = await self._final_summary_from_notes(topic, completed_notes, sources_all)
            level, phrase = describe_evidence(sources_all)
            await publish(write_text_artifact, folder / "SUMMARY.md",
                f"# Research summary — {topic}\n\n_{utc_stamp()}_\n\n"
                f"_Evidence: {phrase}; numbered in sources.md._\n\n{summary}\n",
                artifact_paths=lambda path: [path],
            )
            self._remember(topic, folder, summary)
            sourced = bool(sources_all)
            state = "completed" if sourced else "failed"
            summary_path = folder / "SUMMARY.md"
            # Said when it is so: a run that read no page rests on snippets,
            # headlines or an encyclopaedia summary, and the owner must not
            # hear it as researched from the web.
            caveat = (f"It rests on {phrase}, so treat it as unverified. "
                      if sourced and level != "pages" else "")
            result = ToolResult(
                (f"Research on '{topic}' is complete. {caveat}" if sourced else
                 f"An unverified draft on '{topic}' was saved; no external sources were retrieved. ")
                + f"Location: {summary_path}", ok=sourced,
                receipt={"research_state": state, "topic": topic,
                         "source_count": len(sources_all), "evidence_level": level,
                         "pages_read": sum(1 for s in sources_all if evidence_kind(s) == "page"),
                         "file_paths": [str(summary_path), str(folder)],
                         "selected_path": str(summary_path)})
            self._active.update(state=state, result=result)
            self.bus.log.emit(f"RESEARCH: {state} — {done} note(s) in {folder.name}.")
            activity.finish("done" if sourced else "draft", folder=str(folder))
            self.bus.banner.emit(f"{'RESEARCH' if sourced else 'DRAFT'} READY: {topic}", 3)
            return result
        except asyncio.CancelledError:
            self._active.update(state="cancelled")
            self.bus.log.emit(f"RESEARCH: cancelled '{topic}'.")
            activity.finish("cancelled")
            raise
        except Exception as exc:
            result = ToolResult(f"Research on '{topic}' failed: {first_line(exc, 160)}", ok=False,
                                receipt={"research_state": "failed", "topic": topic, "file_paths": []})
            self._active.update(state="failed", result=result)
            self.bus.log.emit(f"RESEARCH: fault - {first_line(exc, 160)}")
            activity.finish("failed")
            return result
        finally:
            activity.finish("failed")          # no-op unless a path above missed it

    @staticmethod
    def _register(registry: dict[str, dict[str, Any]],
                  sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """*sources* numbered in the run's one registry (copies, with "n").
        The same page found by two questions keeps one number; a later read
        of it replaces an earlier snippet of it."""
        from .live_research import evidence_kind
        out: list[dict[str, Any]] = []
        for source in sources or []:
            if not isinstance(source, dict):
                continue
            key = str(source.get("url") or source.get("title") or "").strip().lower().rstrip("/")
            if not key:
                continue
            row = registry.get(key)
            if row is None:
                row = registry[key] = {**source, "n": len(registry) + 1}
            elif evidence_kind(source) == "page" and evidence_kind(row) != "page":
                row.update({**source, "n": row["n"]})
            if all(item["n"] != row["n"] for item in out):
                out.append({**source, "n": row["n"]})
        return out

    # ── research paper ────────────────────────────────────────────────────────

    async def write_paper(self, topic: str, folder: Optional[Path] = None) -> ToolResult:
        topic = SecuritySanitiser.guard_text(str(topic or "").strip(), "research.paper")
        if not topic:
            return ToolResult("What should the paper be about?", ok=False)
        from .research_docx import _publish_new
        from .research_runs import publish, verified_paths

        paper_dir: Path | None = None
        paper_path: Path | None = None
        sources: list[dict[str, str]] = []

        def receipt(state: str, **extra: Any) -> dict[str, Any]:
            paths = verified_paths([paper_path, paper_dir / "sources.md" if paper_dir else None])
            return {"research_state": state, "topic": topic, "source_count": len(sources),
                    "file_paths": paths, "selected_path": paths[0] if paths else None, **extra}

        try:
            folder = Path(folder) if folder is not None else await asyncio.to_thread(self._make_folder, topic)
            paper_dir = await asyncio.to_thread(self._reserve_paper_folder, folder)
            self.bus.log.emit(f"RESEARCH: writing paper on '{topic}'.")
            sources = await self._gather_sources(topic)
            # Each invocation owns its sources as well as its paper. A caller's
            # existing research folder is not a mutable scratch directory.
            await publish(self._write_sources, paper_dir, sources,
                          artifact_paths=lambda _result: [paper_dir / "sources.md"])
            sections = ["Abstract", "Introduction", "Background and context",
                        "Analysis", "Discussion", "Conclusion"]
            parts = [f"# {topic.title()}\n", f"_Compiled by O.R.I.O.N. — {utc_stamp()}_\n"]
            if not sources:
                parts.append("_Unverified draft: no sources were retrieved; these claims have not been verified._\n")
            source_context = "\n".join(f"- {s['title']} ({s.get('url', '')}): "
                f"{str(s.get('snippet') or s.get('extract') or s.get('summary') or '')[:1200]}"
                for s in sources[:8])
            for section in sections:
                body = str(await self._write_section(topic, section, source_context) or "").strip()
                if not body or body.startswith("[Model unavailable"):
                    raise RuntimeError(f"No usable text was generated for {section}")
                parts.append(f"\n## {section}\n\n{body}\n")
            if sources:
                parts.append("\n## References\n")
                for index, source in enumerate(sources[:15], 1):
                    link = f" — {source['url']}" if source.get("url") else ""
                    parts.append(f"{index}. {source['title']}{link}\n")
            content = "\n".join(parts)
            paper_path = await publish(_publish_new, paper_dir, "paper", ".md",
                lambda path: path.write_text(content, encoding="utf-8"), artifact_paths=lambda path: [path])
            complete = bool(sources)
            if complete:
                self._remember(topic, paper_dir, f"Research paper written to {paper_path.name}")
            self.bus.dashboard_event.emit("research", {"topic": topic,
                "status": "paper" if complete else "incomplete", "folder": str(paper_dir)})
            text = (f"Paper complete: '{topic}'. {len(sections)} sections and {len(sources)} reference(s), "
                    if complete else f"Unverified draft for '{topic}': no sources were retrieved; ")
            return ToolResult(text + f"saved to {paper_path.resolve()}.", ok=complete,
                receipt=receipt("completed" if complete else "failed", section_count=len(sections)))
        except Exception as exc:
            return ToolResult(f"Paper on '{topic}' failed: {first_line(exc, 200)}", ok=False,
                              receipt=receipt("failed"))

    @staticmethod
    def _reserve_paper_folder(folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(1000):
            destination = folder / ("paper" if index == 0 else f"paper_{index + 1}")
            try:
                destination.mkdir()
            except FileExistsError:
                continue
            return destination
        raise FileExistsError("No unused paper run directory was available")

    # ── model-backed helpers ──────────────────────────────────────────────────

    async def _decompose(self, topic: str) -> list[str]:
        prompt = (
            f"Break the research topic '{topic}' into 5 to 7 focused sub-questions "
            "that together give thorough coverage. Return each question on its own "
            "line, no numbering, no preamble."
        )
        text = await self._ask(prompt, "You are a meticulous research planner.")
        questions = [re.sub(r"^\s*[-*\d.)]+\s*", "", ln).strip()
                     for ln in text.splitlines() if len(ln.strip()) > 8]
        questions = [q for q in questions if q.endswith("?") or len(q.split()) > 3][:7]
        if not questions:
            questions = [
                f"What is {topic} and why does it matter?",
                f"What are the key facts and current state of {topic}?",
                f"What are the main challenges or debates around {topic}?",
                f"What does the evidence and data say about {topic}?",
                f"What are the practical implications and outlook for {topic}?",
            ]
        return questions

    async def _synthesise(self, topic: str, question: str, sources: list[dict[str, str]]) -> str:
        """One note, every claim tied to a numbered source - and checked.

        Mark XXXX.9: the note used to get unnumbered snippets, an instruction
        to "cite the level of evidence", and "if sources are absent, reason
        from established knowledge"; the file it wrote listed no source at all.
        Now the writer sees numbered sources, pages read first with the
        passages that bear on the question, and must cite them; afterwards a
        citation to a number it was never given is removed, a claim resting on
        nothing is marked [unsourced], and the note ends with the sources it
        actually cited. Still one model call."""
        from .live_research import PASSAGE_CHARS, check_citations, evidence_kind
        rank ={"page": 0, "encyclopaedia": 1, "snippet": 2, "headline": 3}
        given = sorted(_numbered([s for s in sources or [] if isinstance(s, dict)]),
                       key=lambda pair: rank[evidence_kind(pair[1])])[:NOTE_SOURCES]
        if not given:
            prompt = (
                f"Research topic: {topic}\nSub-question: {question}\n"
                "No source could be retrieved for this question. Write a short draft "
                "(100-150 words) from established knowledge, flag uncertainty, and cite "
                "nothing: it will be filed as unverified.")
            text = await self._ask(prompt, "You are a rigorous research analyst writing notes.")
            return f"_Unverified: no source was retrieved for this question._\n\n{text}"
        blocks = []
        for number, source in given:
            body = str(source.get("snippet") or source.get("extract") or source.get("summary") or "")
            limit = PASSAGE_CHARS if evidence_kind(source) == "page" else 400
            blocks.append(f"{_source_line(number, source)}\n{body[:limit]}".rstrip())
        prompt = (
            f"Research topic: {topic}\nSub-question: {question}\n\n"
            "Sources I found, numbered. \"page read\" means the page itself was opened and "
            "the text is its own words; of the others only a search snippet, a headline "
            "or an encyclopaedia summary was seen.\n\n" + "\n\n".join(blocks) + "\n\n"
            "Write a concise research note (150-250 words) answering the sub-question "
            "from these sources.\n"
            "- End every factual sentence with the number of the source that states it, "
            "e.g. [2] or [2][4]. Cite only the numbers listed above.\n"
            "- Prefer what a page read says; a snippet or headline is not proof.\n"
            "- Where sources disagree, write a line beginning \"Sources disagree:\" saying "
            "which numbers say what.\n"
            "- Anything you add that no source states goes on a final line beginning "
            "\"Not in the sources:\", uncited.\n"
            "- If the sources do not answer the question, say so plainly rather than "
            "answering from memory.")
        text = await self._ask(prompt, "You are a rigorous research analyst writing notes.")
        kinds = {number: evidence_kind(source) for number, source in given}
        checked, check = check_citations(text, kinds)
        cited = [(n, s) for n, s in given if n in check.numbers]
        footer = ["", "**Sources cited**"]
        footer += [f"- {_source_line(n, s)}" for n, s in sorted(cited, key=lambda pair: pair[0])]
        if not cited:
            footer.append("- No listed source was cited; treat this note as unverified.")
        footer.append(f"- {check.line()}")
        return checked.rstrip() + "\n" + "\n".join(footer)

    async def _write_section(self, topic: str, section: str, source_context: str) -> str:
        prompt = (
            f"Write the '{section}' section of a research paper on '{topic}'. "
            + (f"Relevant sources:\n{source_context}\n" if source_context else "")
            + "Be scholarly but readable; 150-300 words; British English. "
            "Do not repeat the section heading."
        )
        return await self._ask(prompt, "You are writing a structured research paper.")

    async def _final_summary(self, topic: str, questions: list[str]) -> str:
        prompt = (
            f"Summarise the research on '{topic}' as an executive briefing (200 words): "
            "the most important findings, the level of confidence, and recommended next "
            "steps. The sub-questions investigated were:\n" + "\n".join(f"- {q}" for q in questions)
        )
        return await self._ask(prompt, "You are ORION delivering a research briefing.")

    async def _final_summary_from_notes(self, topic: str, notes: list[tuple[str, str]],
                                        sources: list[dict[str, Any]] | None = None) -> str:
        """Synthesis receives the findings, not just the questions asked.

        With the run's numbered *sources* (Mark XXXX.9) the briefing keeps the
        notes' [n] on each finding, carries their disagreements and unsourced
        claims as such, and is checked like a note: a citation to a number
        that does not exist is removed and an uncited finding is marked."""
        if not notes:
            raise RuntimeError("No completed research notes were available for the summary")
        index = ""
        if sources:
            index = ("SOURCE INDEX:\n" + "\n".join(
                _source_line(n, s) for n, s in _numbered(sources)[:40]))[:2600] + "\n\n"
        preamble = (
            f"Summarise the research on '{topic}' as an executive briefing (200 words). "
            "Use only the completed notes below, retain their source URLs"
            + (" and keep their [n] source numbers on each finding (only numbers in the "
               "source index exist)" if index else "")
            + ", flag disagreement on a line beginning \"Sources disagree:\", list what the "
            "notes mark [unsourced] or \"Not in the sources\" under \"Unverified:\" rather "
            "than as findings, flag missing evidence, and do not fill gaps from general "
            "knowledge. Retrieved snippets or headlines are not proof that a full page was "
            "read.\n\n")
        budget = max(2000, SUMMARY_PROMPT_CHARS - len(preamble) - len(index))
        evidence = "\n\n".join(f"Question: {question}\n{body[:2400]}"
                                for question, body in notes[:12])[:budget]
        text = await self._ask(preamble + index + evidence,
                               "You are ORION delivering an evidence-grounded research briefing.")
        if not sources:
            return text
        from .live_research import check_citations, evidence_kind
        checked, _check = check_citations(text, {n: evidence_kind(s) for n, s in _numbered(sources)})
        return checked

    async def _ask(self, prompt: str, persona: str) -> str:
        try:
            _profile, text = await self.router.generate_text(prompt, system_extra=persona)
            text = str(text or "").strip()
            if not text or text.startswith("[Model unavailable"):
                raise RuntimeError("The research provider returned no usable text")
            return text
        except Exception as exc:
            raise RuntimeError(f"Research generation failed: {first_line(exc, 160)}") from exc

    # ── source gathering (online best-effort, offline-safe) ───────────────────

    async def gather_sources(self, query: str, *, read_pages: int = 0) -> list[dict[str, str]]:
        """Silent live-source retrieval: encyclopaedia summary plus headlines.

        The reasoning engine's ``web_lookup`` instrument needs facts from the
        world mid-thought, and the ordinary ``web_search`` tool cannot serve it:
        that one opens a browser window at the user, which is the wrong thing to
        do several times inside a single question. This is the same gathering
        the autonomous runs use, exposed without the folder and note machinery,
        and it returns an empty list offline rather than raising.

        *read_pages* (Mark XXXX.9): how many search results to open and read.
        0 for the reasoning engine's quick look-up, which must stay quick; the
        dossier asks for READ_PAGES so its notes rest on pages, not snippets.
        """
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "research.lookup")
        if not query:
            return []
        gather = self._gather_sources
        sources = await (gather(query, read_pages=read_pages) if _accepts(gather, keyword="read_pages")
                         else gather(query))
        # Same reasoning as the briefing: these carry titles and links, and a
        # link that is only ever spoken is not a link.
        try:
            if sources:
                self.bus.content_results.emit(list(sources))
        except Exception:
            pass
        return sources

    async def _gather_sources(self, query: str, *, read_pages: int = READ_PAGES) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        timeout = ClientTimeout(total=12.0, connect=4.0)
        try:
            async with ClientSession(timeout=timeout) as session:
                wiki = await self._wikipedia(session, query)
                if wiki:
                    sources.append(wiki)
                sources.extend(await self._news(session, query))
        except Exception:
            pass  # offline or blocked — the model reasons from knowledge instead
        # A real web search too — an encyclopaedia summary and headlines were
        # all this ever had, which is why "research" read like the model's own
        # memory. Search results carry the page and what it covers.
        try:
            from . import web_search_backends as backends

            for hit in await backends.search(query, limit=6):
                if all(hit.url != s.get("url") for s in sources):
                    sources.append({"title": hit.title, "url": hit.url,
                                    "snippet": hit.snippet, "evidence": "snippet"})
        except Exception:
            pass
        if read_pages > 0:
            await self.read_sources(query, sources, limit=read_pages)
        return sources

    async def read_sources(self, question: str, sources: list[dict[str, Any]],
                           limit: int = READ_PAGES) -> list[dict[str, Any]]:
        """Open the top search results and keep what they say about *question*.

        Each page read replaces its snippet with the passages that bear on the
        question (live_research.relevant_passages) and is marked
        evidence="page". Concurrent and bounded - three readers, a timeout a
        page, a refused page handing its slot to the next result - and never
        raising: an unreadable page simply stays a snippet. Network only; no
        model call."""
        from .live_research import SKIP_HOSTS, _host, evidence_kind, read_page, relevant_passages

        queue = [s for s in sources if isinstance(s, dict) and s.get("url")
                 and evidence_kind(s) == "snippet"
                 and _host(str(s["url"])) not in SKIP_HOSTS
                 and not _host(str(s["url"])).endswith(_UNREADABLE_HOSTS)][:READ_CANDIDATES]
        read: list[dict[str, Any]] = []

        async def reader() -> None:
            while queue and len(read) < limit:
                source = queue.pop(0)
                try:
                    _title, text = await asyncio.wait_for(read_page(str(source["url"])), READ_TIMEOUT_S)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    continue
                passage = relevant_passages(text, question)
                if len(passage) < 200 or len(read) >= limit:
                    continue
                source.update(snippet=passage, evidence="page", chars_read=len(text))
                read.append(source)

        await asyncio.gather(*(reader() for _ in range(max(0, min(3, limit)))))
        return read

    async def _wikipedia(self, session: ClientSession, query: str) -> Optional[dict[str, str]]:
        try:
            # A title, not a query string: quote_plus turned "circadian rhythm"
            # into "circadian+rhythm", a page Wikipedia does not have, so every
            # multi-word lookup came back 404 (Mark XXXX.9).
            title = quote(re.sub(r"\s+", "_", query.strip()), safe="")
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            async with session.get(url, headers={"User-Agent": "ORION-Research/1.0"}) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            extract = str(data.get("extract") or "").strip()
            if not extract:
                return None
            return {"title": f"Wikipedia: {data.get('title', query)}", "evidence": "encyclopaedia",
                    "url": (data.get("content_urls", {}).get("desktop", {}) or {}).get("page", ""),
                    "snippet": extract}
        except Exception:
            return None

    async def _news(self, session: ClientSession, query: str) -> list[dict[str, str]]:
        try:
            url = ("https://news.google.com/rss/search?q="
                   f"{quote_plus(query)}&hl=en-GB&gl=GB&ceid=GB:en")
            async with session.get(url) as r:
                if r.status != 200:
                    return []
                raw = await r.text()
        except Exception:
            return []
        out: list[dict[str, str]] = []
        for block in re.findall(r"<item>(.*?)</item>", raw, re.S)[:5]:
            title = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", block, re.S)
            link = re.search(r"<link>(.*?)</link>", block, re.S)
            published = re.search(r"<pubDate>(.*?)</pubDate>", block, re.S)
            summary = re.search(r"<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>",
                                block, re.S)
            if title:
                out.append({"title": re.sub(r"\s+", " ", title.group(1)).strip()[:160], "evidence": "headline",
                            "url": link.group(1).strip() if link else "",
                            "published": re.sub(r"\s+", " ", published.group(1)).strip()[:40]
                            if published else "",
                            "snippet": _plain(summary.group(1))[:300] if summary else ""})
        await self._add_excerpts(out[:NEWS_EXCERPTS])
        for item in out:
            # The date leads the snippet: a headline with no date is how a
            # dossier came to call last year's model "verified current".
            date = item.pop("published", "")
            if date:
                item["snippet"] = f"Published {date}. {item['snippet']}".strip()
        return out

    async def _add_excerpts(self, items: list[dict[str, str]]) -> None:
        """Replace each top headline's empty snippet with a bounded excerpt of
        the article itself, fetched concurrently through the bounded page
        reader (no browser, size and redirect limits). A page that cannot be
        read keeps the RSS summary; nothing here ever raises."""
        if not items:
            return
        from . import background_http

        async def excerpt(item: dict[str, str]) -> None:
            url = item.get("url") or ""
            if not url:
                return
            try:
                page = await asyncio.wait_for(
                    background_http.fetch_text(url, max_length=4000), NEWS_EXCERPT_TIMEOUT_S)
            except asyncio.CancelledError:
                raise
            except Exception:
                return
            if not getattr(page, "ok", False):
                return
            text = " ".join(line for line in str(page.text or "").splitlines()
                            if len(line.split()) >= 8)      # prose, not menus
            if len(text) >= 200:
                item["snippet"] = (text[:NEWS_EXCERPT_CHARS].rsplit(" ", 1)[0] + " …"
                                   if len(text) > NEWS_EXCERPT_CHARS else text)
                item["evidence"] = "page"            # the article itself was read

        await asyncio.gather(*(excerpt(item) for item in items), return_exceptions=True)

    # ── filesystem + memory ───────────────────────────────────────────────────

    def _make_folder(self, topic: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d")
        self.RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
        stem = f"{stamp}_{self._slug(topic)[:48]}"
        for index in range(1000):
            folder = self.RESEARCH_ROOT / f"{stem}{'_' + str(index + 1) if index else ''}"
            try:
                folder.mkdir()
            except FileExistsError:
                continue
            (folder / "notes").mkdir()
            return folder
        raise FileExistsError("No unused research folder was available")

    def _write_sources(self, folder: Path, sources: list[dict[str, str]]) -> None:
        if not sources:
            return
        from .live_research import EVIDENCE_LABELS, evidence_kind
        seen: set[str] = set()
        # Numbered as the notes and the summary cite them, and each says how
        # it was seen - a page read or only a snippet/headline (Mark XXXX.9).
        lines = ["# Sources\n"]
        for index, s in enumerate(sources, 1):
            key = s.get("url") or s["title"]
            if key in seen:
                continue
            seen.add(key)
            link = f" — {s['url']}" if s.get("url") else ""
            lines.append(f"- [{s.get('n') or index}] ({EVIDENCE_LABELS[evidence_kind(s)]}) "
                         f"**{s['title']}**{link}")
            if s.get("snippet"):
                lines.append(f"  - {s['snippet'][:300]}")
        from .research_docx import write_text_artifact
        write_text_artifact(folder / "sources.md", "\n".join(lines) + "\n")

    def _remember(self, topic: str, folder: Path, summary: str) -> None:
        try:
            self.memory.remember(
                "project", f"research_{self._slug(topic)[:32]}",
                f"{summary[:400]} (folder: {folder.resolve()})",
            )
        except Exception:
            pass

    def _readme(self, topic: str, questions: list[str]) -> str:
        return (
            f"# Research: {topic}\n\n_Started {utc_stamp()} by O.R.I.O.N._\n\n"
            "## Sub-questions\n" + "\n".join(f"- {q}" for q in questions)
            + "\n\n## Layout\n- `notes/` — one note per sub-question\n"
            "- `sources.md` — gathered sources\n- `SUMMARY.md` — executive briefing\n"
            "- `paper/paper.md` — full paper (if requested)\n"
        )

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-") or "topic"
