"""
Investigations that run while you are not here.

``ResearchAgent`` already crawls, synthesises and writes papers — it is the
engine, and this does not replace it. What was missing is everything around an
*unattended* investigation: a clean source of live intelligence, a filed
dossier with citations you can check, and something that tells you when it is
done, because a report nobody knows exists is a report nobody reads.

Search through MCP, not through the DOM
---------------------------------------
Gathering currently comes from an encyclopaedia summary and news headlines
fetched directly. That works and stays as the fallback, but it is thin for a
real investigation, and scraping search-engine HTML is the fragile alternative
— it breaks on a layout change, silently, returning nothing while looking
fine.

So a search MCP server (Tavily or Brave) is preferred when one is connected:
it returns titles, URLs and extracts as data, with no markup to parse and
nothing to break when a page is redesigned. Absent one, the existing path runs
and the dossier says which was used, because "no sources found" and "no search
server configured" are very different problems and should not look alike.

Citations are deduplicated by URL
---------------------------------
Several searches on related questions return the same page repeatedly. Left
alone the citation list is mostly the same three links, which makes a thin
dossier look well-sourced — the one failure mode that matters for something
you are meant to trust without having watched it happen.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .constants import CONFIG_DIR

#: Where dossiers are filed.
DOSSIER_DIR = CONFIG_DIR / "research" / "dossiers"

#: Most sources cited in one dossier. Beyond this the list stops being a
#: bibliography and becomes a log.
MAX_CITATIONS = 40

#: Questions a topic is broken into. Enough to cover a subject from several
#: sides; few enough that an unattended run finishes in minutes, not hours.
DEFAULT_QUESTIONS = 5


def slugify(topic: str) -> str:
    """A filename that is still recognisable as the topic it came from."""
    text = re.sub(r"[^\w\s-]", "", str(topic or "").strip().lower())
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return (text or "untitled")[:60]


@dataclass
class Source:
    """One thing that was read, and where it came from."""

    title: str
    url: str = ""
    extract: str = ""
    origin: str = ""          # which gatherer produced it
    #: How it was seen: "page" (opened and read), "snippet", "headline" or
    #: "encyclopaedia" (live_research.EVIDENCE_LABELS). Mark XXXX.9.
    evidence: str = ""

    @property
    def key(self) -> str:
        """What makes two sources the same.

        The URL when there is one, because the same page reached by two
        searches is one source. Failing that the title, lowercased — an
        encyclopaedia summary has no URL but is still the same summary.
        """
        return (self.url or self.title).strip().lower().rstrip("/")


def dedupe(sources: Iterable[Source]) -> list[Source]:
    """Distinct sources, in the order they were first seen.

    Several searches on related questions return the same page repeatedly.
    Left alone the citation list is mostly the same three links, which makes a
    thin dossier look well-sourced.
    """
    seen: set[str] = set()
    out: list[Source] = []
    for source in sources:
        key = source.key
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(source)
    return out


@dataclass
class Dossier:
    """What an investigation produced."""

    topic: str
    started_at: float
    summary: str = ""
    takeaways: list[str] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    gatherer: str = ""
    #: live_research.describe_evidence level: pages, snippets_only,
    #: encyclopaedia_only or none.
    evidence_level: str = ""
    path: Path | None = None
    filing_error: str = ""
    generation_errors: list[str] = field(default_factory=list)

    @property
    def minutes(self) -> float:
        return (time.time() - self.started_at) / 60.0

    def as_markdown(self) -> str:
        when = datetime.fromtimestamp(self.started_at)
        lines = [
            f"# {self.topic}",
            "",
            f"*Compiled by O.R.I.O.N. on {when.strftime('%A %d %B %Y at %H:%M')}"
            f" — {self.minutes:.0f} minute{'s' if self.minutes >= 2 else ''},"
            f" {len(self.sources)} source{'s' if len(self.sources) != 1 else ''}"
            f" via {self.gatherer or 'unknown'}.*",
            "",
            "## Executive summary",
            "",
            self.summary.strip() or "_No summary was produced._",
            "",
        ]
        if self.takeaways:
            lines += ["## Key takeaways", ""]
            lines += [f"{n}. {point}" for n, point in enumerate(self.takeaways, 1)]
            lines.append("")
        if self.generation_errors:
            lines += ["## Incomplete investigation", ""]
            lines += [f"- {error}" for error in self.generation_errors]
            lines.append("")
        for heading, body in self.sections:
            lines += [f"## {heading}", "", body.strip(), ""]

        lines += ["## Sources", ""]
        if self.sources:
            from .live_research import EVIDENCE_LABELS, evidence_kind
            # Numbered as the sections cite them, each saying whether the page
            # was read or only its snippet or headline seen (Mark XXXX.9).
            for number, source in enumerate(self.sources[:MAX_CITATIONS], 1):
                label = EVIDENCE_LABELS[evidence_kind(source)]
                if source.url:
                    lines.append(f"{number}. [{source.title}]({source.url}) — {label}")
                else:
                    lines.append(f"{number}. {source.title} — {label}")
        else:
            # Said plainly rather than left as an empty heading: a dossier with
            # no sources is a dossier written from the model's own memory, and
            # the reader has to know that.
            lines.append("_None. Everything above is the model's own knowledge,"
                         " not anything read today._")
        lines.append("")
        return "\n".join(lines)


class ResearchWorker:
    """Runs an investigation end to end and files the result.

    The heavy lifting — decomposition, synthesis, section writing — belongs to
    ``ResearchAgent`` and is called, not reimplemented. This owns the parts
    that only matter when nobody is watching.
    """

    def __init__(self, agent: Any = None, mcp: Any = None, bus: Any = None,
                 reminders: Any = None, telephony: Any = None,
                 root: Path | str | None = None, *, announce_results: bool = True) -> None:
        self.agent = agent
        self.mcp = mcp
        self.bus = bus
        self.reminders = reminders
        self.telephony = telephony
        self.announce_results = announce_results
        self.root = Path(root) if root else DOSSIER_DIR

    # ── gathering ────────────────────────────────────────────────────────────

    def search_server(self) -> str:
        """The name of a connected search MCP server, or "".

        Tavily first, then Brave — both return structured results; the
        preference is arbitrary and either is a large improvement on parsing
        a results page.
        """
        host = self.mcp
        if host is None:
            return ""
        try:
            names = [str(n).lower() for n in host.server_names()]
        except Exception:
            try:
                names = [str(n).lower() for n in getattr(host, "servers", {})]
            except Exception:
                return ""
        for wanted in ("tavily", "brave"):
            for name in names:
                if wanted in name:
                    return name
        return ""

    async def gather(self, question: str) -> tuple[list[Source], str]:
        """Sources for one question, and which gatherer found them."""
        from .research import READ_PAGES, _accepts

        server = self.search_server()
        if server:
            found = await self._search_mcp(server, question)
            if found:
                # A search server's results are extracts too; the top ones are
                # opened and read like any other (Mark XXXX.9).
                reader = getattr(self.agent, "read_sources", None)
                if callable(reader):
                    rows = [{"title": s.title, "url": s.url, "snippet": s.extract,
                             "evidence": "snippet"} for s in found]
                    try:
                        await reader(question, rows, READ_PAGES)
                    except Exception as exc:
                        self._log(f"reading {server} results failed - {exc}")
                    for source, row in zip(found, rows):
                        source.extract, source.evidence = row["snippet"], row["evidence"]
                return found, server
            # An empty result from a configured server is reported as such
            # rather than silently falling back — a search server that returns
            # nothing is a different problem from not having one.
            self._log(f"{server} returned nothing for {question!r}")

        if self.agent is None:
            return [], server or "none"
        try:
            gather = self.agent.gather_sources
            # Pages opened and read, not just search snippets and headlines:
            # "11 sources via the built-in encyclopaedia and news lookup" in
            # the owner's log (30 Oct 2026) was not one page read.
            raw = await (gather(question, read_pages=READ_PAGES)
                         if _accepts(gather, keyword="read_pages") else gather(question))
        except Exception as exc:
            self._log(f"gathering failed - {exc}")
            return [], server or "none"
        return ([Source(title=str(row.get("title") or "").strip(),
                        url=str(row.get("url") or "").strip(),
                        extract=str(row.get("snippet") or row.get("summary") or row.get("extract") or ""),
                        origin="builtin", evidence=str(row.get("evidence") or ""))
                 for row in (raw or []) if row.get("title")],
                "the built-in web search, encyclopaedia and news lookup")

    async def _search_mcp(self, server: str, question: str) -> list[Source]:
        """Ask a search MCP server, tolerating its tool being named anything."""
        for tool in ("search", "tavily_search", "brave_web_search",
                     "web_search"):
            try:
                result = await self.mcp.call_tool(
                    server, tool, {"query": question, "max_results": 8})
            except Exception:
                continue
            if not getattr(result, "ok", True):
                continue
            return _parse_results(str(getattr(result, "text", result) or ""),
                                  origin=server)
        return []

    # ── the investigation ────────────────────────────────────────────────────

    async def investigate(self, topic: str, *,
                          questions: int = DEFAULT_QUESTIONS) -> Dossier:
        """Research *topic*, write it up, file it, and say so."""
        topic = str(topic or "").strip()
        dossier = Dossier(topic=topic or "untitled", started_at=time.time())
        if not topic:
            dossier.summary = "No topic was given."
            return dossier

        self._log(f"investigating {topic!r}")
        # The dossier's real start, current question and single finish, for
        # the BRAIN and NEURAL pages (research_runs.ActivitySignal).
        from .research_runs import ActivitySignal
        activity = ActivitySignal(self.bus, topic, "dossier")
        activity.start()
        status = "failed"
        try:
            await self._investigate(dossier, topic, questions, activity)
            status = ("done" if dossier.sources and not dossier.generation_errors
                      else "draft")
            return dossier
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            activity.finish(status)

    async def _investigate(self, dossier: Dossier, topic: str, questions: int,
                           activity: Any) -> None:
        from .live_research import describe_evidence, evidence_kind

        asked = await self._questions(topic, questions)
        collected: list[Source] = []
        evidence: list[tuple[str, str]] = []
        #: One numbering for the dossier, the order its Sources list prints
        #: (dedupe keeps first-seen order), so a section's [3] is entry 3.
        numbers: dict[str, int] = {}
        for index, question in enumerate(asked, 1):
            activity.progress(question, index=index, total=len(asked))
            found, gatherer = await self.gather(question)
            dossier.gatherer = dossier.gatherer or gatherer
            collected.extend(found)
            for source in found:
                if source.key and source.key not in numbers:
                    numbers[source.key] = len(numbers) + 1
            try:
                body = await self._write(topic, question, found, numbers)
                if not body:
                    raise RuntimeError("No research note was produced")
            except Exception as exc:
                dossier.generation_errors.append(f"{question}: {str(exc)[:240]}")
                body = ""
            if body:
                dossier.sections.append((question, body))
                evidence.append((question, "\n".join(
                    f"Source [{numbers.get(source.key, '?')}] "
                    f"({'page read' if evidence_kind(source) == 'page' else 'not opened'}): "
                    f"{source.url or source.title}" for source in found[:8]) + "\n" + body))

        # The page read wins over an earlier snippet of the same page.
        read = {s.key: s for s in collected if evidence_kind(s) == "page"}
        dossier.sources = [read.get(s.key, s) for s in dedupe(collected)]
        level, phrase = describe_evidence(dossier.sources)
        dossier.evidence_level = level
        if dossier.sources:
            # What the sources really were, not just who was asked: an
            # encyclopaedia-only or snippet-only dossier says so (Mark XXXX.9).
            dossier.gatherer = f"{dossier.gatherer or 'the built-in lookup'} ({phrase})"
        index_rows = [{"title": s.title, "url": s.url, "evidence": s.evidence, "n": n}
                      for n, s in enumerate(dossier.sources, 1)]
        try:
            dossier.summary = await self._summarise(topic, asked, evidence, index_rows)
            if not dossier.summary:
                raise RuntimeError("No research summary was produced")
        except Exception as exc:
            dossier.generation_errors.append(f"Summary: {str(exc)[:240]}")
        dossier.takeaways = await self._takeaways(topic, dossier)
        from .research_runs import publish
        dossier.path = await publish(self.file, dossier, artifact_paths=lambda path: [path] if path else [])
        if self.announce_results:
            await self.announce(dossier)

    async def _questions(self, topic: str, count: int) -> list[str]:
        agent = self.agent
        if agent is not None and hasattr(agent, "_decompose"):
            try:
                found = await agent._decompose(topic)
                if found:
                    return list(found)[:count]
            except Exception:
                pass
        # A usable default rather than nothing: these four angles cover most
        # subjects well enough for an unattended first pass.
        return [f"What is {topic}?",
                f"What has changed recently in {topic}?",
                f"Who are the main people and organisations in {topic}?",
                f"What are the risks or open questions in {topic}?"][:count]

    async def _write(self, topic: str, question: str,
                     sources: list[Source], numbers: dict[str, int] | None = None) -> str:
        agent = self.agent
        if agent is None:
            return ""
        context = "\n\n".join(
            f"{s.title}\n{s.extract}".strip() for s in sources if s.title)
        errors = []
        for method in ("_synthesise", "_write_section"):
            writer = getattr(agent, method, None)
            if writer is None:
                continue
            try:
                if method == "_synthesise":
                    rows = [{"title": s.title, "url": s.url, "evidence": s.evidence,
                             "snippet": s.extract, "summary": s.extract, "extract": s.extract,
                             **({"n": numbers[s.key]} if numbers and s.key in numbers else {})}
                            for s in sources]
                    return str(await writer(topic, question, rows) or "")
                return str(await writer(topic, question, context) or "")
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError(errors[-1])
        return ""

    async def _summarise(self, topic: str, questions: list[str],
                         notes: list[tuple[str, str]],
                         sources: list[dict[str, Any]] | None = None) -> str:
        agent = self.agent
        grounded = getattr(agent, "_final_summary_from_notes", None) if agent else None
        if grounded is not None:
            from .research import _accepts
            if sources and _accepts(grounded, 3):
                return str(await grounded(topic, notes, sources) or "")
            return str(await grounded(topic, notes) or "")
        summary = getattr(agent, "_final_summary", None) if agent else None
        if summary is None:
            return ""
        return str(await summary(topic, questions) or "")

    async def _takeaways(self, topic: str, dossier: Dossier) -> list[str]:
        """The three or four things worth acting on."""
        agent = self.agent
        ask = getattr(agent, "_ask", None) if agent else None
        if ask is None:
            return []
        body = dossier.summary or "\n".join(b for _h, b in dossier.sections)
        if not body.strip():
            return []
        try:
            answer = await ask(
                f"From this research on {topic}, give the 3-4 things that "
                f"actually matter, one per line, no numbering, no preamble:"
                f"\n\n{body[:4000]}",
                "You state conclusions plainly and never pad.")
        except Exception:
            return []
        lines = [re.sub(r"^\s*[-*•\d.)]+\s*", "", line).strip()
                 for line in str(answer or "").splitlines()]
        return [line for line in lines if len(line) > 12][:4]

    # ── filing and telling ───────────────────────────────────────────────────

    def file(self, dossier: Dossier) -> Path | None:
        """Publish a complete new dossier; never overwrite an earlier report."""
        from .research_docx import _publish_new

        dossier.filing_error = ""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.fromtimestamp(dossier.started_at).strftime(
                "%Y-%m-%d_%H%M")

            def write(path: Path) -> None:
                with path.open("w", encoding="utf-8") as handle:
                    handle.write(dossier.as_markdown())
                    handle.flush()
                    os.fsync(handle.fileno())

            stem = f"{slugify(dossier.topic)}_{stamp}"
            return _publish_new(self.root, stem, ".md", write)
        except OSError as exc:
            dossier.filing_error = f"Could not save the dossier in {self.root}: {exc}"
            self._log(dossier.filing_error)
            return None

    async def announce(self, dossier: Dossier) -> None:
        """Say it is ready, on whichever channel is available.

        A report nobody knows exists is a report nobody reads — and this runs
        unattended, so there is no moment where the user would notice by
        themselves.
        """
        saved = dossier.path is not None and dossier.path.is_file()
        if saved and dossier.generation_errors:
            line = (f"I saved an incomplete investigation on {dossier.topic} at {dossier.path}. "
                    "Writing or summarising failed; the research is not complete.")
        elif saved and dossier.sources:
            line = (f"I've finished researching {dossier.topic}. "
                    f"{len(dossier.sources)} source"
                    f"{'s' if len(dossier.sources) != 1 else ''}, saved at {dossier.path}.")
            if dossier.evidence_level in {"snippets_only", "encyclopaedia_only"}:
                line += " No web page could be read, so treat it as unverified."
        elif saved:
            line = (f"I saved an unverified draft on {dossier.topic} at {dossier.path}. "
                    "No external sources were retrieved, so the research is not complete.")
        else:
            line = (f"The research on {dossier.topic} is ready, but no report was saved. "
                    + (dossier.filing_error or "No file location was returned."))
        self._log(line)

        if self.reminders is not None:
            try:
                result = await asyncio.to_thread(self.reminders.add, text=line, minutes=0.02)
                if getattr(result, "ok", False):
                    return
            except Exception:
                pass
        if self.bus is not None:
            try:
                self.bus.speak_request.emit(line)
            except Exception:
                pass

    def _log(self, line: str) -> None:
        if self.bus is None:
            return
        try:
            self.bus.log.emit(f"[Research] {line}")
        except Exception:
            pass


def _parse_results(payload: str, origin: str = "") -> list[Source]:
    """Search results out of whatever shape the server returned them in.

    MCP servers return a JSON array, a JSON object with a "results" key, or
    prose with links in it, depending on the server and the version. All three
    are read rather than one being assumed, because assuming is how a working
    search server produces an empty dossier.
    """
    import json

    text = (payload or "").strip()
    if not text:
        return []

    rows: list[dict] = []
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0),
                default=-1)
    if start >= 0:
        for closer in ("]", "}"):
            end = text.rfind(closer)
            if end <= start:
                continue
            try:
                parsed = json.loads(text[start:end + 1])
            except ValueError:
                continue
            if isinstance(parsed, dict):
                parsed = (parsed.get("results") or parsed.get("web", {})
                          .get("results") or [])
            if isinstance(parsed, list):
                rows = [row for row in parsed if isinstance(row, dict)]
                break

    sources = []
    for row in rows:
        title = str(row.get("title") or row.get("name") or "").strip()
        url = str(row.get("url") or row.get("link") or "").strip()
        extract = str(row.get("content") or row.get("description")
                      or row.get("snippet") or row.get("extract") or "").strip()
        if title or url:
            sources.append(Source(title=title or url, url=url,
                                  extract=extract, origin=origin))
    if sources:
        return sources

    # No JSON anywhere — fall back to reading markdown links out of prose.
    for match in re.finditer(r"\[([^\]]{3,160})\]\((https?://[^\s)]+)\)", text):
        sources.append(Source(title=match.group(1), url=match.group(2),
                              origin=origin))
    return sources


__all__ = [
    "DEFAULT_QUESTIONS", "DOSSIER_DIR", "MAX_CITATIONS",
    "Dossier", "ResearchWorker", "Source", "dedupe", "slugify",
]
