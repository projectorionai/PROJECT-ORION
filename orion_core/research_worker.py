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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

#: Where dossiers are filed.
DOSSIER_DIR = Path("research") / "dossiers"

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
    path: Path | None = None

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
        for heading, body in self.sections:
            lines += [f"## {heading}", "", body.strip(), ""]

        lines += ["## Sources", ""]
        if self.sources:
            for number, source in enumerate(self.sources[:MAX_CITATIONS], 1):
                if source.url:
                    lines.append(f"{number}. [{source.title}]({source.url})")
                else:
                    lines.append(f"{number}. {source.title}")
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
                 root: Path | str | None = None) -> None:
        self.agent = agent
        self.mcp = mcp
        self.bus = bus
        self.reminders = reminders
        self.telephony = telephony
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
        server = self.search_server()
        if server:
            found = await self._search_mcp(server, question)
            if found:
                return found, server
            # An empty result from a configured server is reported as such
            # rather than silently falling back — a search server that returns
            # nothing is a different problem from not having one.
            self._log(f"{server} returned nothing for {question!r}")

        if self.agent is None:
            return [], server or "none"
        try:
            raw = await self.agent.gather_sources(question)
        except Exception as exc:
            self._log(f"gathering failed - {exc}")
            return [], server or "none"
        return ([Source(title=str(row.get("title") or "").strip(),
                        url=str(row.get("url") or "").strip(),
                        extract=str(row.get("summary") or row.get("extract") or ""),
                        origin="builtin")
                 for row in (raw or []) if row.get("title")],
                "the built-in encyclopaedia and news lookup")

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
        asked = await self._questions(topic, questions)

        collected: list[Source] = []
        for question in asked:
            found, gatherer = await self.gather(question)
            dossier.gatherer = dossier.gatherer or gatherer
            collected.extend(found)
            body = await self._write(topic, question, found)
            if body:
                dossier.sections.append((question, body))

        dossier.sources = dedupe(collected)
        dossier.summary = await self._summarise(topic, asked)
        dossier.takeaways = await self._takeaways(topic, dossier)
        dossier.path = self.file(dossier)
        await self.announce(dossier)
        return dossier

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
                     sources: list[Source]) -> str:
        agent = self.agent
        if agent is None:
            return ""
        context = "\n\n".join(
            f"{s.title}\n{s.extract}".strip() for s in sources if s.title)
        for method in ("_synthesise", "_write_section"):
            writer = getattr(agent, method, None)
            if writer is None:
                continue
            try:
                if method == "_synthesise":
                    rows = [{"title": s.title, "url": s.url,
                             "summary": s.extract} for s in sources]
                    return str(await writer(topic, question, rows) or "")
                return str(await writer(topic, question, context) or "")
            except Exception:
                continue
        return ""

    async def _summarise(self, topic: str, questions: list[str]) -> str:
        agent = self.agent
        summary = getattr(agent, "_final_summary", None) if agent else None
        if summary is None:
            return ""
        try:
            return str(await summary(topic, questions) or "")
        except Exception:
            return ""

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
        """Write the dossier to research/dossiers/. Returns where it went."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.fromtimestamp(dossier.started_at).strftime(
                "%Y-%m-%d_%H%M")
            path = self.root / f"{slugify(dossier.topic)}_{stamp}.md"
            path.write_text(dossier.as_markdown(), encoding="utf-8")
            return path
        except OSError as exc:
            self._log(f"the dossier could not be filed - {exc}")
            return None

    async def announce(self, dossier: Dossier) -> None:
        """Say it is ready, on whichever channel is available.

        A report nobody knows exists is a report nobody reads — and this runs
        unattended, so there is no moment where the user would notice by
        themselves.
        """
        where = dossier.path.name if dossier.path else "memory"
        line = (f"I've finished researching {dossier.topic}. "
                f"{len(dossier.sources)} source"
                f"{'s' if len(dossier.sources) != 1 else ''}, filed as {where}.")
        self._log(line)

        if self.reminders is not None:
            try:
                self.reminders.add(text=line, minutes=0.02)
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
