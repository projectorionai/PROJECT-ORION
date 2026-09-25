"""
Research the way a person does it: look things up, read them, take notes.

What ORION did before was not research. He fetched a Wikipedia summary and a
few news headlines, handed those to the model as context, and asked it to
write. The prose came out fluent and fast and almost entirely from the model's
own memory — which is a fine way to produce an essay and a poor way to find
anything out. It cannot tell you what a page actually said, it cannot notice
that two sources disagree, and its citations point at things it never read.

This does the other thing. It searches, opens each result, reads the page,
writes down what that page says with the URL attached, and lets what it finds
decide what to look up next. It is slower on purpose. A person reading twelve
pages takes a few minutes, and the output is worth more than a paragraph
generated in four seconds.

Watching it work
----------------
Every step is announced on the bus before it happens — the search being run,
the page being opened, the note being taken, the question that reading it
raised. That is not decoration: a long job with no visible progress is
indistinguishable from a hung one, and the notes are more interesting than the
summary anyway. ``on_event`` receives the same stream for anything that would
rather render it than log it.

Reading a page without a browser
--------------------------------
HTML is fetched and stripped here rather than driven through Chromium. A
headless browser is the right tool when a page builds itself with JavaScript,
and the wrong one for reading forty articles: it costs a second and a hundred
megabytes each time. ``browser_copilot`` remains for the pages that genuinely
need it.

Where the pages come from
-------------------------
``web_search_backends`` finds them: Google through Gemini's search grounding
(ORION's own key), then Brave, DuckDuckGo (MCP server, then HTML) and
Wikipedia — keyless results must share the question's words, and adult hosts
are never handed on. Pages are ranked by meaning before any is opened, a
refused page hands its slot to the next candidate, and a 403 on the plain
client is retried with a Chrome TLS fingerprint (curl_cffi). The keyless routes
are not reliable on their own, and it is worth knowing why rather than being
surprised.
DuckDuckGo's HTML endpoint answered the first query of a session with forty
results and every one after with ``202 Accepted`` and an empty page: a
success-shaped refusal. Wikipedia always answers and only knows what has an
article. A free Brave key makes the whole thing dependable, and its absence
degrades this from "research" to "encyclopaedia lookup" rather than breaking
it.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote_plus, urlparse
from .atomic_io import atomic_write_text

#: How long to wait for one page. Generous enough for a slow news site,
#: short enough that one bad host does not hold up the whole investigation.
PAGE_TIMEOUT_S = 12.0
SEARCH_TIMEOUT_S = 15.0

#: Most characters kept from one page. Enough for the model to work from,
#: bounded so a single enormous article cannot swallow the context budget.
MAX_PAGE_CHARS = 12_000

#: Pages read at once. Small deliberately — this is meant to look like
#: reading, the remote hosts have not agreed to be hammered, and a burst of
#: forty simultaneous requests is how an IP gets blocked.
CONCURRENT_READS = 3
#: Questions researched at the same time (see investigate()).
CONCURRENT_QUESTIONS = 2

#: Hosts that never carry the substance: search engines, social shells, and
#: the aggregators that only ever restate someone else's page.
SKIP_HOSTS = frozenset({
    "duckduckgo.com", "google.com", "www.google.com", "bing.com",
    "www.bing.com", "facebook.com", "www.facebook.com", "twitter.com",
    "x.com", "instagram.com", "www.instagram.com", "pinterest.com",
    "youtube.com", "www.youtube.com", "tiktok.com",
})

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style|noscript|svg|nav|footer|header|form)\b.*?</\1>",
                     re.IGNORECASE | re.DOTALL)
_SPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK = re.compile(r"\n{3,}")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_RESULT = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL)
_SNIPPET = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL)


def _clean(raw: str) -> str:
    """Tags out, entities decoded, whitespace made readable."""
    text = _SCRIPT.sub(" ", raw or "")
    text = _TAG.sub("\n", text)
    text = html.unescape(text)
    text = _SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK.sub("\n\n", text).strip()


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


@dataclass
class Note:
    """One thing ORION read, and what he made of it."""

    question: str
    title: str
    url: str
    note: str
    at: float = field(default_factory=time.time)
    #: Roughly how much of the page was actually read.
    chars: int = 0

    def as_markdown(self) -> str:
        source = f"[{self.title or self.url}]({self.url})" if self.url else self.title
        return f"**{source}**\n\n{self.note.strip()}\n"


@dataclass
class Finding:
    """A search result, before anybody has read it."""

    title: str
    url: str
    snippet: str = ""


@dataclass
class Investigation:
    """Everything one run learned, in the order it learned it."""

    topic: str
    questions: list[str] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float = 0.0

    @property
    def seconds(self) -> float:
        return (self.finished or time.time()) - self.started

    @property
    def sources(self) -> list[str]:
        seen, out = set(), []
        for note in self.notes:
            if note.url and note.url not in seen:
                seen.add(note.url)
                out.append(note.url)
        return out


def _rank_by_meaning(question: str, findings: list["Finding"]) -> list["Finding"]:
    """*findings* reordered by how closely title + snippet match *question*.

    Unchanged when the sentence encoder is not warm (semantic.py): search
    order is a perfectly good fallback, just a less informed one.
    """
    try:
        from . import semantic

        if len(findings) < 2 or not semantic.ENCODER.ready():
            return findings
        vectors = semantic.ENCODER.encode(
            [f"{f.title}. {f.snippet}"[:600] for f in findings])
        target = semantic.ENCODER.encode_one(question)
        if vectors is None or target is None:
            return findings
        scores = vectors @ target
        order = sorted(range(len(findings)), key=lambda i: -float(scores[i]))
        return [findings[i] for i in order]
    except Exception:
        return findings


#: Statuses that mean "not you" rather than "not here": worth one retry as a
#: browser. A 404 is an answer and is not retried.
_REFUSALS = frozenset({401, 403, 429, 503})


async def _fetch_as_browser(url: str, timeout: float, status: int) -> tuple[str, int]:
    """Retry *url* with Chrome's TLS fingerprint (curl_cffi), if installed.

    Returns (html, status); html is "" when it is still refused or the
    package is missing, and status is then the best one seen.
    """
    try:
        from curl_cffi.requests import AsyncSession
    except Exception:
        return "", status
    try:
        async with AsyncSession(impersonate="chrome", timeout=timeout) as session:
            response = await session.get(url, allow_redirects=True)
    except Exception:
        return "", status
    kind = str(response.headers.get("content-type") or "")
    if response.status_code != 200:
        return "", response.status_code
    if "html" not in kind and "text" not in kind:
        return "", 200
    return response.text, 200


class LiveResearcher:
    """Searches, reads, and takes notes — announcing each step as it goes.

    *ask* is an async callable ``(prompt, persona) -> str``; everything that
    needs judgement goes through it, and everything that needs facts goes
    through the network. Neither is asked to do the other's job, which is the
    whole difference from what this replaces.
    """

    def __init__(self, ask: Callable[..., Any], bus: Any = None,
                 on_event: Callable[[dict], None] | None = None,
                 root: Path | str | None = None) -> None:
        self.ask = ask
        self.bus = bus
        self.on_event = on_event
        self.root = Path(root) if root else None
        self._seen: set[str] = set()

    # ── saying what is happening ─────────────────────────────────────────────

    def _say(self, kind: str, message: str, **extra: Any) -> None:
        """Announce a step BEFORE doing it.

        Before, not after: the interesting moment in "reading
        nature.com/articles/…" is while it is happening, and a progress line
        that only appears once the work is done is a progress line that never
        told anybody anything.
        """
        payload = {"kind": kind, "message": message, "at": time.time(), **extra}
        if self.bus is not None:
            try:
                self.bus.log.emit(f"[Research] {message}")
            except Exception:
                pass
            try:
                self.bus.dashboard_event.emit("research_step", payload)
            except Exception:
                pass
        if self.on_event is not None:
            try:
                self.on_event(payload)
            except Exception:
                pass

    # ── the web ──────────────────────────────────────────────────────────────

    async def search(self, query: str, limit: int = 8) -> list[Finding]:
        """Find pages, through whichever backend is answering today.

        Delegated to web_search_backends rather than scraping one engine
        here, because the single-engine version failed in the way that
        matters: DuckDuckGo answered the first query with forty results and
        every query after with 202 and an empty page. That reads as "nothing
        to find" and sounds like it looked.
        """
        from . import web_search_backends as backends

        self._say("search", f"searching the web for: {query}", query=query)
        try:
            hits = await backends.search(query, limit=limit, say=self._say)
        except backends.SearchUnavailable as exc:
            # Raised, not empty: the caller has to tell "nothing exists" from
            # "nothing could be asked", and so does whoever reads the report.
            self._say("search_failed", str(exc), query=query)
            return []
        findings = [Finding(title=h.title, url=h.url, snippet=h.snippet)
                    for h in hits if _host(h.url) not in SKIP_HOSTS]
        self._say("search_done",
                  f"found {len(findings)} page(s) worth opening",
                  count=len(findings))
        return findings

    async def read(self, finding: Finding) -> str:
        """Open a page and return its readable text, or "" if it will not."""
        self._say("reading", f"reading {finding.title or finding.url}",
                  url=finding.url, title=finding.title)
        body = await self._fetch(finding.url, PAGE_TIMEOUT_S)
        if not body:
            return ""
        title = _TITLE.search(body)
        if title and not finding.title:
            finding.title = _clean(title.group(1))[:200]
        text = _clean(body)
        # A page that yields almost nothing is a cookie wall, a login page or
        # a JavaScript shell. Saying so is better than taking a note from it.
        if len(text) < 400:
            self._say("thin", f"{_host(finding.url)} gave almost no text — "
                              f"probably a cookie wall or a script-built page",
                      url=finding.url)
            return ""
        # What was actually read, so the research console can show the page's
        # own words beside the note taken from it.
        self._say("read", f"read {min(len(text), MAX_PAGE_CHARS):,} characters "
                          f"from {_host(finding.url)}",
                  url=finding.url, title=finding.title,
                  excerpt=text[:1600])
        return text[:MAX_PAGE_CHARS]

    async def _fetch(self, url: str, timeout: float) -> str:
        """One HTTP GET, returning "" for anything that is not readable text."""
        try:
            from aiohttp import ClientSession, ClientTimeout
        except Exception:
            return ""
        # The agent depends on who is being asked. Most sites serve a stub or
        # a 403 to anything that announces itself as a script, so they get a
        # browser string. Wikimedia does the exact opposite: its robot policy
        # REFUSES a generic agent and asks you to identify yourself, and a
        # browser string earned a 403 on every article. Sending one agent to
        # both was why every page came back unreadable.
        from .web_search_backends import _USER_AGENT, _WIKI_AGENT

        host = _host(url)
        wikimedia = host.endswith("wikipedia.org") or host.endswith("wikimedia.org")
        headers = {
            "User-Agent": _WIKI_AGENT if wikimedia else _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-GB,en;q=0.9",
        }
        status = 0
        try:
            async with ClientSession(timeout=ClientTimeout(total=timeout)) as session:
                async with session.get(url, headers=headers,
                                       allow_redirects=True) as response:
                    status = response.status
                    if status == 200:
                        kind = str(response.headers.get("Content-Type") or "")
                        if "html" not in kind and "text" not in kind:
                            return ""
                        return await response.text(errors="replace")
        except asyncio.TimeoutError:
            self._say("unreachable", f"{_host(url)} took too long", url=url)
            return ""
        except Exception as exc:
            self._say("unreachable",
                      f"{_host(url)} could not be read ({type(exc).__name__})",
                      url=url)
            return ""
        if status in _REFUSALS and not wikimedia:
            # Refused on the connection's fingerprint, not its headers — a
            # full browser header set changed nothing on the sites that did
            # this (ceh.ac.uk, merriam-webster.com), a Chrome TLS handshake
            # got a 200 from both.
            body, status = await _fetch_as_browser(url, timeout, status)
            if body:
                return body
        self._say("unreachable", f"{_host(url)} answered {status}",
                  url=url, status=status)
        return ""

    # ── the judgement ────────────────────────────────────────────────────────

    async def plan(self, topic: str, count: int = 5) -> list[str]:
        """Break a topic into the questions a person would actually look up."""
        self._say("planning", f"working out what to find out about: {topic}")
        prompt = (
            f"I am about to research this properly: {topic}\n\n"
            f"Give me {count} specific questions to look up, one per line, "
            f"no numbering and no commentary. Each should be something a "
            f"search engine can actually answer — concrete, narrow, and "
            f"different from the others. Prefer the questions whose answers "
            f"would change what someone concludes.")
        raw = await self._ask(prompt, "research planner")
        questions = [
            re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
            for line in str(raw or "").splitlines() if line.strip()]
        questions = [q for q in questions if len(q) > 12][:count]
        if not questions:
            questions = [topic]
        for question in questions:
            self._say("question", f"question: {question}", question=question)
        return questions

    async def take_note(self, question: str, finding: Finding,
                        text: str) -> Note | None:
        """Write down what THIS page says about the question.

        Grounded deliberately: the instruction is to report what the page
        says, and to say so when it does not answer the question, rather than
        to answer from what the model already believes. A note that quietly
        substitutes the model's own knowledge for the page's is worse than no
        note, because it carries a citation.
        """
        prompt = (
            f"QUESTION: {question}\n\n"
            f"I have just read this page. Write a short note — three or four "
            f"sentences — of what THIS PAGE says that bears on the question. "
            f"Quote a figure or a phrase where the page gives one.\n\n"
            f"Report only what is on the page. If it does not really address "
            f"the question, say exactly that in one line instead of writing a "
            f"note from your own knowledge.\n\n"
            f"PAGE: {finding.title}\n"
            f"URL: {finding.url}\n\n"
            f"{text}")
        body = str(await self._ask(prompt, "research reader") or "").strip()
        if not body:
            return None
        note = Note(question=question, title=finding.title, url=finding.url,
                    note=body, chars=len(text))
        self._say("note", f"note taken from {_host(finding.url)}",
                  url=finding.url, note=body[:400], question=question)
        return note

    async def follow_up(self, topic: str, notes: list[Note],
                        count: int = 2) -> list[str]:
        """What reading those raised. This is the part that makes it research.

        A fixed list of questions answered once is a survey. Letting what was
        found decide what to look at next is the difference, and it is where
        the interesting material usually turns up.
        """
        if not notes:
            return []
        digest = "\n\n".join(f"- {n.note[:400]}" for n in notes[-8:])
        prompt = (
            f"Researching: {topic}\n\nSo far I have read and noted:\n\n"
            f"{digest}\n\n"
            f"What {count} things are still missing or now worth checking? "
            f"Give me searchable questions, one per line, no numbering. If a "
            f"note contradicts another, a question that settles it is the best "
            f"possible answer here. If nothing is genuinely missing, reply "
            f"with the single word NOTHING.")
        raw = str(await self._ask(prompt, "research planner") or "").strip()
        if raw.upper().startswith("NOTHING"):
            self._say("satisfied", "nothing important still missing")
            return []
        questions = [re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
                     for line in raw.splitlines() if line.strip()]
        questions = [q for q in questions if len(q) > 12][:count]
        for question in questions:
            self._say("follow_up", f"that raises: {question}", question=question)
        return questions

    async def _ask(self, prompt: str, persona: str) -> str:
        try:
            return await self.ask(prompt, persona)
        except TypeError:
            return await self.ask(prompt)

    # ── the whole thing ──────────────────────────────────────────────────────

    async def investigate(self, topic: str, *, questions: int = 5,
                          per_question: int = 3, rounds: int = 2,
                          deadline_s: float = 0.0) -> Investigation:
        """Search, read, note, follow up — and say so throughout.

        *rounds* is how many times what was found is allowed to decide what to
        look at next. One round is a survey; two or more is research.
        """
        run = Investigation(topic=topic)
        self._say("start", f"starting research: {topic}", topic=topic)
        ends_at = (time.monotonic() + deadline_s) if deadline_s else 0.0

        pending = await self.plan(topic, questions)
        run.questions.extend(pending)

        for round_number in range(max(1, rounds)):
            if not pending:
                break
            if ends_at and time.monotonic() >= ends_at:
                self._say("time", "out of time — writing up what I have")
                break
            # Two questions at once: the reading is network-bound and the notes
            # are model calls, so doing them strictly in turn left most of the
            # run waiting. More than two trips provider rate limits (a paper's
            # notes once used every per-minute allowance) for little gain.
            gate = asyncio.Semaphore(CONCURRENT_QUESTIONS)

            async def _one(question: str) -> None:
                async with gate:
                    if ends_at and time.monotonic() >= ends_at:
                        return
                    await self._work_question(run, question, per_question, ends_at)

            await asyncio.gather(*(_one(q) for q in pending))

            if round_number + 1 >= max(1, rounds):
                break
            pending = await self.follow_up(topic, run.notes)
            run.questions.extend(pending)

        run.finished = time.time()
        self._say("done",
                  f"read {len(run.read)} page(s), took {len(run.notes)} note(s) "
                  f"in {run.seconds / 60:.1f} minutes",
                  notes=len(run.notes), pages=len(run.read))
        return run

    async def _work_question(self, run: Investigation, question: str,
                             per_question: int, ends_at: float) -> None:
        candidates = [f for f in await self.search(question, limit=per_question * 2)
                      if f.url not in self._seen]
        if not candidates:
            run.failed.append((question, "nothing found worth opening"))
            return
        # Most relevant first, by meaning, when the encoder is warm: the reads
        # are the expensive part (a model note each), so they go to the pages
        # that look most like an answer rather than to search-engine order.
        candidates = await asyncio.to_thread(_rank_by_meaning, question, candidates)
        # A shared queue rather than a fixed slice: a page that will not open
        # (403, cookie wall, script shell) hands its slot to the next-best
        # candidate instead of leaving the question a note short.
        queue = list(candidates)
        wanted = per_question
        notes: list[Note] = []

        async def worker() -> None:
            while queue and len(notes) < wanted:
                if ends_at and time.monotonic() >= ends_at:
                    return
                finding = queue.pop(0)
                # Claimed here, not at search time: with questions worked in
                # parallel, two can find the same page, and only the first to
                # reach it reads it (no await between the check and the claim).
                if finding.url in self._seen:
                    continue
                self._seen.add(finding.url)
                text = await self.read(finding)
                if not text:
                    run.failed.append((finding.url, "unreadable"))
                    continue
                run.read.append(finding.url)
                try:
                    note = await self.take_note(question, finding, text)
                except Exception as exc:
                    run.failed.append((finding.url, f"note failed: {exc}"))
                    continue
                if note is not None and len(notes) < wanted:
                    notes.append(note)

        await asyncio.gather(*(worker() for _ in range(min(CONCURRENT_READS, wanted))))
        run.notes.extend(notes)

    # ── writing it up ────────────────────────────────────────────────────────

    async def summarise(self, run: Investigation) -> str:
        """A summary built from the notes, not from the model's own memory."""
        if not run.notes:
            return (f"I could not read anything useful about {run.topic}. "
                    f"{len(run.failed)} page(s) would not open or carried no "
                    f"text.")
        self._say("writing", "writing up what I found")
        digest = "\n\n".join(
            f"[{i}] {n.title or n.url}\n{n.note}" for i, n in
            enumerate(run.notes, start=1))
        prompt = (
            f"TOPIC: {run.topic}\n\n"
            f"These are notes I took from pages I actually read:\n\n{digest}\n\n"
            f"Write the findings in British English. Say what the sources "
            f"agree on, name any disagreement explicitly and say which sources "
            f"disagree, and state plainly what is still unanswered. Cite as "
            f"[1], [2] against the numbered notes.\n\n"
            f"Use only these notes. Where they do not cover something, say so "
            f"rather than filling the gap from your own knowledge — the point "
            f"of this exercise is what the sources said.")
        return str(await self._ask(prompt, "research writer") or "").strip()

    def write_folder(self, run: Investigation, summary: str = "") -> Path | None:
        """Save the notes and the summary, one file per note.

        Per note, because that is what makes them usable afterwards: a single
        transcript is something nobody opens twice, and each of these carries
        its own source.
        """
        if self.root is None:
            return None
        slug = re.sub(r"[^a-z0-9]+", "_", run.topic.lower()).strip("_")[:60]
        folder = self.root / f"{time.strftime('%Y-%m-%d')}_{slug or 'research'}"
        try:
            (folder / "notes").mkdir(parents=True, exist_ok=True)
            for index, note in enumerate(run.notes, start=1):
                name = re.sub(r"[^a-z0-9]+", "_",
                              (note.title or _host(note.url)).lower())[:50]
                (folder / "notes" / f"{index:02d}_{name or 'note'}.md").write_text(
                    f"# {note.title or note.url}\n\n"
                    f"Source: {note.url}\n\n"
                    f"Question: {note.question}\n\n{note.note}\n",
                    encoding="utf-8")
            if summary:
                (folder / "SUMMARY.md").write_text(
                    f"# {run.topic}\n\n{summary}\n\n## Sources\n\n"
                    + "\n".join(f"- {url}" for url in run.sources) + "\n",
                    encoding="utf-8")
            atomic_write_text((folder / "run.json"), json.dumps({
                "topic": run.topic,
                "questions": run.questions,
                "pages_read": run.read,
                "failed": run.failed,
                "seconds": round(run.seconds, 1),
            }, indent=2), encoding="utf-8")
        except OSError as exc:
            self._say("save_failed", f"could not save the notes ({exc})")
            return None
        self._say("saved", f"notes filed in {folder.name}", folder=str(folder))
        return folder


__all__ = [
    "CONCURRENT_READS", "MAX_PAGE_CHARS", "PAGE_TIMEOUT_S", "SKIP_HOSTS",
    "Finding", "Investigation", "LiveResearcher", "Note",
]
