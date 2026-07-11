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
from urllib.parse import quote_plus

from aiohttp import ClientSession, ClientTimeout

from .bus import OrionBus
from .constants import BASE_DIR
from .data import ToolResult
from .security import SecuritySanitiser
from .utils import first_line, utc_stamp


class ResearchAgent:
    RESEARCH_ROOT = BASE_DIR / "research"

    def __init__(self, bus: OrionBus, router: Any, memory: Any,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.router = router
        self.memory = memory
        self.telemetry = telemetry
        self._active: dict[str, Any] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def start_research(self, topic: str, minutes: float = 30.0) -> ToolResult:
        """Kick off a background research run and return immediately."""
        topic = SecuritySanitiser.guard_text(str(topic or "").strip(), "research.topic")
        if not topic:
            return ToolResult("What should I research, sir?", ok=False)
        if self._active.get("task") and not self._active["task"].done():
            return ToolResult(
                f"I'm already researching '{self._active.get('topic')}'. "
                "Let me finish that first, or say 'stop research'.",
                ok=False,
            )
        minutes = max(1.0, min(180.0, float(minutes)))
        folder = self._make_folder(topic)
        task = asyncio.create_task(self._run(topic, minutes, folder))
        self._active = {"topic": topic, "folder": folder, "task": task,
                        "started": time.monotonic(), "minutes": minutes}
        return ToolResult(
            f"Very good, sir. I'll research '{topic}' for about {int(minutes)} minutes "
            f"and organise my findings in {folder.relative_to(BASE_DIR)}. "
            "Go ahead — I'll have it ready when you return."
        )

    def stop_research(self) -> ToolResult:
        task = self._active.get("task")
        if task and not task.done():
            task.cancel()
            return ToolResult(f"Stopping research on '{self._active.get('topic')}', sir.")
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
                f"{self._active['folder'].relative_to(BASE_DIR)}."
            )
        return ToolResult(f"Research on '{self._active['topic']}' is complete. "
                          f"See {self._active['folder'].relative_to(BASE_DIR)}.")

    # ── the autonomous run ────────────────────────────────────────────────────

    async def _run(self, topic: str, minutes: float, folder: Path) -> None:
        deadline = time.monotonic() + minutes * 60.0
        notes_dir = folder / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        self.bus.log.emit(f"RESEARCH: started '{topic}' → {folder.name}")
        self.bus.dashboard_event.emit("research", {"topic": topic, "status": "running"})
        try:
            questions = await self._decompose(topic)
            (folder / "README.md").write_text(
                self._readme(topic, questions), encoding="utf-8"
            )
            sources_all: list[dict[str, str]] = []
            done = 0
            for index, question in enumerate(questions, 1):
                if time.monotonic() > deadline:
                    break
                self.bus.log.emit(f"RESEARCH: investigating — {question}")
                sources = await self._gather_sources(question)
                sources_all.extend(sources)
                note = await self._synthesise(topic, question, sources)
                (notes_dir / f"{index:02d}_{self._slug(question)[:40]}.md").write_text(
                    f"# {question}\n\n_{utc_stamp()}_\n\n{note}\n", encoding="utf-8"
                )
                done += 1
                if self.telemetry is not None:
                    self.telemetry.metrics.incr("research.notes")
                # Pace so the run genuinely spans the requested window rather
                # than finishing in seconds and idling.
                remaining = deadline - time.monotonic()
                gap = remaining / max(1, len(questions) - index) if index < len(questions) else 0
                await asyncio.sleep(max(0.0, min(gap, 20.0)))
            # Final synthesis + sources index.
            self._write_sources(folder, sources_all)
            summary = await self._final_summary(topic, questions)
            (folder / "SUMMARY.md").write_text(
                f"# Research summary — {topic}\n\n_{utc_stamp()}_\n\n{summary}\n",
                encoding="utf-8",
            )
            self._remember(topic, folder, summary)
            self.bus.log.emit(f"RESEARCH: complete — {done} note(s) in {folder.name}.")
            self.bus.dashboard_event.emit("research", {"topic": topic, "status": "done",
                                                       "folder": str(folder)})
            self.bus.banner.emit(f"RESEARCH READY: {topic}", 3)
        except asyncio.CancelledError:
            self.bus.log.emit(f"RESEARCH: cancelled '{topic}'.")
            raise
        except Exception as exc:
            self.bus.log.emit(f"RESEARCH: fault - {first_line(exc, 160)}")

    # ── research paper ────────────────────────────────────────────────────────

    async def write_paper(self, topic: str, folder: Optional[Path] = None) -> ToolResult:
        topic = SecuritySanitiser.guard_text(str(topic or "").strip(), "research.paper")
        if not topic:
            return ToolResult("What should the paper be about, sir?", ok=False)
        folder = folder or self._make_folder(topic)
        paper_dir = folder / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        self.bus.log.emit(f"RESEARCH: writing paper on '{topic}'.")
        sources = await self._gather_sources(topic)
        self._write_sources(folder, sources)
        sections = [
            "Abstract", "Introduction", "Background and context",
            "Analysis", "Discussion", "Conclusion",
        ]
        parts: list[str] = [f"# {topic.title()}\n", f"_Compiled by O.R.I.O.N. — {utc_stamp()}_\n"]
        source_context = "\n".join(f"- {s['title']}: {s.get('snippet','')[:200]}"
                                   for s in sources[:8])
        for section in sections:
            body = await self._write_section(topic, section, source_context)
            parts.append(f"\n## {section}\n\n{body}\n")
        if sources:
            parts.append("\n## References\n")
            for i, s in enumerate(sources[:15], 1):
                link = f" — {s['url']}" if s.get("url") else ""
                parts.append(f"{i}. {s['title']}{link}\n")
        paper_path = paper_dir / "paper.md"
        paper_path.write_text("\n".join(parts), encoding="utf-8")
        self._remember(topic, folder, f"Research paper written to {paper_path.name}")
        self.bus.dashboard_event.emit("research", {"topic": topic, "status": "paper",
                                                   "folder": str(folder)})
        return ToolResult(
            f"Paper complete, sir: '{topic}'. {len(sections)} sections and "
            f"{len(sources)} reference(s), saved to {paper_path.relative_to(BASE_DIR)}."
        )

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
        ctx = "\n".join(f"- {s['title']}: {s.get('snippet','')[:280]}" for s in sources[:6])
        prompt = (
            f"Research topic: {topic}\nSub-question: {question}\n"
            + (f"Sources I found:\n{ctx}\n" if ctx else "")
            + "Write a concise, well-structured research note (150-250 words) answering "
            "the sub-question. Be factual, cite the level of evidence, and flag "
            "uncertainty. If sources are absent, reason from established knowledge."
        )
        return await self._ask(prompt, "You are a rigorous research analyst writing notes.")

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
        return await self._ask(prompt, "You are ORION delivering a research briefing to sir.")

    async def _ask(self, prompt: str, persona: str) -> str:
        try:
            _profile, text = await self.router.generate_text(prompt, system_extra=persona)
            return text.strip()
        except Exception as exc:
            return (f"[Model unavailable — {first_line(exc, 80)}. This note is a "
                    "placeholder; re-run when a cloud or local model is reachable.]")

    # ── source gathering (online best-effort, offline-safe) ───────────────────

    async def _gather_sources(self, query: str) -> list[dict[str, str]]:
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
        return sources

    async def _wikipedia(self, session: ClientSession, query: str) -> Optional[dict[str, str]]:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}"
            async with session.get(url, headers={"User-Agent": "ORION-Research/1.0"}) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            extract = str(data.get("extract") or "").strip()
            if not extract:
                return None
            return {"title": f"Wikipedia: {data.get('title', query)}",
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
            if title:
                out.append({"title": re.sub(r"\s+", " ", title.group(1)).strip()[:160],
                            "url": link.group(1).strip() if link else "",
                            "snippet": ""})
        return out

    # ── filesystem + memory ───────────────────────────────────────────────────

    def _make_folder(self, topic: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d")
        folder = self.RESEARCH_ROOT / f"{stamp}_{self._slug(topic)[:48]}"
        (folder / "notes").mkdir(parents=True, exist_ok=True)
        return folder

    def _write_sources(self, folder: Path, sources: list[dict[str, str]]) -> None:
        if not sources:
            return
        seen: set[str] = set()
        lines = ["# Sources\n"]
        for s in sources:
            key = s.get("url") or s["title"]
            if key in seen:
                continue
            seen.add(key)
            link = f" — {s['url']}" if s.get("url") else ""
            lines.append(f"- **{s['title']}**{link}")
            if s.get("snippet"):
                lines.append(f"  - {s['snippet'][:300]}")
        (folder / "sources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _remember(self, topic: str, folder: Path, summary: str) -> None:
        try:
            self.memory.remember(
                "project", f"research_{self._slug(topic)[:32]}",
                f"{summary[:400]} (folder: {folder.name})",
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
