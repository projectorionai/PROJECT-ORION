"""
LearningService — feed ORION information and he learns it.

Give ORION raw text, a local file, or a URL and he distils the key facts and
commits them to the persistent KNOWLEDGE memory tier, tagged by topic and
cross-referenced against what he already knows.  This is how ORION "gets fed
information like a very smart AI": every ingest permanently expands what he can
recall, online or offline.

Distillation uses the provider router (cloud or local Ollama) to extract crisp,
atomic facts; with no model reachable it falls back to an extractive summary
(salient sentences), so learning never fully fails.  Fetching/parsing runs off
the event loop via ``asyncio.to_thread``; the learned digest is emitted on
``bus.dashboard_event('learned', …)`` for the GUI.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from aiohttp import ClientSession, ClientTimeout

from .bus import OrionBus
from .constants import BASE_DIR
from .data import ToolResult
from .memory import MemoryAgent, MemoryTier
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line, utc_stamp

_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]")
_TAG_RE = re.compile(r"[a-z0-9]+")


class LearningService:
    MAX_FACTS = 12

    def __init__(self, bus: OrionBus, memory: MemoryAgent, router: Any,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.router = router
        self.telemetry = telemetry

    # ── public API ────────────────────────────────────────────────────────────

    async def learn(self, source: str, topic: str = "") -> ToolResult:
        """Learn from raw text, a file path, or a URL."""
        source = str(source or "").strip()
        if not source:
            return ToolResult("What would you like me to learn, sir?", ok=False)
        try:
            SecuritySanitiser.guard_text(source[:400], "learn.source")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)

        kind, text = await self._materialise(source)
        if not text or len(text.strip()) < 20:
            return ToolResult(
                f"I couldn't get enough text from that {kind}, sir.", ok=False)

        topic = topic.strip() or self._infer_topic(text, source)
        facts = await self._distil(text, topic)
        if not facts:
            return ToolResult("I read it but couldn't extract anything worth keeping, sir.", ok=False)

        stored = await asyncio.to_thread(self._store, topic, facts, kind, source)
        self.bus.dashboard_event.emit("learned", {"topic": topic, "facts": facts[:6],
                                                   "at": utc_stamp(), "source": kind})
        if self.telemetry is not None:
            self.telemetry.metrics.incr("learning.ingested", float(stored))
        preview = "\n".join(f"  • {f}" for f in facts[:5])
        return ToolResult(
            f"Learned and committed to memory, sir — {stored} fact(s) on '{topic}' "
            f"(from {kind}). I'll recall these anytime.\n{preview}"
        )

    # ── source materialisation ────────────────────────────────────────────────

    async def _materialise(self, source: str) -> tuple[str, str]:
        # URL?
        parsed = urlparse(source if "://" in source else "")
        if parsed.scheme in {"http", "https"}:
            return "web page", await self._fetch_url(source)
        # File?
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        if path.is_file():
            return "file", await asyncio.to_thread(self._read_file, path)
        # Otherwise treat the input itself as the material to learn.
        return "text", source

    async def _fetch_url(self, url: str) -> str:
        try:
            timeout = ClientTimeout(total=15.0, connect=5.0)
            headers = {"User-Agent": "Mozilla/5.0 (ORION learning)"}
            async with ClientSession(timeout=timeout) as s:
                async with s.get(url, headers=headers) as r:
                    if r.status != 200:
                        return ""
                    raw = await r.text()
        except Exception as exc:
            self.bus.log.emit(f"LEARN: fetch failed - {first_line(exc, 80)}")
            return ""
        # Strip scripts/styles/tags to plain text.
        raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
        text = re.sub(r"(?s)<[^>]+>", " ", raw)
        text = re.sub(r"&[a-z]+;", " ", text)
        return re.sub(r"\s+", " ", text).strip()[:12000]

    # Text-like source files ORION can read directly, and rich docs it can parse.
    TEXT_SUFFIXES = frozenset({
        ".txt", ".md", ".markdown", ".rst", ".text", ".log", ".tex",
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
        ".cs", ".go", ".rs", ".rb", ".php", ".sql", ".sh", ".ps1",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv",
        ".html", ".htm", ".xml",
    })
    DOC_SUFFIXES = frozenset({".pdf", ".docx"})

    def _read_file(self, path: Path, cap: int = 12000) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                return "\n".join((p.extract_text() or "") for p in reader.pages[:60])[:cap]
            except Exception:
                return ""
        if suffix == ".docx":
            try:
                from docx import Document
                doc = Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs)[:cap]
            except Exception:
                return ""
        if suffix in {".html", ".htm", ".xml"}:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
            raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
            return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", raw)).strip()[:cap]
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:cap]
        except OSError:
            return ""

    # ── distillation ──────────────────────────────────────────────────────────

    async def _distil(self, text: str, topic: str) -> list[str]:
        if self.router.has_text_fallback():
            persona = (
                "You extract durable, atomic facts from text for a knowledge base. Return 5-10 "
                "concise standalone facts, one per line, no numbering, no preamble. Each fact must "
                "be self-contained and true to the text."
            )
            prompt = f"Topic: {topic}\n\nText:\n{text[:6000]}\n\nExtract the key facts."
            try:
                _profile, answer = await self.router.generate_text(prompt, system_extra=persona)
                facts = [re.sub(r"^\s*[-*\d.)]+\s*", "", ln).strip()
                         for ln in answer.splitlines() if len(ln.strip()) > 12]
                facts = [f for f in facts if f][: self.MAX_FACTS]
                if facts:
                    return facts
            except Exception as exc:
                self.bus.log.emit(f"LEARN: model distil failed - {first_line(exc, 80)}")
        # Extractive fallback: the most informative sentences.
        return self._extractive(text)

    def _extractive(self, text: str) -> list[str]:
        sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if 40 <= len(s.strip()) <= 300]
        if not sentences:
            return []
        # Rank by word-frequency salience.
        words = _TAG_RE.findall(text.lower())
        freq: dict[str, int] = {}
        for w in words:
            if len(w) > 3:
                freq[w] = freq.get(w, 0) + 1
        scored = sorted(
            sentences,
            key=lambda s: sum(freq.get(w, 0) for w in _TAG_RE.findall(s.lower())),
            reverse=True,
        )
        return scored[:8]

    # ── storage ───────────────────────────────────────────────────────────────

    def _store(self, topic: str, facts: list[str], kind: str, source: str) -> int:
        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:40] or "topic"
        stored = 0
        try:
            self.memory.remember(MemoryTier.KNOWLEDGE, f"learned_{slug}",
                                 f"Learned about {topic} (from {kind}: {source[:80]})")
            for i, fact in enumerate(facts[: self.MAX_FACTS], 1):
                self.memory.remember(MemoryTier.KNOWLEDGE, f"learned_{slug}_{i}",
                                     f"[{topic}] {fact}")
                stored += 1
        except Exception as exc:
            self.bus.log.emit(f"LEARN: store recovered - {first_line(exc, 80)}")
        return stored

    def _infer_topic(self, text: str, source: str) -> str:
        parsed = urlparse(source if "://" in source else "")
        if parsed.netloc:
            return parsed.netloc.replace("www.", "")
        # First few salient words.
        words = [w for w in _TAG_RE.findall(text) if len(w) > 3][:4]
        return " ".join(words).title() or "general"

    async def recall(self, query: str) -> ToolResult:
        """Recall previously-learned facts (KNOWLEDGE tier)."""
        query = str(query or "").strip()
        if not query:
            return ToolResult("What shall I recall, sir?", ok=False)
        rows = await asyncio.to_thread(self.memory.query, query, 8)
        learned = [r.get("value", "") for r in rows if str(r.get("key_ref", "")).startswith("learned_")]
        others = [r.get("value", "") for r in rows if not str(r.get("key_ref", "")).startswith("learned_")]
        hits = (learned + others)[:8]
        if not hits:
            return ToolResult(f"I haven't learned anything matching '{query}' yet, sir.")
        return ToolResult("Here's what I know, sir:\n" + "\n".join(f"- {h}" for h in hits if h))

    # ── bulk ingestion — the practical path to "gigabytes" of knowledge ────────

    async def learn_folder(self, folder: str, topic: str = "", recursive: bool = True,
                           deep: bool = False, max_files: int = 4000) -> ToolResult:
        """
        Ingest an entire folder of documents into ORION's KNOWLEDGE tier.

        Point this at a library of textbooks, papers, notes or code (PDF, DOCX,
        Markdown, text, HTML, source files) and ORION reads every file and
        commits its salient facts, permanently and offline-recallable. This is
        how ORION realistically learns gigabytes: you supply the corpus locally,
        he indexes it.

        ``deep=False`` (default) uses fast extractive summarisation — no API
        cost, scales to very large libraries. ``deep=True`` distils each file
        with the language model for higher-quality facts (slower; best on
        smaller, high-value sets). Fully non-blocking: walking and extraction
        run off the event loop.
        """
        root = Path(str(folder or "").strip()).expanduser()
        if not root.is_absolute():
            root = BASE_DIR / root
        if not root.is_dir():
            return ToolResult(f"That folder doesn't exist, sir: {root}", ok=False)
        base_topic = topic.strip() or root.name
        slug = re.sub(r"[^a-z0-9]+", "_", base_topic.lower()).strip("_")[:40] or "library"
        self.bus.log.emit(f"LEARN: ingesting folder {root} as '{base_topic}' (deep={deep}).")

        supported = self.TEXT_SUFFIXES | self.DOC_SUFFIXES
        walker = root.rglob("*") if recursive else root.glob("*")
        files = [p for p in walker if p.is_file() and p.suffix.lower() in supported][:max_files]
        if not files:
            return ToolResult(f"No readable documents found in {root}, sir.", ok=False)

        total_files = 0
        total_facts = 0
        total_bytes = 0
        for index, path in enumerate(files, 1):
            try:
                text = await asyncio.to_thread(self._read_file, path, 16000)
            except Exception:
                text = ""
            if not text or len(text.strip()) < 60:
                continue
            total_bytes += len(text)
            if deep and self.router.has_text_fallback():
                facts = await self._distil(text, f"{base_topic}: {path.stem}")
            else:
                facts = await asyncio.to_thread(self._extractive, text)
            if not facts:
                continue
            file_slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")[:36] or "doc"
            stored = await asyncio.to_thread(
                self._store_bulk, slug, file_slug, base_topic, facts, path.name)
            total_facts += stored
            total_files += 1
            if index % 25 == 0:
                self.bus.dashboard_event.emit("learning", {
                    "topic": base_topic, "files": total_files, "facts": total_facts,
                    "progress": f"{index}/{len(files)}", "at": utc_stamp()})
                self.bus.log.emit(f"LEARN: {index}/{len(files)} files, {total_facts} facts so far.")

        if self.telemetry is not None:
            self.telemetry.metrics.incr("learning.bulk_files", float(total_files))
            self.telemetry.metrics.incr("learning.bulk_facts", float(total_facts))
        self.bus.dashboard_event.emit("learned", {
            "topic": base_topic, "facts": [f"{total_facts} facts from {total_files} files"],
            "at": utc_stamp(), "source": "folder"})
        mb = total_bytes / 1_000_000.0
        return ToolResult(
            f"Ingested '{base_topic}', sir — {total_facts} fact(s) committed from "
            f"{total_files} document(s) ({mb:.1f} MB of text read). I can recall any of "
            f"it now, online or offline. Say 'forget {base_topic}' to remove it.")

    def _store_bulk(self, lib_slug: str, file_slug: str, topic: str,
                    facts: list[str], filename: str) -> int:
        stored = 0
        try:
            self.memory.remember(
                MemoryTier.KNOWLEDGE, f"learned_{lib_slug}",
                f"Learned library '{topic}' (bulk ingest).")
            for i, fact in enumerate(facts[:8], 1):
                self.memory.remember(
                    MemoryTier.KNOWLEDGE, f"learned_{lib_slug}_{file_slug}_{i}",
                    f"[{topic}] {fact}")
                stored += 1
        except Exception as exc:
            self.bus.log.emit(f"LEARN: bulk store recovered - {first_line(exc, 80)}")
        return stored

    # ── correction learning — fix or remove what ORION learned ─────────────────

    async def correct(self, topic: str, correction: str) -> ToolResult:
        """Record an authoritative correction that overrides earlier learning."""
        topic = str(topic or "").strip()
        correction = str(correction or "").strip()
        if not correction:
            return ToolResult("What's the correction, sir?", ok=False)
        try:
            SecuritySanitiser.guard_text(correction[:400], "learn.correction")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        slug = re.sub(r"[^a-z0-9]+", "_", (topic or correction).lower()).strip("_")[:40] or "note"
        stamp = utc_stamp()
        await asyncio.to_thread(
            self.memory.remember, MemoryTier.KNOWLEDGE, f"correction_{slug}",
            f"CORRECTION{(' — ' + topic) if topic else ''}: {correction} (as of {stamp})")
        self.bus.dashboard_event.emit("learned", {
            "topic": topic or "correction", "facts": [correction], "at": stamp,
            "source": "correction"})
        return ToolResult(
            f"Understood, sir — I've corrected that and will treat it as authoritative"
            f"{(' on ' + topic) if topic else ''} from now on.")

    async def forget(self, query: str) -> ToolResult:
        """Remove learned facts matching a topic/library (KNOWLEDGE tier only)."""
        query = str(query or "").strip()
        if not query:
            return ToolResult("What would you like me to forget, sir?", ok=False)
        slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:40]
        removed = 0
        if slug:
            removed += await asyncio.to_thread(
                self.memory.forget, "knowledge", f"learned_{slug}")
        # Also sweep learned facts whose stored value mentions the topic.
        removed += await asyncio.to_thread(
            self.memory.forget, "knowledge", "learned_", query)
        if removed:
            self.bus.log.emit(f"LEARN: forgot {removed} learned record(s) for '{query}'.")
            return ToolResult(
                f"Done, sir — I've forgotten {removed} learned record(s) relating to "
                f"'{query}'. Curated built-in knowledge is untouched.")
        return ToolResult(
            f"I found nothing I'd learned about '{query}' to forget, sir (built-in "
            "knowledge isn't removed this way).")
