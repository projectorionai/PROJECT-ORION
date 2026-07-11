"""
Academic Intake & Concept Graph (additive module).

``LiteratureIntakeService`` performs deep reading of scientific papers (PDF or
text) and distils them into a concept graph seeded into the persistent
KNOWLEDGE memory tier.  For each paper it isolates:

    • citations          — DOIs, numbered references and author–year cites;
    • data tables        — captioned tables and tabular numeric rows;
    • core mechanisms    — sentences describing biophysical mechanisms
                           (ion channels, membrane dynamics, synaptic
                           processes, receptors, plasticity, …).

Extracted mechanisms are tagged with the paper, cross-referenced against
existing KNOWLEDGE entries (so a new paper links to what ORION already knows),
and written into memory via the ``MemoryAgent`` so they surface in ordinary
recall and in the offline LocalBrain.

PDF extraction uses ``pypdf`` / ``PyPDF2`` when available and degrades with a
clear ``bus.log`` warning otherwise.  All parsing runs off the event loop via
``asyncio.to_thread`` and emits ``bus.dashboard_event("literature", …)`` so the
Studio Deck reader updates live.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import BASE_DIR
from .data import ToolResult
from .memory import MemoryAgent, MemoryTier
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line, utc_stamp

# Mechanism vocabulary — a hit makes a sentence "mechanism-first".
MECHANISM_TERMS = (
    "ion channel", "sodium channel", "potassium channel", "calcium channel",
    "membrane potential", "action potential", "depolaris", "hyperpolaris",
    "synap", "receptor", "neurotransmitter", "glutamate", "gaba", "dopamine",
    "conductance", "gating", "phosphorylat", "kinase", "plasticity", "ltp", "ltd",
    "excitator", "inhibitor", "axon", "dendrit", "myelin", "spike", "firing rate",
    "electrode", "impedance", "signal-to-noise", "decoding", "encoding",
    "optogenetic", "calcium imaging", "patch clamp", "voltage clamp",
    "mechanism", "pathway", "cascade", "feedback", "homeostas",
)

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
NUM_REF_RE = re.compile(r"(?m)^\s*\[?(\d{1,3})\]?\.?\s+[A-Z][^\n]{15,200}")
AUTHOR_YEAR_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+(?:et al\.?|and\s+[A-Z][a-z]+))?),?\s+\(?(19|20)\d{2}\)?")
TABLE_CAPTION_RE = re.compile(r"(?im)\b(table\s+\d+[.:]?\s*[^\n]{0,120})")
TABULAR_ROW_RE = re.compile(r"^\s*\S+(?:\s+[-+]?\d[\d.,%]*){2,}\s*$", re.MULTILINE)
SENTENCE_RE = re.compile(r"[^.!?]*[.!?]")


@dataclass
class PaperRecord:
    slug: str
    title: str
    path: str
    at: str
    citations: list[str] = field(default_factory=list)
    dois: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)
    word_count: int = 0

    def summary(self) -> str:
        return (
            f"'{self.title}' — {self.word_count} words; "
            f"{len(self.mechanisms)} mechanism(s), {len(self.tables)} table(s), "
            f"{len(self.citations)} citation(s), {len(self.dois)} DOI(s)."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug, "title": self.title, "at": self.at,
            "mechanisms": self.mechanisms[:8], "tables": self.tables[:6],
            "citations": self.citations[:8], "dois": self.dois[:8],
            "cross_refs": self.cross_refs[:8], "word_count": self.word_count,
        }


class LiteratureIntakeService:
    """Deep-reads papers into a KNOWLEDGE-tier concept graph."""

    MAX_MECHANISMS = 24
    MAX_CITATIONS = 40

    def __init__(self, bus: OrionBus, memory: MemoryAgent, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.telemetry = telemetry
        self._papers: dict[str, PaperRecord] = {}

    # ── ingestion ─────────────────────────────────────────────────────────────

    async def ingest_paper(self, path: str, title: str = "") -> ToolResult:
        try:
            path = SecuritySanitiser.guard_text(str(path or "").strip(), "lit.path")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        source = Path(os.path.expandvars(os.path.expanduser(path)))
        if not source.is_absolute():
            source = BASE_DIR / source
        if not source.is_file():
            return ToolResult(f"Paper not found: {source}", ok=False)
        text = await asyncio.to_thread(self._extract_text, source)
        if not text:
            return ToolResult(
                f"Could not extract text from '{source.name}'. If it is a PDF, install "
                "pypdf (pip install pypdf); scanned PDFs need OCR.",
                ok=False,
            )
        record = await asyncio.to_thread(self._analyse, source, text, title)
        # Seed into the KNOWLEDGE tier (SQLite writes off the event loop).
        await asyncio.to_thread(self._seed_memory, record)
        self._papers[record.slug] = record
        self.bus.dashboard_event.emit("literature", record.to_dict())
        if self.telemetry is not None:
            self.telemetry.metrics.incr("literature.ingested")
            self.telemetry.metrics.gauge("literature.papers", float(len(self._papers)))
        self.bus.log.emit(f"LIT: ingested {record.summary()}")
        cross = (" Cross-referenced with: " + ", ".join(record.cross_refs[:4]) + ".") \
            if record.cross_refs else ""
        return ToolResult(f"Ingested and indexed {record.summary()}{cross}")

    # ── extraction ────────────────────────────────────────────────────────────

    def _extract_text(self, source: Path) -> str:
        suffix = source.suffix.lower()
        if suffix in {".txt", ".md", ".rst"}:
            try:
                return source.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
        if suffix == ".pdf":
            return self._extract_pdf(source)
        # Unknown — try as text (many .tei/.xml/.html papers are readable).
        try:
            raw = source.read_text(encoding="utf-8", errors="replace")
            return re.sub(r"<[^>]+>", " ", raw)   # strip tags if any
        except OSError:
            return ""

    def _extract_pdf(self, source: Path) -> str:
        reader_cls: Any = None
        try:
            from pypdf import PdfReader as reader_cls  # type: ignore
        except Exception:
            try:
                from PyPDF2 import PdfReader as reader_cls  # type: ignore
            except Exception:
                reader_cls = None
        if reader_cls is None:
            self.bus.log.emit("LIT: pypdf not installed; PDF reading disabled (pip install pypdf).")
            return ""
        try:
            reader = reader_cls(str(source))
            chunks: list[str] = []
            for page in reader.pages[:60]:
                try:
                    chunks.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(chunks)
        except Exception as exc:
            self.bus.log.emit(f"LIT: PDF parse recovered - {first_line(exc, 80)}")
            return ""

    def _analyse(self, source: Path, text: str, title: str) -> PaperRecord:
        clean = re.sub(r"[ \t]+", " ", text)
        title = (title.strip() or self._guess_title(clean, source))
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:48] or source.stem.lower()[:48]
        record = PaperRecord(slug=slug, title=title[:160], path=str(source),
                             at=utc_stamp(), word_count=len(clean.split()))
        record.dois = sorted(set(DOI_RE.findall(clean)))[: self.MAX_CITATIONS]
        record.citations = self._extract_citations(clean)
        record.tables = self._extract_tables(text)
        record.mechanisms = self._extract_mechanisms(clean)
        record.cross_refs = self._cross_reference(record.mechanisms)
        return record

    def _guess_title(self, text: str, source: Path) -> str:
        for line in text.splitlines():
            line = line.strip()
            if 20 <= len(line) <= 160 and not line.lower().startswith(("abstract", "doi", "http")):
                return line
        return source.stem.replace("_", " ").title()

    def _extract_citations(self, text: str) -> list[str]:
        cites: list[str] = []
        # Numbered reference-list entries ("[12] Author, ... 2020 ...").
        for m in NUM_REF_RE.finditer(text):
            cites.append(re.sub(r"\s+", " ", m.group(0)).strip()[:180])
        # Fall back to inline author–year cites if few numbered refs were found.
        if len(cites) < 5:
            for m in AUTHOR_YEAR_RE.finditer(text):
                cites.append(re.sub(r"\s+", " ", m.group(0)).strip())
        # Dedupe preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for c in cites:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out[: self.MAX_CITATIONS]

    def _extract_tables(self, text: str) -> list[str]:
        tables: list[str] = []
        for m in TABLE_CAPTION_RE.finditer(text):
            tables.append(re.sub(r"\s+", " ", m.group(1)).strip()[:160])
        rows = TABULAR_ROW_RE.findall(text)
        if rows:
            tables.append(f"{len(rows)} tabular numeric row(s) detected")
        # Dedupe.
        return list(dict.fromkeys(tables))[:12]

    def _extract_mechanisms(self, text: str) -> list[str]:
        mechanisms: list[str] = []
        for sentence in SENTENCE_RE.findall(text):
            s = sentence.strip()
            if not (40 <= len(s) <= 320):
                continue
            low = s.lower()
            if any(term in low for term in MECHANISM_TERMS):
                mechanisms.append(re.sub(r"\s+", " ", s))
                if len(mechanisms) >= self.MAX_MECHANISMS:
                    break
        return mechanisms

    def _cross_reference(self, mechanisms: list[str]) -> list[str]:
        """Link new mechanisms to existing KNOWLEDGE entries by shared terms."""
        refs: list[str] = []
        seen: set[str] = set()
        for term in MECHANISM_TERMS:
            if not any(term in m.lower() for m in mechanisms):
                continue
            try:
                for row in self.memory.query(term, limit=2):
                    label = f"{row.get('category', '')}/{row.get('key_ref', '')}"
                    if label not in seen and "lit_" not in label:
                        seen.add(label)
                        refs.append(label)
            except Exception:
                continue
            if len(refs) >= 8:
                break
        return refs

    # ── memory seeding ────────────────────────────────────────────────────────

    def _seed_memory(self, record: PaperRecord) -> None:
        try:
            self.memory.remember(MemoryTier.KNOWLEDGE, f"lit_{record.slug}",
                                 f"Paper: {record.title}. {record.summary()}")
            for i, mechanism in enumerate(record.mechanisms[:12], 1):
                self.memory.remember(MemoryTier.KNOWLEDGE, f"lit_{record.slug}_mech_{i}",
                                     f"[{record.title[:60]}] {mechanism}")
        except Exception as exc:
            self.bus.log.emit(f"LIT: memory seed recovered - {first_line(exc, 80)}")

    # ── queries ───────────────────────────────────────────────────────────────

    async def query_mechanisms(self, query: str) -> ToolResult:
        query = str(query or "").strip()
        if not query:
            return ToolResult("What mechanism shall I look up, sir?", ok=False)
        # Search freshly-ingested papers plus the persistent KNOWLEDGE tier.
        hits: list[str] = []
        low = query.lower()
        for record in self._papers.values():
            for mechanism in record.mechanisms:
                if low in mechanism.lower():
                    hits.append(f"[{record.title[:50]}] {mechanism}")
        try:
            for row in await asyncio.to_thread(self.memory.query, query, 8):
                if str(row.get("key_ref", "")).startswith("lit_"):
                    hits.append(row.get("value", ""))
        except Exception:
            pass
        hits = list(dict.fromkeys(h for h in hits if h))[:10]
        if not hits:
            return ToolResult(f"No indexed mechanism matches '{query}', sir.")
        return ToolResult("Mechanism-first findings, sir:\n" + "\n".join(f"- {h}" for h in hits))

    async def generate_citation_summary(self, slug: str = "") -> ToolResult:
        record: Optional[PaperRecord] = None
        if slug:
            record = self._papers.get(slug) or next(
                (r for r in self._papers.values() if slug.lower() in r.slug or slug.lower() in r.title.lower()),
                None,
            )
        elif self._papers:
            record = list(self._papers.values())[-1]   # most recent
        if record is None:
            return ToolResult("No paper is ingested yet, sir. Use ingest_paper first.", ok=False)
        lines = [f"Citation summary — {record.title}", ""]
        if record.dois:
            lines.append("DOIs:")
            lines.extend(f"  • {doi}" for doi in record.dois[:12])
        if record.citations:
            lines.append("References:")
            lines.extend(f"  {c}" for c in record.citations[:20])
        if not record.dois and not record.citations:
            lines.append("No citations were detected in this document.")
        return ToolResult("\n".join(lines))

    # ── snapshot for the GUI ──────────────────────────────────────────────────

    def snapshot(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in list(self._papers.values())[-12:]]
