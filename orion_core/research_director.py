"""
ResearchDirector (Phase 3) — persistent, multi-topic autonomous research.

The existing ResearchAgent runs one time-boxed topic and writes an organised
folder. The director sits above it and gives ORION a research PROGRAMME:

    - a persistent agenda of topics (queued → running → complete → reviewed)
      stored in cognitive state, so research survives restarts;
    - automatic advancement: when one topic finishes, the next queued topic
      starts without being asked;
    - knowledge acquisition: every completed run is harvested — factual
      sentences from the summary and notes become claims in the
      EvidenceEngine (with provenance and confidence) and a record in the
      knowledge graph, so research genuinely expands what ORION knows;
    - findings on demand: claims, confidence and detected contradictions
      for any topic, ready for a briefing.

Composes: ResearchAgent (the runner), CognitiveStateManager (the durable
agenda), EvidenceEngine (claims), KnowledgeGraphEngine (optional). Owns no
model calls of its own; everything degrades gracefully offline.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .evidence import EvidenceEngine, score_confidence
from .utils import first_line

_FACT_MIN_LEN = 40
_FACT_MARKERS = re.compile(
    r"\b(is|are|was|were|has|have|will|can|show|found|according|percent|%|\d)\b", re.I)

_URL_PATTERN = re.compile(r"https?://([^\s/\)\]>\"']+)", re.I)

# Domain-credibility tiers for source validation. A claim is only as strong
# as where it came from; these priors weight the wording-based confidence.
_SOURCE_TIERS: tuple[tuple[float, str, tuple[str, ...]], ...] = (
    (0.90, "primary/institutional",
     (".gov", ".edu", ".ac.uk", ".nhs.uk", "nature.com", "science.org",
      "nih.gov", "pubmed", "who.int", "reuters.com", "bbc.co.uk", "ft.com",
      "ons.gov.uk")),
    (0.75, "reference/technical",
     ("wikipedia.org", "arxiv.org", "github.com", "docs.", "developer.",
      "stackoverflow.com", "ieee.org", "acm.org")),
    (0.35, "social/forum",
     ("reddit.com", "quora.com", "tiktok.com", "twitter.com", "x.com",
      "facebook.com", "instagram.com", "pinterest.", "forum")),
)
_DEFAULT_SOURCE_SCORE = 0.60


class SourceValidator:
    """Domain-tier credibility scoring for research sources."""

    @staticmethod
    def credibility(url_or_domain: str) -> tuple[float, str]:
        domain = str(url_or_domain or "").strip().lower()
        match = _URL_PATTERN.search(domain)
        if match:
            domain = match.group(1)
        for score, label, needles in _SOURCE_TIERS:
            if any(n in domain for n in needles):
                return score, label
        return _DEFAULT_SOURCE_SCORE, "general web"

    @classmethod
    def assess_text(cls, text: str) -> dict[str, Any]:
        """Score every URL in *text*; returns per-domain results + the mean."""
        seen: dict[str, tuple[float, str]] = {}
        for match in _URL_PATTERN.finditer(str(text or "")):
            domain = match.group(1).lower()
            if domain not in seen:
                seen[domain] = cls.credibility(domain)
        if not seen:
            return {"sources": {}, "mean": _DEFAULT_SOURCE_SCORE}
        mean = sum(s for s, _ in seen.values()) / len(seen)
        return {"sources": seen, "mean": round(mean, 2)}


class ResearchDirector:
    """Persistent research programme on top of the single-run ResearchAgent."""

    def __init__(
        self,
        bus: OrionBus,
        research: Any,
        cognition: Any,
        evidence: EvidenceEngine,
        graph: Any | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.bus = bus
        self.research = research
        self.cognition = cognition
        self.evidence = evidence
        self.graph = graph
        self.telemetry = telemetry
        self.validator = SourceValidator()
        self.missions = None            # attached after MissionEngine wiring
        self.auto_advance = True
        if research is not None:
            research.on_complete = self._on_run_complete
        if telemetry is not None:
            telemetry.health.register("research_director")

    # ── agenda management ─────────────────────────────────────────────────────

    async def queue_topic(self, topic: str, minutes: float = 20.0) -> ToolResult:
        topic = str(topic or "").strip()
        if not topic:
            return ToolResult("What should go on the research agenda?", ok=False)
        minutes = max(1.0, min(180.0, float(minutes)))
        record = await asyncio.to_thread(
            self.cognition.upsert_research_session, topic,
            {"status": "queued", "minutes": minutes, "reviewed": False})
        started = await self._advance()
        if started == record.get("topic"):
            return ToolResult(f"Research on '{topic}' is underway now.")
        return ToolResult(
            f"'{topic}' is on the research agenda ({minutes:.0f} min). "
            "I'll start it as soon as the current run finishes.")

    async def agenda(self) -> ToolResult:
        sessions = await self._sessions()
        if not sessions:
            return ToolResult("The research agenda is empty.")
        order = {"running": 0, "queued": 1, "complete": 2, "reviewed": 3}
        rows = sorted(sessions.values(),
                      key=lambda s: order.get(str(s.get("status")), 4))
        lines = ["Research agenda:"]
        for s in rows[:12]:
            status = str(s.get("status") or "queued")
            extra = ""
            if status == "complete" and not s.get("reviewed"):
                extra = " — awaiting review"
            lines.append(f"- [{status}] {str(s.get('topic'))[:70]}{extra}")
        return ToolResult("\n".join(lines))

    async def mark_reviewed(self, topic: str) -> ToolResult:
        topic = str(topic or "").strip()
        if not topic:
            return ToolResult("Which topic shall I mark as reviewed?", ok=False)
        await asyncio.to_thread(
            self.cognition.upsert_research_session, topic,
            {"status": "reviewed", "reviewed": True})
        return ToolResult(f"Research on '{topic}' marked as reviewed.")

    async def resume_pending(self) -> str:
        """Called at launch: restart the agenda where the last session left off."""
        # A run that was 'running' when ORION shut down never finished — requeue it.
        sessions = await self._sessions()
        for s in sessions.values():
            if str(s.get("status")) == "running":
                await asyncio.to_thread(
                    self.cognition.upsert_research_session,
                    str(s.get("topic") or ""), {"status": "queued"})
        return await self._advance()

    # ── findings ──────────────────────────────────────────────────────────────

    async def findings(self, topic: str) -> ToolResult:
        topic = str(topic or "").strip()
        if not topic:
            return ToolResult("Which topic's findings would you like?", ok=False)
        claims = await asyncio.to_thread(self.evidence.claims_for, topic, 12)
        if not claims:
            return ToolResult(
                f"No harvested findings for '{topic}' yet. Queue it for research "
                "and I'll build the evidence base.")
        conflicts = await asyncio.to_thread(self.evidence.detect_contradictions, topic)
        lines = [f"Findings — {topic}:"]
        for c in claims[:8]:
            lines.append(f"- ({c['confidence']:.0%}) {c['claim'][:180]}  [{c['source'][:50]}]")
        if conflicts:
            lines.append(f"Contradictions detected ({len(conflicts)}):")
            for x in conflicts[:3]:
                lines.append(f"- '{x['claim_a'][:80]}' vs '{x['claim_b'][:80]}'")
        else:
            lines.append("No contradictions detected across sources.")
        return ToolResult("\n".join(lines))

    async def validate_sources(self, text: str) -> ToolResult:
        """Credibility report for every URL in *text* (or a topic's sources)."""
        text = str(text or "").strip()
        if not text:
            return ToolResult("Paste the sources (or text containing URLs) "
                              "to validate.", ok=False)
        report = await asyncio.to_thread(self.validator.assess_text, text)
        sources = report["sources"]
        if not sources:
            return ToolResult("No URLs found in that text.")
        lines = [f"Source validation — {len(sources)} domain(s), "
                 f"mean credibility {report['mean']:.0%}:"]
        for domain, (score, label) in sorted(
                sources.items(), key=lambda p: -p[1][0])[:12]:
            lines.append(f"- {score:.0%} [{label}] {domain[:60]}")
        return ToolResult("\n".join(lines))

    # ── strategic opportunity scanning ────────────────────────────────────────

    async def opportunities(self) -> ToolResult:
        """Where should research go next? Gaps, conflicts and stale ground."""
        sessions = await self._sessions()
        lines: list[str] = []
        # 1 contradictions worth resolving (strongest signal there is).
        for s in list(sessions.values())[:10]:
            topic = str(s.get("topic") or "")
            if str(s.get("status")) not in {"complete", "reviewed"} or not topic:
                continue
            conflicts = await asyncio.to_thread(
                self.evidence.detect_contradictions, topic)
            if conflicts:
                lines.append(f"- Resolve {len(conflicts)} contradiction(s) in "
                             f"'{topic[:50]}' — the evidence disagrees.")
        # 2 completed research never reviewed by the user.
        unreviewed = [str(s.get("topic")) for s in sessions.values()
                      if str(s.get("status")) == "complete"
                      and not s.get("reviewed")]
        if unreviewed:
            lines.append("- Review completed research: "
                         + ", ".join(t[:40] for t in unreviewed[:3]) + ".")
        # 3 missions with no research base at all.
        if self.missions is not None:
            try:
                snap = await asyncio.to_thread(self.missions.panel_snapshot)
                researched = {str(s.get("topic", "")).lower()
                              for s in sessions.values()}
                for m in snap.get("missions", []):
                    if (m.get("status") == "active"
                            and "no research linked" in " ".join(m.get("risks", []))
                            and m["name"].lower() not in researched):
                        lines.append(f"- Mission '{m['name']}' has no research "
                                     "behind it — queue a foundation topic.")
            except Exception:
                pass
        if not lines:
            return ToolResult("No pressing research opportunities — the "
                              "evidence base is consistent and current.")
        return ToolResult("Research opportunities:\n" + "\n".join(lines[:8]))

    # ── the supervised lifecycle ──────────────────────────────────────────────

    async def _advance(self) -> str:
        """Start the next queued topic if the runner is idle. Returns its topic."""
        if not self.auto_advance or self.research is None:
            return ""
        active = getattr(self.research, "_active", {})
        task = active.get("task")
        if task is not None and not task.done():
            return ""
        sessions = await self._sessions()
        queued = [s for s in sessions.values() if str(s.get("status")) == "queued"]
        if not queued:
            return ""
        queued.sort(key=lambda s: str(s.get("updated_at") or ""))
        nxt = queued[0]
        topic = str(nxt.get("topic") or "")
        minutes = float(nxt.get("minutes") or 20.0)
        result = self.research.start_research(topic, minutes=minutes)
        if getattr(result, "ok", False):
            await asyncio.to_thread(
                self.cognition.upsert_research_session, topic, {"status": "running"})
            self.bus.log.emit(f"DIRECTOR: research advanced to '{topic}'.")
            return topic
        return ""

    async def _on_run_complete(self, topic: str, folder: Path) -> None:
        """ResearchAgent completion hook: harvest, record, advance."""
        try:
            harvested = await asyncio.to_thread(self.harvest_folder, topic, folder)
            await asyncio.to_thread(
                self.cognition.upsert_research_session, topic,
                {"status": "complete", "folder": str(folder), "claims": harvested})
            self.bus.dashboard_event.emit(
                "research_director",
                {"topic": topic, "status": "complete", "claims": harvested})
            if self.telemetry is not None:
                self.telemetry.metrics.incr("research.director.completed")
        except Exception as exc:
            self.bus.log.emit(f"DIRECTOR: harvest fault - {first_line(exc)}")
        await self._advance()

    # ── knowledge acquisition ─────────────────────────────────────────────────

    def harvest_folder(self, topic: str, folder: Path) -> int:
        """Turn a completed research folder into evidence claims + graph records."""
        harvested = 0
        for name in ("SUMMARY.md", "sources.md"):
            path = folder / name
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore")
                # Source validation: the credibility of the URLs behind a file
                # sets the confidence prior for every claim harvested from it.
                base = self.validator.assess_text(text)["mean"] \
                    if name == "sources.md" else 0.6
                harvested += self._harvest_text(
                    topic, text, source=f"research:{folder.name}/{name}",
                    base_confidence=base)
        notes_dir = folder / "notes"
        if notes_dir.exists():
            for note in sorted(notes_dir.glob("*.md"))[:12]:
                harvested += self._harvest_text(
                    topic, note.read_text(encoding="utf-8", errors="ignore"),
                    source=f"research:{folder.name}/notes/{note.name}")
        if self.graph is not None and harvested:
            try:
                self.graph.ingest_record(
                    "research", topic,
                    f"Autonomous research completed: {harvested} claim(s) harvested "
                    f"into the evidence store from {folder.name}.",
                    {"topic": topic, "folder": folder.name})
            except Exception:
                pass
        return harvested

    def _harvest_text(self, topic: str, text: str, source: str,
                      base_confidence: float = 0.6) -> int:
        count = 0
        for raw in re.split(r"(?<=[.!?])\s+", text):
            sentence = re.sub(r"[#*_`>\[\]]+", "", raw).strip()
            if len(sentence) < _FACT_MIN_LEN or sentence.startswith(("_", "-")):
                continue
            if "[Model unavailable" in sentence or not _FACT_MARKERS.search(sentence):
                continue
            if self.evidence.record_claim(
                    topic, sentence, source,
                    confidence=score_confidence(sentence, base=base_confidence)):
                count += 1
            if count >= 25:
                break
        return count

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _sessions(self) -> dict[str, Any]:
        state = await asyncio.to_thread(self.cognition.snapshot)
        sessions = state.get("research_sessions") or {}
        return {k: v for k, v in sessions.items() if isinstance(v, dict)}


__all__ = ["ResearchDirector", "SourceValidator"]
