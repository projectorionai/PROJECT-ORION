"""
ProactiveReportingService (Mark X.5) — scheduled intelligence reports.

ORION writes his own reports without being asked:

    Morning Briefing               — on demand (delivery stays consent-gated
                                     through the live worker; this service
                                     only PERSISTS a written copy)
    Daily Business Report          — every day at the configured hour
    Weekly Product Intelligence    — Sundays
    Monthly Growth Report          — the 1st of each month

Each report gathers whatever local sources are healthy — proactive findings,
commerce research, the agency pipeline, cognitive state, priority email
(attach-only; Outlook is never launched for a scheduled report) and calendar
— then writes an executive document through the DocumentExporterService, so
every report lands on disk (Markdown + HTML + DOCX) with export history.

Schedule bookkeeping persists in ``config/reporting.json`` so a restart never
double-generates and a machine that slept through a slot catches up on the
next cycle.  Content generation prefers the model router; with no model the
report is assembled extractively from the same data — offline never means
report-less.  Nothing here speaks: completed reports surface as a dashboard
event and a log line.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .utils import first_line, utc_stamp

STATE_PATH = CONFIG_DIR / "reporting.json"

REPORT_KINDS = ("daily_business", "weekly_product", "monthly_growth")


class ProactiveReportingService:
    """Generates and stores scheduled executive reports, fully offline-capable."""

    CHECK_INTERVAL_S = 600.0        # schedule sweep cadence
    FIRST_RUN_DELAY_S = 120.0
    DAILY_HOUR = 18                 # daily business report after 18:00 local
    WEEKLY_DAY = 6                  # Sunday (date.weekday())
    MONTHLY_DAY = 1                 # 1st of the month

    def __init__(
        self,
        bus: OrionBus,
        exporter: Any,
        router: Any | None = None,
        proactive: Any | None = None,
        commerce: Any | None = None,
        pipeline: Any | None = None,
        cognition: Any | None = None,
        outlook: Any | None = None,
        notion: Any | None = None,
        memory: Any | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.bus = bus
        self.exporter = exporter            # DocumentExporterService
        self.router = router
        self.proactive = proactive
        self.commerce = commerce            # CommerceSuite
        self.pipeline = pipeline            # AgencyPipelineService
        self.cognition = cognition          # CognitiveStateManager
        self.outlook = outlook
        self.notion = notion
        self.memory = memory
        self.telemetry = telemetry
        self._stop = asyncio.Event()
        self._state = self._load_state()
        if self.telemetry is not None:
            self.telemetry.health.register("reporting")

    # ── schedule bookkeeping ──────────────────────────────────────────────────

    def _load_state(self) -> dict[str, str]:
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _slot(self, kind: str, now: datetime) -> str:
        """The schedule slot identifier a run of *kind* would satisfy now."""
        if kind == "daily_business":
            return now.strftime("%Y-%m-%d")
        if kind == "weekly_product":
            return f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
        return now.strftime("%Y-%m")        # monthly_growth

    def _due(self, kind: str, now: datetime) -> bool:
        if self._state.get(kind) == self._slot(kind, now):
            return False
        if kind == "daily_business":
            return now.hour >= self.DAILY_HOUR
        if kind == "weekly_product":
            return now.weekday() == self.WEEKLY_DAY and now.hour >= self.DAILY_HOUR
        return now.day == self.MONTHLY_DAY and now.hour >= 9

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Background schedule loop; cancel-safe, one report at a time."""
        try:
            await asyncio.sleep(self.FIRST_RUN_DELAY_S)
            while not self._stop.is_set():
                now = datetime.now()
                for kind in REPORT_KINDS:
                    if self._stop.is_set():
                        break
                    if not self._due(kind, now):
                        continue
                    try:
                        await self.generate(kind)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self.bus.log.emit(
                            f"REPORT: scheduled {kind} recovered - {first_line(exc)}"
                        )
                if self.telemetry is not None:
                    self.telemetry.health.beat("reporting", "OK", "schedule watched")
                await asyncio.sleep(self.CHECK_INTERVAL_S)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()

    # ── generation ────────────────────────────────────────────────────────────

    async def generate(self, kind: str) -> ToolResult:
        """Generate a report now (scheduled or on demand) and persist it."""
        kind = str(kind or "").strip().lower()
        if kind in {"daily", "business", "daily_business_report"}:
            kind = "daily_business"
        elif kind in {"weekly", "product", "product_intelligence"}:
            kind = "weekly_product"
        elif kind in {"monthly", "growth", "growth_report"}:
            kind = "monthly_growth"
        if kind not in REPORT_KINDS:
            return ToolResult(
                f"Unknown report kind '{kind}'. Use daily_business, "
                "weekly_product, or monthly_growth.",
                ok=False,
            )
        now = datetime.now()
        title, sections = await self._compose(kind, now)
        summary = await self._executive_summary(title, sections)
        result = await self.exporter.export_report(title, summary, sections)
        if result.ok:
            self._state[kind] = self._slot(kind, now)
            await asyncio.to_thread(self._save_state)
            self.bus.dashboard_event.emit(
                "report", {"kind": kind, "title": title, "at": utc_stamp()}
            )
            self.bus.log.emit(f"REPORT: {title} generated and stored.")
            if self.telemetry is not None:
                self.telemetry.metrics.incr(f"report.{kind}")
        return result

    async def _compose(self, kind: str, now: datetime) -> tuple[str, list[dict[str, str]]]:
        """Gather sections from every healthy local source; each may fail alone."""
        sections: list[dict[str, str]] = []

        async def _add(heading: str, coro: Any) -> None:
            try:
                text = await coro
            except Exception as exc:
                text = ""
                self.bus.log.emit(f"REPORT: '{heading}' source skipped - {first_line(exc)}")
            if text and str(text).strip():
                sections.append({"heading": heading, "content": str(text).strip()})

        if kind == "daily_business":
            title = f"Daily Business Report — {now:%A %d %B %Y}"
            await _add("Attention items", self._proactive_findings())
            await _add("Priority email", self._priority_email())
            await _add("Calendar", self._calendar())
            await _add("Pipeline", self._pipeline_snapshot())
            await _add("Project state", self._project_state())
        elif kind == "weekly_product":
            title = f"Weekly Product Intelligence — week {now.isocalendar().week}, {now:%Y}"
            await _add("Product research this week", self._product_research())
            await _add("Opportunity scores", self._opportunities())
            await _add("Channel intelligence", self._channel_intel())
        else:
            title = f"Monthly Growth Report — {now:%B %Y}"
            await _add("Pipeline", self._pipeline_snapshot())
            await _add("Product research", self._product_research())
            await _add("Project state", self._project_state())
            await _add("Growth recommendations", self._growth_recommendations())
        if not sections:
            sections.append({
                "heading": "Status",
                "content": "No source produced material for this period — all "
                           "integrations were unavailable or quiet.",
            })
        return title, sections

    async def _executive_summary(self, title: str, sections: list[dict[str, str]]) -> str:
        """LLM summary when a model is up; extractive first-lines otherwise."""
        corpus = "\n\n".join(f"{s['heading']}:\n{s['content'][:800]}" for s in sections)
        if self.router is not None and getattr(self.router, "has_text_fallback", lambda: False)():
            try:
                _profile, text = await self.router.generate_text(
                    f"Write a 3-5 sentence executive summary for '{title}' from this "
                    f"material. Lead with what needs action. British English.\n\n{corpus[:5000]}",
                    system_extra="You write crisp executive summaries.",
                )
                return text
            except Exception as exc:
                self.bus.log.emit(f"REPORT: summary via model failed - {first_line(exc)}")
        return "\n".join(
            f"• {s['heading']}: {s['content'].splitlines()[0][:160]}" for s in sections
        )

    # ── sources (each returns plain text or '') ───────────────────────────────

    async def _proactive_findings(self) -> str:
        if self.proactive is None:
            return ""
        result = await self.proactive.check_now()
        return result.text if result.ok else ""

    async def _priority_email(self) -> str:
        # Attach-only: a scheduled report must never launch Outlook.
        if self.outlook is None or not getattr(self.outlook, "available", False):
            return ""
        result = await self.outlook.priority_emails(limit=5)
        return result.text if result.ok else ""

    async def _calendar(self) -> str:
        if self.notion is None or not getattr(self.notion, "available", False):
            return ""
        result = await self.notion.upcoming_events(days=2, limit=10)
        return result.text if result.ok else ""

    async def _pipeline_snapshot(self) -> str:
        if self.pipeline is None:
            return ""
        result = await asyncio.to_thread(self.pipeline.snapshot_text)
        return result.text if result.ok else ""

    async def _project_state(self) -> str:
        if self.cognition is None:
            return ""
        state = await asyncio.to_thread(self.cognition.snapshot)
        lines: list[str] = []
        projects = state.get("active_projects") or {}
        if projects:
            lines.append(f"Active projects ({len(projects)}): " + ", ".join(list(projects)[:8]))
        tasks = state.get("pending_tasks") or {}
        if tasks:
            lines.append(f"Open tasks: {len(tasks)}")
        priorities = state.get("user_priorities") or []
        if priorities:
            lines.append("Priorities: " + "; ".join(str(p)[:60] for p in priorities[:5]))
        return "\n".join(lines)

    async def _product_research(self) -> str:
        if self.commerce is None or self.memory is None:
            return ""
        rows = await asyncio.to_thread(self.commerce.dropship.research_log, 10)
        if not rows:
            return ""
        return "\n".join(
            f"- {r.get('key_ref', '?')}: {r.get('value', '')[:160]}" for r in rows
        )

    async def _opportunities(self) -> str:
        if self.commerce is None:
            return ""
        result = await self.commerce.tiktok.trend_report()
        return result.text[:2500] if result.ok else ""

    async def _channel_intel(self) -> str:
        if self.commerce is None:
            return ""
        result = await self.commerce.instagram.weekly_report()
        return result.text[:2500] if result.ok else ""

    async def _growth_recommendations(self) -> str:
        if self.commerce is None:
            return ""
        result = await self.commerce.advisor.growth_plan("coming month")
        return result.text[:3000] if result.ok else ""
