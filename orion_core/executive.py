"""
ExecutiveAssistantMode (Mark X.5) — JARVIS mode.

ORION as an executive operating partner rather than a conversational tool:
project tracking, scheduling, task prioritisation, meeting summaries,
workflow planning and progress monitoring, all grounded in the durable
cognitive state so nothing evaporates between sessions.

This layer OWNS no data and NO transport: it composes the services that do —

    CognitiveStateManager  — projects, tasks, goals, priorities (durable)
    ReminderService        — spoken time-based nudges
    NotionService          — external tasks/calendar when configured
    ProviderRouter         — drafting/summarising when a model is available
    KnowledgeGraphEngine   — meeting summaries filed into the second brain

Everything degrades gracefully: with no model, summaries and plans fall back
to deterministic extraction; with no Notion, scheduling stays local through
reminders and cognitive tasks.  All blocking work runs off the event loop.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .memory import MemoryAgent, MemoryTier
from .utils import first_line

_PRIORITY_TERMS = (
    ("urgent", 5.0), ("asap", 5.0), ("today", 4.0), ("deadline", 4.0),
    ("client", 3.0), ("payment", 3.0), ("launch", 3.0), ("blocker", 4.0),
    ("critical", 5.0), ("important", 2.5),
)


class ExecutiveAssistantMode:
    """The executive operating partner behind the 'executive' tool."""

    def __init__(
        self,
        bus: OrionBus,
        memory: MemoryAgent,
        cognition: Any,
        reminders: Any | None = None,
        notion: Any | None = None,
        router: Any | None = None,
        graph: Any | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.cognition = cognition
        self.reminders = reminders
        self.notion = notion
        self.router = router
        self.graph = graph
        self.telemetry = telemetry
        if self.telemetry is not None:
            self.telemetry.health.register("executive")

    # ── project tracking ──────────────────────────────────────────────────────

    async def track_project(self, name: str, note: str = "") -> ToolResult:
        name = str(name or "").strip()
        if not name:
            return ToolResult("A project name is required, sir.", ok=False)
        details = {"note": note[:400]} if note.strip() else {}
        await asyncio.to_thread(self.cognition.upsert_project, name, details)
        self.memory.set_active_project(name)
        if self.graph is not None:
            try:
                await asyncio.to_thread(
                    self.graph.upsert_entity, name, "project",
                )
            except Exception:
                pass
        return ToolResult(f"Project '{name}' is now under executive tracking, sir.")

    async def status(self) -> ToolResult:
        """The executive picture: projects, goals, tasks, priorities, progress."""
        state = await asyncio.to_thread(self.cognition.snapshot)
        lines = [f"Executive status — {datetime.now():%A %d %B, %H:%M}"]
        projects = state.get("active_projects") or {}
        lines.append(f"Projects tracked: {len(projects)}"
                     + (f" ({', '.join(list(projects)[:6])})" if projects else ""))
        tasks = state.get("pending_tasks") or {}
        lines.append(f"Open tasks: {len(tasks)}")
        goals = state.get("active_goals") or []
        if goals:
            lines.append(f"Active goals: {len(goals)}")
        priorities = state.get("user_priorities") or []
        if priorities:
            lines.append("Standing priorities: "
                         + "; ".join(str(p)[:60] for p in priorities[:5]))
        ranked = self._prioritised_tasks(state)[:5]
        if ranked:
            lines.append("Top of the queue:")
            lines.extend(f"  {i + 1}. {label}" for i, (label, _s) in enumerate(ranked))
        return ToolResult("\n".join(lines))

    # ── task prioritisation ───────────────────────────────────────────────────

    def _prioritised_tasks(self, state: dict[str, Any]) -> list[tuple[str, float]]:
        """Deterministic urgency ranking: due-time pressure + priority terms."""
        now = datetime.now()
        ranked: list[tuple[str, float]] = []
        for task in (state.get("pending_tasks") or {}).values():
            if not isinstance(task, dict):
                continue
            title = str(task.get("title") or "task")[:90]
            score = 1.0
            text = title.lower()
            for term, weight in _PRIORITY_TERMS:
                if term in text:
                    score += weight
            due_raw = str(task.get("due") or "").strip()
            label = title
            if due_raw:
                for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                    try:
                        due = datetime.strptime(due_raw[: len(now.strftime(fmt))], fmt)
                    except ValueError:
                        continue
                    hours = (due - now).total_seconds() / 3600.0
                    if hours < 0:
                        score += 8.0
                        label = f"{title} (OVERDUE)"
                    elif hours <= 24:
                        score += 6.0
                        label = f"{title} (due in {hours:.0f}h)"
                    elif hours <= 72:
                        score += 3.0
                        label = f"{title} (due in {hours / 24:.0f}d)"
                    break
            ranked.append((label, score))
        ranked.sort(key=lambda item: -item[1])
        return ranked

    async def prioritise(self) -> ToolResult:
        state = await asyncio.to_thread(self.cognition.snapshot)
        ranked = self._prioritised_tasks(state)
        if not ranked:
            return ToolResult("The task queue is clear, sir. Nothing to prioritise.")
        lines = ["Task prioritisation (urgency-ranked):"]
        lines.extend(
            f"{i + 1}. {label}  [score {score:.1f}]"
            for i, (label, score) in enumerate(ranked[:10])
        )
        return ToolResult("\n".join(lines))

    # ── scheduling ────────────────────────────────────────────────────────────

    async def schedule(self, title: str, when: str = "", notes: str = "") -> ToolResult:
        """Schedule locally (cognitive task + reminder) and in Notion when up."""
        title = str(title or "").strip()
        if not title:
            return ToolResult("A title is required to schedule anything, sir.", ok=False)
        outcomes: list[str] = []
        await asyncio.to_thread(self.cognition.add_task, title, self.memory.active_project, when)
        outcomes.append("tracked in cognitive state")
        if when.strip() and self.reminders is not None:
            try:
                result = self.reminders.add(f"remind me at {when} to {title}", at=when)
                if getattr(result, "ok", False):
                    outcomes.append("spoken reminder set")
            except Exception as exc:
                self.bus.log.emit(f"EXEC: reminder skipped - {first_line(exc)}")
        if self.notion is not None and getattr(self.notion, "available", False):
            try:
                result = await self.notion.create_task(title, due=when, notes=notes)
                if getattr(result, "ok", False):
                    outcomes.append("filed in Notion")
            except Exception as exc:
                self.bus.log.emit(f"EXEC: Notion scheduling skipped - {first_line(exc)}")
        return ToolResult(f"Scheduled '{title}'" + (f" for {when}" if when else "")
                          + " — " + ", ".join(outcomes) + ".")

    # ── meeting summaries ─────────────────────────────────────────────────────

    async def summarise_meeting(self, transcript: str, title: str = "") -> ToolResult:
        transcript = str(transcript or "").strip()
        if len(transcript) < 40:
            return ToolResult(
                "I need the meeting transcript or notes text to summarise, sir.",
                ok=False,
            )
        title = str(title or "").strip() or f"Meeting {datetime.now():%d %b %Y %H:%M}"
        summary = ""
        if self.router is not None and getattr(self.router, "has_text_fallback", lambda: False)():
            try:
                _profile, summary = await self.router.generate_text(
                    "Summarise this meeting into: decisions, action items (with "
                    "owners where stated), and open questions. Be concise. "
                    f"British English.\n\n{transcript[:8000]}",
                    system_extra="You write disciplined executive meeting minutes.",
                )
            except Exception as exc:
                self.bus.log.emit(f"EXEC: model summary failed - {first_line(exc)}")
        if not summary:
            # Extractive fallback: decision/action-bearing sentences.
            picks = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript)
                if re.search(r"\b(decid|agree|will|action|due|next step|owner)\b", s, re.I)
            ][:8]
            summary = ("Key points (extracted offline):\n"
                       + "\n".join(f"- {p[:200]}" for p in picks)) if picks \
                      else f"Summary unavailable offline; transcript stored ({len(transcript)} chars)."
        await asyncio.to_thread(
            self.memory.remember, MemoryTier.KNOWLEDGE,
            f"meeting_{datetime.now():%Y%m%d_%H%M}", f"{title}: {summary[:900]}",
        )
        if self.graph is not None:
            try:
                await asyncio.to_thread(
                    self.graph.ingest_record, "meeting", title, summary,
                    {"title": title},
                )
            except Exception:
                pass
        return ToolResult(f"Meeting summary — {title}:\n{summary}")

    # ── workflow planning ─────────────────────────────────────────────────────

    async def plan_workflow(self, objective: str) -> ToolResult:
        objective = str(objective or "").strip()
        if not objective:
            return ToolResult("An objective is required to plan a workflow, sir.", ok=False)
        state = await asyncio.to_thread(self.cognition.snapshot)
        context = (
            f"Active project: {self.memory.active_project or 'none'}. "
            f"Open tasks: {len(state.get('pending_tasks') or {})}. "
            f"Priorities: {'; '.join(str(p)[:50] for p in (state.get('user_priorities') or [])[:3])}"
        )
        plan = ""
        if self.router is not None and getattr(self.router, "has_text_fallback", lambda: False)():
            try:
                _profile, plan = await self.router.generate_text(
                    f"Plan the workflow for: {objective}\nContext: {context}\n"
                    "Produce 4-8 ordered steps, each with a completion criterion. "
                    "British English, terse.",
                    system_extra="You are an operations planner.",
                )
            except Exception as exc:
                self.bus.log.emit(f"EXEC: model planning failed - {first_line(exc)}")
        if not plan:
            plan = (
                "1. Define the outcome and its acceptance criteria.\n"
                "2. List constraints, owners and required inputs.\n"
                "3. Break the work into ordered, verifiable steps.\n"
                "4. Schedule the first step and set its reminder.\n"
                "5. Review progress at each completion and adjust."
            )
        await asyncio.to_thread(
            self.cognition.upsert_workflow, objective[:80], {"plan": plan[:1500]}
        )
        return ToolResult(f"Workflow plan — {objective}:\n{plan}")

    # ── progress monitoring ───────────────────────────────────────────────────

    async def progress(self) -> ToolResult:
        state = await asyncio.to_thread(self.cognition.snapshot)
        goals = state.get("goals") or {}
        done = sum(1 for g in goals.values()
                   if isinstance(g, dict) and g.get("status") == "completed")
        active = sum(1 for g in goals.values()
                     if isinstance(g, dict) and g.get("status") == "active")
        workflows = state.get("open_workflows") or {}
        lines = [
            "Progress report:",
            f"- Goals: {active} active, {done} completed, {len(goals)} total",
            f"- Open workflows: {len(workflows)}",
            f"- Open tasks: {len(state.get('pending_tasks') or {})}",
        ]
        if workflows:
            lines.append("- Workflows in flight: "
                         + "; ".join(list(workflows)[:5]))
        return ToolResult("\n".join(lines))
