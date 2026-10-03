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
from datetime import datetime, timedelta
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
        outlook: Any | None = None,
    ) -> None:
        self.bus = bus
        self.outlook = outlook
        #: The slots last proposed by find_time, for "book option 2".
        self._proposals: list[tuple[Any, str]] = []
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
            return ToolResult("A project name is required.", ok=False)
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
        return ToolResult(f"Project '{name}' is now under executive tracking.")

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
            return ToolResult("The task queue is clear. Nothing to prioritise.")
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
            return ToolResult("A title is required to schedule anything.", ok=False)
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

    # ── finding time (conflict-aware scheduling, §2.4) ────────────────────────

    async def find_time(self, minutes: int = 60, days: int = 7, title: str = "",
                        deadline: str = "", hours: str = "",
                        weekends: bool = False) -> ToolResult:
        """Propose free slots across every calendar ORION can read.

        Never books: the slots are offered, and only "book option N" schedules
        one (brief §2.4 keeps this confirm-tier). Calendars that could not be
        read are named, because "you are free" from a calendar that was never
        consulted is the one wrong answer here that costs a double-booking.
        """
        from .calendar_sources import configured_feeds, describe_feed, parse_ics, read_feed
        from .scheduling import DEFAULT_WORK_HOURS, DEFAULT_WORKDAYS, find_slots
        from .time_service import TIME

        try:
            minutes = max(15, min(8 * 60, int(float(minutes or 60))))
            days = max(1, min(31, int(float(days or 7))))
        except (TypeError, ValueError):
            return ToolResult("How long do you need, in minutes?", ok=False)
        now = TIME.now()
        zone = now.tzinfo
        window_end = now + timedelta(days=days)
        work_hours = _parse_hours(hours) or DEFAULT_WORK_HOURS
        workdays = frozenset(range(7)) if weekends else DEFAULT_WORKDAYS
        hard_deadline = _parse_deadline(deadline, zone)
        if deadline.strip() and hard_deadline is None:
            # A misread deadline proposes slots after it — ask instead.
            return ToolResult(f"I couldn't read the deadline '{deadline}'. "
                              "Give it as a date, like 2026-10-09.", ok=False)

        busy: list[Any] = []
        consulted: list[str] = []
        unread: list[str] = []
        for feed in configured_feeds():
            name = describe_feed(feed)
            try:
                text = await asyncio.to_thread(read_feed, feed)
                blocks = parse_ics(text, now, window_end, zone, source=name)
                busy.extend(blocks)
                consulted.append(f"{name} ({len(blocks)} busy)")
            except Exception as exc:
                unread.append(f"{name}: {first_line(exc, 80)}")
        for label, service in (("Outlook", self.outlook), ("Notion", self.notion)):
            if service is None or not getattr(service, "available", False):
                continue
            try:
                result = await service.busy_blocks(now, window_end, zone)
            except Exception as exc:
                unread.append(f"{label}: {first_line(exc, 80)}")
                continue
            if result.ok:
                blocks = [item["block"] for item in (result.evidence or [])]
                busy.extend(blocks)
                consulted.append(f"{label} ({len(blocks)} busy)")
            else:
                unread.append(f"{label}: {first_line(result.text, 80)}")

        slots = find_slots([(b.start, b.end) for b in busy], now, window_end,
                           timedelta(minutes=minutes), work_hours=work_hours,
                           workdays=workdays, deadline=hard_deadline, limit=3)
        label = title.strip() or "that"
        self._proposals = [(slot, title.strip()) for slot in slots]
        span = (f"before {hard_deadline:%a %d %b}" if hard_deadline
                else f"in the next {days} day(s)")
        if not slots:
            lines = [f"I can't find {minutes} free minutes for {label} {span} "
                     f"within {work_hours[0]:02d}:00–{work_hours[1]:02d}:00."]
        else:
            lines = [f"Free {minutes}-minute slots for {label} {span}:"]
            for number, slot in enumerate(slots, 1):
                lines.append(f"{number}. {slot.start:%a %d %b, %H:%M}–{slot.end:%H:%M}")
        if consulted:
            lines.append("Checked: " + ", ".join(consulted) + ".")
        else:
            lines.append("No calendar could be read, so these are free by working "
                         "hours alone — add an ICS feed to config/calendars.json, or "
                         "open Outlook, for real availability.")
        if unread:
            lines.append("Not checked: " + "; ".join(unread) + ".")
        if slots:
            lines.append("Nothing is booked — say 'book option 1' (or 2, 3) to schedule one.")
        return ToolResult("\n".join(lines), evidence=[
            {"start": s.start.isoformat(), "end": s.end.isoformat()} for s in slots])

    async def book_slot(self, option: int = 1, title: str = "") -> ToolResult:
        """Schedule one of the slots find_time last proposed."""
        if not self._proposals:
            return ToolResult("There are no proposed slots — ask me to find time first.",
                              ok=False)
        try:
            index = int(option or 1) - 1
        except (TypeError, ValueError):
            index = -1
        if not 0 <= index < len(self._proposals):
            return ToolResult(f"Choose option 1 to {len(self._proposals)}.", ok=False)
        slot, proposed_title = self._proposals[index]
        name = title.strip() or proposed_title or "Focus time"
        result = await self.schedule(name, slot.start.strftime("%Y-%m-%d %H:%M"),
                                     f"Until {slot.end:%H:%M}.")
        if result.ok:
            self._proposals = []
        return result

    # ── meeting summaries ─────────────────────────────────────────────────────

    async def summarise_meeting(self, transcript: str, title: str = "") -> ToolResult:
        transcript = str(transcript or "").strip()
        if len(transcript) < 40:
            return ToolResult(
                "I need the meeting transcript or notes text to summarise.",
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
            return ToolResult("An objective is required to plan a workflow.", ok=False)
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


def _parse_hours(text: str) -> tuple[int, int] | None:
    """'8-17', '08:00–17:30' or '9 to 6pm' → (start_hour, end_hour)."""
    numbers = re.findall(r"(\d{1,2})(?::\d{2})?\s*(am|pm)?", str(text or "").lower())
    if len(numbers) < 2:
        return None
    hours = []
    for value, meridiem in numbers[:2]:
        hour = int(value) % 24
        if meridiem == "pm" and hour < 12:
            hour += 12
        hours.append(hour)
    start, end = hours
    if end <= start and end + 12 <= 24 and end + 12 > start:
        end += 12                               # "9 to 6" means 09:00–18:00
    return (start, end) if 0 <= start < end <= 24 else None


def _parse_deadline(text: str, zone: Any) -> datetime | None:
    """An ISO date or date-time; a bare date means the end of that day."""
    text = str(text or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=zone)
    if len(text) <= 10:
        moment = moment.replace(hour=23, minute=59)
    return moment
