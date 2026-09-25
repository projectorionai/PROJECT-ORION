"""
DynamicBriefingEngine (Phase 3) — real-time adaptive briefings.

The MorningBriefingService composes the classic wake-up brief (news, markets,
calendar). This engine makes briefing a CONTINUOUS capability driven by what
the system is actually doing:

    ContextMonitor   — tracks what changed in cognitive state since the last
                       briefing (tasks completed/added, projects touched)
    EventMonitor     — accumulates noteworthy system events (research
                       complete, workflow finished, security alerts, agent
                       discoveries) as they happen on the bus
    PriorityMonitor  — watches deadlines; anything falling due soon becomes
                       briefing-worthy on its own

Briefings adapt to the time of day (morning / midday / evening) and to the
accumulated events — "While you were away I completed research on X and two
workflows finished." Everything is deterministic and offline; the classic
news briefing remains the morning service's job and is referenced, not
duplicated.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .executive_core import _age_days, _parse_stamp

def _clock_now() -> datetime:
    """ORION's clock (TimeService), which also decides the briefing's part of
    the day: on a machine set to another zone the header's period and time
    otherwise came from two different clocks."""
    from .time_service import TIME
    return TIME.now()


_EVENT_LIMIT = 60
_DEADLINE_HOURS = 24.0

# Channels on the dashboard_event bus that are worth telling the user about.
_NOTEWORTHY = {
    "research": "research",
    "research_director": "research",
    "workflow": "workflow",
    "security": "security",
    "forge": "capability",
}


class EventMonitor:
    """Accumulates noteworthy system events for the next briefing."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def observe(self, channel: str, payload: Any) -> None:
        kind = _NOTEWORTHY.get(str(channel or ""))
        if kind is None or not isinstance(payload, dict):
            return
        status = str(payload.get("status") or "")
        # Only terminal states are briefing-worthy; progress ticks are noise.
        if status and status not in {"done", "complete", "paper", "failed",
                                     "alert", "found"} \
                and not status.startswith("failed"):
            return
        self.events.append({
            "kind": kind,
            "detail": str(payload.get("topic") or payload.get("workflow")
                          or payload.get("name") or payload.get("detail") or "")[:120],
            "status": status or "noted",
            "at": time.time(),
        })
        del self.events[:-_EVENT_LIMIT]

    def drain(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events


class ContextMonitor:
    """What changed in cognitive state since the last briefing."""

    def __init__(self, cognition: Any) -> None:
        self.cognition = cognition
        self._last: dict[str, Any] | None = None

    def delta(self) -> dict[str, Any]:
        state = self.cognition.snapshot()
        current = {
            "tasks": set((state.get("pending_tasks") or {}).keys()),
            "projects": set((state.get("active_projects") or {}).keys()),
            "goals": set(
                k for k, g in (state.get("goals") or {}).items()
                if isinstance(g, dict) and str(g.get("status", "active")) == "active"),
        }
        previous, self._last = self._last, current
        if previous is None:
            return {"first": True, "state": state}
        return {
            "first": False,
            "state": state,
            "tasks_added": len(current["tasks"] - previous["tasks"]),
            "tasks_cleared": len(previous["tasks"] - current["tasks"]),
            "projects_added": sorted(current["projects"] - previous["projects"]),
            "goals_completed": len(previous["goals"] - current["goals"]),
        }


class PriorityMonitor:
    """Deadline pressure: what falls due inside the alert window."""

    def due_soon(self, state: dict[str, Any],
                 hours: float = _DEADLINE_HOURS) -> list[str]:
        now = datetime.now(timezone.utc)
        pressing: list[tuple[float, str]] = []
        for task in (state.get("pending_tasks") or {}).values():
            if not isinstance(task, dict):
                continue
            due = _parse_stamp(str(task.get("due") or ""))
            if due is None:
                continue
            remaining = (due - now).total_seconds() / 3600.0
            title = str(task.get("title") or "task")[:70]
            if remaining < 0:
                pressing.append((remaining, f"{title} — OVERDUE"))
            elif remaining <= hours:
                pressing.append((remaining, f"{title} — due in {remaining:.0f}h"))
        pressing.sort()
        return [label for _, label in pressing[:6]]


class DynamicBriefingEngine:
    """Adaptive briefings from live system activity, by period or on demand."""

    def __init__(
        self,
        bus: OrionBus,
        cognition: Any,
        briefing: Any | None = None,      # MorningBriefingService (classic brief)
        companion: Any | None = None,     # CompanionEngine (continuity)
        executive_core: Any | None = None,
        telemetry: Any | None = None,
        research_director: Any | None = None,
        missions: Any | None = None,
        security: Any | None = None,
    ) -> None:
        self.bus = bus
        self.cognition = cognition
        self.briefing = briefing
        self.companion = companion
        self.executive_core = executive_core
        self.telemetry = telemetry
        self.research_director = research_director
        self.missions = missions
        self.security = security
        self.events = EventMonitor()
        self.context = ContextMonitor(cognition)
        self.priority = PriorityMonitor()
        self._last_briefed = 0.0
        if telemetry is not None:
            telemetry.health.register("briefing_engine")

    # Wired to bus.dashboard_event at app start-up.
    def observe_event(self, channel: str, payload: Any) -> None:
        self.events.observe(channel, payload)

    @staticmethod
    def period(moment: datetime | None = None) -> str:
        """morning / midday / evening for the adaptive brief.

        Keeps this engine's own 'midday' label (its templates key off it) but
        takes the BOUNDARIES from the shared band table, so it splits the day
        at the same instants as every other subsystem."""
        from .time_service import AFTERNOON, TIME, classify
        period = classify(moment) if moment is not None else TIME.time_of_day()
        if period == "morning":
            return "morning"
        return "midday" if period == AFTERNOON else "evening"

    # ── the adaptive brief ────────────────────────────────────────────────────

    async def brief(self, period: str = "") -> ToolResult:
        period = (str(period or "").strip().lower()
                  or self.period())
        # Specialist briefings: generated from the relevant engine's live
        # state rather than the time of day.
        if period in {"research"}:
            return await self._research_brief()
        if period in {"project", "mission", "missions"}:
            return await self._mission_brief()
        if period in {"security"}:
            return await self._security_brief()
        if period in {"opportunity", "opportunities"}:
            return await self._opportunity_brief()
        state = await asyncio.to_thread(self.cognition.snapshot)
        delta = await asyncio.to_thread(self.context.delta)
        events = self.events.drain()
        lines = [f"{period.title()} briefing — {_clock_now():%A %d %B, %H:%M}"]
        away = self._away_section(events)
        if away:
            lines.extend(away)
        if period == "morning":
            lines.extend(self._morning_section(state))
        elif period == "midday":
            lines.extend(self._midday_section(state, delta))
        else:
            lines.extend(self._evening_section(state, delta))
        pressing = self.priority.due_soon(state)
        if pressing:
            lines.append("Deadline watch:")
            lines.extend(f"  - {p}" for p in pressing)
        self._last_briefed = time.time()
        if self.telemetry is not None:
            self.telemetry.metrics.incr(f"briefing.{period}")
        self.bus.dashboard_event.emit("briefing", {"period": period,
                                                   "events": len(events)})
        return ToolResult("\n".join(lines))

    # ── event-triggered alerts ────────────────────────────────────────────────

    async def check_triggers(self) -> ToolResult:
        """Anything worth interrupting for right now? (deadlines + events)"""
        state = await asyncio.to_thread(self.cognition.snapshot)
        pressing = self.priority.due_soon(state, hours=4.0)
        pending = list(self.events.events)
        if not pressing and not pending:
            return ToolResult("Nothing needs your attention right now.")
        lines: list[str] = []
        if pressing:
            lines.append("Time-sensitive:")
            lines.extend(f"- {p}" for p in pressing)
        if pending:
            lines.append("Since you last checked:")
            lines.extend(f"- {self._event_line(e)}" for e in pending[-6:])
        return ToolResult("\n".join(lines))

    # ── specialist briefings ──────────────────────────────────────────────────

    async def _research_brief(self) -> ToolResult:
        lines = [f"Research briefing — {_clock_now():%A %d %B, %H:%M}"]
        state = await asyncio.to_thread(self.cognition.snapshot)
        sessions = {k: v for k, v in (state.get("research_sessions") or {}).items()
                    if isinstance(v, dict)}
        by_status: dict[str, list[str]] = {}
        for s in sessions.values():
            by_status.setdefault(str(s.get("status") or "queued"), []).append(
                str(s.get("topic") or "")[:60])
        for status in ("running", "queued", "complete", "reviewed"):
            topics = by_status.get(status)
            if topics:
                lines.append(f"{status.title()}: " + "; ".join(topics[:4]))
        if self.research_director is not None:
            try:
                opportunities = await self.research_director.opportunities()
                lines.append(getattr(opportunities, "text", str(opportunities)))
            except Exception:
                pass
        if len(lines) == 1:
            lines.append("No research programme activity yet — queue a topic "
                         "and I'll take it from there.")
        self._mark_briefed("research")
        return ToolResult("\n".join(lines))

    async def _mission_brief(self) -> ToolResult:
        lines = [f"Mission briefing — {_clock_now():%A %d %B, %H:%M}"]
        if self.missions is None:
            return ToolResult("Mission engine offline — no mission briefing "
                              "available.", ok=False)
        snap = await asyncio.to_thread(self.missions.panel_snapshot)
        current = snap.get("current") or ""
        for m in snap.get("missions", []):
            marker = "▶" if m["name"] == current else "•"
            line = (f"{marker} {m['name']}: {m['progress']:.0f}%, "
                    f"{m['open_tasks']} open task(s)")
            if m.get("risks"):
                line += " — RISK: " + "; ".join(m["risks"][:2])
            lines.append(line)
        focus = next((m for m in snap.get("missions", [])
                      if m["name"] == current), None)
        if focus is not None and focus.get("recommendation"):
            lines.append(f"Recommended next move: {focus['recommendation']}")
        self._mark_briefed("mission")
        return ToolResult("\n".join(lines))

    async def _security_brief(self) -> ToolResult:
        lines = [f"Security briefing — {_clock_now():%A %d %B, %H:%M}"]
        security_events = [e for e in self.events.events
                           if e.get("kind") == "security"]
        if security_events:
            lines.append(f"{len(security_events)} security event(s) since the "
                         "last review:")
            lines.extend(f"  - {e.get('detail', '')[:80]}"
                         for e in security_events[-5:])
        if self.security is not None:
            for method in ("summary", "status_report", "snapshot"):
                try:
                    report = getattr(self.security, method)()
                    if report:
                        lines.append(str(report).split("\n")[0][:160])
                        break
                except Exception:
                    continue
        if len(lines) == 1:
            lines.append("No security events recorded — perimeter quiet.")
        self._mark_briefed("security")
        return ToolResult("\n".join(lines))

    async def _opportunity_brief(self) -> ToolResult:
        lines = [f"Opportunity briefing — {_clock_now():%A %d %B, %H:%M}"]
        if self.research_director is not None:
            try:
                result = await self.research_director.opportunities()
                lines.append(getattr(result, "text", str(result)))
            except Exception:
                pass
        if self.missions is not None:
            try:
                snap = await asyncio.to_thread(self.missions.panel_snapshot)
                quiet = [m["name"] for m in snap.get("missions", [])
                         if m.get("status") == "active"
                         and m.get("open_tasks", 0) == 0]
                if quiet:
                    lines.append("Missions with no queued work (room to push): "
                                 + ", ".join(quiet[:3]) + ".")
            except Exception:
                pass
        if len(lines) == 1:
            lines.append("No stand-out opportunities right now — steady as "
                         "she goes.")
        self._mark_briefed("opportunity")
        return ToolResult("\n".join(lines))

    def _mark_briefed(self, kind: str) -> None:
        self._last_briefed = time.time()
        if self.telemetry is not None:
            self.telemetry.metrics.incr(f"briefing.{kind}")
        self.bus.dashboard_event.emit("briefing", {"period": kind, "events": 0})

    # ── section builders ──────────────────────────────────────────────────────

    def _away_section(self, events: list[dict[str, Any]]) -> list[str]:
        if not events:
            return []
        lines = ["While you were away:"]
        lines.extend(f"  - {self._event_line(e)}" for e in events[-8:])
        return lines

    @staticmethod
    def _event_line(event: dict[str, Any]) -> str:
        kind = event.get("kind")
        detail = event.get("detail") or "an item"
        status = str(event.get("status") or "")
        if kind == "research":
            return (f"research on '{detail}' completed" if status != "failed"
                    else f"research on '{detail}' failed")
        if kind == "workflow":
            return f"workflow '{detail}' {status or 'finished'}"
        if kind == "security":
            return f"security: {detail}"
        if kind == "capability":
            return f"new capability forged: {detail}"
        return f"{kind}: {detail}"

    def _morning_section(self, state: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        priorities = state.get("user_priorities") or []
        if priorities:
            lines.append("Standing priorities: "
                         + "; ".join(str(p)[:50] for p in priorities[:3]))
        tasks = state.get("pending_tasks") or {}
        projects = state.get("active_projects") or {}
        lines.append(f"On the books: {len(tasks)} open task(s) across "
                     f"{len(projects)} project(s).")
        if self.briefing is not None:
            lines.append("Say 'morning briefing' for the full news, markets "
                         "and calendar brief.")
        return lines

    def _midday_section(self, state: dict[str, Any],
                        delta: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if not delta.get("first"):
            cleared = delta.get("tasks_cleared", 0)
            added = delta.get("tasks_added", 0)
            if cleared or added:
                lines.append(f"Progress since last brief: {cleared} task(s) "
                             f"cleared, {added} added.")
            else:
                lines.append("No task movement since the last brief — "
                             "worth a deliberate push this afternoon.")
        sessions = state.get("research_sessions") or {}
        running = [str(s.get("topic")) for s in sessions.values()
                   if isinstance(s, dict) and str(s.get("status")) == "running"]
        if running:
            lines.append("Research in progress: " + ", ".join(running[:3]) + ".")
        return lines

    def _evening_section(self, state: dict[str, Any],
                         delta: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if not delta.get("first"):
            cleared = delta.get("tasks_cleared", 0)
            goals_done = delta.get("goals_completed", 0)
            accomplished: list[str] = []
            if cleared:
                accomplished.append(f"{cleared} task(s) cleared")
            if goals_done:
                accomplished.append(f"{goals_done} goal(s) completed")
            lines.append("Accomplished today: "
                         + (", ".join(accomplished) if accomplished
                            else "quiet on the scoreboard."))
        tasks = state.get("pending_tasks") or {}
        if tasks:
            first = next(iter(tasks.values()), {})
            title = str(first.get("title") or "")[:60] if isinstance(first, dict) else ""
            lines.append(f"Unfinished: {len(tasks)} task(s) carry over"
                         + (f" — starting with '{title}'." if title else "."))
        if self.companion is not None:
            try:
                continuity = self.companion.continuity_brief()
                if continuity:
                    lines.append(str(continuity).split("\n")[0][:160])
            except Exception:
                pass
        return lines


__all__ = ["ContextMonitor", "DynamicBriefingEngine", "EventMonitor",
           "PriorityMonitor"]
