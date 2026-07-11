"""
CognitiveLoopManager (Mark X.5) — the continuous cognitive loop.

A single low-priority background loop that keeps ORION persistently AWARE of
the user's world — projects, tasks, workspaces, conversations, deadlines and
priorities — without ever acting on it.  Awareness, not autonomy: the loop
monitors and remembers; it never dispatches tools, never opens applications
and never speaks.  Anything it learns is surfaced passively (dashboard event,
log line) and stored so the conversational channels can draw on it.

It composes the durable primitives that already exist rather than duplicating
them:

    CognitiveStateManager  — goals, projects, workflows, tasks, priorities
    IntentTracker          — recurring user intent (via the state manager)
    KnowledgeGraphEngine   — the second brain, fed compressed conversation
    WorkspaceManager       — live desktop focus
    MemoryAgent            — episodic turns and the active project

Every cycle runs its blocking work through ``asyncio.to_thread`` so the
qasync GUI loop is never touched, and every failure is contained — a bad
cycle logs once and the loop carries on.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .memory import MemoryAgent
from .utils import first_line


class CognitiveLoopManager:
    """Persistent awareness loop — monitors and remembers, never executes."""

    DEFAULT_INTERVAL_S = 120.0
    FIRST_RUN_DELAY_S = 60.0
    DEADLINE_HORIZON_HOURS = 72.0

    def __init__(
        self,
        bus: OrionBus,
        memory: MemoryAgent,
        cognition: Any,
        graph: Any | None = None,
        workspace: Any | None = None,
        telemetry: Any | None = None,
        interval_s: float | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.cognition = cognition          # CognitiveStateManager
        self.graph = graph                  # KnowledgeGraphEngine (optional)
        self.workspace = workspace          # WorkspaceManager (optional)
        self.telemetry = telemetry
        self.interval_s = float(interval_s or self.DEFAULT_INTERVAL_S)
        self._stop = asyncio.Event()
        self._ingested: set[str] = set()    # episode hashes already graphed
        self._cycles = 0
        self._last_digest: dict[str, Any] = {}
        if self.telemetry is not None:
            self.telemetry.health.register("cognitive_loop")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Background awareness loop; cancel-safe, failure-contained."""
        try:
            await asyncio.sleep(self.FIRST_RUN_DELAY_S)
            while not self._stop.is_set():
                try:
                    await self._cycle()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.bus.log.emit(f"COG: awareness cycle recovered - {first_line(exc)}")
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()

    # ── the awareness cycle (observe → remember → surface) ───────────────────

    async def _cycle(self) -> None:
        self._cycles += 1
        # 1. Desktop focus — where is the user's attention right now?
        focus = ""
        if self.workspace is not None:
            try:
                snap = await self.workspace.snapshot_workspace()
                focus = snap.active_window[:120]
                await asyncio.to_thread(
                    self.cognition.update_workspace,
                    focus,
                    windows=len(snap.windows),
                    browser_tabs=len(snap.browser_tabs),
                    active_project=self.memory.active_project,
                )
            except Exception as exc:
                self.bus.log.emit(f"COG: workspace observation skipped - {first_line(exc)}")

        # 2. Conversation awareness — fold fresh user turns into intent
        #    tracking and the second brain (deduplicated, idempotent).
        turns = self.memory.recent_turns(limit=12)
        fresh = 0
        for turn in turns:
            content = str(turn.get("content") or "").strip()
            role = str(turn.get("role") or "")
            if not content or not role.startswith("user"):
                continue
            digest = hashlib.sha1(content[:240].encode("utf-8")).hexdigest()
            if digest in self._ingested:
                continue
            self._ingested.add(digest)
            fresh += 1
            try:
                await asyncio.to_thread(
                    self.cognition.intent.observe_request, content,
                    self.memory.active_project,
                )
            except Exception:
                pass
            if self.graph is not None:
                try:
                    await asyncio.to_thread(
                        self.graph.ingest_record, "conversation",
                        content[:80], content,
                        {"project": self.memory.active_project},
                    )
                except Exception:
                    pass  # sanitiser or storage refusal must not stall the loop
        if len(self._ingested) > 4000:
            # Bounded memory: keep the newest half of the dedup window.
            self._ingested = set(list(self._ingested)[-2000:])

        # 3. Deadline awareness — pending tasks approaching their due time.
        deadlines = await asyncio.to_thread(self._due_soon)

        # 4. Surface the digest passively (never spoken, never acted on).
        digest = {
            "at": datetime.now().strftime("%H:%M:%S"),
            "cycle": self._cycles,
            "focus": focus,
            "active_project": self.memory.active_project,
            "fresh_turns": fresh,
            "deadlines": deadlines[:6],
            "graph": self.graph.stats() if self.graph is not None else {},
        }
        self._last_digest = digest
        self.bus.dashboard_event.emit("awareness", digest)
        if self.telemetry is not None:
            self.telemetry.health.beat(
                "cognitive_loop", "OK",
                f"cycle {self._cycles}; {len(deadlines)} deadline(s) in view",
            )
            self.telemetry.metrics.incr("cognition.cycles")

    def _due_soon(self) -> list[str]:
        """Pending tasks whose due date falls inside the awareness horizon."""
        out: list[str] = []
        try:
            state = self.cognition.snapshot()
        except Exception:
            return out
        now = datetime.now()
        for task in (state.get("pending_tasks") or {}).values():
            if not isinstance(task, dict):
                continue
            due_raw = str(task.get("due") or "").strip()
            title = str(task.get("title") or "task")[:80]
            if not due_raw:
                continue
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                try:
                    due = datetime.strptime(due_raw[: len(now.strftime(fmt))], fmt)
                except ValueError:
                    continue
                hours = (due - now).total_seconds() / 3600.0
                if hours < 0:
                    out.append(f"OVERDUE: '{title}' ({abs(hours):.0f}h ago)")
                elif hours <= self.DEADLINE_HORIZON_HOURS:
                    out.append(f"'{title}' due in {hours:.0f}h")
                break
        return sorted(out)

    # ── on-demand awareness (the voice/tool channel asks; loop never speaks) ──

    def situation_report(self) -> ToolResult:
        """What ORION is currently aware of — for the 'awareness' tool."""
        state = self.cognition.snapshot()
        lines = ["Cognitive awareness report:"]
        project = self.memory.active_project or "none"
        lines.append(f"- Active project: {project}")
        focus = (state.get("current_workspace") or {}).get("name") or "unknown"
        lines.append(f"- Workspace focus: {focus}")
        goals = state.get("active_goals") or []
        if goals:
            lines.append(f"- Active goals: {len(goals)}")
        priorities = state.get("user_priorities") or []
        if priorities:
            lines.append("- Priorities: " + "; ".join(str(p)[:60] for p in priorities[:5]))
        deadlines = self._due_soon()
        if deadlines:
            lines.append("- Deadlines: " + "; ".join(deadlines[:5]))
        try:
            top = self.cognition.intent.top_intents(limit=4)
            if top:
                lines.append("- Recurring intents: " + "; ".join(
                    str(i.get("label") or i.get("key") or "?")[:50] for i in top
                ))
        except Exception:
            pass
        if self.graph is not None:
            stats = self.graph.stats()
            lines.append(
                f"- Second brain: {stats.get('entities', 0)} entities, "
                f"{stats.get('events', 0)} events, "
                f"{stats.get('relationships', 0)} relationships"
            )
        lines.append(f"- Awareness cycles this session: {self._cycles}")
        return ToolResult("\n".join(lines))

    def last_digest(self) -> dict[str, Any]:
        return dict(self._last_digest)
