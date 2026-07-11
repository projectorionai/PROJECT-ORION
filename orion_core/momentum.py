"""
MomentumEngine — ORION's shipping coach for an entrepreneur.

The cognitive stack already *tracks* goals, projects, tasks and deadlines, and
the cognitive loop *monitors* them — but neither one tells you decisively what to
do next. Momentum closes that gap: it turns the current state into shipping
pressure — the single highest-leverage next action, what's blocking or overdue,
and a concrete focus block — so projects actually get finished.

Three modes (all offline-capable; ``plan`` uses the model when reachable):
    • ``focus``   — one next action for the active project + blockers + a timed
                    focus block. The "what do I do right now" answer.
    • ``standup`` — a cross-project shipping stand-up: in-progress, next, blocked.
    • ``plan``    — break a project/goal into milestones with a definition-of-done
                    and the immediate next actions (model-drafted, template
                    fallback offline).

It only reads cognition state and writes nothing destructive, so it never fights
the awareness loop; it complements executive mode with a bias toward *finishing*.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from .bus import OrionBus
from .data import ToolResult
from .utils import first_line


class MomentumEngine:
    def __init__(self, bus: OrionBus, memory: Any, cognition: Any,
                 router: Any | None = None, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.cognition = cognition
        self.router = router
        self.telemetry = telemetry

    # ── helpers ────────────────────────────────────────────────────────────────

    def _snapshot(self) -> dict[str, Any]:
        try:
            return self.cognition.snapshot() if self.cognition is not None else {}
        except Exception:
            return {}

    @staticmethod
    def _active_project(state: dict[str, Any]) -> str:
        ws = state.get("current_workspace") or {}
        return str(ws.get("active_project") or "")

    @staticmethod
    def _pending(state: dict[str, Any]) -> list[dict[str, Any]]:
        tasks = state.get("pending_tasks") or {}
        return [t for t in tasks.values() if str(t.get("status", "pending")) != "completed"]

    @staticmethod
    def _due_bucket(due: str) -> tuple[int, str]:
        """Sort key: overdue first, then soonest dated, then undated last."""
        due = str(due or "").strip()
        if len(due) >= 10 and due[4] == "-" and due[7] == "-":
            today = date.today().isoformat()
            return (0 if due < today else 1, due)
        return (2, "9999-99-99")

    def _rank_tasks(self, tasks: list[dict[str, Any]], project: str = "") -> list[dict[str, Any]]:
        scoped = [t for t in tasks if not project or str(t.get("project", "")) == project]
        pool = scoped or tasks
        return sorted(pool, key=lambda t: self._due_bucket(t.get("due", "")))

    @staticmethod
    def _is_overdue(due: str) -> bool:
        due = str(due or "").strip()
        return len(due) >= 10 and due[4] == "-" and due < date.today().isoformat()

    # ── modes ──────────────────────────────────────────────────────────────────

    def focus(self) -> ToolResult:
        """The single next action + blockers + a focus block."""
        state = self._snapshot()
        project = self._active_project(state)
        tasks = self._pending(state)
        if not tasks:
            hint = (f" on '{project}'" if project else "")
            return ToolResult(
                f"No open tasks are tracked{hint}, sir. Tell me the goal and I'll break it "
                "into next actions, or add a task and I'll keep you shipping.")
        ranked = self._rank_tasks(tasks, project)
        nxt = ranked[0]
        overdue = [t for t in ranked if self._is_overdue(t.get("due", ""))]
        lines = [f"Next action{(' on ' + project) if project else ''}, sir: "
                 f"**{nxt.get('title', 'untitled')}**"
                 + (f" (due {nxt['due']})" if nxt.get("due") else "") + "."]
        if overdue:
            od = ", ".join(t.get("title", "?") for t in overdue[:3])
            lines.append(f"Overdue / blocking: {od}.")
        remaining = len(ranked) - 1
        if remaining > 0:
            lines.append(f"{remaining} other task(s) queued after it.")
        lines.append("Suggested focus block: 50 minutes on that one action, phone face-down, "
                     "then a 10-minute break. Say 'done' when it's shipped and I'll tee up the next.")
        return ToolResult("\n".join(lines))

    def standup(self) -> ToolResult:
        """A cross-project shipping stand-up."""
        state = self._snapshot()
        tasks = self._pending(state)
        projects = state.get("active_projects") or {}
        priorities = state.get("user_priorities") or []
        if not tasks and not projects and not priorities:
            return ToolResult(
                "Nothing is tracked yet, sir. Name a project and its goal and I'll set up "
                "milestones and next actions so we can start shipping.")
        ranked = self._rank_tasks(tasks)
        overdue = [t for t in ranked if self._is_overdue(t.get("due", ""))]
        lines = ["Shipping stand-up, sir:"]
        if projects:
            lines.append("Active projects: " + ", ".join(list(projects)[:6]) + ".")
        if priorities:
            lines.append("Priorities: " + "; ".join(str(p) for p in priorities[:4]) + ".")
        if ranked:
            lines.append("Do next: " + "; ".join(
                f"{t.get('title', '?')}" + (f" (due {t['due']})" if t.get("due") else "")
                for t in ranked[:3]) + ".")
        if overdue:
            lines.append(f"⚠ {len(overdue)} overdue — clear these first: "
                         + ", ".join(t.get("title", "?") for t in overdue[:3]) + ".")
        else:
            lines.append("Nothing overdue — good. Keep the streak.")
        return ToolResult("\n".join(lines))

    async def plan(self, project: str = "", goal: str = "") -> ToolResult:
        """Break a project/goal into milestones + a definition-of-done."""
        state = self._snapshot()
        project = str(project or self._active_project(state) or "").strip()
        goal = str(goal or "").strip()
        if not project and not goal:
            return ToolResult("Which project or goal shall I plan, sir?", ok=False)
        subject = goal or project
        tasks = [t.get("title", "") for t in self._rank_tasks(self._pending(state), project)][:8]
        task_note = ("\nAlready tracked: " + "; ".join(t for t in tasks if t)) if tasks else ""

        if self.router is not None and self.router.has_text_fallback():
            persona = (
                "SPECIALIST MODE — Delivery lead for a solo entrepreneur. Turn a project/goal into a "
                "concrete plan that gets it SHIPPED: 3-5 milestones in order, each with a one-line "
                "definition-of-done; then the immediate next 3 actions (concrete, today-sized); then "
                "the single biggest risk and how to de-risk it. Be specific and lean; no filler."
            )
            prompt = f"Project: {project or '(unnamed)'}\nGoal: {goal or project}{task_note}\n\nProduce the shipping plan."
            try:
                _profile, answer = await self.router.generate_text(prompt, system_extra=persona)
                if answer.strip():
                    self.bus.dashboard_event.emit("momentum", {"project": project, "goal": subject})
                    return ToolResult(f"Shipping plan for '{subject}', sir:\n\n{answer.strip()}")
            except Exception as exc:
                self.bus.log.emit(f"MOMENTUM: model plan failed - {first_line(exc, 80)}")

        # Offline template — still actionable.
        template = (
            f"Shipping plan for '{subject}', sir (offline template — I'll sharpen it when a model is reachable):\n"
            "1. Define done — write the one sentence that means this project is finished.\n"
            "2. Slice — break it into 3-5 milestones, each shippable on its own.\n"
            "3. Next 3 actions — the smallest concrete steps you can do today.\n"
            "4. De-risk — name the one thing most likely to stall it, and tackle that first.\n"
            "5. Cadence — one 50-minute focus block per day beats a heroic weekend."
        )
        return ToolResult(template)
