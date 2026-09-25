"""
ExecutiveCore (Phase 3) — ORION's central intelligence layer.

The coordinating system ABOVE the individual agents and services. Where
``ExecutiveAssistantMode`` (Mark X.5) handles the assistant mechanics —
scheduling, meeting minutes, task tracking — the ExecutiveCore reasons about
the whole operation:

    DecisionEngine   — constructive challenge: assumptions, risks,
                       alternatives and blind spots for any proposed decision.
                       ORION questions weak ideas rather than agreeing.
    PriorityEngine   — one urgency-ranked queue across tasks, goals and
                       workflows, each entry with a stated reason.
    StrategicEngine  — recommendations from live system state: stalled
                       projects, drifting goals, overloaded queues,
                       unreviewed research.
    GoalEngine       — goal portfolio evaluation on top of GoalManager.

Everything composes existing services (CognitiveStateManager, GoalManager,
ExecutiveAssistantMode, ProviderRouter, KnowledgeGraphEngine); this layer
owns no storage and no transport. All engines degrade gracefully offline:
the model enriches analysis when reachable, and a deterministic
critical-thinking core produces the challenge, ranking and recommendations
on its own when it is not. Blocking state reads run off the event loop.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .utils import first_line

# ── deterministic critical-thinking core ─────────────────────────────────────
# Domain-keyed risk libraries: when a decision mentions a domain, its risks
# are raised as questions. The generic set always applies.

_DOMAIN_RISKS: dict[str, tuple[str, ...]] = {
    "launch": (
        "market saturation in the niche you are entering",
        "creator or audience fatigue with the format",
        "CPM and acquisition-cost trends moving against you",
        "competitor pricing and counter-offers",
        "fulfilment and support load once volume arrives",
    ),
    "product": (
        "evidence of real demand beyond your own conviction",
        "unit economics after returns, refunds and ad spend",
        "differentiation a competitor cannot copy in a week",
        "supplier reliability and lead times",
    ),
    "content": (
        "platform algorithm changes cutting reach overnight",
        "hook fatigue — the format that worked last month decaying",
        "over-reliance on a single channel or creator",
        "production cost per video versus attributable revenue",
    ),
    "creator": (
        "creator churn and key-person dependency",
        "brief clarity — whether creators actually know what good looks like",
        "usage-rights and licensing terms on the content",
        "incentive alignment between flat fees and performance",
    ),
    "hire": (
        "whether the role is a proven need or a hopeful bet",
        "ramp-up time before the hire pays for itself",
        "what happens if the hire does not work out",
    ),
    "price": (
        "price-elasticity of your actual audience, not the market average",
        "competitor response to the price move",
        "margin after platform fees, shipping and returns",
    ),
    "invest": (
        "opportunity cost against your current best-performing channel",
        "time to payback and what you do if it doubles",
        "whether the downside is capped or open-ended",
    ),
    "automat": (
        "failure modes when the automation runs unattended",
        "the cost of silent errors versus visible manual work",
        "who maintains it when the workflow changes",
    ),
}

_GENERIC_CHALLENGES: tuple[str, ...] = (
    "What evidence supports this beyond intuition — and what would change your mind?",
    "What is the strongest argument AGAINST doing this?",
    "Is this reversible? If it fails, what exactly is lost?",
    "What is the second-order effect once it works — what does success break?",
    "Who has tried this before, and why did they stop?",
    "What is the cheapest test that would validate this in a week?",
)

_STALL_DAYS_PROJECT = 7.0
_STALL_DAYS_GOAL = 5.0


def _parse_stamp(raw: str) -> datetime | None:
    """Parse the utc_stamp()/date formats used across cognitive state."""
    raw = str(raw or "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(raw[: len(datetime.now().strftime(fmt))], fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _age_days(raw: str) -> float | None:
    stamp = _parse_stamp(raw)
    if stamp is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds() / 86400.0)


class DecisionEngine:
    """Constructive challenge for proposed decisions — never passive agreement."""

    def __init__(self, router: Any | None = None) -> None:
        self.router = router

    def challenge_offline(self, decision: str) -> list[str]:
        """Deterministic challenge points drawn from the risk library."""
        text = decision.lower()
        points: list[str] = []
        for stem, risks in _DOMAIN_RISKS.items():
            if stem in text:
                points.extend(f"Have you considered {risk}?" for risk in risks)
        generic = list(_GENERIC_CHALLENGES)
        # Deterministic but decision-dependent selection of generic prompts.
        offset = sum(ord(c) for c in text) % len(generic)
        for i in range(3):
            points.append(generic[(offset + i) % len(generic)])
        seen: set[str] = set()
        unique = [p for p in points if not (p in seen or seen.add(p))]
        return unique[:9]

    async def challenge(self, decision: str, context: str = "") -> str:
        points = self.challenge_offline(decision)
        body = "\n".join(f"- {p}" for p in points)
        enriched = ""
        if self.router is not None and getattr(self.router, "has_text_fallback", lambda: False)():
            try:
                _profile, enriched = await self.router.generate_text(
                    f"Proposed decision: {decision}\n"
                    + (f"Context: {context}\n" if context else "")
                    + "Play the constructive sceptic. In under 180 words: the two "
                    "weakest assumptions, the biggest risk, one alternative worth "
                    "considering, and the cheapest validating test. British English.",
                    system_extra="You are a strategic advisor who challenges ideas "
                                 "to strengthen them, never to dismiss them.",
                )
            except Exception:
                enriched = ""
        result = f"Before committing to '{decision.strip()}', consider:\n{body}"
        if enriched.strip():
            result += f"\n\nStrategic view:\n{enriched.strip()}"
        return result


class PriorityEngine:
    """One urgency-ranked queue across tasks, goals and workflows."""

    _TERMS = (
        ("urgent", 5.0), ("asap", 5.0), ("today", 4.0), ("deadline", 4.0),
        ("blocker", 4.0), ("critical", 5.0), ("client", 3.0), ("payment", 3.0),
        ("launch", 3.0), ("important", 2.5),
    )

    def rank(self, state: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
        now = datetime.now()
        queue: list[dict[str, Any]] = []
        for task in (state.get("pending_tasks") or {}).values():
            if not isinstance(task, dict):
                continue
            title = str(task.get("title") or "task")[:90]
            score, reason = 1.0, "open task"
            lowered = title.lower()
            for term, weight in self._TERMS:
                if term in lowered:
                    score += weight
                    reason = f"mentions '{term}'"
            due_raw = str(task.get("due") or "").strip()
            if due_raw:
                due = _parse_stamp(due_raw)
                if due is not None:
                    hours = (due.replace(tzinfo=None) - now).total_seconds() / 3600.0
                    if hours < 0:
                        score, reason = score + 8.0, "OVERDUE"
                    elif hours <= 24:
                        score, reason = score + 6.0, f"due in {hours:.0f}h"
                    elif hours <= 72:
                        score, reason = score + 3.0, f"due in {hours / 24:.0f}d"
            queue.append({"kind": "task", "label": title, "score": score, "reason": reason})
        for goal in (state.get("goals") or {}).values():
            if not isinstance(goal, dict) or str(goal.get("status", "active")) != "active":
                continue
            priority = float(goal.get("priority") or 0.5)
            progress = float(goal.get("progress") or 0.0)
            score = 1.0 + priority * 4.0 + (1.0 - progress) * 2.0
            age = _age_days(str(goal.get("updated_at") or ""))
            reason = f"goal at {progress * 100:.0f}%"
            if age is not None and age > _STALL_DAYS_GOAL:
                score += 2.0
                reason += f", untouched {age:.0f}d"
            queue.append({"kind": "goal", "label": str(goal.get("title") or "goal")[:90],
                          "score": score, "reason": reason})
        for name, wf in (state.get("open_workflows") or {}).items():
            if not isinstance(wf, dict):
                continue
            queue.append({"kind": "workflow", "label": str(name)[:90],
                          "score": 2.0, "reason": "workflow in flight"})
        queue.sort(key=lambda item: -item["score"])
        return queue[: max(1, limit)]


class StrategicEngine:
    """Recommendations and blind spots from live system state."""

    def recommendations(self, state: dict[str, Any]) -> list[str]:
        recs: list[str] = []
        for name, project in (state.get("active_projects") or {}).items():
            if not isinstance(project, dict):
                continue
            age = _age_days(str(project.get("updated_at") or ""))
            if age is not None and age > _STALL_DAYS_PROJECT:
                recs.append(f"Project '{name}' has had no update in {age:.0f} days — "
                            "revive it or archive it deliberately.")
        overdue = 0
        for task in (state.get("pending_tasks") or {}).values():
            if isinstance(task, dict):
                due = _parse_stamp(str(task.get("due") or ""))
                if due is not None and due < datetime.now(timezone.utc):
                    overdue += 1
        if overdue:
            recs.append(f"{overdue} task(s) are overdue — clear or reschedule them "
                        "before adding new work.")
        stale_goals = [
            str(g.get("title") or "goal")
            for g in (state.get("goals") or {}).values()
            if isinstance(g, dict) and str(g.get("status", "active")) == "active"
            and (_age_days(str(g.get("updated_at") or "")) or 0.0) > _STALL_DAYS_GOAL
            and float(g.get("progress") or 0.0) < 1.0
        ]
        if stale_goals:
            recs.append("Goals drifting without progress: "
                        + ", ".join(f"'{t[:40]}'" for t in stale_goals[:4])
                        + " — schedule one concrete step for each.")
        workflows = state.get("open_workflows") or {}
        if len(workflows) > 5:
            recs.append(f"{len(workflows)} workflows are open at once — "
                        "finish or close some before starting more.")
        sessions = state.get("research_sessions") or {}
        unreviewed = [
            str(s.get("topic") or name)
            for name, s in sessions.items()
            if isinstance(s, dict) and str(s.get("status") or "") == "complete"
            and not s.get("reviewed")
        ]
        if unreviewed:
            recs.append("Completed research awaiting review: "
                        + ", ".join(t[:40] for t in unreviewed[:3]) + ".")
        if not recs:
            recs.append("The operation is in order — no stalled projects, "
                        "overdue tasks or drifting goals detected.")
        return recs

    def blind_spots(self, state: dict[str, Any]) -> list[str]:
        spots: list[str] = []
        if not (state.get("goals") or {}):
            spots.append("No goals are recorded — activity without goals is "
                         "motion, not progress.")
        if not (state.get("user_priorities") or []):
            spots.append("No standing priorities are set — everything urgent "
                         "will crowd out everything important.")
        projects = state.get("active_projects") or {}
        tasks = state.get("pending_tasks") or {}
        if projects and not tasks:
            spots.append("Projects are tracked but the task queue is empty — "
                         "each project should have at least one next action.")
        if len(tasks) > 15:
            spots.append(f"{len(tasks)} open tasks is beyond working memory — "
                         "batch, delegate or drop the bottom third.")
        return spots


class GoalEngine:
    """Portfolio evaluation on top of GoalManager."""

    def __init__(self, goals: Any | None = None) -> None:
        self.goals = goals  # GoalManager

    def evaluate(self, state: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        records = [g for g in (state.get("goals") or {}).values() if isinstance(g, dict)]
        if not records:
            return ["No goals on record. Set one and I will hold you to it."]
        active = [g for g in records if str(g.get("status", "active")) == "active"]
        done = [g for g in records if str(g.get("status")) == "completed"]
        lines.append(f"Goal portfolio: {len(active)} active, {len(done)} completed.")
        for goal in sorted(active, key=lambda g: -float(g.get("priority") or 0.5))[:6]:
            title = str(goal.get("title") or "goal")[:60]
            progress = float(goal.get("progress") or 0.0) * 100.0
            age = _age_days(str(goal.get("updated_at") or ""))
            note = f" — untouched {age:.0f}d" if age is not None and age > _STALL_DAYS_GOAL else ""
            lines.append(f"- {title}: {progress:.0f}%{note}")
        return lines


class ExecutiveCore:
    """The coordinating intelligence layer behind the Phase 3 'executive' actions."""

    def __init__(
        self,
        bus: OrionBus,
        cognition: Any,
        goals: Any | None = None,
        executive: Any | None = None,
        router: Any | None = None,
        graph: Any | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.bus = bus
        self.cognition = cognition
        self.executive = executive
        self.decisions = DecisionEngine(router)
        self.priorities = PriorityEngine()
        self.strategy = StrategicEngine()
        self.goal_engine = GoalEngine(goals)
        self.graph = graph
        self.telemetry = telemetry
        if telemetry is not None:
            telemetry.health.register("executive_core")

    async def _state(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.cognition.snapshot)

    # ── constructive challenge ────────────────────────────────────────────────

    async def challenge(self, decision: str, context: str = "") -> ToolResult:
        decision = str(decision or "").strip()
        if not decision:
            return ToolResult("Tell me the decision or idea to stress-test.", ok=False)
        try:
            analysis = await self.decisions.challenge(decision, context)
        except Exception as exc:
            return ToolResult(f"Decision analysis failed: {first_line(exc)}", ok=False)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("executive.challenges")
        return ToolResult(analysis)

    # ── executive focus: priorities + recommendations in one view ─────────────

    async def focus(self) -> ToolResult:
        state = await self._state()
        queue = self.priorities.rank(state, limit=7)
        recs = self.strategy.recommendations(state)
        lines = [f"Executive focus — {datetime.now():%A %d %B, %H:%M}"]
        if queue:
            lines.append("Priority queue:")
            lines.extend(
                f"  {i}. [{item['kind']}] {item['label']} ({item['reason']})"
                for i, item in enumerate(queue, 1)
            )
        lines.append("Recommendations:")
        lines.extend(f"  - {r}" for r in recs[:5])
        spots = self.strategy.blind_spots(state)
        if spots:
            lines.append("Blind spots:")
            lines.extend(f"  - {s}" for s in spots[:4])
        return ToolResult("\n".join(lines))

    async def recommend(self) -> ToolResult:
        state = await self._state()
        recs = self.strategy.recommendations(state)
        return ToolResult("Recommendations:\n" + "\n".join(f"- {r}" for r in recs))

    async def review_goals(self) -> ToolResult:
        state = await self._state()
        return ToolResult("\n".join(self.goal_engine.evaluate(state)))

    async def blind_spots(self) -> ToolResult:
        state = await self._state()
        spots = self.strategy.blind_spots(state)
        if not spots:
            return ToolResult("No structural blind spots detected in the current state.")
        return ToolResult("Blind spots:\n" + "\n".join(f"- {s}" for s in spots))


__all__ = [
    "DecisionEngine", "ExecutiveCore", "GoalEngine",
    "PriorityEngine", "StrategicEngine",
]
