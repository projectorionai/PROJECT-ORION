"""
Mark XI cognitive continuity layer.

This module keeps ORION's working state outside any single model session:
goals, projects, workflows, pending tasks, research sessions, workspace focus
and recurring user intent.  It is deliberately local-first: state is stored as
an atomic JSON document under ``config/`` and important signals are mirrored
into ``MemoryAgent`` so they survive model switches, crashes and reboots.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Literal, Optional

from .bus import OrionBus
from .constants import CONFIG_DIR
from .memory import MemoryAgent, MemoryTier
from .security import SecuritySanitiser
from .utils import first_line, utc_stamp

GoalHorizon = Literal["long_term", "medium_term", "session"]
GoalStatus = Literal["active", "completed", "archived"]


def _slug(value: str, fallback: str = "item", limit: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return (slug or fallback)[:limit]


def _tokens(value: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about",
        "what", "when", "where", "your", "you", "sir", "orion", "please",
    }
    return {
        t for t in re.findall(r"[a-z0-9]{3,}", str(value or "").lower())
        if t not in stop
    }


@dataclass
class GoalRecord:
    id: str
    title: str
    description: str = ""
    horizon: GoalHorizon = "session"
    priority: float = 0.5
    progress: float = 0.0
    relevance: float = 0.0
    status: GoalStatus = "active"
    project: str = ""
    tags: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_stamp)
    updated_at: str = field(default_factory=utc_stamp)
    completed_at: str = ""
    archived_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoalRecord":
        return cls(
            id=str(data.get("id") or _slug(data.get("title", "goal"))),
            title=str(data.get("title") or "Untitled goal"),
            description=str(data.get("description") or ""),
            horizon=str(data.get("horizon") or "session"),  # type: ignore[arg-type]
            priority=max(0.0, min(1.0, float(data.get("priority", 0.5)))),
            progress=max(0.0, min(1.0, float(data.get("progress", 0.0)))),
            relevance=max(0.0, min(1.0, float(data.get("relevance", 0.0)))),
            status=str(data.get("status") or "active"),  # type: ignore[arg-type]
            project=str(data.get("project") or ""),
            tags=[str(t)[:48] for t in data.get("tags", []) if str(t).strip()],
            evidence=[str(e)[:280] for e in data.get("evidence", []) if str(e).strip()],
            created_at=str(data.get("created_at") or utc_stamp()),
            updated_at=str(data.get("updated_at") or utc_stamp()),
            completed_at=str(data.get("completed_at") or ""),
            archived_at=str(data.get("archived_at") or ""),
        )

    def score(self, context: str = "", active_project: str = "") -> None:
        context_tokens = _tokens(context)
        goal_tokens = _tokens(" ".join([self.title, self.description, self.project, " ".join(self.tags)]))
        overlap = len(context_tokens & goal_tokens)
        base = min(1.0, overlap / max(4, len(goal_tokens) or 1))
        if active_project and self.project and _slug(active_project) == _slug(self.project):
            base = max(base, 0.72)
        if self.status != "active":
            base *= 0.15
        self.relevance = round(max(0.0, min(1.0, base)), 3)
        self.priority = round(max(0.0, min(1.0, self.priority)), 3)
        self.progress = round(max(0.0, min(1.0, self.progress)), 3)
        self.updated_at = utc_stamp()


class CognitiveStateManager:
    """Durable state co-ordinator for goals, projects and resumed work."""

    SCHEMA = "orion.mark_xi.cognitive_state.v1"

    def __init__(
        self,
        bus: OrionBus,
        memory: MemoryAgent,
        path: Path | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.telemetry = telemetry
        self.path = path or (CONFIG_DIR / "cognitive_state.json")
        self._lock = RLock()
        self._state: dict[str, Any] = self._default_state()
        self.goals = GoalManager(self)
        self.intent = IntentTracker(self, memory)
        if self.telemetry is not None:
            self.telemetry.health.register("cognition")

    def _default_state(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "created_at": utc_stamp(),
            "updated_at": utc_stamp(),
            "last_launch_at": "",
            "active_goals": [],
            "goals": {},
            "active_projects": {},
            "open_workflows": {},
            "user_priorities": [],
            "current_workspace": {},
            "pending_tasks": {},
            "research_sessions": {},
            "intent_tracker": {"requests": {}, "interests": {}, "habits": {}},
        }

    async def restore_on_launch(self) -> str:
        """Load persisted state off the event loop and prime memory context."""
        summary = await asyncio.to_thread(self._restore_sync)
        if summary:
            self.memory.note_session_fact("cognitive_resume", summary)
            await asyncio.to_thread(
                self.memory.remember,
                MemoryTier.LONG_TERM,
                "last_cognitive_resume",
                summary,
            )
        self.bus.dashboard_event.emit("cognition", self.snapshot())
        return summary

    def _restore_sync(self) -> str:
        with self._lock:
            self._load_locked()
            self._state["last_launch_at"] = utc_stamp()
            self._save_locked()
            active_project = self._state.get("current_workspace", {}).get("active_project", "")
            if active_project:
                self.memory.set_active_project(str(active_project))
            summary = self.launch_summary_locked()
        self._log("COG: cognitive continuity restored.")
        if self.telemetry is not None:
            self.telemetry.health.beat("cognition", "OK", "state restored")
        return summary

    def _load_locked(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._save_locked()
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("state root is not an object")
            merged = self._default_state()
            merged.update(raw)
            for key in ("goals", "active_projects", "open_workflows", "pending_tasks", "research_sessions"):
                if not isinstance(merged.get(key), dict):
                    merged[key] = {}
            if not isinstance(merged.get("active_goals"), list):
                merged["active_goals"] = []
            if not isinstance(merged.get("user_priorities"), list):
                merged["user_priorities"] = []
            self._state = merged
        except Exception as exc:
            backup = self.path.with_suffix(f".corrupt-{int(time.time())}.json")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            self._state = self._default_state()
            self._log(f"COG: corrupt state ignored - {first_line(exc)}")
            self._save_locked()

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state["schema"] = self.SCHEMA
        self._state["updated_at"] = utc_stamp()
        tmp = self.path.with_suffix(".tmp")
        payload = json.dumps(self._state, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("cognition.state_saves")

    def _mutate(self, mutator: Any) -> Any:
        with self._lock:
            result = mutator(self._state)
            self._save_locked()
            snapshot = self.snapshot_locked()
        self.bus.dashboard_event.emit("cognition", snapshot)
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.snapshot_locked()

    def snapshot_locked(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._state, ensure_ascii=False))

    def launch_summary_locked(self) -> str:
        goals = [
            GoalRecord.from_dict(g)
            for g in self._state.get("goals", {}).values()
            if str(g.get("status", "active")) == "active"
        ]
        tasks = self._state.get("pending_tasks", {})
        workflows = self._state.get("open_workflows", {})
        research = self._state.get("research_sessions", {})
        workspace = self._state.get("current_workspace", {})
        lines = ["Cognitive continuity restored."]
        if goals:
            ranked = sorted(goals, key=lambda g: (g.priority, g.relevance, g.updated_at), reverse=True)[:5]
            lines.append("Active goals: " + "; ".join(f"{g.title} ({int(g.progress * 100)}%)" for g in ranked))
        if tasks:
            lines.append(f"Pending tasks: {len(tasks)}.")
        if workflows:
            lines.append("Open workflows: " + ", ".join(list(workflows)[:6]) + ".")
        if research:
            lines.append("Research sessions: " + ", ".join(list(research)[:6]) + ".")
        if workspace:
            label = workspace.get("name") or workspace.get("active_project") or "latest workspace"
            lines.append(f"Workspace focus: {label}.")
        return " ".join(lines)

    def launch_summary(self) -> str:
        with self._lock:
            return self.launch_summary_locked()

    def update_workspace(self, name: str = "", **metadata: Any) -> dict[str, Any]:
        clean_name = SecuritySanitiser.guard_text(name, "cognition.workspace")[:120]

        def _do(state: dict[str, Any]) -> dict[str, Any]:
            workspace = dict(state.get("current_workspace") or {})
            workspace.update({k: v for k, v in metadata.items() if v not in (None, "")})
            if clean_name:
                workspace["name"] = clean_name
            workspace["updated_at"] = utc_stamp()
            state["current_workspace"] = workspace
            return workspace

        return self._mutate(_do)

    def upsert_project(self, name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        clean = SecuritySanitiser.guard_text(name, "cognition.project")[:160]
        key = _slug(clean, "project")

        def _do(state: dict[str, Any]) -> dict[str, Any]:
            projects = state.setdefault("active_projects", {})
            record = dict(projects.get(key) or {})
            record.update(details or {})
            record.update({"id": key, "name": clean, "updated_at": utc_stamp()})
            projects[key] = record
            state.setdefault("current_workspace", {})["active_project"] = key
            return record

        record = self._mutate(_do)
        self.memory.set_active_project(key)
        return record

    def upsert_workflow(self, name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        clean = SecuritySanitiser.guard_text(name, "cognition.workflow")[:180]
        key = _slug(clean, "workflow")

        def _do(state: dict[str, Any]) -> dict[str, Any]:
            workflows = state.setdefault("open_workflows", {})
            record = dict(workflows.get(key) or {})
            record.update(details or {})
            record.update({"id": key, "name": clean, "status": "open", "updated_at": utc_stamp()})
            workflows[key] = record
            return record

        return self._mutate(_do)

    def close_workflow(self, name: str) -> bool:
        key = _slug(name, "workflow")
        return bool(self._mutate(lambda state: state.setdefault("open_workflows", {}).pop(key, None)))

    def add_priority(self, text: str) -> list[str]:
        clean = SecuritySanitiser.guard_text(text, "cognition.priority")[:240]
        if not clean:
            return list(self._state.get("user_priorities", []))

        def _do(state: dict[str, Any]) -> list[str]:
            priorities = [p for p in state.setdefault("user_priorities", []) if p != clean]
            priorities.insert(0, clean)
            state["user_priorities"] = priorities[:24]
            return state["user_priorities"]

        return self._mutate(_do)

    def add_task(self, title: str, project: str = "", due: str = "") -> dict[str, Any]:
        clean = SecuritySanitiser.guard_text(title, "cognition.task")[:220]
        key = _slug(clean, "task")

        def _do(state: dict[str, Any]) -> dict[str, Any]:
            tasks = state.setdefault("pending_tasks", {})
            record = dict(tasks.get(key) or {})
            record.update({
                "id": key,
                "title": clean,
                "project": _slug(project, "", 80),
                "due": str(due or ""),
                "status": "pending",
                "updated_at": utc_stamp(),
            })
            tasks[key] = record
            return record

        return self._mutate(_do)

    def complete_task(self, key_or_title: str) -> bool:
        key = _slug(key_or_title, "task")

        def _do(state: dict[str, Any]) -> bool:
            tasks = state.setdefault("pending_tasks", {})
            match = key if key in tasks else next(
                (k for k, v in tasks.items() if key in k or key in _slug(v.get("title", ""))),
                "",
            )
            if not match:
                return False
            tasks[match]["status"] = "completed"
            tasks[match]["completed_at"] = utc_stamp()
            tasks.pop(match, None)
            return True

        return bool(self._mutate(_do))

    def upsert_research_session(self, topic: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        clean = SecuritySanitiser.guard_text(topic, "cognition.research")[:180]
        key = _slug(clean, "research")

        def _do(state: dict[str, Any]) -> dict[str, Any]:
            sessions = state.setdefault("research_sessions", {})
            record = dict(sessions.get(key) or {})
            record.update(details or {})
            record.update({"id": key, "topic": clean, "updated_at": utc_stamp()})
            sessions[key] = record
            return record

        return self._mutate(_do)

    def _log(self, message: str) -> None:
        if self.telemetry is not None:
            try:
                self.telemetry.log.info("COG", message.replace("COG: ", ""))
                return
            except Exception:
                pass
        self.bus.log.emit(message)


class GoalManager:
    """Lifecycle and scoring for long, medium and session goals."""

    def __init__(self, state: CognitiveStateManager) -> None:
        self.state = state

    def create_goal(
        self,
        title: str,
        horizon: GoalHorizon = "session",
        priority: float = 0.5,
        description: str = "",
        project: str = "",
        tags: Iterable[str] | None = None,
    ) -> GoalRecord:
        clean_title = SecuritySanitiser.guard_text(title, "goal.title")[:180]
        clean_description = SecuritySanitiser.guard_text(description, "goal.description")[:800]
        if not clean_title:
            raise ValueError("goal title is required")
        now_id = f"{_slug(clean_title, 'goal', 48)}_{int(time.time())}"
        goal = GoalRecord(
            id=now_id,
            title=clean_title,
            description=clean_description,
            horizon=horizon if horizon in {"long_term", "medium_term", "session"} else "session",
            priority=max(0.0, min(1.0, float(priority))),
            project=_slug(project, "", 80),
            tags=[_slug(t, "tag", 48) for t in (tags or []) if str(t).strip()],
        )
        goal.score(context=clean_title + " " + clean_description, active_project=self.state.memory.active_project)

        def _do(state: dict[str, Any]) -> GoalRecord:
            state.setdefault("goals", {})[goal.id] = asdict(goal)
            active = [g for g in state.setdefault("active_goals", []) if g != goal.id]
            active.insert(0, goal.id)
            state["active_goals"] = active[:80]
            return goal

        created = self.state._mutate(_do)
        self.state.memory.remember(
            MemoryTier.PROJECT if goal.project else MemoryTier.LONG_TERM,
            f"goal_{goal.id}",
            f"{goal.title} | horizon={goal.horizon} | priority={goal.priority:.2f}",
            project=goal.project,
        )
        return created

    def update_goal(self, goal_id: str, **fields: Any) -> Optional[GoalRecord]:
        key = str(goal_id or "").strip()

        def _do(state: dict[str, Any]) -> Optional[GoalRecord]:
            raw = self._find_goal_locked(state, key)
            if raw is None:
                return None
            goal = GoalRecord.from_dict(raw)
            for name in ("title", "description", "project"):
                if name in fields and fields[name] is not None:
                    setattr(goal, name, SecuritySanitiser.guard_text(str(fields[name]), f"goal.{name}")[:800])
            if "horizon" in fields and fields["horizon"] in {"long_term", "medium_term", "session"}:
                goal.horizon = fields["horizon"]
            if "priority" in fields:
                goal.priority = max(0.0, min(1.0, float(fields["priority"])))
            if "progress" in fields:
                goal.progress = max(0.0, min(1.0, float(fields["progress"])))
            if "evidence" in fields and fields["evidence"]:
                goal.evidence.append(SecuritySanitiser.guard_text(str(fields["evidence"]), "goal.evidence")[:280])
                goal.evidence = goal.evidence[-12:]
            goal.score(context=str(fields.get("context") or ""), active_project=self.state.memory.active_project)
            state["goals"][goal.id] = asdict(goal)
            return goal

        return self.state._mutate(_do)

    def complete_goal(self, goal_id: str, evidence: str = "") -> Optional[GoalRecord]:
        goal = self.update_goal(goal_id, progress=1.0, evidence=evidence)
        if goal is None:
            return None

        def _do(state: dict[str, Any]) -> GoalRecord:
            raw = state["goals"][goal.id]
            raw["status"] = "completed"
            raw["completed_at"] = utc_stamp()
            raw["updated_at"] = utc_stamp()
            state["active_goals"] = [g for g in state.get("active_goals", []) if g != goal.id]
            return GoalRecord.from_dict(raw)

        completed = self.state._mutate(_do)
        self.state.memory.remember(MemoryTier.LONG_TERM, f"completed_goal_{completed.id}", completed.title)
        return completed

    def archive_goal(self, goal_id: str) -> Optional[GoalRecord]:
        def _do(state: dict[str, Any]) -> Optional[GoalRecord]:
            raw = self._find_goal_locked(state, goal_id)
            if raw is None:
                return None
            raw["status"] = "archived"
            raw["archived_at"] = utc_stamp()
            raw["updated_at"] = utc_stamp()
            state["active_goals"] = [g for g in state.get("active_goals", []) if g != raw["id"]]
            return GoalRecord.from_dict(raw)

        return self.state._mutate(_do)

    def retrieve_goal(self, goal_id: str) -> Optional[GoalRecord]:
        with self.state._lock:
            raw = self._find_goal_locked(self.state._state, goal_id)
            return GoalRecord.from_dict(raw) if raw is not None else None

    def list_goals(self, include_archived: bool = False, limit: int = 20) -> list[GoalRecord]:
        with self.state._lock:
            goals = [GoalRecord.from_dict(g) for g in self.state._state.get("goals", {}).values()]
        if not include_archived:
            goals = [g for g in goals if g.status != "archived"]
        goals.sort(key=lambda g: (g.status != "active", -g.priority, -g.relevance, g.updated_at), reverse=False)
        return goals[: max(1, min(80, int(limit or 20)))]

    def score_goals(self, context: str = "") -> list[GoalRecord]:
        def _do(state: dict[str, Any]) -> list[GoalRecord]:
            out: list[GoalRecord] = []
            for gid, raw in state.setdefault("goals", {}).items():
                goal = GoalRecord.from_dict(raw)
                goal.score(context=context, active_project=self.state.memory.active_project)
                state["goals"][gid] = asdict(goal)
                out.append(goal)
            return sorted(out, key=lambda g: (g.status == "active", g.priority, g.relevance), reverse=True)

        return self.state._mutate(_do)

    def _find_goal_locked(self, state: dict[str, Any], goal_id: str) -> Optional[dict[str, Any]]:
        goals = state.setdefault("goals", {})
        key = str(goal_id or "").strip()
        if key in goals:
            return goals[key]
        slug = _slug(key, "")
        for gid, raw in goals.items():
            if slug and (slug == _slug(raw.get("title", "")) or slug in gid):
                return raw
        return None


class IntentTracker:
    """Observes repeated requests and turns them into confidence metrics."""

    def __init__(self, state: CognitiveStateManager, memory: MemoryAgent) -> None:
        self.state = state
        self.memory = memory

    def observe_request(self, text: str, project: str = "") -> dict[str, Any]:
        clean = SecuritySanitiser.guard_text(text, "intent.request")[:1000]
        if not clean:
            return {}
        tokens = sorted(_tokens(clean))
        key = "_".join(tokens[:7]) or _slug(clean, "request", 80)
        project_key = _slug(project or self.memory.active_project, "", 64)

        def _do(state: dict[str, Any]) -> dict[str, Any]:
            root = state.setdefault("intent_tracker", {}).setdefault("requests", {})
            record = dict(root.get(key) or {
                "id": key,
                "example": clean[:240],
                "count": 0,
                "first_seen": utc_stamp(),
                "last_seen": "",
                "projects": {},
                "confidence": 0.0,
            })
            record["count"] = int(record.get("count", 0)) + 1
            record["last_seen"] = utc_stamp()
            if project_key:
                projects = dict(record.get("projects") or {})
                projects[project_key] = int(projects.get(project_key, 0)) + 1
                record["projects"] = projects
            project_bonus = min(0.18, len(record.get("projects", {})) * 0.04)
            count_score = min(0.72, int(record["count"]) / 7.0)
            record["confidence"] = round(min(0.96, 0.12 + count_score + project_bonus), 3)
            root[key] = record
            return record

        record = self.state._mutate(_do)
        if float(record.get("confidence", 0.0)) >= 0.62:
            self.memory.remember(
                MemoryTier.LONG_TERM,
                f"intent_{record['id']}",
                (
                    f"Repeated request pattern: {record['example']} "
                    f"(confidence {record['confidence']}, count {record['count']})"
                ),
            )
        return record

    def top_intents(self, limit: int = 8) -> list[dict[str, Any]]:
        with self.state._lock:
            rows = list(
                self.state._state.get("intent_tracker", {})
                .get("requests", {})
                .values()
            )
        rows.sort(key=lambda r: (float(r.get("confidence", 0.0)), int(r.get("count", 0))), reverse=True)
        return rows[: max(1, min(30, int(limit or 8)))]
