"""
MissionEngine — ORION's mission-based operating model.

ORION stops being a pile of tools and starts operating around MISSIONS: the
user's real, long-running endeavours.  Each mission is a durable container —

    files · research topics · tasks · goals · memory notes · agents ·
    reports · workflows

— stored in ``config/missions.json`` (same durable-JSON pattern as the creator
roster), so missions survive restarts and are cheap to read from any panel.

Awareness, not bookkeeping: the engine computes progress from its own task and
goal ledgers, surfaces outstanding work, flags risks (stalled missions, tasks
past due, research never linked) and recommends the next move.  The current
mission gives every other layer context — briefings lead with it, the deck's
Mission Control panel renders it, and the executive core can weigh it.

Composes (all optional, degrade silently): CognitiveStateManager for
cross-linking pending tasks, ResearchDirector for queueing linked research,
and the bus for dashboard events.  Deterministic and offline throughout.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import user_profile
from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .utils import utc_stamp
from .atomic_io import atomic_write_text

MISSIONS_PATH = CONFIG_DIR / "missions.json"

# Starter missions — seeded once, never re-imposed after edits.  The names are
# personal, so the publishable defaults live in user_profile and the operator's
# real endeavours come from config/profile.json (which is never committed).
DEFAULT_MISSIONS = tuple((name, brief) for name, brief in user_profile.DEFAULTS["missions"])

_STALL_DAYS = 7.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _age_days(stamp: str) -> float:
    try:
        then = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max(0.0, (_now() - then).total_seconds() / 86_400.0)
    except Exception:
        return 0.0


class MissionEngine:
    """Durable missions with computed progress, risks and recommendations."""

    def __init__(self, bus: OrionBus | None = None,
                 cognition: Any | None = None,
                 research_director: Any | None = None,
                 telemetry: Any | None = None,
                 path: Path | None = None) -> None:
        self.bus = bus
        self.cognition = cognition
        self.research_director = research_director
        self.telemetry = telemetry
        self.path = Path(path) if path is not None else MISSIONS_PATH
        self._lock = threading.Lock()
        self._seed()
        if telemetry is not None:
            telemetry.health.register("missions")

    # ── storage ───────────────────────────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(data, indent=2), encoding="utf-8")

    def _seed(self) -> None:
        with self._lock:
            data = self.load()
            if data.get("missions"):
                return
            starters = user_profile.missions()
            current = next((name for name, _ in starters if name == "Develop ORION"), starters[0][0])
            data = {"current": current, "missions": {}}
            for name, brief in starters:
                data["missions"][name] = self._blank(name, brief)
            self._save(data)

    @staticmethod
    def _blank(name: str, brief: str = "") -> dict[str, Any]:
        return {
            "name": name, "brief": brief, "status": "active",
            "created_at": utc_stamp(), "updated_at": utc_stamp(),
            "goals": [], "tasks": [], "files": [], "research": [],
            "notes": [], "reports": [], "workflows": [], "agents": [],
        }

    # ── mutation helpers ──────────────────────────────────────────────────────

    def _with_mission(self, name: str, mutate: Any) -> dict[str, Any] | None:
        """Run *mutate(record)* under the lock; returns the record or None."""
        name = str(name or "").strip()
        with self._lock:
            data = self.load()
            missions = data.setdefault("missions", {})
            record = self._resolve(missions, name)
            if record is None:
                return None
            mutate(record)
            record["updated_at"] = utc_stamp()
            self._save(data)
        if self.bus is not None:
            self.bus.dashboard_event.emit("mission", {
                "name": record["name"], "status": "updated"})
        return record

    @staticmethod
    def _resolve(missions: dict[str, Any], name: str) -> dict[str, Any] | None:
        if not name:
            return None
        if name in missions:
            return missions[name]
        lowered = name.lower()
        for key, record in missions.items():
            if lowered in key.lower():
                return record
        return None

    # ── the operating surface ─────────────────────────────────────────────────

    def current(self) -> dict[str, Any] | None:
        data = self.load()
        return self._resolve(data.get("missions") or {},
                             str(data.get("current") or ""))

    def set_current(self, name: str) -> ToolResult:
        with self._lock:
            data = self.load()
            record = self._resolve(data.get("missions") or {}, str(name or ""))
            if record is None:
                return ToolResult(f"No mission matching '{name}'.", ok=False)
            data["current"] = record["name"]
            self._save(data)
        if self.bus is not None:
            self.bus.dashboard_event.emit("mission", {
                "name": record["name"], "status": "current"})
        return ToolResult(f"Current mission set: {record['name']}.")

    def create(self, name: str, brief: str = "") -> ToolResult:
        name = str(name or "").strip()[:80]
        if not name:
            return ToolResult("Every mission needs a name.", ok=False)
        with self._lock:
            data = self.load()
            missions = data.setdefault("missions", {})
            if name in missions:
                return ToolResult(f"Mission '{name}' already exists.", ok=False)
            missions[name] = self._blank(name, str(brief or "")[:300])
            self._save(data)
        return ToolResult(f"Mission '{name}' is on the board.")

    def add_goal(self, mission: str, goal: str) -> ToolResult:
        goal = str(goal or "").strip()[:200]
        record = self._with_mission(
            mission, lambda r: r["goals"].append(
                {"goal": goal, "done": False, "at": utc_stamp()}))
        if record is None or not goal:
            return ToolResult("Mission or goal missing.", ok=False)
        return ToolResult(f"Goal added to {record['name']}: {goal}")

    def add_task(self, mission: str, task: str, due: str = "") -> ToolResult:
        task = str(task or "").strip()[:200]
        record = self._with_mission(
            mission, lambda r: r["tasks"].append(
                {"task": task, "done": False, "due": str(due or ""),
                 "at": utc_stamp()}))
        if record is None or not task:
            return ToolResult("Mission or task missing.", ok=False)
        return ToolResult(f"Task logged against {record['name']}: {task}")

    def complete(self, mission: str, item: str) -> ToolResult:
        """Mark the first open task or goal containing *item* as done."""
        item_l = str(item or "").strip().lower()
        hit: list[str] = []

        def mutate(record: dict[str, Any]) -> None:
            for kind in ("tasks", "goals"):
                for entry in record[kind]:
                    text = str(entry.get("task") or entry.get("goal") or "")
                    if not entry.get("done") and item_l and item_l in text.lower():
                        entry["done"] = True
                        entry["done_at"] = utc_stamp()
                        hit.append(text)
                        return

        record = self._with_mission(mission, mutate)
        if record is None:
            return ToolResult(f"No mission matching '{mission}'.", ok=False)
        if not hit:
            return ToolResult(f"Nothing open matching '{item}' on "
                              f"{record['name']}.", ok=False)
        return ToolResult(f"Done and logged on {record['name']}: {hit[0]}")

    # ── removal ───────────────────────────────────────────────────────────────
    #
    # The board had no way to take anything OFF it. You could create a mission,
    # add goals and tasks, and mark them done — but a mission started by
    # mistake, or a task that stopped being relevant, stayed on the deck for
    # ever. "I want ORION to be able to get rid of things inside of MISSION
    # without restarting."
    #
    # Everything here saves immediately and emits on the bus, so the deck
    # re-renders from the same tick — no restart, and nothing to remember to
    # refresh.

    def remove_item(self, mission: str, item: str, kind: str = "") -> ToolResult:
        """Delete a goal, task or attachment matching *item*.

        Matches on substring, case-insensitively, and takes the FIRST hit —
        the same rule ``complete`` uses, so "get rid of X" and "mark X done"
        behave consistently rather than being two different searches.
        """
        needle = str(item or "").strip().lower()
        if not needle:
            return ToolResult("Name the item to remove.", ok=False)
        wanted = self._ITEM_KINDS.get(str(kind or "").strip().lower())
        searched = (wanted,) if wanted else (
            "tasks", "goals", "notes", "files", "research", "reports",
            "workflows", "agents")
        removed: list[str] = []

        def mutate(record: dict[str, Any]) -> None:
            for bucket in searched:
                entries = record.get(bucket)
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    text = self._entry_text(entry)
                    if needle in text.lower():
                        entries.pop(index)
                        removed.append(f"{text} (from {bucket})")
                        return

        record = self._with_mission(mission, mutate)
        if record is None:
            return ToolResult(f"No mission matching '{mission}'.", ok=False)
        if not removed:
            return ToolResult(
                f"Nothing on {record['name']} matches '{item}'.", ok=False)
        return ToolResult(f"Removed from {record['name']}: {removed[0]}")

    def delete(self, name: str, confirm: bool = True) -> ToolResult:
        """Delete a whole mission and everything on it.

        Irreversible, so it reports exactly what went with it — a mission
        carrying twelve tasks disappearing silently would be alarming.
        """
        wanted = str(name or "").strip()
        if not wanted:
            return ToolResult("Name the mission to delete.", ok=False)
        with self._lock:
            data = self.load()
            missions = data.setdefault("missions", {})
            record = self._resolve(missions, wanted)
            if record is None:
                return ToolResult(f"No mission matching '{name}'.", ok=False)
            real = record["name"]
            key = next((k for k, v in missions.items() if v is record), real)
            tasks = len(record.get("tasks") or [])
            goals = len(record.get("goals") or [])
            missions.pop(key, None)
            if str(data.get("current") or "") in (key, real):
                # Never leave the board pointing at something that is gone.
                data["current"] = next(iter(missions), "")
            self._save(data)
        if self.bus is not None:
            self.bus.dashboard_event.emit("mission", {"name": real,
                                                      "status": "deleted"})
        carried = f" It carried {goals} goal(s) and {tasks} task(s)." if (goals or tasks) else ""
        return ToolResult(f"Deleted the {real} mission.{carried}")

    def archive(self, name: str) -> ToolResult:
        """Take a mission off the active board without destroying it.

        The gentler option, and the right default for "I'm finished with this":
        the record and its history survive, it simply stops occupying the deck.
        """
        def mutate(record: dict[str, Any]) -> None:
            record["status"] = "archived"

        record = self._with_mission(name, mutate)
        if record is None:
            return ToolResult(f"No mission matching '{name}'.", ok=False)
        with self._lock:
            data = self.load()
            if str(data.get("current") or "") == record["name"]:
                active = [k for k, v in (data.get("missions") or {}).items()
                          if v.get("status") != "archived"]
                data["current"] = active[0] if active else ""
                self._save(data)
        return ToolResult(f"Archived {record['name']} — it is off the board but "
                          "nothing was lost.")

    def clear_completed(self, mission: str = "") -> ToolResult:
        """Sweep every finished task and goal off one mission, or all of them."""
        swept = {"n": 0}

        def sweep(record: dict[str, Any]) -> None:
            for bucket in ("tasks", "goals"):
                entries = record.get(bucket)
                if not isinstance(entries, list):
                    continue
                keep = [e for e in entries if not e.get("done")]
                swept["n"] += len(entries) - len(keep)
                record[bucket] = keep

        if mission.strip():
            record = self._with_mission(mission, sweep)
            if record is None:
                return ToolResult(f"No mission matching '{mission}'.", ok=False)
            where = record["name"]
        else:
            with self._lock:
                data = self.load()
                for record in (data.get("missions") or {}).values():
                    sweep(record)
                    record["updated_at"] = utc_stamp()
                self._save(data)
            if self.bus is not None:
                self.bus.dashboard_event.emit("mission", {"name": "*",
                                                          "status": "updated"})
            where = "the whole board"
        if not swept["n"]:
            return ToolResult(f"Nothing completed to clear from {where}.")
        return ToolResult(f"Cleared {swept['n']} completed item(s) from {where}.")

    #: Spoken names for the buckets, so "remove the note about X" works.
    _ITEM_KINDS = {
        "task": "tasks", "tasks": "tasks",
        "goal": "goals", "goals": "goals",
        "note": "notes", "notes": "notes", "memory": "notes",
        "file": "files", "files": "files",
        "research": "research", "report": "reports", "reports": "reports",
        "workflow": "workflows", "workflows": "workflows",
        "agent": "agents", "agents": "agents",
    }

    @staticmethod
    def _entry_text(entry: Any) -> str:
        """The human text of an entry, whatever shape its bucket uses."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            for key in ("task", "goal", "note", "value", "text", "name", "title"):
                if entry.get(key):
                    return str(entry[key])
            return str(next(iter(entry.values()), ""))
        return str(entry)

    def attach(self, mission: str, kind: str, value: str) -> ToolResult:
        """Attach a file / research topic / note / report / workflow / agent."""
        kind = {"file": "files", "research": "research", "note": "notes",
                "memory": "notes", "report": "reports", "workflow": "workflows",
                "agent": "agents"}.get(str(kind or "").lower().strip(), "")
        value = str(value or "").strip()[:400]
        if not kind or not value:
            return ToolResult(
                "Attach what? kind=file|research|note|report|workflow|agent "
                "plus the value.", ok=False)
        record = self._with_mission(
            mission, lambda r: r[kind].append({"value": value, "at": utc_stamp()}))
        if record is None:
            return ToolResult(f"No mission matching '{mission}'.", ok=False)
        # Research attachments can go straight onto the research agenda.
        if kind == "research" and self.research_director is not None:
            # spawn, not create_task: asyncio keeps only a WEAK reference,
            # so a discarded task can be collected mid-flight and simply
            # stop — reporting "research attached" while the topic never
            # reached the agenda.
            from .tasks import spawn

            spawn(self.research_director.queue_topic(value),
                  bus=getattr(self, "bus", None),
                  label=f"queueing research on {value!r}")
        return ToolResult(f"{kind[:-1].title()} attached to {record['name']}.")

    # ── awareness ─────────────────────────────────────────────────────────────

    @staticmethod
    def progress(record: dict[str, Any]) -> float:
        """0-100 from the task and goal ledgers (goals weigh double)."""
        tasks = record.get("tasks") or []
        goals = record.get("goals") or []
        weight = len(tasks) + 2 * len(goals)
        if weight == 0:
            return 0.0
        done = (sum(1 for t in tasks if t.get("done"))
                + 2 * sum(1 for g in goals if g.get("done")))
        return round(100.0 * done / weight, 1)

    def risks(self, record: dict[str, Any]) -> list[str]:
        risks: list[str] = []
        age = _age_days(str(record.get("updated_at") or ""))
        if age > _STALL_DAYS and str(record.get("status")) == "active":
            risks.append(f"stalled — no activity for {age:.0f} days")
        overdue = [t for t in record.get("tasks") or []
                   if not t.get("done") and t.get("due")
                   and _age_days(str(t["due"])) > 0.0
                   and str(t["due"]) < utc_stamp()]
        if overdue:
            risks.append(f"{len(overdue)} task(s) past due")
        if (record.get("goals") and not record.get("research")
                and str(record.get("status")) == "active"):
            risks.append("no research linked — decisions may be unsupported")
        return risks

    def recommendations(self, record: dict[str, Any]) -> list[str]:
        recs: list[str] = []
        open_tasks = [t for t in record.get("tasks") or [] if not t.get("done")]
        if open_tasks:
            recs.append(f"next task: {open_tasks[0].get('task', '')[:80]}")
        elif record.get("goals"):
            open_goals = [g for g in record["goals"] if not g.get("done")]
            if open_goals:
                recs.append("goals are set but no tasks break them down — "
                            f"decompose '{open_goals[0].get('goal', '')[:60]}'")
        else:
            recs.append("define the first goal so progress becomes measurable")
        if not record.get("research"):
            recs.append("queue one research topic to build the evidence base")
        return recs

    # ── tool surface ──────────────────────────────────────────────────────────

    def overview(self) -> ToolResult:
        data = self.load()
        missions = data.get("missions") or {}
        if not missions:
            return ToolResult("No missions on the board yet.")
        current = str(data.get("current") or "")
        lines = ["Mission board:"]
        for name, record in missions.items():
            mark = "▶" if name == current else "•"
            open_tasks = sum(1 for t in record.get("tasks") or []
                             if not t.get("done"))
            lines.append(
                f"{mark} {name} — {self.progress(record):.0f}% · "
                f"{open_tasks} open task(s) · {record.get('status', 'active')}")
        lines.append(f"Current mission: {current or 'none set'}.")
        return ToolResult("\n".join(lines))

    def status(self, name: str = "") -> ToolResult:
        data = self.load()
        record = self._resolve(data.get("missions") or {},
                               str(name or data.get("current") or ""))
        if record is None:
            return ToolResult(f"No mission matching '{name}'.", ok=False)
        lines = [f"Mission — {record['name']} "
                 f"({self.progress(record):.0f}% complete)"]
        if record.get("brief"):
            lines.append(record["brief"])
        open_tasks = [t for t in record.get("tasks") or [] if not t.get("done")]
        if open_tasks:
            lines.append("Outstanding:")
            lines.extend(f"  - {t.get('task', '')[:80]}"
                         + (f" (due {t['due'][:10]})" if t.get("due") else "")
                         for t in open_tasks[:6])
        goals = [g for g in record.get("goals") or [] if not g.get("done")]
        if goals:
            lines.append("Open goals: " + "; ".join(
                str(g.get("goal", ""))[:50] for g in goals[:3]))
        for label, key in (("Research", "research"), ("Files", "files"),
                           ("Reports", "reports")):
            items = record.get(key) or []
            if items:
                lines.append(f"{label}: {len(items)} linked — latest "
                             f"{str(items[-1].get('value', ''))[:60]}")
        risks = self.risks(record)
        if risks:
            lines.append("Risks: " + "; ".join(risks))
        recs = self.recommendations(record)
        if recs:
            lines.append("Recommended: " + "; ".join(recs))
        return ToolResult("\n".join(lines))

    def panel_snapshot(self) -> dict[str, Any]:
        """Cheap read for the Mission Control panel: everything, computed."""
        data = self.load()
        missions = data.get("missions") or {}
        return {
            "current": str(data.get("current") or ""),
            "missions": [{
                "name": record.get("name", name),
                "status": record.get("status", "active"),
                "progress": self.progress(record),
                "open_tasks": sum(1 for t in record.get("tasks") or []
                                  if not t.get("done")),
                "risks": self.risks(record),
                "recommendation": (self.recommendations(record) or [""])[0],
            } for name, record in missions.items()],
        }

    async def handle(self, args: dict[str, Any]) -> ToolResult:
        """Dispatcher-facing surface for the `mission` tool."""
        action = str(args.get("action") or "").lower().strip()
        name = str(args.get("mission") or args.get("name") or "")
        text = str(args.get("text") or args.get("value") or "")
        if action in {"overview", "board", "list", ""}:
            return self.overview()
        if action in {"status", "report", "progress"}:
            return self.status(name)
        if action in {"current", "switch", "set_current", "focus"}:
            return self.set_current(name or text)
        if action in {"create", "new"}:
            return self.create(name or text, str(args.get("brief") or ""))
        if action in {"goal", "add_goal"}:
            return self.add_goal(name, text)
        if action in {"task", "add_task"}:
            return self.add_task(name, text, str(args.get("due") or ""))
        if action in {"complete", "done", "finish"}:
            return self.complete(name, text)
        if action == "attach":
            return self.attach(name, str(args.get("kind") or ""), text)
        # ── removal ──────────────────────────────────────────────────────────
        # "remove"/"delete" are ambiguous in speech: "delete that task" means an
        # ITEM, "delete the mission" means the whole thing. Resolved by whether
        # item text was supplied, which is what the user actually said.
        if action in {"remove", "delete", "drop", "get_rid_of", "clear"}:
            if text:
                return self.remove_item(name, text, str(args.get("kind") or ""))
            if action == "clear":
                return self.clear_completed(name)
            return self.delete(name)
        if action in {"remove_item", "delete_item", "remove_task", "delete_task"}:
            return self.remove_item(name, text, str(args.get("kind") or ""))
        if action in {"delete_mission", "remove_mission"}:
            return self.delete(name or text)
        if action in {"archive", "shelve", "retire"}:
            return self.archive(name or text)
        if action in {"clear_completed", "sweep", "tidy"}:
            return self.clear_completed(name)
        return ToolResult(
            "Unsupported mission action. Use overview, status, current, "
            "create, goal, task, complete, attach, remove, delete, archive "
            "or clear_completed.", ok=False)


__all__ = ["DEFAULT_MISSIONS", "MissionEngine"]
