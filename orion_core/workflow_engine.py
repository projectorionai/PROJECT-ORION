"""
WorkflowEngine + AutomationManager (Phase 3) — repeatable processes.

A workflow is a NAMED, ORDERED chain of tool calls the dispatcher already
knows how to execute — which means workflows inherit the entire security
layer (sanitiser, system guard, protected paths) for free and cannot do
anything a tool call could not.

    WorkflowEngine     — definitions (persisted to config/workflows.json),
                         background execution with per-step retries,
                         progress tracking on the bus, cancellation
    AutomationManager  — the dispatcher-facing facade; seeds the built-in
                         workflow library on first run

Step arguments support two placeholders, resolved at run time:
    {input}  — the text the workflow was started with
    {prev}   — the text result of the previous step

Run state is mirrored into cognitive state (open_workflows) so an
interrupted run is visible after a restart rather than silently vanishing.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .security import SecuritySanitiser
from .utils import first_line, utc_stamp
from .atomic_io import atomic_write_text

WORKFLOWS_PATH = CONFIG_DIR / "workflows.json"

_MAX_STEPS = 12
#: Workflows started from inside other workflows, at most this deep.
_MAX_NESTING = 3
#: Workflow runs executing at once.
_MAX_RUNNING = 8
#: The chain of workflow names the current task is running inside. A step that
#: starts a workflow runs in a task created from this context, so a workflow
#: that (directly or through another) starts ITSELF is seen and refused —
#: start() returns at once and spawns a task, so without this a self-starting
#: workflow re-spawned itself forever.
_CHAIN: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "orion_workflow_chain", default=())
_MAX_RETRIES = 3


class WorkflowEngine:
    """Definition store + background executor for tool-chain workflows."""

    def __init__(self, bus: OrionBus, dispatcher: Any = None,
                 cognition: Any | None = None, telemetry: Any | None = None,
                 path: Path | None = None) -> None:
        self.bus = bus
        self.dispatcher = dispatcher
        self.cognition = cognition
        self.telemetry = telemetry
        self.path = Path(path) if path is not None else WORKFLOWS_PATH
        self.definitions: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self._load()
        if telemetry is not None:
            telemetry.health.register("workflows")

    # ── definitions ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.definitions = {
                    k: v for k, v in raw.items()
                    if isinstance(v, dict) and isinstance(v.get("steps"), list)
                }
        except Exception:
            self.definitions = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path,
                json.dumps(self.definitions, indent=2), encoding="utf-8")
        except Exception as exc:
            self.bus.log.emit(f"WORKFLOW: save failed - {first_line(exc)}")

    def define(self, name: str, steps: list[dict[str, Any]],
               description: str = "") -> ToolResult:
        name = self._slug(name)
        if not name:
            return ToolResult("A workflow needs a name.", ok=False)
        cleaned: list[dict[str, Any]] = []
        known = set(self.dispatcher.handler_table()) if self.dispatcher is not None else set()
        for step in steps[:_MAX_STEPS]:
            if not isinstance(step, dict):
                continue
            tool = SecuritySanitiser.guard_text(str(step.get("tool") or ""), "workflow.tool")
            if not tool:
                continue
            if known and tool not in known:
                return ToolResult(f"Unknown tool in workflow: '{tool}'.", ok=False)
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            if tool == "workflow" and self._slug(str(args.get("name") or "")) == name:
                return ToolResult(f"A step of '{name}' would run '{name}' again — "
                                  "a workflow cannot start itself.", ok=False)
            cleaned.append({
                "tool": tool,
                "args": {str(k)[:60]: v for k, v in list(args.items())[:12]},
                "retries": max(0, min(_MAX_RETRIES, int(step.get("retries", 1)))),
            })
        if not cleaned:
            return ToolResult("A workflow needs at least one valid step.", ok=False)
        self.definitions[name] = {
            "description": SecuritySanitiser.guard_text(
                str(description or ""), "workflow.description")[:300],
            "steps": cleaned,
            "updated_at": utc_stamp(),
        }
        self._save()
        return ToolResult(f"Workflow '{name}' defined with {len(cleaned)} step(s).")

    def remove(self, name: str) -> ToolResult:
        if self.definitions.pop(self._slug(name), None) is None:
            return ToolResult(f"No workflow named '{name}'.", ok=False)
        self._save()
        return ToolResult(f"Workflow '{name}' removed.")

    def list_workflows(self) -> ToolResult:
        if not self.definitions:
            return ToolResult("No workflows defined yet.")
        lines = ["Workflow library:"]
        for name, definition in sorted(self.definitions.items()):
            steps = definition.get("steps") or []
            chain = " → ".join(str(s.get("tool")) for s in steps[:6])
            lines.append(f"- {name} ({len(steps)} steps: {chain})"
                         + (f" — {definition.get('description')}"
                            if definition.get("description") else ""))
        return ToolResult("\n".join(lines))

    # ── execution ─────────────────────────────────────────────────────────────

    def start(self, name: str, input_text: str = "") -> ToolResult:
        name = self._slug(name)
        definition = self.definitions.get(name)
        if definition is None:
            return ToolResult(f"No workflow named '{name}'. "
                              "Say 'list workflows' to see the library.", ok=False)
        if self.dispatcher is None:
            return ToolResult("The dispatcher is not attached; workflows "
                              "cannot execute yet.", ok=False)
        chain = _CHAIN.get()
        if name in chain:
            return ToolResult(f"'{name}' is already running in this chain "
                              f"({' → '.join(chain)}); not starting it again.", ok=False)
        if len(chain) >= _MAX_NESTING:
            return ToolResult(f"Workflows are nested {len(chain)} deep already; "
                              "not starting another from inside them.", ok=False)
        running = sum(1 for r in self.runs.values() if r.get("status") == "running")
        if running >= _MAX_RUNNING:
            return ToolResult(f"{running} workflows are already running; wait for "
                              "one to finish.", ok=False)
        run_id = f"{name}_{int(time.time())}"
        run = {
            "id": run_id, "workflow": name, "status": "running",
            "step": 0, "total": len(definition["steps"]),
            "input": str(input_text or "")[:2000], "transcript": [],
            "started_at": utc_stamp(),
        }
        self.runs[run_id] = run
        token = _CHAIN.set(chain + (name,))
        try:
            # create_task copies the CURRENT context, so the run (and any
            # workflow its steps start) carries the chain with it.
            run["task"] = asyncio.create_task(self._execute(run, definition))
        finally:
            _CHAIN.reset(token)
        if self.cognition is not None:
            try:
                self.cognition.upsert_workflow(
                    name, {"run": run_id, "status": "running"})
            except Exception:
                pass
        return ToolResult(f"Workflow '{name}' is running in the background "
                          f"({run['total']} steps). I'll report when it completes.")

    async def _execute(self, run: dict[str, Any], definition: dict[str, Any]) -> None:
        prev_text = ""
        status = "complete"
        try:
            for index, step in enumerate(definition["steps"], 1):
                run["step"] = index
                self.bus.dashboard_event.emit("workflow", {
                    "run": run["id"], "workflow": run["workflow"],
                    "step": index, "total": run["total"], "status": "running"})
                args = self._resolve_args(step.get("args") or {},
                                          run["input"], prev_text)
                result: ToolResult | None = None
                attempts = int(step.get("retries", 1)) + 1
                for attempt in range(1, attempts + 1):
                    result = await self.dispatcher.dispatch(step["tool"], args)
                    if result.ok:
                        break
                    if attempt < attempts:
                        self.bus.log.emit(
                            f"WORKFLOW: step {index} ({step['tool']}) failed, "
                            f"retry {attempt}/{attempts - 1}")
                        await asyncio.sleep(min(2.0 * attempt, 6.0))
                prev_text = result.text if result is not None else ""
                run["transcript"].append({
                    "step": index, "tool": step["tool"],
                    "ok": bool(result and result.ok),
                    "text": prev_text[:400],
                })
                if result is None or not result.ok:
                    status = f"failed at step {index} ({step['tool']})"
                    break
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as exc:
            status = f"fault: {first_line(exc, 120)}"
        run["status"] = status
        run["finished_at"] = utc_stamp()
        self.bus.dashboard_event.emit("workflow", {
            "run": run["id"], "workflow": run["workflow"],
            "step": run["step"], "total": run["total"], "status": status})
        self.bus.log.emit(f"WORKFLOW: '{run['workflow']}' {status}.")
        if status == "complete":
            self.bus.banner.emit(f"WORKFLOW COMPLETE: {run['workflow']}", 3)
        if self.telemetry is not None:
            self.telemetry.metrics.incr(
                "workflow.complete" if status == "complete" else "workflow.failed")
        if self.cognition is not None:
            try:
                if status == "complete":
                    self.cognition.close_workflow(run["workflow"])
                else:
                    self.cognition.upsert_workflow(
                        run["workflow"], {"run": run["id"], "status": status})
            except Exception:
                pass

    def cancel(self, run_id: str = "") -> ToolResult:
        candidates = [r for r in self.runs.values()
                      if r.get("status") == "running"
                      and (not run_id or r["id"] == run_id)]
        if not candidates:
            return ToolResult("No running workflow to cancel.")
        for run in candidates:
            task = run.get("task")
            if task is not None and not task.done():
                task.cancel()
        return ToolResult(f"Cancelling {len(candidates)} workflow run(s).")

    def status(self) -> ToolResult:
        if not self.runs:
            return ToolResult("No workflow runs this session.")
        lines = ["Workflow runs:"]
        for run in list(self.runs.values())[-8:]:
            lines.append(f"- {run['workflow']} [{run['status']}] "
                         f"step {run['step']}/{run['total']}")
            for entry in run["transcript"][-2:]:
                mark = "ok" if entry["ok"] else "FAILED"
                lines.append(f"    {entry['step']}. {entry['tool']} ({mark})")
        return ToolResult("\n".join(lines))

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_args(args: dict[str, Any], input_text: str,
                      prev_text: str) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                value = value.replace("{input}", input_text)
                value = value.replace("{prev}", prev_text[:2000])
            resolved[key] = value
        return resolved

    @staticmethod
    def _slug(name: str) -> str:
        return "".join(c for c in str(name or "").lower().replace(" ", "_")
                       if c.isalnum() or c == "_").strip("_")[:60]


class AutomationManager:
    """Dispatcher-facing facade; seeds the built-in workflow library."""

    def __init__(self, engine: WorkflowEngine) -> None:
        self.engine = engine
        self._seed_builtins()

    def _seed_builtins(self) -> None:
        for name, definition in _BUILTIN_WORKFLOWS.items():
            if name not in self.engine.definitions:
                self.engine.definitions[name] = dict(definition,
                                                     updated_at=utc_stamp())
        self.engine._save()

    async def handle(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        name = str(args.get("name") or args.get("workflow") or "")
        if action in {"list", "library"}:
            return self.engine.list_workflows()
        if action in {"run", "start", "execute"}:
            return self.engine.start(name, str(args.get("input")
                                               or args.get("text") or ""))
        if action in {"status", "progress"}:
            return self.engine.status()
        if action in {"cancel", "stop"}:
            return self.engine.cancel(str(args.get("run") or ""))
        if action in {"define", "create"}:
            steps = args.get("steps")
            if isinstance(steps, str):
                try:
                    steps = json.loads(steps)
                except Exception:
                    return ToolResult("Steps must be a JSON list of "
                                      "{tool, args} objects.", ok=False)
            if not isinstance(steps, list):
                return ToolResult("Steps must be a list of {tool, args} "
                                  "objects.", ok=False)
            return self.engine.define(name, steps,
                                      str(args.get("description") or ""))
        if action in {"remove", "delete"}:
            return self.engine.remove(name)
        return ToolResult("Unsupported workflow action. Use list, run, status, "
                          "cancel, define, or remove.", ok=False)


_BUILTIN_WORKFLOWS: dict[str, dict[str, Any]] = {
    "research_workflow": {
        "description": "Queue autonomous research on a topic, then show the agenda.",
        "steps": [
            {"tool": "research", "args": {"action": "queue", "topic": "{input}"},
             "retries": 1},
            {"tool": "research", "args": {"action": "agenda"}, "retries": 0},
        ],
    },
    "creator_review_workflow": {
        "description": "Score a creator script (hook, structure, CTA) and log the review.",
        "steps": [
            {"tool": "creator_intel",
             "args": {"action": "review_script", "script": "{input}"},
             "retries": 1},
        ],
    },
    "product_analysis_workflow": {
        "description": "Analyse a product for short-form commerce potential.",
        "steps": [
            {"tool": "creator_intel",
             "args": {"action": "product", "product": "{input}"}, "retries": 1},
        ],
    },
    "daily_review_workflow": {
        "description": "Executive focus, goal review and companion continuity in one pass.",
        "steps": [
            {"tool": "executive", "args": {"action": "focus"}, "retries": 0},
            {"tool": "executive", "args": {"action": "goals"}, "retries": 0},
            {"tool": "companion", "args": {"action": "brief"}, "retries": 0},
        ],
    },
}


__all__ = ["AutomationManager", "WorkflowEngine", "WORKFLOWS_PATH"]
