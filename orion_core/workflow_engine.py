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

Runs are also durable jobs (``job_store``) admitted as USER work
(``admission``): a run beyond the running limit is queued and says so, each
step's progress and a checkpoint (next step, previous result) are recorded,
pause and cancel land between steps, and after a restart an unfinished run is
INTERRUPTED. ``resume`` continues it from the step that was in flight — at
once if that step is read-only, and only after an approval bound to that
exact step if it changes anything, because it may already have happened.
Run ids come from a uuid; they used to be ``name_<unix seconds>``, so two
starts of one workflow in the same second overwrote each other.

A finished run keeps what it found (``result``): every step output that no
later step read through {prev}, bounded to the job record's 4,000
characters, on the run and in its job record, so "what did my workflow find"
has an answer this session and after a restart. A step marked ``optional``
may fail without stopping the run (one offline source must not sink a
morning check). ``WorkflowScheduler`` runs a named workflow at a time of day
(once, daily, on weekdays or weekly); see its docstring for how it waits for
a quiet moment and why a restart can never run an occurrence twice.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import re
import uuid
from dataclasses import replace
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from typing import Any

from . import job_store as _store_mod
from .admission import Busy, WorkClass
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
_MAX_LIBRARY_CHARS = 4_000_000
_MAX_PREVIOUS_CHARS = 2000
#: What a finished run keeps of what it found: the job record's own limit.
_MAX_RESULT_CHARS = 4000
#: Marks text that was shortened to fit; nothing is cut silently.
_CUT_MARK = " [...]"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate workflow fields")
        result[key] = value
    return result


def _read_library(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            text = stream.read(_MAX_LIBRARY_CHARS + 1)
    except FileNotFoundError:
        return None
    if len(text) > _MAX_LIBRARY_CHARS:
        raise ValueError("workflow library exceeds the read limit")
    return text


def _check_argument_budget(value: Any, label: str) -> None:
    """Use the same serialized JSON budget as the durable job store."""
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ValueError(f"{label}: arguments must be complete finite JSON.") from None
    if len(encoded) > _store_mod.MAX_ARGS_CHARS:
        raise ValueError(f"{label}: serialized JSON exceeds the 20,000-character limit; "
                         "no shortened data was used. Save large content and pass its path.")


class WorkflowEngine:
    """Definition store + background executor for tool-chain workflows."""

    def __init__(self, bus: OrionBus, dispatcher: Any = None,
                 cognition: Any | None = None, telemetry: Any | None = None,
                 path: Path | None = None, *, store: Any = None,
                 admission: Any = None) -> None:
        self.bus = bus
        self.dispatcher = dispatcher
        self.cognition = cognition
        self.telemetry = telemetry
        self.path = Path(path) if path is not None else WORKFLOWS_PATH
        self.definitions: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self._store = store
        self._admission = admission
        self._load_error = False
        self._loaded_text: str | None = None
        self._load()
        from .jobs import register_resumer
        register_resumer("workflow", self.resume_job)
        if telemetry is not None:
            telemetry.health.register("workflows")

    @property
    def store(self) -> Any:
        return self._store if self._store is not None else _store_mod.default_store()

    def _admission_obj(self) -> Any:
        if self._admission is not None:
            return self._admission
        from . import admission
        return admission.ADMISSION

    # ── definitions ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            text = _read_library(self.path)
            raw = json.loads(text, object_pairs_hook=_unique_object) if text is not None else {}
            if not isinstance(raw, dict):
                raise ValueError("workflow library must be an object")
            for name, definition in raw.items():
                steps = definition.get("steps") if isinstance(definition, dict) else None
                if not name or not isinstance(steps, list) or not 1 <= len(steps) <= _MAX_STEPS:
                    raise ValueError("invalid stored workflow")
                for step in steps:
                    if not isinstance(step, dict) or not isinstance(step.get("tool"), str) or not step["tool"].strip():
                        raise ValueError("invalid stored workflow step")
                    args = step.get("args", {})
                    retries = step.get("retries", 1)
                    if (not isinstance(args, dict) or isinstance(retries, bool) or not isinstance(retries, int)
                            or not 0 <= retries <= _MAX_RETRIES
                            or not isinstance(step.get("optional", False), bool)
                            or len(json.dumps(args, allow_nan=False)) > _store_mod.MAX_ARGS_CHARS):
                        raise ValueError("invalid stored workflow arguments")
            self.definitions = raw
            self._loaded_text = text
            self._load_error = False
        except (OSError, ValueError, TypeError, RecursionError):
            self.definitions = {}
            self._load_error = True

    def _unavailable(self) -> ToolResult:
        return ToolResult("The workflow library is unreadable or has changed outside this session. "
                          "Its file has been retained. Repair or review workflows.json, then reload ORION "
                          "before changing or starting workflows.", ok=False)

    def _library_is_current(self) -> bool:
        """Refuse new work from an observed stale/unreadable library.

        Already admitted jobs own their snapshot. This is a bounded observation
        before admission, not a lock or an interprocess transaction.
        """
        if self._load_error:
            return False
        try:
            if _read_library(self.path) == self._loaded_text:
                return True
        except (OSError, ValueError, TypeError, RecursionError):
            pass
        self._load_error = True
        self.bus.log.emit("WORKFLOW: library changed or became unreadable; "
                          "reload before starting or recovering workflows.")
        return False

    def _save(self, definitions: dict[str, dict[str, Any]] | None = None) -> bool:
        if self._load_error:
            return False
        try:
            if _read_library(self.path) != self._loaded_text:
                self._load_error = True
                self.bus.log.emit("WORKFLOW: library changed outside this session; refusing to overwrite it.")
                return False
            text = json.dumps(self.definitions if definitions is None else definitions, indent=2, allow_nan=False)
            if len(text) > _MAX_LIBRARY_CHARS:
                raise ValueError("workflow library exceeds the size limit")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path, text, encoding="utf-8")
            self._loaded_text = text
        except Exception as exc:
            self.bus.log.emit(f"WORKFLOW: save failed - {first_line(exc)}")
            return False
        return True

    def define(self, name: str, steps: list[dict[str, Any]],
               description: str = "") -> ToolResult:
        if self._load_error:
            return self._unavailable()
        name = self._slug(name)
        if not name:
            return ToolResult("A workflow needs a name.", ok=False)
        if not isinstance(steps, list) or not 1 <= len(steps) <= _MAX_STEPS:
            return ToolResult(f"A workflow needs 1 to {_MAX_STEPS} steps; no steps were saved.", ok=False)
        cleaned: list[dict[str, Any]] = []
        known = set(self.dispatcher.handler_table()) if self.dispatcher is not None else set()
        for index, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                return ToolResult(f"Workflow step {index} must be an object; nothing was saved.", ok=False)
            tool = SecuritySanitiser.guard_text(str(step.get("tool") or ""), "workflow.tool")
            if not tool:
                return ToolResult(f"Workflow step {index} needs a tool; nothing was saved.", ok=False)
            if known and tool not in known:
                return ToolResult(f"Unknown tool in workflow: '{tool}'.", ok=False)
            args = step.get("args", {})
            if not isinstance(args, dict) or not all(isinstance(key, str) for key in args):
                return ToolResult(f"Workflow step {index} needs an argument object with text keys.", ok=False)
            try:
                encoded = json.dumps(args, allow_nan=False)
                if len(encoded) > _store_mod.MAX_ARGS_CHARS:
                    raise ValueError("arguments exceed the durable step limit")
                args = json.loads(encoded)
                retries = int(step.get("retries", 1))
            except (ValueError, TypeError, OverflowError, RecursionError) as exc:
                return ToolResult(f"Invalid workflow step {index}: {first_line(exc, 120)}. Nothing was saved.", ok=False)
            if tool == "workflow" and self._slug(str(args.get("name") or "")) == name:
                return ToolResult(f"A step of '{name}' would run '{name}' again — "
                                  "a workflow cannot start itself.", ok=False)
            cleaned.append({
                "tool": tool,
                "args": args,
                "retries": max(0, min(_MAX_RETRIES, retries)),
            })
            # Only an explicit yes: bool("false") is True.
            optional = step.get("optional", False)
            if optional is True or str(optional).strip().lower() in {"true", "yes", "1"}:
                cleaned[-1]["optional"] = True
        if not cleaned:
            return ToolResult("A workflow needs at least one valid step.", ok=False)
        updated = dict(self.definitions)
        updated[name] = {
            "description": SecuritySanitiser.guard_text(
                str(description or ""), "workflow.description")[:300],
            "steps": cleaned,
            "updated_at": utc_stamp(),
        }
        if not self._save(updated):
            return ToolResult(f"Workflow '{name}' could not be saved; its definition "
                              "has not changed. Check the log and try again.", ok=False)
        self.definitions = updated
        return ToolResult(f"Workflow '{name}' defined with {len(cleaned)} step(s).")

    def remove(self, name: str) -> ToolResult:
        if self._load_error:
            return self._unavailable()
        key = self._slug(name)
        if key not in self.definitions:
            return ToolResult(f"No workflow named '{name}'.", ok=False)
        updated = dict(self.definitions)
        del updated[key]
        if not self._save(updated):
            return ToolResult(f"Workflow '{name}' could not be removed from disk; "
                              "it remains available. Check the log and try again.", ok=False)
        self.definitions = updated
        return ToolResult(f"Workflow '{name}' removed.")

    def list_workflows(self) -> ToolResult:
        if not self._library_is_current():
            return self._unavailable()
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
        if not self._library_is_current():
            return self._unavailable()
        input_text = str(input_text or "")
        if len(input_text) > 2000:
            return ToolResult("Workflow input exceeds 2,000 characters. Shorten it explicitly; "
                              "no shortened job was started.", ok=False)
        name = self._slug(name)
        definition = self.definitions.get(name)
        if definition is None:
            return ToolResult(f"No workflow named '{name}'. "
                              "Say 'list workflows' to see the library.", ok=False)
        if self.dispatcher is None:
            return ToolResult("The dispatcher is not attached; workflows "
                              "cannot execute yet.", ok=False)
        # Queued and running work owns the definition it was started with.
        # Later library edits or caller-owned nested dicts cannot change it.
        definition = copy.deepcopy(definition)
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
        try:
            job_args = {"workflow": name, "input": input_text,
                        "definition_digest": self._definition_digest(definition)}
            _check_argument_budget(job_args, "Workflow job metadata")
        except ValueError as exc:
            return ToolResult(first_line(exc, 220), ok=False)
        from .jobs import JobControl
        try:
            ticket = self._admission_obj().submit(WorkClass.USER, label=f"workflow {name}")
        except Busy as busy:
            return ToolResult(f"Workflow '{name}' not started — {busy.reason}", ok=False)
        try:
            control = JobControl.create(
                kind="workflow", name=name, label=f"workflow {name}", cls=WorkClass.USER,
                args=job_args,
                steps=[{"index": i, "tool": s.get("tool"), "args": s.get("args") or {}}
                       for i, s in enumerate(definition["steps"], 1)],
                store=self.store, admission=self._admission, ticket=ticket)
        except Exception as exc:
            # create() has already given the slot back.
            self.bus.log.emit(f"WORKFLOW: could not record '{name}' - {first_line(exc)}")
            return ToolResult(f"Workflow '{name}' could not be recorded and was not "
                              "started. Check the log and try again.", ok=False)
        run_id = f"{name}_{uuid.uuid4().hex[:8]}"
        run = {
            "id": run_id, "workflow": name,
            "status": "running" if ticket.running else "queued",
            "step": 0, "total": len(definition["steps"]),
            "input": input_text, "transcript": [],
            "started_at": utc_stamp(), "job_id": control.id,
        }
        self.runs[run_id] = run
        self._launch(run, definition, control, chain + (name,))
        self._sync_cognition(name)
        if run["status"] == "queued":
            return ToolResult(f"Workflow '{name}' is queued behind other background work "
                              f"({ticket.position() or 1} in line, {run['total']} steps). "
                              "I'll report when it completes.")
        return ToolResult(f"Workflow '{name}' is running in the background "
                          f"({run['total']} steps). I'll report when it completes.")

    def _launch(self, run: dict[str, Any], definition: dict[str, Any], control: Any,
                chain: tuple[str, ...], start: int = 1, prev_text: str = "",
                prev_complete: bool = True,
                saved_outputs: list[tuple[int, dict[str, Any], str, bool]] | None = None,
                outputs_truncated: bool = False) -> None:
        control.on_state = lambda state, run=run: self._on_job_state(run, state)
        token = _CHAIN.set(chain)
        try:
            # create_task copies the CURRENT context, so the run (and any
            # workflow its steps start) carries the chain with it.
            run["task"] = asyncio.create_task(
                self._execute(run, definition, control, start, prev_text, prev_complete,
                              saved_outputs, outputs_truncated))
        finally:
            _CHAIN.reset(token)
        control.task = run["task"]
        run["task"].add_done_callback(
            lambda task, run=run, control=control: self._reconcile(run, control, task))

    def _reconcile(self, run: dict[str, Any], control: Any, task: Any) -> None:
        """A task cancelled before its first step never runs _execute's body,
        so nothing would release its admission slot or close its record."""
        from .jobs import CONTROLS
        from .job_store import State
        if control.id not in CONTROLS:
            return
        control.finish(State.CANCELLED, ok=False, result="cancelled before it started")
        if run.get("status") in self._LIVE_STATUSES:
            run["status"] = "cancelled"
            run["finished_at"] = utc_stamp()
            self._sync_cognition(run["workflow"])

    def _on_job_state(self, run: dict[str, Any], state: Any) -> None:
        """Mirror queued / paused / waiting into the run the deck displays."""
        value = getattr(state, "value", str(state))
        if value in {"queued", "running", "paused", "waiting_for_approval"} \
                and run.get("status") in {"queued", "running", "paused",
                                          "waiting for approval"}:
            run["status"] = value.replace("_", " ")
            self.bus.dashboard_event.emit("workflow", {
                "run": run["id"], "workflow": run["workflow"], "step": run["step"],
                "total": run["total"], "status": run["status"]})
            self._sync_cognition(run["workflow"])

    async def _execute(self, run: dict[str, Any], definition: dict[str, Any],
                       control: Any = None, start: int = 1, prev_text: str = "",
                       prev_complete: bool = True,
                       saved_outputs: list[tuple[int, dict[str, Any], str, bool]] | None = None,
                       outputs_truncated: bool = False) -> None:
        from .execution_policy import background_scope
        with background_scope("workflow", run_id=str(run.get("id") or "")):
            await self._execute_background(run, definition, control, start, prev_text,
                                           prev_complete, saved_outputs, outputs_truncated)

    async def _execute_background(self, run: dict[str, Any], definition: dict[str, Any],
                                  control: Any = None, start: int = 1, prev_text: str = "",
                                  prev_complete: bool = True,
                                  saved_outputs: list[tuple[int, dict[str, Any], str, bool]] | None = None,
                                  outputs_truncated: bool = False) -> None:
        from .jobs import _CURRENT_CONTROL, JobCancelled
        from .job_store import State, READ_ONLY, side_effect_class
        status = "complete"
        # What each step returned (index, step, text, ok), for the result kept
        # on the run. A resumed run starts from the output its checkpoint
        # carried, when that was kept whole.
        outputs: list[tuple[int, dict[str, Any], str, bool]] = list(saved_outputs or [])
        if (saved_outputs is None and 1 < start <= len(definition["steps"]) + 1
                and prev_text and prev_complete is True):
            outputs.append((start - 1, definition["steps"][start - 2], prev_text, True))
        token = _CURRENT_CONTROL.set(control)
        try:
            if control is not None:
                await control.admit()
                run["status"] = "running"
            for index, step in enumerate(definition["steps"], 1):
                if index < start:
                    continue
                if control is not None:
                    # Between steps is where a pause or a cancel lands.
                    await control.checkpoint()
                    run["status"] = "running"
                run["step"] = index
                self.bus.dashboard_event.emit("workflow", {
                    "run": run["id"], "workflow": run["workflow"],
                    "step": index, "total": run["total"], "status": "running"})
                args = self._resolve_args(step.get("args") or {},
                                          run["input"], prev_text, prev_complete=prev_complete)
                if control is not None and run.get("approval_gate") == index:
                    # Admitted and about to dispatch: the restart approval for
                    # this exact step is used up now, or the run stops here.
                    run["approval_gate"] = 0
                    approved, why = control.store.consume_approval(
                        control.id, index, step["tool"], args)
                    if not approved:
                        control.step(index, state="skipped", detail=f"not re-run: {why}")
                        status = f"cancelled: step {index} ({step['tool']}) not re-run - {why}"
                        break
                if control is not None:
                    control.step(index, state="running")
                    control.store.update(
                        control.id, step=index, fraction=(index - 1) / max(1, run["total"]),
                        checkpoint=self._checkpoint(definition, outputs, index, prev_text,
                                                    prev_complete, outputs_truncated))
                result: ToolResult | None = None
                attempts = int(step.get("retries", 1)) + 1
                tried = 0
                for attempt in range(1, attempts + 1):
                    tried = attempt
                    result = await self.dispatcher.dispatch(step["tool"], args)
                    if result.ok:
                        break
                    if side_effect_class(step["tool"], args) != READ_ONLY:
                        result = replace(result, text=result.text +
                                         " (not retried: it may already have taken effect)")
                        break
                    if attempt < attempts:
                        self.bus.log.emit(
                            f"WORKFLOW: step {index} ({step['tool']}) failed, "
                            f"retry {attempt}/{attempts - 1}")
                        await asyncio.sleep(min(2.0 * attempt, 6.0))
                prev_text = result.text if result is not None else ""
                prev_complete = True
                step_ok = bool(result and result.ok)
                outputs.append((index, step, prev_text, step_ok))
                run["transcript"].append({
                    "step": index, "tool": step["tool"],
                    "ok": step_ok,
                    "text": prev_text[:400],
                })
                if control is not None:
                    control.step(index, state="completed" if step_ok else "failed",
                                 attempts=tried, detail=first_line(prev_text, 200))
                    control.store.update(control.id,
                                         checkpoint=self._checkpoint(
                                             definition, outputs, index + 1, prev_text,
                                             prev_complete, outputs_truncated))
                if not step_ok and step.get("optional") is not True:
                    status = f"failed at step {index} ({step['tool']})"
                    break
                if not step_ok:
                    self.bus.log.emit(f"WORKFLOW: step {index} ({step['tool']}) failed; it is "
                                      "optional, so the run carries on.")
        except asyncio.CancelledError:
            status = "cancelled"
        except JobCancelled:
            status = "cancelled"
        except Busy as busy:            # ORION is shutting down before it started
            status = f"cancelled: {first_line(busy.reason, 100)}"
        except Exception as exc:
            status = f"fault: {first_line(exc, 120)}"
        finally:
            _CURRENT_CONTROL.reset(token)
        try:
            report, shortened = self._report(definition, outputs)
        except Exception:               # the run's own record must still be closed below
            report, shortened = "", False
        run["result"] = report
        run["result_truncated"] = shortened or outputs_truncated
        if control is not None:
            final = (State.COMPLETED if status == "complete" else
                     State.CANCELLED if status.startswith("cancelled") else State.FAILED)
            try:
                control.finish(final, ok=status == "complete",
                               result=self._job_result(status, report),
                               fraction=1.0 if status == "complete" else None)
            except Exception as exc:
                # finish() retired the slot and the live control regardless.
                # Raising here skipped everything below, so the deck showed the
                # run "running" for ever; end it there and say the durable
                # record did not take the outcome.
                self.bus.log.emit(f"WORKFLOW: '{run['workflow']}' outcome could not be "
                                  f"recorded - {first_line(exc, 160)}")
        run["status"] = status
        run["finished_at"] = utc_stamp()
        self.bus.dashboard_event.emit("workflow", {
            "run": run["id"], "workflow": run["workflow"],
            "step": run["step"], "total": run["total"], "status": status})
        found = f" Found: {first_line(report, 160)}" if report and status == "complete" else ""
        self.bus.log.emit(f"WORKFLOW: '{run['workflow']}' {status}.{found}")
        if status == "complete":
            self.bus.banner.emit(f"WORKFLOW COMPLETE: {run['workflow']}"
                                 + (" - ask what it found" if report else ""), 3)
        if self.telemetry is not None:
            self.telemetry.metrics.incr(
                "workflow.complete" if status == "complete" else "workflow.failed")
        self._sync_cognition(run["workflow"])

    _LIVE_STATUSES = frozenset({"running", "queued", "paused", "waiting for approval"})

    def cancel(self, run_id: str = "") -> ToolResult:
        candidates = [r for r in self.runs.values()
                      if r.get("status") in self._LIVE_STATUSES
                      and (not run_id or r["id"] == run_id or r.get("job_id") == run_id)]
        if not candidates:
            return ToolResult("No running workflow to cancel.")
        from .jobs import CONTROLS
        for run in candidates:
            control = CONTROLS.get(run.get("job_id") or "")
            if control is not None:
                control.cancel()
            task = run.get("task")
            if task is not None and not task.done():
                task.cancel()
        return ToolResult(f"Cancelling {len(candidates)} workflow run(s).")

    def resume(self, run_or_job_id: str) -> Any:
        """Resume a paused run, or one a restart INTERRUPTED (see resume_job)."""
        from .jobs import CONTROLS
        key = str(run_or_job_id or "").strip()
        run = self.runs.get(key)
        job_id = run.get("job_id") if run else key
        control = CONTROLS.get(job_id or "")
        if control is not None:
            if control.resume():
                return ToolResult(f"Resuming workflow run {run['id'] if run else job_id}.")
            return ToolResult(f"{job_id} is not paused.", ok=False)
        record = self.store.get(job_id) if job_id else None
        if record is None:
            recoverable = [r for r in self.store.list(states=["interrupted", "waiting_for_approval"],
                                                      kinds=["workflow"], limit=20)
                           if r["id"] not in CONTROLS and (not key or r["name"] == self._slug(key))]
            record = recoverable[0] if recoverable else None
        if record is None:
            return ToolResult(f"No paused or interrupted workflow run matches '{key}'.",
                              ok=False)
        return self.resume_job(record)

    def resume_job(self, record: dict[str, Any]) -> ToolResult:
        """Continue an INTERRUPTED run from the step that was in flight.

        A read-only step runs again (it is idempotent). A step that changes
        something may already have happened, so it runs again only after an
        approval bound to that exact step and its resolved arguments; declining
        stops the run there rather than carrying on without it.
        """
        if not self._library_is_current():
            return self._unavailable()
        from .jobs import CONTROLS, JobControl
        from .job_store import READ_ONLY, State, side_effect_class
        job_id = record["id"]
        if record.get("state") not in {State.INTERRUPTED.value, State.WAITING_FOR_APPROVAL.value}:
            return ToolResult(f"{job_id} is {record.get('state')}; nothing to resume.",
                              ok=False)
        if job_id in CONTROLS:
            return ToolResult(f"{job_id} is already running.", ok=False)
        if self.dispatcher is None:
            return ToolResult("The dispatcher is not attached; workflows "
                              "cannot execute yet.", ok=False)
        spec = record.get("args") or {}
        name = str(spec.get("workflow") or record.get("name") or "")
        definition = copy.deepcopy(self.definitions.get(name))
        recorded = self.store.steps(job_id)
        if (definition is None or not record.get("args_complete", True)
                or [s.get("tool") for s in definition["steps"]] != [s["tool"] for s in recorded]
                or (spec.get("definition_digest") is not None
                    and spec["definition_digest"] != self._definition_digest(definition))
                or (spec.get("definition_digest") is None and any(
                    _store_mod.args_digest(s["tool"], s.get("args"))
                    != _store_mod.args_digest(old["tool"], old.get("args"))
                    for s, old in zip(definition["steps"], recorded)))):
            return ToolResult(f"Workflow '{name}' has changed or been removed since "
                              f"{job_id} stopped, so it cannot be continued safely; "
                              "start it again.", ok=False)
        checkpoint = record.get("checkpoint", {})
        if not isinstance(checkpoint, dict) or checkpoint.get("_truncated"):
            return ToolResult("Workflow checkpoint is incomplete or invalid; review the "
                              "recorded steps before recovery. No work was resumed.", ok=False)
        prev_text = str(checkpoint.get("prev") or "")
        # Older checkpoints have no marker. Exactly the old truncation bound
        # cannot prove completeness, so only consumers of {prev} must refuse it.
        prev_complete = checkpoint.get("prev_complete", len(prev_text) < _MAX_PREVIOUS_CHARS)
        input_text = str(spec.get("input") or "")
        in_flight = next((s for s in recorded if s["state"] in {"interrupted", "running"}),
                         None)
        start = in_flight["idx"] if in_flight else checkpoint.get("next", 1)
        # Durable completion is evidence of an effect, even if a crash prevented
        # its next-step checkpoint from being written. Never default back to it
        # or mark later unexecuted steps complete from inconsistent metadata.
        def settled(index: int, old: dict[str, Any]) -> bool:
            return old["state"] == "completed" or (
                old["state"] == "failed" and definition["steps"][index - 1].get("optional") is True)

        expected = next((i for i, old in enumerate(recorded, 1)
                         if not settled(i, old)), len(recorded) + 1)
        if (type(start) is not int or start != expected
                or [old["idx"] for old in recorded] != list(range(1, len(recorded) + 1))
                or any(old["state"] not in {"completed", "pending", "running", "interrupted"}
                       and not settled(i, old) for i, old in enumerate(recorded, 1))
                or any(settled(i, old) for i, old in enumerate(recorded[start - 1:], start))):
            return ToolResult("Workflow checkpoint conflicts with the recorded step progress; "
                              "review it before recovery. No work was resumed.", ok=False)
        try:
            saved_outputs, outputs_truncated = self._restore_outputs(
                definition, recorded, checkpoint, start)
        except ValueError:
            return ToolResult("Workflow checkpoint contains invalid saved findings; review it "
                              "before recovery. No work was resumed.", ok=False)
        if start <= len(definition["steps"]):
            try:
                resumed_args = self._resolve_args(definition["steps"][start - 1].get("args") or {},
                                                 input_text, prev_text, prev_complete=prev_complete)
            except ValueError as exc:
                return ToolResult(f"Workflow cannot resume: {first_line(exc, 200)}", ok=False)
        # The recorded class was calculated from the definition's template.
        # Expanding {input}/{prev} may select a writing action. Keep either
        # source of uncertainty conservative, including older saved jobs.
        gated_step = 0
        if in_flight is not None and (
                in_flight.get("side_effect") != READ_ONLY
                or side_effect_class(definition["steps"][start - 1]["tool"], resumed_args) != READ_ONLY):
            step = definition["steps"][start - 1]
            args = resumed_args
            # Only checked here: the run uses it up at that step's dispatch,
            # after admission, so a queued or cancelled resume spends nothing.
            approved, _why = self.store.peek_approval(job_id, start, step["tool"], args)
            if approved:
                gated_step = start
            if not approved:
                latest = self.store.latest_approval(job_id, start)
                if latest is not None and latest["state"] == "denied":
                    self.store.transition(job_id, State.CANCELLED,
                                          note=f"not continued: step {start} "
                                               f"({step['tool']}) was not re-run")
                    return ToolResult(f"Understood — {job_id} stops at step {start} "
                                      f"({step['tool']}) and will not run it again.")
                if latest is None or latest["state"] != "pending":
                    latest = self.store.request_approval(
                        job_id, start, step["tool"], args,
                        reason=f"step {start} ({step['tool']}) was running when ORION "
                               "stopped and may already have taken effect")
                return ToolResult(
                    f"Workflow '{name}' ({job_id}) stopped during step {start} "
                    f"({step['tool']}), which may already have happened. It will not run "
                    f"again without your approval (approval {latest['id']}); approve or "
                    "decline it, then resume.", ok=False)
        if start > len(definition["steps"]):
            # INTERRUPTED/WAITING cannot transition straight to COMPLETED.
            # Reconcile only after every durable step is proven complete;
            # no dispatcher call or admission ticket is needed for this.
            report, _shortened = self._report(definition, saved_outputs or [])
            if (self.store.transition(job_id, State.RUNNING,
                                      note="reconciling recorded completed steps") is None
                    or self.store.transition(job_id, State.COMPLETED, ok=True,
                                             result=report or "complete (every step had finished)",
                                             fraction=1.0) is None):
                return ToolResult(f"{job_id}'s completed progress could not be reconciled; "
                                  "no work was resumed.", ok=False)
            return ToolResult(f"{job_id} had already finished every step.")
        try:
            ticket = self._admission_obj().submit(WorkClass.USER, label=f"workflow {name}")
        except Busy as busy:
            return ToolResult(f"Workflow '{name}' not resumed — {busy.reason}", ok=False)
        control = JobControl(self.store, record, cls=WorkClass.USER,
                             label=f"workflow {name}", admission=self._admission)
        control.ticket = ticket
        CONTROLS[control.id] = control
        try:
            control.set_state(State.QUEUED, note="resuming after a restart")
        except Exception as exc:
            control.release()
            CONTROLS.pop(control.id, None)
            self.bus.log.emit(f"WORKFLOW: could not record the resume of '{name}' - "
                              f"{first_line(exc)}")
            return ToolResult(f"Workflow '{name}' ({job_id}) could not be recorded as resuming, "
                              "so nothing was run. Check the log and try again.", ok=False)
        run_id = f"{name}_{uuid.uuid4().hex[:8]}"
        run = {"id": run_id, "workflow": name, "status": "queued", "step": start - 1,
               "total": len(definition["steps"]), "input": input_text, "transcript": [],
               "started_at": utc_stamp(), "job_id": job_id, "approval_gate": gated_step}
        self.runs[run_id] = run
        self._launch(run, definition, control, (name,), start, prev_text, prev_complete,
                     saved_outputs, outputs_truncated)
        self._sync_cognition(name)
        return ToolResult(f"Resuming workflow '{name}' ({job_id}) from step {start}.")

    def status(self) -> ToolResult:
        from .jobs import CONTROLS
        recoverable = [record for record in self.store.list(
            states=["interrupted", "waiting_for_approval"], kinds=["workflow"], limit=8)
            if record["id"] not in CONTROLS]
        if not self.runs and not recoverable:
            last = self._last_record()
            if last is None:
                return ToolResult("No workflow runs this session or interrupted runs to resume.")
            return ToolResult(f"No workflow runs this session. The last one before ORION restarted "
                              f"was '{last['name']}' ({last['state']}); ask for the workflow "
                              "result to hear what it found.")
        lines = ["Workflow runs:"] if self.runs else []
        for run in list(self.runs.values())[-8:]:
            lines.append(f"- {run['workflow']} [{run['status']}] "
                         f"step {run['step']}/{run['total']}"
                         + (" (scheduled)" if run.get("scheduled") else ""))
            for entry in run["transcript"][-2:]:
                mark = "ok" if entry["ok"] else "FAILED"
                lines.append(f"    {entry['step']}. {entry['tool']} ({mark})")
            if run.get("result") and run.get("status") not in self._LIVE_STATUSES:
                lines.append(f"    found: {first_line(run['result'], 160)}")
        if recoverable:
            lines.append("Saved workflow runs interrupted or awaiting approval:")
            for record in recoverable:
                lines.append(f"- {record['name']} [{record['state']}] ({record['id']}) "
                             f"step {record['step']}/{record['total']}")
            lines.append("Use resume with the saved job ID to continue a run.")
        if any(run.get("result") for run in self.runs.values()):
            lines.append("Ask for the workflow result (with a name for one workflow) "
                         "for everything a run found.")
        return ToolResult("\n".join(lines))

    def result(self, ref: str = "") -> ToolResult:
        """What the latest finished run found - of the workflow or run *ref*
        names, if any: this session's run, else (after a restart) the run's
        job record, which keeps the same text."""
        ref = str(ref or "").strip()
        key = self._slug(ref)
        finished = [run for run in self.runs.values()
                    if run.get("status") not in self._LIVE_STATUSES
                    and (not key or key == run["workflow"] or ref == run["id"]
                         or ref == run.get("job_id"))]
        if finished:
            run = max(finished, key=lambda r: str(r.get("finished_at") or ""))
            shortened = (" (each part shortened to fit, marked [...])"
                         if run.get("result_truncated") else "")
            return ToolResult(f"Workflow '{run['workflow']}' {run['status']} at "
                              f"{run.get('finished_at') or 'an unknown time'}{shortened}:\n"
                              f"{run.get('result') or 'It returned no output.'}")
        record = self._last_record(ref)
        if record is not None:
            return ToolResult(f"Workflow '{record['name']}' {record['state']} ({record['id']}, "
                              "before ORION last restarted):\n"
                              f"{record.get('result') or 'It returned no output.'}")
        live = [run for run in self.runs.values()
                if run.get("status") in self._LIVE_STATUSES and (not key or key == run["workflow"])]
        if live:
            run = live[-1]
            return ToolResult(f"Workflow '{run['workflow']}' is still {run['status']} (step "
                              f"{run['step']} of {run['total']}); what it finds is kept when it ends.")
        return ToolResult("No workflow run has finished yet"
                          + (f" for '{ref}'." if ref else "."))

    def _last_record(self, key: str = "") -> dict[str, Any] | None:
        """The newest finished workflow job record (of workflow *key*), or None."""
        try:
            if key:
                exact = self.store.get(key)
                if (exact is not None and exact.get("kind") == "workflow"
                        and exact.get("state") in {"completed", "failed", "cancelled"}):
                    return exact
            name = self._slug(key)
            records = self.store.list(states=["completed", "failed", "cancelled"],
                                      kinds=["workflow"],
                                      limit=_store_mod.RETAINED_FINISHED if name else 1)
        except Exception:
            return None
        return next((r for r in records if not name or r.get("name") == name), None)

    # ── what a run found ──────────────────────────────────────────────────────

    @staticmethod
    def _retained_outputs(definition: dict[str, Any],
                          outputs: list[tuple[int, dict[str, Any], str, bool]]
                          ) -> list[tuple[int, dict[str, Any], str, bool]]:
        steps = definition.get("steps") or []
        successful = {index for index, _step, _text, ok in outputs if ok}
        return [(index, step, str(text or "").strip(), ok)
                for index, step, text, ok in outputs
                if str(text or "").strip() and (not ok or index + 1 not in successful
                    or index >= len(steps)
                    or "{prev}" not in json.dumps(steps[index].get("args") or {}))]

    @classmethod
    def _checkpoint(cls, definition: dict[str, Any],
                    outputs: list[tuple[int, dict[str, Any], str, bool]],
                    next_step: int, prev_text: str, prev_complete: bool,
                    already_truncated: bool = False) -> dict[str, Any]:
        """Keep findings as well as continuation input, within the JSON limit.

        The retained texts are display data only; {prev} always uses its separate
        completeness marker. Escaped Unicode counts towards the store's budget.
        """
        base = {"next": next_step, "prev": prev_text[:_MAX_PREVIOUS_CHARS],
                "prev_complete": prev_complete is True and len(prev_text) <= _MAX_PREVIOUS_CHARS}
        retained = cls._retained_outputs(definition, outputs)

        def pack(budget: int) -> dict[str, Any]:
            texts, shortened = cls._fit_parts([item[2] for item in retained], budget)
            return dict(base, outputs=[{"index": item[0], "text": text, "ok": item[3]}
                                      for item, text in zip(retained, texts)],
                        outputs_truncated=already_truncated or shortened)

        checkpoint = pack(_MAX_RESULT_CHARS)
        if len(json.dumps(checkpoint)) <= _store_mod.MAX_ARGS_CHARS:
            return checkpoint
        low, high = len(retained) * len(_CUT_MARK), _MAX_RESULT_CHARS
        while low < high:
            middle = (low + high + 1) // 2
            if len(json.dumps(pack(middle))) <= _store_mod.MAX_ARGS_CHARS:
                low = middle
            else:
                high = middle - 1
        return pack(low)

    @staticmethod
    def _restore_outputs(definition: dict[str, Any], recorded: list[dict[str, Any]],
                         checkpoint: dict[str, Any], start: int
                         ) -> tuple[list[tuple[int, dict[str, Any], str, bool]] | None, bool]:
        if "outputs" not in checkpoint:          # older job: only its last {prev} survived
            return None, False
        raw = checkpoint["outputs"]
        shortened = checkpoint.get("outputs_truncated", False)
        if not isinstance(raw, list) or len(raw) > _MAX_STEPS or type(shortened) is not bool:
            raise ValueError("invalid saved findings")
        outputs = []
        seen = set()
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("invalid saved finding")
            index, text, ok = item.get("index"), item.get("text"), item.get("ok")
            if (type(index) is not int or not 1 <= index < start or index in seen
                    or not isinstance(text, str) or type(ok) is not bool
                    or (recorded[index - 1]["state"] == "completed") is not ok):
                raise ValueError("saved finding conflicts with completed progress")
            seen.add(index)
            outputs.append((index, definition["steps"][index - 1], text, ok))
        return outputs, shortened

    @classmethod
    def _report(cls, definition: dict[str, Any],
                outputs: list[tuple[int, dict[str, Any], str, bool]]) -> tuple[str, bool]:
        """What a run found, and whether it had to be shortened.

        The output of every step that the next step did not read through
        {prev}: in a chain that is the final step's, for independent steps (a
        morning check) it is each of them, labelled with its tool. A failed
        step's message is kept too. Bounded to the job record's 4,000
        characters with the room shared fairly; a cut is marked, never silent.
        """
        kept = [(step, text, ok) for _index, step, text, ok in cls._retained_outputs(definition, outputs)]
        if not kept:
            return "", False
        if len(kept) == 1:
            return cls._fit([kept[0][1]], _MAX_RESULT_CHARS)
        return cls._fit([f"{cls._step_label(step)}{'' if ok else ' (failed)'}: {text}"
                         for step, text, ok in kept], _MAX_RESULT_CHARS)

    @staticmethod
    def _step_label(step: dict[str, Any]) -> str:
        action = (step.get("args") or {}).get("action")
        tool = str(step.get("tool") or "step")
        return f"{tool} {action}" if isinstance(action, str) and action and "{" not in action else tool

    @staticmethod
    def _fit(parts: list[str], budget: int) -> tuple[str, bool]:
        fitted, shortened = WorkflowEngine._fit_parts(parts, budget)
        return "\n\n".join(fitted), shortened

    @staticmethod
    def _fit_parts(parts: list[str], budget: int) -> tuple[list[str], bool]:
        """*parts* joined within *budget*: short parts whole, the rest sharing
        what is left equally, each cut part marked."""
        separator = "\n\n"
        room = budget - len(separator) * (len(parts) - 1)
        if sum(len(part) for part in parts) <= room:
            return parts, False
        limits: dict[int, int] = {}
        pending = sorted(range(len(parts)), key=lambda i: len(parts[i]))
        left = room
        while pending:
            share = left // len(pending)
            if len(parts[pending[0]]) <= share:
                shortest = pending.pop(0)
                limits[shortest] = len(parts[shortest])
                left -= limits[shortest]
                continue
            for i in pending:
                limits[i] = share
            break
        cut = [part if len(part) <= limits[i]
               else part[:max(0, limits[i] - len(_CUT_MARK))].rstrip() + _CUT_MARK
               for i, part in enumerate(parts)]
        return cut, True

    @staticmethod
    def _job_result(status: str, report: str) -> str:
        """The job record's result: what was found (after the outcome, unless
        the run simply completed), within the record's limit, a cut marked."""
        if not report:
            return status
        text = report if status == "complete" else f"{status}: {report}"
        if len(text) > _MAX_RESULT_CHARS:
            text = text[:_MAX_RESULT_CHARS - len(_CUT_MARK)].rstrip() + _CUT_MARK
        return text

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sync_cognition(self, name: str) -> None:
        """Preserve unfinished work across admitted runs with the same name.

        Same-name runs have separate job identities. A completed sibling must
        not erase another run's work. Cognition reserves ``status`` for whether
        a plan is open, so runtime state also has its own ``run_status`` field.
        """
        if self.cognition is None:
            return
        unfinished = [run for run in self.runs.values()
                      if run["workflow"] == name and run.get("status") != "complete"]
        try:
            if not unfinished:
                self.cognition.close_workflow(name)
                return
            representative = next((run for run in reversed(unfinished)
                                   if run.get("status") in self._LIVE_STATUSES),
                                  unfinished[-1])
            self.cognition.upsert_workflow(name, {
                "run": representative["id"], "job_id": representative.get("job_id", ""),
                "status": representative["status"], "run_status": representative["status"],
            })
        except Exception:
            pass

    @staticmethod
    def _definition_digest(definition: dict[str, Any]) -> str:
        # Bind restart recovery to arguments AND retry policy, including steps
        # that had not yet started when the process stopped.
        return _store_mod.args_digest("workflow", {"steps": definition["steps"]})

    @staticmethod
    def _resolve_args(args: dict[str, Any], input_text: str,
                      prev_text: str, *, prev_complete: bool = True) -> dict[str, Any]:
        replacements = {"input": input_text, "prev": prev_text}

        def substitute(match: Any) -> str:
            key = match[1]
            if key == "prev" and (prev_complete is not True or len(prev_text) > _MAX_PREVIOUS_CHARS):
                raise ValueError("Previous workflow output is incomplete or exceeds 2,000 characters; "
                                 "no shortened argument was used. Save the full output and pass its path.")
            return replacements[key]

        def resolve(value: Any) -> Any:
            if isinstance(value, str):
                # Substitute template tokens once: user text containing a
                # token is data, not a second template to expand.
                return re.sub(r"\{(input|prev)\}", substitute, value)
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            return copy.deepcopy(value)

        resolved = resolve(args)
        _check_argument_budget(resolved, "Expanded workflow arguments")
        return resolved

    @staticmethod
    def _slug(name: str) -> str:
        return "".join(c for c in str(name or "").lower().replace(" ", "_")
                       if c.isalnum() or c == "_").strip("_")[:60]


# ── schedules: a named workflow at a time of day ──────────────────────────────

#: Kept beside the workflow library (config/workflow_schedules.json).
SCHEDULES_FILENAME = "workflow_schedules.json"
_MAX_SCHEDULES = 32
_REPEATS = ("once", "daily", "weekdays", "weekly")
_DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_DAY_WORDS: dict[str, int] = {name: i for i, name in enumerate(_DAY_NAMES)}
_DAY_WORDS.update({"mon": 0, "tue": 1, "tues": 1, "wed": 2, "weds": 2, "thu": 3, "thur": 3,
                   "thurs": 3, "fri": 4, "sat": 5, "sun": 6})
_DAY_RE = re.compile(r"\b(" + "|".join(sorted(_DAY_WORDS, key=len, reverse=True)) + r")s?\b")
_PLURAL_DAY_RE = re.compile(r"\b(?:" + "|".join(_DAY_NAMES) + r")s\b")
_WEEKDAYS_RE = re.compile(r"\b(?:weekdays?|working days?|work ?days?|monday to friday|mon-fri)\b")
_DAILY_RE = re.compile(r"\b(?:daily|every ?day|each day|every (?:morning|afternoon|evening|night))\b")
_RELATIVE_RE = re.compile(r"\bin\s+(\d{1,4})\s*(minutes?|mins?|hours?|hrs?)\b")
_REPEAT_ALIASES = {"day": "daily", "every day": "daily", "everyday": "daily",
                   "weekday": "weekdays", "week": "weekly", "every week": "weekly",
                   "one-off": "once", "one off": "once", "one time": "once"}


class ScheduleError(ValueError):
    """A schedule that cannot be understood, with a sentence to show."""


def _parse_clock(text: Any) -> tuple[int, int] | None:
    """(hour, minute) from '7:30', '07.30', '7am', '7:30 pm', 'noon', 'at 9' or '9'."""
    text = str(text or "").lower()
    if re.search(r"\b(?:noon|midday)\b", text):
        return 12, 0
    if re.search(r"\bmidnight\b", text):
        return 0, 0
    half = None
    found = re.search(r"(?<!\d)(\d{1,2})[:.](\d{2})(?!\d)\s*(a\.?m\.?|p\.?m\.?)?", text)
    if found:
        hour, minute, half = int(found.group(1)), int(found.group(2)), found.group(3)
    else:
        found = re.search(r"(?<!\d)(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", text)
        if found:
            hour, minute, half = int(found.group(1)), 0, found.group(2)
        else:
            found = (re.search(r"\bat\s+(\d{1,2})(?![\d:.])", text)
                     or re.fullmatch(r"\s*(\d{1,2})\s*", text))
            if found is None:
                return None
            hour, minute = int(found.group(1)), 0
    if half:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if half.startswith("p") else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _days_named(text: Any) -> set[int]:
    return {_DAY_WORDS[found.group(1)] for found in _DAY_RE.finditer(str(text or "").lower())}


def _local_now() -> datetime:
    from .time_service import TIME
    return TIME.now()


def _moment(day: date, hour: int, minute: int, tz: Any) -> datetime:
    return datetime.combine(day, clock_time(hour, minute), tzinfo=tz)


def parse_when(text: Any = "", now: datetime | None = None, *, repeat: Any = "", at: Any = "",
               day: Any = "", on_date: Any = "") -> dict[str, Any]:
    """A schedule from words and/or fields, in ORION's local time.

    'every weekday at 7:30', 'daily at 18:00', 'every monday at 9', 'mondays
    at 6pm', 'tomorrow at 9am', 'today at 17:45', 'friday at noon' (once, the
    coming Friday), 'in 90 minutes', '2026-10-20 09:00', or just 'at 6pm'
    (once, at the next 6pm). Fields win over words: repeat (once, daily,
    weekdays, weekly), at ('07:30'), day ('monday'), on_date ('2026-10-20').
    Returns {'repeat', 'time', 'weekday', 'date'}; raises ScheduleError.
    """
    now = now or _local_now()
    words = " ".join(str(text or "").lower().replace(",", " ").split())
    chosen = str(repeat or "").strip().lower()
    chosen = _REPEAT_ALIASES.get(chosen, chosen)
    if chosen and chosen not in _REPEATS:
        raise ScheduleError("Repeat it once, daily, on weekdays or weekly.")
    relative = _RELATIVE_RE.search(words)
    if relative and not str(at or "").strip() and chosen in {"", "once"}:
        amount = int(relative.group(1))
        delta = (timedelta(hours=amount) if relative.group(2).startswith("h")
                 else timedelta(minutes=amount))
        if not timedelta(minutes=1) <= delta <= timedelta(days=7):
            raise ScheduleError("Give a delay between a minute and a week, or a time of day.")
        moment = (now + delta).replace(second=0, microsecond=0)
        return {"repeat": "once", "time": f"{moment:%H:%M}", "weekday": None,
                "date": moment.date().isoformat()}
    clock = _parse_clock(at) if str(at or "").strip() else _parse_clock(words)
    if clock is None:
        raise ScheduleError("When should it run? Give a time: 'every weekday at 7:30', "
                            "'daily at 18:00', 'every monday at 9' or 'tomorrow at 9am'.")
    hour, minute = clock
    days = _days_named(day) if str(day or "").strip() else _days_named(words)
    if len(days) > 1:
        raise ScheduleError("One day per schedule: add one for each day, e.g. 'every monday "
                            "at 9' and then 'every thursday at 9'.")
    weekday = next(iter(days)) if days else None
    iso = str(on_date or "").strip()[:10]
    if not iso:
        found = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", words)
        iso = found.group(1) if found else ""
    if not chosen:
        if iso:
            chosen = "once"
        elif _WEEKDAYS_RE.search(words):
            chosen = "weekdays"
        elif (weekday is not None and (re.search(r"\b(?:every|each|weekly)\b", words)
                                       or _PLURAL_DAY_RE.search(words))) \
                or re.search(r"\bweekly\b", words):
            chosen = "weekly"
        elif _DAILY_RE.search(words):
            chosen = "daily"
        else:
            chosen = "once"
    time_text = f"{hour:02d}:{minute:02d}"
    if chosen == "weekly":
        if weekday is None:
            raise ScheduleError("Weekly on which day? Say e.g. 'every monday at 9'.")
        return {"repeat": "weekly", "time": time_text, "weekday": weekday, "date": ""}
    if chosen in {"daily", "weekdays"}:
        return {"repeat": chosen, "time": time_text, "weekday": None, "date": ""}
    today = now.date()
    if iso:
        try:
            target = date.fromisoformat(iso)
        except ValueError:
            raise ScheduleError(f"'{iso}' is not a date I can read; write it as 2026-10-20.") from None
    elif re.search(r"\btomorrow\b", words):
        target = today + timedelta(days=1)
    elif weekday is not None:
        target = today + timedelta(days=(weekday - today.weekday()) % 7)
        if target == today and (_moment(target, hour, minute, now.tzinfo) <= now
                                or re.search(r"\bnext\b", words)):
            target += timedelta(days=7)
    else:
        target = today
        if (_moment(today, hour, minute, now.tzinfo) <= now
                and not re.search(r"\b(?:today|tonight)\b", words)):
            target = today + timedelta(days=1)
    if _moment(target, hour, minute, now.tzinfo) <= now:
        raise ScheduleError(f"{target:%A %d %B} at {time_text} has already passed.")
    return {"repeat": "once", "time": time_text, "weekday": None, "date": target.isoformat()}


def describe_schedule(entry: dict[str, Any]) -> str:
    """'every day at 07:30', 'on weekdays at 08:00', 'every Monday at 09:00', 'once, ...'."""
    at = entry.get("time", "?")
    repeat = entry.get("repeat")
    if repeat == "daily":
        return f"every day at {at}"
    if repeat == "weekdays":
        return f"on weekdays at {at}"
    if repeat == "weekly":
        return f"every {_DAY_NAMES[int(entry.get('weekday') or 0)].capitalize()} at {at}"
    try:
        return f"once, on {date.fromisoformat(str(entry.get('date'))):%a %d %b} at {at}"
    except ValueError:
        return f"once at {at}"


class WorkflowScheduler:
    """Runs named workflows at a time of day: once, daily, on weekdays or weekly.

    Kept in workflow_schedules.json beside the library. Each occurrence is
    claimed - written to disk - BEFORE its run starts, so neither a crash nor
    a restart can run it twice; the run itself is an ordinary durable workflow
    job, interrupted and resumable like any other, and it keeps what it found
    (WorkflowEngine.result). An occurrence missed while ORION was off is
    caught up once if it is less than MISSED_GRACE_S late, and skipped
    otherwise; only the latest missed occurrence ever runs, never a backlog.

    A due run waits for a quiet moment. Admission must have no reason to defer
    background work (memory pressure, a request being answered, a busy
    processor - that last stops counting after CPU_PATIENCE_S) and ORION must
    be neither speaking nor mid-turn. Nothing here speaks.
    """

    TICK_S = 30.0
    FIRST_TICK_S = 90.0
    MISSED_GRACE_S = 6 * 3600.0
    CPU_PATIENCE_S = 600.0

    def __init__(self, engine: WorkflowEngine, *, path: Path | None = None,
                 clock: Any = None, quiet: Any = None) -> None:
        self.engine = engine
        self.path = Path(path) if path is not None else engine.path.with_name(SCHEDULES_FILENAME)
        self._clock = clock
        self._quiet = quiet
        self._entries: list[dict[str, Any]] | None = None
        self._load_error = False
        self._waiting: dict[str, str] = {}
        self._stopped = False
        self._task: Any = None

    # ── the file ──────────────────────────────────────────────────────────────

    def _now(self) -> datetime:
        return self._clock() if self._clock is not None else _local_now()

    def _log(self, text: str) -> None:
        try:
            self.engine.bus.log.emit(text)
        except Exception:
            pass

    def _load(self) -> bool:
        """Read the schedules once. An unreadable file is kept as it is and
        nothing scheduled runs (or is saved over it) until it is repaired."""
        if self._entries is not None:
            return not self._load_error
        try:
            text = _read_library(self.path)
            data = json.loads(text, object_pairs_hook=_unique_object) if text is not None else {}
            entries = data.get("schedules", []) if isinstance(data, dict) else None
            if not isinstance(entries, list) or len(entries) > _MAX_SCHEDULES:
                raise ValueError("invalid schedule list")
            for entry in entries:
                self._check_entry(entry)
            self._entries, self._load_error = entries, False
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            self._entries, self._load_error = [], True
            self._log(f"WORKFLOW: {self.path.name} is unreadable; it is kept as it is and "
                      "nothing scheduled runs until it is repaired.")
        return not self._load_error

    @staticmethod
    def _check_entry(entry: Any) -> None:
        if not isinstance(entry, dict):
            raise ValueError("a schedule must be an object")
        for field_name in ("id", "workflow", "input", "time", "created_at", "last_due"):
            if not isinstance(entry.get(field_name), str):
                raise ValueError(f"schedule field {field_name} must be text")
        if entry.get("repeat") not in _REPEATS or not isinstance(entry.get("enabled"), bool):
            raise ValueError("invalid schedule repeat")
        hour, minute = (int(part) for part in entry["time"].split(":"))
        if not re.fullmatch(r"\d{2}:\d{2}", entry["time"]) or hour > 23 or minute > 59:
            raise ValueError("invalid schedule time")
        weekday = entry.get("weekday")
        if entry["repeat"] == "weekly" and not (type(weekday) is int and 0 <= weekday <= 6):
            raise ValueError("invalid schedule weekday")
        if entry["repeat"] == "once":
            date.fromisoformat(str(entry.get("date")))
        if not entry["created_at"]:
            raise ValueError("a schedule needs the time it was made")
        for stamp in (entry["created_at"], entry["last_due"]):
            if stamp and datetime.fromisoformat(stamp).tzinfo is None:
                raise ValueError("schedule times must carry their zone")

    def _save(self) -> bool:
        if self._load_error or self._entries is None:
            return False
        try:
            text = json.dumps({"version": 1, "schedules": self._entries}, indent=2, allow_nan=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path, text, encoding="utf-8")
        except Exception as exc:
            self._log(f"WORKFLOW: schedules could not be saved - {first_line(exc)}")
            return False
        return True

    def _unreadable(self) -> ToolResult:
        return ToolResult(f"{self.path.name} is unreadable and has been kept as it is; repair or "
                          "remove it, then reload ORION, before changing schedules.", ok=False)

    # ── the tool's actions ────────────────────────────────────────────────────

    def add(self, name: str, when: str = "", input_text: str = "", **fields: Any) -> ToolResult:
        """Schedule workflow *name*: *when* in words, *fields* as parse_when's."""
        if not self._load():
            return self._unreadable()
        key = self.engine._slug(name)
        if not key or key not in self.engine.definitions:
            return ToolResult(f"No workflow named '{name}'. Say 'list workflows' to see the "
                              "library; nothing was scheduled.", ok=False)
        input_text = str(input_text or "")
        if len(input_text) > 2000:
            return ToolResult("Workflow input exceeds 2,000 characters; nothing was scheduled.",
                              ok=False)
        now = self._now()
        try:
            spec = parse_when(when, now, **fields)
        except ScheduleError as exc:
            return ToolResult(f"{exc} Nothing was scheduled.", ok=False)
        for entry in self._entries:
            if (entry["enabled"] and entry["workflow"] == key and entry["input"] == input_text
                    and all(entry.get(k) == spec[k] for k in ("repeat", "time", "weekday", "date"))):
                return ToolResult(f"'{key}' is already scheduled {describe_schedule(entry)} "
                                  f"({entry['id']}).")
        entries = list(self._entries)
        if len(entries) >= _MAX_SCHEDULES:
            # Finished one-off schedules make room first; a live one never.
            entries = [e for e in entries if e["enabled"]]
        if len(entries) >= _MAX_SCHEDULES:
            return ToolResult(f"There are already {_MAX_SCHEDULES} schedules; remove one first.",
                              ok=False)
        entry = {"id": f"sch-{uuid.uuid4().hex[:8]}", "workflow": key, "input": input_text,
                 **spec, "enabled": True, "created_at": now.isoformat(timespec="seconds"),
                 "last_due": "", "last_started_at": "", "last_outcome": "", "last_job_id": ""}
        before, self._entries = self._entries, entries + [entry]
        if not self._save():
            self._entries = before
            return ToolResult("The schedule could not be saved, so nothing was scheduled. "
                              "Check the log.", ok=False)
        upcoming = self.next_due(entry, now)
        return ToolResult(f"Scheduled '{key}' {describe_schedule(entry)} ({entry['id']})"
                          + (f"; next run {upcoming:%a %d %b %H:%M}" if upcoming else "")
                          + ". It runs in the background at a quiet moment and keeps what it "
                          "finds - ask for the workflow result afterwards.")

    def remove(self, ref: str) -> ToolResult:
        """Remove the schedule *ref* names (its id), or every schedule of a workflow."""
        if not self._load():
            return self._unreadable()
        ref = str(ref or "").strip()
        if not ref:
            return ToolResult("Which schedule? Give its id from 'schedules', or the "
                              "workflow's name.", ok=False)
        key = self.engine._slug(ref)
        doomed = [e for e in self._entries if e["id"] == ref or e["workflow"] == key]
        if not doomed:
            return ToolResult(f"No schedule matches '{ref}'.", ok=False)
        before = self._entries
        self._entries = [e for e in before if e not in doomed]
        if not self._save():
            self._entries = before
            return ToolResult("The schedule could not be removed from disk; it is still "
                              "active. Check the log.", ok=False)
        return ToolResult("Removed " + "; ".join(
            f"'{e['workflow']}' {describe_schedule(e)} ({e['id']})" for e in doomed)
            + ". A run already started carries on.")

    def describe(self) -> ToolResult:
        if not self._load():
            return self._unreadable()
        if not self._entries:
            return ToolResult("Nothing is scheduled. Say e.g. 'run morning_check every "
                              "weekday at 7:30'.")
        now = self._now()
        lines = ["Scheduled workflows:"]
        for entry in self._entries:
            upcoming = self.next_due(entry, now) if entry["enabled"] else None
            bits = [f"- {entry['id']}: '{entry['workflow']}' {describe_schedule(entry)}"
                    + ("" if entry["enabled"] else " [finished]")]
            if upcoming is not None:
                bits.append(f"next {upcoming:%a %d %b %H:%M}")
            if entry.get("last_outcome"):
                bits.append(f"last: {first_line(entry['last_outcome'], 120)}")
            lines.append("; ".join(bits))
        return ToolResult("\n".join(lines))

    def summary(self) -> str:
        """One line for the workflow status, or '' when nothing is scheduled."""
        if not self._load() or not self._entries:
            return ""
        live = sum(1 for e in self._entries if e["enabled"])
        return f"{live} workflow schedule(s) active - ask for 'schedules' to list them."

    # ── when ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _falls_on(entry: dict[str, Any], day: date) -> bool:
        repeat = entry["repeat"]
        if repeat == "daily":
            return True
        if repeat == "weekdays":
            return day.weekday() < 5
        if repeat == "weekly":
            return day.weekday() == entry["weekday"]
        return day.isoformat() == entry.get("date")

    def _at(self, entry: dict[str, Any], day: date, tz: Any) -> datetime:
        hour, minute = (int(part) for part in entry["time"].split(":"))
        return _moment(day, hour, minute, tz)

    def latest_due(self, entry: dict[str, Any], now: datetime) -> datetime | None:
        """The latest occurrence at or before *now*, if any."""
        if entry["repeat"] == "once":
            moment = self._at(entry, date.fromisoformat(entry["date"]), now.tzinfo)
            return moment if moment <= now else None
        for back in range(0, 8):
            day = now.date() - timedelta(days=back)
            if self._falls_on(entry, day):
                moment = self._at(entry, day, now.tzinfo)
                if moment <= now:
                    return moment
        return None

    def next_due(self, entry: dict[str, Any], now: datetime) -> datetime | None:
        """The first occurrence after *now*, if any."""
        if entry["repeat"] == "once":
            moment = self._at(entry, date.fromisoformat(entry["date"]), now.tzinfo)
            return moment if moment > now else None
        for ahead in range(0, 8):
            day = now.date() + timedelta(days=ahead)
            if self._falls_on(entry, day):
                moment = self._at(entry, day, now.tzinfo)
                if moment > now:
                    return moment
        return None

    def _pending(self, entry: dict[str, Any], now: datetime) -> datetime | None:
        """The occurrence that is due and not yet claimed, if any."""
        latest = self.latest_due(entry, now)
        if latest is None:
            return None
        anchor = datetime.fromisoformat(entry["last_due"] or entry["created_at"])
        return latest if latest > anchor else None

    def _deferral(self, waited_s: float) -> str:
        """Why a due run should wait, or '' to start it now."""
        try:
            reason = self.engine._admission_obj().defer_reason(WorkClass.BACKGROUND)
        except Exception:
            reason = ""
        if reason.startswith("the processor") and waited_s >= self.CPU_PATIENCE_S:
            reason = ""
        if not reason and self._quiet is not None:
            try:
                if not self._quiet():
                    reason = "ORION is speaking or in a conversation"
            except Exception:
                pass                    # a broken probe must not stop schedules for good
        return reason

    # ── running ───────────────────────────────────────────────────────────────

    async def tick(self, now: datetime | None = None) -> list[str]:
        """Start whatever is due and may start now; the ids of those started."""
        if not self._load():
            return []
        now = now or self._now()
        started: list[str] = []
        for entry in list(self._entries):
            if not entry["enabled"]:
                continue
            due = self._pending(entry, now)
            if due is None:
                continue
            if entry["workflow"] not in self.engine.definitions:
                self._close(entry, due, f"not run on {due:%a %d %b %H:%M}: the workflow "
                                        "no longer exists, so this schedule was switched off",
                            switch_off=True)
                continue
            late = (now - due).total_seconds()
            if late > self.MISSED_GRACE_S:
                waited = self._waiting.pop(entry["id"], "")
                why = (f"it was still waiting ({waited.split('|', 1)[-1]})" if waited
                       else "ORION was not running then")
                self._close(entry, due, f"missed the {due:%a %d %b %H:%M} run - {why}")
                continue
            reason = self._deferral(late)
            if reason:
                marker = f"{due.isoformat()}|{reason}"
                if self._waiting.get(entry["id"]) != marker:
                    self._waiting[entry["id"]] = marker
                    self._log(f"WORKFLOW: scheduled '{entry['workflow']}' is due and waiting "
                              f"- {reason}.")
                continue
            if self._start(entry, due, now):
                started.append(entry["id"])
        return started

    def _close(self, entry: dict[str, Any], due: datetime, outcome: str,
               switch_off: bool = False) -> None:
        """Consume an occurrence without running it, and say why."""
        entry["last_due"] = due.isoformat()
        entry["last_outcome"] = outcome
        if switch_off or entry["repeat"] == "once":
            entry["enabled"] = False
        self._save()
        self._log(f"WORKFLOW: scheduled '{entry['workflow']}' {outcome}.")

    def _start(self, entry: dict[str, Any], due: datetime, now: datetime) -> bool:
        claimed = dict(entry)
        entry["last_due"] = due.isoformat()
        entry["last_started_at"] = now.isoformat(timespec="seconds")
        if entry["repeat"] == "once":
            entry["enabled"] = False
        if not self._save():
            entry.clear()
            entry.update(claimed)
            self._log(f"WORKFLOW: scheduled '{entry['workflow']}' not started - its claim could "
                      "not be saved, and without it a restart could run it twice.")
            return False
        self._waiting.pop(entry["id"], None)
        known = set(self.engine.runs)
        try:
            result = self.engine.start(entry["workflow"], entry["input"])
        except Exception as exc:          # claimed already: reported, never retried
            result = ToolResult(f"could not start - {first_line(exc, 160)}", ok=False)
        run = next((r for run_id, r in self.engine.runs.items() if run_id not in known), None)
        if run is not None:
            run["scheduled"] = entry["id"]
        entry["last_outcome"] = first_line(result.text, 200)
        entry["last_job_id"] = str(run.get("job_id") or "") if run is not None else ""
        self._save()
        self._log(f"WORKFLOW: scheduled '{entry['workflow']}' ({describe_schedule(entry)}) - "
                  f"{first_line(result.text, 160)}")
        if not result.ok:
            try:
                self.engine.bus.banner.emit(
                    f"Scheduled workflow '{entry['workflow']}' did not start", 2)
            except Exception:
                pass
        return bool(result.ok)

    def _listen(self) -> Any:
        """A quiet-conversation probe fed by ORION's states, or None."""
        try:
            from .semantic import ConversationQuiet
            tracker = ConversationQuiet()
            self.engine.bus.state.connect(tracker.note_state)
        except Exception:
            return None
        return tracker.quiet

    async def run(self) -> None:
        if self._quiet is None:
            self._quiet = self._listen()
        await asyncio.sleep(self.FIRST_TICK_S)
        while not self._stopped:
            try:
                await self.tick()
            except Exception as exc:
                self._log(f"WORKFLOW: schedule check recovered - {first_line(exc, 120)}")
            await asyncio.sleep(self.TICK_S)

    def start(self) -> Any:
        """Begin checking on the running loop; a second call is a no-op."""
        if self._task is not None and not self._task.done():
            return self._task
        from . import background
        self._task = background.spawn(self.run(), name="orion-workflow-schedules")
        return self._task

    def stop(self) -> None:
        self._stopped = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()


class AutomationManager:
    """Dispatcher-facing facade; seeds the built-in workflow library and owns
    the scheduler (started by app.py once the loop is running)."""

    def __init__(self, engine: WorkflowEngine, *,
                 scheduler: WorkflowScheduler | None = None) -> None:
        self.engine = engine
        self.scheduler = scheduler if scheduler is not None else WorkflowScheduler(engine)
        self._seed_builtins()

    def start_scheduler(self) -> Any:
        """Run due schedules from now on (needs the running event loop)."""
        return self.scheduler.start()

    def _uses_input(self, name: str) -> bool:
        definition = self.engine.definitions.get(self.engine._slug(name)) or {}
        return "{input}" in json.dumps(definition.get("steps") or [])

    def _seed_builtins(self) -> None:
        if self.engine._load_error:
            return
        updated = dict(self.engine.definitions)
        for name, definition in _BUILTIN_WORKFLOWS.items():
            if name not in updated:
                updated[name] = copy.deepcopy(dict(definition, updated_at=utc_stamp()))
        if updated != self.engine.definitions and self.engine._save(updated):
            self.engine.definitions = updated

    async def handle(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        name = str(args.get("name") or args.get("workflow") or "")
        if action in {"list", "library"}:
            return self.engine.list_workflows()
        if action in {"run", "start", "execute"}:
            return self.engine.start(name, str(args.get("input")
                                               or args.get("text") or ""))
        if action in {"status", "progress"}:
            status = self.engine.status()
            scheduled = self.scheduler.summary()
            return replace(status, text=f"{status.text}\n{scheduled}") if scheduled else status
        if action in {"result", "results", "output", "found", "findings", "last", "last_result"}:
            return self.engine.result(str(args.get("run") or name))
        if action in {"schedule", "schedule_run", "add_schedule"}:
            when = str(args.get("when") or args.get("time") or "")
            input_text = str(args.get("input") or args.get("text") or "")
            if not when and input_text and not self._uses_input(name):
                # The schedule said as the input of a workflow that takes none.
                when, input_text = input_text, ""
            return self.scheduler.add(name, when, input_text,
                                      repeat=args.get("repeat") or "", at=args.get("at") or "",
                                      day=args.get("day") or "", on_date=args.get("date") or "")
        if action in {"schedules", "scheduled", "list_schedules", "timetable"}:
            return self.scheduler.describe()
        if action in {"unschedule", "remove_schedule", "cancel_schedule", "stop_schedule",
                      "delete_schedule"}:
            return self.scheduler.remove(str(args.get("schedule") or args.get("id") or name))
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
        if action in {"resume", "continue"}:
            return self.engine.resume(str(args.get("run") or name))
        if action in {"approve", "decline", "deny"}:
            # An approval is bound to one step of one run; deciding it here
            # only records the decision — "resume" re-checks it before the
            # step can run.
            approval = str(args.get("approval") or args.get("run") or "").strip()
            decided = self.engine.store.decide_approval(approval, action == "approve")
            if decided is None:
                return ToolResult("That approval is no longer pending (it expired, was "
                                  "replaced, or was already decided).", ok=False)
            return ToolResult(f"{'Approved' if action == 'approve' else 'Declined'} step "
                              f"{decided['step_idx']} ({decided['tool']}). Say resume to "
                              "continue the workflow.")
        return ToolResult("Unsupported workflow action. Use list, run, status, result, "
                          "schedule, schedules, unschedule, cancel, define, remove, resume, "
                          "approve or decline.", ok=False)


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
    # Read-only checks the owner can schedule ("run morning_check every weekday
    # at 7:30"). Each step is optional, so one source being offline (the
    # weather, Notion) costs that part, not the whole check. Deliberately not
    # morning_briefing: it marks the stories it reads as heard, so a briefing
    # run in the background would empty the one he asks for out loud.
    "morning_check": {
        "description": "Start of the day: what ORION is aware of, deadlines, today's calendar "
                       "and priority email, the weather, and goals needing attention. Read-only.",
        "steps": [
            {"tool": "awareness", "args": {"action": "situation"}, "retries": 1, "optional": True},
            {"tool": "proactive_check", "args": {}, "retries": 1, "optional": True},
            {"tool": "geo", "args": {"action": "weather", "when": "today"}, "retries": 1,
             "optional": True},
            {"tool": "goals", "args": {"action": "attention"}, "retries": 1, "optional": True},
        ],
    },
    "research_digest": {
        "description": "What the research programme has found: its agenda, the latest findings "
                       "and open opportunities. Read-only.",
        "steps": [
            {"tool": "research", "args": {"action": "agenda"}, "retries": 1, "optional": True},
            {"tool": "research", "args": {"action": "findings"}, "retries": 1, "optional": True},
            {"tool": "research", "args": {"action": "opportunities"}, "retries": 1,
             "optional": True},
        ],
    },
    "downloads_tidy_preview": {
        "description": "Dry run of tidying Downloads: what would move where, and the largest "
                       "files there. Moves nothing.",
        "steps": [
            # preview only: organise_files' preview never moves a file.
            {"tool": "organise_files", "args": {"action": "preview", "folder": "Downloads"},
             "retries": 0, "optional": True},
            {"tool": "find_files", "args": {"folder": "Downloads", "sort": "largest", "limit": 10},
             "retries": 1, "optional": True},
        ],
    },
    "weekly_goal_review": {
        "description": "The weekly goals review: what moved, what slipped, and what needs "
                       "attention next. Read-only.",
        "steps": [
            {"tool": "goals", "args": {"action": "review", "days": 7}, "retries": 1},
            {"tool": "goals", "args": {"action": "attention"}, "retries": 1, "optional": True},
        ],
    },
}


__all__ = ["AutomationManager", "ScheduleError", "WorkflowEngine", "WorkflowScheduler",
           "WORKFLOWS_PATH", "describe_schedule", "parse_when"]
