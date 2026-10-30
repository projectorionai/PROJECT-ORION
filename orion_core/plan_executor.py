"""
Self-verifying plan executor (improvements #1, #3; Track B: concurrency).

``PlanExecutor`` runs a multi-step autonomous plan and verifies each step
before moving on, with bounded retries — so ORION's autonomy is reliable
rather than fire-and-forget.

Verification strategy per step:
    • screen-changing desktop_control / web_control — a pixel-diff check via
      VisualVerificationEngine. An uncertain effect is not repeated.
    • read-only observations and other tools — ``ToolResult.ok`` is the verdict.

Each step may declare an ``on_fail`` policy: ``retry`` (bounded, then carry
on), ``continue`` (log and move on) or ``abort`` (stop the plan).  A
consolidated, honest report is returned: what was attempted, what verified,
what failed and why.  This runs entirely on the qasync loop; the verification
captures happen off-thread inside the verification engine.

DEFAULT ON FAILURE (Mark XXXVII, 23 Oct 2026).  The default used to be
``retry``, which for anything that changes something meant "carry on": "open
Word, then type the letter" typed the letter into whatever window had focus
when Word failed to open. With no ``on_fail`` given, a failed step that
CHANGES something now stops a linear plan (the rest is skipped and reported);
in a dependency-graph plan its dependants are skipped and steps that declared
no dependency on it still run. A failed read-only step is retried (bounded)
and the plan carries on, exactly as before.

STEP OUTPUT.  An argument value may contain ``{prev}`` (the previous step's
result text) or ``{step:<id>}`` (a named earlier step's). The text passed on
is bounded (``MAX_PASSED_CHARS``, cut at a line and marked when shortened); a
step whose input step did not succeed is skipped rather than run with nothing.
A step that consumes another's output never runs in a parallel batch, and is
treated as changing something: its real action is only known once expanded.

SCHEDULING (Track B).  The executor was a strict sequential loop, so a plan
whose first three steps were independent lookups paid for all three in series.
It now schedules two ways:

  • **Linear plans** (no step declares a dependency) keep their exact order,
    but adjacent steps the concurrency classifier calls read-only are gathered
    into one batch. Existing plans get faster with no change to how they are
    written.
  • **DAG plans** — any step carrying ``id`` and/or ``after`` — are executed in
    dependency waves: everything whose prerequisites are met runs, with the
    parallel-safe members of each wave running together. A step whose
    dependency failed is *skipped and reported*, never silently run against a
    precondition that never happened.

Visual steps are excluded from batching regardless of class: their verdict is
a before/after pixel diff of one screen, and two of them running at once would
each be measuring the other's changes.

STRICT VALIDATION (2026-09-26).  The WHOLE plan is validated before anything
is dispatched, and an invalid plan dispatches nothing. Rejected: duplicate
step ids, prerequisites that name no step, dependency cycles, unknown tools,
tools not permitted inside a plan, arguments that fail the tool's declared
parameters, malformed steps, and plans over ``MAX_PLAN_STEPS``. This replaced
three silent repairs — an edge to an unknown step was DROPPED (so a misspelt
prerequisite let its dependant run with its precondition removed), a
duplicate id was renamed, and a long plan was truncated — each of which ran a
different plan from the one written.

DURABILITY.  A plan is recorded in the job store as it runs, step by step,
with each step's side-effect class. After a restart an unfinished plan is
INTERRUPTED; resuming it re-runs a read-only step that was in flight, but a
side-effecting one is never replayed blindly — it waits for an approval bound
to that exact step, or is skipped (with its dependants) if the approval is
declined.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Iterable, Optional

from . import job_store as _store_mod
from .admission import Busy, WorkClass
from .bus import OrionBus
from .concurrency import ToolClass, classify, max_parallel, parallelism_enabled
from .data import ToolResult
from .job_store import READ_ONLY, SIDE_EFFECT, State, side_effect_class
from .jobs import CONTROLS, JobCancelled, JobControl, _CURRENT_CONTROL, register_resumer
from .utils import first_line

# Tools whose success is judged by whether the screen actually changed.
VISUAL_TOOLS = {"desktop_control", "web_control", "vision_verify"}

MAX_PLAN_STEPS = 24

#: Lifecycle, self-modification and nested plans. A plan that restarts ORION,
#: repairs its source or launches another plan is not something to run as one
#: step among many; the inline fallback in the dispatcher already refused the
#: nested plan and the restart.
FORBIDDEN_IN_PLAN = frozenset({"shutdown_orion", "restart_orion", "self_repair",
                               "execute_plan", "autoplan"})

#: "auto" is what a step that names no policy gets: stop on a failed step that
#: changes something, retry-then-continue on a read-only one (module docstring).
_ON_FAIL = {"retry": "retry", "continue": "continue", "abort": "abort",
            "stop": "abort", "skip": "continue", "ignore": "continue", "auto": "auto"}

#: ``{prev}`` or ``{step:<id>}`` inside an argument value.
_PLACEHOLDER = re.compile(r"\{(prev|step:([A-Za-z0-9_.\-]+))\}")
#: Most of one step's result text passed into another's argument. Search
#: results typed into a document fit; a whole web page does not, and is cut at
#: a line boundary with a visible marker rather than silently.
MAX_PASSED_CHARS = 4000
#: Kept per successful step for later {prev}/{step:id} use (memory bound).
_KEPT_OUTPUT_CHARS = 20_000


@dataclass
class StepResult:
    index: int
    tool: str
    ok: bool
    verified: bool
    attempts: int
    detail: str
    step_id: str = ""
    skipped: bool = False
    #: The step's full result text (bounded), for {prev}/{step:id}; never shown.
    output: str = ""


@dataclass
class PlanReport:
    objective: str
    steps: list[StepResult] = field(default_factory=list)
    aborted: bool = False
    schedule: str = ""
    cancelled: bool = False
    job_id: str = ""
    #: Why the plan stopped early, when a failed step stopped it.
    stopped_by: str = ""
    #: step_id -> result text of the steps that SUCCEEDED in this run.
    outputs: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> int:
        return sum(1 for s in self.steps if s.ok)

    @property
    def verified(self) -> int:
        return sum(1 for s in self.steps if s.verified)

    @property
    def skipped(self) -> int:
        return sum(1 for s in self.steps if s.skipped)

    def to_tool_result(self) -> ToolResult:
        lines = [f"PLAN: {self.objective}" if self.objective else "PLAN executed:"]
        for s in sorted(self.steps, key=lambda item: item.index):
            mark = "✓" if s.ok else ("–" if s.skipped else "✗")
            v = "verified" if s.verified else "unverified"
            lines.append(f"  {mark} [{s.index}] {s.tool} ({v}, {s.attempts} try): {s.detail[:110]}")
        summary = (
            f"RESULT: {self.succeeded}/{len(self.steps)} succeeded, "
            f"{self.verified} visually verified"
        )
        on_dependency = sum(1 for s in self.steps
                            if s.skipped and s.detail.startswith("skipped — dependency"))
        other = self.skipped - on_dependency
        if on_dependency:
            summary += f", {on_dependency} skipped on a failed dependency"
        if other:
            summary += f", {other} not run"
        if self.cancelled:
            summary += " — plan cancelled."
        elif self.aborted and self.stopped_by:
            summary += (f" — plan aborted early: {self.stopped_by}, so the steps after it "
                        "were not run.")
        else:
            summary += " — plan aborted early." if self.aborted else "."
        lines.append(summary)
        if self.schedule:
            lines.append(f"SCHEDULE: {self.schedule}")
        if self.job_id:
            lines.append(f"JOB: {self.job_id}")
        return ToolResult(
            "\n".join(lines),
            ok=(not self.aborted and not self.cancelled
                and self.succeeded == len(self.steps)),
            evidence=[{
                "index": step.index,
                "step_id": step.step_id,
                "tool": step.tool,
                "ok": step.ok,
                # The executor's visual check establishes only that pixels
                # changed, not that the requested postcondition was met.
                "evidence": "screen_change" if step.verified else "unknown",
                "skipped": step.skipped,
                "detail": step.detail,
            } for step in sorted(self.steps, key=lambda item: item.index)],
        )


@dataclass
class _Step:
    """One parsed plan step, with its dependency edges resolved."""

    index: int
    step_id: str
    tool: str
    args: dict[str, Any]
    on_fail: str
    after: list[str] = field(default_factory=list)
    #: Placeholder as written ("prev", "a", "2") -> the step_id it names.
    inputs: dict[str, str] = field(default_factory=dict)
    #: Sources whose text was cut to MAX_PASSED_CHARS when filled in.
    shortened: tuple[str, ...] = ()

    @property
    def batchable(self) -> bool:
        """Safe to run alongside other steps in the same wave.

        Visual steps are excluded whatever their class: their verdict is a
        pixel diff of the whole screen, so two running at once would each
        measure the other's changes. A step consuming another's output must
        wait for it, so it never shares a batch either.
        """
        return (not self.inputs
                and classify(self.tool, self.args) is ToolClass.PARALLEL
                and self.tool not in VISUAL_TOOLS)

    @property
    def side_effect(self) -> str:
        found = side_effect_class(self.tool, self.args)
        if self.inputs and found == READ_ONLY:
            # "{prev}" in an action could expand to a writing branch, so the
            # template's class proves nothing: never retried, never replayed
            # after a restart without approval.
            return SIDE_EFFECT
        return found

    def stops_plan_on_failure(self, linear: bool) -> bool:
        """Whether this step failing stops the rest of the plan."""
        if self.on_fail == "abort":
            return True
        # Default: something that changes the machine failed, so what follows
        # (written assuming it worked) must not run. In a dependency graph the
        # dependants are skipped instead and declared-independent work runs.
        return self.on_fail == "auto" and linear and self.side_effect != READ_ONLY


# ── the tool catalogue plans are validated against ────────────────────────────

class ToolCatalogue:
    """Which tools exist, and what arguments each declares."""

    def __init__(self, declarations: Iterable[dict[str, Any]],
                 extra_names: Iterable[str] = ()) -> None:
        self._declarations = {str(d.get("name")): d for d in declarations if d.get("name")}
        self._names = set(self._declarations) | {str(n) for n in extra_names}

    @classmethod
    def from_dispatch(cls, dispatch: Any) -> "ToolCatalogue":
        """The live table when *dispatch* is a dispatcher's bound method (so
        MCP, forged and plugin tools count), else the static declarations."""
        from .dispatch_schema import TOOL_DECLARATIONS
        owner = getattr(dispatch, "__self__", None)
        declarations = list(getattr(owner, "TOOL_DECLARATIONS", None) or TOOL_DECLARATIONS)
        names: set[str] = set()
        table = getattr(owner, "handler_table", None)
        if callable(table):
            try:
                names |= set(table())
            except Exception:
                pass
        forged = getattr(owner, "_tool_handlers", None)
        if isinstance(forged, dict):
            names |= set(forged)
        return cls(declarations, names)

    def known(self, name: str) -> bool:
        return name in self._names

    def check_args(self, name: str, args: dict[str, Any]) -> str | None:
        """None when *args* satisfy *name*'s declared parameters, else why not.

        Uses the dispatcher's own normalisation first (numeric strings and
        "true"/"false" are accepted exactly as the dispatcher would accept
        them), then requires what the declaration requires and rejects values
        of the wrong kind. A tool with no declaration (a forged tool) has
        nothing to check against.
        """
        declaration = self._declarations.get(name)
        if declaration is None:
            return None
        from .tool_args import coerce_args, declared_types
        coerced, error = coerce_args(args, declaration)
        if error:
            return error
        parameters = declaration.get("parameters") or {}
        required = [str(r) for r in (parameters.get("required") or [])]
        missing = [key for key in required
                   if coerced.get(key) is None
                   or (isinstance(coerced.get(key), str) and not coerced[key].strip())]
        if missing:
            return f"missing required argument(s): {', '.join(missing)}"
        for key, kind in declared_types(declaration).items():
            if key not in coerced:
                continue
            value = coerced[key]
            if kind in {"INTEGER", "NUMBER"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return f"'{key}' must be a number, but got {value!r}"
                if kind == "INTEGER" and isinstance(value, float) and not value.is_integer():
                    return f"'{key}' must be a whole number, but got {value!r}"
            elif kind == "ARRAY" and not isinstance(value, (list, tuple)):
                return f"'{key}' must be a list"
            elif kind == "OBJECT" and not isinstance(value, dict):
                return f"'{key}' must be an object"
            elif kind == "STRING" and isinstance(value, (dict, list, tuple)):
                return f"'{key}' must be text"
        return None


def validate_plan(raw_steps: Any, catalogue: ToolCatalogue) -> tuple[list[_Step], list[str]]:
    """Parse and check a whole plan. Returns (steps, problems); any problem
    means the plan must not run at all."""
    if not isinstance(raw_steps, list) or not raw_steps:
        return [], ["No plan steps supplied."]
    if len(raw_steps) > MAX_PLAN_STEPS:
        return [], [f"The plan has {len(raw_steps)} steps; the limit is {MAX_PLAN_STEPS}. "
                    "Split it into smaller plans rather than have steps dropped."]
    problems: list[str] = []
    parsed: list[_Step] = []
    first_at: dict[str, int] = {}
    raw_edges: list[tuple[_Step, list[Any]]] = []
    for index, raw in enumerate(raw_steps, 1):
        if not isinstance(raw, dict):
            problems.append(f"Step {index} is not an object with a tool and args.")
            continue
        tool = str(raw.get("tool") or raw.get("name") or "").strip()
        if not tool:
            problems.append(f"Step {index} names no tool.")
            continue
        args = raw.get("args")
        if args is None:
            text = raw.get("args_json")
            if text in (None, ""):
                args = {}
            else:
                try:
                    args = json.loads(str(text))
                except Exception:
                    problems.append(f"Step {index} ({tool}): args_json is not valid JSON.")
                    continue
        if not isinstance(args, dict):
            problems.append(f"Step {index} ({tool}): its args must be an object.")
            continue
        policy = _ON_FAIL.get(str(raw.get("on_fail") or "auto").strip().lower())
        if policy is None:
            problems.append(f"Step {index} ({tool}): on_fail must be stop, continue or "
                            f"retry, not {raw.get('on_fail')!r}.")
            continue
        step_id = str(raw.get("id") or raw.get("step_id") or "").strip() or f"step_{index}"
        if step_id in first_at:
            problems.append(f"Steps {first_at[step_id]} and {index} share the id "
                            f"'{step_id}'; every step needs its own.")
        else:
            first_at[step_id] = index
        raw_after = raw.get("after", raw.get("depends_on"))
        if raw_after is None:
            raw_after = []
        elif isinstance(raw_after, (str, int)) and not isinstance(raw_after, bool):
            raw_after = [raw_after]
        elif not isinstance(raw_after, (list, tuple)):
            problems.append(f"Step {index} ({tool}): 'after' must list step ids.")
            raw_after = []
        if tool in FORBIDDEN_IN_PLAN:
            problems.append(f"Step {index} ({tool}) is not permitted inside a plan.")
        elif not catalogue.known(tool):
            problems.append(f"Step {index} uses an unknown tool '{tool}'.")
        else:
            reason = catalogue.check_args(tool, args)
            if reason:
                problems.append(f"Step {index} ({tool}): {reason}.")
        step = _Step(index=index, step_id=step_id, tool=tool, args=args, on_fail=policy)
        parsed.append(step)
        raw_edges.append((step, list(raw_after)))

    ids = {step.step_id for step in parsed}
    by_index = {step.index: step.step_id for step in parsed}
    for step, raw_after in raw_edges:
        for dep in raw_after:
            name = str(dep).strip()
            if not name:
                problems.append(f"Step '{step.step_id}' has an empty prerequisite.")
                continue
            if name not in ids:
                # A number refers to a step by position ("after": [1]).
                if name.isdigit() and int(name) in by_index:
                    name = by_index[int(name)]
                else:
                    problems.append(f"Step '{step.step_id}' depends on '{name}', which is "
                                    "not a step in this plan.")
                    continue
            if name == step.step_id:
                problems.append(f"Step '{step.step_id}' depends on itself.")
                continue
            if name not in step.after:
                step.after.append(name)

    # {prev} / {step:<id>}: each must name an EARLIER step of this plan. In a
    # dependency graph the consumer also waits for its source, so an output is
    # never read before the step that makes it has run.
    graph = any(step.after for step in parsed)
    order = {step.step_id: step.index for step in parsed}
    for position, step in enumerate(parsed):
        for raw_ref in _references(step.args):
            if raw_ref in step.inputs:
                continue
            if raw_ref == "prev":
                if position == 0:
                    problems.append(f"Step '{step.step_id}' uses {{prev}}, but no step comes "
                                    "before it.")
                    continue
                source = parsed[position - 1].step_id
            elif raw_ref in ids:
                source = raw_ref
            elif raw_ref.isdigit() and int(raw_ref) in by_index:
                source = by_index[int(raw_ref)]
            else:
                problems.append(f"Step '{step.step_id}' uses the output of '{raw_ref}', which "
                                "is not a step in this plan.")
                continue
            if order[source] >= step.index:
                problems.append(f"Step '{step.step_id}' can only use the output of an earlier "
                                f"step, not '{source}'.")
                continue
            step.inputs[raw_ref] = source
            if graph and source not in step.after:
                step.after.append(source)

    if not problems:
        cycle = _cycle(parsed)
        if cycle:
            problems.append("These steps depend on each other in a cycle, so none could "
                            f"ever start: {', '.join(cycle)}.")
    return parsed, problems


def _cycle(steps: list[_Step]) -> list[str]:
    """Step ids left over by a topological sort — non-empty means a cycle."""
    remaining = {step.step_id: set(step.after) for step in steps}
    while True:
        free = [sid for sid, deps in remaining.items() if not deps]
        if not free:
            return sorted(remaining)
        for sid in free:
            remaining.pop(sid)
        for deps in remaining.values():
            deps.difference_update(free)
        if not remaining:
            return []


def _references(value: Any) -> list[str]:
    """Every placeholder in *value*, as written: "prev" or a step's id/number."""
    found: list[str] = []
    if isinstance(value, str):
        for match in _PLACEHOLDER.finditer(value):
            found.append("prev" if match.group(1) == "prev" else match.group(2))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_references(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_references(item))
    return found


def _passed(text: str) -> tuple[str, bool]:
    """*text* as it may be passed into another step: whole, or cut at a line
    near MAX_PASSED_CHARS and visibly marked. Returns (text, shortened)."""
    text = str(text or "")
    if len(text) <= MAX_PASSED_CHARS:
        return text, False
    cut = text[:MAX_PASSED_CHARS]
    line = cut.rfind("\n")
    if line > MAX_PASSED_CHARS * 0.6:
        cut = cut[:line]
    return cut.rstrip() + " […]", True


def resolve_inputs(step: _Step, outputs: dict[str, str]) -> tuple[_Step | None, str]:
    """*step* with its {prev}/{step:id} placeholders replaced by those steps'
    results, or (None, why not). A placeholder whose step did not succeed in
    this run is never replaced by nothing: the step is not run."""
    if not step.inputs:
        return step, ""
    missing = [source for source in step.inputs.values() if source not in outputs]
    if missing:
        return None, (f"skipped — its input, the result of step '{missing[0]}', is not "
                      "available (that step did not succeed in this run)")
    shortened: list[str] = []

    def substitute(match: Any) -> str:
        raw = "prev" if match.group(1) == "prev" else match.group(2)
        source = step.inputs.get(raw)
        if source is None:                  # validation names every placeholder
            return match.group(0)
        text, cut = _passed(outputs[source])
        if cut:
            shortened.append(source)
        return text

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            # One pass: a result that itself contains "{prev}" is data.
            return _PLACEHOLDER.sub(substitute, value)
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    args = walk(step.args)
    try:
        size = len(json.dumps(args, default=str))
    except (TypeError, ValueError):
        size = _store_mod.MAX_ARGS_CHARS + 1
    if size > _store_mod.MAX_ARGS_CHARS:
        return None, ("not run — with the earlier results filled in, its arguments exceed "
                      f"{_store_mod.MAX_ARGS_CHARS:,} characters")
    return replace(step, args=args, shortened=tuple(dict.fromkeys(shortened))), ""


class PlanExecutor:
    def __init__(self, bus: OrionBus, dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolResult]],
                 verifier: Any | None = None, telemetry: Any | None = None, *,
                 catalogue: ToolCatalogue | Callable[[], ToolCatalogue] | None = None,
                 store: Any = None, admission: Any = None) -> None:
        self.bus = bus
        self.dispatch = dispatch
        self.verifier = verifier
        self.telemetry = telemetry
        self._catalogue_source = catalogue
        self._store = store
        self._admission = admission
        register_resumer("plan", self.resume_job)

    def catalogue(self) -> ToolCatalogue:
        """Built per plan: MCP, forged and plugin tools register at runtime."""
        source = self._catalogue_source
        if isinstance(source, ToolCatalogue):
            return source
        if callable(source):
            return source()
        return ToolCatalogue.from_dispatch(self.dispatch)

    @property
    def store(self) -> Any:
        return self._store if self._store is not None else _store_mod.default_store()

    def _count_metric(self, name: str) -> None:
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr(name)
            except Exception:
                pass

    async def execute(self, steps: list[dict[str, Any]], objective: str = "",
                      max_retries: int = 2) -> ToolResult:
        if not isinstance(steps, list) or not steps:
            return ToolResult("No plan steps supplied.", ok=False)
        parsed, problems = validate_plan(list(steps), self.catalogue())
        if problems:
            # Nothing has been dispatched, and nothing will be: a partial or
            # repaired plan is a different plan from the one written.
            self.bus.log.emit(f"PLAN: rejected before running — {problems[0]}")
            self._count_metric("plan.rejected")
            return ToolResult(
                "PLAN REJECTED — nothing was run:\n"
                + "\n".join(f"  - {problem}" for problem in problems[:12])
                + "\nFix the plan and submit it again.", ok=False)

        objective = str(objective or "").strip()
        try:
            ticket = self._admission_obj().submit(WorkClass.INTERACTIVE,
                                                  label=objective or "plan")
        except Busy as busy:
            return ToolResult(f"PLAN not started — {busy.reason}", ok=False)
        try:
            control = JobControl.create(
                kind="plan", name=objective or "plan", label=objective or "plan",
                cls=WorkClass.INTERACTIVE,
                args={"objective": objective, "steps": list(steps),
                      "max_retries": int(max_retries)},
                steps=[{"index": s.index, "step_id": s.step_id, "tool": s.tool, "args": s.args,
                        "side_effect": s.side_effect} for s in parsed],
                store=self.store, admission=self._admission, ticket=ticket)
        except Exception as exc:
            # create() has already given the slot back.
            self.bus.log.emit(f"PLAN: could not be recorded — {first_line(exc, 160)}")
            return ToolResult("PLAN not started — it could not be recorded, so nothing "
                              "was run. Check the log and try again.", ok=False)
        # Asked for by the owner: his own turn for its every step, even once
        # Live has answered and left it to finish (execution_policy).
        from .execution_policy import attended_scope
        with attended_scope("plan"):
            return await self._run_plan(control, parsed, objective, max_retries, preset={})

    def _admission_obj(self) -> Any:
        if self._admission is not None:
            return self._admission
        from . import admission
        return admission.ADMISSION

    async def _run_plan(self, control: JobControl, parsed: list[_Step], objective: str,
                        max_retries: int, preset: dict[str, StepResult]) -> ToolResult:
        report = PlanReport(objective=objective, job_id=control.id)
        has_dag = any(step.after for step in parsed)
        self.bus.log.emit(
            f"PLAN: executing {len(parsed) - len(preset)} step(s) with self-verification "
            f"({'dependency graph' if has_dag else 'linear'}).")
        token = _CURRENT_CONTROL.set(control)
        try:
            # Nested so a record write failing inside one of these handlers
            # (marking the rest skipped) still reaches the fault branch below
            # and retires the slot, instead of escaping with the control live.
            try:
                await control.admit()
                if has_dag:
                    await self._execute_dag(parsed, report, max_retries, control, preset)
                else:
                    await self._execute_linear(parsed, report, max_retries, control, preset)
            except _PlanAborted:
                report.aborted = True
            except JobCancelled:
                report.cancelled = True
                self._skip_remaining(parsed, report, "skipped — plan cancelled", control)
            except Busy as busy:
                report.cancelled = True
                self._skip_remaining(parsed, report, f"skipped — {busy.reason}", control)
        except asyncio.CancelledError:
            # A record write failing here must not turn the cancellation into
            # a different exception: the caller is unwinding and needs to see
            # CancelledError. finish() retires the slot and control regardless.
            try:
                self._skip_remaining(parsed, report, "skipped — plan cancelled", control)
            except Exception:
                pass
            try:
                control.finish(State.CANCELLED, ok=False, result="cancelled")
            except Exception:
                pass
            raise
        except Exception as exc:
            # An unexpected fault (a progress write, say) can land before a
            # dispatch or after a tool already returned. Retire the slot and
            # the live control, record what is known, and never replay a step:
            # anything recorded as running may already have happened.
            self.bus.log.emit(f"PLAN: stopped by an internal fault — {first_line(exc, 160)}")
            try:
                control.finish(State.FAILED, ok=False,
                               result=f"stopped by an internal fault ({type(exc).__name__}); "
                                      "a step recorded as running may already have happened")
            except Exception:
                pass                    # finish() has retired the slot and control regardless
            raise
        finally:
            _CURRENT_CONTROL.reset(token)
        result = report.to_tool_result()
        final = State.CANCELLED if report.cancelled else (
            State.COMPLETED if result.ok else State.FAILED)
        try:
            control.finish(final, ok=result.ok, result=result.text,
                           fraction=1.0 if final is State.COMPLETED else None)
        except Exception as exc:
            # finish() has retired the slot and the live control regardless.
            # The steps above DID run, so raising here would hide from the
            # conversation what was done; say instead that the record of it
            # was not saved, rather than letting the job list imply it was.
            self.bus.log.emit(f"PLAN: outcome could not be recorded — {first_line(exc, 160)}")
            result = replace(result, text=result.text + (
                f"\nJOB RECORD: the outcome could not be saved ({type(exc).__name__}); "
                "the job list may still show this plan as unfinished."))
        self._count_metric("plan.executed")
        return result

    def _skip_remaining(self, parsed: list[_Step], report: PlanReport, detail: str,
                        control: JobControl | None) -> None:
        reported = {r.index for r in report.steps}
        for step in parsed:
            if step.index in reported:
                continue
            report.steps.append(StepResult(step.index, step.tool, False, False, 0, detail,
                                           step.step_id, skipped=True))
            if control is not None:
                control.step(step.index, state="skipped", detail=detail)

    # ── linear scheduling (order preserved, read-only neighbours batched) ─────

    async def _execute_linear(self, steps: list[_Step], report: PlanReport,
                              max_retries: int, control: JobControl | None = None,
                              preset: dict[str, StepResult] | None = None) -> None:
        preset = preset or {}
        ceiling = max_parallel() if parallelism_enabled() else 1
        batch: list[_Step] = []
        parallel_batches = 0

        async def _flush() -> None:
            nonlocal batch, parallel_batches
            if not batch:
                return
            if len(batch) > 1:
                parallel_batches += 1
            await self._run_wave(batch, report, max_retries, control, linear=True)
            batch = []

        try:
            for step in steps:
                if step.step_id in preset:
                    report.steps.append(preset[step.step_id])
                    continue
                if step.batchable and ceiling > 1:
                    batch.append(step)
                    if len(batch) >= ceiling:
                        await _flush()
                    continue
                await _flush()
                await self._run_wave([step], report, max_retries, control, linear=True)
            await _flush()
        except _PlanAborted:
            self._skip_remaining(steps, report, self._abort_detail(report), control)
            raise
        if parallel_batches:
            report.schedule = f"{parallel_batches} batch(es) ran in parallel"

    # ── dependency-graph scheduling ──────────────────────────────────────────

    async def _execute_dag(self, steps: list[_Step], report: PlanReport,
                           max_retries: int, control: JobControl | None = None,
                           preset: dict[str, StepResult] | None = None) -> None:
        preset = preset or {}
        ceiling = max_parallel() if parallelism_enabled() else 1
        pending = {step.step_id: step for step in steps}
        done: dict[str, bool] = {}          # step_id → succeeded
        waves = 0

        # Steps settled by an earlier run (a resumed plan) count as done.
        for step_id, result in preset.items():
            if step_id in pending:
                pending.pop(step_id)
                done[step_id] = result.ok
                report.steps.append(result)

        # Validation has already rejected unknown prerequisites and cycles,
        # so every remaining step becomes ready eventually.
        while pending:
            ready = [s for s in pending.values()
                     if all(dep in done for dep in s.after)]
            if not ready:                   # pragma: no cover - validation forbids it
                self._skip_remaining(steps, report, "skipped — dependency cycle", control)
                report.aborted = True
                break

            # A step whose prerequisite failed must not run: its precondition
            # never happened, so running it anyway is how a plan does damage.
            blocked = [s for s in ready if not all(done.get(d) for d in s.after)]
            for step in blocked:
                failed = [d for d in step.after if not done.get(d)]
                pending.pop(step.step_id, None)
                done[step.step_id] = False
                detail = f"skipped — dependency '{failed[0]}' failed"
                report.steps.append(StepResult(
                    step.index, step.tool, False, False, 0, detail, step.step_id,
                    skipped=True))
                if control is not None:
                    control.step(step.index, state="skipped", detail=detail)
                self.bus.log.emit(
                    f"PLAN: skipping '{step.step_id}' — dependency "
                    f"'{failed[0]}' did not succeed.")
            runnable = [s for s in ready if s not in blocked]
            if not runnable:
                continue

            runnable.sort(key=lambda s: s.index)
            wave: list[_Step] = []
            for step in runnable:
                if step.batchable and ceiling > 1 and len(wave) < ceiling:
                    wave.append(step)
            if not wave:
                wave = [runnable[0]]
            if len(wave) > 1:
                waves += 1

            try:
                results = await self._run_wave(wave, report, max_retries, control)
            except _PlanAborted:
                self.bus.log.emit("PLAN: aborted — the remaining steps will not run.")
                self._skip_remaining(steps, report, self._abort_detail(report), control)
                raise
            for step, result in zip(wave, results):
                pending.pop(step.step_id, None)
                done[step.step_id] = result.ok
        if waves:
            report.schedule = f"{waves} dependency wave(s) ran in parallel"

    # ── execution ────────────────────────────────────────────────────────────

    async def _run_wave(self, wave: list[_Step], report: PlanReport,
                        max_retries: int, control: JobControl | None = None, *,
                        linear: bool = False) -> list[StepResult]:
        """Run one batch of steps, concurrently when there is more than one."""
        if control is not None:
            # Between waves is where a cancel lands and a pause takes effect.
            await control.checkpoint()
        if len(wave) == 1:
            step = wave[0]
            result = await self._run_with_inputs(step, report, max_retries, control)
            report.steps.append(result)
            self._keep_output(report, result)
            self._count(result)
            self._progress(control, report)
            self._stop_if_needed(step, result, report, linear)
            return [result]

        self.bus.log.emit(
            "PLAN: running " + " ‖ ".join(s.tool for s in wave) + " concurrently.")
        gathered = await asyncio.gather(
            *(self._run_with_inputs(step, report, max_retries, control) for step in wave),
            return_exceptions=True,
        )
        results: list[StepResult] = []
        for step, outcome in zip(wave, gathered):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                outcome = StepResult(step.index, step.tool, False, False, 1,
                                     first_line(outcome, 120), step.step_id)
            results.append(outcome)
            report.steps.append(outcome)
            self._keep_output(report, outcome)
            self._count(outcome)
        self._progress(control, report)
        # A batch is only entered by steps that cannot interfere, so an abort
        # policy is honoured after the whole batch rather than mid-flight.
        for step, result in zip(wave, results):
            self._stop_if_needed(step, result, report, linear)
        return results

    def _stop_if_needed(self, step: _Step, result: StepResult, report: PlanReport,
                        linear: bool) -> None:
        if result.ok or not step.stops_plan_on_failure(linear):
            return
        report.aborted = True
        report.stopped_by = (f"step {step.index} ({step.tool}) "
                             f"{'was not run' if result.skipped else 'failed'}")
        self.bus.log.emit(f"PLAN: aborted at step {step.index} ({step.tool}).")
        raise _PlanAborted()

    @staticmethod
    def _abort_detail(report: PlanReport) -> str:
        return (f"skipped — not run because {report.stopped_by}" if report.stopped_by
                else "skipped — plan aborted")

    @staticmethod
    def _keep_output(report: PlanReport, result: StepResult) -> None:
        if result.ok and result.step_id:
            report.outputs[result.step_id] = result.output

    async def _run_with_inputs(self, step: _Step, report: PlanReport, max_retries: int,
                               control: JobControl | None) -> StepResult:
        """Fill in {prev}/{step:id} from this run's results, then run the step."""
        resolved, why = resolve_inputs(step, report.outputs)
        if resolved is None:
            if control is not None:
                control.step(step.index, state="skipped", detail=why)
            return StepResult(step.index, step.tool, False, False, 0, why, step.step_id,
                              skipped=True)
        result = await self._run_recorded(resolved, max_retries, control)
        if resolved.shortened:
            result = replace(result, detail=(
                f"{result.detail} (the result of step '{resolved.shortened[0]}' was "
                f"passed on shortened to {MAX_PASSED_CHARS:,} characters)"))
        return result

    def _progress(self, control: JobControl | None, report: PlanReport) -> None:
        if control is None:
            return
        total = max(1, len(control.store.steps(control.id)) or len(report.steps))
        control.store.update(control.id, fraction=min(1.0, len(report.steps) / total),
                             step=len(report.steps))

    def _count(self, result: StepResult) -> None:
        if self.telemetry is not None and not result.skipped:
            self.telemetry.metrics.incr("plan.step" if result.ok else "plan.step_failed")

    async def _run_recorded(self, step: _Step, max_retries: int,
                            control: JobControl | None) -> StepResult:
        gate = getattr(control, "approval_gate", None) if control is not None else None
        if gate and step.index in gate:
            # The resumed plan has been admitted: use up the approval for this
            # exact step now, immediately before its dispatch, or not at all.
            tool, args = gate.pop(step.index)
            approved, why = control.store.consume_approval(control.id, step.index, tool, args)
            if not approved:
                detail = f"skipped — not re-run: {why}"
                control.step(step.index, state="skipped", detail=detail)
                return StepResult(step.index, step.tool, False, False, 0, detail,
                                  step.step_id, skipped=True)
        if control is not None:
            control.step(step.index, state="running")
        result = await self._run_step(step, max_retries)
        if control is not None:
            control.step(step.index,
                         state="skipped" if result.skipped else
                         ("completed" if result.ok else "failed"),
                         attempts=result.attempts, detail=result.detail)
        return result

    async def _run_step(self, step: _Step, max_retries: int) -> StepResult:
        if step.tool in FORBIDDEN_IN_PLAN:      # validation refuses these; belt and braces
            return StepResult(step.index, step.tool, False, False, 0,
                              "skipped — not permitted inside a plan", step.step_id,
                              skipped=True)
        attempts = 0
        last_detail = ""
        # Failure can follow a partial effect (e.g. SMTP accepts one recipient
        # but refuses another). Neither ok=False nor an exception establishes
        # that repeating a mutation is safe. Only read-only steps retry here.
        idempotent = step.side_effect == READ_ONLY
        observation = str(step.args.get("action") or "").strip().lower() in {
            "desktop_control": {"list_windows", "windows"},
            "web_control": {"read", "read_page"},
        }.get(step.tool, set())
        visual = (step.tool in VISUAL_TOOLS and self.verifier is not None
                  and not idempotent and not observation)
        for attempt in range(1, max(1, max_retries) + 1):
            attempts = attempt
            unknown_outcome = False
            try:
                if visual:
                    verified, detail, text = await self._run_visual(step.tool, step.args)
                    last_detail = detail
                    if verified:
                        return StepResult(step.index, step.tool, True, True, attempts,
                                          detail, step.step_id,
                                          output=text[:_KEPT_OUTPUT_CHARS])
                    # A missing pixel change cannot establish that a click or
                    # write did not happen, even if a nested verifier failed.
                    unknown_outcome = True
                else:
                    result = await self.dispatch(step.tool, step.args)
                    last_detail = first_line(result.text, 140)
                    if result.ok:
                        return StepResult(step.index, step.tool, True, False, attempts,
                                          last_detail, step.step_id,
                                          output=str(result.text or "")[:_KEPT_OUTPUT_CHARS])
                    unknown_outcome = not idempotent
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_detail = first_line(exc, 120)
                unknown_outcome = True
            if unknown_outcome and not idempotent:
                last_detail = f"{last_detail} (not retried: it may already have taken effect)"
                break
            if attempt < max_retries:
                self.bus.log.emit(f"PLAN: step {step.index} ({step.tool}) unverified; retrying "
                                  f"({attempt}/{max_retries}).")
                await asyncio.sleep(0.4)
        return StepResult(step.index, step.tool, False, False, attempts,
                          last_detail or "no result", step.step_id)

    async def _run_visual(self, tool: str, args: dict[str, Any]) -> tuple[bool, str, str]:
        """Run a screen-affecting tool and confirm the display actually changed.
        Returns (changed, detail, the tool's full result text)."""
        before = await self.verifier._capture(None)
        result = await self.dispatch(tool, args)
        if not result.ok:
            return False, first_line(result.text, 120), str(result.text or "")
        await asyncio.sleep(0.4)
        after = await self.verifier._capture(None)
        ratio = self.verifier._change_ratio(before, after)
        detail = f"{first_line(result.text, 90)} [{ratio*100:.2f}% screen change]"
        return (ratio >= 0.002), detail, str(result.text or "")

    # ── resuming after a restart ─────────────────────────────────────────────

    async def resume_job(self, record: dict[str, Any]) -> ToolResult:
        """Continue an INTERRUPTED plan without repeating anything it did.

        Completed steps stay completed. A read-only step that was in flight
        runs again (it is idempotent). A side-effecting step that was in
        flight may already have happened, so it needs an approval bound to
        exactly that step before it runs again; if the approval is declined it
        is skipped, and so is everything that depends on it.
        """
        job_id = record["id"]
        if record.get("state") not in {State.INTERRUPTED.value, State.WAITING_FOR_APPROVAL.value}:
            return ToolResult(f"{job_id} is {record.get('state')}; nothing to resume.", ok=False)
        if job_id in CONTROLS:
            return ToolResult(f"{job_id} is already running.", ok=False)
        spec = record.get("args") or {}
        if not record.get("args_complete", True) or not isinstance(spec.get("steps"), list):
            return ToolResult(f"{job_id}'s plan was too large to keep; submit it again.",
                              ok=False)
        parsed, problems = validate_plan(spec["steps"], self.catalogue())
        if problems:
            return ToolResult(f"{job_id} can no longer run as written:\n"
                              + "\n".join(f"  - {p}" for p in problems[:8]), ok=False)
        store = self.store
        rows = store.steps(job_id)
        recorded = {s["idx"]: s for s in rows}
        # Missing or mismatched progress is uncertainty about an effect, not
        # evidence that the step never ran. Refuse before consuming approvals
        # or admitting any new execution.
        if (len(rows) != len(parsed) or len(recorded) != len(parsed)
                or set(recorded) != {s.index for s in parsed}
                or any(recorded[s.index]["tool"] != s.tool
                       or _store_mod.args_digest(s.tool, recorded[s.index].get("args"))
                       != _store_mod.args_digest(s.tool, s.args)
                       or recorded[s.index]["side_effect"] != s.side_effect
                       or recorded[s.index]["state"] not in {
                           "pending", "running", "interrupted", "completed", "failed", "skipped", "cancelled"}
                       for s in parsed)):
            return ToolResult(f"{job_id}'s durable step evidence is missing or inconsistent; "
                              "review the recorded effects before recovery. Nothing was resumed.", ok=False)
        preset: dict[str, StepResult] = {}
        for step in parsed:
            row = recorded.get(step.index)
            if row is None or row["state"] in {"pending", "interrupted", "running"}:
                continue
            ok = row["state"] == "completed"
            preset[step.step_id] = StepResult(
                step.index, step.tool, ok, False, int(row["attempts"] or 0),
                f"(before restart) {row['detail']}".strip(), step.step_id,
                skipped=row["state"] == "skipped")
        in_flight = [s for s in parsed if recorded.get(s.index, {}).get("state")
                     in {"interrupted", "running"}]
        # Approvals are only CHECKED here. Each is used up at that step's own
        # dispatch, after admission: a resume that waits, is cancelled or
        # fails first never spends permission for a step it did not run, and
        # an approval that expires while queued is caught at the boundary.
        gated: dict[int, tuple[str, dict[str, Any]]] = {}
        for step in in_flight:
            if step.side_effect == READ_ONLY:
                continue
            approved, why = store.peek_approval(job_id, step.index, step.tool, step.args)
            if approved:
                gated[step.index] = (step.tool, step.args)
                continue
            latest = store.latest_approval(job_id, step.index)
            if latest is not None and latest["state"] == "denied":
                detail = "skipped — not re-run: it may already have happened"
                preset[step.step_id] = StepResult(step.index, step.tool, False, False, 0,
                                                  detail, step.step_id, skipped=True)
                store.set_step(job_id, step.index, state="skipped", detail=detail)
                continue
            if latest is None or latest["state"] != "pending":
                latest = store.request_approval(
                    job_id, step.index, step.tool, step.args,
                    reason=f"step {step.index} ({step.tool}) was running when ORION "
                           "stopped and may already have taken effect")
            return ToolResult(
                f"{job_id}: step {step.index} ({step.tool}) was running when ORION stopped, "
                "so it may already have happened. It will not run again without your "
                f"approval (approval {latest['id']}, valid for "
                f"{int(latest['expires_at'] - latest['requested_at'])}s). Approve it to "
                "run it again, or decline to skip it and the steps that depend on it; "
                "then resume the plan.", ok=False)
        control = JobControl(store, record, cls=WorkClass.INTERACTIVE,
                             label=str(record.get("label") or "plan"),
                             admission=self._admission)
        control.approval_gate = gated
        CONTROLS[control.id] = control
        try:
            control.set_state(State.QUEUED, note="resuming after a restart")
        except Exception as exc:
            # Nothing has been admitted or dispatched; the control must not be
            # left registered as live work it will never do.
            CONTROLS.pop(control.id, None)
            self.bus.log.emit(f"PLAN: could not record the resume of {job_id} — "
                              f"{first_line(exc, 160)}")
            return ToolResult(f"{job_id} could not be recorded as resuming, so nothing was "
                              "run. Check the log and try again.", ok=False)
        return await self._run_plan(control, parsed, str(spec.get("objective") or ""),
                                    int(spec.get("max_retries") or 2), preset)

    # ── parsing (kept for callers that parse one step) ───────────────────────

    def _parse_step(self, raw: Any) -> tuple[str, dict[str, Any], str, str, list[str]]:
        if not isinstance(raw, dict):
            return "", {}, "retry", "", []
        tool = str(raw.get("tool") or raw.get("name") or "").strip()
        args = raw.get("args")
        if not isinstance(args, dict):
            try:
                args = json.loads(str(raw.get("args_json") or "{}"))
            except Exception:
                args = {}
        on_fail = _ON_FAIL.get(str(raw.get("on_fail") or "retry").lower().strip(), "retry")
        step_id = str(raw.get("id") or raw.get("step_id") or "").strip()
        raw_after = raw.get("after") or raw.get("depends_on") or []
        if isinstance(raw_after, str):
            raw_after = [raw_after]
        after = [str(dep).strip() for dep in raw_after if str(dep).strip()] \
            if isinstance(raw_after, (list, tuple)) else []
        return tool, args if isinstance(args, dict) else {}, on_fail, step_id, after


class _PlanAborted(Exception):
    """Internal signal: an abort-policy step failed and the plan must stop."""


__all__ = ["FORBIDDEN_IN_PLAN", "MAX_PLAN_STEPS", "PlanExecutor", "PlanReport", "StepResult",
           "ToolCatalogue", "VISUAL_TOOLS", "validate_plan"]
