"""
Self-verifying plan executor (improvements #1, #3; Track B: concurrency).

``PlanExecutor`` runs a multi-step autonomous plan and verifies each step
before moving on, with bounded retries — so ORION's autonomy is reliable
rather than fire-and-forget.

Verification strategy per step:
    • desktop_control / web_control / vision_verify — a *visual* check: the
      screen must have changed after the action (pixel-diff via the
      VisualVerificationEngine).  If it did not, the step is retried.
    • every other tool — the tool's own ``ToolResult.ok`` is the verdict.

Each step may declare an ``on_fail`` policy: ``retry`` (default, bounded),
``continue`` (log and move on) or ``abort`` (stop the plan).  A consolidated,
honest report is returned: what was attempted, what verified, what failed and
why.  This runs entirely on the qasync loop; the verification captures happen
off-thread inside the verification engine.

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
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .bus import OrionBus
from .concurrency import ToolClass, classify, max_parallel, parallelism_enabled
from .data import ToolResult
from .utils import first_line

# Tools whose success is judged by whether the screen actually changed.
VISUAL_TOOLS = {"desktop_control", "web_control", "vision_verify"}

MAX_PLAN_STEPS = 24


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


@dataclass
class PlanReport:
    objective: str
    steps: list[StepResult] = field(default_factory=list)
    aborted: bool = False
    schedule: str = ""

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
        if self.skipped:
            summary += f", {self.skipped} skipped on a failed dependency"
        summary += " — plan aborted early." if self.aborted else "."
        lines.append(summary)
        if self.schedule:
            lines.append(f"SCHEDULE: {self.schedule}")
        return ToolResult(
            "\n".join(lines),
            ok=(not self.aborted and self.succeeded == len(self.steps)),
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

    @property
    def batchable(self) -> bool:
        """Safe to run alongside other steps in the same wave.

        Visual steps are excluded whatever their class: their verdict is a
        pixel diff of the whole screen, so two running at once would each
        measure the other's changes.
        """
        return (classify(self.tool, self.args) is ToolClass.PARALLEL
                and self.tool not in VISUAL_TOOLS)


class PlanExecutor:
    def __init__(self, bus: OrionBus, dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolResult]],
                 verifier: Any | None = None, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.dispatch = dispatch
        self.verifier = verifier
        self.telemetry = telemetry

    async def execute(self, steps: list[dict[str, Any]], objective: str = "",
                      max_retries: int = 2) -> ToolResult:
        if not isinstance(steps, list) or not steps:
            return ToolResult("No plan steps supplied.", ok=False)
        parsed = self._parse_plan(list(steps)[:MAX_PLAN_STEPS])
        if not parsed:
            return ToolResult("No runnable steps in the plan.", ok=False)

        report = PlanReport(objective=str(objective or "").strip())
        has_dag = any(step.after for step in parsed)
        self.bus.log.emit(
            f"PLAN: executing {len(parsed)} step(s) with self-verification "
            f"({'dependency graph' if has_dag else 'linear'})."
        )
        try:
            if has_dag:
                await self._execute_dag(parsed, report, max_retries)
            else:
                await self._execute_linear(parsed, report, max_retries)
        except _PlanAborted:
            report.aborted = True
        if self.telemetry is not None:
            self.telemetry.metrics.incr("plan.executed")
        return report.to_tool_result()

    # ── linear scheduling (order preserved, read-only neighbours batched) ─────

    async def _execute_linear(self, steps: list[_Step], report: PlanReport,
                              max_retries: int) -> None:
        ceiling = max_parallel() if parallelism_enabled() else 1
        batch: list[_Step] = []
        parallel_batches = 0

        async def _flush() -> None:
            nonlocal batch, parallel_batches
            if not batch:
                return
            if len(batch) > 1:
                parallel_batches += 1
            await self._run_wave(batch, report, max_retries)
            batch = []

        for step in steps:
            if step.batchable and ceiling > 1:
                batch.append(step)
                if len(batch) >= ceiling:
                    await _flush()
                continue
            await _flush()
            await self._run_wave([step], report, max_retries)
        await _flush()
        if parallel_batches:
            report.schedule = f"{parallel_batches} batch(es) ran in parallel"

    # ── dependency-graph scheduling ──────────────────────────────────────────

    async def _execute_dag(self, steps: list[_Step], report: PlanReport,
                           max_retries: int) -> None:
        ceiling = max_parallel() if parallelism_enabled() else 1
        by_id = {step.step_id: step for step in steps}
        pending = {step.step_id: step for step in steps}
        done: dict[str, bool] = {}          # step_id → succeeded
        waves = 0

        # An edge to a step that does not exist can never be satisfied; drop it
        # rather than deadlocking the whole plan on a typo.
        for step in steps:
            unknown = [dep for dep in step.after if dep not in by_id]
            for dep in unknown:
                step.after.remove(dep)
                self.bus.log.emit(
                    f"PLAN: step '{step.step_id}' depends on unknown step "
                    f"'{dep}' — edge ignored.")

        while pending:
            ready = [s for s in pending.values()
                     if all(dep in done for dep in s.after)]
            if not ready:
                # Every remaining step waits on another remaining step.
                cycle = ", ".join(sorted(pending))
                self.bus.log.emit(f"PLAN: dependency cycle among [{cycle}] — stopping.")
                for step in pending.values():
                    report.steps.append(StepResult(
                        step.index, step.tool, False, False, 0,
                        "skipped — dependency cycle", step.step_id, skipped=True))
                report.aborted = True
                break

            # A step whose prerequisite failed must not run: its precondition
            # never happened, so running it anyway is how a plan does damage.
            blocked = [s for s in ready if not all(done.get(d) for d in s.after)]
            for step in blocked:
                failed = [d for d in step.after if not done.get(d)]
                pending.pop(step.step_id, None)
                done[step.step_id] = False
                report.steps.append(StepResult(
                    step.index, step.tool, False, False, 0,
                    f"skipped — dependency '{failed[0]}' failed", step.step_id,
                    skipped=True))
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

            results = await self._run_wave(wave, report, max_retries)
            for step, result in zip(wave, results):
                pending.pop(step.step_id, None)
                done[step.step_id] = result.ok
                if not result.ok and step.on_fail == "abort":
                    self.bus.log.emit(f"PLAN: aborted at '{step.step_id}' ({step.tool}).")
                    for remaining in pending.values():
                        report.steps.append(StepResult(
                            remaining.index, remaining.tool, False, False, 0,
                            "skipped — plan aborted", remaining.step_id, skipped=True))
                    report.aborted = True
                    pending.clear()
                    break
        if waves:
            report.schedule = f"{waves} dependency wave(s) ran in parallel"

    # ── execution ────────────────────────────────────────────────────────────

    async def _run_wave(self, wave: list[_Step], report: PlanReport,
                        max_retries: int) -> list[StepResult]:
        """Run one batch of steps, concurrently when there is more than one."""
        if len(wave) == 1:
            step = wave[0]
            result = await self._run_step(step, max_retries)
            report.steps.append(result)
            self._count(result)
            if not result.ok and step.on_fail == "abort":
                self.bus.log.emit(f"PLAN: aborted at step {step.index} ({step.tool}).")
                report.aborted = True
                raise _PlanAborted()
            return [result]

        self.bus.log.emit(
            "PLAN: running " + " ‖ ".join(s.tool for s in wave) + " concurrently.")
        gathered = await asyncio.gather(
            *(self._run_step(step, max_retries) for step in wave),
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
            self._count(outcome)
        # A batch is only entered by steps that cannot interfere, so an abort
        # policy is honoured after the whole batch rather than mid-flight.
        for step, result in zip(wave, results):
            if not result.ok and step.on_fail == "abort":
                self.bus.log.emit(f"PLAN: aborted at step {step.index} ({step.tool}).")
                report.aborted = True
                raise _PlanAborted()
        return results

    def _count(self, result: StepResult) -> None:
        if self.telemetry is not None and not result.skipped:
            self.telemetry.metrics.incr("plan.step" if result.ok else "plan.step_failed")

    async def _run_step(self, step: _Step, max_retries: int) -> StepResult:
        if step.tool in {"shutdown_orion", "self_repair"}:
            return StepResult(step.index, step.tool, False, False, 0,
                              "skipped — not permitted inside a plan", step.step_id,
                              skipped=True)
        attempts = 0
        last_detail = ""
        visual = step.tool in VISUAL_TOOLS and self.verifier is not None
        for attempt in range(1, max(1, max_retries) + 1):
            attempts = attempt
            try:
                if visual:
                    verified, detail = await self._run_visual(step.tool, step.args)
                    last_detail = detail
                    if verified:
                        return StepResult(step.index, step.tool, True, True, attempts,
                                          detail, step.step_id)
                else:
                    result = await self.dispatch(step.tool, step.args)
                    last_detail = first_line(result.text, 140)
                    if result.ok:
                        return StepResult(step.index, step.tool, True, False, attempts,
                                          last_detail, step.step_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_detail = first_line(exc, 120)
            if attempt < max_retries:
                self.bus.log.emit(f"PLAN: step {step.index} ({step.tool}) unverified; retrying "
                                  f"({attempt}/{max_retries}).")
                await asyncio.sleep(0.4)
        return StepResult(step.index, step.tool, False, False, attempts,
                          last_detail or "no result", step.step_id)

    async def _run_visual(self, tool: str, args: dict[str, Any]) -> tuple[bool, str]:
        """Run a screen-affecting tool and confirm the display actually changed."""
        before = await self.verifier._capture(None)
        result = await self.dispatch(tool, args)
        if not result.ok:
            return False, first_line(result.text, 120)
        await asyncio.sleep(0.4)
        after = await self.verifier._capture(None)
        ratio = self.verifier._change_ratio(before, after)
        detail = f"{first_line(result.text, 90)} [{ratio*100:.2f}% screen change]"
        return (ratio >= 0.002), detail

    # ── parsing ──────────────────────────────────────────────────────────────

    def _parse_plan(self, raw_steps: list[Any]) -> list[_Step]:
        parsed: list[_Step] = []
        used_ids: set[str] = set()
        for i, raw in enumerate(raw_steps, 1):
            tool, args, on_fail, step_id, after = self._parse_step(raw)
            if not tool:
                continue
            if not step_id or step_id in used_ids:
                step_id = f"step_{i}"
            used_ids.add(step_id)
            parsed.append(_Step(index=i, step_id=step_id, tool=tool, args=args,
                                on_fail=on_fail, after=after))
        return parsed

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
        on_fail = str(raw.get("on_fail") or "retry").lower().strip()
        if on_fail not in {"retry", "continue", "abort"}:
            on_fail = "retry"
        step_id = str(raw.get("id") or raw.get("step_id") or "").strip()
        raw_after = raw.get("after") or raw.get("depends_on") or []
        if isinstance(raw_after, str):
            raw_after = [raw_after]
        after = [str(dep).strip() for dep in raw_after if str(dep).strip()] \
            if isinstance(raw_after, (list, tuple)) else []
        return tool, args if isinstance(args, dict) else {}, on_fail, step_id, after


class _PlanAborted(Exception):
    """Internal signal: an abort-policy step failed and the plan must stop."""


__all__ = ["MAX_PLAN_STEPS", "PlanExecutor", "PlanReport", "StepResult", "VISUAL_TOOLS"]
