"""
Self-verifying plan executor (improvements #1, #3).

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
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .bus import OrionBus
from .data import ToolResult
from .utils import first_line

# Tools whose success is judged by whether the screen actually changed.
VISUAL_TOOLS = {"desktop_control", "web_control", "vision_verify"}


@dataclass
class StepResult:
    index: int
    tool: str
    ok: bool
    verified: bool
    attempts: int
    detail: str


@dataclass
class PlanReport:
    objective: str
    steps: list[StepResult] = field(default_factory=list)
    aborted: bool = False

    @property
    def succeeded(self) -> int:
        return sum(1 for s in self.steps if s.ok)

    @property
    def verified(self) -> int:
        return sum(1 for s in self.steps if s.verified)

    def to_tool_result(self) -> ToolResult:
        lines = [f"PLAN: {self.objective}" if self.objective else "PLAN executed:"]
        for s in self.steps:
            mark = "✓" if s.ok else "✗"
            v = "verified" if s.verified else "unverified"
            lines.append(f"  {mark} [{s.index}] {s.tool} ({v}, {s.attempts} try): {s.detail[:110]}")
        lines.append(
            f"RESULT: {self.succeeded}/{len(self.steps)} succeeded, "
            f"{self.verified} visually verified"
            + (" — plan aborted early." if self.aborted else ".")
        )
        return ToolResult("\n".join(lines), ok=(not self.aborted and self.succeeded == len(self.steps)))


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
        report = PlanReport(objective=str(objective or "").strip())
        self.bus.log.emit(f"PLAN: executing {len(steps)} step(s) with self-verification.")
        for i, raw in enumerate(list(steps)[:12], 1):
            tool, args, on_fail = self._parse_step(raw)
            if not tool:
                continue
            if tool in {"shutdown_orion", "self_repair"}:
                report.steps.append(StepResult(i, tool, False, False, 0,
                                               "skipped — not permitted inside a plan"))
                continue
            step_result = await self._run_step(i, tool, args, max_retries)
            report.steps.append(step_result)
            if not step_result.ok and on_fail == "abort":
                report.aborted = True
                self.bus.log.emit(f"PLAN: aborted at step {i} ({tool}).")
                break
            if self.telemetry is not None:
                self.telemetry.metrics.incr("plan.step" if step_result.ok else "plan.step_failed")
        if self.telemetry is not None:
            self.telemetry.metrics.incr("plan.executed")
        return report.to_tool_result()

    async def _run_step(self, index: int, tool: str, args: dict[str, Any],
                        max_retries: int) -> StepResult:
        attempts = 0
        last_detail = ""
        visual = tool in VISUAL_TOOLS and self.verifier is not None
        for attempt in range(1, max(1, max_retries) + 1):
            attempts = attempt
            try:
                if visual:
                    verified, detail = await self._run_visual(tool, args)
                    last_detail = detail
                    if verified:
                        return StepResult(index, tool, True, True, attempts, detail)
                else:
                    result = await self.dispatch(tool, args)
                    last_detail = first_line(result.text, 140)
                    if result.ok:
                        return StepResult(index, tool, True, False, attempts, last_detail)
            except Exception as exc:
                last_detail = first_line(exc, 120)
            if attempt < max_retries:
                self.bus.log.emit(f"PLAN: step {index} ({tool}) unverified; retrying "
                                  f"({attempt}/{max_retries}).")
                await asyncio.sleep(0.4)
        return StepResult(index, tool, False, False, attempts, last_detail or "no result")

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

    def _parse_step(self, raw: Any) -> tuple[str, dict[str, Any], str]:
        if not isinstance(raw, dict):
            return "", {}, "retry"
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
        return tool, args if isinstance(args, dict) else {}, on_fail
