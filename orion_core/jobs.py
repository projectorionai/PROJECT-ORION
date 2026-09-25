"""
Background job manager (Track B).

Every tool call blocked the conversation. ``_handle_tool_call`` set
``tool_busy``, drained the microphone and held the turn in PROCESSING until the
tool returned — so asking ORION for a deep research run meant sitting in
silence until it finished, and a genuinely long task was effectively
unaskable. There was no way to say "start that, and let's carry on talking".

A job is a tool invocation that has been detached from the turn. It runs as a
task on the same event loop, the turn ends immediately with an acknowledgement,
and when the job finishes ORION announces it. The result is retained so it can
be collected by name afterwards.

Design constraints that matter:

  * **Only parallel-safe tools may be backgrounded.** A job runs concurrently
    with whatever the conversation does next, which is exactly the situation
    the concurrency classifier exists to reason about — so backgrounding uses
    the same table, and refuses anything that drives the machine. A detached
    ``open_app`` racing a foreground click is precisely the bug this must not
    introduce.
  * **Finished jobs are kept, not dropped.** The whole point is that the user
    is elsewhere when it completes; a result that evaporates on completion is
    useless.
  * **Failure is a result, not an exception.** A background task that dies
    silently is worse than one that never started.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from .bus import OrionBus
from .concurrency import ToolClass, classify
from .data import ToolResult
from .utils import first_line, utc_stamp

DEFAULT_MAX_ACTIVE = 4
RETAINED_JOBS = 40

# Progress is only worth emitting when it has actually moved.
PROGRESS_STEP = 0.02


class JobState(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self is not JobState.RUNNING


@dataclass
class Job:
    """One detached tool invocation."""

    id: str
    tool: str
    args: dict[str, Any]
    label: str
    state: JobState = JobState.RUNNING
    started_at: str = field(default_factory=utc_stamp)
    finished_at: str = ""
    duration_s: float = 0.0
    result: str = ""
    ok: bool = False
    # Progress, when the running work chooses to report any. ``fraction`` is
    # None until something reports one — an unknown fraction must not render as
    # 0%, which reads as "stuck" rather than "not saying".
    fraction: float | None = None
    note: str = ""
    _task: Any = field(default=None, repr=False, compare=False)
    _monotonic: float = field(default_factory=time.monotonic, repr=False)
    _reporter: Any = field(default=None, repr=False, compare=False)
    _last_emitted: float = field(default=-1.0, repr=False, compare=False)

    def _record_progress(self, fraction: float | None, note: str) -> bool:
        """Fold a progress report in, and broadcast only when it moved.

        Progress is held MONOTONIC: a report lower than the last one is kept as
        the note but not the number. Work that genuinely re-scopes downward is
        real, but a bar that slides backwards reads as a bug to everyone who
        sees it, and the note is the honest place to say what happened.
        """
        changed = False
        if fraction is not None:
            try:
                value = max(0.0, min(1.0, float(fraction)))
            except (TypeError, ValueError):
                value = None
            if value is not None and (self.fraction is None or value > self.fraction):
                self.fraction = value
                changed = True
        note = str(note or "").strip()[:160]
        if note and note != self.note:
            self.note = note
            changed = True
        if not changed:
            return False
        moved = self.fraction is not None and (
            self._last_emitted < 0 or self.fraction - self._last_emitted >= PROGRESS_STEP)
        if (moved or note) and self._reporter is not None:
            self._last_emitted = self.fraction if self.fraction is not None else -1.0
            self._reporter(self)
        return True

    @property
    def elapsed_s(self) -> float:
        return self.duration_s if self.state.finished else time.monotonic() - self._monotonic

    def progress_text(self) -> str:
        """Whatever is actually known about how far along this is."""
        parts: list[str] = []
        if self.fraction is not None:
            parts.append(f"{self.fraction * 100:.0f}%")
        if self.note:
            parts.append(self.note)
        return " — ".join(parts)

    def eta_s(self) -> float | None:
        """Seconds remaining, extrapolated linearly from progress so far.

        Deliberately absent below 5% — extrapolating from a sliver produces
        confident nonsense, and no estimate is better than a wrong one.
        """
        if self.state.finished or self.fraction is None or self.fraction < 0.05:
            return None
        elapsed = self.elapsed_s
        return max(0.0, elapsed / self.fraction - elapsed)

    def summary(self) -> str:
        if self.state is JobState.RUNNING:
            detail = self.progress_text()
            eta = self.eta_s()
            if eta is not None:
                detail = f"{detail}, about {eta:.0f}s left" if detail else f"about {eta:.0f}s left"
            return (f"[{self.id}] {self.label} — running for "
                    f"{self.elapsed_s:.0f}s" + (f" ({detail})" if detail else ""))
        mark = {"succeeded": "✓", "failed": "✗", "cancelled": "–"}[self.state.value]
        return f"[{self.id}] {mark} {self.label} ({self.duration_s:.1f}s)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "tool": self.tool, "label": self.label,
            "state": self.state.value, "ok": self.ok,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 2),
            "fraction": self.fraction, "note": self.note,
            "eta_s": self.eta_s(),
            "result": self.result[:2000],
        }


# The job a coroutine is currently running inside, if any.
#
# A context variable rather than a parameter threaded through ``dispatch``:
# progress is reported from deep inside a tool — a per-file loop, a per-section
# writer — and every layer between would otherwise need a reporter argument it
# does not care about. ``asyncio.create_task`` copies the current context, so a
# job's task carries this automatically across every await inside it, and code
# running in the foreground simply sees None and reports nothing.
_CURRENT_JOB: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "orion_current_job", default=None)


def report_progress(fraction: float | None = None, note: str = "") -> bool:
    """Report progress from inside a background job. True when it landed.

    Safe to call from anywhere, including the foreground, where it is a no-op —
    a tool should not have to know whether it was backgrounded to say how far
    along it is.
    """
    job = _CURRENT_JOB.get()
    if job is None or job.state.finished:
        return False
    return job._record_progress(fraction, note)


class JobManager:
    """Runs tool calls detached from the conversation turn."""

    def __init__(self, bus: OrionBus,
                 dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolResult]],
                 telemetry: Any | None = None,
                 max_active: int = DEFAULT_MAX_ACTIVE) -> None:
        self.bus = bus
        self.dispatch = dispatch
        self.telemetry = telemetry
        self.max_active = max(1, int(max_active))
        self._jobs: dict[str, Job] = {}
        self._counter = 0
        if telemetry is not None:
            try:
                telemetry.health.register("jobs")
            except Exception:
                pass

    # ── inspection ───────────────────────────────────────────────────────────

    def active(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.state is JobState.RUNNING]

    def jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def get(self, job_id: str) -> Job | None:
        key = str(job_id or "").strip()
        if key in self._jobs:
            return self._jobs[key]
        # Accept a tool name or a label fragment: the user says "is the
        # research done?", not "is job-3 done?".
        folded = key.lower()
        matches = [
            j for j in self._jobs.values()
            if folded and (folded in j.tool.lower() or folded in j.label.lower())
        ]
        return matches[-1] if matches else None

    def can_background(self, tool: str, args: dict[str, Any] | None = None) -> tuple[bool, str]:
        """Whether this call may be detached, and why not when it may not.

        *args* matters: a detached job runs alongside the conversation, so a
        read-only tool asked to take its write branch (``research start``,
        ``awareness add_task``) is refused exactly as a genuinely
        side-effecting tool would be.
        """
        if classify(tool, args) is not ToolClass.PARALLEL:
            return False, (
                f"'{tool}' changes state or drives the machine, so it must run "
                "in the conversation where you can see it — only read-only "
                "work can be backgrounded.")
        if len(self.active()) >= self.max_active:
            return False, (
                f"{self.max_active} background jobs are already running; let one "
                "finish first.")
        return True, ""

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self, tool: str, args: dict[str, Any] | None = None,
              label: str = "") -> Job | ToolResult:
        """Detach *tool* onto the event loop. Returns the Job, or a refusal."""
        tool = str(tool or "").strip()
        allowed, reason = self.can_background(tool, args)
        if not allowed:
            return ToolResult(reason, ok=False)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return ToolResult(
                "No event loop is running, so nothing can be backgrounded.", ok=False)

        self._counter += 1
        job = Job(
            id=f"job-{self._counter}",
            tool=tool,
            args=dict(args or {}),
            label=str(label or "").strip() or tool,
        )
        job._reporter = self._on_progress
        self._jobs[job.id] = job
        job._task = loop.create_task(self._run(job), name=f"orion-{job.id}")
        job._task.add_done_callback(lambda task: self._reconcile(job, task))
        self.bus.log.emit(f"JOB: {job.id} started in the background — {job.label}")
        self._broadcast(job)
        self._prune()
        return job

    def _reconcile(self, job: Job, task: Any) -> None:
        """Guarantee a finished task leaves its job in a terminal state.

        ``create_task`` only SCHEDULES the coroutine, so a job cancelled in the
        same tick it was started never enters ``_run`` at all — none of the
        handling in there executes, and the job would sit at RUNNING forever
        with no task behind it. This callback fires however the task ended.
        """
        if job.state is not JobState.RUNNING:
            return                      # _run already recorded the outcome
        if task.cancelled():
            job.state = JobState.CANCELLED
            job.result = "cancelled before it started"
        else:
            exc = task.exception()
            if exc is None:
                return                  # completed without setting state: nothing to do
            job.state = JobState.FAILED
            job.result = f"{type(exc).__name__}: {exc}"
        job.ok = False
        self._finish(job)

    async def _run(self, job: Job) -> None:
        # Set inside the coroutine so everything the tool awaits inherits it.
        _CURRENT_JOB.set(job)
        try:
            result = await self.dispatch(job.tool, job.args)
            job.ok = bool(result.ok)
            job.result = str(result.text or "")
            job.state = JobState.SUCCEEDED if job.ok else JobState.FAILED
        except asyncio.CancelledError:
            job.state = JobState.CANCELLED
            job.result = "cancelled before completion"
            job.ok = False
            self._finish(job)
            raise
        except Exception as exc:
            job.state = JobState.FAILED
            job.ok = False
            job.result = f"{type(exc).__name__}: {exc}"
        self._finish(job)

    def _finish(self, job: Job) -> None:
        job.duration_s = max(0.0, time.monotonic() - job._monotonic)
        job.finished_at = utc_stamp()
        job._task = None
        verdict = "finished" if job.ok else job.state.value
        self.bus.log.emit(
            f"JOB: {job.id} {verdict} after {job.duration_s:.1f}s — "
            f"{first_line(job.result, 140)}")
        self._broadcast(job)
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr(
                    "jobs.succeeded" if job.ok else "jobs.failed")
            except Exception:
                pass

    def _broadcast(self, job: Job) -> None:
        """Tell the interface a job changed, so completion can be announced."""
        try:
            self.bus.dashboard_event.emit("job", job.to_dict())
        except Exception:
            pass

    def _on_progress(self, job: Job) -> None:
        """A running job moved. Kept off the log: progress is for the display,
        and a line per percent would drown everything else in it."""
        self._broadcast(job)

    def set_progress(self, job_id: str, fraction: float | None = None,
                     note: str = "") -> bool:
        """Report progress against a named job from outside its own context.

        (Not ``report`` — that name already belongs to the whole-board summary
        below, and a silent override would have made one of them unreachable.)
        """
        job = self.get(job_id)
        if job is None or job.state.finished:
            return False
        return job._record_progress(fraction, note)

    def cancel(self, job_id: str) -> ToolResult:
        job = self.get(job_id)
        if job is None:
            return ToolResult(f"No job matches '{job_id}'.", ok=False)
        if job.state is not JobState.RUNNING or job._task is None:
            return ToolResult(f"{job.id} already {job.state.value}.", ok=False)
        job._task.cancel()
        return ToolResult(f"Cancelling {job.id} ({job.label}).")

    def _prune(self) -> None:
        """Keep the newest RETAINED_JOBS finished records; running ones stay."""
        finished = [j for j in self._jobs.values() if j.state.finished]
        excess = len(finished) - RETAINED_JOBS
        for job in finished[:max(0, excess)]:
            self._jobs.pop(job.id, None)

    # ── reporting ────────────────────────────────────────────────────────────

    def report(self) -> ToolResult:
        if not self._jobs:
            return ToolResult("No background jobs have run this session.")
        running = self.active()
        done = [j for j in self._jobs.values() if j.state.finished]
        lines: list[str] = []
        if running:
            lines.append(f"Running ({len(running)}):")
            lines.extend(f"  {j.summary()}" for j in running)
        if done:
            lines.append(f"Finished ({len(done)}):")
            lines.extend(f"  {j.summary()}" for j in done[-8:])
        return ToolResult("\n".join(lines))

    def collect(self, job_id: str) -> ToolResult:
        job = self.get(job_id)
        if job is None:
            return ToolResult(f"No job matches '{job_id}'.", ok=False)
        if job.state is JobState.RUNNING:
            return ToolResult(
                f"{job.id} ({job.label}) is still running — "
                f"{time.monotonic() - job._monotonic:.0f}s so far.", ok=False)
        header = f"{job.id} — {job.label} ({job.state.value}, {job.duration_s:.1f}s)"
        return ToolResult(f"{header}\n\n{job.result}", ok=job.ok)

    async def drain(self, timeout: float = 5.0) -> None:
        """Await outstanding jobs at shutdown, briefly, then let them go."""
        tasks = [j._task for j in self.active() if j._task is not None]
        if not tasks:
            return
        self.bus.log.emit(f"JOB: waiting up to {timeout:.0f}s for "
                          f"{len(tasks)} background job(s).")
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()

    def describe(self) -> dict[str, Any]:
        return {
            "active": len(self.active()),
            "max_active": self.max_active,
            "total": len(self._jobs),
        }


__all__ = ["DEFAULT_MAX_ACTIVE", "Job", "JobManager", "JobState", "PROGRESS_STEP",
           "RETAINED_JOBS", "report_progress"]
