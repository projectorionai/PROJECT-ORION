"""
Background job manager (Track B) — and the one control surface for every job.

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

One owner (architecture review 2026-09-26, finding 6)
-----------------------------------------------------
Background jobs, workflow runs and plans each had their own lifecycle, ids,
limits and cancellation. They now share:

  * the durable record in ``job_store`` (uuid ids, legal state transitions,
    restart marks unfinished work INTERRUPTED, approvals bound to one step);
  * ``admission`` for when they may run — a job beyond the running limit is
    QUEUED and visible, past the pending limit it is refused with a reason;
  * ``JobControl`` for cooperative cancel, pause/resume and reprioritisation.

Cancelling an asyncio task does not stop a native call already running on a
thread. Long work therefore calls ``checkpoint()`` (or ``checkpoint_sync()``
on a worker thread) between units of work: that is where a cancel lands, and
where a pause releases its admission slot until it is resumed.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from . import admission as _admission_mod
from . import job_store as _store_mod
from .admission import Busy, WorkClass, work_class
from .bus import OrionBus
from .concurrency import ToolClass, classify
from .data import ToolResult
from .job_store import JobStore, State
from .utils import first_line, utc_stamp

#: Historical ceiling; the machine-wide USER limit in ``admission`` now governs.
DEFAULT_MAX_ACTIVE = 4
RETAINED_JOBS = 40

# Progress is only worth emitting when it has actually moved.
PROGRESS_STEP = 0.02


def approval_step_number(index: Any, kind: str) -> int:
    """Plans/workflows record one-based steps; tool jobs record zero-based steps."""
    return int(index) if kind in {"plan", "workflow"} else int(index) + 1


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    #: The old name for COMPLETED, kept so existing callers still read.
    SUCCEEDED = "completed"

    @property
    def finished(self) -> bool:
        """No longer running in this process (done, failed, cancelled, or
        interrupted by a restart)."""
        return self.value in {"completed", "failed", "cancelled", "interrupted"}

    @property
    def active(self) -> bool:
        return not self.finished


class JobCancelled(Exception):
    """Raised at a checkpoint once cancellation of the job was requested."""


# ── cooperative control ───────────────────────────────────────────────────────

#: Every job currently owned by this process, by id — tool jobs, workflow runs
#: and plans alike. Entries leave when the job reaches a final state.
CONTROLS: dict[str, "JobControl"] = {}

_CURRENT_CONTROL: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "orion_current_job_control", default=None)

#: kind -> callable(record) that resumes an INTERRUPTED job of that kind.
_RESUMERS: dict[str, Callable[[dict[str, Any]], Any]] = {}


def register_resumer(kind: str, resumer: Callable[[dict[str, Any]], Any]) -> None:
    """Say how to resume an interrupted job of *kind* (workflow, plan, tool)."""
    _RESUMERS[str(kind)] = resumer


class JobControl:
    """The live handle on one job: its record, its admission slot, and the
    flags a checkpoint obeys."""

    def __init__(self, store: JobStore, record: dict[str, Any], *, cls: Any = WorkClass.USER,
                 label: str = "", admission: Any = None) -> None:
        self.store = store
        self.id = str(record["id"])
        self.kind = str(record.get("kind") or "")
        self.cls = work_class(cls)
        self.label = str(label or record.get("label") or record.get("name") or "")
        self.state = State(record["state"])
        self.cancel_requested = False
        self.ticket: Any = None
        #: The task to cancel as well, when cancelling it is safe (a detached
        #: job or workflow run). Never set for a plan: that runs inside the
        #: conversation's own tool call, which must not be cancelled from here.
        self.task: Any = None
        self.on_state: Callable[[State], Any] | None = None
        self._run_gate = threading.Event()
        self._run_gate.set()
        self._admission_override = admission
        #: Step index -> (tool, args) whose restart approval was checked but
        #: not yet used: a resumed plan consumes each at that step's dispatch.
        self.approval_gate: dict[int, tuple[str, Any]] = {}
        #: Set by suspend_all() at shutdown: the record already says
        #: INTERRUPTED, and the cancellation that follows must not turn that
        #: into CANCELLED (which could never be resumed).
        self.frozen = False

    # ── creation ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls_, *, kind: str, name: str, label: str = "", cls: Any = WorkClass.USER,
               args: Any = None, steps: Any = (), origin: str = "", total: int | None = None,
               store: JobStore | None = None, admission: Any = None,
               ticket: Any = None) -> "JobControl":
        store = store if store is not None else _store_mod.default_store()
        klass = work_class(cls)
        state = State.QUEUED if ticket is None or not ticket.running else State.RUNNING
        try:
            record = store.create(kind=kind, name=name, label=label or name, cls=klass.label,
                                  state=state, args=args, origin=origin, steps=steps, total=total)
        except BaseException:
            # No control exists to own the slot, so it is given back HERE,
            # exactly once - callers report the failure but do not release.
            if ticket is not None:
                ticket.release()
            raise
        control = cls_(store, record, cls=klass, label=label or name, admission=admission)
        control.ticket = ticket
        CONTROLS[control.id] = control
        return control

    def _admission(self) -> Any:
        return self._admission_override or _admission_mod.ADMISSION

    # ── state ───────────────────────────────────────────────────────────────

    def set_state(self, state: State | str, **fields: Any) -> bool:
        if self.frozen:
            return False
        record = self.store.transition(self.id, state, **fields)
        if record is None:
            return False
        changed = State(record["state"]) is not self.state
        self.state = State(record["state"])
        if changed and self.on_state is not None:
            try:
                self.on_state(self.state)
            except Exception:
                pass
        return True

    @property
    def pause_requested(self) -> bool:
        return not self._run_gate.is_set()

    def position(self) -> int | None:
        ticket = self.ticket
        return ticket.position() if ticket is not None else None

    # ── admission ───────────────────────────────────────────────────────────

    def submit(self) -> Any:
        """Ask admission for a slot (never blocks; raises Busy)."""
        if self.ticket is None or self.ticket.state not in {"queued", "running"}:
            self.ticket = self._admission().submit(self.cls, label=self.label)
        if self.ticket.running and self.state is not State.RUNNING:
            self.set_state(State.RUNNING)
        return self.ticket

    async def admit(self) -> None:
        """Wait for a slot, honouring cancel and pause while queued."""
        while True:
            if self.cancel_requested:
                raise JobCancelled(f"{self.label} was cancelled before it started.")
            if self.pause_requested:
                await self._wait_paused()
                if not self.cancel_requested and self.state is State.PAUSED:
                    self.set_state(State.QUEUED, note="resuming")
                continue
            ticket = self.submit()
            if ticket.running:
                return
            position = ticket.position()
            if position:
                self.store.update(self.id, note=f"queued — {position} in line")
            try:
                await ticket.wait()
            except _admission_mod.Superseded:
                raise
            except Busy:
                if self.pause_requested or self.cancel_requested:
                    continue            # withdrawn by pause() / cancel(): loop decides
                raise
            self.set_state(State.RUNNING, note="")
            return

    def release(self) -> None:
        ticket, self.ticket = self.ticket, None
        if ticket is not None:
            ticket.release()

    # ── checkpoints ─────────────────────────────────────────────────────────

    async def checkpoint(self) -> None:
        """Where a cancel lands and a pause takes effect (async work)."""
        if self.cancel_requested:
            raise JobCancelled(f"{self.label} was cancelled.")
        if self.pause_requested:
            self.release()
            await self._wait_paused()
            if self.cancel_requested:
                raise JobCancelled(f"{self.label} was cancelled while paused.")
            self.set_state(State.QUEUED, note="resuming")
            await self.admit()

    async def _wait_paused(self) -> None:
        if self.state is not State.PAUSED:
            self.set_state(State.PAUSED, note="paused")
        while self.pause_requested and not self.cancel_requested:
            await asyncio.sleep(0.05)

    def checkpoint_sync(self) -> None:
        """The same for work on a worker thread (it may block; it is not the loop)."""
        if self.cancel_requested:
            raise JobCancelled(f"{self.label} was cancelled.")
        if self.pause_requested:
            self.release()
            if self.state is not State.PAUSED:
                self.set_state(State.PAUSED, note="paused")
            while self.pause_requested and not self.cancel_requested:
                self._run_gate.wait(0.1)
            if self.cancel_requested:
                raise JobCancelled(f"{self.label} was cancelled while paused.")
            self.set_state(State.QUEUED, note="resuming")
            ticket = self.submit()
            ticket.wait_blocking()
            self.set_state(State.RUNNING, note="")

    # ── requests from outside ───────────────────────────────────────────────

    def pause(self) -> bool:
        if self.state.value in {"completed", "failed", "cancelled", "interrupted"}:
            return False
        self._run_gate.clear()
        ticket = self.ticket
        if ticket is not None and ticket.queued:
            # Not started yet: step out of the queue now rather than hold a place.
            self.release()
            self.set_state(State.PAUSED, note="paused")
        return True

    def resume(self) -> bool:
        if not self.pause_requested:
            return False
        self._run_gate.set()
        return True

    def cancel(self) -> bool:
        if self.state.value in {"completed", "failed", "cancelled"}:
            return False
        self.cancel_requested = True
        self._run_gate.set()
        ticket = self.ticket
        if ticket is not None and ticket.queued:
            self.release()
        task = self.task
        if task is not None and not task.done():
            task.cancel()
        return True

    def reprioritise(self, cls: Any) -> None:
        self.cls = work_class(cls)
        if self.ticket is not None:
            self._admission().reprioritise(self.ticket, self.cls)
        self.store.update(self.id, cls=self.cls.label)

    # ── progress and completion ─────────────────────────────────────────────

    def progress(self, fraction: float | None = None, note: str = "") -> None:
        if self.frozen:
            return
        fields: dict[str, Any] = {}
        if fraction is not None:
            fields["fraction"] = fraction
        if note:
            fields["note"] = note
        if fields:
            self.store.update(self.id, **fields)

    def step(self, idx: int, **fields: Any) -> None:
        if not self.frozen:
            self.store.set_step(self.id, idx, **fields)

    def finish(self, state: State | str, *, ok: bool = False, result: str = "",
               **fields: Any) -> None:
        try:
            self.release()
            self.set_state(state, ok=ok, result=result, **fields)
        finally:
            # A failing durable transition still raises to the caller, but it
            # must not leave a dead control registered as live work.
            CONTROLS.pop(self.id, None)


async def suspend_all(timeout: float = 2.0,
                      note: str = "ORION was shut down while this was in progress") -> int:
    """Shutdown: record every live job as INTERRUPTED — resumable next start,
    exactly as if the process had died, but written deliberately — then
    cancel what can be cancelled and wait (bounded) for it to unwind, so no
    task is left "destroyed but pending" and no admission slot stays held.
    Returns how many jobs were suspended."""
    controls = list(CONTROLS.values())
    tasks = []
    for control in controls:
        try:
            control.store.mark_interrupted(control.id, note)
        except Exception:
            pass
        control.frozen = True
        control.cancel()
        if control.task is not None and not control.task.done():
            tasks.append(control.task)
    if tasks:
        await asyncio.wait(tasks, timeout=max(0.0, timeout))
    for control in controls:
        control.release()
        CONTROLS.pop(control.id, None)
    return len(controls)


def find_control(job_id: str) -> JobControl | None:
    key = str(job_id or "").strip()
    if key in CONTROLS:
        return CONTROLS[key]
    folded = key.lower()
    matches = [c for c in CONTROLS.values()
               if folded and (c.id.startswith(folded) or folded in c.label.lower())]
    return matches[-1] if matches else None


def current_control() -> JobControl | None:
    """The job the calling code is running inside, if any."""
    job = _CURRENT_JOB.get()
    control = getattr(job, "_control", None) if job is not None else None
    return control or _CURRENT_CONTROL.get()


async def checkpoint(fraction: float | None = None, note: str = "") -> None:
    """Cooperative checkpoint for long async work: reports progress, and is
    where a cancel (JobCancelled) or a pause lands. A no-op outside a job."""
    if fraction is not None or note:
        report_progress(fraction, note)
    control = current_control()
    if control is not None:
        await control.checkpoint()


def checkpoint_sync(fraction: float | None = None, note: str = "") -> None:
    """``checkpoint`` for code on a worker thread (``asyncio.to_thread`` and
    the bulk pool copy the context, so the job is known there too)."""
    if fraction is not None or note:
        report_progress(fraction, note)
    control = current_control()
    if control is not None:
        control.checkpoint_sync()


# ── background tool jobs ──────────────────────────────────────────────────────

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
    cls: str = "user"
    _task: Any = field(default=None, repr=False, compare=False)
    _monotonic: float = field(default_factory=time.monotonic, repr=False)
    _reporter: Any = field(default=None, repr=False, compare=False)
    _last_emitted: float = field(default=-1.0, repr=False, compare=False)
    _control: Any = field(default=None, repr=False, compare=False)

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
        if self.state is JobState.QUEUED:
            position = self._control.position() if self._control is not None else None
            return (f"[{self.id}] {self.label} — queued"
                    + (f" ({position} in line)" if position else ""))
        if self.state is JobState.PAUSED:
            return f"[{self.id}] {self.label} — paused"
        if self.state.active:
            detail = self.progress_text()
            eta = self.eta_s()
            if eta is not None:
                detail = f"{detail}, about {eta:.0f}s left" if detail else f"about {eta:.0f}s left"
            return (f"[{self.id}] {self.label} — running for "
                    f"{self.elapsed_s:.0f}s" + (f" ({detail})" if detail else ""))
        mark = {"completed": "✓", "failed": "✗", "cancelled": "–",
                "interrupted": "!"}[self.state.value]
        return f"[{self.id}] {mark} {self.label} ({self.duration_s:.1f}s)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "tool": self.tool, "label": self.label,
            "state": self.state.value, "ok": self.ok, "class": self.cls,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 2),
            "fraction": self.fraction, "indeterminate": self.fraction is None,
            "note": self.note,
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
    """Runs tool calls detached from the conversation turn, and is the one
    place a status view asks about, pauses, resumes, reprioritises, approves
    or cancels ANY job (tool, workflow or plan)."""

    def __init__(self, bus: OrionBus,
                 dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolResult]],
                 telemetry: Any | None = None,
                 max_active: int | None = None, *,
                 store: JobStore | None = None, admission: Any = None) -> None:
        self.bus = bus
        self.dispatch = dispatch
        self.telemetry = telemetry
        self._store = store
        self._admission_override = admission
        #: Reported, not enforced here: the machine-wide USER running limit in
        #: ``admission`` decides how many jobs run; the rest queue visibly.
        self.max_active = max(1, int(max_active)) if max_active else \
            self._admission().policy[WorkClass.USER].running
        self._jobs: dict[str, Job] = {}
        register_resumer("tool", self._resume_tool_job)
        if telemetry is not None:
            try:
                telemetry.health.register("jobs")
            except Exception:
                pass

    @property
    def store(self) -> JobStore:
        return self._store if self._store is not None else _store_mod.default_store()

    def _admission(self) -> Any:
        return self._admission_override or _admission_mod.ADMISSION

    # ── inspection ───────────────────────────────────────────────────────────

    def active(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.state.active]

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
            if folded and (j.id.startswith(folded) or folded in j.tool.lower()
                           or folded in j.label.lower())
        ]
        return matches[-1] if matches else None

    def can_background(self, tool: str, args: dict[str, Any] | None = None) -> tuple[bool, str]:
        """Whether this call may be detached, and why not when it may not.

        *args* matters: a detached job runs alongside the conversation, so a
        read-only tool asked to take its write branch (``research start``,
        ``awareness add_task``) is refused exactly as a genuinely
        side-effecting tool would be. How MANY may run is admission's call:
        past the running limit a job queues, visibly; past the pending limit
        it is refused with a reason.
        """
        if classify(tool, args) is not ToolClass.PARALLEL:
            return False, (
                f"'{tool}' changes state or drives the machine, so it must run "
                "in the conversation where you can see it — only read-only "
                "work can be backgrounded.")
        return True, ""

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self, tool: str, args: dict[str, Any] | None = None,
              label: str = "", *, priority: Any = WorkClass.USER,
              key: Any = None, replaceable: bool = False) -> Job | ToolResult:
        """Detach *tool* onto the event loop. Returns the Job (running or
        queued), or a refusal that says why."""
        tool = str(tool or "").strip()
        allowed, reason = self.can_background(tool, args)
        if not allowed:
            return ToolResult(reason, ok=False)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return ToolResult(
                "No event loop is running, so nothing can be backgrounded.", ok=False)
        klass = work_class(priority)
        if klass is WorkClass.INTERACTIVE:
            klass = WorkClass.USER          # detached work is never "the turn"
        label = str(label or "").strip() or tool
        call_args = dict(args or {})
        try:
            ticket = self._admission().submit(klass, key=key, replaceable=replaceable,
                                              label=label)
        except Busy as busy:
            if self.telemetry is not None:
                try:
                    self.telemetry.metrics.incr("jobs.refused")
                except Exception:
                    pass
            return ToolResult(
                f"{busy.reason} ({len(self.active())} background job(s) already "
                "running or queued.)", ok=False)
        try:
            control = JobControl.create(
                kind="tool", name=tool, label=label, cls=klass,
                args={"tool": tool, "args": call_args},
                steps=[{"tool": tool, "args": call_args}],
                store=self.store, admission=self._admission_override, ticket=ticket)
        except Exception as exc:
            # create() has already given the slot back.
            self.bus.log.emit(f"JOB: could not record '{label}' - {first_line(exc, 160)}")
            return ToolResult(f"'{label}' could not be recorded, so it was not started. "
                              "Check the log and try again.", ok=False)
        job = Job(id=control.id, tool=tool, args=call_args, label=label,
                  state=JobState.RUNNING if ticket.running else JobState.QUEUED,
                  cls=klass.label)
        job._control = control
        job._reporter = self._on_progress
        control.on_state = lambda state, job=job: self._on_state(job, state)
        self._jobs[job.id] = job
        job._task = loop.create_task(self._run(job), name=f"orion-{job.id}")
        control.task = job._task
        job._task.add_done_callback(lambda task: self._reconcile(job, task))
        if job.state is JobState.QUEUED:
            self.bus.log.emit(f"JOB: {job.id} queued ({ticket.position() or 1} in line) "
                              f"— {job.label}")
        else:
            self.bus.log.emit(f"JOB: {job.id} started in the background — {job.label}")
        self._broadcast(job)
        self._prune()
        return job

    def _on_state(self, job: Job, state: State) -> None:
        if job.state.finished:
            return
        job.state = JobState(state.value)
        if state is State.RUNNING and job._last_emitted < 0 and job.fraction is None:
            job._monotonic = time.monotonic()
        self._broadcast(job)

    def _reconcile(self, job: Job, task: Any) -> None:
        """Guarantee a finished task leaves its job in a terminal state.

        ``create_task`` only SCHEDULES the coroutine, so a job cancelled in the
        same tick it was started never enters ``_run`` at all — none of the
        handling in there executes, and the job would sit at RUNNING forever
        with no task behind it. This callback fires however the task ended.
        """
        if job.state.finished:
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
        from .execution_policy import background_scope
        with background_scope("job", run_id=job.id):
            await self._run_background(job)

    async def _run_background(self, job: Job) -> None:
        # Set inside the coroutine so everything the tool awaits inherits it.
        _CURRENT_JOB.set(job)
        control: JobControl = job._control
        try:
            await control.admit()
            job.state = JobState.RUNNING
            control.step(0, state="running", attempts=1)
            result = await self.dispatch(job.tool, job.args)
            job.ok = bool(result.ok)
            job.result = str(result.text or "")
            job.state = JobState.COMPLETED if job.ok else JobState.FAILED
            control.step(0, state="completed" if job.ok else "failed",
                         detail=first_line(job.result, 200))
        except asyncio.CancelledError:
            job.state = JobState.CANCELLED
            job.result = "cancelled before completion"
            job.ok = False
            self._finish(job)
            raise
        except JobCancelled as exc:
            job.state, job.ok, job.result = JobState.CANCELLED, False, str(exc)
        except Busy as exc:             # superseded, or ORION is shutting down
            job.state, job.ok, job.result = JobState.CANCELLED, False, exc.reason
        except Exception as exc:
            job.state = JobState.FAILED
            job.ok = False
            job.result = f"{type(exc).__name__}: {exc}"
        self._finish(job)

    def _finish(self, job: Job) -> None:
        job.duration_s = max(0.0, time.monotonic() - job._monotonic)
        job.finished_at = utc_stamp()
        job._task = None
        control = job._control
        if control is not None:
            try:
                control.finish(State(job.state.value), ok=job.ok, result=job.result,
                               fraction=job.fraction)
            except Exception as exc:
                # The slot and live control are already retired by finish();
                # say plainly that the durable record did not take the outcome,
                # and still tell the interface the job has ended.
                self.bus.log.emit(f"JOB: {job.id} outcome could not be recorded - "
                                  f"{first_line(exc, 160)}")
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
        if job._control is not None:
            job._control.progress(job.fraction, job.note)
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
            control = find_control(job_id)
            if control is not None and control.cancel():
                return ToolResult(f"Cancelling {control.id} ({control.label}).")
            return ToolResult(f"No job matches '{job_id}'.", ok=False)
        if job.state.finished or job._task is None:
            return ToolResult(f"{job.id} already {job.state.value}.", ok=False)
        job._control.cancel()
        return ToolResult(f"Cancelling {job.id} ({job.label}).")

    # ── pause, resume, priority, approval (any kind of job) ─────────────────

    def pause(self, job_id: str) -> ToolResult:
        control = self._control(job_id)
        if control is None or not control.pause():
            return ToolResult(f"No running or queued job matches '{job_id}'.", ok=False)
        waits = "" if control.state is State.PAUSED else \
            " It stops at its next checkpoint; work already inside a call finishes first."
        return ToolResult(f"Pausing {control.id} ({control.label}).{waits}")

    async def resume(self, job_id: str) -> ToolResult:
        """Resume a paused job, or an INTERRUPTED one left by a restart."""
        control = self._control(job_id)
        if control is not None:
            if control.resume():
                return ToolResult(f"Resuming {control.id} ({control.label}).")
            return ToolResult(f"{control.id} is not paused.", ok=False)
        record = self._record(job_id)
        if record is None:
            return ToolResult(f"No job matches '{job_id}'.", ok=False)
        # Recovery approvals deliberately leave an unowned job waiting until
        # the user asks to continue. Its resumer still checks the decision,
        # exact step/args binding and expiry before dispatching anything.
        if record["state"] not in {State.INTERRUPTED.value, State.WAITING_FOR_APPROVAL.value}:
            return ToolResult(f"{record['id']} is {record['state']}; nothing to resume.",
                              ok=False)
        resumer = _RESUMERS.get(record["kind"])
        if resumer is None:
            return ToolResult(
                f"{record['id']} ({record['label']}) cannot be resumed automatically — "
                "start it again once you have checked what it already did.", ok=False)
        outcome = resumer(record)
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        return outcome if isinstance(outcome, ToolResult) else ToolResult(str(outcome))

    def reprioritise(self, job_id: str, priority: Any) -> ToolResult:
        control = self._control(job_id)
        if control is None:
            return ToolResult(f"No live job matches '{job_id}'.", ok=False)
        try:
            klass = work_class(priority)
        except ValueError:
            return ToolResult("Priority must be user, background or training.", ok=False)
        if klass is WorkClass.INTERACTIVE and control.kind != "plan":
            klass = WorkClass.USER
        control.reprioritise(klass)
        job = self._jobs.get(control.id)
        if job is not None:
            job.cls = klass.label
            self._broadcast(job)
        return ToolResult(f"{control.id} ({control.label}) now runs as {klass.label} work.")

    def approve(self, approval_id: str, approve: bool = True, by: str = "user") -> ToolResult:
        pending = self.store.approval(approval_id)
        record = self.store.get(pending["job_id"]) if pending is not None else None
        decided = self.store.decide_approval(approval_id, approve, by=by)
        if decided is None:
            return ToolResult("That approval is no longer pending (it expired, was "
                              "replaced, or was already decided).", ok=False)
        verb = "Approved" if approve else "Declined"
        step = approval_step_number(decided["step_idx"], (record or {}).get("kind", ""))
        return ToolResult(f"{verb}: step {step} ({decided['tool']}) "
                          f"of {decided['job_id']}.")

    def _control(self, job_id: str) -> JobControl | None:
        job = self.get(job_id)
        if job is not None and job._control is not None and job.state.active:
            return job._control
        return find_control(job_id)

    def _record(self, job_id: str) -> dict[str, Any] | None:
        key = str(job_id or "").strip()
        record = self.store.get(key)
        if record is not None:
            return record
        folded = key.lower()
        for candidate in self.store.list(limit=200):
            if folded and (candidate["id"].startswith(folded)
                           or folded in str(candidate["label"]).lower()):
                return candidate
        return None

    async def _resume_tool_job(self, record: dict[str, Any]) -> ToolResult:
        """Background jobs are read-only by construction (PARALLEL only), so
        re-running an interrupted one cannot duplicate a side effect."""
        spec = record.get("args") or {}
        tool = str(spec.get("tool") or record.get("name") or "")
        args = spec.get("args") if isinstance(spec.get("args"), dict) else {}
        if not record.get("args_complete", True):
            return ToolResult(f"{record['id']}'s arguments were too large to keep; "
                              "start it again.", ok=False)
        self.store.transition(record["id"], State.CANCELLED,
                              note="superseded by a resumed run")
        outcome = self.start(tool, args, label=str(record.get("label") or tool),
                             priority=record.get("cls") or "user")
        if isinstance(outcome, ToolResult):
            return outcome
        return ToolResult(f"Resumed {record['id']} as {outcome.id} ({outcome.label}).")

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
            # A job from before a restart: its record, and result, survive.
            record = self._record(job_id)
            if record is None:
                return ToolResult(f"No job matches '{job_id}'.", ok=False)
            header = f"{record['id']} — {record['label']} ({record['state']})"
            if JobState(record["state"]).active:
                return ToolResult(f"{header} — still in progress.\n"
                                  f"{record.get('note') or ''}", ok=False)
            return ToolResult(f"{header}\n\n{record.get('result') or record.get('note') or ''}",
                              ok=record["state"] == State.COMPLETED.value
                              and bool(record.get("ok")))
        if job.state.active:
            return ToolResult(
                f"{job.id} ({job.label}) is still {job.state.value.replace('_', ' ')} — "
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
        if pending:
            await asyncio.wait(pending, timeout=1.0)

    def describe(self) -> dict[str, Any]:
        return {
            "active": len(self.active()),
            "max_active": self.max_active,
            "total": len(self._jobs),
        }

    def snapshot(self, limit: int = 50, since_seq: int = 0) -> dict[str, Any]:
        """Every job (tool, workflow, plan) for a task view — see the report
        for the shape. Cheap: one indexed query plus the admission counters."""
        view = self.store.snapshot(limit=limit, since_seq=since_seq)
        for record in view["jobs"]:
            control = CONTROLS.get(record["id"])
            record["live"] = control is not None
            record["position"] = control.position() if control is not None else None
            record["pause_requested"] = bool(control and control.pause_requested)
        view["approvals"] = self.store.pending_approvals()
        try:
            view["admission"] = self._admission().snapshot()
        except Exception:
            view["admission"] = {}
        return view


__all__ = ["CONTROLS", "DEFAULT_MAX_ACTIVE", "Job", "JobCancelled", "JobControl",
           "JobManager", "JobState", "PROGRESS_STEP", "RETAINED_JOBS", "checkpoint",
           "approval_step_number", "checkpoint_sync", "current_control", "find_control", "register_resumer",
           "report_progress", "suspend_all"]
