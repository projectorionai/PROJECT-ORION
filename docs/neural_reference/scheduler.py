from __future__ import annotations

import asyncio
import copy
import heapq
import math
import time
import uuid
from dataclasses import dataclass, field

from .async_utils import atomic_async
from .contracts import Outcome, Plan


@dataclass
class Job:
    id: str
    owner: str
    plan: Plan
    priority: int = 50  # 0 is most urgent
    created: float = field(default_factory=time.time)
    paused: bool = False
    cancelled: bool = False
    states: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, dict] = field(default_factory=dict)
    revision: int = 0

    def snapshot(self):
        return {
            "id": self.id,
            "owner": self.owner,
            "plan": self.plan.model_dump(),
            "priority": self.priority,
            "created": self.created,
            "paused": self.paused,
            "cancelled": self.cancelled,
            "states": dict(self.states),
            "outputs": copy.deepcopy(self.outputs),
            "revision": self.revision,
        }


class Scheduler:
    """Single desktop executor, bounded admission, cooperative pause/replanning.

    Rebuild a small priority heap at each dispatch: no stale queue entries after
    reprioritisation. A durable claim precedes every side effect. No automatic
    retry of writes, and no cancellation pretending to undo an OS operation.
    """

    def __init__(self, agent, journal):
        self.agent, self.journal = agent, journal
        self.jobs: dict[str, Job] = {}
        self.condition = asyncio.Condition()
        self.pressure = False
        self.closing = False
        self.worker = None
        self.fatal_error = ""

    async def _persist(self, job, event, *args):
        # All callers hold the scheduler condition while changing state.
        try:
            await self.journal.record(job.snapshot(), event, *args)
        except Exception as exc:
            self.fatal_error = type(exc).__name__
            self.closing = True
            self.condition.notify_all()
            raise

    @atomic_async
    async def start(self):
        async with self.condition:
            if self.worker is not None or self.closing:
                raise RuntimeError("Scheduler already started or closed")
            recovered = {}
            for raw in await asyncio.to_thread(self.journal.load):
                raw = copy.deepcopy(raw)
                raw["plan"] = Plan.model_validate(raw["plan"])
                job = Job(**raw)
                if (
                    set(job.states) != {s.id for s in job.plan.steps}
                    or set(job.states.values())
                    - {
                        "pending",
                        "running",
                        "succeeded",
                        "failed",
                        "blocked",
                        "cancelled",
                        "uncertain",
                    }
                    or not math.isfinite(job.created)
                ):
                    raise ValueError("Invalid persisted job state")
                # A crash can occur after the side effect and before its receipt.
                job.states = {
                    k: ("uncertain" if v == "running" else v)
                    for k, v in job.states.items()
                }
                job.paused = True
                recovered[job.id] = job
            self.jobs.update(recovered)
            self.worker = asyncio.create_task(self._run(), name="orion-neural-executor")

    def _available(self):
        if self.closing or self.fatal_error:
            raise RuntimeError("Scheduler unavailable")

    @staticmethod
    def _priority(priority):
        if type(priority) is not int or not 0 <= priority <= 100:
            raise ValueError("Priority must be an integer between 0 and 100")
        return priority

    @atomic_async
    async def submit(self, owner, plan: Plan, priority=50):
        self.agent.authorise(plan)
        async with self.condition:
            self._available()
            active = sum(
                any(v in {"pending", "running", "uncertain"} for v in j.states.values())
                for j in self.jobs.values()
            )
            if active >= 128:
                raise ValueError("Queue capacity reached")
            job = Job(uuid.uuid4().hex, owner, plan, self._priority(priority))
            job.states = {s.id: "pending" for s in plan.steps}
            await self._persist(job, "submitted")
            self.jobs[job.id] = job
            self.condition.notify_all()
            return job.id

    def owned(self, job_id, owner):
        job = self.jobs.get(job_id)
        if job is None or job.owner != owner:
            raise KeyError("Unknown job")
        return job

    @atomic_async
    async def control(self, job_id, owner, *, paused=None, priority=None, cancel=False):
        async with self.condition:
            self._available()
            job = copy.deepcopy(self.owned(job_id, owner))
            if paused is False and "uncertain" in job.states.values():
                raise ValueError("Reconcile uncertain actions before resuming")
            if paused is not None:
                job.paused = paused
            if priority is not None:
                job.priority = self._priority(priority)
            if cancel:
                job.cancelled = True
                job.states = {
                    k: ("cancelled" if v == "pending" else v)
                    for k, v in job.states.items()
                }
            job.revision += 1
            await self._persist(job, "controlled")
            self.jobs[job_id] = job
            self.condition.notify_all()

    @atomic_async
    async def revise(
        self, job_id, owner, plan: Plan, *, expected_revision=None, priority=None
    ):
        self.agent.authorise(plan)
        async with self.condition:
            self._available()
            job = copy.deepcopy(self.owned(job_id, owner))
            if expected_revision is not None and job.revision != expected_revision:
                raise ValueError("Job changed during planning; retry")
            if (
                not job.paused
                or job.cancelled
                or any(v in {"running", "uncertain"} for v in job.states.values())
            ):
                raise ValueError("Pause and wait for a known, quiescent state first")
            previous = {s.id: s for s in job.plan.steps}
            replacement = {s.id: s for s in plan.steps}
            done = {sid for sid, state in job.states.items() if state == "succeeded"}
            if any(replacement.get(sid) != previous[sid] for sid in done):
                raise ValueError("Completed actions are immutable and must remain")
            if any(
                (sid in job.outputs and sid not in done)
                or job.states.get(sid) in {"failed", "blocked", "cancelled"}
                for sid in replacement
            ):
                raise ValueError("Use fresh IDs for new attempts")
            job.plan = plan
            if priority is not None:
                job.priority = self._priority(priority)
            job.states = {
                s.id: ("succeeded" if s.id in done else "pending") for s in plan.steps
            }
            job.revision += 1
            await self._persist(job, "revised")
            self.jobs[job_id] = job
            self.condition.notify_all()

    async def set_pressure(self, high: bool):
        async with self.condition:
            self.pressure = high
            self.condition.notify_all()

    def _next(self):
        ready = []
        now = time.time()
        for job in self.jobs.values():
            if job.paused or job.cancelled:
                continue
            for step in job.plan.steps:
                if job.states[step.id] != "pending":
                    continue
                if all(job.states[d] == "succeeded" for d in step.depends_on):
                    # Age old jobs so low-priority work eventually gets a turn.
                    score = job.priority - int(max(0, now - job.created) // 60)
                    ready.append((score, job.created, job.id, step.id, step))
        if not ready:
            return None
        heapq.heapify(ready)
        _, _, job_id, _, step = heapq.heappop(ready)
        return self.jobs[job_id], step

    async def _run(self):
        try:
            while True:
                async with self.condition:
                    while True:
                        if self.closing:
                            return
                        candidate = None if self.pressure else self._next()
                        if candidate:
                            break
                        await self.condition.wait()
                    job, step = candidate
                    job = copy.deepcopy(job)
                    job.states[step.id] = "running"
                    job.revision += 1
                    await self._persist(job, "claimed", step.action.tool)
                    self.jobs[job.id] = job
                    self.condition.notify_all()
                started = time.monotonic()
                try:
                    result = await self.agent.execute(step.action)
                except Exception as exc:  # noqa: BLE001 - tool failure boundary
                    result = Outcome(ok=False, error=type(exc).__name__)
                async with self.condition:
                    # Control requests may have replaced the job while the
                    # action ran. Merge its receipt into the latest snapshot.
                    job = copy.deepcopy(self.jobs[job.id])
                    job.states[step.id] = "succeeded" if result.ok else "failed"
                    job.outputs[step.id] = result.model_dump()
                    # Failures block descendants, but independent work can continue.
                    changed = True
                    while changed:
                        changed = False
                        for node in job.plan.steps:
                            if job.states[node.id] == "pending" and any(
                                job.states[d] in {"failed", "blocked", "cancelled"}
                                for d in node.depends_on
                            ):
                                job.states[node.id] = "blocked"
                                changed = True
                    job.revision += 1
                    await self._persist(
                        job,
                        "finished",
                        step.action.tool,
                        result.ok,
                        time.monotonic() - started,
                    )
                    self.jobs[job.id] = job
                    self.condition.notify_all()
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 - worker supervisor
            # A persistence failure must stop dispatch, not silently lose receipts.
            async with self.condition:
                self.fatal_error = type(exc).__name__
                self.closing = True
                for job in self.jobs.values():
                    if "running" in job.states.values():
                        job.states = {
                            k: ("uncertain" if v == "running" else v)
                            for k, v in job.states.items()
                        }
                        job.paused = True
                self.condition.notify_all()

    @atomic_async
    async def close(self):
        async with self.condition:
            self.closing = True
            self.condition.notify_all()
        if self.worker:
            await self.worker  # finish the current atomic action; do not start more
