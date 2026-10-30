"""
Admission — one machine-wide answer to "may this start now?".

The problem (architecture review 2026-09-26, finding 3, verified against the
code): ORION's limits were local and overlapped. The default executor bounded
its threads but queued without limit; intent training queued without limit;
bulk work had two workers, background jobs four, a tool batch six and
workflows eight — and none of them knew the others existed. A research run, an
index refresh and a training pass could each be "within its limit" while the
user's next voice command waited behind all three.

This is the shared policy those pieces consult. It does not replace them: the
executors still own threads, the resource governor still owns memory pressure,
the job manager still owns job records. Admission decides ORDER and HOW MANY.

Four classes, highest priority first:

    INTERACTIVE   a voice or typed command being answered now
    USER          work the user asked for that runs detached (jobs, workflows)
    BACKGROUND    indexing, research refreshes, periodic self-maintenance
    TRAINING      learning passes — the most optional work there is

The rules:

* Every class has a RUNNING limit and a PENDING limit. Past the pending limit
  ``submit`` raises ``Busy`` — something the caller can show — so nothing
  queues without bound, nothing is dropped silently and no caller is ever
  blocked on a full queue.
* INTERACTIVE never draws on the shared budget, so a burst of detached work
  cannot take the capacity a user turn needs.
* Waiting work AGES: every ``age_step_s`` spent queued lifts a ticket one
  class for the shared budget, so background work behind a steady stream of
  user jobs still runs.
* Replaceable work COALESCES by key: a newer refresh takes the queued one's
  place and keeps its age; the older waiter is told it was superseded.
* Pressure DEFERS optional work, with the governor's hysteresis (memory
  states, plus CPU load and GPU memory). Missing GPU telemetry is reported as
  unknown, never as free.
* ``lease("gpu")`` is exclusive by default, so a training pass and a local
  generation never share the card. A higher class waiting for a lease sets
  ``preempt_requested`` on the holder; a cooperative holder yields at its next
  checkpoint. A native call already running cannot be pre-empted, only asked.

Thread-safe: the state sits behind one lock, asyncio waiters are woken with
``call_soon_threadsafe`` and worker threads can wait with ``wait_blocking`` —
which refuses to run on a thread that is driving an event loop.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, AsyncIterator, Callable, Iterator


class WorkClass(IntEnum):
    """Priority classes; a LOWER value is served first."""

    INTERACTIVE = 0
    USER = 1
    BACKGROUND = 2
    TRAINING = 3

    @property
    def label(self) -> str:
        return self.name.lower()


_ALIASES: dict[str, WorkClass] = {
    "interactive": WorkClass.INTERACTIVE, "voice": WorkClass.INTERACTIVE,
    "foreground": WorkClass.INTERACTIVE, "command": WorkClass.INTERACTIVE,
    "user": WorkClass.USER, "job": WorkClass.USER, "jobs": WorkClass.USER,
    "normal": WorkClass.USER, "workflow": WorkClass.USER,
    "background": WorkClass.BACKGROUND, "research": WorkClass.BACKGROUND,
    "indexing": WorkClass.BACKGROUND, "refresh": WorkClass.BACKGROUND,
    "training": WorkClass.TRAINING, "train": WorkClass.TRAINING,
    "learning": WorkClass.TRAINING,
}


def work_class(value: Any) -> WorkClass:
    """A WorkClass from a class, its number, or a name such as 'background'."""
    if isinstance(value, WorkClass):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return WorkClass(value)
    found = _ALIASES.get(str(value or "").strip().lower())
    if found is None:
        raise ValueError(f"unknown work class {value!r}")
    return found


class Busy(Exception):
    """Work that could not be admitted, with a sentence the caller can show."""

    def __init__(self, reason: str, *, cls: WorkClass = WorkClass.USER,
                 running: int = 0, pending: int = 0, limit: int = 0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cls = cls
        self.running = running
        self.pending = pending
        self.limit = limit

    def to_dict(self) -> dict[str, Any]:
        return {"busy": True, "reason": self.reason, "class": self.cls.label,
                "running": self.running, "pending": self.pending, "limit": self.limit}

    def to_tool_result(self) -> Any:
        from .data import ToolResult
        return ToolResult(self.reason, ok=False)


class Superseded(Busy):
    """A newer request with the same key took this queued one's place."""


@dataclass(frozen=True)
class ClassPolicy:
    running: int
    pending: int


#: Starting policy for a 6-core / 16 GB / 8 GB-VRAM desktop. USER running
#: matches the job manager's historical ceiling of four; BACKGROUND matches the
#: bulk pool's two workers; TRAINING is one pass at a time.
DEFAULT_POLICY: dict[WorkClass, ClassPolicy] = {
    WorkClass.INTERACTIVE: ClassPolicy(running=8, pending=32),
    WorkClass.USER: ClassPolicy(running=4, pending=16),
    WorkClass.BACKGROUND: ClassPolicy(running=2, pending=32),
    WorkClass.TRAINING: ClassPolicy(running=1, pending=4),
}
#: Detached work running at once across USER + BACKGROUND + TRAINING. Less
#: than the sum of their limits on purpose: that contention is what ageing
#: arbitrates, and six physical cores is the real ceiling.
DEFAULT_SHARED_RUNNING = 5
AGE_STEP_S = 20.0
#: After the last interactive request, how long the foreground counts as busy.
FOREGROUND_QUIET_S = 4.0
LEASE_PENDING = 8


def _lease_capacity(resource: str) -> int:
    """GPU leases are exclusive unless ORION_GPU_LEASES says otherwise."""
    if resource != "gpu":
        return 1
    raw = os.getenv("ORION_GPU_LEASES", "").strip()
    return max(1, min(4, int(raw))) if raw.isdigit() else 1


class Ticket:
    """One admission request: queued, then running, then released."""

    def __init__(self, owner: "Admission", cls: WorkClass, *, key: Any = None,
                 label: str = "", replaceable: bool = False,
                 resource: str | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.cls = cls
        self.key = key
        self.label = str(label or "")[:80]
        self.replaceable = bool(replaceable)
        self.resource = resource
        self.state = "queued"           # queued / running / released / cancelled / superseded
        self.enqueued_at = owner.clock()
        self.granted_at: float | None = None
        self.error: Busy | None = None
        #: Set on a lease holder when a higher class is waiting for it.
        self.preempt_requested = False
        self._owner = owner
        self._event = threading.Event()
        self._future: Any = None
        self._loop: Any = None
        self._recorded = False

    # ── inspection ──────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def queued(self) -> bool:
        return self.state == "queued"

    def position(self) -> int | None:
        """1-based place in its queue while queued, else None."""
        return self._owner._position(self)

    def waited_s(self) -> float:
        end = self.granted_at if self.granted_at is not None else self._owner.clock()
        return max(0.0, end - self.enqueued_at)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "class": self.cls.label, "state": self.state,
                "label": self.label, "resource": self.resource,
                "position": self.position(), "waited_s": round(self.waited_s(), 3),
                "preempt_requested": self.preempt_requested}

    # ── waiting ─────────────────────────────────────────────────────────────

    async def wait(self, timeout: float | None = None) -> "Ticket":
        """Until this ticket runs. Raises Busy on timeout, Superseded when a
        newer request replaced it, and Busy when admission closed."""
        owner = self._owner
        with owner._lock:
            if self.state == "queued":
                loop = asyncio.get_running_loop()
                self._loop = loop
                self._future = loop.create_future()
        if self.state == "running":
            self._record()
            return self
        if self.error is not None:
            raise self.error
        if self.state != "queued":
            raise Busy(f"{self.label or 'That work'} is no longer waiting ({self.state}).",
                       cls=self.cls)
        try:
            if timeout is None:
                await self._future
            else:
                await asyncio.wait_for(asyncio.shield(self._future), timeout)
        except asyncio.TimeoutError:
            owner._release(self)
            raise Busy(f"{self.label or 'That work'} waited {timeout:.0f}s and could "
                       "not start; it has been withdrawn.", cls=self.cls) from None
        except asyncio.CancelledError:
            owner._release(self)
            raise
        if self.error is not None:
            raise self.error
        if self.state != "running":
            raise Busy(f"{self.label or 'That work'} was withdrawn before it started.",
                       cls=self.cls)
        self._record()
        return self

    def wait_blocking(self, timeout: float | None = None) -> "Ticket":
        """The same, for worker threads. Never waits on an event-loop thread."""
        if self.state == "queued":
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError("wait_blocking would stall the event loop; "
                                   "await ticket.wait() there instead")
        if not self._event.wait(timeout):
            self._owner._release(self)
            raise Busy(f"{self.label or 'That work'} waited {timeout:.0f}s and could "
                       "not start; it has been withdrawn.", cls=self.cls)
        if self.error is not None:
            raise self.error
        if self.state != "running":
            raise Busy(f"{self.label or 'That work'} was withdrawn before it started.",
                       cls=self.cls)
        self._record()
        return self

    def release(self) -> None:
        """Give the slot (or the place in the queue) back. Idempotent."""
        self._owner._release(self)

    def _record(self) -> None:
        if not self._recorded and self.granted_at is not None:
            self._recorded = True
            self._owner._record_wait(self)


@dataclass
class _Resource:
    capacity: int
    holders: list[Ticket] = field(default_factory=list)
    waiters: list[Ticket] = field(default_factory=list)


@dataclass
class _OptionalLoop:
    name: str
    cls: WorkClass
    pause: Callable[[], Any] | None = None
    resume: Callable[[], Any] | None = None
    paused: bool = False
    reason: str = ""


def _resolve_future(future: Any) -> None:
    if not future.done():
        future.set_result(None)


class Admission:
    """The machine-wide admission policy. See the module docstring."""

    def __init__(self, policy: dict[WorkClass, ClassPolicy] | None = None, *,
                 shared_running: int = DEFAULT_SHARED_RUNNING,
                 age_step_s: float = AGE_STEP_S,
                 quiet_s: float = FOREGROUND_QUIET_S,
                 governor: Any = None, metrics: Any = None, bus: Any = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = dict(DEFAULT_POLICY)
        self.policy.update(policy or {})
        self.shared_running = max(1, int(shared_running))
        self.age_step_s = max(0.001, float(age_step_s))
        self.quiet_s = max(0.0, float(quiet_s))
        self.governor = governor
        self.metrics = metrics
        self.bus = bus
        self.clock = clock
        self._lock = threading.RLock()
        self._queues: dict[WorkClass, deque[Ticket]] = {c: deque() for c in WorkClass}
        self._running: dict[WorkClass, int] = {c: 0 for c in WorkClass}
        self._shared = 0
        self._resources: dict[str, _Resource] = {}
        self._loops: dict[str, _OptionalLoop] = {}
        self._last_foreground = -1e18
        self._closed = False
        self._refresh_handle: Any = None
        self._stats: dict[WorkClass, dict[str, float]] = {
            c: {"admitted": 0, "queued": 0, "rejected": 0, "superseded": 0,
                "cancelled": 0, "waits": 0, "wait_ms_total": 0.0, "wait_ms_max": 0.0}
            for c in WorkClass}

    def attach(self, *, governor: Any = None, metrics: Any = None, bus: Any = None) -> None:
        """Wire the live governor, metrics registry and log (app.py)."""
        if governor is not None:
            self.governor = governor
        if metrics is not None:
            self.metrics = metrics
        if bus is not None:
            self.bus = bus

    # ── conditions ──────────────────────────────────────────────────────────

    def _governor(self) -> Any:
        if self.governor is not None:
            return self.governor
        from . import resource_governor
        return resource_governor.GOVERNOR

    def _conditions(self) -> tuple[int, bool | None, bool | None]:
        """(memory level 0-3, cpu busy, gpu busy) — None means unknown."""
        governor = self._governor()
        if governor is None or not getattr(governor, "acting", True):
            return 0, None, None
        try:
            memory = int(governor.effective())
        except Exception:
            memory = 0
        cpu = getattr(governor, "cpu_busy", None)
        gpu = getattr(governor, "gpu_busy", None)
        try:
            cpu = cpu() if callable(cpu) else None
        except Exception:
            cpu = None
        try:
            gpu = gpu() if callable(gpu) else None
        except Exception:
            gpu = None
        return memory, cpu, gpu

    def _foreground_locked(self, now: float) -> bool:
        return self._running[WorkClass.INTERACTIVE] > 0 or (now - self._last_foreground) < self.quiet_s

    def foreground_active(self) -> bool:
        with self._lock:
            return self._foreground_locked(self.clock())

    def defer_reason(self, cls: Any = WorkClass.BACKGROUND) -> str:
        """Why optional work of *cls* should wait right now, or '' to go."""
        cls = work_class(cls)
        if self._closed:
            return "shutting down"
        if cls <= WorkClass.USER:
            return ""
        memory, cpu, gpu = self._conditions()
        with self._lock:
            foreground = self._foreground_locked(self.clock())
        if memory >= 2:
            return "memory is under pressure"
        if foreground:
            return "a request is being answered"
        if cpu:
            return "the processor is busy"
        if cls is WorkClass.TRAINING:
            if memory >= 1:
                return "memory is constrained"
            if gpu:
                return "graphics memory is nearly full"
        return ""

    def _effective_limit_locked(self, cls: WorkClass) -> int:
        base = self.policy[cls].running
        if self._closed:
            return 0
        if cls is WorkClass.INTERACTIVE:
            return base
        memory, cpu, gpu = self._conditions()
        foreground = self._foreground_locked(self.clock())
        if cls is WorkClass.USER:
            return min(base, 1) if memory >= 3 else base
        if cls is WorkClass.BACKGROUND:
            if memory >= 2:
                return 0
            # Never 0 for CPU or the foreground: one background task still
            # progresses, which is what keeps it from starving.
            return min(base, 1) if (cpu or foreground or memory >= 1) else base
        # TRAINING: fully deferred by anything the user would notice.
        if memory >= 1 or cpu or gpu or foreground:
            return 0
        return base

    def _can_start_locked(self, cls: WorkClass) -> bool:
        if self._running[cls] >= self._effective_limit_locked(cls):
            return False
        return cls is WorkClass.INTERACTIVE or self._shared < self.shared_running

    def _rank(self, ticket: Ticket, now: float) -> int:
        return max(0, int(ticket.cls) - int((now - ticket.enqueued_at) // self.age_step_s))

    # ── granting ────────────────────────────────────────────────────────────

    def _start_locked(self, ticket: Ticket, now: float) -> bool:
        """Mark *ticket* running. True when this began a foreground spell."""
        ticket.state = "running"
        ticket.granted_at = now
        ticket._event.set()
        self._stats[ticket.cls]["admitted"] += 1
        if ticket.resource is not None:
            return False
        if ticket.cls is WorkClass.INTERACTIVE:
            began = not self._foreground_locked(now)
            self._running[ticket.cls] += 1
            self._last_foreground = now
            return began
        self._running[ticket.cls] += 1
        self._shared += 1
        return False

    def _pump_locked(self) -> list[Ticket]:
        granted: list[Ticket] = []
        now = self.clock()
        while True:
            best: Ticket | None = None
            best_key: tuple[int, float] | None = None
            for cls, queue in self._queues.items():
                if not queue or not self._can_start_locked(cls):
                    continue
                head = queue[0]
                key = (self._rank(head, now), head.enqueued_at)
                if best_key is None or key < best_key:
                    best, best_key = head, key
            if best is None:
                return granted
            self._queues[best.cls].popleft()
            self._start_locked(best, now)
            granted.append(best)

    def _notify(self, ticket: Ticket) -> None:
        future, loop = ticket._future, ticket._loop
        if future is None or loop is None:
            return
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is loop:
            _resolve_future(future)
            return
        try:
            loop.call_soon_threadsafe(_resolve_future, future)
        except RuntimeError:
            # The waiter's loop is gone; nobody will ever use this slot.
            self._release(ticket)

    def submit(self, cls: Any = WorkClass.USER, *, key: Any = None,
               replaceable: bool = False, label: str = "") -> Ticket:
        """Ask to run. Returns a Ticket that is running or queued; raises Busy
        when the class's queue is full. Never blocks."""
        cls = work_class(cls)
        ticket = Ticket(self, cls, key=key, label=label, replaceable=replaceable)
        notify: list[Ticket] = []
        began = False
        with self._lock:
            if self._closed:
                raise Busy("ORION is shutting down, so nothing new is starting.", cls=cls)
            notify.extend(self._pump_locked())
            queue = self._queues[cls]
            if key is not None and replaceable:
                for index, other in enumerate(queue):
                    if other.key == key:
                        # Coalesce: the newer request takes the older's place
                        # AND its age, so replacing never resets the clock.
                        ticket.enqueued_at = other.enqueued_at
                        queue[index] = ticket
                        other.state = "superseded"
                        other.error = Superseded(
                            f"{other.label or 'That work'} was replaced by a newer "
                            "request for the same thing.", cls=cls)
                        other._event.set()
                        notify.append(other)
                        self._stats[cls]["superseded"] += 1
                        break
            if ticket.state == "queued" and ticket not in queue:
                if not queue and self._can_start_locked(cls):
                    began = self._start_locked(ticket, self.clock())
                else:
                    limit = self.policy[cls].pending
                    if len(queue) >= limit:
                        self._stats[cls]["rejected"] += 1
                        running = self._running[cls]
                        error = Busy(
                            f"{label or 'That'} can't start yet: {len(queue)} "
                            f"{cls.label} task(s) are already waiting, which is the "
                            "limit. Try again when one finishes.",
                            cls=cls, running=running, pending=len(queue), limit=limit)
                    else:
                        error = None
                        queue.append(ticket)
                        self._stats[cls]["queued"] += 1
                        notify.extend(self._pump_locked())
                    if error is not None:
                        self._gauge(cls)
                        self._count(f"admission.rejected.{cls.label}")
                        raise error
        for waiter in notify:
            self._notify(waiter)
        if ticket.state == "running":
            ticket._record()
        self._gauge(cls)
        if began:
            self.evaluate()
        return ticket

    def _position(self, ticket: Ticket) -> int | None:
        with self._lock:
            if ticket.state != "queued":
                return None
            if ticket.resource is not None:
                resource = self._resources.get(ticket.resource)
                waiters = sorted(resource.waiters, key=lambda t: (t.cls, t.enqueued_at)) \
                    if resource else []
            else:
                waiters = list(self._queues[ticket.cls])
            try:
                return waiters.index(ticket) + 1
            except ValueError:
                return None

    def _release(self, ticket: Ticket) -> None:
        notify: list[Ticket] = []
        ended_foreground = False
        with self._lock:
            now = self.clock()
            if ticket.state == "running":
                ticket.state = "released"
                if ticket.resource is not None:
                    resource = self._resources.get(ticket.resource)
                    if resource is not None and ticket in resource.holders:
                        resource.holders.remove(ticket)
                        notify.extend(self._pump_resource_locked(resource))
                else:
                    self._running[ticket.cls] = max(0, self._running[ticket.cls] - 1)
                    if ticket.cls is WorkClass.INTERACTIVE:
                        self._last_foreground = now
                        ended_foreground = self._running[ticket.cls] == 0
                    else:
                        self._shared = max(0, self._shared - 1)
                    notify.extend(self._pump_locked())
            elif ticket.state == "queued":
                ticket.state = "cancelled"
                self._stats[ticket.cls]["cancelled"] += 1
                if ticket.resource is not None:
                    resource = self._resources.get(ticket.resource)
                    if resource is not None and ticket in resource.waiters:
                        resource.waiters.remove(ticket)
                        self._update_preempt_locked(resource)
                else:
                    try:
                        self._queues[ticket.cls].remove(ticket)
                    except ValueError:
                        pass
                ticket._event.set()
                notify.append(ticket)       # wake its waiter: it was withdrawn
            else:
                return
        for waiter in notify:
            self._notify(waiter)
        self._gauge(ticket.cls)
        if ended_foreground:
            self._schedule_refresh()

    def reprioritise(self, ticket: Ticket, cls: Any) -> None:
        """Move a ticket to another class. A queued ticket keeps its arrival
        time (so its age), a running one moves its slot; either may let
        something else start."""
        cls = work_class(cls)
        notify: list[Ticket] = []
        with self._lock:
            old = ticket.cls
            if old is cls:
                return
            if ticket.resource is not None:
                ticket.cls = cls
                resource = self._resources.get(ticket.resource)
                if resource is not None:
                    self._update_preempt_locked(resource)
                    notify.extend(self._pump_resource_locked(resource))
            elif ticket.state == "queued":
                try:
                    self._queues[old].remove(ticket)
                except ValueError:
                    pass
                ticket.cls = cls
                queue = self._queues[cls]
                index = next((i for i, other in enumerate(queue)
                              if other.enqueued_at > ticket.enqueued_at), len(queue))
                queue.insert(index, ticket)
            elif ticket.state == "running":
                self._running[old] = max(0, self._running[old] - 1)
                self._running[cls] += 1
                if old is WorkClass.INTERACTIVE:
                    self._shared += 1
                elif cls is WorkClass.INTERACTIVE:
                    self._shared = max(0, self._shared - 1)
                ticket.cls = cls
            else:
                ticket.cls = cls
                return
            notify.extend(self._pump_locked())
        for waiter in notify:
            self._notify(waiter)

    def refresh(self) -> None:
        """Re-run the policy after conditions changed (a governor tick, the
        foreground going quiet): start what may now start, pause or resume
        registered loops."""
        with self._lock:
            notify = self._pump_locked()
            for resource in self._resources.values():
                notify.extend(self._pump_resource_locked(resource))
        for waiter in notify:
            self._notify(waiter)
        self.evaluate()

    def on_resources(self, *_args: Any) -> None:
        """ResourceGovernor listener: called after every sample."""
        self.refresh()

    def _schedule_refresh(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        handle = self._refresh_handle
        if handle is not None:
            handle.cancel()
        self._refresh_handle = loop.call_later(self.quiet_s + 0.05, self.refresh)

    def note_foreground(self, label: str = "") -> None:
        """Say the user is being served without taking a slot (a voice turn
        starting): optional loops defer until the foreground goes quiet."""
        with self._lock:
            now = self.clock()
            began = not self._foreground_locked(now)
            self._last_foreground = now
        if began:
            self.evaluate()
        self._schedule_refresh()

    # ── convenience ─────────────────────────────────────────────────────────

    @asynccontextmanager
    async def slot(self, cls: Any = WorkClass.USER, *, key: Any = None,
                   replaceable: bool = False, label: str = "",
                   timeout: float | None = None) -> AsyncIterator[Ticket]:
        """``async with ADMISSION.slot("background", label="index"):`` — waits
        its turn (raising Busy when the queue is full) and always releases."""
        ticket = self.submit(cls, key=key, replaceable=replaceable, label=label)
        try:
            await ticket.wait(timeout)
            yield ticket
        finally:
            ticket.release()

    @contextmanager
    def slot_blocking(self, cls: Any = WorkClass.BACKGROUND, *, key: Any = None,
                      replaceable: bool = False, label: str = "",
                      timeout: float | None = None) -> Iterator[Ticket]:
        """``slot`` for worker threads."""
        ticket = self.submit(cls, key=key, replaceable=replaceable, label=label)
        try:
            ticket.wait_blocking(timeout)
            yield ticket
        finally:
            ticket.release()

    def interactive(self, label: str = "") -> Any:
        """Mark a user request in flight: ``async with ADMISSION.interactive():``."""
        return self.slot(WorkClass.INTERACTIVE, label=label)

    def should_run(self, name_or_cls: Any = WorkClass.BACKGROUND) -> bool:
        """The cheap "should I run now?" for an optional loop's next pass."""
        entry = self._loops.get(name_or_cls) if isinstance(name_or_cls, str) else None
        if entry is not None:
            cls = entry.cls
        else:
            try:
                cls = work_class(name_or_cls)
            except ValueError:
                cls = WorkClass.BACKGROUND        # an unregistered loop's own name
        return not self.defer_reason(cls)

    # ── exclusive resources ─────────────────────────────────────────────────

    def acquire_lease(self, resource: str = "gpu", cls: Any = WorkClass.INTERACTIVE, *,
                      label: str = "") -> Ticket:
        """Queue for *resource*; returns a running or queued Ticket, or raises
        Busy when too many are already waiting."""
        cls = work_class(cls)
        name = str(resource or "gpu").strip().lower()
        ticket = Ticket(self, cls, label=label, resource=name)
        notify: list[Ticket] = []
        with self._lock:
            if self._closed:
                raise Busy("ORION is shutting down, so nothing new is starting.", cls=cls)
            res = self._resources.get(name)
            if res is None:
                res = self._resources[name] = _Resource(capacity=_lease_capacity(name))
            if not res.waiters and len(res.holders) < res.capacity:
                self._start_locked(ticket, self.clock())
                res.holders.append(ticket)
            elif len(res.waiters) >= LEASE_PENDING:
                self._stats[cls]["rejected"] += 1
                holder = res.holders[0] if res.holders else None
                raise Busy(f"The {name} is busy"
                           + (f" with {holder.label or holder.cls.label + ' work'}" if holder else "")
                           + f" and {len(res.waiters)} requests are already waiting for it.",
                           cls=cls, running=len(res.holders), pending=len(res.waiters),
                           limit=LEASE_PENDING)
            else:
                res.waiters.append(ticket)
                self._stats[cls]["queued"] += 1
                notify.extend(self._pump_resource_locked(res))
        for waiter in notify:
            self._notify(waiter)
        if ticket.state == "running":
            ticket._record()
        return ticket

    def _pump_resource_locked(self, res: _Resource) -> list[Ticket]:
        granted: list[Ticket] = []
        now = self.clock()
        while res.waiters and len(res.holders) < res.capacity and not self._closed:
            best = min(res.waiters, key=lambda t: (self._rank(t, now), t.enqueued_at))
            res.waiters.remove(best)
            self._start_locked(best, now)
            res.holders.append(best)
            granted.append(best)
        self._update_preempt_locked(res)
        return granted

    def _update_preempt_locked(self, res: _Resource) -> None:
        top = min((w.cls for w in res.waiters), default=None)
        for holder in res.holders:
            holder.preempt_requested = top is not None and top < holder.cls

    @asynccontextmanager
    async def lease(self, resource: str = "gpu", cls: Any = WorkClass.INTERACTIVE, *,
                    label: str = "", timeout: float | None = None) -> AsyncIterator[Ticket]:
        """``async with ADMISSION.lease("gpu", cls="training"):`` — exclusive
        use of *resource*. The yielded ticket's ``preempt_requested`` turns
        True when a higher class is waiting; check it at checkpoints."""
        ticket = self.acquire_lease(resource, cls, label=label)
        try:
            await ticket.wait(timeout)
            yield ticket
        finally:
            ticket.release()

    @contextmanager
    def lease_blocking(self, resource: str = "gpu", cls: Any = WorkClass.TRAINING, *,
                       label: str = "", timeout: float | None = None) -> Iterator[Ticket]:
        """``lease`` for worker threads (a training pass on its own thread)."""
        ticket = self.acquire_lease(resource, cls, label=label)
        try:
            ticket.wait_blocking(timeout)
            yield ticket
        finally:
            ticket.release()

    # ── optional loops ──────────────────────────────────────────────────────

    def register_loop(self, name: str, *, cls: Any = WorkClass.BACKGROUND,
                      pause: Callable[[], Any] | None = None,
                      resume: Callable[[], Any] | None = None) -> None:
        """Register an optional background loop. It is paused (``pause()``)
        while pressure or the foreground defers its class and resumed after;
        a loop without callbacks can simply ask ``should_run(name)``."""
        with self._lock:
            self._loops[str(name)] = _OptionalLoop(str(name), work_class(cls), pause, resume)

    def unregister_loop(self, name: str) -> None:
        with self._lock:
            self._loops.pop(str(name), None)

    def evaluate(self) -> None:
        """Apply pause/resume edges to the registered loops."""
        with self._lock:
            loops = list(self._loops.values())
        for entry in loops:
            reason = self.defer_reason(entry.cls)
            if reason and not entry.paused:
                entry.paused, entry.reason = True, reason
                self._call(entry.pause, f"pause {entry.name}")
            elif not reason and entry.paused:
                entry.paused, entry.reason = False, ""
                self._call(entry.resume, f"resume {entry.name}")

    def _call(self, fn: Callable[[], Any] | None, what: str) -> None:
        if fn is None:
            return
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    result.close()
                else:
                    # Owned and observed: a bare create_task could be
                    # collected mid-flight, and its failure went unreported.
                    from . import background
                    background.spawn(result, name=f"admission: {what}")
        except Exception as exc:
            self._log(f"ADMISSION: could not {what} - {exc}")

    # ── observability ───────────────────────────────────────────────────────

    def _record_wait(self, ticket: Ticket) -> None:
        ms = max(0.0, ((ticket.granted_at or 0.0) - ticket.enqueued_at) * 1000.0)
        with self._lock:
            stats = self._stats[ticket.cls]
            stats["waits"] += 1
            stats["wait_ms_total"] += ms
            stats["wait_ms_max"] = max(stats["wait_ms_max"], ms)
        try:
            from . import request_trace
            request_trace.mark("queue_wait", cls=ticket.cls.label, ms=ms,
                               resource=ticket.resource or "slot")
        except Exception:
            pass
        if self.metrics is not None:
            try:
                self.metrics.observe(f"admission.queue_wait_ms.{ticket.cls.label}", ms)
            except Exception:
                pass

    def _gauge(self, cls: WorkClass) -> None:
        if self.metrics is None:
            return
        try:
            self.metrics.gauge(f"admission.depth.{cls.label}", float(len(self._queues[cls])))
            self.metrics.gauge(f"admission.running.{cls.label}", float(self._running[cls]))
        except Exception:
            pass

    def _count(self, name: str) -> None:
        if self.metrics is not None:
            try:
                self.metrics.incr(name)
            except Exception:
                pass

    def _log(self, text: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(text)
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        """Everything a status view needs, cheap enough for a UI timer."""
        memory, cpu, gpu = self._conditions()
        governor = self._governor()
        with self._lock:
            now = self.clock()
            classes: dict[str, Any] = {}
            for cls in WorkClass:
                queue = self._queues[cls]
                stats = self._stats[cls]
                waits = stats["waits"]
                classes[cls.label] = {
                    "running": self._running[cls],
                    "pending": len(queue),
                    "limit": self._effective_limit_locked(cls),
                    "max_running": self.policy[cls].running,
                    "max_pending": self.policy[cls].pending,
                    "oldest_wait_s": round(now - queue[0].enqueued_at, 3) if queue else 0.0,
                    "admitted": int(stats["admitted"]),
                    "rejected": int(stats["rejected"]),
                    "superseded": int(stats["superseded"]),
                    "mean_wait_ms": round(stats["wait_ms_total"] / waits, 1) if waits else 0.0,
                    "max_wait_ms": round(stats["wait_ms_max"], 1),
                }
            leases = {
                name: {
                    "capacity": res.capacity,
                    "holders": [{"class": h.cls.label, "label": h.label,
                                 "held_s": round(now - (h.granted_at or now), 3),
                                 "preempt_requested": h.preempt_requested}
                                for h in res.holders],
                    "waiting": len(res.waiters),
                }
                for name, res in self._resources.items()
            }
            loops = {entry.name: {"class": entry.cls.label, "paused": entry.paused,
                                  "reason": entry.reason}
                     for entry in self._loops.values()}
            foreground = self._foreground_locked(now)
            shared = {"running": self._shared, "limit": self.shared_running}
        cpu_value = getattr(getattr(governor, "cpu", None), "value", None)
        gpu_value = getattr(getattr(governor, "gpu", None), "value", None)
        return {
            "classes": classes,
            "shared": shared,
            "foreground_active": foreground,
            "resources": {
                "memory": ("normal", "constrained", "pressured", "critical")[max(0, min(3, memory))],
                "cpu_percent": cpu_value,
                "cpu": "unknown" if cpu is None else ("busy" if cpu else "ok"),
                "gpu_mem_percent": gpu_value,
                "gpu": "unknown" if gpu is None else ("busy" if gpu else "ok"),
            },
            "leases": leases,
            "loops": loops,
            "closed": self._closed,
        }

    # ── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> int:
        """Refuse new work and withdraw everything still queued (shutdown).
        Running work is untouched — its owners release it. Returns how many
        waiters were withdrawn."""
        withdrawn: list[Ticket] = []
        with self._lock:
            self._closed = True
            for cls, queue in self._queues.items():
                while queue:
                    withdrawn.append(queue.popleft())
            for res in self._resources.values():
                withdrawn.extend(res.waiters)
                res.waiters.clear()
            for ticket in withdrawn:
                ticket.state = "cancelled"
                ticket.error = Busy("ORION is shutting down; this was not started.",
                                    cls=ticket.cls)
                ticket._event.set()
                self._stats[ticket.cls]["cancelled"] += 1
        handle, self._refresh_handle = self._refresh_handle, None
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass
        for ticket in withdrawn:
            self._notify(ticket)
        return len(withdrawn)

    def in_use(self) -> int:
        """Slots and leases currently held — zero after a clean shutdown."""
        with self._lock:
            return sum(self._running.values()) + sum(len(r.holders) for r in self._resources.values())


#: The process-wide policy. Consumers read ``admission.ADMISSION`` at call
#: time (never ``from .admission import ADMISSION``) so tests can swap it.
ADMISSION = Admission()


__all__ = ["ADMISSION", "Admission", "Busy", "ClassPolicy", "DEFAULT_POLICY",
           "Superseded", "Ticket", "WorkClass", "work_class"]
