"""
Shared admission — one policy for who runs, who waits and who is told no.

The review of 2026-09-26 found every limit in ORION was local: an unbounded
executor queue, unbounded training queues, four jobs, six tool calls, eight
workflows, none aware of the others. These tests hold the shared policy to
what it promises:

  * a burst never grows a queue past its bound, and overload is VISIBLE
    (Busy with a reason, counters in the snapshot) rather than a silent drop;
  * waiting work ages, so background work is not starved by a stream of
    higher-priority jobs;
  * a newer refresh replaces a queued one instead of queueing behind it;
  * CPU/GPU pressure defers optional work with hysteresis, and missing GPU
    telemetry reads as unknown, not free;
  * the GPU lease is exclusive, and a waiting higher class asks the holder
    to yield;
  * shutdown withdraws every waiter and leaves nothing held;
  * nothing here ever blocks the event loop.

Offline: fake clocks and fake readers, no hardware.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import request_trace  # noqa: E402
from orion_core.admission import (  # noqa: E402
    Admission,
    Busy,
    ClassPolicy,
    Superseded,
    WorkClass,
    work_class,
)
from orion_core.resource_governor import LoadGauge, ResourceGovernor  # noqa: E402


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Metrics:
    def __init__(self) -> None:
        self.counts: dict[str, float] = {}
        self.gauges: dict[str, float] = {}
        self.observed: list[tuple[str, float]] = []

    def incr(self, name, amount=1.0):
        self.counts[name] = self.counts.get(name, 0.0) + amount

    def gauge(self, name, value):
        self.gauges[name] = value

    def observe(self, name, ms):
        self.observed.append((name, ms))


class _Governor:
    """Just the surface admission reads from the resource governor."""

    def __init__(self) -> None:
        self.acting = True
        self.memory = 0
        self.cpu_state = None
        self.gpu_state = None

    def effective(self):
        return self.memory

    def cpu_busy(self):
        return self.cpu_state

    def gpu_busy(self):
        return self.gpu_state


def _admission(**policy) -> tuple[Admission, _Clock, _Governor]:
    clock = _Clock()
    governor = _Governor()
    table = {work_class(name): ClassPolicy(*limits) for name, limits in policy.items()}
    return Admission(table, clock=clock, governor=governor, quiet_s=4.0), clock, governor


# ── bounded bursts and visible overload ──────────────────────────────────────

def test_a_burst_keeps_a_stable_bounded_depth_and_every_refusal_is_visible():
    admission, _, _ = _admission(user=(2, 5))
    admission.metrics = metrics = _Metrics()
    accepted, refused = [], []
    for index in range(50):
        try:
            accepted.append(admission.submit("user", label=f"job {index}"))
        except Busy as busy:
            refused.append(busy)
    snapshot = admission.snapshot()["classes"]["user"]
    assert snapshot["running"] == 2 and snapshot["pending"] == 5, "depth must stay bounded"
    assert len(refused) == 43 and snapshot["rejected"] == 43
    assert "already waiting" in refused[0].reason, "a refusal must say why"
    assert refused[0].to_dict()["busy"] is True and not refused[0].to_tool_result().ok
    assert metrics.counts["admission.rejected.user"] == 43
    assert [t.position() for t in accepted[2:]] == [1, 2, 3, 4, 5]


def test_submit_never_waits_even_when_everything_is_full():
    admission, _, _ = _admission(background=(1, 1))
    admission.submit("background")
    admission.submit("background")
    started = time.perf_counter()
    with pytest.raises(Busy):
        admission.submit("background")
    assert time.perf_counter() - started < 0.05


def test_interactive_work_does_not_draw_on_the_shared_budget():
    admission, _, _ = _admission(user=(4, 4), background=(4, 4))
    admission.shared_running = 2
    admission.submit("user")
    admission.submit("background")
    assert admission.submit("background").queued, "the shared budget is spent"
    assert admission.submit("interactive").running, "a user turn is never behind it"


# ── ageing: no starvation ────────────────────────────────────────────────────

def _one_shared_slot(age_step_s: float):
    clock = _Clock()
    admission = Admission({WorkClass.USER: ClassPolicy(1, 8),
                           WorkClass.BACKGROUND: ClassPolicy(1, 8)},
                          shared_running=1, age_step_s=age_step_s, clock=clock,
                          governor=_Governor())
    return admission, clock


def test_waiting_background_work_ages_past_newer_user_work():
    admission, clock = _one_shared_slot(age_step_s=20.0)
    running = admission.submit("user")
    background = admission.submit("background", label="index refresh")
    clock.now += 30.0                           # background has aged one class
    newer_user = admission.submit("user", label="later job")
    running.release()
    assert background.running, "aged background work must get the freed slot"
    assert newer_user.queued


def test_without_ageing_the_higher_class_would_win():
    """The control case: it is the ageing that changes the outcome."""
    admission, clock = _one_shared_slot(age_step_s=1e9)
    running = admission.submit("user")
    background = admission.submit("background")
    clock.now += 30.0
    newer_user = admission.submit("user")
    running.release()
    assert newer_user.running and background.queued


# ── coalescing replaceable work ──────────────────────────────────────────────

async def test_a_newer_refresh_replaces_the_queued_one_and_keeps_its_age():
    admission, clock, _ = _admission(background=(1, 4))
    blocker = admission.submit("background")
    first = admission.submit("background", key="index:notes", replaceable=True)
    clock.now += 5.0
    second = admission.submit("background", key="index:notes", replaceable=True)
    assert admission.snapshot()["classes"]["background"]["pending"] == 1
    assert second.enqueued_at == first.enqueued_at, "replacing must not reset the clock"
    with pytest.raises(Superseded):
        await first.wait(timeout=1.0)
    blocker.release()
    await second.wait(timeout=1.0)
    assert second.running


def test_non_replaceable_work_with_the_same_key_is_queued_not_merged():
    admission, _, _ = _admission(background=(1, 4))
    admission.submit("background")
    admission.submit("background", key="send", replaceable=False)
    admission.submit("background", key="send", replaceable=False)
    assert admission.snapshot()["classes"]["background"]["pending"] == 2


# ── resource pressure with hysteresis ────────────────────────────────────────

def test_the_load_gauge_needs_sustained_readings_both_ways():
    gauge = LoadGauge(enter=85, leave=70)
    assert gauge.observe(95) is False, "one spike changes nothing"
    assert gauge.observe(95) is True
    assert gauge.observe(80) is True, "between the thresholds it holds"
    gauge.observe(60)
    gauge.observe(60)
    assert gauge.state is True
    assert gauge.observe(60) is False
    assert gauge.observe(None) is None, "no reading is unknown, not idle"


def test_the_governor_samples_cpu_and_gpu_on_its_tick_and_reports_unknown():
    readings = iter([90.0, 90.0, 90.0])
    governor = ResourceGovernor(reader=lambda: (50.0, 8000.0),
                                cpu_reader=lambda: next(readings),
                                gpu_reader=lambda: None)
    seen = []
    governor.add_listener(lambda g: seen.append(g.cpu_busy()))
    governor.tick()
    governor.tick()
    assert seen == [False, True]
    status = governor.status()
    assert status["cpu_busy"] is True and status["cpu_percent"] == 90.0
    assert status["gpu_busy"] is None and status["gpu_memory_percent"] is None


def test_cpu_pressure_defers_training_and_throttles_background():
    admission, _, governor = _admission(background=(2, 8), training=(1, 4))
    governor.cpu_state = True
    first = admission.submit("background")
    second = admission.submit("background")
    training = admission.submit("training")
    assert first.running and second.queued, "background keeps ONE slot under load"
    assert training.queued, "training waits for a quiet machine"
    assert admission.defer_reason("training") == "the processor is busy"
    governor.cpu_state = False
    admission.refresh()
    assert second.running and training.running


def test_memory_pressure_stops_background_but_not_user_work():
    admission, _, governor = _admission()
    governor.memory = 2                         # PRESSURED
    assert admission.submit("background").queued
    assert admission.submit("user").running


def test_unknown_gpu_telemetry_is_reported_as_unknown_not_free():
    admission, _, governor = _admission()
    governor.gpu_state = None
    snapshot = admission.snapshot()["resources"]
    assert snapshot["gpu"] == "unknown"
    governor.gpu_state = True
    assert admission.snapshot()["resources"]["gpu"] == "busy"
    assert admission.defer_reason("training") == "graphics memory is nearly full"


# ── the GPU lease ────────────────────────────────────────────────────────────

async def test_training_and_generation_never_hold_the_gpu_together():
    admission, _, _ = _admission()
    holders: list[str] = []
    peak = 0
    training_entered = asyncio.Event()

    async def use(cls: str, hold_s: float, entered: asyncio.Event | None = None):
        nonlocal peak
        async with admission.lease("gpu", cls=cls, label=cls) as lease:
            holders.append(cls)
            peak = max(peak, len(holders))
            if entered is not None:
                entered.set()
            deadline = time.monotonic() + hold_s
            while time.monotonic() < deadline and not lease.preempt_requested:
                await asyncio.sleep(0.005)      # a cooperative checkpoint
            holders.remove(cls)
            return lease.preempt_requested

    training = asyncio.create_task(use("training", 5.0, training_entered))
    await training_entered.wait()
    generation = asyncio.create_task(use("interactive", 0.01))
    asked_to_yield = await asyncio.wait_for(training, 2.0)
    await asyncio.wait_for(generation, 2.0)
    assert peak == 1, "the lease must be exclusive"
    assert asked_to_yield, "a waiting interactive request asks training to yield"
    assert admission.in_use() == 0


def test_a_worker_thread_can_hold_the_lease_while_the_loop_waits():
    admission, _, _ = _admission()
    order: list[str] = []
    ready = threading.Event()

    def trainer():
        with admission.lease_blocking("gpu", cls="training", label="intent pass"):
            order.append("training")
            ready.set()
            time.sleep(0.1)
            order.append("training done")

    thread = threading.Thread(target=trainer)
    thread.start()
    ready.wait(2.0)

    async def generate():
        async with admission.lease("gpu", cls="interactive"):
            order.append("generation")

    asyncio.run(asyncio.wait_for(generate(), 2.0))
    thread.join(2.0)
    assert order == ["training", "training done", "generation"]


# ── waiting safely ───────────────────────────────────────────────────────────

async def test_a_queued_ticket_is_granted_across_threads():
    admission, _, _ = _admission(user=(1, 4))
    holder = admission.submit("user")
    waiter = admission.submit("user")
    threading.Timer(0.05, holder.release).start()
    await asyncio.wait_for(waiter.wait(), 2.0)
    assert waiter.running


async def test_blocking_wait_refuses_to_stall_the_event_loop():
    admission, _, _ = _admission(user=(1, 4))
    admission.submit("user")
    queued = admission.submit("user")
    with pytest.raises(RuntimeError, match="event loop"):
        queued.wait_blocking(1.0)


async def test_a_timed_out_waiter_leaves_the_queue():
    admission, _, _ = _admission(user=(1, 4))
    admission.submit("user")
    queued = admission.submit("user")
    with pytest.raises(Busy, match="withdrawn"):
        await queued.wait(timeout=0.02)
    assert admission.snapshot()["classes"]["user"]["pending"] == 0


async def test_cancelling_a_waiter_withdraws_it():
    admission, _, _ = _admission(user=(1, 4))
    admission.submit("user")
    queued = admission.submit("user")
    task = asyncio.create_task(queued.wait())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert queued.state == "cancelled"
    assert admission.snapshot()["classes"]["user"]["pending"] == 0


async def test_the_queue_wait_is_marked_on_the_request_that_waited():
    admission, _, _ = _admission(user=(1, 4))
    holder = admission.submit("user")
    trace = request_trace.TraceStore().begin("remote")
    with request_trace.bound(trace):
        queued = admission.submit("user")
        asyncio.get_running_loop().call_later(0.02, holder.release)
        await queued.wait(1.0)
    marks = trace.marks_named("queue_wait")
    assert marks and marks[-1].fields["cls"] == "user"


# ── optional loops ───────────────────────────────────────────────────────────

def test_optional_loops_pause_for_the_foreground_and_resume_after_it():
    admission, clock, governor = _admission()
    events: list[str] = []
    admission.register_loop("research refresh", pause=lambda: events.append("pause"),
                            resume=lambda: events.append("resume"))
    assert admission.should_run("research refresh")
    admission.note_foreground("voice")
    assert events == ["pause"] and not admission.should_run("research refresh")
    assert admission.snapshot()["loops"]["research refresh"]["paused"]
    clock.now += 5.0                            # past the quiet window
    admission.refresh()
    assert events == ["pause", "resume"]
    governor.memory = 2
    admission.refresh()
    assert events[-1] == "pause", "memory pressure defers it too"


def test_an_interactive_slot_counts_as_foreground():
    admission, _, _ = _admission()
    ticket = admission.submit("interactive", label="voice turn")
    assert admission.foreground_active()
    assert admission.defer_reason("background") == "a request is being answered"
    ticket.release()


async def test_withdrawing_a_queued_ticket_wakes_its_waiter():
    """A pause withdraws a queued job from the queue; its waiter must wake and
    learn it was withdrawn, not sleep forever or be told it was admitted."""
    admission, _, _ = _admission(user=(1, 4))
    admission.submit("user")
    queued = admission.submit("user")
    waiting = asyncio.create_task(queued.wait())
    await asyncio.sleep(0)
    queued.release()
    with pytest.raises(Busy, match="withdrawn"):
        await asyncio.wait_for(waiting, 1.0)


def test_reprioritising_moves_a_queued_ticket_and_keeps_its_age():
    admission, clock, _ = _admission(user=(1, 4), background=(1, 4))
    admission.shared_running = 1
    running = admission.submit("user")
    background = admission.submit("background", label="index")
    clock.now += 5
    later_user = admission.submit("user", label="report")
    admission.reprioritise(background, "user")
    assert background.cls is WorkClass.USER
    assert background.position() == 1, "it keeps its place by arrival time"
    running.release()
    assert background.running and later_user.queued


def test_a_gpu_lease_is_refused_visibly_when_too_many_wait():
    from orion_core import admission as admission_mod
    admission, _, _ = _admission()
    holder = admission.acquire_lease("gpu", "training")
    waiters = [admission.acquire_lease("gpu", "background")
               for _ in range(admission_mod.LEASE_PENDING)]
    with pytest.raises(Busy, match="gpu is busy"):
        admission.acquire_lease("gpu", "interactive")
    assert holder.preempt_requested is True, "training is the most optional work there is"
    for ticket in waiters + [holder]:
        ticket.release()
    assert admission.in_use() == 0


# ── shutdown ─────────────────────────────────────────────────────────────────

async def test_shutdown_withdraws_waiters_and_leaves_nothing_held():
    admission, _, _ = _admission(user=(1, 4))
    running = admission.submit("user")
    queued = admission.submit("user")
    lease_holder = admission.acquire_lease("gpu", "training")
    lease_waiter = admission.acquire_lease("gpu", "interactive")
    waiting = asyncio.create_task(queued.wait())
    await asyncio.sleep(0)
    assert admission.close() == 2
    with pytest.raises(Busy, match="shutting down"):
        await waiting
    with pytest.raises(Busy):
        await lease_waiter.wait(0.1)
    with pytest.raises(Busy):
        admission.submit("user")
    running.release()
    lease_holder.release()
    assert admission.in_use() == 0


def test_work_class_accepts_names_and_rejects_nonsense():
    assert work_class("voice") is WorkClass.INTERACTIVE
    assert work_class(3) is WorkClass.TRAINING
    with pytest.raises(ValueError):
        work_class("urgent-ish")
