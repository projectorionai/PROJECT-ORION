"""Background tasks that survive being started.

asyncio keeps only a WEAK reference to running tasks, so this —

    asyncio.get_running_loop().create_task(do_the_thing())

— is eligible for collection while it is still running, and when it is
collected it simply stops. No exception, no log line, no partial result. The
code looks finished and works perfectly until the interpreter collects at the
wrong moment, which is the worst kind of bug to have in something that reports
success before the work is done.

Two places in ORION did exactly that: attaching research to a mission (which
said "attached" while the topic might never reach the agenda) and running a
protocol from the entrepreneur panel.
"""

from __future__ import annotations

import asyncio
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import tasks  # noqa: E402


class _Signal:
    def __init__(self):
        self.lines = []

    def emit(self, *args, **kwargs):
        if args:
            self.lines.append(str(args[0]))


class _Bus:
    def __init__(self):
        self.log = _Signal()


# ── the reference is held ────────────────────────────────────────────────────

async def test_a_spawned_task_runs_to_completion():
    done = []

    async def work():
        await asyncio.sleep(0)
        done.append(True)

    task = tasks.spawn(work())
    assert task is not None
    await asyncio.sleep(0.05)
    assert done == [True]


async def test_a_spawned_task_survives_a_collection():
    """The actual failure mode. Nothing else references it, so without a
    strong reference here the collector is free to take it mid-flight."""
    done = []

    async def work():
        await asyncio.sleep(0.02)
        done.append(True)

    tasks.spawn(work())          # deliberately not stored by the caller
    gc.collect()
    await asyncio.sleep(0.1)
    assert done == [True], "the task was collected before it finished"


async def test_the_reference_is_released_afterwards():
    """Holding them forever would trade a lost task for a slow leak."""
    async def work():
        await asyncio.sleep(0)

    before = tasks.in_flight()
    tasks.spawn(work())
    await asyncio.sleep(0.05)
    assert tasks.in_flight() == before


# ── failures go somewhere ────────────────────────────────────────────────────

async def test_a_failure_reaches_the_handler():
    seen = []

    async def boom():
        raise ValueError("no")

    tasks.spawn(boom(), on_error=seen.append)
    await asyncio.sleep(0.05)
    assert seen and isinstance(seen[0], ValueError)


async def test_a_failure_reaches_the_bus_when_there_is_no_handler():
    """A background failure that goes nowhere is the same as no failure at
    all, until something downstream is mysteriously empty."""
    bus = _Bus()

    async def boom():
        raise ValueError("the thing broke")

    tasks.spawn(boom(), bus=bus, label="doing the thing")
    await asyncio.sleep(0.05)
    assert any("doing the thing" in line for line in bus.log.lines)
    assert any("the thing broke" in line for line in bus.log.lines)


async def test_a_failure_with_nowhere_to_go_is_still_retrieved():
    """Otherwise it resurfaces at collection as "Task exception was never
    retrieved", detached from whatever caused it."""
    async def boom():
        raise ValueError("no")

    task = tasks.spawn(boom())
    await asyncio.sleep(0.05)
    assert task.done()
    assert task.exception() is not None      # retrieved, not left dangling


async def test_a_broken_handler_does_not_break_the_callback():
    bus = _Bus()

    async def boom():
        raise ValueError("no")

    def bad(_exc):
        raise RuntimeError("the handler is broken too")

    tasks.spawn(boom(), on_error=bad, bus=bus, label="x")
    await asyncio.sleep(0.05)
    assert bus.log.lines, "the bus was not used after the handler failed"


async def test_a_cancelled_task_is_not_reported_as_a_failure():
    bus = _Bus()

    async def slow():
        await asyncio.sleep(10)

    task = tasks.spawn(slow(), bus=bus, label="slow")
    task.cancel()
    await asyncio.sleep(0.05)
    assert bus.log.lines == []


# ── outside a loop ───────────────────────────────────────────────────────────

def test_with_no_running_loop_it_declines_rather_than_raising():
    """Callers are usually somewhere that means "nothing to do", not
    "crash" — a GUI handler that may or may not be inside qasync."""
    async def work():
        pass

    assert tasks.spawn(work()) is None


# ── the two call sites ───────────────────────────────────────────────────────

def test_attaching_research_to_a_mission_holds_its_task():
    source = (ROOT / "orion_core" / "missions.py").read_text(encoding="utf-8")
    assert "from .tasks import spawn" in source
    # The CALL, not the word: the comment above it explains why create_task
    # is wrong here, and matching prose failed a correct implementation.
    assert ".create_task(" not in source, (
        "a bare create_task is back; it can be collected mid-flight")


def test_the_entrepreneur_panel_holds_its_task():
    source = (ROOT / "orion_core" / "gui" / "entrepreneur.py").read_text(
        encoding="utf-8")
    assert "spawn(" in source
    assert "create_task(_go())" not in source
