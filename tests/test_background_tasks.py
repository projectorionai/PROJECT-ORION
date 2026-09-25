"""
Startup work that runs, and that says so when it doesn't.

app.py launched fourteen background tasks with their handles thrown away.
asyncio keeps only a WEAK reference to a running task, so each of those was a
live object nobody owned — collectable between one await and the next, with no
trace beyond a "Task was destroyed but it is pending" warning.

That is not a hypothetical. app.py's own _boot_phase docstring records it
happening, by name:

    Every task that lost that race was then discarded ("Task was destroyed but
    it is pending"), which silently killed real startup work —
    TemporalPresence.prime_locality, the forge's persisted-tool reload, the
    plugin loader and the research director's resume all never ran.

The Qt-pump race that triggered it got fixed. The ownership bug underneath did
not, and those same three calls were still launched with the handle discarded.

The second half is just as bad: an exception inside a discarded task is never
retrieved, so it is never reported. ORION could start up looking entirely
healthy with his research resume, corpus build or MCP connection having thrown
on line one.
"""

from __future__ import annotations

import asyncio
import gc
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import background  # noqa: E402


class _Log:
    def __init__(self):
        self.messages: list[str] = []

    def emit(self, message):
        self.messages.append(str(message))


class _Bus:
    def __init__(self):
        self.log = _Log()


@pytest.fixture(autouse=True)
def _clean():
    background.cancel_all()
    background._RUNNING.clear()
    background._BUS = None
    yield
    background.cancel_all()
    background._RUNNING.clear()
    background._BUS = None


# ── ownership: the actual bug ────────────────────────────────────────────────

def test_a_spawned_task_survives_losing_its_last_reference():
    """The defect. Without a strong reference the collector may destroy a
    running task, and the work silently never happens."""
    async def main():
        finished = []

        async def work():
            await asyncio.sleep(0.05)
            finished.append(True)

        task = background.spawn(work(), name="orion-work")
        del task
        gc.collect()
        gc.collect()
        assert background.in_flight() == 1, "the task was not retained"
        await asyncio.sleep(0.2)
        assert finished == [True], "the task was collected before it finished"

    asyncio.run(main())


def test_the_registry_does_not_grow_without_bound():
    """Retention must end when the task does, or this becomes a leak."""
    async def main():
        for index in range(20):
            background.spawn(asyncio.sleep(0), name=f"t{index}")
        await asyncio.sleep(0.1)
        assert background.in_flight() == 0
        assert len(background._RUNNING) == 0

    asyncio.run(main())


def test_running_tasks_can_be_named():
    """'What is ORION still doing?' during a slow start needs an answer."""
    async def main():
        background.spawn(asyncio.sleep(0.3), name="orion-forge-reload")
        assert "orion-forge-reload" in background.names()
        background.cancel_all()

    asyncio.run(main())


# ── visibility: the silent half ──────────────────────────────────────────────

def test_a_failing_background_task_is_reported():
    """Discarded tasks never have their exception retrieved, so a startup step
    could throw on line one and leave ORION looking perfectly healthy."""
    async def main():
        bus = _Bus()
        background.attach_bus(bus)

        async def boom():
            raise RuntimeError("could not reach the geo service")

        background.spawn(boom(), name="orion-prime-locality")
        await asyncio.sleep(0.05)
        assert bus.log.messages, "the failure was swallowed"
        line = bus.log.messages[0]
        assert "orion-prime-locality" in line, "the report does not say WHICH task"
        assert "geo service" in line, "the report does not say what went wrong"

    asyncio.run(main())


def test_a_failure_does_not_take_down_anything_else():
    async def main():
        background.attach_bus(_Bus())
        survived = []

        async def boom():
            raise ValueError("nope")

        async def fine():
            await asyncio.sleep(0.05)
            survived.append(True)

        background.spawn(boom(), name="bad")
        background.spawn(fine(), name="good")
        await asyncio.sleep(0.2)
        assert survived == [True]

    asyncio.run(main())


def test_reporting_survives_a_dead_bus():
    """The bus is torn down before the loop finishes draining at shutdown —
    reporting through a deleted Qt object must not raise from a callback."""
    class _Dead:
        class log:
            @staticmethod
            def emit(_):
                raise RuntimeError("wrapped C/C++ object of type OrionBus has been deleted")

    async def main():
        background.attach_bus(_Dead)

        async def boom():
            raise ValueError("x")

        background.spawn(boom(), name="t")
        await asyncio.sleep(0.05)      # must not raise

    asyncio.run(main())


def test_cancellation_is_not_reported_as_a_fault():
    """Shutdown cancels everything in flight. A dozen 'task cancelled' lines on
    every clean exit is exactly the noise this codebase has removed elsewhere."""
    async def main():
        bus = _Bus()
        background.attach_bus(bus)
        background.spawn(asyncio.sleep(60), name="orion-forever")
        assert background.cancel_all() == 1
        await asyncio.sleep(0.05)
        assert bus.log.messages == [], f"clean shutdown logged {bus.log.messages}"

    asyncio.run(main())


# ── draining ─────────────────────────────────────────────────────────────────

def test_drain_waits_for_work_and_reports_stragglers():
    async def main():
        background.spawn(asyncio.sleep(0.02), name="quick")
        assert list(await background.drain(timeout=1.0)) == []
        background.spawn(asyncio.sleep(30), name="slow")
        assert list(await background.drain(timeout=0.05)) == ["slow"]
        background.cancel_all()

    asyncio.run(main())


def test_drain_is_bounded():
    """Shutdown must never hang on a task that will not end."""
    async def main():
        background.spawn(asyncio.sleep(60), name="wedged")
        loop = asyncio.get_running_loop()
        started = loop.time()
        await background.drain(timeout=0.1)
        assert loop.time() - started < 2.0
        background.cancel_all()

    asyncio.run(main())


# ── app.py actually uses it ──────────────────────────────────────────────────

def _app_source() -> str:
    return (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8", errors="replace")


def test_no_fire_and_forget_tasks_remain_in_startup():
    """A bare create_task whose handle is discarded is the bug. One that is
    assigned or awaited is fine — the caller owns it."""
    orphans = []
    for line in _app_source().splitlines():
        stripped = line.strip()
        if stripped.startswith("asyncio.create_task(") or stripped.startswith(
                "lambda: asyncio.create_task("):
            orphans.append(stripped[:80])
    assert not orphans, f"unowned startup tasks: {orphans}"


def test_the_three_tasks_that_were_observed_to_vanish_are_owned():
    """prime_locality, the forge reload and the research resume are named in
    app.py's own record of this failure."""
    source = _app_source()
    for fragment in ("prime_locality", "reload_persisted_tools", "resume_pending"):
        pattern = re.compile(r"background\.spawn\(\s*[^)]*" + fragment)
        assert pattern.search(source), f"{fragment} is still not owned"


def test_failures_have_somewhere_to_be_reported():
    assert "background.attach_bus(bus)" in _app_source()


def test_background_work_is_cancelled_on_shutdown():
    assert "background.cancel_all" in _app_source()
