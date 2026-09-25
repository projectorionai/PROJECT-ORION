"""
Background tasks that actually run, and that say so when they don't.

Two separate defects, both invisible, both fixed here.

**1. The task can be garbage-collected mid-flight.**

asyncio keeps only a WEAK reference to a running task. ``asyncio.create_task(
something())`` with the handle discarded is therefore a live object nobody
owns: the garbage collector is free to destroy it between one await and the
next, and the only trace is a "Task was destroyed but it is pending" warning —
if anything is watching stderr at all.

This is not theoretical here. app.py's own ``_boot_phase`` docstring records it
happening, by name:

    Every task that lost that race was then discarded ("Task was destroyed but
    it is pending"), which silently killed real startup work —
    TemporalPresence.prime_locality, the forge's persisted-tool reload, the
    plugin loader and the research director's resume all never ran.

The Qt-pump race that triggered it was fixed; the underlying ownership bug was
not, and fourteen startup tasks — including those same three — were still being
launched with their handles thrown away. ``command_router.run_soon`` already
solved exactly this for the GUI→loop direction, so the pattern was understood;
it just never reached app.py.

**2. A failing background task is silent.**

An exception inside a discarded task is never retrieved, so asyncio has nowhere
to report it and the work simply did not happen. ORION would start up looking
perfectly healthy with his research resume, his corpus build or his MCP
connection having thrown on line one. Every task spawned here reports its own
failure, once, in plain terms.

Cancellation is deliberately NOT reported: shutdown cancels everything still in
flight, and a dozen "task cancelled" lines on every clean exit is precisely the
noise this codebase has removed elsewhere.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Iterable

#: Strong references to everything in flight. THIS is the fix for defect 1 —
#: membership here is what keeps a task alive; the done-callback removes it, so
#: the set cannot grow without bound.
_RUNNING: set[asyncio.Task] = set()

#: Set once by app.py so failures have somewhere to be reported.
_BUS: Any = None


def attach_bus(bus: Any) -> None:
    """Wire the log so background failures become visible."""
    global _BUS
    _BUS = bus


def _report(task: asyncio.Task) -> None:
    """Retrieve the outcome so a failure is neither lost nor unretrieved."""
    _RUNNING.discard(task)
    if task.cancelled():
        return                      # shutdown cancels; that is not a fault
    exception = task.exception()
    if exception is None:
        return
    name = task.get_name() or "background task"
    message = f"{type(exception).__name__}: {exception}"
    if _BUS is not None:
        try:
            _BUS.log.emit(
                f"SYS: background task '{name}' failed - {message[:200]} "
                "(startup continued; that piece of work did not run)")
            return
        except Exception:
            pass
    print(f"SYS: background task '{name}' failed - {message[:200]}")


def spawn(coro: Awaitable[Any], name: str = "") -> asyncio.Task | None:
    """Start *coro* as a background task that is owned and observed.

    Use this instead of a bare ``asyncio.create_task`` for any work whose
    result nobody awaits — which is most of startup.

    Returns None when there is no running loop to attach to, having CLOSED the
    coroutine rather than leaving it to be collected. That distinction matters:
    an un-awaited coroutine emits "coroutine ... was never awaited" from
    whatever line the collector happens to run on, which is how a shutdown
    ends up printing warnings that point at unrelated code. Closing it is the
    same outcome, stated once, in the right place.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            coro.close()          # type: ignore[attr-defined]
        except AttributeError:
            pass
        if _BUS is not None:
            try:
                _BUS.log.emit(
                    f"SYS: '{name or 'background task'}' not started - "
                    "no running event loop (shutting down).")
            except Exception:
                pass
        return None
    task = asyncio.ensure_future(coro)
    if name:
        try:
            task.set_name(name)
        except AttributeError:      # a Future rather than a Task
            pass
    _RUNNING.add(task)
    task.add_done_callback(_report)
    return task


def in_flight() -> int:
    """How many owned background tasks are still running."""
    return sum(1 for task in _RUNNING if not task.done())


def names() -> list[str]:
    """Names of the tasks still running — for diagnostics that ask 'what is
    ORION still doing?' during a slow start."""
    return sorted(task.get_name() for task in _RUNNING if not task.done())


async def drain(timeout: float = 5.0) -> Iterable[str]:
    """Wait for background work to finish, returning anything still running.

    Bounded: shutdown must never hang on a task that will not end.
    """
    pending = [task for task in _RUNNING if not task.done()]
    if not pending:
        return []
    done, still = await asyncio.wait(pending, timeout=max(0.0, timeout))
    return sorted(task.get_name() for task in still)


def cancel_all() -> int:
    """Cancel everything still in flight. Returns how many were cancelled."""
    cancelled = 0
    for task in list(_RUNNING):
        if not task.done():
            task.cancel()
            cancelled += 1
    return cancelled


__all__ = ["attach_bus", "cancel_all", "drain", "in_flight", "names", "spawn"]
