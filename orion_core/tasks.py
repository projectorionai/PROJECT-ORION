"""
Background tasks that survive being started.

``asyncio`` keeps only a **weak** reference to the tasks it is running. A task
created and immediately discarded —

    asyncio.get_running_loop().create_task(do_the_thing())

— is therefore eligible for garbage collection while it is still running, and
when it is collected it simply stops. No exception, no log line, no partial
result: the work quietly does not happen. The documentation says to keep a
reference and it is easy to read past, because the code looks finished and
works perfectly until the interpreter happens to collect at the wrong moment.

The second half of the same problem is the exception nobody retrieves. A task
that raises with nothing awaiting it holds the exception until collection and
then prints "Task exception was never retrieved" — which arrives detached from
whatever caused it, sometimes minutes later, and tells you almost nothing.

So :func:`spawn` holds the reference until the task finishes, and routes any
failure somewhere a person will see it attached to the thing that failed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

#: Tasks in flight. A strong reference each, discarded by the done-callback,
#: so this is a live set rather than a slow leak.
_RUNNING: set[asyncio.Task] = set()


def spawn(coro: Coroutine, *, name: str = "",
          on_error: Callable[[BaseException], Any] | None = None,
          bus: Any = None, label: str = "") -> asyncio.Task | None:
    """Start *coro* in the background and keep it alive until it finishes.

    Returns the task, or None when there is no running loop — callers are
    usually in a place where that means "nothing to do", not "crash".

    *on_error* is called with the exception if the task raises. Failing that,
    *bus* gets a log line naming *label*, because a background failure that
    goes nowhere is the same as no failure at all until something downstream
    is mysteriously empty.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        coro.close()            # never started; closing avoids a warning
        return None

    task = loop.create_task(coro, name=name or label or None)
    _RUNNING.add(task)

    def done(finished: asyncio.Task) -> None:
        _RUNNING.discard(finished)
        if finished.cancelled():
            return
        error = finished.exception()
        if error is None:
            return
        if on_error is not None:
            try:
                on_error(error)
                return
            except Exception:
                pass
        if bus is not None:
            try:
                what = label or name or "a background task"
                bus.log.emit(f"TASK: {what} failed - "
                             f"{str(error).splitlines()[0][:120]}")
                return
            except Exception:
                pass
        # Nowhere to report it. Retrieved anyway, so it does not resurface
        # later as "Task exception was never retrieved" with no context.

    task.add_done_callback(done)
    return task


def in_flight() -> int:
    """How many spawned tasks are still running. For diagnostics."""
    return len(_RUNNING)


async def drain(timeout: float = 5.0) -> int:
    """Wait for spawned tasks to finish, for a clean shutdown.

    Returns how many were still running when the timeout expired. Cancelling
    is deliberately NOT done here: several of these write to disk, and a
    half-written note is worse than a slow exit.
    """
    if not _RUNNING:
        return 0
    pending = set(_RUNNING)
    done, still = await asyncio.wait(pending, timeout=timeout)
    return len(still)


__all__ = ["drain", "in_flight", "spawn"]
