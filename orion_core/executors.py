"""
Bounded executor for bulk work (Mark X.12 §5.3).

``asyncio.to_thread`` runs on the event loop's *default* executor — a pool of
``min(32, cpu_count + 4)`` threads shared by everything, including
latency-sensitive audio persistence and bus-critical work. Bulk, best-effort
jobs — ``learn_folder`` walking a large tree, OCR over many files, folder
ingestion — can saturate that pool and add head-of-line latency to a voice turn.

This gives bulk work its own small, bounded pool so it can never starve the
default one. ``run_bulk(fn, *args)`` is a drop-in for ``asyncio.to_thread(fn,
*args)`` for anything throughput-oriented rather than latency-sensitive. The
worker count is deliberately low (``ORION_BULK_WORKERS`` overrides, default 2):
the point is to *cede* CPU to the latency path, not to race it.

Both pools — this one and the loop's default executor, which main() installs
from ``default_executor()`` — are ``DaemonThreadExecutor``s, so work still in
flight when ORION quits can never hold the process open. See that class.

Both pools also BOUND their pending queue. It was an unbounded SimpleQueue: a
burst of ``to_thread`` calls behind sixteen busy workers queued without limit,
each one adding head-of-line delay to whatever came after — including the
voice path's own persistence. Past ``max_pending`` a submit now fails at once
with ``ExecutorBusy`` instead. Failing is deliberate: blocking the submitter
would stall the event loop (``run_in_executor`` submits from it) and could
deadlock a worker that submits to its own pool, while queueing further only
moves the delay somewhere nobody can see it. Ordering and priority between
kinds of work is ``admission``'s job; this is the backstop underneath it.
"""

from __future__ import annotations

import asyncio
import functools
import os
import queue
import threading
import time
from contextvars import copy_context
from concurrent.futures import Executor, Future
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


class ExecutorBusy(RuntimeError):
    """A pool's pending queue is full; the call was refused, not queued."""


def _default_max_pending(max_workers: int) -> int:
    raw = os.getenv("ORION_EXECUTOR_MAX_PENDING", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    # Generous: normal operation peaks at tens of queued calls. Hitting this
    # means minutes of backlog, at which point an error beats more waiting.
    return max(256, 64 * max_workers)


class DaemonThreadExecutor(Executor):
    """A thread pool whose workers can never keep the process alive.

    Both pools ORION had held the exit hostage to whatever was still running:

    * the event loop's default executor (every ``asyncio.to_thread``) is
      qasync's QThreadExecutor, and qasync's ``close()`` shuts it down with
      ``wait=True`` — so a scan or network probe still in flight at quit held
      the process after teardown had finished. Measured on 2026-09-23: teardown
      done at 8.0 s, ORION.exe gone at 13.7 s, the WebEngine children long
      since exited. Its workers are QThreads, invisible to
      ``threading.enumerate()``, so nothing reported what was being waited for.
    * the bulk pool was a ThreadPoolExecutor. ``shutdown(wait=False)`` looked
      like the fix, but concurrent.futures registers its own exit hook that
      joins every worker regardless, so a long ingest still held the exit.

    Here workers are daemon threads, and ``shutdown(wait=True)`` waits at most
    ``join_timeout`` seconds in total. Work that outlives that is abandoned
    when the interpreter exits, which is the right trade on the way out:
    teardown has already closed the stores, the JSON state is written
    atomically, and SQLite rolls back an interrupted transaction.

    Threads start on demand, up to ``max_workers``, and an idle one is reused
    before another is started — the same policy as ThreadPoolExecutor.
    """

    def __init__(self, max_workers: int, *, name: str = "orion-worker",
                 join_timeout: float = 1.0, max_pending: int | None = None) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._max_workers = max_workers
        self._name = name
        self._join_timeout = join_timeout
        self._max_pending = int(max_pending) if max_pending and max_pending > 0 \
            else _default_max_pending(max_workers)
        # Still a SimpleQueue — put() never blocks — with the bound enforced by
        # the counter below under the lock, so a full pool refuses instead of
        # stalling the caller.
        self._queue: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
        self._idle = threading.Semaphore(0)
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False
        self._pending = 0
        self._busy = 0
        self._high_water = 0
        self._rejected = 0

    def submit(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> "Future[_T]":
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if self._pending >= self._max_pending:
                self._rejected += 1
                raise ExecutorBusy(
                    f"{self._name}: {self._pending} calls are already waiting for "
                    f"{self._max_workers} worker(s); refusing rather than queueing "
                    "without bound")
            future: "Future[_T]" = Future()
            self._pending += 1
            if self._pending > self._high_water:
                self._high_water = self._pending
            # Unlike asyncio.to_thread, direct run_in_executor/submit does
            # not copy task-local policy. Bind every owned pool invocation
            # to a fresh context so restrictions and revocation survive it.
            context = copy_context()
            self._queue.put((future, functools.partial(context.run, fn), args, kwargs))
            if not self._idle.acquire(timeout=0) and len(self._threads) < self._max_workers:
                thread = threading.Thread(target=self._work, daemon=True,
                                          name=f"{self._name}_{len(self._threads)}")
                self._threads.append(thread)
                thread.start()
            return future

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, fn, args, kwargs = item
            del item
            with self._lock:
                self._pending -= 1
                self._busy += 1
            try:
                # A future cancelled while queued is skipped here. One already
                # running is NOT stopped by cancel(): Python cannot interrupt a
                # native call, so long work must poll a checkpoint instead.
                if future.set_running_or_notify_cancel():
                    try:
                        result = fn(*args, **kwargs)
                    except BaseException as exc:          # noqa: BLE001 - delivered to the caller
                        future.set_exception(exc)
                    else:
                        future.set_result(result)
            finally:
                with self._lock:
                    self._busy -= 1
            del future, fn, args, kwargs
            self._idle.release()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            if cancel_futures:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None:
                        self._pending -= 1
                        item[0].cancel()
            threads = list(self._threads)
            for _ in threads:
                self._queue.put(None)
        if wait:
            deadline = time.monotonic() + self._join_timeout
            for thread in threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def stats(self) -> dict[str, Any]:
        """Depth and pressure, for the status view and the admission snapshot."""
        with self._lock:
            return {"name": self._name, "workers": len(self._threads),
                    "max_workers": self._max_workers, "busy": self._busy,
                    "pending": self._pending, "max_pending": self._max_pending,
                    "high_water": self._high_water, "rejected": self._rejected,
                    "shutdown": self._shutdown}


_default_pool: DaemonThreadExecutor | None = None


def default_executor() -> DaemonThreadExecutor:
    """The pool main() installs as the event loop's default executor, sized as
    asyncio sizes its own (min(32, cpu_count + 4))."""
    global _default_pool
    _default_pool = DaemonThreadExecutor(min(32, (os.cpu_count() or 1) + 4),
                                         name="orion-worker")
    return _default_pool


def pool_stats() -> dict[str, Any]:
    """Both pools' depth — whichever exist yet."""
    out: dict[str, Any] = {}
    if _default_pool is not None:
        out["default"] = _default_pool.stats()
    if _bulk_executor is not None:
        out["bulk"] = _bulk_executor.stats()
    return out


def _configured_workers() -> int:
    raw = os.getenv("ORION_BULK_WORKERS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return min(int(raw), 8)
    return 2


_BULK_MAX_WORKERS = _configured_workers()
_bulk_executor: DaemonThreadExecutor | None = None


def max_workers() -> int:
    return _BULK_MAX_WORKERS


def bulk_executor() -> DaemonThreadExecutor:
    """The lazily-created, process-wide bounded pool for bulk work."""
    global _bulk_executor
    if _bulk_executor is None:
        _bulk_executor = DaemonThreadExecutor(_BULK_MAX_WORKERS, name="orion-bulk")
    return _bulk_executor


async def run_bulk(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run *fn* on the bounded bulk pool. Drop-in for ``asyncio.to_thread`` for
    throughput-oriented work that must not starve latency-sensitive tasks."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(bulk_executor(), functools.partial(fn, *args, **kwargs))


def shutdown_bulk_executor(wait: bool = False) -> None:
    global _bulk_executor
    if _bulk_executor is not None:
        _bulk_executor.shutdown(wait=wait)
        _bulk_executor = None
