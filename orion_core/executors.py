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
"""

from __future__ import annotations

import asyncio
import functools
import os
import queue
import threading
import time
from concurrent.futures import Executor, Future
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


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
                 join_timeout: float = 1.0) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._max_workers = max_workers
        self._name = name
        self._join_timeout = join_timeout
        self._queue: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
        self._idle = threading.Semaphore(0)
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def submit(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> "Future[_T]":
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            future: "Future[_T]" = Future()
            self._queue.put((future, fn, args, kwargs))
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
            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:          # noqa: BLE001 - delivered to the caller
                    future.set_exception(exc)
                else:
                    future.set_result(result)
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
                        item[0].cancel()
            threads = list(self._threads)
            for _ in threads:
                self._queue.put(None)
        if wait:
            deadline = time.monotonic() + self._join_timeout
            for thread in threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))


def default_executor() -> DaemonThreadExecutor:
    """The pool main() installs as the event loop's default executor, sized as
    asyncio sizes its own (min(32, cpu_count + 4))."""
    return DaemonThreadExecutor(min(32, (os.cpu_count() or 1) + 4), name="orion-worker")


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
