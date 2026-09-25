"""Tests for the bounded bulk executor (Mark X.12 §5.3)."""

from __future__ import annotations

import asyncio
import threading
import time

from orion_core import executors
from orion_core.executors import bulk_executor, max_workers, run_bulk


def test_worker_count_is_bounded_and_small():
    assert 1 <= max_workers() <= 8
    assert bulk_executor()._max_workers == max_workers()


def test_run_bulk_returns_the_callables_result():
    async def drive():
        return await run_bulk(lambda a, b: a + b, 2, 3)
    assert asyncio.run(drive()) == 5


def test_run_bulk_runs_on_the_bulk_pool_not_the_default():
    async def drive():
        return await run_bulk(lambda: threading.current_thread().name)
    assert asyncio.run(drive()).startswith("orion-bulk")


def test_concurrency_never_exceeds_the_bound():
    peak = 0
    current = 0
    lock = threading.Lock()

    def work(i: int) -> int:
        nonlocal peak, current
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.05)          # hold the worker so tasks genuinely overlap
        with lock:
            current -= 1
        return i

    async def drive():
        return await asyncio.gather(*(run_bulk(work, i) for i in range(8)))

    results = asyncio.run(drive())
    assert sorted(results) == list(range(8))     # all ran to completion
    assert peak <= max_workers()                 # but never more than the bound at once


# ── lifecycle: the pool that nobody shut down (Mark XXVI) ───────────────────
#
# executors.shutdown_bulk_executor() has always existed and was never called.
# The pool was a ThreadPoolExecutor, whose non-daemon workers concurrent.futures
# joins at exit — a bulk job still running at shutdown then held the whole
# process open. Calling shutdown(wait=False) did NOT fix that on its own (the
# exit hook joins them regardless); the pool is a DaemonThreadExecutor now.

import time as _time
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]


def teardown_function(_fn):
    executors.shutdown_bulk_executor(wait=False)


def test_the_pool_is_created_lazily_and_reused():
    executors.shutdown_bulk_executor(wait=False)
    assert executors._bulk_executor is None
    first = executors.bulk_executor()
    assert first is executors.bulk_executor(), "a new pool per call would leak"


def test_shutdown_releases_the_pool():
    executors.bulk_executor()
    executors.shutdown_bulk_executor(wait=False)
    assert executors._bulk_executor is None


def test_shutdown_is_safe_when_nothing_was_created():
    executors.shutdown_bulk_executor(wait=False)
    executors.shutdown_bulk_executor(wait=False)      # must not raise


def test_the_pool_rebuilds_after_shutdown():
    """A restart-in-place must not leave ORION with a dead pool."""
    executors.shutdown_bulk_executor(wait=False)
    pool = executors.bulk_executor()
    assert pool.submit(lambda: 21 * 2).result(timeout=30) == 42


def test_shutdown_does_not_wait_for_running_work():
    """wait=False is the point: shutdown must not block on a long ingest."""
    pool = executors.bulk_executor()
    pool.submit(lambda: _time.sleep(0.4))
    started = _time.perf_counter()
    executors.shutdown_bulk_executor(wait=False)
    assert (_time.perf_counter() - started) < 0.3, (
        "shutdown blocked on in-flight bulk work")


def test_the_shutdown_sequence_actually_calls_it():
    src = (_ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "shutdown_bulk_executor" in src, (
        "the bulk pool is never shut down — its non-daemon threads are left to "
        "concurrent.futures' atexit hook, which can hang the process")
    assert "wait=False" in src, "shutdown must not block on in-flight bulk work"


def test_it_is_shut_down_before_the_stores_are_closed():
    """Bulk work touches the databases. Closing them first would let an
    in-flight job write to a closed handle."""
    src = (_ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert src.index("shutdown_bulk_executor") < src.index('_safe("memory"')


# ── DaemonThreadExecutor: work in flight can never hold the exit ────────────

def test_workers_are_daemon_threads():
    pool = executors.DaemonThreadExecutor(2, name="probe")
    try:
        assert pool.submit(lambda: threading.current_thread().daemon).result(timeout=10)
    finally:
        pool.shutdown()


def test_shutdown_waits_only_a_bounded_time_for_stuck_work():
    """qasync's close() calls shutdown() with wait=True. That wait is what held
    ORION.exe open after teardown; here it is bounded."""
    release = threading.Event()
    pool = executors.DaemonThreadExecutor(2, name="probe", join_timeout=0.3)
    pool.submit(release.wait)
    started = _time.perf_counter()
    pool.shutdown(wait=True)
    elapsed = _time.perf_counter() - started
    release.set()
    assert elapsed < 1.5, f"shutdown blocked {elapsed:.2f}s on in-flight work"


def test_results_and_exceptions_reach_the_caller():
    pool = executors.DaemonThreadExecutor(2, name="probe")
    try:
        assert pool.submit(pow, 2, 10).result(timeout=10) == 1024
        failing = pool.submit(int, "not a number")
        try:
            failing.result(timeout=10)
        except ValueError:
            pass
        else:
            raise AssertionError("the exception was swallowed")
    finally:
        pool.shutdown()


def test_an_idle_worker_is_reused_before_another_starts():
    pool = executors.DaemonThreadExecutor(4, name="probe")
    try:
        names = {pool.submit(lambda: threading.current_thread().name).result(timeout=10)
                 for _ in range(5)}
        assert names == {"probe_0"}, names
    finally:
        pool.shutdown()


def test_nothing_new_is_accepted_after_shutdown_and_queued_work_can_be_cancelled():
    release = threading.Event()
    pool = executors.DaemonThreadExecutor(1, name="probe", join_timeout=0.1)
    pool.submit(release.wait)
    queued = pool.submit(lambda: "never")
    pool.shutdown(wait=False, cancel_futures=True)
    release.set()
    assert queued.cancelled()
    try:
        pool.submit(lambda: None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a shut-down pool accepted work")


def test_the_event_loop_can_run_work_on_it():
    """What qasync does with its default executor for every asyncio.to_thread.
    (Plain asyncio's set_default_executor only accepts a ThreadPoolExecutor;
    qasync's does not check, which is what ORION runs on.)"""
    pool = executors.default_executor()

    async def drive():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, lambda: threading.current_thread().name)
    try:
        assert asyncio.run(drive()).startswith("orion-worker")
    finally:
        pool.shutdown()


def test_main_installs_it_as_the_loops_default_executor():
    src = (_ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "loop.set_default_executor(_default_executor())" in src
