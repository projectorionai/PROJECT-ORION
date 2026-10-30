"""Optional telemetry I/O cannot hold the GUI loop or outlive cancellation."""
import asyncio
import threading
from types import SimpleNamespace

from orion_core.app import _sample_metrics_history
from orion_core.telemetry import Telemetry


async def _wait_entered(entered, task):
    async def _poll():
        while not entered.is_set():
            if task.done():
                await task
                raise AssertionError("sampler task finished before entering target")
            await asyncio.sleep(0.001)

    await asyncio.wait_for(_poll(), timeout=5.0)


def test_slow_history_samples_do_not_block_the_event_loop():
    entered, release = threading.Event(), threading.Event()
    threads = []
    owner = threading.get_ident()

    def tick():
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(3)

    async def scenario():
        sampler = asyncio.create_task(_sample_metrics_history(SimpleNamespace(tick_history=tick), period=0.001))
        try:
            await _wait_entered(entered, sampler)
            # This turn can run while the store remains blocked in its thread.
            assert not release.is_set() and threads == [threads[0]] and threads[0] != owner
        finally:
            release.set()
            sampler.cancel()
            try:
                await sampler
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())


def test_cancel_during_history_initialization_closes_late_store(tmp_path, monkeypatch):
    from orion_core import metrics_history
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class History:
        def __init__(self, *_args, **_kwargs):
            entered.set()
            assert release.wait(3)

        def close(self):
            closed.set()

    monkeypatch.setattr(metrics_history, "MetricsHistory", History)
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    telemetry = Telemetry(bus)

    async def scenario():
        sampler = asyncio.create_task(_sample_metrics_history(telemetry, initialize_path=tmp_path / "metrics.db"))
        try:
            await _wait_entered(entered, sampler)
            sampler.cancel()
            try:
                await sampler
            except asyncio.CancelledError:
                pass
            assert telemetry.history is None
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 3)
        assert telemetry.history is None

    asyncio.run(scenario())
