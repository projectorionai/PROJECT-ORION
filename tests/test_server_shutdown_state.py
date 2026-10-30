"""Headless shutdown must close persisted stores despite another stop failing."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from orion_core import server
from orion_core.server import _close_headless_state


def test_headless_closes_every_store_and_checkpoints_after_stop_failures():
    calls: list[str] = []
    messages: list[str] = []

    class Gateway:
        async def stop(self):
            calls.append("gateway")
            raise RuntimeError("gateway failed")

    class History:
        def close(self):
            calls.append("history")
            raise RuntimeError("history failed")

    class Memory:
        def close(self):
            calls.append("memory")
            raise RuntimeError("memory failed")

    class Housekeeper:
        def checkpoint_all(self):
            calls.append("checkpoint")
            raise RuntimeError("checkpoint failed")

    telemetry = SimpleNamespace(history=History())
    bus = SimpleNamespace(log=SimpleNamespace(emit=messages.append))
    asyncio.run(_close_headless_state(
        Gateway(), telemetry, Memory(), Housekeeper(), bus))

    assert calls == ["gateway", "history", "memory", "checkpoint"]
    assert [message.split(" failed - ")[0] for message in messages] == [
        "SERVER: gateway stop", "SERVER: metrics history close",
        "SERVER: memory close", "SERVER: final checkpoint",
    ]


def test_headless_retires_ingress_before_provider_connections_and_stores():
    calls = []

    async def stop_gateway():
        calls.append("gateway")

    async def close_router():
        calls.append("router")
        raise RuntimeError("close failed")

    asyncio.run(_close_headless_state(
        SimpleNamespace(stop=stop_gateway),
        SimpleNamespace(history=SimpleNamespace(close=lambda: calls.append("history"))),
        SimpleNamespace(close=lambda: calls.append("memory")),
        SimpleNamespace(checkpoint_all=lambda: calls.append("checkpoint")),
        SimpleNamespace(log=SimpleNamespace(emit=lambda _message: None)),
        router=SimpleNamespace(aclose=close_router),
    ))
    assert calls == ["gateway", "router", "history", "memory", "checkpoint"]


def _isolated_startup(monkeypatch, failure):
    """Use no real state, sockets, keys or platform signal registrations."""
    calls = []
    messages = []
    bus = SimpleNamespace(log=SimpleNamespace(emit=messages.append))
    monkeypatch.setattr(server, "OrionBus", lambda: bus)
    monkeypatch.setattr(server, "attach_console_log", lambda _bus: None)
    monkeypatch.delenv("ORION_SYNC_JOURNAL", raising=False)

    history = SimpleNamespace(close=lambda: calls.append("history"))

    def enable_history(_path):
        if failure == "history":
            raise OSError("history unavailable")

    monkeypatch.setattr(server, "Telemetry", lambda _bus: SimpleNamespace(
        history=history, enable_history=enable_history,
        health=SimpleNamespace(register=lambda _name: None), tick_history=lambda: None))
    matrix = SimpleNamespace(close=lambda: calls.append("matrix"))
    monkeypatch.setattr(server, "OrionMemoryMatrix", lambda *_args: matrix)

    def memory_factory(_matrix, _bus):
        if failure == "memory":
            raise RuntimeError("memory unavailable")
        return SimpleNamespace(close=lambda: calls.append("memory"))

    monkeypatch.setattr(server, "MemoryAgent", memory_factory)
    monkeypatch.setattr(server, "read_provider_settings", lambda: object())

    def write_settings(_settings):
        if failure == "settings":
            raise PermissionError("store read-only")

    monkeypatch.setattr(server, "write_provider_settings", write_settings)
    monkeypatch.setattr(server, "IdentityManager", lambda *_args: SimpleNamespace(announce=lambda: None))

    async def wait_until_cancelled():
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "ConnectivityMonitor", lambda *_args: SimpleNamespace(
        mode=lambda: "MODE A", run=wait_until_cancelled, stop=lambda: calls.append("connectivity")))
    monkeypatch.setattr(server, "OllamaManager", lambda *_args: SimpleNamespace(register=lambda _settings: None))

    def attach_identity(_identity):
        if failure == "router":
            raise RuntimeError("identity unavailable")

    async def close_router():
        calls.append("router")

    monkeypatch.setattr(server, "ProviderRouter", lambda *_args, **_kwargs: SimpleNamespace(
        attach_identity=attach_identity, aclose=close_router))
    monkeypatch.setattr(server, "ConversationMemoryEngine", lambda *_args: None)
    for name in ("NeuroKnowledgeBase", "ProgrammingKnowledgeBase", "CyberKnowledgeBase"):
        monkeypatch.setattr(server, name, lambda *_args: SimpleNamespace(seed=lambda _memory: 0))
    monkeypatch.setattr(server, "KnowledgePackManager", lambda *_args: SimpleNamespace(seed_builtin=lambda: None))
    monkeypatch.setattr(server, "LearningService", lambda *_args: None)

    async def start_gateway():
        if failure == "cancelled":
            raise asyncio.CancelledError
        raise OSError("port already in use")

    async def stop_gateway():
        calls.append("gateway")

    monkeypatch.setattr(server, "RemoteGateway", lambda *_args, **_kwargs: SimpleNamespace(
        start=start_gateway, stop=stop_gateway))
    monkeypatch.setattr(server, "DatabaseHousekeeper", lambda _bus: SimpleNamespace(
        stop=lambda: calls.append("housekeeper"),
        checkpoint_all=lambda: calls.append("checkpoint"),
        run=wait_until_cancelled, run_checkpoints=wait_until_cancelled))
    return calls, messages


@pytest.mark.parametrize("failure,closed", [
    ("history", ["housekeeper", "history", "checkpoint"]),
    ("memory", ["housekeeper", "history", "matrix", "checkpoint"]),
    ("settings", ["housekeeper", "history", "memory", "checkpoint"]),
    ("router", ["connectivity", "housekeeper", "router", "history", "memory", "checkpoint"]),
    ("gateway", ["connectivity", "housekeeper", "gateway", "router", "history", "memory", "checkpoint"]),
    ("cancelled", ["connectivity", "housekeeper", "gateway", "router", "history", "memory", "checkpoint"]),
])
def test_failed_startup_closes_only_acquired_resources_and_never_claims_ready(monkeypatch, failure, closed):
    calls, messages = _isolated_startup(monkeypatch, failure)
    error = asyncio.CancelledError if failure == "cancelled" else Exception
    with pytest.raises(error):
        asyncio.run(server.run_headless())
    assert calls == closed
    assert not any("headless node ready" in message for message in messages)


def test_fatal_server_fault_exits_nonzero_for_the_service_manager(monkeypatch, capsys):
    class Loop:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def run_until_complete(self, coroutine):
            coroutine.close()
            raise RuntimeError("startup failed")

    monkeypatch.setattr(server, "QCoreApplication", lambda _argv: SimpleNamespace(setApplicationName=lambda _name: None))
    monkeypatch.setattr(server.qasync, "QEventLoop", lambda _app: Loop())
    monkeypatch.setattr(server.asyncio, "set_event_loop", lambda _loop: None)
    with pytest.raises(SystemExit) as exit_info:
        server.main()
    assert exit_info.value.code == 1
    assert "startup failed" in capsys.readouterr().err


def _shutdown_fakes(calls, messages):
    async def stop_gateway():
        calls.append("gateway")

    async def close_router():
        calls.append("router")

    return (
        SimpleNamespace(stop=stop_gateway),
        SimpleNamespace(history=SimpleNamespace(close=lambda: calls.append("history"))),
        SimpleNamespace(close=lambda: calls.append("memory")),
        SimpleNamespace(checkpoint_all=lambda: calls.append("checkpoint")),
        SimpleNamespace(log=SimpleNamespace(emit=messages.append)),
        SimpleNamespace(aclose=close_router),
    )


@pytest.mark.parametrize("phase", ["gateway", "router"])
def test_headless_cooperative_stalled_close_cannot_consume_final_write_budget(monkeypatch, phase):
    monkeypatch.setattr(server, "_SHUTDOWN_BUDGET_S", 0.08, raising=False)

    async def scenario():
        calls, messages = [], []
        gateway, telemetry, memory, housekeeper, bus, router = _shutdown_fakes(calls, messages)
        started = asyncio.Event()

        async def stalled():
            calls.append(phase)
            started.set()
            await asyncio.Event().wait()

        if phase == "gateway":
            gateway.stop = stalled
        else:
            router.aclose = stalled
        closing = asyncio.create_task(_close_headless_state(
            gateway, telemetry, memory, housekeeper, bus, router=router))
        await started.wait()
        try:
            done, _pending = await asyncio.wait({closing}, timeout=0.25)
            assert done, "a stalled close prevents the final stores/checkpoint from running"
            assert calls == ["gateway", "router", "history", "memory", "checkpoint"]
            assert any(phase in message and "timed out" in message for message in messages)
        finally:
            closing.cancel()
            await asyncio.gather(closing, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["history", "memory"])
def test_headless_store_close_runs_off_loop_and_final_checkpoint_follows_it(monkeypatch, phase):
    monkeypatch.setattr(server, "_SHUTDOWN_BUDGET_S", 0.5, raising=False)

    async def scenario():
        calls, messages = [], []
        gateway, telemetry, memory, housekeeper, bus, router = _shutdown_fakes(calls, messages)
        owner_thread = threading.get_ident()
        close_threads = []
        release = threading.Event()
        heartbeat = asyncio.Event()

        def close_store():
            close_threads.append(threading.get_ident())
            release.wait(0.12)
            calls.append(phase)

        if phase == "history":
            telemetry.history.close = close_store
        else:
            memory.close = close_store

        def tick():
            heartbeat.set()
            release.set()

        timer = asyncio.get_running_loop().call_later(0.02, tick)
        try:
            await _close_headless_state(gateway, telemetry, memory, housekeeper, bus, router=router)
            assert len(close_threads) == 1 and close_threads[0] != owner_thread
            assert heartbeat.is_set(), "store close blocked the loop's release callback"
            assert calls == ["gateway", "router", "history", "memory", "checkpoint"]
        finally:
            timer.cancel()
            release.set()

    asyncio.run(scenario())


def test_headless_background_cancel_resistance_is_owned_before_stores_close(monkeypatch):
    monkeypatch.setattr(server, "_SHUTDOWN_BUDGET_S", 0.06, raising=False)

    async def scenario():
        calls, messages = [], []
        gateway, telemetry, memory, housekeeper, bus, router = _shutdown_fakes(calls, messages)
        started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def writer():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                calls.append("writer finished")

        writer_task = asyncio.create_task(writer(), name="orion-test-checkpoint-writer")
        await started.wait()
        try:
            pending = await _close_headless_state(
                gateway, telemetry, memory, housekeeper, bus, router=router,
                background_tasks=[writer_task])
            assert cancelled.is_set() and not writer_task.done()
            assert pending and "orion-test-checkpoint-writer" in " ".join(pending)
            assert calls == ["gateway"], "stores/providers must remain open for the live writer"
            assert any("still running" in message and "orion-test-checkpoint-writer" in message
                       for message in messages)
            owned = [task for task in asyncio.all_tasks()
                     if task.get_name().startswith("orion-headless-shutdown") and not task.done()]
            assert owned, "timing out must retain cleanup ownership"
        finally:
            release.set()
            if not writer_task.done() and not cancelled.is_set():
                writer_task.cancel()
            await asyncio.gather(writer_task, return_exceptions=True)
            owned = [task for task in asyncio.all_tasks()
                     if task.get_name().startswith("orion-headless-shutdown") and not task.done()]
            if owned:
                await asyncio.wait_for(asyncio.gather(*owned, return_exceptions=True), 0.5)
        assert calls == ["gateway", "writer finished", "router", "history", "memory", "checkpoint"]

    asyncio.run(scenario())


def test_headless_blocked_worker_is_reported_and_checkpoint_waits_for_store_close(monkeypatch):
    monkeypatch.setattr(server, "_SHUTDOWN_BUDGET_S", 0.06, raising=False)

    async def scenario():
        calls, messages = [], []
        gateway, telemetry, memory, housekeeper, bus, router = _shutdown_fakes(calls, messages)
        started, release = threading.Event(), threading.Event()

        def close_history():
            started.set()
            release.wait(0.4)
            calls.append("history")

        telemetry.history.close = close_history
        try:
            before = asyncio.get_running_loop().time()
            pending = await _close_headless_state(gateway, telemetry, memory, housekeeper, bus, router=router)
            elapsed = asyncio.get_running_loop().time() - before
            assert started.is_set()
            assert elapsed < 0.25, "blocking store close ignored the total shutdown budget"
            assert pending and any("stores" in name for name in pending)
            assert calls == ["gateway", "router"]
            assert any("still running" in message and "checkpoint" in message for message in messages)
        finally:
            release.set()
            owned = [task for task in asyncio.all_tasks()
                     if task.get_name().startswith("orion-headless-shutdown") and not task.done()]
            if owned:
                await asyncio.wait_for(asyncio.gather(*owned, return_exceptions=True), 0.5)
        assert calls == ["gateway", "router", "history", "memory", "checkpoint"]

    asyncio.run(scenario())


def test_headless_caller_cancellation_retains_final_write_cleanup(monkeypatch):
    monkeypatch.setattr(server, "_SHUTDOWN_BUDGET_S", 0.5, raising=False)

    async def scenario():
        calls, messages = [], []
        gateway, telemetry, memory, housekeeper, bus, router = _shutdown_fakes(calls, messages)
        started, release = asyncio.Event(), asyncio.Event()

        async def stop_gateway():
            calls.append("gateway")
            started.set()
            await release.wait()

        gateway.stop = stop_gateway
        closing = asyncio.create_task(_close_headless_state(
            gateway, telemetry, memory, housekeeper, bus, router=router))
        await started.wait()
        closing.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await closing
            owned = [task for task in asyncio.all_tasks()
                     if task.get_name().startswith("orion-headless-shutdown") and not task.done()]
            assert owned, "caller cancellation abandoned cleanup and final checkpoint"
        finally:
            release.set()
            owned = [task for task in asyncio.all_tasks()
                     if task.get_name().startswith("orion-headless-shutdown") and not task.done()]
            if owned:
                await asyncio.wait_for(asyncio.gather(*owned, return_exceptions=True), 0.5)
        assert calls == ["gateway", "router", "history", "memory", "checkpoint"]

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["orion-housekeeping", "orion-wal-checkpoint"])
def test_headless_native_background_writer_finishes_before_its_store_closes(monkeypatch, name):
    monkeypatch.setattr(server, "_SHUTDOWN_BUDGET_S", 0.06, raising=False)

    async def scenario():
        calls, messages = [], []
        gateway, telemetry, memory, housekeeper, bus, router = _shutdown_fakes(calls, messages)
        started, release, finished = threading.Event(), threading.Event(), threading.Event()

        def native_write():
            started.set()
            release.wait(0.4)
            calls.append("native write finished")
            finished.set()

        writer_task = asyncio.create_task(asyncio.to_thread(native_write), name=name)
        while not started.is_set():
            await asyncio.sleep(0)
        try:
            pending = await _close_headless_state(
                gateway, telemetry, memory, housekeeper, bus, router=router,
                background_tasks=[writer_task])
            assert not finished.is_set(), "the synthetic worker must still be running at the deadline"
            assert not writer_task.done(), "cancelling its Future conceals a live native writer"
            assert pending and name in pending
            assert calls == ["gateway"], "store closure raced an in-flight native database write"
        finally:
            release.set()
            await asyncio.gather(writer_task, return_exceptions=True)
            await asyncio.to_thread(finished.wait, 0.5)
            owned = [task for task in asyncio.all_tasks()
                     if task.get_name().startswith("orion-headless-shutdown") and not task.done()]
            if owned:
                await asyncio.wait_for(asyncio.gather(*owned, return_exceptions=True), 0.5)
        assert calls == ["gateway", "native write finished", "router", "history", "memory", "checkpoint"]

    asyncio.run(scenario())
