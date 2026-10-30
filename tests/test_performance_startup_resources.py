"""Storage workers yield startup, with owned disposal across cancellation races."""
import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from orion_core import startup_resources as resources


async def entered(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 3)


def test_delayed_storage_constructor_keeps_controller_loop_running():
    owner = threading.get_ident()
    waiting, release = threading.Event(), threading.Event()
    calls = []

    def construct():
        calls.append(threading.get_ident())
        waiting.set()
        assert release.wait(5)
        return object()

    async def scenario():
        task = asyncio.create_task(resources.construct_startup_resource(construct))
        try:
            await entered(waiting)
            assert not task.done() and calls == [calls[0]] and calls[0] != owner
        finally:
            release.set()
            await task

    asyncio.run(scenario())


def test_cancelled_constructor_disposes_late_handle_once_without_waiting_for_it():
    waiting, release, closed = threading.Event(), threading.Event(), threading.Event()
    close_threads = []
    owner = threading.get_ident()

    class Store:
        def close(self):
            close_threads.append(threading.get_ident())
            closed.set()

    def construct():
        waiting.set()
        assert release.wait(5)
        return Store()

    async def scenario():
        task = asyncio.create_task(resources.construct_startup_resource(construct))
        try:
            await entered(waiting)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not closed.is_set()  # the real constructor is still blocked
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 3)
        assert len(close_threads) == 1 and close_threads[0] != owner

    asyncio.run(scenario())


def test_cancel_after_worker_result_before_delivery_still_closes_handle(monkeypatch):
    closed = threading.Event()
    threads = []
    owner = threading.get_ident()
    entered, release = threading.Event(), threading.Event()

    class Store:
        def __init__(self):
            entered.set()
            assert release.wait(5)

        def close(self):
            threads.append(threading.get_ident())
            closed.set()

    async def scenario():
        loop = asyncio.get_running_loop()
        real_submit = loop.run_in_executor
        task = None

        def submit(_executor, function, *args):
            future = real_submit(_executor, function, *args)
            future.add_done_callback(lambda _future: task.cancel())
            assert entered.wait(5)
            release.set()
            return future

        monkeypatch.setattr(loop, "run_in_executor", submit)
        task = asyncio.create_task(resources.construct_startup_resource(Store))
        with pytest.raises(asyncio.CancelledError):
            await task
        monkeypatch.setattr(loop, "run_in_executor", real_submit)
        assert await asyncio.to_thread(closed.wait, 3)
        assert len(threads) == 1 and threads[0] != owner

    asyncio.run(scenario())


def test_boot_scope_closes_ready_stores_when_a_later_phase_fails():
    closing, release, closed = threading.Event(), threading.Event(), threading.Event()
    threads = []
    owner = threading.get_ident()

    class Store:
        def close(self):
            threads.append(threading.get_ident())
            closing.set()
            assert release.wait(5)
            closed.set()

    @resources.startup_lifetime
    async def boot():
        await resources.construct_startup_resource(Store)
        raise ValueError("synthetic later startup failure")

    async def scenario():
        try:
            with pytest.raises(ValueError, match="later startup failure"):
                await boot()
            await entered(closing)
            assert not closed.is_set() and threads[0] != owner
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 3)
        assert len(threads) == 1

    asyncio.run(scenario())


def test_normal_handoff_keeps_store_open_for_existing_shutdown_owner():
    closed = []
    store = SimpleNamespace(close=lambda: closed.append(True))

    @resources.startup_lifetime
    async def boot():
        value = await resources.construct_startup_resource(lambda: store)
        resources.handoff_startup_resources()
        return value

    assert asyncio.run(boot()) is store and closed == []
    store.close()
    assert closed == [True]


def test_cancel_before_executor_admission_never_runs_storage_factory():
    waiting, release = threading.Event(), threading.Event()
    calls = []

    def occupy():
        waiting.set()
        assert release.wait(5)

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        blocker = loop.run_in_executor(None, occupy)
        try:
            await entered(waiting)
            task = asyncio.create_task(resources.construct_startup_resource(lambda: calls.append(True)))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await blocker

    asyncio.run(scenario())
    assert calls == []


def test_worker_context_and_late_failure_are_observed():
    marker = contextvars.ContextVar("synthetic_startup_context")
    waiting, release = threading.Event(), threading.Event()
    observed = []

    def construct():
        observed.append(marker.get())
        waiting.set()
        assert release.wait(5)
        raise OSError("synthetic late open failure")

    async def scenario():
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: errors.append(context))
        token = marker.set("request scope")
        task = asyncio.create_task(resources.construct_startup_resource(construct))
        try:
            await entered(waiting)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            marker.reset(token)
            release.set()
        async def settled():
            while resources._PENDING:
                await asyncio.sleep(0)
        await asyncio.wait_for(settled(), 3)
        await asyncio.sleep(0)
        assert observed == ["request scope"] and errors == []

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["memory", "usage", "graph"])
def test_real_temporary_store_remains_usable_on_owner_thread_after_worker_creation(name, tmp_path, monkeypatch):
    from orion_core import memory, token_usage, knowledge_graph, constants
    from orion_core import app
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    monkeypatch.setattr(memory, "BASE_DIR", tmp_path)
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path)
    monkeypatch.setattr(app, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(app, "CORE_DB_PATH", tmp_path / "memory.db")
    factories = {
        "memory": lambda: app._make_startup_memory(bus),
        "usage": lambda: token_usage.TokenUsageLedger(tmp_path / "usage.db"),
        "graph": lambda: knowledge_graph.KnowledgeGraphEngine(bus, None, db_path=tmp_path / "graph.db"),
    }
    store = asyncio.run(resources.construct_startup_resource(factories[name]))
    try:
        connection = store.matrix.conn if name == "memory" else store.conn
        assert connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0] > 0
    finally:
        store.close()


def test_memory_agent_initialization_failure_closes_its_created_matrix(monkeypatch):
    from orion_core import app
    closed = []
    monkeypatch.setattr(app, "OrionMemoryMatrix", lambda *_: SimpleNamespace(close=lambda: closed.append(True)))

    def fail(*_args):
        raise OSError("synthetic transcript path initialization failure")

    monkeypatch.setattr(app, "MemoryAgent", fail)
    with pytest.raises(OSError, match="transcript path"):
        app._make_startup_memory(None)
    assert closed == [True]


def test_real_skill_registry_is_complete_before_owner_publishes_it(tmp_path):
    from orion_core.skills import SkillManager
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    manager = asyncio.run(resources.construct_startup_resource(SkillManager, bus, root=tmp_path))
    assert manager.registry.all()
    assert manager.prompt_context()
    # The owner can immediately read a complete registry and use its disk
    # mutation lock after worker construction; no late partial publication.
    skill = manager.registry.all()[0]
    assert manager.set_enabled(skill.name, False).ok
    assert not manager.registry.get(skill.name).enabled


def test_real_control_backend_setup_stays_off_owner_without_starting_actions(monkeypatch):
    from orion_core import control
    owner, backend_threads = threading.get_ident(), []
    backend = object()

    def configure(self):
        backend_threads.append(threading.get_ident())
        self._pg = backend

    monkeypatch.setattr(control.AutonomousControlLayer, "_configure_backends", configure)
    display, desktop = object(), object()
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    manager = asyncio.run(resources.construct_startup_resource(
        control.AutonomousControlLayer, bus, display, None, desktop))
    assert backend_threads and backend_threads[0] != owner
    assert manager._pg is backend and manager.display is display and manager.desktop is desktop
    assert manager.last_action == {}  # startup configures; it does not perform input
