"""
Headless server — ORION's brain without the desktop, for the cloud.

This is the deployable node for **Oracle Cloud** (or any display-less Linux VM)
and the endpoint the **private Android PWA** connects to over the internet or
mobile data. It builds only the portable, non-GUI half of ORION:

    OrionBus (a QObject) → Telemetry → Memory → Identity → Connectivity →
    Ollama → ProviderRouter → ConversationMemory → knowledge seeds → LearningService
    → RemoteGateway (installable PWA + JSON API)

No windows, no audio capture, no screen grabbing, no Windows COM — so it runs on
a headless Ubuntu/Oracle-Linux box with nothing but Python, aiohttp, PyQt6-Core
and (optionally) a local Ollama or a cloud API key. The full desktop build is
unchanged; this is an additional entry point selected with ``--headless`` or
``ORION_HEADLESS=1``.

Answers come from the language-model router (cloud API when reachable, local
Ollama otherwise), grounded in ORION's frozen identity and memory. Teach it new
facts remotely with the learning path; each node persists to its own SQLite
store. Keep desktop and cloud stores separate whilst both are running. Use a
consistent backup and restore process to transfer state between them.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from typing import Any, Awaitable

import qasync
from PyQt6.QtCore import QCoreApplication

from .bus import OrionBus
from .connectivity import ConnectivityMonitor
from .constants import APP_NAME, CONFIG_DIR, CORE_DB_PATH
from .conversation_memory import ConversationMemoryEngine
from .identity import IdentityManager
from .cyber_knowledge import CyberKnowledgeBase
from .knowledge import NeuroKnowledgeBase
from .knowledge_packs import KnowledgePackManager
from .learning import LearningService
from .programming_knowledge import ProgrammingKnowledgeBase
from .local_models import OllamaManager
from .memory import MemoryAgent, OrionMemoryMatrix
from .providers import (
    ProviderRouter,
    read_provider_settings,
    write_provider_settings,
)
from .maintenance import DatabaseHousekeeper
from .remote import RemoteGateway
from .telemetry import Telemetry


_SHUTDOWN_BUDGET_S = 8.0
_DURABLE_HEADLESS_TASKS = frozenset({"orion-housekeeping", "orion-wal-checkpoint"})
# A deadline does not stop a native store close or a cancellation-resistant
# coroutine. Keep the cleanup owner until it really finishes, including after
# its caller is cancelled, and report what remains rather than discarding it.
_HEADLESS_SHUTDOWN_TASKS: set[asyncio.Task[Any]] = set()


def _shutdown_log(bus: Any, message: str) -> None:
    try:
        bus.log.emit(message)
    except Exception:
        pass


def attach_console_log(bus: Any, stream: Any = None) -> None:
    """Print every log line to stdout, timestamped.

    The desktop shows the log in its window; a headless node has no window,
    and nothing else listened to ``bus.log`` — so every line went nowhere,
    including the first-run pairing code, which made a fresh node impossible
    to pair. systemd (journalctl) and Docker (docker logs) read stdout.
    """
    def _print(message: Any) -> None:
        try:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}",
                  file=stream or sys.stdout, flush=True)
        except Exception:
            pass

    bus.log.connect(_print)


async def _close_headless_state(gateway: Any, telemetry: Any, memory: Any,
                                housekeeper: DatabaseHousekeeper, bus: OrionBus,
                                *, router: Any = None,
                                background_tasks: list[asyncio.Task[Any]] | None = None,
                                ) -> tuple[str, ...]:
    """Bound the wait, preserving ordered cleanup and reporting unfinished work.

    Slow gateway/provider phases get a fraction of the total budget before
    cancellation, leaving time for final writes. A phase which ignores that
    cancellation remains owned; stores are not closed underneath its writers.
    One worker closes both stores and checkpoints in order, so a late native
    close cannot race a checkpoint or stall the event loop.
    """
    loop = asyncio.get_running_loop()
    budget = max(0.0, _SHUTDOWN_BUDGET_S)
    deadline = loop.time() + budget
    writers = list(background_tasks or ())
    deferred = False

    def remaining() -> float:
        return max(0.0, deadline - loop.time())

    async def close_async(name: str, operation: Awaitable[Any]) -> None:
        owner = asyncio.current_task()
        if owner is not None:
            owner.set_name(f"orion-headless-shutdown: {name}")
        phase = asyncio.create_task(operation, name=f"orion-headless-close: {name}")
        done, _pending = await asyncio.wait({phase}, timeout=min(budget / 4.0, remaining()))
        if not done:
            _shutdown_log(bus, f"SERVER: {name} timed out - cancellation requested; "
                              "final cleanup still owns the operation.")
            phase.cancel()
        try:
            # Waiting here is intentional: the outer total deadline is bounded
            # independently, while a late closer must finish before stores close.
            await asyncio.shield(phase)
        except asyncio.CancelledError:
            _shutdown_log(bus, f"SERVER: {name} cancelled before completion.")
        except Exception as exc:
            _shutdown_log(bus, f"SERVER: {name} failed - {exc}")

    def close_stores() -> None:
        try:
            if telemetry is not None and telemetry.history is not None:
                telemetry.history.close()
        except Exception as exc:
            _shutdown_log(bus, f"SERVER: metrics history close failed - {exc}")
        try:
            if memory is not None:
                memory.close()
        except Exception as exc:
            _shutdown_log(bus, f"SERVER: memory close failed - {exc}")
        try:
            # Include writes made while retiring ingress and closing stores,
            # even for a node which ran less than one checkpoint interval.
            housekeeper.checkpoint_all()
        except Exception as exc:
            _shutdown_log(bus, f"SERVER: final checkpoint failed - {exc}")

    async def cleanup() -> None:
        for task in writers:
            # Housekeeper.stop() already stops future passes. Its current
            # to_thread call must retire naturally: cancelling the await marks
            # the Task done while its native database writer can still run.
            if not task.done() and task.get_name() not in _DURABLE_HEADLESS_TASKS:
                task.cancel()
        if gateway is not None:
            await close_async("gateway stop", gateway.stop())
        pending_writers = {task for task in writers if not task.done()}
        if pending_writers:
            owner = asyncio.current_task()
            if owner is not None:
                owner.set_name("orion-headless-shutdown: background retirement")
            # They were already cancelled once. Repeated cancellation can cut
            # off a writer's durable finally block instead of helping it retire.
            await asyncio.wait(pending_writers)
        for task in writers:
            if not task.cancelled():
                error = task.exception()
                if error is not None:
                    _shutdown_log(bus, f"SERVER: background '{task.get_name()}' failed - {error}")
        if router is not None:
            await close_async("provider router close", router.aclose())
        owner = asyncio.current_task()
        if owner is not None:
            owner.set_name("orion-headless-shutdown: stores and final checkpoint")
        await asyncio.to_thread(close_stores)

    owner = asyncio.create_task(cleanup(), name="orion-headless-shutdown: cleanup")
    _HEADLESS_SHUTDOWN_TASKS.add(owner)

    def finished(task: asyncio.Task[Any]) -> None:
        _HEADLESS_SHUTDOWN_TASKS.discard(task)
        if task.cancelled():
            _shutdown_log(bus, "SERVER: shutdown cleanup cancelled - final writes are unverified.")
            return
        error = task.exception()
        if error is not None:
            _shutdown_log(bus, f"SERVER: shutdown cleanup failed - {error}")
        elif deferred:
            _shutdown_log(bus, "SERVER: deferred shutdown cleanup finished.")

    owner.add_done_callback(finished)
    try:
        done, _pending = await asyncio.wait({owner}, timeout=remaining())
    except asyncio.CancelledError:
        deferred = True
        _shutdown_log(bus, "SERVER: shutdown caller cancelled - cleanup remains owned; "
                          "final checkpoint is not yet verified.")
        raise
    if done:
        await owner
        return ()
    deferred = True
    pending = tuple(sorted({owner.get_name(), *(task.get_name() for task in writers if not task.done())}))
    _shutdown_log(bus, "SERVER: shutdown budget exhausted - still running: "
                      f"{', '.join(pending)}; final checkpoint is not yet verified.")
    return pending


async def run_headless() -> None:
    """Compose and run the brain-only node until interrupted."""
    bus       = OrionBus()
    attach_console_log(bus)
    # Startup can fail after opening stores but before the HTTP listener exists.
    # Keep every acquired resource in the same finally block as normal shutdown.
    telemetry = matrix = memory = router = gateway = connectivity = None
    housekeeper = DatabaseHousekeeper(bus)
    background_tasks: list[asyncio.Task[Any]] = []
    try:
        telemetry = Telemetry(bus)
        telemetry.enable_history(CONFIG_DIR / "metrics_history.db")   # rolling trends
        for component in ("connectivity", "ollama", "remote", "identity"):
            try:
                telemetry.health.register(component)
            except Exception:
                pass

        matrix = OrionMemoryMatrix(CORE_DB_PATH, CONFIG_DIR, bus)
        memory = MemoryAgent(matrix, bus)
        # C2: opt-in change journal (see app.py). ORION_SYNC_JOURNAL=1 to enable.
        if os.getenv("ORION_SYNC_JOURNAL", "").strip().lower() in {"1", "true", "on", "yes"}:
            from .sync import SyncJournal
            matrix.journal = SyncJournal()
            bus.log.emit("SYNC: change journal enabled (C2).")

        # Provider settings are read from config/api_keys.json (or ORION_* env);
        # unlike the desktop we never pop a dialog — a cloud node is unattended.
        settings = read_provider_settings()
        if not getattr(settings, "configuration_error", ""):
            write_provider_settings(settings)

        identity = IdentityManager(bus, telemetry)
        identity.announce()

        connectivity = ConnectivityMonitor(bus, telemetry)
        ollama = OllamaManager(bus, telemetry)
        try:
            ollama.register(settings)   # enable local_ollama if a server is running
        except Exception as exc:
            bus.log.emit(f"SERVER: ollama probe skipped - {exc}")

        router = ProviderRouter(settings, bus, memory, connectivity=connectivity)
        router.attach_identity(identity)

        conversation = ConversationMemoryEngine(bus, memory, router, telemetry)

        # Seed the offline knowledge so answers stay grounded even with no cloud —
        # neuroscience, programming and cybersecurity all travel to the cloud node.
        try:
            seeded = NeuroKnowledgeBase(telemetry).seed(memory)
            seeded += ProgrammingKnowledgeBase(telemetry).seed(memory)
            seeded += CyberKnowledgeBase(telemetry).seed(memory)
            if seeded:
                bus.log.emit(f"SERVER: knowledge corpora seeded ({seeded} entries).")
        except Exception as exc:
            bus.log.emit(f"SERVER: knowledge seed skipped - {exc}")
        try:
            packs = KnowledgePackManager(bus, memory, telemetry)
            packs.seed_builtin()
        except Exception as exc:
            bus.log.emit(f"SERVER: knowledge packs skipped - {exc}")

        # Remote teaching: POST future facts through the learning path if desired.
        _learning = LearningService(bus, memory, router, telemetry)  # noqa: F841

        bus.log.emit(f"SERVER: {connectivity.mode()} at startup ({APP_NAME} headless node).")

        # The uplink is the whole point of the headless node, so it is ON by default
        # here (ORION_REMOTE_ACCESS still forces it, but never disables it).
        gateway = RemoteGateway(
            router, memory, bus,
            identity=identity, conversation=conversation, telemetry=telemetry,
        )
        await gateway.start()

        background_tasks.append(asyncio.create_task(
            connectivity.run(), name="orion-connectivity"))
        # Keep WAL data current every few minutes and compact stores weekly on an
        # unattended node where a missing checkpoint can go unnoticed for days.
        background_tasks.append(asyncio.create_task(
            housekeeper.run(), name="orion-housekeeping"))
        background_tasks.append(asyncio.create_task(
            housekeeper.run_checkpoints(), name="orion-wal-checkpoint"))

        # Graceful shutdown on SIGINT/SIGTERM (POSIX) or KeyboardInterrupt (Windows).
        shutdown = asyncio.Event()
        loop = asyncio.get_running_loop()

        async def _history_sampler() -> None:
            """Drive the rolling metrics store on a headless node (the GUI's
            Command Centre timer does this on the desktop)."""
            while not shutdown.is_set():
                telemetry.tick_history()          # throttled internally
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=60.0)
                except asyncio.TimeoutError:
                    continue
        background_tasks.append(asyncio.create_task(
            _history_sampler(), name="orion-history-sampler"))

        def _request_stop(*_: Any) -> None:
            shutdown.set()

        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError):
                # Windows event loops don't support add_signal_handler for SIGTERM.
                try:
                    signal.signal(sig, lambda *_: _request_stop())
                except Exception:
                    pass

        bus.log.emit("SERVER: headless node ready — awaiting remote turns.")
        await shutdown.wait()
    finally:
        bus.log.emit("SERVER: shutting down headless node.")
        try:
            if connectivity is not None:
                connectivity.stop()
        except Exception as exc:
            _shutdown_log(bus, f"SERVER: connectivity stop failed - {exc}")
        try:
            housekeeper.stop()
        except Exception as exc:
            _shutdown_log(bus, f"SERVER: housekeeping stop failed - {exc}")
        # Stop admitting remote work before closing sessions used by that work.
        # If MemoryAgent construction failed, close the matrix it would own.
        await _close_headless_state(
            gateway, telemetry, memory if memory is not None else matrix,
            housekeeper, bus, router=router, background_tasks=background_tasks)


def main() -> None:
    """Headless entry point: a QCoreApplication (no GUI) + qasync loop."""
    app = QCoreApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    try:
        with loop:
            loop.run_until_complete(run_headless())
    except KeyboardInterrupt:
        print("O.R.I.O.N. headless node stopped from console.")
    except SystemExit:
        raise
    except Exception:
        import traceback
        print("O.R.I.O.N. headless node terminated after a fault:")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
