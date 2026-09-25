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
facts remotely with the learning path; everything persists to the same SQLite
memory the desktop uses, so a cloud node and a desktop node can share a config
directory (or a synced volume) and stay in lock-step.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from typing import Any

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


async def run_headless() -> None:
    """Compose and run the brain-only node until interrupted."""
    bus       = OrionBus()
    attach_console_log(bus)
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

    connectivity_task = asyncio.create_task(connectivity.run(), name="orion-connectivity")
    # §5.2: the unattended cloud node is where the SQLite stores grow without a
    # human noticing, so the weekly checkpoint/VACUUM sweep matters most here.
    housekeeper = DatabaseHousekeeper(bus)
    housekeeping_task = asyncio.create_task(housekeeper.run(), name="orion-housekeeping")

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
    history_task = asyncio.create_task(_history_sampler(), name="orion-history-sampler")

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
    try:
        await shutdown.wait()
    finally:
        bus.log.emit("SERVER: shutting down headless node.")
        connectivity.stop()
        housekeeper.stop()
        connectivity_task.cancel()
        history_task.cancel()
        housekeeping_task.cancel()
        for task in (connectivity_task, history_task, housekeeping_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        await gateway.stop()
        if telemetry.history is not None:
            telemetry.history.close()
        memory.close()


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


if __name__ == "__main__":
    main()
