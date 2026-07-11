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
from .remote import RemoteGateway
from .telemetry import Telemetry


async def run_headless() -> None:
    """Compose and run the brain-only node until interrupted."""
    bus       = OrionBus()
    telemetry = Telemetry(bus)
    for component in ("connectivity", "ollama", "remote", "identity"):
        try:
            telemetry.health.register(component)
        except Exception:
            pass

    matrix = OrionMemoryMatrix(CORE_DB_PATH, CONFIG_DIR, bus)
    memory = MemoryAgent(matrix, bus)

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

    # Graceful shutdown on SIGINT/SIGTERM (POSIX) or KeyboardInterrupt (Windows).
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

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
        connectivity_task.cancel()
        try:
            await connectivity_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        await gateway.stop()
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
