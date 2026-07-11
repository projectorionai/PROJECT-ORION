"""
Application bootstrap — composition root for O.R.I.O.N. Mark X.5.

Everything is wired here, and only here: services never construct their own
dependencies, which keeps every module independently testable and swappable.

Construction order (each layer only sees the layers above it):

    OrionBus                                    signal hub
    OrionMemoryMatrix → MemoryAgent             memory (session + persistent)
    OrionCoreWindow                             window 1 (needs bus + memory)
    provider settings (dialog if unconfigured)
    IdentityManager                             one persona for every channel
    ProviderRouter                              model transport + orchestration
    VolatileScreenGrabber / LocalFileIntelligence / VisionAgent
    DesktopAgent / OutlookService / NotionService
    AgentManager                                specialist workforce
    MorningBriefingService
    CognitiveStateManager / KnowledgeGraphEngine   the second brain
    OrionDispatcher                             tool routing
    GenAILiveWorker                             realtime session brain
    CognitiveLoopManager / ProactiveReportingService   awareness + reports
    UnifiedDashboard                            window 2 (monitor 1)
    RemoteGateway (opt-in)

Mark X.5 startup layout: the core window opens maximised on the primary
monitor and the unified Command Deck maximised on the second monitor; with a
single monitor the deck docks as a managed workspace panel instead.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from typing import Optional

import qasync
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from .agents import AgentManager, DesktopAgent
from .briefing import MorningBriefingService
from .bus import OrionBus
from .constants import APP_NAME, CONFIG_DIR, CORE_DB_PATH
from .control import AutonomousControlLayer
from .copilot import DeveloperCopilot
from .dispatcher import OrionDispatcher
from .display import DisplayTopologyManager
from .gui import (
    ApiKeyDialog,
    CommandCentreWindow,
    HolographicToggle,
    OrionCoreWindow,
    WidgetDashboardWindow,
)
from .audio_studio import AudioStudioService
from .backup_manager import BackupManager
from .changelog import Changelog
from .cognition import CognitiveStateManager
from .cyber_knowledge import CyberKnowledgeBase
from .cognitive_loop import CognitiveLoopManager
from .diagnostics import DiagnosticsEngine
from .dynamic_loader import ReflectiveModuleLoader
from .executive import ExecutiveAssistantMode
from .exporter import DocumentExporterService
from .file_organiser import FileOrganiser
from .forge import ForgeOrchestrationManager
from .identity import IdentityManager
from .knowledge_graph import KnowledgeGraphEngine
from .learning import LearningService
from .reporting import ProactiveReportingService
from .dependencies import DynamicPackageResolver
from .sandbox import SandboxVerificationHarness
from .gui.cursor_overlay import CursorOverlay
from .gui.diagnostics_centre import DiagnosticsCentreView
from .gui.entrepreneur import EntrepreneurDeck
from .ocr_engine import OcrEngine
from .plan_executor import PlanExecutor
from .programming_knowledge import ProgrammingKnowledgeBase
from .reports import ReportDrafter
from .security_sentinel import SecuritySentinel
from .gui.globe import GlobeView
from .gui.studio_deck import StudioDeckView
from .gui.unified_dashboard import UnifiedDashboard
from .literature import LiteratureIntakeService
from .pipeline import AgencyPipelineService
from .knowledge import NeuroKnowledgeBase
from .live_worker import GenAILiveWorker
from .local_brain import LocalBrain
from .memory import MemoryAgent, OrionMemoryMatrix
from .mind_expansion import KnowledgeCorpusBuilder
from .momentum import MomentumEngine
from .presence import PresenceMonitor
from .protocols import ProtocolManager
from .reminders import ReminderService
from .research import ResearchAgent
from .sentinel import SentinelAgent
from .notion import NotionService
from .outlook import OutlookService
from .proactive import ProactiveIntelligence
from .providers import (
    OrionProviderSettings,
    ProviderRouter,
    _default_provider_payload,
    _settings_from_payload,
    read_provider_settings,
    write_provider_settings,
)
from .remote import RemoteGateway
from .selfrepair import SelfRepairAgent
from .telemetry import Telemetry
from .connectivity import ConnectivityMonitor
from .local_models import OllamaManager, AIModeInfo
from .knowledge_packs import KnowledgePackManager
from .conversation_memory import ConversationMemoryEngine
from .commerce import CommerceSuite
from .community import CommunityHub, EcommerceHub
from .speech_offline import OfflineTranscriber
from .verification import VisualVerificationEngine
from .vision import LocalFileIntelligence, VisionAgent, VolatileScreenGrabber
from .web import WebController
from .workspace import DesktopMemoryManager
from .web_automation import WebAutomationService
from .peripherals import PeripheralController
from .messaging import MessagingGateway
from .gaming import GamingClientService
from .entertainment import EntertainmentService


# ──────────────────────────────────────────────────────────────────────────────
# DUAL-SCREEN STARTUP LAYOUT (Mark X.5)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_startup_layout(
    window: QWidget, deck: QWidget, display: "DisplayTopologyManager", bus: OrionBus
) -> None:
    """
    Place the core window maximised on monitor 0 and the Command Deck
    maximised on monitor 1, preserving each monitor's offset and DPI scale.

    The DisplayTopologyManager provides the authoritative physical topology
    (and the log record); window placement itself goes through Qt's QScreen
    objects because Qt geometry is expressed in device-independent pixels —
    mixing the two coordinate spaces is exactly what misplaces windows on
    scaled monitors.  With a single monitor the deck docks as a managed
    workspace panel on the right half instead of hiding.
    """
    topology = display.topology()
    screens = QGuiApplication.screens()
    primary_screen = QGuiApplication.primaryScreen()
    secondary = next((s for s in screens if s is not primary_screen), None)

    if primary_screen is not None:
        try:
            window.setScreen(primary_screen)
        except Exception:
            pass  # Qt < 6.1 fallback: move() below still lands it correctly
        window.move(primary_screen.availableGeometry().topLeft())
    window.showMaximized()
    primary_monitor = topology.primary
    bus.log.emit(
        "DISPLAY: core window → monitor "
        f"{primary_monitor.index if primary_monitor else 0} (primary), maximised."
    )

    if secondary is not None:
        try:
            deck.setScreen(secondary)
        except Exception:
            pass
        deck.move(secondary.availableGeometry().topLeft())
        deck.showMaximized()
        second_monitor = next(
            (m for m in topology.monitors if not m.is_primary), None
        )
        bus.log.emit(
            "DISPLAY: command deck → monitor "
            f"{second_monitor.index if second_monitor else 1} "
            f"('{secondary.name()}', scale "
            f"{int((second_monitor.scale if second_monitor else 1.0) * 100)}%), maximised."
        )
    else:
        geometry = (primary_screen.availableGeometry()
                    if primary_screen is not None else None)
        if geometry is not None:
            half = geometry.width() // 2
            deck.setGeometry(geometry.x() + half, geometry.y(), half, geometry.height())
        deck.show()
        bus.log.emit(
            "DISPLAY: single monitor — command deck docked as a managed "
            "workspace panel (right half); Ctrl+D toggles it as an overlay."
        )


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION BOOTSTRAP (GUI-assisted)
# ──────────────────────────────────────────────────────────────────────────────

def ensure_provider_settings(window: Optional[QWidget] = None) -> OrionProviderSettings:
    settings = read_provider_settings()
    if any(profile.enabled and (profile.api_key or profile.base_url) for profile in settings.providers.values()):
        write_provider_settings(settings)
        return settings
    dialog = ApiKeyDialog(window)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        raise SystemExit(0)
    settings = _settings_from_payload(_default_provider_payload(dialog.key()))
    write_provider_settings(settings)
    return settings


def ensure_api_key(window: Optional[QWidget] = None) -> str:
    """Legacy wrapper retained for external imports."""
    settings = ensure_provider_settings(window)
    gemini = settings.providers.get("gemini")
    return gemini.api_key if gemini is not None else ""


# ──────────────────────────────────────────────────────────────────────────────
# APPLICATION LIFECYCLE
# ──────────────────────────────────────────────────────────────────────────────

async def run_application(app: QApplication) -> None:
    # ── observability + memory before windows ───────────────────────────────
    bus       = OrionBus()
    telemetry = Telemetry(bus)
    for component in ("live_worker", "audio", "dispatcher", "vision", "control",
                      "proactive", "self_repair", "workspace"):
        telemetry.health.register(component)
    matrix = OrionMemoryMatrix(CORE_DB_PATH, CONFIG_DIR, bus)
    memory = MemoryAgent(matrix, bus)

    window = OrionCoreWindow(bus, memory)
    window.show()

    toggle = HolographicToggle(window)
    toggle.move(28, 28)
    toggle.show()

    bus.log.emit("SYS: O.R.I.O.N. Mark X.5 autonomous AI operating system initialised.")

    settings = ensure_provider_settings(window)

    # ── personality consistency engine: one persona across every channel ─────
    identity = IdentityManager(bus, telemetry)
    identity.announce()

    # ── dual-mode intelligence: connectivity + local models (MODE A/B) ───────
    for component in ("connectivity", "ollama", "commerce", "knowledge_packs"):
        telemetry.health.register(component)
    connectivity = ConnectivityMonitor(bus, telemetry)
    ollama = OllamaManager(bus, telemetry)
    ollama.register(settings)          # enable local_ollama with the best pulled model
    bus.log.emit(f"SYS: {connectivity.mode()} at startup.")

    # ── core services ────────────────────────────────────────────────────────
    router     = ProviderRouter(settings, bus, memory, connectivity=connectivity)
    router.attach_identity(identity)
    grabber    = VolatileScreenGrabber(bus)
    file_intel = LocalFileIntelligence(bus)
    vision     = VisionAgent(bus, grabber, file_intel)
    desktop    = DesktopAgent(bus)
    outlook    = OutlookService(bus)
    notion     = NotionService(bus, settings.integration("notion"))
    agents     = AgentManager(router, bus)
    # Mark X.7: memory is passed for compatibility, but story dedup now lives
    # in the briefing's own TTL signature cache — the old memory-based check
    # was never wired here (the handle was omitted), so stories repeated.
    briefing   = MorningBriefingService(bus, notion, outlook, memory=memory)

    # ── neuroscience expertise: seed the KNOWLEDGE tier (idempotent) ─────────
    knowledge = NeuroKnowledgeBase(telemetry)
    seeded = knowledge.seed(memory)
    if seeded:
        bus.log.emit(f"KNOWLEDGE: neuroscience corpus seeded ({seeded} entries).")

    # ── Mark IX autonomy stack ───────────────────────────────────────────────
    display   = DisplayTopologyManager(telemetry)
    control   = AutonomousControlLayer(bus, display, telemetry, desktop)
    verifier  = VisualVerificationEngine(bus, control, vision, display, telemetry)
    web       = WebController(bus, control, vision, verifier, telemetry)
    # DesktopMemoryManager IS a WorkspaceManager, extended with named
    # workspace restoration ("restore marketing workspace") — Mark X.5.
    workspace = DesktopMemoryManager(bus, memory, desktop, telemetry)
    copilot   = DeveloperCopilot(bus, memory, router, telemetry)
    selfrepair = SelfRepairAgent(bus, telemetry, router)
    proactive = ProactiveIntelligence(bus, outlook, notion, memory, telemetry, workspace)
    bus.log.emit("SYS: " + display.summary())

    # ── entrepreneurial + offline intelligence layer ─────────────────────────
    packs = KnowledgePackManager(bus, memory, telemetry)
    packs.seed_builtin()
    offline_stt = OfflineTranscriber(bus, telemetry)
    conversation = ConversationMemoryEngine(bus, memory, router, telemetry)
    commerce = CommerceSuite(bus, memory, router, knowledge=packs, telemetry=telemetry)
    community = CommunityHub(bus, memory, packs, telemetry)
    hub = EcommerceHub(bus, memory, commerce.dropship, packs)
    ai_info = AIModeInfo(router, connectivity, ollama, offline_stt)

    # ── Mark X.5: second brain + durable cognitive state ─────────────────────
    cognition = CognitiveStateManager(bus, memory, telemetry=telemetry)
    graph = KnowledgeGraphEngine(bus, memory, telemetry=telemetry)
    resume_summary = await cognition.restore_on_launch()
    if resume_summary:
        bus.log.emit(f"COG: {resume_summary[:160]}")

    # ── JARVIS Subsystems: web automation, peripherals, messaging, gaming, entertainment ──
    web_automation = WebAutomationService(bus)
    peripherals = PeripheralController(bus)
    messaging = MessagingGateway(bus)
    gaming = GamingClientService(bus)
    entertainment = EntertainmentService(bus)

    dispatcher = OrionDispatcher(
        bus, memory, grabber, file_intel,
        desktop, vision, outlook, notion, agents, briefing,
        control=control, verifier=verifier, web=web, workspace=workspace,
        copilot=copilot, selfrepair=selfrepair, proactive=proactive,
        display=display, telemetry=telemetry, knowledge=knowledge,
        packs=packs, conversation=conversation, commerce=commerce,
        community=community, hub=hub, ai=ai_info,
    )
    # Offline conversational brain: keeps ORION talking + task-capable with no API.
    local_brain = LocalBrain(bus, memory, dispatcher, knowledge, router, telemetry)
    worker = GenAILiveWorker(settings, bus, memory, dispatcher, router, telemetry, local_brain)

    # ── Mark X.7: sentiment-driven expression + temporal presence ────────────
    # The emotion engine listens to the bus (state + sentiment) and broadcasts
    # full rendering parameter sets; the face subscribes in the core window.
    from .emotion import EmotionStateManager
    from .temporal import TemporalPresence
    emotion = EmotionStateManager(bus, telemetry)
    temporal = TemporalPresence(bus, memory=memory, notion=notion, telemetry=telemetry)
    worker.temporal = temporal          # contextual startup greeting (Phase 3)
    dispatcher.emotion = emotion        # the 'emotion' voice tool
    router.attach_emotion(emotion)      # emotion colours tone, pacing, wording

    # ── Phase 8: worldwide geospatial intelligence (towns/villages/districts) ─
    from .geo import GeoIntelligenceEngine
    geo = GeoIntelligenceEngine(bus, telemetry)
    dispatcher.geo = geo                # the 'geo' locate/nearby tool
    # Resolve the PC's current location once and tell ORION where he is, so
    # weather, news and 'near me' default there without having to ask.
    asyncio.create_task(temporal.prime_locality(router))

    # ── Mark X: autonomous research, mind expansion, narrated web ────────────
    research = ResearchAgent(bus, router, memory, telemetry)
    corpus = KnowledgeCorpusBuilder(telemetry)
    dispatcher.research = research
    dispatcher.corpus = corpus
    web.router = router
    web.set_narrator(worker._say)          # ORION narrates browsing aloud
    # Build the 50 MB knowledge corpus once (idempotent), off the event loop.
    asyncio.create_task(asyncio.to_thread(corpus.build, memory))

    # ── JARVIS layer: protocols, reminders, sentinel, presence ───────────────
    protocols = ProtocolManager(bus, memory, telemetry)
    protocols.bind_dispatch(dispatcher.dispatch)
    reminders = ReminderService(bus, telemetry)
    sentinel = SentinelAgent(bus, telemetry)
    presence = PresenceMonitor(bus, telemetry)
    dispatcher.protocols = protocols
    dispatcher.reminders = reminders
    dispatcher.sentinel = sentinel

    # ── Studio deck: creative audio, academic intake, agency pipeline ────────
    audio_studio = AudioStudioService(bus, telemetry)
    literature = LiteratureIntakeService(bus, memory, telemetry)
    pipeline = AgencyPipelineService(bus, telemetry)
    dispatcher.audio_studio = audio_studio
    dispatcher.literature = literature
    dispatcher.pipeline = pipeline

    # ── Mark X.5: AI Operating System layer ──────────────────────────────────
    # Document production, the continuous cognitive loop, executive assistant
    # mode and scheduled reporting — all bus-decoupled, all offline-capable.
    exporter = DocumentExporterService(bus, telemetry)
    cognitive_loop = CognitiveLoopManager(
        bus, memory, cognition, graph=graph, workspace=workspace, telemetry=telemetry,
    )
    executive_mode = ExecutiveAssistantMode(
        bus, memory, cognition, reminders=reminders, notion=notion,
        router=router, graph=graph, telemetry=telemetry,
    )
    reporting = ProactiveReportingService(
        bus, exporter, router=router, proactive=proactive, commerce=commerce,
        pipeline=pipeline, cognition=cognition, outlook=outlook, notion=notion,
        memory=memory, telemetry=telemetry,
    )
    dispatcher.exporter = exporter
    dispatcher.reporting = reporting
    dispatcher.cognition = cognition
    dispatcher.cognitive_loop = cognitive_loop
    dispatcher.graph = graph
    dispatcher.executive = executive_mode
    # Momentum: turns tracked goals/tasks into decisive "ship this next" pressure.
    dispatcher.momentum = MomentumEngine(bus, memory, cognition, router=router, telemetry=telemetry)

    # ── Realistic-improvements batch ─────────────────────────────────────────
    ocr_engine = OcrEngine(bus)
    vision.attach_ocr_engine(ocr_engine)          # pluggable local OCR (#2)
    plan_executor = PlanExecutor(bus, dispatcher.dispatch, verifier, telemetry)  # (#3)
    file_organiser = FileOrganiser(bus, telemetry)                              # (#18)
    security = SecuritySentinel(bus, telemetry)                                 # (#17)
    backup = BackupManager(bus, telemetry)                                      # (#30)
    reports = ReportDrafter(bus, router, telemetry)                            # (#22,#19)
    dispatcher.plan_executor = plan_executor
    dispatcher.file_organiser = file_organiser
    dispatcher.security = security
    dispatcher.backup = backup
    dispatcher.reports = reports
    # JARVIS subsystems: web automation, peripherals, messaging, gaming, entertainment
    dispatcher.web_automation = web_automation
    dispatcher.peripherals = peripherals
    dispatcher.messaging = messaging
    dispatcher.gaming = gaming
    dispatcher.entertainment = entertainment
    # ── Mark X.7+: Forge capability-forging engine ────────────────────────────
    # ForgeOrchestrationManager creates its own sandbox, resolver, and loader internally.
    forge = ForgeOrchestrationManager(bus)
    dispatcher.forge = forge
    # ──────────────────────────────────────────────────────────────────────────
    # Full self-diagnostics + a visible cursor halo the user can see ORION drive.
    diagnostics = DiagnosticsEngine(bus, memory, telemetry, dispatcher)
    cursor_overlay = CursorOverlay(bus)
    dispatcher.diagnostics = diagnostics
    dispatcher.cursor_overlay = cursor_overlay

    # ── Living Memory: patch notes, learning intake, programming expertise ────
    changelog = Changelog()
    learning = LearningService(bus, memory, router, telemetry)
    programming = ProgrammingKnowledgeBase(telemetry)
    prog_seeded = programming.seed(memory)
    if prog_seeded:
        bus.log.emit(f"KNOWLEDGE: programming corpus seeded ({prog_seeded} entries).")
    cyber = CyberKnowledgeBase(telemetry)
    cyber_seeded = cyber.seed(memory)
    if cyber_seeded:
        bus.log.emit(f"KNOWLEDGE: cybersecurity corpus seeded ({cyber_seeded} entries).")
    dispatcher.changelog = changelog
    dispatcher.learning = learning
    dispatcher.programming = programming
    dispatcher.cyber = cyber

    # ── window 2: the unified, swipeable Command Deck ────────────────────────
    # The widget dashboard, command centre and global-intelligence globe are
    # merged into one window; swipe / arrow-keys / tabs move between them.
    dashboard = WidgetDashboardWindow(bus, agents, outlook, notion, dispatcher)
    command_centre = CommandCentreWindow(
        bus, telemetry, worker, memory, display, workspace, dispatcher, control,
    )
    globe = GlobeView(bus, geo=geo)     # town-accurate geocoding on the globe
    toolkit = EntrepreneurDeck(bus, memory, reminders=reminders, protocols=protocols)
    studio_deck = StudioDeckView(bus, memory, audio_studio, literature, pipeline)
    diagnostics_centre = DiagnosticsCentreView(bus, telemetry, dispatcher, worker)
    deck = UnifiedDashboard(bus, [
        ("WIDGETS", dashboard),
        ("TOOLKIT", toolkit),
        ("STUDIO", studio_deck),
        ("COMMAND CENTRE", command_centre),
        ("DIAGNOSTICS", diagnostics_centre),
        ("GLOBE", globe),
    ])
    deck.setStyleSheet(window.styleSheet())
    # Both header buttons / shortcuts open the same deck on the right page.
    window.attach_dashboard(deck)
    window.attach_command_centre(deck)
    # Mark X.5 dual-screen startup: core window on monitor 0, deck on
    # monitor 1 (or docked as a workspace panel on single-monitor hosts).
    _apply_startup_layout(window, deck, display, bus)

    window.attach_worker(worker)
    window.memory_view.refresh()
    window.telemetry_view.attach_env_refresh(
        lambda: asyncio.create_task(window.refresh_environment_widgets())
    )

    gateway: RemoteGateway | None = None
    if os.getenv("ORION_REMOTE_ACCESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        gateway = RemoteGateway(
            router, memory, bus,
            identity=identity, local_brain=local_brain,
            conversation=conversation, telemetry=telemetry,
        )
        try:
            await gateway.start()
        except Exception as exc:
            gateway = None
            bus.log.emit(f"REMOTE: uplink unavailable - {exc}")

    shutdown_event = asyncio.Event()

    def request_shutdown() -> None:
        if not shutdown_event.is_set():
            shutdown_event.set()

    app.aboutToQuit.connect(request_shutdown)
    bus.request_shutdown.connect(request_shutdown)

    # Self-healing runtime: capture faults on this loop and via sys.excepthook.
    loop = asyncio.get_running_loop()
    selfrepair.install(loop)

    telemetry_task = asyncio.create_task(window.start_telemetry(), name="orion-telemetry")
    asyncio.create_task(window.refresh_environment_widgets(), name="orion-environment-refresh")
    # Surface which microphone and speaker ORION is using, so a silent voice
    # is diagnosable at a glance (set with the audio_devices tool).
    from . import audio_devices as _audio_devices
    _audio_devices.log_startup(bus)

    worker_task = asyncio.create_task(worker.run(), name="orion-live-worker")
    briefing_task = asyncio.create_task(
        worker.offer_startup_briefing(), name="orion-startup-briefing"
    )
    proactive_task = asyncio.create_task(proactive.run(), name="orion-proactive")
    connectivity_task = asyncio.create_task(connectivity.run(), name="orion-connectivity")
    # JARVIS layer background loops.
    reminders_task = asyncio.create_task(reminders.run(), name="orion-reminders")
    sentinel_task = asyncio.create_task(sentinel.run(), name="orion-sentinel")
    presence_task = asyncio.create_task(presence.run(), name="orion-presence")
    # Studio deck background loops (directory monitor + pipeline heartbeat).
    audio_studio_task = asyncio.create_task(audio_studio.run(), name="orion-audio-studio")
    pipeline_task = asyncio.create_task(pipeline.run(), name="orion-pipeline")
    # Proactive cybersecurity monitor.
    security_task = asyncio.create_task(security.run(), name="orion-security")
    # Mark X.5: the continuous cognitive loop (awareness, never autonomy) and
    # the scheduled report generator.
    cognitive_loop_task = asyncio.create_task(cognitive_loop.run(), name="orion-cognitive-loop")
    reporting_task = asyncio.create_task(reporting.run(), name="orion-reporting")
    # Capture a workspace baseline so proactive change-tracking has a reference.
    asyncio.create_task(workspace.snapshot_workspace(), name="orion-workspace-baseline")

    try:
        await shutdown_event.wait()
    finally:
        bus.state.emit("SHUTTING DOWN")
        proactive.stop()
        connectivity.stop()
        reminders.stop()
        sentinel.stop()
        presence.stop()
        audio_studio.stop()
        pipeline.stop()
        security.stop()
        cognitive_loop.stop()
        reporting.stop()
        cursor_overlay.stop()
        telemetry_task.cancel()
        briefing_task.cancel()
        proactive_task.cancel()
        connectivity_task.cancel()
        reminders_task.cancel()
        sentinel_task.cancel()
        presence_task.cancel()
        audio_studio_task.cancel()
        pipeline_task.cancel()
        security_task.cancel()
        cognitive_loop_task.cancel()
        reporting_task.cancel()
        await worker.stop()
        if gateway is not None:
            await gateway.stop()
        worker_task.cancel()
        for task in (telemetry_task, briefing_task, worker_task, proactive_task,
                     connectivity_task, reminders_task, sentinel_task, presence_task,
                     audio_studio_task, pipeline_task, security_task,
                     cognitive_loop_task, reporting_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                bus.log.emit(f"SYS: shutdown task reported - {exc}")
        graph.close()
        geo.close()
        memory.close()


def main() -> None:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    # QtWebEngine (the Three.js globe) requires a shared OpenGL context set —
    # and ideally the module imported — BEFORE the QApplication is created.
    try:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        import PyQt6.QtWebEngineWidgets  # noqa: F401  (import-order requirement)
    except Exception:
        pass  # globe degrades gracefully if WebEngine is unavailable
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    # Terminal Ctrl+C should stop ORION cleanly. When launched with `py orion.py`
    # from a console, the Qt/asyncio loop spends most of its time in C code, which
    # starves Python's signal handling — so a plain Ctrl+C (and closing the last
    # window) previously left the process running. Fix both: install a SIGINT
    # handler that asks the app to quit (aboutToQuit → shutdown), and run a
    # periodic no-op timer so the interpreter regularly regains control to
    # actually deliver the signal.
    import signal
    from PyQt6.QtCore import QTimer
    try:
        signal.signal(signal.SIGINT, lambda *_: app.quit())
    except (ValueError, OSError):
        pass  # not on the main thread on some platforms — timer path still helps
    _sigint_timer = QTimer(app)
    _sigint_timer.setInterval(250)
    _sigint_timer.timeout.connect(lambda: None)
    _sigint_timer.start()

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    try:
        with loop:
            loop.run_until_complete(run_application(app))
    except KeyboardInterrupt:
        print("O.R.I.O.N. shutdown requested from console.")
    except SystemExit:
        raise
    except Exception:
        print("O.R.I.O.N. terminated after a controlled fault:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
