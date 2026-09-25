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
import time
import traceback
from typing import Optional

_APP_IMPORT_STARTED_AT = time.perf_counter()

import qasync
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

# NOTE: the ~80 service imports that used to sit here now load inside
# run_application(), immediately after the window has PAINTED. They cost about
# 160 ms and pull in 113 of ORION's 292 modules — antivirus, breach_monitor,
# the cyber curriculum, the forge — none of which the shell needs in order to
# appear. Paying for them here made every launch wait for subsystems the user
# cannot yet see. Only what the window itself needs is imported at module
# level; see the deferred block in run_application for the rest.
from .bus import OrionBus
from .constants import APP_NAME, CONFIG_DIR, CORE_DB_PATH
from .display import DisplayTopologyManager
from .gui import (
    ApiKeyDialog,
    CommandCentreWindow,
    HolographicToggle,
    OrionCoreWindow,
    WidgetDashboardWindow,
)
# Defined beside the window it configures, not here — see its docstring.
from .gui.core_window import unified_shell_enabled
from . import background
from .dynamic_loader import ReflectiveModuleLoader
from .dependencies import DynamicPackageResolver
from .sandbox import SandboxVerificationHarness
from .memory import MemoryAgent, OrionMemoryMatrix
# .providers is NOT imported here: it reaches cyber_knowledge and nine other
# modules, and the only things that need it are the two functions below (whose
# annotations are strings under `from __future__ import annotations`) and the
# deferred block in run_application. See tests/test_startup_imports.py.
from .telemetry import Telemetry
from .startup_budget import StartupBudget


# ──────────────────────────────────────────────────────────────────────────────
# DUAL-SCREEN STARTUP LAYOUT (Mark X.5)
# ──────────────────────────────────────────────────────────────────────────────

def _maximise_properly(win: QWidget) -> None:
    """Maximise a window so it ACTUALLY fills the screen, not just claims to.

    The reported bug: the Command Deck's title bar said "maximised" while the
    window sat small in the top-left corner. That is the exact signature of
    calling ``showMaximized()`` (or ``move()`` then ``showMaximized()``) on a
    window whose platform peer does not exist yet — Qt records the maximised
    STATE but computes the maximised GEOMETRY against a screen it has not
    resolved, so the frame never grows. It bites the SECOND window far more
    than the first, which is why the core window looked fine and the deck did
    not.

    The fix is to apply the maximised state after the window has actually been
    shown and the event loop has run once, by which point the peer exists and
    the correct screen geometry is known. A belt-and-braces explicit resize to
    the available geometry covers the platforms where the state flag alone is
    still not honoured.
    """
    from PyQt6.QtCore import QTimer

    win.show()

    def _do() -> None:
        try:
            screen = win.screen() or QGuiApplication.primaryScreen()
            if screen is not None:
                area = screen.availableGeometry()
                # Set the normal geometry to the full work area first, so that
                # even if the maximised flag is dropped later (a window-flags
                # change rebuilds the peer — a known hazard in this codebase)
                # the window still fills the screen rather than snapping to a
                # tiny constructor default.
                win.setGeometry(area)
            state = win.windowState()
            state &= ~Qt.WindowState.WindowMinimized
            win.setWindowState(state | Qt.WindowState.WindowMaximized)
            win.raise_()
        except Exception:
            try:
                win.showMaximized()
            except Exception:
                pass

    # Once now (covers the common case) and once on the next tick (covers the
    # not-yet-realised case that produced the top-left bug).
    _do()
    QTimer.singleShot(0, _do)


def _apply_unified_layout(
    window: QWidget, deck: QWidget, display: "DisplayTopologyManager", bus: OrionBus
) -> None:
    """One window: the face and the Command Deck share a single shell.

    The deck is docked into the core window rather than reconstructed, so it
    keeps every page, timer and piece of state it already had. Sizing simply
    maximises on the primary screen — with both surfaces in one window there
    is no second window to place, which is the entire point.
    """
    window.attach_deck_inline(deck)
    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        try:
            window.setScreen(screen)
        except Exception:
            pass
        window.move(screen.availableGeometry().topLeft())
    _maximise_properly(window)
    monitors = len(QGuiApplication.screens())
    bus.log.emit(
        f"DISPLAY: unified shell — face + Command Deck in one window "
        f"({monitors} monitor(s) detected; the second is left free).")


def _apply_startup_layout(
    window: QWidget, deck: QWidget, display: "DisplayTopologyManager", bus: OrionBus
) -> None:
    """
    Two monitors: core window maximised on monitor 0, Command Deck maximised
    on monitor 1, preserving each monitor's offset and DPI scale.  One
    monitor: core window docks the left half, deck docks the right half —
    sharing the one screen cleanly ("face on one screen, deck on the right")
    instead of the deck docking right while the core window stayed maximised
    full-screen underneath it.

    The DisplayTopologyManager provides the authoritative physical topology
    (and the log record); window placement itself goes through Qt's QScreen
    objects because Qt geometry is expressed in device-independent pixels —
    mixing the two coordinate spaces is exactly what misplaces windows on
    scaled monitors.
    """
    if unified_shell_enabled():
        _apply_unified_layout(window, deck, display, bus)
        return

    topology = display.topology()
    screens = QGuiApplication.screens()
    primary_screen = QGuiApplication.primaryScreen()
    secondary = next((s for s in screens if s is not primary_screen), None)

    if primary_screen is not None:
        try:
            window.setScreen(primary_screen)
        except Exception:
            pass  # Qt < 6.1 fallback: the geometry calls below still land it correctly

    if secondary is not None:
        if primary_screen is not None:
            window.move(primary_screen.availableGeometry().topLeft())
        _maximise_properly(window)
        primary_monitor = topology.primary
        bus.log.emit(
            "DISPLAY: core window → monitor "
            f"{primary_monitor.index if primary_monitor else 0} (primary), maximised."
        )
        try:
            deck.setScreen(secondary)
        except Exception:
            pass
        deck.move(secondary.availableGeometry().topLeft())
        _maximise_properly(deck)
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
            window.setGeometry(geometry.x(), geometry.y(), half, geometry.height())
            deck.setGeometry(geometry.x() + half, geometry.y(), half, geometry.height())
        # Maximised within their half, so each window fills the space it was
        # given rather than opening at its constructor's default size and
        # leaving the user to drag both out on every launch.
        _maximise_properly(window)
        _maximise_properly(deck)
        bus.log.emit(
            "DISPLAY: single monitor — core window docked left half, command "
            "deck docked right half; Ctrl+D toggles the deck as an overlay."
        )


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION BOOTSTRAP (GUI-assisted)
# ──────────────────────────────────────────────────────────────────────────────

def ensure_provider_settings(window: Optional[QWidget] = None) -> "OrionProviderSettings":
    from .providers import (_default_provider_payload, _settings_from_payload,
                            read_provider_settings, write_provider_settings)
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


_shutdown_watchdog_armed = False


def _arm_shutdown_watchdog(grace: float = 25.0) -> None:
    """Guarantee ORION actually terminates once shutdown is requested.

    The graceful teardown cancels ~20 tasks and closes a dozen services, and a
    single stubborn one — a wedged audio device, a GL context, a torch/mediapipe
    worker thread, an ``await task`` on something that will not cancel — can
    leave the process alive AFTER the goodbye and the deck have gone. That is
    exactly the "he says goodbye but doesn't fully shut down" symptom (and the
    same hang that makes the test process refuse to exit).

    A daemon thread waits ``grace`` seconds — comfortably longer than a real
    farewell plus teardown — and then FORCE-exits. The clean path returns well
    before this fires and the process is already gone; this only ever catches a
    genuine hang, turning "runs forever" into "gone within `grace` seconds"."""
    global _shutdown_watchdog_armed
    if _shutdown_watchdog_armed:
        return
    _shutdown_watchdog_armed = True
    import os
    import threading

    from .shutdown_trace import TRACE
    TRACE.begin()

    def _force_exit() -> None:
        time.sleep(max(1.0, float(grace)))
        # Cleanup already ran in the teardown; this is the last-resort kill for
        # a thread that will not let the interpreter exit on its own. Say
        # WHICH step it was stuck in first: os._exit leaves no other trace.
        TRACE.write(CONFIG_DIR / "diagnostics" / "last_shutdown.txt", forced=True)
        os._exit(0)

    threading.Thread(target=_force_exit, name="orion-shutdown-watchdog",
                     daemon=True).start()


async def _sample_metrics_history(telemetry: Any, period: float = 30.0) -> None:
    """Record ORION's metrics history whether or not any page is on screen.

    The only desktop caller of ``tick_history()`` used to be the Command
    Centre's refresh — which, sensibly, stopped refreshing while hidden. That
    optimisation also stopped the recording: the live metrics_history.db holds
    nothing after 2026-08-27, although ORION opened it every day since. A
    rolling history that only fills while one particular page is visible is not
    a history. ``tick_history`` is throttled internally (about once a minute)
    and writes one WAL row, so polling it here costs nothing noticeable.
    """
    while True:
        await asyncio.sleep(period)
        try:
            telemetry.tick_history()
        except Exception:
            pass          # a metrics write must never take the app down


def _release_portaudio_atexit() -> None:
    """Stop the interpreter's exit from tearing PortAudio down under a live stream.

    ``sounddevice`` registers an ``atexit`` handler that calls
    ``Pa_Terminate()``. ORION's audio threads are daemons, so when ``main()``
    returns one can still be inside ``stream.close()`` — and terminating
    PortAudio underneath it frees the state that close is using. That is not
    theoretical: of the 16 fatal crashes in the installed build's
    faulthandler.log, every one faulted inside ``sounddevice.close`` on an
    audio thread, and four show ``_exit_handler -> _terminate`` running at the
    same moment (STATUS_HEAP_CORRUPTION / access violation).

    Terminating PortAudio at process exit buys nothing — Windows releases every
    device handle when the process ends — so the handler is simply removed once
    ORION's own teardown has run. Every other ``atexit`` handler still runs.
    """
    sd = sys.modules.get("sounddevice")
    handler = getattr(sd, "_exit_handler", None)
    if handler is not None:
        import atexit
        atexit.unregister(handler)


async def _await_farewell(speech: Any, limit: float = 16.0) -> None:
    """Let ORION actually FINISH saying goodbye before teardown continues.

    The previous version waited a flat 2.5 s, which cut him off mid-sentence
    on anything but the shortest farewell — the goodbye is the last thing the
    user hears, and clipping it is worse than not saying it at all.

    This polls the speech pipeline's own is_busy() (speaking, or utterances
    still queued) and returns the moment it goes idle, so a short goodbye is
    still quick. `limit` is a hard ceiling so a wedged audio device can never
    hold shutdown open: the same bounded-wait guarantee as before, just
    driven by the real end of speech rather than a guess.

    It AWAITS rather than pumping Qt. It used to call processEvents() in a
    time.sleep() loop, on the belief that the asyncio side had already stopped
    by now — which was only true because bus.request_shutdown was also wired
    to QApplication.quit. Pumping Qt from inside run_application() dispatches
    other tasks' steps re-entrantly ("Cannot enter into task <orion-sentinel>
    while another task <run_application()> is being executed", seen on
    2026-09-23 for the turn watchdog, sentinel and reminders). The loop now
    stays up until teardown returns (see _run_until_torn_down), so a plain
    await keeps Qt painting and every task stepping in turn.
    """
    deadline = time.monotonic() + max(0.0, float(limit))
    # A brief grace period first: speak_text queues asynchronously, so
    # is_busy() can still read False for a moment after the call returns and
    # an immediate poll would exit before he has drawn breath.
    grace = time.monotonic() + 0.6
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        if time.monotonic() < grace:
            continue
        try:
            if not speech.is_busy():
                # Tail: a synthesis pipeline reports idle the moment it stops
                # FEEDING the device, but the last word or two is still draining
                # through the output buffer. A short tail lets him actually
                # finish the sentence instead of clipping the final syllables.
                await asyncio.sleep(1.3)
                return
        except Exception:
            return          # no usable pipeline — do not hold shutdown open


def _take_off_screen(window: Any) -> None:
    """Hide every top-level window and the tray icon.

    Hidden, not closed: closing the core window emits request_shutdown again,
    and a closed widget can be deleted under teardown steps that still hold it.
    The tray goes first so no ghost icon outlives the process.
    """
    tray = getattr(window, "_tray", None)
    if tray is not None:
        tray.hide()
    qapp = QApplication.instance()
    if qapp is not None:
        for widget in qapp.topLevelWidgets():
            widget.hide()


#: A boot phase slower than this is reported, always.  Startup was previously
#: unobservable: `budget.mark()` recorded a phase but nothing emitted it, so a
#: phase that stalled produced NO log line at all and the last thing you saw
#: was whatever happened to log just before it.  Observed on this machine: boot
#: reaches "[FORGE] Loader: starting load of 'dependency_manager_tool.py'" and
#: then goes silent for minutes with no indication of which phase is hung.
#: Reporting slow phases turns that into a named line.  ORION_BOOT_TRACE=1
#: reports every phase with its duration.
_BOOT_SLOW_PHASE_S = 1.5
_BOOT_TRACE = os.getenv("ORION_BOOT_TRACE", "").strip().lower() in {
    "1", "true", "yes", "on"}


async def _boot_phase(budget: "StartupBudget", name: str) -> None:
    """Mark a startup phase complete and let Qt actually service its event queue.

    The window is shown early (see the 'core window visible' phase below),
    but everything after that runs as one long block of service construction
    on the same task as the GUI — if Qt never gets a turn to process
    paint/input events until the whole thing finishes, that is exactly what
    "the whole program is unresponsive while loading" looks like.

    This SUSPENDS instead of pumping Qt directly, and the distinction matters.
    An earlier version called ``QApplication.processEvents()`` here on the
    reasoning that a zero-delay sleep does not guarantee a pass through Qt's
    own dispatch. Under qasync that reasoning is wrong and actively harmful:
    qasync schedules every asyncio callback as a **Qt timer event**
    (_QEventLoop.call_later -> _SimpleTimer.startTimer), so pumping Qt from
    inside a running task dispatches those timers re-entrantly and asyncio
    refuses with

        RuntimeError: Cannot enter into task <X> while another task
        <run_application()> is being executed

    Every task that lost that race was then discarded ("Task was destroyed
    but it is pending"), which silently killed real startup work —
    TemporalPresence.prime_locality, the forge's persisted-tool reload, the
    plugin loader and the research director's resume all never ran.

    Awaiting instead hands control back to qasync, which IS the Qt event
    loop: Qt paints and pumps its message queue exactly as it would
    normally, and the pending asyncio callbacks run through the proper path
    rather than re-entrantly. A small real delay rather than sleep(0) so Qt
    gets a definite slice instead of an immediately-requeued timer.
    """
    # mark() already returns the duration of the phase that just finished.
    elapsed = budget.mark(name)
    if _BOOT_TRACE or elapsed >= _BOOT_SLOW_PHASE_S:
        bus = getattr(budget, "bus", None)
        message = f"BOOT: '{name}' took {elapsed:.2f}s."
        if bus is not None:
            try:
                bus.log.emit(message)
            except Exception:
                pass
        else:
            print(message, flush=True)
    await asyncio.sleep(_BOOT_YIELD_S)


# Long enough that Qt reliably gets a scheduling slice for paints, short
# enough that ~40 boot phases add well under a tenth of a second in total.
_BOOT_YIELD_S = 0.002


# ──────────────────────────────────────────────────────────────────────────────
# APPLICATION LIFECYCLE
# ──────────────────────────────────────────────────────────────────────────────

# Both live in console_hygiene, which has no Qt dependency — a pure predicate
# must be importable without dragging QtWebEngine in behind it. Re-exported
# here so existing import sites are unchanged.
from .console_hygiene import QASYNC_STOPPED as _QASYNC_STOPPED  # noqa: E402
from .console_hygiene import is_clean_shutdown  # noqa: E402,F401


async def run_application(app: QApplication, *, started_at: float | None = None) -> bool:
    """Runs ORION until shutdown; returns True when a self-restart was asked."""
    _boot_t0 = time.perf_counter() if started_at is None else started_at
    budget = StartupBudget(target_seconds=5.0, started_at=_boot_t0)
    budget.mark("launcher + imports + Qt")
    # ── observability + memory before windows ───────────────────────────────
    bus       = OrionBus()
    # The budget carries the bus so _boot_phase can name a slow phase in the
    # log; without it a stalled phase produced no output whatsoever.
    budget.bus = bus
    telemetry = Telemetry(bus)
    for component in ("live_worker", "audio", "dispatcher", "vision", "control",
                      "proactive", "self_repair", "workspace"):
        telemetry.health.register(component)
    matrix = OrionMemoryMatrix(CORE_DB_PATH, CONFIG_DIR, bus)
    memory = MemoryAgent(matrix, bus)
    # Meaning-based recall (semantic.py): the sentence-embedding model loads
    # in a worker thread once start-up has settled, then everything already
    # remembered is embedded. Until then — and on a machine without the
    # model — recall runs on words exactly as before.
    import threading as _sem_threading
    from . import semantic as _semantic

    #: Stores indexed once the encoder is warm, filled in as they are built.
    _semantic_targets: dict[str, Any] = {}

    def _semantic_ready() -> None:
        indexed = matrix.index_semantics()
        if indexed:
            bus.log.emit(f"MEMORY: {indexed} memories and conversation turns indexed "
                         "by meaning — recall now understands paraphrases.")
        # The library (built further down this function; the timer fires long
        # after) is searched the same way once its chunks are embedded.
        library = _semantic_targets.get("library")
        if library is not None:
            chunks = library.index_semantics()
            if chunks:
                bus.log.emit(f"LIBRARY: {chunks} document chunks indexed by meaning.")
        graph_engine = _semantic_targets.get("graph")
        if graph_engine is not None:
            events = graph_engine.index_semantics()
            if events:
                bus.log.emit(f"GRAPH: {events} knowledge-graph events indexed by meaning.")
        # Tool purposes too, here in the worker, so the first routed turn
        # does not pay ~2 s embedding every tool on the turn's own thread.
        from . import tool_resolver as _tool_resolver
        _tool_resolver.warm_semantic()

    _sem_timer = _sem_threading.Timer(
        25.0, lambda: _semantic.warm_in_background(on_ready=_semantic_ready,
                                                   log=bus.log.emit))
    _sem_timer.daemon = True
    _sem_timer.start()
    # C2: opt-in change journal for desktop⇄cloud memory continuity. Off by
    # default until the /v1/sync replicator lands (an append-only log needs a
    # consumer before it earns its keep); ORION_SYNC_JOURNAL=1 turns it on.
    sync_journal = None
    if os.getenv("ORION_SYNC_JOURNAL", "").strip().lower() in {"1", "true", "on", "yes"}:
        from .sync import SyncJournal
        sync_journal = SyncJournal()          # one journal per node, shared below
        matrix.journal = sync_journal
        bus.log.emit("SYNC: change journal enabled (C2).")
    await _boot_phase(budget, "observability + memory")

    window = OrionCoreWindow(bus, memory)
    window.show()

    toggle = HolographicToggle(window)
    toggle.move(28, 28)
    toggle.show()

    # System-tray presence (Mark XXII): ORION stays resident in the notification
    # area with his own icon, menu and state tooltip, so he is always there like
    # a real OS assistant even with every window minimised. Held on the window so
    # it outlives this function; a no-op where there is no tray. When a tray is
    # present, closing a window hides ORION to it rather than quitting (an
    # explicit "Quit" still shuts him down) — see OrionCoreWindow.closeEvent.
    try:
        from . import desktop_app as _desktop_app
        from .system_tray import OrionTray
        from .utils import first_line as _first_line
        window._tray = OrionTray(app, window, bus,
                                 icon_path=str(_desktop_app.ICON_PATH))
        window._tray.show()
        if window._tray.active:
            app.setQuitOnLastWindowClosed(False)
    except Exception as exc:
        try:
            from .utils import first_line as _first_line
            bus.log.emit(f"SYS: system tray unavailable - {_first_line(exc)}")
        except Exception:
            pass

    # BOOT: hand the event loop a beat so Qt PAINTS the window now, before the
    # ~350 lines of service construction below run — the shell appears in a
    # fraction of the time instead of after everything is wired.
    await asyncio.sleep(0)
    budget.mark("core window visible")
    telemetry.metrics.gauge("startup.window_visible_seconds", budget.total)

    from .providers import OrionProviderSettings, ProviderRouter  # noqa: F401
    # ── deferred service imports ────────────────────────────────────────────
    # The shell is on screen and painted; everything below is needed only to
    # build the services that follow, so it is imported HERE rather than at
    # module level. Python caches modules, so this costs the same total work —
    # what changes is that the user is looking at ORION while it happens
    # instead of at nothing. Keep these together and keep them after the
    # paint: moving any one of them back to the header silently restores the
    # cost this block exists to defer (tests/test_startup_imports.py guards it).
    from .briefing import MorningBriefingService
    from .control import AutonomousControlLayer
    from .copilot import DeveloperCopilot
    from .dispatcher import OrionDispatcher, TOOL_DECLARATIONS
    from .audio_studio import AudioStudioService
    from .backup_manager import BackupManager
    from .change_awareness import SourceChangeTracker
    from .changelog import Changelog
    from .cognition import CognitiveStateManager
    from .cyber_knowledge import CyberKnowledgeBase
    from .cognitive_loop import CognitiveLoopManager
    from .diagnostics import DiagnosticsEngine
    from .executive import ExecutiveAssistantMode
    from .exporter import DocumentExporterService
    from .file_organiser import FileOrganiser
    from .forge import ForgeOrchestrationManager, ImprovementHeartbeat, LlmForgeBrain
    from .forge_lessons import ForgeLessonStore
    from .identity import IdentityManager
    from .knowledge_graph import KnowledgeGraphEngine
    from .learning import LearningService
    from .reporting import ProactiveReportingService
    from .gui.cursor_overlay import CursorOverlay
    from .gui.diagnostics_centre import DiagnosticsCentreView
    from .gui.entrepreneur import EntrepreneurDeck
    from .gui.views import LogConsoleView, MemoryMatrixView, TelemetryView
    from .ocr_engine import OcrEngine
    from .jobs import JobManager
    from .plan_executor import PlanExecutor
    from .programming_knowledge import ProgrammingKnowledgeBase
    from .reports import ReportDrafter
    from .security_sentinel import SecuritySentinel
    from .breach_monitor import BreachMonitor
    from .antivirus import AntivirusMonitor
    from .gui.library_deck import LibraryDeckView
    from .gui.unified_dashboard import UnifiedDashboard
    from .ingestion import IngestionEngine
    from .literature import LiteratureIntakeService
    from .pipeline import AgencyPipelineService
    from .knowledge import NeuroKnowledgeBase
    from .local_brain import LocalBrain
    from .mind_expansion import KnowledgeCorpusBuilder
    from .momentum import MomentumEngine
    from .presence import PresenceMonitor
    from .command_router import CommandRouter
    from .emergency import EmergencyProtocol
    from .health_model import HealthModel
    from .protocols import ProtocolManager
    from .request_trace import TRACES
    from .reminders import ReminderService
    from .research import ResearchAgent
    from .sentinel import SentinelAgent
    from .notion import NotionService
    from .outlook import OutlookService
    from .proactive import ProactiveIntelligence
    from .remote import RemoteGateway
    from .selfrepair import SelfRepairAgent
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
    from .maintenance import DatabaseHousekeeper
    from .browser_copilot import BrowserCopilot
    from .peripherals import PeripheralController
    from .messaging import MessagingGateway
    from .social_automation import SocialAutomationService
    from .security_recon import SecurityReconService
    from .debugger import DebuggerService
    from .navigation_trace import NavigationTrace
    from .voice_presence import VoicePresence
    from .voice_speaker_id import SpeakerIdentificationService
    from .pattern_detector import WorkflowPatternDetector
    from .gaming import GamingClientService
    from .entertainment import EntertainmentService


    # The window exists; now pull the heavy third-party imports in on a daemon
    # thread. google.genai alone is ~1.28 s and is imported lazily so it does
    # not delay this moment — but it is first NEEDED inside connect(), which
    # runs on the qasync loop, so warming it here is what stops that cost
    # simply reappearing as a freeze. See orion_core/lazy_import.warm.
    from .lazy_import import warm as _warm_imports
    _warm_imports(log=lambda msg: bus.log.emit(f"BOOT: {msg}"))

    # Historical telemetry opens/migrates SQLite. It is not required to paint
    # the shell; wait off the GUI thread now that the window is visible. The
    # in-memory metrics registry already collects the early startup timings.
    await asyncio.to_thread(telemetry.enable_history, CONFIG_DIR / "metrics_history.db")
    background.spawn(_sample_metrics_history(telemetry), name="orion-metrics-history")
    await _boot_phase(budget, "telemetry history")

    # Watch the event loop for stalls. Every lag bug in ORION's history was
    # something blocking this thread; the watchdog catches it live and names the
    # function, so "he stuttered" becomes a measurement instead of a hunt.
    # Ask him via diagnostics(action='latency'). ORION_STALL_WATCH=0 disables.
    if os.getenv("ORION_STALL_WATCH", "1").strip().lower() not in {"0", "false", "off", "no"}:
        from .latency import DETECTOR as _stall_detector
        _stall_detector.on_stall = lambda s: bus.log.emit(
            f"PERF: event loop blocked {s.ms:.0f} ms by {s.culprit()}")
        _stall_detector.start()
        window._stall_detector = _stall_detector      # keep it alive with the window
    bus.log.emit(f"BOOT: window visible at {time.perf_counter() - _boot_t0:.2f}s.")

    bus.log.emit(f"SYS: {APP_NAME} autonomous AI operating system initialised.")

    settings = ensure_provider_settings(window)

    await _boot_phase(budget, "identity")
    # ── personality consistency engine: one persona across every channel ─────
    identity = IdentityManager(bus, telemetry)
    identity.announce()

    await _boot_phase(budget, "identity announce")
    # ── dual-mode intelligence: connectivity + local models (MODE A/B) ───────
    for component in ("connectivity", "ollama", "commerce", "knowledge_packs"):
        telemetry.health.register(component)
    connectivity = ConnectivityMonitor(bus, telemetry)
    ollama = OllamaManager(bus, telemetry)
    # NOT ollama.register(settings) here.
    #
    # Measured: that call cost 2.04 s on this machine, all of it waiting for a
    # probe of 127.0.0.1:11434 that hangs rather than being refused. It ran on
    # the same task as the GUI, so it was two seconds in which Qt could not
    # pump a message — a visible chunk of "not responding". It is warmed in
    # the background instead (see the warm-up task below); until it lands,
    # ORION simply has no local provider, which is exactly the state he is in
    # anyway when Ollama is not installed.
    bus.log.emit(f"SYS: {connectivity.mode()} at startup.")

    await _boot_phase(budget, "connectivity + local models")
    # ── core services ────────────────────────────────────────────────────────
    router     = ProviderRouter(settings, bus, memory, connectivity=connectivity)
    # "No AI text provider available" should be something ORION tries to FIX.
    # If the user has Ollama installed but not running, this starts it and
    # registers a local model rather than simply reporting the dead end.
    router.attach_local_recovery(ollama.recover)
    router.attach_identity(identity)
    # Providers retire models without notice (Groq withdrew the one ORION was
    # configured with, and every request 404'd for days). One cheap listing
    # per provider, off the boot path, moves each onto a model it still serves.
    background.spawn(router.validate_models(), name="orion-model-check")
    # Learning about the user from what they say (fact_memory.py): lasting
    # facts in a user turn are kept without "remember that" — gated locally,
    # phrased by the model in the background, never storing a secret.
    from .fact_memory import FactHarvester
    fact_harvester = FactHarvester(memory, router, bus)
    fact_harvester.start()
    memory.fact_harvester = fact_harvester
    # Token-usage ledger (Section 6): records authoritative usage per text turn,
    # deduped by request id.  Degrades to a no-op if the DB can't be opened.
    try:
        from .token_usage import TokenUsageLedger
        router.attach_token_ledger(TokenUsageLedger())
    except Exception as exc:
        bus.log.emit(f"TELEMETRY: token ledger unavailable - {exc}")
    grabber    = VolatileScreenGrabber(bus)
    file_intel = LocalFileIntelligence(bus)
    vision     = VisionAgent(bus, grabber, file_intel)
    # Imported here, not at module scope: nothing above needs the agent
    # workforce, and at module scope it was ~61 ms spent before the window
    # could exist. warm() has been pulling it in on its own thread since
    # first paint, so by now this is a sys.modules lookup.
    from .agents import AgentManager, DesktopAgent
    desktop    = DesktopAgent(bus)
    outlook    = OutlookService(bus)
    notion     = NotionService(bus, settings.integration("notion"))
    agents     = AgentManager(router, bus)
    # Mark X.7: memory is passed for compatibility, but story dedup now lives
    # in the briefing's own TTL signature cache — the old memory-based check
    # was never wired here (the handle was omitted), so stories repeated.
    briefing   = MorningBriefingService(bus, notion, outlook, memory=memory)

    await _boot_phase(budget, "core services")
    # ── neuroscience expertise (KNOWLEDGE tier) — SEEDED off the boot path ────
    # (see the deferred _seed_corpora task below; construction here is cheap).
    knowledge = NeuroKnowledgeBase(telemetry)

    await _boot_phase(budget, "knowledge base")
    # ── Mark IX autonomy stack ───────────────────────────────────────────────
    display   = DisplayTopologyManager(telemetry)
    control   = AutonomousControlLayer(bus, display, telemetry, desktop)
    # Mark XXI, Track D7: one shared trace of every UI-targeting attempt,
    # built first so D1-D6's navigation-accuracy fixes could be measured
    # against real data rather than a feeling — also readable from
    # Diagnostics via dispatcher.navigation_trace.
    navigation_trace = NavigationTrace(bus)
    verifier  = VisualVerificationEngine(bus, control, vision, display, telemetry,
                                         trace=navigation_trace)
    web       = WebController(bus, control, vision, verifier, telemetry)
    # DesktopMemoryManager IS a WorkspaceManager, extended with named
    # workspace restoration ("restore marketing workspace") — Mark X.5.
    workspace = DesktopMemoryManager(bus, memory, desktop, telemetry)
    copilot   = DeveloperCopilot(bus, memory, router, telemetry)
    selfrepair = SelfRepairAgent(bus, telemetry, router)
    # Fed by the dispatcher on every tool call; read by the proactive survey,
    # which offers to name any run of tools that keeps recurring.
    pattern_detector = WorkflowPatternDetector()
    proactive = ProactiveIntelligence(bus, outlook, notion, memory, telemetry, workspace,
                                       pattern_detector=pattern_detector)
    bus.log.emit("SYS: " + display.summary())

    await _boot_phase(budget, "autonomy stack")
    # ── entrepreneurial + offline intelligence layer ─────────────────────────
    packs = KnowledgePackManager(bus, memory, telemetry)   # seeded off the boot path below
    # defer=True: loading faster-whisper costs ~2.8 s (measured) and nothing
    # needs offline transcription in the first seconds of a session. Warmed in
    # the background; any earlier caller triggers the load itself.
    offline_stt = OfflineTranscriber(bus, telemetry, defer=True)
    from . import speech_offline as _speech_offline
    _speech_offline.SHARED = offline_stt     # one model per process, not one per caller
    conversation = ConversationMemoryEngine(bus, memory, router, telemetry)
    commerce = CommerceSuite(bus, memory, router, knowledge=packs, telemetry=telemetry)
    community = CommunityHub(bus, memory, packs, telemetry)
    hub = EcommerceHub(bus, memory, commerce.dropship, packs)
    ai_info = AIModeInfo(router, connectivity, ollama, offline_stt)

    async def _warm_local_stack() -> None:
        """Pay for the slow local subsystems OFF the GUI thread.

        Together these were ~4.9 s of blocking work on ORION's startup path —
        the bulk of the window being unresponsive. Neither is needed to accept
        a command, so both are warmed here, in a worker thread, after the UI is
        already interactive. Failures are logged and dropped: a missing local
        model is a degraded mode, never a failed startup.
        """
        from .utils import first_line
        try:
            found = await asyncio.to_thread(ollama.register, settings)
            if found:
                bus.log.emit("SYS: local Ollama provider is now available (MODE B ready).")
        except Exception as exc:
            bus.log.emit(f"SYS: Ollama warm-up failed - {first_line(exc, 90)}")
        # The Whisper transcriber is NOT warmed here any more. The offline
        # voice loop runs on Vosk; Whisper serves phone dictation and video
        # analysis only, so warming it cost ~310 MB (measured) in every session
        # for work most never do. It loads on first use and is released after
        # ten idle minutes by its reaper.
        background.spawn(offline_stt.idle_reaper(), name="orion-stt-reaper")
        try:
            if await asyncio.to_thread(worker.vad.ensure_ready):
                bus.log.emit("AUDIO: Silero voice-activity model warmed.")
        except Exception as exc:
            bus.log.emit(f"AUDIO: VAD warm-up failed - {first_line(exc, 90)}")
        try:
            # Enumerating the shell's AppsFolder over COM costs ~1.5 s, and it
            # is what makes Store apps (WhatsApp, TikTok, Sticky Notes)
            # findable at all. Paid here rather than on the first "open
            # WhatsApp", which is exactly when a delay is most obvious.
            from .app_resolver import CATALOGUE

            await asyncio.to_thread(CATALOGUE.build)
            bus.log.emit(f"APP: {len(CATALOGUE.candidates)} launchable "
                         "applications catalogued.")
        except Exception as exc:
            bus.log.emit(f"APP: catalogue warm-up failed - {first_line(exc, 90)}")

    # Started at the END of startup (see below), not here: launching it mid-chain
    # just moves the stall to the next phase rather than removing it.
    await _boot_phase(budget, "entrepreneurial + offline intelligence")
    # ── Mark X.5: second brain + durable cognitive state ─────────────────────
    cognition = CognitiveStateManager(bus, memory, telemetry=telemetry)
    proactive.cognition = cognition
    graph = KnowledgeGraphEngine(bus, memory, telemetry=telemetry)
    _semantic_targets["graph"] = graph
    if sync_journal is not None:
        graph.journal = sync_journal          # C2: share the node's one journal
    # Companion Intelligence: the cross-session continuity ledger ("you were
    # working on X yesterday") over activities, goals, habits and achievements.
    from .companion import CompanionAgent, CompanionEngine
    companion = CompanionEngine(bus, memory=memory, cognition=cognition,
                                telemetry=telemetry)
    companion.begin_session()
    companion_agent = CompanionAgent(companion, bus)
    resume_summary = await cognition.restore_on_launch()
    if resume_summary:
        bus.log.emit(f"COG: {resume_summary[:160]}")

    await _boot_phase(budget, "second brain + cognitive state")
    # ── JARVIS Subsystems: web automation, peripherals, messaging, gaming, entertainment ──
    # The `web_automation` tool is served entirely by the real visible-browser
    # co-pilot (#11); the earlier no-op WebAutomationService stub was retired.
    web_copilot = BrowserCopilot(bus)      # #11 — real visible-browser co-pilot
    peripherals = PeripheralController(bus)
    messaging = MessagingGateway(bus)
    social = SocialAutomationService(bus)  # real-account TikTok/Instagram automation
    security_recon = SecurityReconService(bus)  # authorized-target pentest tooling
    gaming = GamingClientService(bus)
    entertainment = EntertainmentService(bus)
    debugger_service = DebuggerService(bus)  # real pdb-backed debug sessions (Development deck)
    voiceprint_service = SpeakerIdentificationService(bus)  # which PERSON is speaking
    # Who is speaking + how they sound, from the same utterance. Hung off the
    # transcriber because that is where a complete utterance exists off the
    # real-time audio thread; see voice_presence for why not in the gate.
    voice_presence = VoicePresence(bus, speaker_id=voiceprint_service)
    offline_stt.presence = voice_presence
    router.attach_voice_presence(voice_presence)   # tone reaches the model's context

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
    # The single heaviest import in the tree (~103 ms: it pulls audio, and
    # audio pulls numpy). Deferred off the pre-window path and warmed.
    from .live_worker import GenAILiveWorker
    worker = GenAILiveWorker(settings, bus, memory, dispatcher, router, telemetry, local_brain)
    dispatcher.live_worker = worker    # for the live self-description (capabilities)
    # Speaker gender recognition (#14): expose the worker's tracker to the tool
    # layer so "who was just talking?" can be answered.
    dispatcher.speaker_tracker = worker.speaker_tracker
    # And the voiceprint reader, so a LIVE session reads who is speaking and
    # how they sound — not only the offline transcriber. Without this line the
    # tone tool answers from a stale reading, the enrolled voiceprint is never
    # compared against anything, and the only-my-voice gate cannot apply to
    # the path most conversations actually take.
    worker.voice_presence = voice_presence
    # The HUD's mute button (and Ctrl+Shift+M) — the spoken "mute yourself"
    # reaches the same method from inside the worker.
    bus.voice_mute_request.connect(worker.set_voice_muted)
    if os.environ.get("ORION_START_MUTED", "").strip().lower() in {"1", "true", "yes", "on"}:
        # Start with the voice muted (a late-night launch, a demo, a check
        # run): replies arrive as text until "unmute" or the speaker button.
        worker.set_voice_muted(True, announce=False)

    await _boot_phase(budget, "jarvis subsystems")
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

    await _boot_phase(budget, "sentiment + temporal presence")
    # ── Phase 8: worldwide geospatial intelligence (towns/villages/districts) ─
    from .geo import GeoIntelligenceEngine
    geo = GeoIntelligenceEngine(bus, telemetry)
    dispatcher.geo = geo                # the 'geo' locate/nearby tool
    # Resolve the PC's current location once and tell ORION where he is, so
    # weather, news and 'near me' default there without having to ask.
    background.spawn(temporal.prime_locality(router))

    await _boot_phase(budget, "geospatial intelligence")
    # ── Mark X: autonomous research, mind expansion, narrated web ────────────
    research = ResearchAgent(bus, router, memory, telemetry)
    corpus = KnowledgeCorpusBuilder(telemetry)
    dispatcher.research = research
    dispatcher.corpus = corpus
    web.router = router
    web.set_narrator(worker._say)          # ORION narrates browsing aloud
    # Build the 50 MB knowledge corpus once (idempotent), off the event loop.
    background.spawn(asyncio.to_thread(corpus.build, memory))

    await _boot_phase(budget, "research + mind expansion")
    # ── JARVIS layer: protocols, reminders, sentinel, presence ───────────────
    protocols = ProtocolManager(bus, memory, telemetry)
    protocols.bind_dispatch(dispatcher.dispatch)
    # The emergency protocol is not a macro — it weighs source reliability and
    # whether an event actually reaches the user — so it runs its own pipeline
    # and is bound here rather than being expressed as tool steps.
    emergency = EmergencyProtocol(bus, briefing=getattr(dispatcher, "briefing", None),
                                  router=router, telemetry=telemetry)
    protocols.bind_native("emergency", lambda: emergency.run())
    dispatcher.emergency = emergency
    # One health model, fed by real probes; subsystems are added below as they
    # come into existence, so nothing is ever shown green merely by default.
    health = HealthModel(bus)
    emergency.health = health
    dispatcher.health_model = health
    reminders = ReminderService(bus, telemetry)
    sentinel = SentinelAgent(bus, telemetry)
    presence = PresenceMonitor(bus, telemetry)
    dispatcher.protocols = protocols
    dispatcher.reminders = reminders
    dispatcher.sentinel = sentinel

    await _boot_phase(budget, "protocols + reminders + sentinel")
    # ── Studio deck: creative audio, academic intake, agency pipeline ────────
    audio_studio = AudioStudioService(bus, telemetry)
    literature = LiteratureIntakeService(bus, memory, telemetry, graph=graph)
    pipeline = AgencyPipelineService(bus, telemetry)
    dispatcher.audio_studio = audio_studio
    dispatcher.literature = literature
    dispatcher.pipeline = pipeline

    def _studio_outcome(phase: str, data: Any) -> None:
        # The Studio Deck that drew this telemetry was retired in Mark XXXI;
        # the outcomes still matter, so they go to the activity log. Progress
        # and inventory ticks are left out — they were only ever animation.
        data = data if isinstance(data, dict) else {}
        if phase == "processed":
            bus.log.emit(f"STUDIO: processed {data.get('file', 'a take')}.")
        elif phase == "package":
            bus.log.emit(f"STUDIO: packaged {data.get('count', 0)} stems "
                         f"as {data.get('package', 'a package')}.")
        elif phase == "error":
            detail = f" - {data['error']}" if data.get("error") else ""
            bus.log.emit(f"STUDIO: could not process "
                         f"{data.get('file', 'a file')}{detail}")

    bus.audio_studio_activity.connect(_studio_outcome)

    await _boot_phase(budget, "studio deck")
    # ── Mark X.5: AI Operating System layer ──────────────────────────────────
    # Document production, the continuous cognitive loop, executive assistant
    # mode and scheduled reporting — all bus-decoupled, all offline-capable.
    exporter = DocumentExporterService(bus, telemetry)
    cognitive_loop = CognitiveLoopManager(
        bus, memory, cognition, graph=graph, workspace=workspace, telemetry=telemetry,
    )
    # Free-thinking stream: ORION's inner monologue on its own provider slot —
    # periodic reflections plus 'why I did that' decision commentary, surfaced
    # on the command deck, the log and the phone (never spoken, never acting).
    from .thought_stream import ThoughtStream
    thoughts = ThoughtStream(
        bus, memory, router, cognitive_loop=cognitive_loop, telemetry=telemetry,
    )
    # Noticing without acting is the least useful half of attention. This lets
    # the thought loop settle what it can MEASURE — keyed to facts rather than
    # to the words of a thought, so it costs no tokens and cannot act on a
    # sentence that was wrong. See orion_core/thought_actions.py.
    from .thought_actions import ThoughtActor
    thoughts.actor = ThoughtActor(bus=bus, config_dir=CONFIG_DIR)
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
    dispatcher.companion = companion
    dispatcher.companion_agent = companion_agent
    dispatcher.executive = executive_mode
    # Momentum: turns tracked goals/tasks into decisive "ship this next" pressure.
    dispatcher.momentum = MomentumEngine(bus, memory, cognition, router=router, telemetry=telemetry)

    await _boot_phase(budget, "AI operating system layer")
    # ── Realistic-improvements batch ─────────────────────────────────────────
    ocr_engine = OcrEngine(bus)
    vision.attach_ocr_engine(ocr_engine)          # pluggable local OCR (#2)
    # Unified file & folder ingestion: fingerprinted, deduplicated, offline —
    # feeds the knowledge graph + KNOWLEDGE memory, backs the Command Deck's
    # File Library page.  Shares the OCR engine so images ingest as text.
    ingestion = IngestionEngine(
        bus, memory=memory, knowledge_graph=graph, telemetry=telemetry,
        ocr=ocr_engine,
    )
    dispatcher.ingestion = ingestion
    _semantic_targets["library"] = ingestion
    plan_executor = PlanExecutor(bus, dispatcher.dispatch, verifier, telemetry)  # (#3)
    # Track B: long read-only work detaches from the conversation turn instead
    # of holding the microphone in PROCESSING until it finishes.  It routes
    # through dispatch_chain so a backgrounded tool behaves exactly as it would
    # in the foreground, including any follow-on chain it derives.
    jobs = JobManager(bus, dispatcher.dispatch_chain, telemetry)
    file_organiser = FileOrganiser(bus, telemetry)                              # (#18)
    security = SecuritySentinel(bus, telemetry)                                 # (#17)
    breach_monitor = BreachMonitor(bus, config=settings, telemetry=telemetry)   # HIBP
    antivirus = AntivirusMonitor(bus, telemetry=telemetry)              # Windows Defender awareness
    backup = BackupManager(bus, telemetry)                                      # (#30)
    reports = ReportDrafter(bus, router, telemetry)                            # (#22,#19)
    dispatcher.plan_executor = plan_executor
    dispatcher.jobs = jobs
    dispatcher.file_organiser = file_organiser
    dispatcher.security = security
    dispatcher.breach_monitor = breach_monitor
    dispatcher.antivirus = antivirus
    dispatcher.backup = backup
    dispatcher.reports = reports
    # JARVIS subsystems: web automation, peripherals, messaging, gaming, entertainment
    dispatcher.web_copilot = web_copilot
    dispatcher.peripherals = peripherals
    dispatcher.messaging = messaging
    dispatcher.social = social
    dispatcher.security_recon = security_recon
    dispatcher.debugger = debugger_service
    dispatcher.voiceprint_service = voiceprint_service
    dispatcher.voice_presence = voice_presence
    dispatcher.identity = identity
    dispatcher.navigation_trace = navigation_trace
    dispatcher.pattern_detector = pattern_detector
    dispatcher.gaming = gaming
    dispatcher.entertainment = entertainment
    await _boot_phase(budget, "realistic-improvements batch")
    # ── Mark X.7+: Forge capability-forging engine ────────────────────────────
    # ForgeOrchestrationManager creates its own sandbox, resolver, and loader
    # internally.  Mark X.8: the Forge finally gets its BRAIN — an LLM code
    # generator + self-healing fixer over the provider router (it was
    # previously constructed without one, so every forge attempt refused with
    # "No code generator configured") — plus the ADA-style autonomous
    # self-improvement heartbeat.
    # Mark II: the Forge keeps a lesson corpus, so a failure class that has
    # killed sessions before is warned about in the next generation prompt
    # instead of being rediscovered; and repairs are aimed by diagnosis, which
    # lets a broken TEST be fixed rather than the module being rewritten
    # forever to satisfy it.
    forge_lessons = ForgeLessonStore()
    forge_brain = LlmForgeBrain(bus, router, lessons=forge_lessons)
    forge = ForgeOrchestrationManager(
        bus,
        code_generator=forge_brain.generate,
        llm_fixer=forge_brain.fix,
        llm_repairer=forge_brain.repair,
        lessons=forge_lessons,
    )
    forge.dispatcher = dispatcher     # verified tools register live, no restart
    forge.repair = selfrepair         # failed sessions become visible incidents
    dispatcher.forge = forge
    # The reviewed plugins bundled with this build go into the live plugin
    # folder first, so the reload below picks up the delivered versions.
    from .plugin_registry import seed_shipped
    shipped = seed_shipped(CONFIG_DIR / "custom_tools")
    if shipped:
        bus.log.emit(f"PLUGIN: installed {len(shipped)} bundled plugin(s) - "
                     f"{', '.join(shipped)}")
    # Previously forged tools come back to life at startup (ADA parity) —
    # in the background, so a slow import never delays the boot path.
    background.spawn(forge.reload_persisted_tools(), name="orion-forge-reload")
    # Third-party plugins (config/custom_tools/*.plugin.json) — the
    # integration step plugin_manifest.py's own docstring calls for.  Reuses
    # forge's loader so a loaded plugin shows up in forge.health() too.
    from .plugin_manifest import load_plugins
    from .plugin_registry import PluginRegistry
    plugins = PluginRegistry(bus, CONFIG_DIR / "custom_tools", telemetry)
    dispatcher.plugins = plugins        # the 'plugin' tool's backing registry
    # Forged tools load through the code-only forge path, so they arrive with
    # NO declared capability tier and the remote gate has to guess at them.
    # Backfilling a manifest at the safe 'confirm' default closes that gap
    # before anything is loaded or any paired device can call one.
    try:
        backfilled = plugins.backfill_manifests()
        if backfilled:
            bus.log.emit(
                f"PLUGIN: {len(backfilled)} forged tool(s) gained a capability "
                f"tier — {', '.join(backfilled)}")
    except Exception as exc:
        bus.log.emit(f"PLUGIN: manifest backfill skipped - {exc}")
    # Mark XXVI: plugins can REACT to bus events (a declared "events" list plus
    # an on_event() handler), not only be called as tools. The bridge isolates
    # every handler — a faulting plugin is muted, never allowed to break ORION.
    from .plugin_events import PluginEventBridge
    plugin_events = PluginEventBridge(bus=bus)
    dispatcher.plugin_events = plugin_events

    # The boot roll-call. A silent boot is indistinguishable from a broken
    # one: a tool the forge quarantined, or an output device that reports
    # success and moves no audio, used to show up only as a capability that
    # mysteriously did not work later. The plugin line is printed by the
    # loader itself, since plugins load in the background and their counts
    # are not true yet here.
    # Where ORION's thinking will come from, and what it will cost. Said out
    # loud because "thinking for free", "not thinking at all" and "about to
    # bill you every five minutes" are three different situations that used
    # to look identical in the log.
    # On a worker thread: the check waits on a connection to the local model
    # server, which when it is not running cost 2.1 s of the event loop on
    # every boot (measured) — most of what the log called "forge engine".
    from .local_mind import ensure_local_mind

    async def _check_local_mind() -> None:
        try:
            await asyncio.to_thread(ensure_local_mind, bus)
        except Exception as exc:
            bus.log.emit(f"MIND: could not be checked - {exc}")

    background.spawn(_check_local_mind(), name="orion-local-mind-check")

    try:
        from .startup_report import report

        report(dispatcher, bus=bus)
    except Exception as exc:
        bus.log.emit(f"BOOT: roll-call skipped - {exc}")
    # Scheduled plugins: a manifest may declare when it wants to run rather
    # than waiting to be called. The supervisor is created before the load so
    # a plugin can be handed to it the moment its manifest is parsed.
    from .plugin_runner import PluginScheduler
    from .plugin_vault import VAULT_NAME, PluginVault

    try:
        plugin_vault = PluginVault(CONFIG_DIR / VAULT_NAME)
        status = plugin_vault.status()
        if not status.encrypted:
            bus.log.emit(f"VAULT: {status.detail}")
    except Exception as exc:
        bus.log.emit(f"VAULT: unavailable - {exc}")
        plugin_vault = None
    scheduler = PluginScheduler(dispatcher, bus, vault=plugin_vault)
    dispatcher.plugin_scheduler = scheduler
    dispatcher.plugin_vault = plugin_vault

    background.spawn(
        load_plugins(forge.loader, dispatcher, CONFIG_DIR / "custom_tools", bus,
                     registry=plugins, events=plugin_events,
                     scheduler=scheduler),
        name="orion-plugin-load",
    )
    background.spawn(scheduler.run(), name="orion-plugin-scheduler")
    improvement = ImprovementHeartbeat(bus, router, forge, telemetry, memory=memory)
    await _boot_phase(budget, "forge engine")
    # ── MCP host: connect ORION to Model Context Protocol servers (Gmail,
    # Google Calendar, filesystem, …) declared in config/mcp_servers.json.
    from .mcp_host import MCPHost
    mcp_host = MCPHost(bus, telemetry)   # telemetry: per-server health (Mark XXI, Track E2)
    dispatcher.mcp_host = mcp_host
    # Real outbound calls and texts. Built here because it needs the MCP host,
    # and until now nothing outside the tests ever built one — so "ring my
    # phone" had nowhere to go however Twilio was configured. Dialling is
    # default-deny: only numbers in config/telephony_contacts.json, and every
    # spoken message is screened first, because a phone call cannot be
    # un-sent and is often recorded at the far end.
    from .telephony import TelephonyGateway

    dispatcher.telephony = TelephonyGateway(
        mcp=mcp_host, bus=bus,
        guard=getattr(dispatcher, "spillage", None))
    # ──────────────────────────────────────────────────────────────────────────
    # Full self-diagnostics + a visible cursor halo the user can see ORION drive.
    diagnostics = DiagnosticsEngine(bus, memory, telemetry, dispatcher,
                                    forge=forge, repair=selfrepair)
    cursor_overlay = CursorOverlay(bus)
    dispatcher.diagnostics = diagnostics
    dispatcher.cursor_overlay = cursor_overlay
    await _boot_phase(budget, "mcp host + diagnostics")

    # ── Living Memory: patch notes, learning intake, programming expertise ────
    changelog = Changelog()
    change_tracker = SourceChangeTracker()              # file-level self-awareness (#15)
    learning = LearningService(bus, memory, router, telemetry)
    programming = ProgrammingKnowledgeBase(telemetry)   # seeded off the boot path below
    cyber = CyberKnowledgeBase(telemetry)               # seeded off the boot path below
    dispatcher.changelog = changelog
    dispatcher.change_tracker = change_tracker
    dispatcher.learning = learning
    dispatcher.programming = programming
    dispatcher.cyber = cyber

    await _boot_phase(budget, "living memory + learning")
    # ── Phase 3: Personal AI Operating System layer ──────────────────────────
    # The coordinating intelligence above the agents: executive core,
    # persistent research programme with an evidence store, installable
    # skills, workflow automation, the Creator Studio creator suite, dynamic
    # briefings and the system registries.
    from .briefing_engine import DynamicBriefingEngine
    from .cognition import GoalManager
    from .creator_intel import CreatorIntelSuite
    from .evidence import EvidenceEngine
    from .executive_core import ExecutiveCore
    from .reasoning import BUDGETS, ReasoningEngine, ReasoningTier
    from .perception import PerceptionLoop
    from .registries import SystemRegistries
    from .strategy import StrategyEngine
    from .research_director import ResearchDirector
    from .skills import SkillManager
    from .workflow_engine import AutomationManager, WorkflowEngine

    goal_manager = GoalManager(cognition)
    executive_core = ExecutiveCore(
        bus, cognition, goals=goal_manager, executive=executive_mode,
        router=router, graph=graph, telemetry=telemetry,
    )
    evidence = EvidenceEngine(bus, telemetry=telemetry)
    research_director = ResearchDirector(
        bus, research, cognition, evidence, graph=graph, telemetry=telemetry,
    )
    # Track C: the deliberation layer over the specialist workforce. Given the
    # dispatcher's own dispatch it can hand specialists read-only instruments
    # (memory, graph, files, situation), the evidence store for verification,
    # and the research agent for silent live-web lookups.
    reasoning = ReasoningEngine(
        bus, router, agents, dispatch=dispatcher.dispatch, evidence=evidence,
        research=research, telemetry=telemetry,
    )
    # Give the everyday agent_dispatch path (one specialist, one pass) the
    # same read-only instrument tray reasoning's STANDARD tier already
    # defines — "one specialist, free to look things up" — instead of it
    # remaining a blind persona-only prompt while the real capability sat
    # unused behind the separate, rarer `reason` tool.
    agents.attach_toolbelt_factory(lambda: reasoning.toolbelt(BUDGETS[ReasoningTier.STANDARD]))
    # Track D: exhaustive option-space search. Given the reasoning engine, its
    # final recommendation gets a panel and a red team rather than one prompt.
    strategy = StrategyEngine(bus, router, reasoning=reasoning, telemetry=telemetry)
    skills = SkillManager(bus, memory=memory, telemetry=telemetry)
    workflow_engine = WorkflowEngine(
        bus, dispatcher=dispatcher, cognition=cognition, telemetry=telemetry,
    )
    automation = AutomationManager(workflow_engine)
    creator_intel = CreatorIntelSuite(
        bus, router=router, memory=memory, telemetry=telemetry,
    )
    await _boot_phase(budget, "phase 3: personal AI OS")
    # ── ULTRON pass: mission-based operating model ───────────────────────────
    from .missions import MissionEngine
    missions = MissionEngine(
        bus, cognition=cognition, research_director=research_director,
        telemetry=telemetry,
    )
    research_director.missions = missions   # opportunity scanning sees missions
    briefing_engine = DynamicBriefingEngine(
        bus, cognition, briefing=briefing, companion=companion,
        executive_core=executive_core, telemetry=telemetry,
        research_director=research_director, missions=missions,
        security=security,
    )
    bus.dashboard_event.connect(briefing_engine.observe_event)
    await _boot_phase(budget, "mission-based operating model")
    # ── ULTRON pass: the avatar nervous system + webcam face tracking ────────
    from .avatar import AvatarController, AvatarEngine
    from .face_tracking import FaceTracker
    from .gesture_control import GestureEngine
    avatar = AvatarController(AvatarEngine())
    face_tracker = (FaceTracker(bus.face_tracking.emit)
                    if FaceTracker.available else None)
    # Gesture control BORROWS the tracker's live frame too, same convention
    # as vision/perception below — it only opens its own camera handle when
    # face tracking isn't running.
    gestures = (
        GestureEngine(
            bus, peripherals, desktop.media_control,
            frame_source=face_tracker.latest_frame if face_tracker is not None else None,
        )
        if GestureEngine.available else None
    )
    dispatcher.gestures = gestures
    # Let on-demand camera analysis BORROW the tracker's live frame instead of
    # opening a second handle on the same webcam (Windows refuses that with the
    # -1072873821 MSMF grab failure).  Harmless when the tracker is idle: it
    # returns None and vision falls back to opening the camera itself.
    if face_tracker is not None:
        vision.set_live_frame_source(face_tracker.latest_frame)
    # Camera visibility: the snapshot preview shows the exact still ORION sent
    # to the model, and the live view shows what the tracker is seeing — both
    # strictly read-only consumers that never open a camera handle themselves.
    from .gui.camera_preview import LiveCameraWindow, SnapshotPreviewWindow
    snapshot_preview = SnapshotPreviewWindow(bus)
    live_camera = LiveCameraWindow(bus, tracker=face_tracker)
    dispatcher.live_camera_window = live_camera
    from .electronics_controller import ElectronicsInspectionController
    from .gui.electronics_workbench import ElectronicsWorkbench
    electronics_workbench = ElectronicsWorkbench(bus)
    electronics_workbench.attach_tracker(face_tracker)
    electronics_controller = ElectronicsInspectionController(bus, vision, router)
    await _boot_phase(budget, "avatar + face tracking")
    # ── Track E: the perception loop ─────────────────────────────────────────
    # It BORROWS the tracker's frames rather than opening the camera itself —
    # two handles on one webcam is the MSMF -1072873821 failure. With no
    # tracker there is nothing to borrow, and the loop reports that honestly
    # instead of competing for the device.
    async def _describe_frame(frame: Any) -> str:
        # The frame the loop already holds, through the router's vision
        # models. This used to call analyse_camera and return its .text —
        # which is the INSTRUCTION to describe the frame, so the loop stored
        # "Give a thorough, forensic description…" as what it had seen.
        from .vision_describe import describe_image, frame_to_jpeg
        jpeg = frame_to_jpeg(frame) if frame is not None else None
        if jpeg is None:
            return ""
        ok, text = await describe_image(
            router, jpeg, "In one or two sentences, say what is in view and "
            "what has changed.", max_tokens=160)
        return text if ok else ""

    # Its instruments: OpenCV measurements (always), a local YOLO object
    # detector (ONNX Runtime, model installed on request), MediaPipe pose, and
    # the "when the camera sees X, do Y" rules that start workflows.
    from .object_detection import ObjectDetector
    from .pose_tracking import PoseTracker
    from .vision_rules import VisionRules
    vision_rules = VisionRules()
    perception = PerceptionLoop(
        bus,
        frame_source=face_tracker.latest_frame if face_tracker is not None else None,
        describe=_describe_frame,
        telemetry=telemetry,
        objects=ObjectDetector(),
        pose=PoseTracker(),
        rules=vision_rules,
        camera=face_tracker,
        on_workflow=workflow_engine.start,
    )
    dispatcher.perception = perception
    live_camera.attach_perception(perception)
    # The language→action network (intent_brain): built off the loop, and late
    # enough not to compete with start-up — seeding it the first time is ~2 s
    # of CPU, and it is not needed until the first tool call has happened.
    import threading as _threading
    from . import intent_brain as _intent_brain
    _intent_timer = _threading.Timer(20.0, _intent_brain.warm)
    _intent_timer.daemon = True
    _intent_timer.start()
    if vision_rules.watch_on_startup and vision_rules.enabled_rules():
        bus.log.emit(f"PERCEPTION: {len(vision_rules.enabled_rules())} vision rule(s) "
                     "armed at start-up — " + perception.start())
    registries = SystemRegistries()
    registries.capabilities.register_from_dispatcher(dispatcher, TOOL_DECLARATIONS)
    from . import __version__ as _orion_version
    # Widened in the Mark XX architectural-audit pass (Track G) from 16 to
    # 33 modules — specifically the ones the audit named as missing: the
    # memory matrix, the provider router, the dispatcher, the bus,
    # workflow_engine, vision, desktop control, Outlook/Notion, the agent
    # manager, and more. Dependencies are populated where the constructor
    # call above makes them unambiguous; an empty tuple means "not yet
    # mapped", not "has none" — an honest gap rather than a guess.
    for module_name, service, role, deps in (
        ("executive_core", executive_core, "decision/priority/strategy engines", ()),
        ("research_director", research_director, "persistent research programme", ()),
        ("evidence", evidence, "claim store with provenance", ()),
        ("reasoning", reasoning, "deliberation, critique and verification", ()),
        ("strategy", strategy, "exhaustive option-space search", ()),
        ("perception", perception, "continuous local vision + scene memory", ("vision",)),
        ("skills", skills, "installable skill packages", ()),
        ("automation", automation, "workflow automation", ("workflow_engine",)),
        ("creator_intel", creator_intel, "Creator Studio creator suite", ()),
        ("briefing_engine", briefing_engine, "dynamic adaptive briefings", ()),
        ("forge", forge, "capability forging", ()),
        ("graph", graph, "knowledge graph", ()),
        ("cognition", cognition, "durable cognitive state", ()),
        ("missions", missions, "mission-based operating model", ()),
        ("avatar", avatar, "avatar state machine + animation channels", ()),
        ("face_tracker", face_tracker, "webcam head tracking for eye contact", ()),
        ("memory", memory, "the memory matrix (working/short/long/semantic/episodic/procedural/project)", ()),
        ("identity", identity, "personality/persona consistency across every provider", ()),
        ("telemetry", telemetry, "structured logging, metrics and component health", ()),
        ("router", router, "multi-provider text/voice routing with failover", ()),
        ("vision", vision, "screen capture, OCR and visual understanding", ()),
        ("desktop", desktop, "OS-level app/window/cursor control", ()),
        ("outlook", outlook, "email read/summarise/draft", ()),
        ("notion", notion, "tasks, scheduling, calendar, project tracking", ()),
        ("agents", agents, "the six specialist agents + toolbelt-driven investigation", ("router",)),
        ("control", control, "verified autonomous desktop control", ("desktop", "telemetry")),
        ("proactive", proactive, "unprompted email/deadline/calendar surfacing", ("outlook", "notion", "memory", "telemetry")),
        ("companion", companion, "continuity ledger — goals, habits, achievements", ("memory", "cognition")),
        ("dispatcher", dispatcher, "the tool-call router every capability is reached through", ("memory", "vision", "desktop", "outlook", "notion", "agents")),
        ("bus", bus, "the Qt signal bus every subsystem communicates over", ()),
        ("workflow_engine", workflow_engine, "named ordered tool-call chains with retries", ("dispatcher", "cognition")),
        ("security_recon", security_recon, "authorized-target pentest tooling", ()),
        ("debugger_service", debugger_service, "real pdb-backed debug sessions", ()),
        ("voiceprint_service", voiceprint_service,
         "which PERSON is speaking, by trained voiceprint (consented enrolment)", ()),
        ("navigation_trace", navigation_trace, "UI-targeting attempt log (Track D7)", ()),
        ("pattern_detector", pattern_detector, "spots repeated tool sequences worth naming as workflows", ("dispatcher",)),
        ("registries", registries, "this registry itself — self-referential, for completeness", ()),
    ):
        registries.modules.register(module_name, service, role,
                                     dependencies=list(deps), version=_orion_version)
    registries.features.register(
        "phase3_executive_core", "decision challenge, priority queue, strategy")
    registries.features.register(
        "phase3_research_programme", "persistent agenda + evidence harvesting")
    registries.features.register(
        "phase3_skills_workflows", "skill packages + workflow automation")
    registries.features.register(
        "track_c_reasoning", "reasoning-budget tiers, tool-using specialists, "
        "multi-agent panel with adversarial critique, evidence-grounded "
        "verification of the synthesised answer")
    registries.features.register(
        "track_d_strategy", "exhaustive decision search: model-defined option "
        "space, thousands of candidates scored locally, Pareto frontier, "
        "Copeland tournament, model judgement on the finalists only")
    registries.features.register(
        "track_e_perception", "continuous local frame worker, motion and "
        "change events, scene memory with object permanence, adaptive cloud "
        "sampling only when something changes")
    registries.features.register(
        "phase3_creator_intel", "Creator Studio intelligence suite")
    registries.features.register(
        "ultron_avatar_system", "AvatarEngine/Controller/StateMachine/"
        "AnimationManager: 8 states, cross-fades, lip envelope, blink, "
        "breathing, face-tracking head follow")
    registries.features.register(
        "ultron_mission_model", "MissionEngine + mission tool + Mission Deck "
        "command center (14 dockable panels, persistent layouts)")
    registries.features.register(
        "ultron_product_pipeline", "CTA/competitor analysers, trend tracker, "
        "full product intelligence pipeline")
    registries.features.register(
        "ultron_briefing_types", "research/mission/security/opportunity "
        "briefings + source validation + opportunity scanning")
    dispatcher.executive_core = executive_core
    dispatcher.research_director = research_director
    dispatcher.evidence = evidence
    dispatcher.reasoning = reasoning
    dispatcher.strategy = strategy
    dispatcher.skills = skills
    dispatcher.automation = automation
    dispatcher.creator_intel = creator_intel
    dispatcher.briefing_engine = briefing_engine
    dispatcher.registries = registries
    dispatcher.missions = missions
    dispatcher.avatar = avatar
    dispatcher.face_tracker = face_tracker
    # Resume any research programme interrupted by the last shutdown.
    background.spawn(research_director.resume_pending(),
                        name="orion-research-resume")

    await _boot_phase(budget, "executive core + registries")
    # ── window 2: the unified, swipeable Command Deck ────────────────────────
    # The widget dashboard, command centre and global-intelligence globe are
    # merged into one window; swipe / arrow-keys / tabs move between them.
    # Each deck page below is a full widget tree; built back-to-back with no
    # yields this was the single longest unbroken block in the whole boot, and
    # the most visible one — it is what the user is staring at while waiting.
    dashboard = WidgetDashboardWindow(bus, agents, outlook, notion, dispatcher)
    await _boot_phase(budget, "deck: widgets")
    command_centre = CommandCentreWindow(
        bus, telemetry, worker, memory, display, workspace, dispatcher, control,
    )
    await _boot_phase(budget, "deck: command centre")
    # DEFERRED. GlobeView is a WebEngine surface: building it brings up
    # Chromium subprocesses and a GL context, and nothing touches it between
    # here and the user opening the page. Built on first navigation instead.
    from .gui.globe import GlobeView
    def globe() -> Any:                 # town-accurate geocoding on the globe
        return GlobeView(bus, geo=geo)
    toolkit = EntrepreneurDeck(bus, memory, reminders=reminders, protocols=protocols)
    # Mark XXXI: the STUDIO page is gone. It drew two file counts, a text log
    # and a kanban nobody used; the audio studio, literature intake and agency
    # pipeline behind it are all still live and reachable by voice.
    await _boot_phase(budget, "deck: toolkit")
    diagnostics_centre = DiagnosticsCentreView(bus, telemetry, dispatcher, worker)
    library_deck = LibraryDeckView(bus, engine=ingestion)
    # The plugin ecosystem's visible face — enable/disable, health and
    # scaffolding, backed by the SAME registry the 'plugin' tool uses so the
    # two paths can never disagree.
    from .gui.plugin_deck import PluginDeckView
    plugin_deck = PluginDeckView(bus, plugins)
    await _boot_phase(budget, "deck: diagnostics + library + plugins")
    # LOG/MEMORY/TELEMETRY used to live on the Core Window; that window is
    # face-only now, so these three become ordinary deck pages like every
    # other view here.
    log_view = LogConsoleView(bus)
    memory_view = MemoryMatrixView(memory)
    telemetry_view = TelemetryView(bus)
    # Chess: real Stockfish play + analysis, built into its own deck page.
    # router/telemetry are passed through so move commentary is conversational
    # (LLM-backed) rather than templated-only, and autonomous-play status can
    # surface through the same telemetry the rest of the app already uses.
    from .chess_engine import ChessService
    from .gui.chess_view import ChessPanel
    chess_service = ChessService(bus, router=router, telemetry=telemetry)
    dispatcher.chess = chess_service
    # DEFERRED — see the globe above. The chess service is live either way;
    # only the board is built on demand.
    def chess_view() -> Any:
        return ChessPanel(bus, chess_service)
    # Security Centre: SecuritySentinel previously had no GUI surface at all —
    # posture/alerts only ever flashed on the HUD banner for a few seconds.
    from .gui.security_centre import SecurityCentreView
    security_centre = SecurityCentreView(bus, security=security)
    await _boot_phase(budget, "deck: chess + security centre")
    from .gui.ops_deck import OperationsDeckView
    ops_deck = OperationsDeckView(
        bus, companion=companion, agents=agents, cognition=cognition,
        memory=memory, sentinel=security, desktop=desktop,
        executive_core=executive_core,
    )
    await _boot_phase(budget, "deck: operations")
    # ULTRON pass: the dockable mission command center leads the deck.
    from .gui.mission_deck import MissionDeckView
    # DEFERRED. Fourteen dock panels, none of which anything reads until the
    # page is opened; every engine it draws from is already live.
    def mission_deck() -> Any:
        return MissionDeckView(
            bus, missions=missions, cognition=cognition, memory=memory,
            research_director=research_director, evidence=evidence,
            telemetry=telemetry, workflow_engine=workflow_engine, graph=graph,
            ingestion=ingestion, companion=companion, security=security,
            workspace=workspace, agents=agents, cognitive_loop=cognitive_loop,
        )
    # Mark XX design-spec: AUTOMATION and DEVELOPMENT were disabled, empty
    # zone tabs — there was no page to enable them with, even though their
    # backends (WorkflowEngine, dev_workbench) were already real and
    # working. These pages read/write the SAME engine instances the
    # dispatcher's own tools use, so nothing here is a parallel or weaker
    # implementation — it's the first GUI surface either has ever had.
    from .gui.automation_deck import AutomationDeckView
    from .gui.development_deck import DevelopmentDeckView
    automation_deck = AutomationDeckView(bus, workflow_engine, dispatcher=dispatcher)
    development_deck = DevelopmentDeckView(bus, dispatcher=dispatcher)
    await _boot_phase(budget, "deck: automation + development")

    # Mark XX design-spec §6: each specialist gets a real professional
    # workspace (Brief/Findings/Context/Instruments) instead of sharing one
    # generic combo-box-and-textbox panel — one template (AgentWorkspaceView),
    # six domain fills, all calling the SAME AgentManager.dispatch() the
    # WIDGETS panel and voice/text agent_dispatch already use.
    #
    # Mark XXXI: only MARKETING keeps a workspace page. DESIGN, FASHION and
    # ENTERTAINMENT were chat personas with a text box and nothing behind
    # them, and are retired along with their agents; CODING duplicated the
    # DEVELOPMENT page, which has the real instruments. RESEARCH is no longer
    # a chat box either — it is the live research console, where every search,
    # page and note of a run is shown as it happens.
    from .gui.agent_workspace import AgentWorkspaceView
    from .gui.research_console import ResearchConsole
    agent_workspaces = {
        "MARKETING": AgentWorkspaceView(
            bus, agents, "marketing", "Digital Marketing Agent", (
                "Draft a launch campaign", "Improve this ad copy",
                "Suggest a content calendar", "Analyse this funnel")),
    }
    research_console = ResearchConsole(bus, dispatcher=dispatcher)
    await _boot_phase(budget, "deck: marketing + research console")

    # Native 2D overview. Keep the BRAIN route key for saved and spoken
    # navigation; the visible title is Overview. Qt bus signals update its
    # labels directly, with no additional browser renderer or transport.
    from .gui.command_overview import CommandOverview
    swarm_view = CommandOverview(
        bus,
        on_open_page=lambda page: window._open_deck_page(str(page)),
    )
    dispatcher.swarm_view = swarm_view   # Track B5: pulse() on every tool call
    # Keep the point-cloud fallback only when the separate face needs it.
    core_swarm_view = None
    if hasattr(window.face, "attach_renderer"):
        from .gui.swarm_view import SwarmDeckView
        core_swarm_view = SwarmDeckView(
            bus, agents=agents, registries=registries, telemetry=telemetry,
            mcp_host=mcp_host, workflow_engine=workflow_engine, memory=memory,
            on_open_agent=lambda name: window._open_deck_page(str(name).upper()),
            on_open_zone=lambda zone: window._open_deck_zone(str(zone)),
            on_open_page=lambda page: window._open_deck_page(str(page)),
        )
        window.attach_face_renderer(core_swarm_view)
    await _boot_phase(budget, "deck: face renderer")

    await _boot_phase(budget, "deck: overview")

    # COGNITION deck page (Mark XXVI, Phase 1): the learning loop on screen. It
    # shares the SAME study/focus engine instances the voice tools use, so the
    # page and the tools are one source of truth (the qasync loop unifies the GUI
    # and event-loop threads). Constructed eagerly here — cheap SQLite opens —
    # and attached to the dispatcher, which also hands the study tool its
    # model-backed card generator.
    from .study import StudyEngine
    from .focus import FocusEngine
    from .gui.cognition_deck import CognitionDeckView
    if getattr(dispatcher, "study", None) is None:
        dispatcher.study = StudyEngine(generate=dispatcher._study_generator())
    if getattr(dispatcher, "focus", None) is None:
        dispatcher.focus = FocusEngine()
    cognition_deck = CognitionDeckView(bus, study=dispatcher.study, focus=dispatcher.focus)
    await _boot_phase(budget, "deck: cognition")

    deck = UnifiedDashboard(bus, [
        ("BRAIN", swarm_view),
        ("WORKBENCH", electronics_workbench),
        ("MISSION", mission_deck),
        ("RESEARCH", research_console),
        ("DEVELOPMENT", development_deck),
        ("WIDGETS", dashboard),
        ("TOOLKIT", toolkit),
        ("MARKETING", agent_workspaces["MARKETING"]),
        ("LIBRARY", library_deck),
        ("OPS", ops_deck),
        ("COGNITION", cognition_deck),
        ("AUTOMATION", automation_deck),
        ("COMMAND CENTRE", command_centre),
        ("PLUGINS", plugin_deck),
        ("DIAGNOSTICS", diagnostics_centre),
        ("GLOBE", globe),
        ("LOG", log_view),
        ("MEMORY", memory_view),
        ("TELEMETRY", telemetry_view),
        ("CHESS", chess_view),
        ("SECURITY", security_centre),
    ])
    # The status strip shows the FULL picture — network, focus block, mission —
    # composed from the live engines behind one guarded call.
    def _is_online(monitor) -> bool:
        try:
            return bool(monitor.is_online())
        except Exception:
            return True        # unknown -> do not claim ORION is offline

    def _op_status(state: str):
        from .operational_status import from_context
        return from_context(
            state=state,
            online=_is_online(connectivity),
            focus_engine=getattr(dispatcher, "focus", None),
            mission_engine=getattr(dispatcher, "missions", None),
            health_model=health,
        )
    deck.attach_status_provider(_op_status)
    # Voice-triggered scans land on the same visible camera/report surface as
    # the Capture button, including when the deck was previously hidden.
    bus.electronics_scan_started.connect(
        lambda _payload: window._toggle_deck_page("WORKBENCH", "Electronics workbench"))
    deck.setStyleSheet(window.styleSheet())
    # So interface_control can tell a real page from a misheard one BEFORE it
    # reports success. Without these it emitted onto the bus and said "done"
    # whatever happened at the other end.
    dispatcher.page_resolver = deck.resolve_page_name
    dispatcher.page_names = deck.page_names
    # Migrate the deck INTO the swarm: every page becomes a node in its zone's
    # cluster, so the graph holds all of ORION and not just his agents. Built
    # from the deck's OWN page list and zone table, so a page added later
    # appears in the swarm without anything here being updated. Pages outside
    # every zone (CHESS is deliberately one) still get a node, under SYSTEM,
    # rather than being the one part of ORION the swarm cannot reach.
    _zone_of_page = {
        page: zone
        for zone, pages in UnifiedDashboard.ZONE_PAGES.items()
        for page in pages
    }
    _deck_pages = {name: _zone_of_page.get(name, "SYSTEM")
                   for name in deck.page_names()}
    for _swarm in (swarm_view, core_swarm_view):
        if _swarm is not None:
            _swarm.attach_deck_pages(_deck_pages)
    # Keep the old compact-navigation setting compatible with existing
    # installations. The new sidebar remains available in either mode.
    if os.getenv("ORION_SWARM_NAV", "").strip().lower() in {"1", "true", "on", "yes"}:
        deck.set_swarm_navigation(True)
        bus.log.emit("SYS: compact deck tabs enabled (ORION_SWARM_NAV=1).")
    # ORION's face moves to the CENTRE of the network — the graph reads as a
    # brain with him at the middle of it and every edge radiating out from
    # him, instead of his face being a small bright panel off to one side.
    # The core node is drawn at radius 0 and the layout holds a keep-out
    # sphere clear at the centre, so he lands in genuinely empty space.
    # ORION_FACE_CENTRE=0 keeps the face in its own side panel.
    # ORION's face owns the Core Window; the Command Deck is its own window.
    # (The legacy "face overlaid at the centre of the swarm" path went with
    # the swarm map in Mark XXXI.)
    window.content_splitter.widget(0).show()
    window.content_splitter.setSizes([max(window.width(), 900), 0, 0])
    window.swarm_rail.hide()
    window.deck_panel.hide()
    bus.log.emit("SYS: ORION's face is on the primary monitor; "
                 "the Command Deck has its own window.")
    # Both header buttons / shortcuts open the same deck on the right page.
    window.attach_dashboard(deck)
    window.attach_command_centre(deck)
    # Mark X.5 dual-screen startup: core window on monitor 0, deck on
    # monitor 1 (or docked as a workspace panel on single-monitor hosts).
    _apply_startup_layout(window, deck, display, bus)

    # The avatar controller takes over the face rig (cross-fades, lip
    # envelope, breathing, blink, head follow) on a 20 Hz behaviour tick.
    window.attach_avatar(avatar)

    window.attach_worker(worker)
    worker.companion_agent = companion_agent   # passive "working on X" observer

    # ── real commands, real health (brief §20, §23) ───────────────────────────
    # The Command Palette used to only ever prefill a text box.  Every entry
    # now resolves to a handler that performs the thing it names, and the
    # result comes back to the UI as an event rather than by touching widgets.
    TRACES.attach_bus(bus)
    background.attach_bus(bus)   # background failures become visible, not silent
    command_router = CommandRouter(
        bus, dispatcher=dispatcher, protocols=protocols, health=health,
        recovery=worker.recovery, worker=worker, deck=deck)
    window.attach_command_router(command_router)
    health.register_defaults(
        worker=worker, dispatcher=dispatcher, router=router, memory=memory,
        recovery=worker.recovery, protocols=protocols, deck=deck,
        command_router=command_router, mcp=mcp_host, security=security)
    bus.log.emit(
        f"SYS: health model watching {len(health.registered())} subsystems; "
        f"command router exposes {len(command_router.commands())} real commands.")

    # NOW warm the slow local subsystems — the whole UI is built and
    # interactive, so a worker thread competing for the loop costs the user
    # nothing. Started here rather than where they are constructed because
    # doing it mid-chain moved the stall instead of removing it.
    background.spawn(_warm_local_stack(), name="orion-warm-local")
    log_view.attach_worker(worker)
    telemetry_view.attach_worker(worker)
    worker.set_microphone_enabled(log_view.mic_toggle.isChecked())
    # The LOG input line and the Environment panel moved off the Core Window
    # onto the deck; attach them late so self-navigation (Ctrl+K prefill,
    # "Orion, refresh the environment") still routes correctly.
    window.attach_log_view(log_view)
    window.attach_environment_panel(ops_deck.environment_panel)
    memory_view.refresh()
    telemetry_view.attach_env_refresh(
        lambda: background.spawn(ops_deck.environment_panel.refresh_environment_widgets())
    )

    # Desktop-only by default (user request: "get rid of ORION on localhost, I
    # want him just on the app"). Now that ORION is a proper desktop application
    # in the taskbar, the local web server is redundant as a LOCAL interface, so
    # a normal session no longer stands one up — nothing listens on localhost.
    # The phone/remote uplink is kept as an opt-in FALLBACK: set
    # ORION_REMOTE_ACCESS=1 (or ask ORION to "turn on phone access") to bring it
    # back for a device on the same Wi-Fi. The headless/cloud node forces it on
    # separately (server.py) — that is its whole reason to exist.
    gateway: RemoteGateway | None = None
    if os.getenv("ORION_REMOTE_ACCESS", "0").strip().lower() in {"1", "true", "yes", "on"}:
        gateway = RemoteGateway(
            router, memory, bus,
            identity=identity, local_brain=local_brain,
            conversation=conversation, telemetry=telemetry,
            dispatcher=dispatcher, knowledge=knowledge,
        )
        try:
            await gateway.start()
            ops_deck.attach_gateway(gateway)   # Security Centre: QR pairing + devices
        except Exception as exc:
            gateway = None
            bus.log.emit(f"REMOTE: uplink unavailable - {exc}")

    # "Turn on/off phone access" reads dispatcher.gateway and calls
    # dispatcher.start_remote_gateway; neither was ever set, so "phone off"
    # always answered "already off" (while the uplink was running) and
    # "phone on" always asked for a restart.
    async def _start_remote_gateway() -> None:
        nonlocal gateway
        if getattr(dispatcher, "gateway", None) is not None:
            return
        started = RemoteGateway(
            router, memory, bus,
            identity=identity, local_brain=local_brain,
            conversation=conversation, telemetry=telemetry,
            dispatcher=dispatcher, knowledge=knowledge,
        )
        await started.start()
        ops_deck.attach_gateway(started)
        gateway = dispatcher.gateway = started

    dispatcher.gateway = gateway
    dispatcher.start_remote_gateway = _start_remote_gateway

    shutdown_event = asyncio.Event()
    restart_requested = False

    def request_shutdown() -> None:
        if not shutdown_event.is_set():
            shutdown_event.set()
            # Guarantee termination even if a background thread refuses to die.
            _arm_shutdown_watchdog()

    def request_restart() -> None:
        # Mark X.6: ORION restarts himself — same clean teardown as shutdown,
        # then main() respawns the process once the loop has closed.
        nonlocal restart_requested
        restart_requested = True
        request_shutdown()

    app.aboutToQuit.connect(request_shutdown)
    bus.request_shutdown.connect(request_shutdown)
    bus.request_restart.connect(request_restart)

    # Self-healing runtime: capture faults on this loop and via sys.excepthook.
    loop = asyncio.get_running_loop()
    selfrepair.install(loop)
    # Black-box crash recorder: writes plain-text reports to config/crash_reports
    # for the failures selfrepair can't see — Qt/WebEngine/GPU fatals and hard
    # C-level render-process crashes (the globe's usual culprits).  Installed
    # AFTER selfrepair so its excepthook wraps (and still calls) selfrepair's.
    from . import crash_reporter as _crash_reporter
    _crash_reporter.install(bus, telemetry)

    telemetry_task = asyncio.create_task(window.start_telemetry(), name="orion-telemetry")
    background.spawn(
        ops_deck.environment_panel.run_environment_loop(), name="orion-environment-refresh"
    )
    # Surface which microphone and speaker ORION is using, so a silent voice
    # is diagnosable at a glance (set with the audio_devices tool).
    from . import audio_devices as _audio_devices
    _audio_devices.log_startup(bus)
    # Fingerprint ORION's own source so he knows which of his files changed this
    # update — reported by the self_changes tool (#15).
    change_tracker.log_startup(bus)

    worker_task = asyncio.create_task(worker.run(), name="orion-live-worker")
    briefing_task = asyncio.create_task(
        worker.offer_startup_briefing(), name="orion-startup-briefing"
    )
    proactive_task = asyncio.create_task(proactive.run(), name="orion-proactive")
    connectivity_task = asyncio.create_task(connectivity.run(), name="orion-connectivity")
    # Mark XXVI: the learning-loop proactivity engine — volunteers a break-due
    # focus block or a review backlog aloud, gated by the shared speech policy.
    # Opt-in (ORION_PROACTIVE_ENGINE): off by default so runtime is unchanged.
    from .proactivity_engine import ProactivityEngine, engine_enabled as _proactive_engine_enabled
    if _proactive_engine_enabled():
        proactivity_engine = ProactivityEngine(
            bus, focus=getattr(dispatcher, "focus", None),
            study=getattr(dispatcher, "study", None))
        # Let it see the learned-reflex candidates, so ORION can offer to answer
        # a phrase instantly instead of banking candidates nobody ever promotes.
        try:
            proactivity_engine.reflex_learner = worker.reflex_learner()
        except Exception:
            pass
        proactivity_engine_task = asyncio.create_task(
            proactivity_engine.run(), name="orion-proactivity-engine")
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
    thoughts_task = asyncio.create_task(thoughts.run(), name="orion-thought-stream")
    reporting_task = asyncio.create_task(reporting.run(), name="orion-reporting")
    # §5.2: weekly WAL-checkpoint + VACUUM sweep over the config SQLite stores.
    housekeeper = DatabaseHousekeeper(bus)
    housekeeping_task = asyncio.create_task(housekeeper.run(), name="orion-housekeeping")
    # Separate from the weekly VACUUM sweep: this folds each store's
    # write-ahead log back into its database every few minutes. config/ lives
    # in a synced OneDrive folder, which treats .db and .db-wal as unrelated
    # files — a .db restored without its WAL is ORION's memory as it stood at
    # the last checkpoint, with everything since silently gone.
    wal_task = asyncio.create_task(
        housekeeper.run_checkpoints(), name="orion-wal-checkpoint")
    # Follow the device the user is actually using. ORION resolved his
    # microphone and speaker once at startup and held them for the life of the
    # process, so plugging in headphones an hour later left him talking to
    # speakers nobody was listening to. A pinned device still wins — that is a
    # decision, and a decision outranks a default.
    from .audio_follow import AudioFollower
    audio_follower = AudioFollower(bus=bus)
    audio_task = asyncio.create_task(
        audio_follower.run(), name="orion-audio-follow")
    # Capture a workspace baseline so proactive change-tracking has a reference.
    background.spawn(workspace.snapshot_workspace(), name="orion-workspace-baseline")
    # ADA-style autonomous self-improvement pulse (ORION_SELF_IMPROVE=0 disables).
    improvement_task = asyncio.create_task(improvement.run(), name="orion-self-improve")
    # The Resource Governor: shed optional work when memory runs short, so a
    # 16 GB machine at 98% degrades the extras instead of flashing the face
    # black (see resource_governor). Voice, face and the task are never shed.
    from . import resource_governor as _rg
    governor = _rg.ResourceGovernor(bus)
    _rg.GOVERNOR = governor
    governor.register("the idle speech-to-text model", _rg.Pressure.CONSTRAINED,
                      lambda: offline_stt.release_if_idle(limit_s=60.0))
    def _park_mcp_sooner():
        # Idle MCP servers normally pause after half an hour; short of memory,
        # after two minutes, starting now. Each restarts on its next call.
        mcp_host.set_pressure_parking(120.0)
        return mcp_host.park_idle()

    governor.register("idle MCP servers", _rg.Pressure.CONSTRAINED, _park_mcp_sooner,
                      lambda: mcp_host.set_pressure_parking(None))
    governor.register("self-review", _rg.Pressure.PRESSURED, lambda: None)
    governor.register("the Command Deck's 3-D pages while it is hidden",
                      _rg.Pressure.PRESSURED,
                      lambda: getattr(deck, "_release_heavy_pages", lambda: None)())
    governor.register("Python's spare memory", _rg.Pressure.CRITICAL,
                      _rg.release_python_memory)
    background.spawn(governor.run(), name="orion-resource-governor")
    # Periodic self-diagnostic so the Diagnostics Centre panel has fresh data
    # without the user asking for it (run_full() already broadcasts over the
    # bus; this just puts it on a schedule).
    diagnostics_task = asyncio.create_task(diagnostics.run(), name="orion-diagnostics")
    # Connect MCP servers in the background so a slow/absent server never delays
    # startup (the tools appear once each server completes its handshake).
    # Mark XXI, Track E1: once connected, each server's tools are registered
    # as first-class "mcp__<server>__<tool>" entries — reachable by name,
    # not only through the generic `mcp` list/call indirection.
    async def _connect_mcp_and_register() -> None:
        await mcp_host.connect_all()
        for server_name, conn in mcp_host.servers.items():
            count = dispatcher.register_mcp_tools(server_name, conn.tools)
            if count:
                bus.log.emit(
                    f"MCP: {count} tool(s) from '{server_name}' registered as "
                    f"first-class tools (mcp__{server_name}__*).")
        # Research searches through the DuckDuckGo MCP server too, as one
        # more engine between Google and the keyless scrapers.
        if "duckduckgo" in mcp_host.servers:
            from . import web_search_backends

            async def _mcp_search(query: str, limit: int) -> str:
                return await mcp_host.call("duckduckgo", "search",
                                           {"query": query, "max_results": int(limit)})
            web_search_backends.attach_mcp_search(_mcp_search)
        # Mark XXI, Track E2: a crashed/dropped server no longer stays gone
        # until the next full restart — this watches for that and
        # reconnects with bounded backoff, re-registering its tools
        # (Track E1) the moment it's back.
        background.spawn(
            mcp_host.supervise(on_reconnect=dispatcher.register_mcp_tools),
            name="orion-mcp-supervise")
    mcp_host._startup_task = background.spawn(  # type: ignore[attr-defined]
        _connect_mcp_and_register(), name="orion-mcp-connect")

    # BOOT: seed the knowledge corpora OFF the startup critical path.  These are
    # idempotent (marker-guarded) and were previously seeded synchronously,
    # blocking the first paint on first launch.  They now populate the KNOWLEDGE
    # tier a moment after ORION is already interactive, each on a worker thread
    # so SQLite writes never touch the event loop.
    async def _seed_corpora() -> None:
        for label, seed_fn in (
            ("neuroscience", lambda: knowledge.seed(memory)),
            ("knowledge packs", packs.seed_builtin),
            ("programming", lambda: programming.seed(memory)),
            ("cybersecurity", lambda: cyber.seed(memory)),
        ):
            try:
                count = await asyncio.to_thread(seed_fn)
                if count:
                    bus.log.emit(f"KNOWLEDGE: {label} corpus seeded ({count} entries).")
            except Exception as exc:
                bus.log.emit(f"KNOWLEDGE: {label} corpus seed deferred-fail - {exc}")
    background.spawn(_seed_corpora(), name="orion-seed-corpora")

    budget.mark("services + background tasks")
    bus.log.emit(
        f"BOOT: all services constructed and background tasks launched at "
        f"{time.perf_counter() - _boot_t0:.2f}s.")
    bus.log.emit(budget.report())   # §5.1 per-phase startup budget
    telemetry.metrics.gauge("startup.total_seconds", budget.total)
    if budget.over_budget():
        telemetry.metrics.incr("startup.over_budget")
        bus.log.emit(f"BOOT: WARN - {budget.breach_summary()}")

    try:
        await shutdown_event.wait()
    finally:
        # Every step below is best-effort: a restart the user explicitly
        # asked for must never silently fail to come back just because one
        # subsystem's teardown raised. Before this, most of these calls had
        # no individual error handling, so a single failing .stop()/.close()
        # anywhere in this ~30-step sequence would propagate out of this
        # `finally` block, skip `return restart_requested` entirely, and the
        # process would simply exit with no respawn — "restart" that quietly
        # never happens. Each step is now isolated so the rest still run and
        # the function always reaches its return.
        from .shutdown_trace import TRACE as _trace

        def _safe(label: str, fn) -> None:
            with _trace.phase(label):
                try:
                    fn()
                except Exception as exc:
                    bus.log.emit(f"SYS: {label} shutdown reported - {exc}")

        async def _safe_async(label: str, coro) -> None:
            with _trace.phase(label):
                try:
                    await coro
                except Exception as exc:
                    bus.log.emit(f"SYS: {label} shutdown reported - {exc}")

        bus.state.emit("SHUTTING DOWN")
        # BEFORE the farewell. The farewell wait used to PUMP QT, and under
        # qasync pumping Qt from inside run_application() dispatches pending
        # asyncio task-steps re-entrantly. Any repair task still queued at
        # that moment started its first step while run_application() was the
        # current task, and asyncio refused:
        #
        #   RuntimeError: Cannot enter into task <attempt_auto_repair()> while
        #   another task <run_application()> is being executed
        #
        # which the loop exception handler then CAPTURED AS A NEW INCIDENT,
        # which scheduled another repair — the "self_repair degraded" spiral
        # (inc-29 … inc-33), all manufactured by the shutdown path itself. The
        # wait awaits now and no longer pumps, but a repair must still not
        # START on the way out, so self-repair is disarmed first regardless.
        _safe("self-repair", selfrepair.shutdown)
        _safe("background tasks", background.cancel_all)
        # Cancelling only SCHEDULES cancellation. Awaiting the cancelled tasks
        # is what consumes their CancelledError and stops Python printing
        # "Task was destroyed but it is pending!" once per task at exit.
        await _safe_async("self-repair drain", selfrepair.drain(timeout=3.0))
        await _safe_async("background drain", background.drain(timeout=2.0))
        # Mark XXI: a spoken farewell on EVERY shutdown path, not only the
        # voice-initiated one (live_worker.py already speaks its own before
        # this ever runs, and sets _farewell_spoken so this doesn't repeat
        # it). Restarting gets its own "I'll be right back" message instead
        # of a goodbye. Must run before worker.stop() (further below) tears
        # down self.speech — and is itself bounded so a wedged audio device
        # can never block shutdown.
        if not restart_requested and not getattr(worker, "_farewell_spoken", False):
            try:
                worker.speech.speak_text(worker.compose_farewell())
            except Exception as exc:
                bus.log.emit(f"SYS: farewell speech failed - {exc}")
            else:
                # Wait for him to finish the sentence, not a fixed guess.
                with _trace.phase("farewell (speaking the goodbye)"):
                    await _await_farewell(worker.speech)
        # He has said goodbye (or is restarting): off the screen now. Nothing
        # below needs a window, and nothing ever closed them, so the last
        # frame sat there frozen until the interpreter finished exiting.
        _safe("off screen", lambda: _take_off_screen(window))
        _safe("proactive", proactive.stop)
        _safe("connectivity", connectivity.stop)
        _safe("reminders", reminders.stop)
        _safe("sentinel", sentinel.stop)
        _safe("presence", presence.stop)
        _safe("audio studio", audio_studio.stop)
        _safe("fact harvester", fact_harvester.stop)
        _safe("pipeline", pipeline.stop)
        _safe("security", security.stop)
        _safe("cognitive loop", cognitive_loop.stop)
        _safe("reporting", reporting.stop)
        _safe("housekeeper", housekeeper.stop)
        # One last fold before the process goes, so a clean exit always leaves
        # a database that is complete on its own.
        _safe("wal checkpoint", housekeeper.checkpoint_all)
        _safe("improvement", improvement.stop)
        _safe("cursor overlay", cursor_overlay.stop)
        await _safe_async("electronics inspection", electronics_controller.shutdown())
        _safe("electronics camera", electronics_workbench.shutdown)
        await _safe_async("perception", perception.stop())   # release the frame loop before the camera
        if face_tracker is not None:
            _safe("face tracker", face_tracker.stop)
        _safe("MCP supervisor", mcp_host.stop_supervising)
        await _safe_async("MCP host", mcp_host.close())
        await _safe_async("browser co-pilot", web_copilot.close())          # #11 — release any driven browser
        await _safe_async("social automation", social.close())              # release any driven social-automation browser
        await _safe_async("chess engine", asyncio.to_thread(chess_service.shutdown))   # quit the Stockfish subprocess
        for task in (telemetry_task, diagnostics_task, briefing_task, proactive_task,
                     connectivity_task, reminders_task, sentinel_task, presence_task,
                     audio_studio_task, pipeline_task, security_task,
                     cognitive_loop_task, thoughts_task, reporting_task, housekeeping_task,
                     improvement_task):
            _safe(f"cancel {getattr(task, 'get_name', lambda: 'task')()}", task.cancel)
        _safe("thought stream", thoughts.stop)
        await _safe_async("worker", worker.stop())
        # The dispatcher's reference is current if phone access was turned
        # on or off by voice; stop() is idempotent, so stopping both is safe.
        for live_gateway in {id(g): g for g in (gateway, getattr(dispatcher, "gateway", None))
                             if g is not None}.values():
            await _safe_async("remote gateway", live_gateway.stop())
        worker_task.cancel()
        wal_task.cancel()
        audio_task.cancel()
        _safe("audio follower", audio_follower.stop)
        for task in (wal_task, audio_task, telemetry_task, briefing_task, worker_task, proactive_task,
                     connectivity_task, reminders_task, sentinel_task, presence_task,
                     audio_studio_task, pipeline_task, security_task,
                     cognitive_loop_task, thoughts_task, reporting_task, improvement_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                bus.log.emit(f"SYS: shutdown task reported - {exc}")
        # The bulk executor holds non-daemon worker threads. Nothing shut it
        # down, so concurrent.futures' atexit hook was left to join them — which
        # means a stuck bulk job (a large ingest, OCR over a folder) would hang
        # the whole shutdown. It has always had a shutdown function; it was
        # simply never wired in. wait=False: shutdown must not block on work.
        from .executors import shutdown_bulk_executor as _shutdown_bulk
        _safe("bulk executor", lambda: _shutdown_bulk(wait=False))
        _safe("companion session", companion.end_session)
        _safe("companion", companion.close)
        _safe("evidence store", evidence.close)
        _safe("knowledge graph", graph.close)
        _safe("geo cache", geo.close)
        if getattr(commerce, "signals", None) is not None:
            _safe("commerce signals", commerce.signals.close)
        _safe("memory", memory.close)
        _trace.write(CONFIG_DIR / "diagnostics" / "last_shutdown.txt")
    return restart_requested


def _report_exit_fault(exc: BaseException) -> None:
    """A genuine crash: one readable line on the console, the detail on disk.

    The full traceback used to be dumped to CMD, where it scrolls past and is
    gone. Writing it to config/reports/console/ means it survives, is
    timestamped, and can be read after the fact — while the console says the
    one thing worth reading at that moment: something broke, and where to look.
    """
    from . import console_hygiene
    path = console_hygiene.report_fault(exc, context="shutdown")
    print(f"ORION stopped after a fault: {type(exc).__name__}: {exc}")
    if path is not None:
        print(f"  full report: {path}")
    else:
        traceback.print_exc()      # nowhere to write it — do not lose it


def _quiet_report() -> None:
    """Say how much was kept off the console, so 'quiet' is never mistaken for
    'nothing happened'."""
    try:
        from . import console_hygiene
        routed = console_hygiene.routed_count()
        if routed:
            print(f"  ({routed} routine third-party notices logged to "
                  f"{console_hygiene.REPORTS_DIR})")
    except Exception:
        pass


def _note_lingering_threads() -> None:
    """Add to the shutdown report whatever the process still has to wait for.

    Only rewritten when there is something to say, so a clean report stays
    exactly what teardown wrote. Never raises."""
    try:
        from .shutdown_trace import TRACE, lingering_threads
        if TRACE.started is None:
            return
        TRACE.lingering = lingering_threads()
        if TRACE.lingering:
            TRACE.write(CONFIG_DIR / "diagnostics" / "last_shutdown.txt")
    except Exception:
        pass


def _run_until_torn_down(loop: Any, coro: Any, *, requested: Any,
                         max_resumes: int = 3) -> Any:
    """Run *coro* to completion, resuming the loop if Qt quits under it.

    Every way of closing ORION ends in QApplication.quit() — Ctrl+C, closing
    the last window, the tray — and until 2026-09-23 the Quit button did too,
    because core_window wired bus.request_shutdown straight to it. quit()
    stops the qasync loop, so run_until_complete() returned while
    run_application() was only a few steps into its teardown, and the rest
    never ran: no services stopped, no databases closed, no shutdown report,
    and every task still pending was destroyed at exit. The process printed
    "ORION has shutdown cleanly." regardless. Measured twice on the same day:
    one quit ended 5.2 s in with the teardown abandoned mid-farewell; another
    stopped the loop inside worker.stop() and sat there until the 25 s
    watchdog killed it.

    So when the loop stops on a requested shutdown with teardown unfinished,
    the loop is simply entered again (qasync supports repeated
    run_until_complete calls) and teardown carries on from where it was
    suspended. Anything else that stops the loop early is still raised, and
    the watchdog armed by request_shutdown still bounds the whole thing.
    """
    task = asyncio.ensure_future(coro, loop=loop)
    resumes = 0
    while True:
        try:
            return loop.run_until_complete(task)
        except RuntimeError:
            if task.done() or not requested() or resumes >= max_resumes:
                raise
            resumes += 1


#: V8 heap ceiling for ORION's own web pages (face, brain, globe), in MB.
WEB_HEAP_CAP_MB = 160


def _cap_web_heaps() -> None:
    """Bound how much garbage each WebEngine page may hoard before collecting.

    MEASURED on the running pages: live JavaScript after a full collection is
    8 MB (face), 4 MB (brain) and 13 MB (globe) — yet before one they held
    293, 152 and 61 MB, because V8 sizes its heap to the machine and on 16 GB
    lets a page grow to ~half a gigabyte before a major GC. The face renderer
    plateaued at 545 MB of which ~500 was garbage: RAM a 16 GB machine running
    at 98% did not have, which is when Chromium discards surfaces and flashes
    black. Capped at 128 the same renderer held flat at 110 MB. 160 leaves
    twelve times the heaviest page's live heap.

    Must run before the web engine starts (it reads the variable once). A
    --js-flags the user set themselves is left alone; ORION_WEB_HEAP_MB=0
    turns the cap off.
    """
    flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "--js-flags" in flags:
        return
    try:
        cap = int(os.getenv("ORION_WEB_HEAP_MB", str(WEB_HEAP_CAP_MB)))
    except ValueError:
        cap = WEB_HEAP_CAP_MB
    if cap <= 0:
        return
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        f"{flags} --js-flags=--max-old-space-size={max(64, cap)}").strip()


def main(*, started_at: float | None = None) -> None:
    _cap_web_heaps()
    # Before anything can log: keeps known-benign third-party chatter off the
    # console and in config/reports/console/ instead. Nothing is discarded.
    from . import console_hygiene
    console_hygiene.install()
    # Before anything can spawn a child: stop pip, Stockfish, ffmpeg and the
    # rest from flashing up their own console windows. One hook on Popen covers
    # ORION's own spawns and its libraries' alike.
    from . import quiet_subprocess
    quiet_subprocess.install()
    # Give ORION his own taskbar identity BEFORE any window exists, so Windows
    # shows him — his icon, his group, his pinned shortcut merged with his
    # running window — instead of a generic "Python" process.
    from . import desktop_app
    desktop_app.set_app_user_model_id()
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
    # BEFORE QApplication, and this order is not incidental. QWebEngine
    # decides whether it can get a GPU compositor at import time; imported
    # after the application exists it decides it cannot, sets
    # QuantumFace3D.available = False, and does so silently. Measured both
    # ways: True when imported first, False when imported second.
    #
    # That single line of ordering is why ORION's real Three.js face — the
    # quantum orb that materialises into him — was never once selected, on
    # any machine, for the whole life of the sculpted head.
    try:
        from .gui.face3d import QuantumFace3D  # noqa: F401
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    # His face in the taskbar, Alt-Tab and title bars.
    desktop_app.apply_window_icon(app)

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

    # Closing ORION is not a crash.
    #
    # app.quit() stops the Qt event loop while run_application() is still
    # awaiting, and qasync's run_until_complete then raises
    #
    #     RuntimeError: Event loop stopped before Future completed.
    #
    # _run_until_torn_down() now resumes the loop so teardown still finishes,
    # and only lets that error out if the loop keeps being stopped under it.
    # When it does, that is still a deliberate shutdown under qasync, not a
    # fault — but it was once being caught by the generic handler below and printed
    # as "terminated after a controlled fault" with a full traceback, every
    # single time ORION was closed or restarted. Alarming, and completely
    # wrong. Intent is tracked explicitly rather than inferred from a message,
    # so a genuine mid-run loop failure is still reported as the fault it is.
    _shutdown = {"requested": False}
    try:
        app.aboutToQuit.connect(lambda: _shutdown.__setitem__("requested", True))
    except Exception:
        pass

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    # Every asyncio.to_thread runs here. qasync's own default (a QThread pool)
    # is joined with no bound when the loop closes, so a job still in flight
    # at quit held the process long after teardown — see DaemonThreadExecutor.
    from .executors import default_executor as _default_executor
    loop.set_default_executor(_default_executor())
    restart_requested = False
    try:
        with loop:
            restart_requested = bool(_run_until_torn_down(loop, run_application(
                app, started_at=_APP_IMPORT_STARTED_AT if started_at is None else started_at),
                requested=lambda: _shutdown["requested"]))
    except KeyboardInterrupt:
        print("O.R.I.O.N. shutdown requested from console.")
    except SystemExit:
        raise
    except RuntimeError as exc:
        if is_clean_shutdown(exc, requested=_shutdown["requested"]):
            print("ORION has shutdown cleanly.")
        else:
            _report_exit_fault(exc)
    except Exception as exc:
        _report_exit_fault(exc)
    else:
        print("ORION has shutdown cleanly.")
    finally:
        _quiet_report()
        try:
            from . import console_hygiene
            console_hygiene.close()
        except Exception:
            pass
        try:
            _release_portaudio_atexit()
        except Exception:
            pass
        _note_lingering_threads()
    if restart_requested:
        # Self-restart (Mark X.6): respawn the same interpreter with the same
        # arguments once the loop has fully closed.  subprocess.Popen is used
        # rather than os.execv because Windows execv mangles arguments that
        # contain spaces (this project lives under 'PROJECT ORION').
        import subprocess

        # Release the single-instance lock BEFORE spawning, or the successor
        # claims it, finds it held by a process that is still exiting, and
        # refuses to start — a restart that quietly becomes a shutdown.
        # NOT `import orion`. orion.py is the entry script, so it is __main__
        # and importing it by name builds a SECOND copy of the module, which
        # re-runs the guard, finds the mutex held by this very process, and
        # raises SystemExit — which is a BaseException and sailed straight
        # through the `except Exception` below, ending the process before the
        # respawn below could run. The restart became a shutdown, silently.
        try:
            from .single_instance import release

            release()
        except BaseException:           # noqa: BLE001 - see above
            pass

        # argv[1:], not argv. Frozen, sys.executable IS ORION.exe and argv[0]
        # is ORION.exe too, so the old form launched him with his own path as
        # a positional argument.
        arguments = sys.argv[1:] if getattr(sys, "frozen", False) else sys.argv
        print("O.R.I.O.N. restarting…")
        subprocess.Popen([sys.executable] + arguments)


if __name__ == "__main__":
    main()
