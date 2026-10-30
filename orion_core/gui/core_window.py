"""
ORION Core Window — face-only shell.  Every other surface (conversation log,
memory matrix, telemetry, inner-voice thought stream, calendar/geo panel)
lives on the Command Deck; this window shows nothing but ORION's face.

Also implements the immersive display modes:

    FULLSCREEN (F11)         — borderless full-monitor operating-system feel.
    OVERLAY    (Ctrl+Shift+O) — compact frameless always-on-top orb that
                                floats over the desktop while ORION keeps
                                listening; drag to reposition, Esc to exit.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
import weakref
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional

import psutil
from PyQt6 import sip
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import APP_BUILD, APP_NAME, APP_SUBTITLE, SPACE
from ..memory import MemoryAgent
from ..utils import utc_stamp
from .face import HologramFace
from .style import APP_STYLESHEET
from .. import background


def native_face_enabled() -> bool:
    """Whether ORION's face is drawn by the native renderer (Mark XXIII).

    The default. His avatar is point-cloud geometry inside the swarm scene —
    depth-tested against the network, occluded by particles passing in front,
    and the literal origin every connection radiates from. That is impossible
    for a QWebEngineView layered over the surface, which is what this
    replaces.

    ORION_NATIVE_FACE=0 falls back to the 2-D face."""
    return os.getenv("ORION_NATIVE_FACE", "").strip().lower() not in {
        "0", "false", "off", "no"}


def holo_head_enabled() -> bool:
    """Whether ORION wears the software-rendered human head (head_mesh + QPainter).

    On by default, and first in the order, because it is the only face here
    with no precondition. The WebEngine face needs a compositor that cannot
    coexist with a QOpenGLWidget in the same window; the native face needs a GL
    context that app.py does not create (it never calls setDefaultFormat, so
    passes can silently draw nothing). This one needs numpy and a QPainter,
    both of which are already required, so it renders identically on a gaming
    rig, a 2013 laptop, a VM and a remote desktop session.

    OFF by default now, and FIRST when explicitly on.

    It renders anywhere, which is why it was once the default — but what it
    renders is a head, and a sculpted crimson bust staring out of the HUD is
    not what ORION looks like. His own face is the Three.js one: a quantum orb
    that holds the screen while he comes up and then materialises into him.
    This was standing in front of it.

    Kept as the escape hatch for a machine where WebEngine will not start, and
    checked BEFORE the WebGL face because an escape hatch that only applies
    once the thing you are escaping has already failed is not an escape hatch.
    It is also how the test suite picks a face it can actually drive.

    ORION_HOLO_HEAD=1 to force it."""
    return os.getenv("ORION_HOLO_HEAD", "").strip().lower() in {
        "1", "true", "on", "yes"}


def webengine_face_possible() -> bool:
    """Whether ORION's Three.js avatar can render in the Core Window.

    Measured, not assumed, and the answer turns entirely on how many windows
    ORION is running as:

      * SHARED window — impossible. Once a QOpenGLWidget has existed in a
        window the WebEngine compositor comes up with zero-size framebuffer
        attachments and every WebGL draw fails with
        GL_INVALID_FRAMEBUFFER_OPERATION. Hiding the GL widget does not help;
        neither does deleting it. The page never recovers.
      * SEPARATE top-level windows — fine. Different native surfaces, no
        conflict: the crimson avatar renders in one window while the swarm
        renders in the other.

    So the unified shell forces the native point-cloud avatar, and the
    two-window layout — the default — gets ORION's real face back.
    """
    if "pytest" in sys.modules:
        # Never inside the test suite. A QWebEngineView starts a Chromium
        # compositor in this process, and once one has existed the later Qt
        # tests wedge — the run stops dead partway through rather than
        # failing, which is the worst shape of breakage because it reads as a
        # slow machine. It could not be verified here in any case: WebEngine
        # will not grab to a pixmap without a real GPU compositor.
        #
        # What the APPLICATION prefers is asserted by reading this source
        # instead, in tests/test_native_renderer_migration.py — a claim about
        # the code rather than a Chromium instance nobody can look at.
        return False
    return not unified_shell_enabled()


def unified_shell_enabled() -> bool:
    """Whether ORION runs as ONE window (Mark XXII, Phase 5).

    OFF by default again. Merging them was technically sound and looked
    cluttered: a face, a swarm rail and a twenty-page deck competing in one
    frame, with the deck's SWARM page and the rail showing the same graph
    twice. Two windows give each surface a monitor and a job — ORION's face on
    the primary, the Command Deck on the second — which is both calmer and
    what the layout was designed around.

    ORION_UNIFIED_SHELL=1 re-enables the single-window shell.

    Deliberately defined HERE rather than in app.py, beside the window it
    configures. app.py is the composition root and pulls in essentially the
    whole system, and merely importing it puts the process in a state where
    recreating a native window that owns a QWebEngineView — which is what
    overlay mode's setWindowFlags does — dies with STATUS_STACK_BUFFER_OVERRUN
    under the offscreen QPA platform. A test that only wanted to read this
    flag was enough to crash unrelated window tests, so the flag lives
    somewhere cheap to import."""
    return os.getenv("ORION_UNIFIED_SHELL", "").strip().lower() in {
        "1", "true", "on", "yes"}


#: How much room the strip of results under the face may take. Bounded so a
#: long list cannot squeeze the face down to nothing; it scrolls instead.
CONTENT_STRIP_HEIGHT = 190


def _new_webgl_face() -> Any:
    """The Three.js face: in its own process (face_host.RemoteFace) unless
    ORION_FACE_PROCESS=0, or after the host failed this run - then in ORION's.

    MEASURED, 10 Oct: with ORION's GUI thread busy 40% of the time the
    in-process page drew 36 fps with 200 ms freezes; in its own process,
    60 fps with no gap over 17 ms."""
    from .face3d import QuantumFace3D, local_three, _register_scheme
    if local_three() is None or not _register_scheme():
        raise RuntimeError("the local avatar assets cannot be served")
    from . import face_host
    if face_host.enabled():
        try:
            return face_host.RemoteFace()
        except Exception as exc:
            print(f"[FACE] face process unavailable ({exc}) - drawing in-process.", flush=True)
    return QuantumFace3D()


def _wear_orion_colours(face: Any) -> Any:
    """Give a freshly built face ORION's crimson.

    Every renderer shipped with its own default — the sculpted head's is a
    cyan, #39b6ff — and that default was only ever replaced if the user
    opened Preferences and picked an accent. So ORION's identity was
    crimson everywhere except the one thing you actually look at.

    This changes the colour and nothing else. The geometry, the rig, the
    expressions and the orb are exactly as they were.
    """
    if face is None or not hasattr(face, "set_palette_colours"):
        return face
    # Imported here because this module keeps its colours local to the
    # functions that use them; a module-level `C` does not exist.
    from .style import C

    try:
        face.set_palette_colours(C.PRI, C.ACCENT)
    except Exception as exc:
        # Reported, not swallowed. A bare `except: pass` here hid this
        # failing entirely — the face stayed the renderer's default cyan
        # and nothing said why. Printed rather than put on the bus because
        # this is deliberately a module-level function: _build_face is
        # driven with a stand-in `self` by the renderer-migration tests, so
        # anything reached through self breaks them.
        from ..startup_report import say

        say(f"[ORION] FACE: could not apply ORION's colours - {exc}")
    return face


#: ORION's state as the header says it (Mark XXXV): words, not a status code.
_STATE_WORDS = {
    "INITIALISING": "Waking up…", "LISTENING": "I'm listening", "SPEAKING": "Speaking",
    "THINKING": "Thinking…", "PROCESSING": "Working on it…", "STANDBY": "Resting",
    "IDLE": "Here when you need me", "PAUSED": "Paused", "OFFLINE": "Offline",
    "RECONNECTING": "Reconnecting…", "MUTED": "Muted", "FOCUS": "Keeping quiet",
    "ERROR": "Something's wrong",
}


def _state_words(state: str) -> str:
    key = str(state or "").upper()
    return _STATE_WORDS.get(key) or key.replace("_", " ").capitalize()


def _clock_words(now: datetime) -> str:
    """'Tue 7 Oct  ·  21:52' - the date and the time, without seconds ticking."""
    return f"{now.strftime('%a')} {now.day} {now.strftime('%b')}  ·  {now.strftime('%H:%M')}"


def _greeting_words(now: datetime, name: str = "") -> str:
    hour = now.hour
    if 5 <= hour < 12:
        words = "Good morning"
    elif 12 <= hour < 18:
        words = "Good afternoon"
    else:
        words = "Good evening"
    first = name.split()[0] if name.strip() else ""
    return f"{words}, {first}" if first else words


def _share_with_live(worker: Any, note: str) -> bool:
    """Put a dropped file's content into Live's context the way other
    background notes go (reward_learning._send_hint): queued, sent while the
    channel is idle and never mid-sentence, and re-sent on a reconnect.

    The review calls this before it returns, so the note is queued ahead of
    the spoken acknowledgement and cannot cut it off on the channel."""
    notes = getattr(worker, "_pending_context_notes", None)
    flush = getattr(worker, "_flush_context_notes", None)
    if notes is None or not callable(flush):
        return False
    notes.append(note)
    try:
        background.spawn(flush(), name="orion-dropped-file-context")
    except RuntimeError:
        pass            # no running loop: the next flush (or reconnect) sends it
    return True


def _note_dropped_file(worker: Any, path: str, result: Any) -> None:
    """Record a dropped file in the activity ledger, as a tool call would be,
    so the continuity brief of a later Live session still knows about it -
    a context note dies with the session that received it."""
    ledger = getattr(worker, "_activity_ledger", None)
    payload = getattr(result, "response_payload", None)
    if ledger is None or not callable(payload):
        return
    try:
        activity = ledger.started("process_file", {"action": "review", "path": str(path)})
        ledger.finished(activity, payload())
    except Exception:
        pass            # the ledger is a courtesy; the review itself is done


class OrionCoreWindow(QMainWindow):
    def __init__(self, bus: OrionBus, memory: MemoryAgent) -> None:
        super().__init__()
        self.bus          = bus
        self.memory       = memory
        self.worker: Any  = None
        self.dashboard: QMainWindow | None = None
        self.command_centre: QMainWindow | None = None
        # The Environment panel (calendar/geo/weather) now lives on the OPS
        # deck page, not on this window — attached late so self-navigation
        # ("Orion, refresh the environment") still has somewhere to route.
        self.environment_panel: Any = None
        # The LOG view's input line also moved to the deck; attached late so
        # the Ctrl+K command palette can still prefill a chosen tool name.
        self.log_view: Any = None
        self.active_state = "INITIALISING"
        self.telemetry: dict[str, Any] = {
            "cpu":          0.0,
            "ram":          0.0,
            "net_bps":      0.0,
            "net_percent":  0.0,
            "state":        self.active_state,
            "mic_active":   True,
            "updated_at":   utc_stamp(),
        }
        # Overlay-mode bookkeeping
        self._overlay_active = False
        self._saved_geometry: Any = None
        self._orb_overlay: Any = None
        self._orb_gestures: Any = None
        self._orb_camera_tracker: Any = None
        self._orb_hands_enabled = False
        self._orb_started_tracker = False
        self._orb_hand_session: int | None = None

        self.setWindowTitle(APP_NAME)
        # Face-only content can comfortably go much smaller than the old
        # splitter+sidebar shell could — kept low enough that docking to the
        # left half of a single 1920-wide monitor (960px) never gets
        # overridden back up to overlap the Command Deck's right half.
        self.setMinimumSize(420, 480)
        self.resize(1360, 820)
        self.setStyleSheet(APP_STYLESHEET)
        self._build_ui()
        self._connect_bus()
        # Depth, after the tree exists. Qt Style Sheets cannot cast a shadow,
        # so elevation is applied to the widgets themselves — once, by walking
        # the tree, rather than at each of the twenty-six places that build a
        # panel and could forget.
        from . import depth

        depth.lift_panels(self)
        # The initial host can fail just like a rebuilt one. Without this
        # watcher its empty container survived the startup timeout forever.
        self._watch_face_health()

    def showEvent(self, event: Any) -> None:
        """Ask the compositor to blur the desktop behind the window.

        This is the one part of the look Qt genuinely cannot fake, and it
        needs a real native window handle — which does not exist until the
        window is shown.
        """
        super().showEvent(event)
        try:
            from . import depth

            depth.apply_backdrop(self)
        except Exception:
            pass

    # ── attachments ───────────────────────────────────────────────────────────

    def attach_worker(self, worker: Any) -> None:
        self.worker = worker

    def attach_orb_controls(self, gestures: Any, camera_tracker: Any = None) -> None:
        """Connect the opt-in two-hand camera mode to the existing orb view."""
        self._orb_gestures = gestures
        self._orb_camera_tracker = camera_tracker
        action = getattr(self, "orb_hands_action", None)
        if action is not None:
            action.setEnabled(gestures is not None)

    def _show_orb_controls(self) -> None:
        QMessageBox.information(
            self, "Orb Controls",
            "Drag to move the orb. Right-drag to rotate it.\n"
            "Scroll or Shift-drag to scatter its particles.\n"
            "Double-click to reset its position and shape.\n\n"
            "Camera hand control is available in the Appearance menu.",
        )

    def set_orb_hand_control(self, enabled: bool) -> bool:
        action = getattr(self, "orb_hands_action", None)
        gestures = getattr(self, "_orb_gestures", None)
        if enabled and gestures is not None:
            self.set_face_form("orb")
            face = getattr(self, "face", None)
            if (not hasattr(face, "set_orb_hands") or
                    not getattr(face, "supports_orb_hands", True)):
                from ..appearance import SCREEN_ONLY_REPLY, form_enabled
                self.bus.log.emit(
                    "GESTURE: camera hand control needs the Three.js orb."
                    if form_enabled("orb") else
                    f"GESTURE: camera hand control belongs to the orb. {SCREEN_ONLY_REPLY}")
                enabled = False
            else:
                tracker = self._orb_camera_tracker
                if tracker is not None and not getattr(tracker, "running", False):
                    result = tracker.start()
                    if not getattr(result, "ok", True):
                        self.bus.log.emit(f"GESTURE: camera unavailable - {result.text}")
                        enabled = False
                    else:
                        self._orb_started_tracker = True
                if enabled:
                    result = gestures.start_orb()
                    if not getattr(result, "ok", True):
                        self.bus.log.emit(f"GESTURE: {result.text}")
                        enabled = False
                    else:
                        self._orb_hand_session = getattr(gestures, "orb_session", None)
        elif enabled:
            self.bus.log.emit("GESTURE: install mediapipe to use camera hand control.")
            enabled = False
        if not enabled:
            if self._orb_hands_enabled and gestures is not None:
                gestures.stop_orb()
            if self._orb_started_tracker and self._orb_camera_tracker is not None:
                # Off the loop: stop() joins the capture thread for up to 2 s
                # while a camera driver finishes read() (Mark XXXV W1.1 audit).
                # A start() meanwhile answers "the camera is still stopping".
                import threading
                threading.Thread(target=self._orb_camera_tracker.stop,
                                 name="orb-camera-stop", daemon=True).start()
            self._orb_started_tracker = False
        self._orb_hands_enabled = enabled
        self._to_face("set_orb_camera_enabled", enabled)
        if action is not None:
            action.setChecked(enabled)
        return enabled

    def _on_orb_gesture(self, sample: Any) -> None:
        if isinstance(sample, dict) and sample.get("stopped"):
            if (sample.get("session") is not None
                    and sample["session"] != self._orb_hand_session):
                return
            if self._orb_hands_enabled:
                self.set_orb_hand_control(False)
            return
        if self._orb_hands_enabled:
            self._to_face("set_orb_hands", sample)

    def attach_dashboard(self, dashboard: QMainWindow) -> None:
        self.dashboard = dashboard
        import weakref
        dashboard._core_window_ref = weakref.ref(self)

    def attach_command_centre(self, centre: QMainWindow) -> None:
        self.command_centre = centre

    def attach_environment_panel(self, panel: Any) -> None:
        self.environment_panel = panel

    def attach_log_view(self, view: Any) -> None:
        self.log_view = view

    def attach_deck_inline(self, deck: Any) -> None:
        """Dock the Command Deck INSIDE this window (Mark XXII, Phase 5).

        The unified shell: ORION's face and the Command Deck stop being two
        top-level windows and become one, sharing a splitter. They already
        shared a process, an event loop and an AI session — what they did not
        share was a window, which is what made them feel like two unrelated
        programs rather than one system.

        The deck's CENTRAL WIDGET is reparented — deliberately not the deck
        object itself. UnifiedDashboard is a QMainWindow, and nesting one
        QMainWindow inside another is a Qt anti-pattern that does not merely
        look wrong: it crashed this window natively (STATUS_STACK_BUFFER_OVERRUN)
        the first time overlay mode called setWindowFlags on it, because
        changing flags forces the native window to be recreated and the nested
        main-window layout cannot survive that.

        Taking the central widget avoids the nesting entirely while keeping
        everything that matters. Every page, timer, signal and piece of state
        lives INSIDE that widget, so the move costs nothing; the deck object
        stays alive (referenced here) as the controller, and its
        show_page_named/page_names still drive the very same QStackedWidget,
        which is now a descendant of this window. Rebuilding the deck instead
        would mean a second construction path that could drift from app.py's.
        """
        self.deck_inline = deck
        layout = self.deck_panel.layout()
        if self._deck_placeholder is not None:
            layout.removeWidget(self._deck_placeholder)
            self._deck_placeholder.deleteLater()
            self._deck_placeholder = None

        body = deck.centralWidget() if hasattr(deck, "centralWidget") else None
        if body is None:
            # Not a QMainWindow (a plain widget, or a stub in a test): embed
            # it directly, which is safe precisely because there is no nesting.
            body = deck
        else:
            # takeCentralWidget hands over ownership cleanly, leaving the deck
            # shell empty rather than half-owning a widget we then reparent.
            taken = deck.takeCentralWidget()
            body = taken if taken is not None else body
        body.setParent(self.deck_panel)
        layout.addWidget(body)
        body.show()
        self.deck_panel.show()
        # The deck owns the working area, but the face keeps a real, visible
        # share — set as explicit pixel sizes rather than stretch factors
        # alone, because stretch only governs how SPARE space is distributed
        # and left the face at whatever tiny width it started with.
        self.content_splitter.setStretchFactor(0, 2)
        self.content_splitter.setStretchFactor(2, 5)
        total = max(self.width(), 1200)
        face_width = max(360, int(total * 0.30))
        self.content_splitter.setSizes([face_width, 0, total - face_width])
        self.bus.log.emit("SYS: unified shell — face and Command Deck in one window.")

    @property
    def deck_is_inline(self) -> bool:
        return getattr(self, "deck_inline", None) is not None

    def attach_swarm_view(self, swarm_view: Any) -> None:
        """Docks a SwarmDeckView (Mark XXI, Track A1) into the collapsible
        rail beside the face — an independent instance from the Command
        Deck's own SWARM page (constructed with the same live backends by
        app.py), never a reparent of that page's widget, so the Deck's page
        is untouched. The rail stays hidden until the user opts in
        (toggle_swarm_rail), so SwarmDeckView's own lazy WebEngine build
        (_ensure_built, on real showEvent) still never fires at startup."""
        self.swarm_view = swarm_view
        rail_layout = self.swarm_rail.layout()
        rail_layout.removeWidget(self._swarm_rail_placeholder)
        self._swarm_rail_placeholder.deleteLater()
        rail_layout.addWidget(swarm_view)

    def attach_face_renderer(self, renderer: Any) -> None:
        """Give the face panel the live scene view it draws ORION with.

        The renderer is the SAME kind of view the Command Deck's SWARM page
        uses, framed on his face instead of on the whole field. Late-attached
        because the window is built long before app.py has any backends, and
        creating a GL context during construction would put it on the startup
        path.

        Returns quietly when the face is the 2-D rig — it draws itself and has
        nothing to attach."""
        self.face_renderer = renderer
        if hasattr(self.face, "attach_renderer"):
            self.face.attach_renderer(renderer)
        else:
            # The face draws itself (the Three.js avatar, or the 2-D rig), so
            # the renderer becomes the collapsible rail's swarm rather than
            # being left orphaned. It builds no GL context until shown — and
            # while the Three.js face is live, toggle_swarm_rail refuses to
            # show it, because a GL surface in this window would take that
            # face down permanently.
            self.attach_swarm_view(renderer)

    def attach_avatar(self, avatar: Any) -> None:
        """Hand the face rig over to the AvatarController.

        The controller becomes the single driver of state / amplitude /
        speaking (it adds eased cross-fades, the lip envelope, breathing,
        blinking and the face-tracking head follow), so the direct
        bus→face connections are removed to avoid double-driving.  Emotion
        parameter sets still stream straight to the rig.
        """
        if getattr(self, "avatar", None) is avatar:
            avatar.attach(self.face)
            return
        self.avatar = avatar
        # The window's own forwarders are what the bus is connected to (see
        # _connect_bus); disconnecting self.face.* would silently match
        # nothing and leave the rig driven twice.
        for signal, slot in ((self.bus.amplitude, self._face_set_amplitude),
                             (self.bus.speaking, self._face_set_speaking)):
            try:
                signal.disconnect(slot)
            except Exception:
                pass
        avatar.attach(self.face)
        self.bus.state.connect(avatar.on_state)
        self.bus.amplitude.connect(avatar.on_amplitude)
        self.bus.voice_spectrum.connect(avatar.on_voice_spectrum)
        self.bus.speaking.connect(avatar.on_speaking)
        # Webcam face samples arrive on the bus from the tracker thread
        # (queued connection → GUI thread safe).
        try:
            self.bus.face_tracking.connect(avatar.on_face_sample)
        except Exception:
            pass
        self._avatar_timer = QTimer(self)
        self._avatar_timer.setInterval(50)      # 20 Hz behaviour tick
        # Routed through a wrapper rather than connected to avatar.tick
        # directly, so the tick can be skipped while this window is hidden.
        self._avatar_timer.timeout.connect(self._tick_avatar_if_visible)
        self._avatar_timer.start()

    def _tick_avatar_if_visible(self) -> None:
        """20 Hz avatar behaviour tick, skipped while this window is hidden.

        The tick only advances the state machine's eased transitions and the
        animation channels (both clamp their own dt), so pausing it costs
        nothing but the frames nobody could see — there is no time-based
        emotion decay or other non-visual side effect riding on it."""
        avatar = getattr(self, "avatar", None)
        if avatar is None or not self.isVisible():
            return
        avatar.tick()

    # ── UI construction ───────────────────────────────────────────────────────

    def _watch_face_health(self) -> None:
        """Check the WebGL face actually came up, and fall back if it did not.

        The page reports its own failure and nothing used to listen, so a
        machine where WebGL will not start showed a message where ORION should
        be — for ever, with no log line and no fallback.
        """
        face = getattr(self, "face", None)
        check = getattr(face, "check_state", None)
        if check is None:
            return                      # not the WebGL face; nothing to watch

        watched = face
        token = object()
        watched._orion_health_watch = token
        attempts = [0]
        pending: list[Any] = [None]

        def current() -> bool:
            try:
                return (getattr(self, "face", None) is watched
                        and not getattr(self, "_shutting_down", False)
                        and not getattr(watched, "_orion_retired", False)
                        and getattr(watched, "_orion_health_watch", None) is token)
            except RuntimeError:
                return False

        def probe() -> None:
            if not current():
                return
            try:
                if not self.isVisible():
                    QTimer.singleShot(4000, probe)
                    return
            except RuntimeError:
                return
            marker = object()
            pending[0] = marker

            def answer(state: Any) -> None:
                if current() and pending[0] is marker:
                    pending[0] = None
                    verdict(state)

            # A dead renderer/pipe may never answer runJavaScript. Its absence
            # must also consume a bounded attempt, with late replies ignored.
            QTimer.singleShot(2000, lambda: answer("loading"))
            self._probe_face(check, answer)

        def verdict(state: Any) -> None:
            if not current():
                return
            state = str(state)
            if state == "alive":
                attempts[0] = 0
                QTimer.singleShot(4000, probe)
                return
            attempts[0] += 1
            if state != "failed" and attempts[0] < 3:
                QTimer.singleShot(4000, probe)
                return
            watched._orion_health_watch = None
            if not getattr(self, "_face_retried", False):
                # One fresh page before giving up. A WebGL context refused
                # under momentary GPU or memory pressure is usually granted a
                # few seconds later, and a fallback latched on a transient is
                # how the old sculpted head replaced ORION for a whole session.
                self._face_retried = True
                self.bus.log.emit("FACE: the WebGL avatar did not start — "
                                  "trying once more with a fresh page.")
                try:
                    self._rebuild_face()
                except Exception as exc:
                    self.bus.log.emit(f"FACE: the retry failed - {exc}")
                else:
                    if getattr(self, "face", None) is not watched:
                        return
            self.bus.log.emit(
                "FACE: the WebGL avatar will not start on this machine right "
                "now — showing ORION's orb, which needs no GPU. His face comes "
                "back on the next start.")
            os.environ["ORION_FACE_FALLBACK"] = "orb"
            try:
                self._swap_face()
            except Exception as exc:
                self.bus.log.emit(f"FACE: the fallback also failed - {exc}")

        if hasattr(watched, "on_failed"):
            # The host reports a missing window immediately, including after
            # it was initially healthy. Queue recovery outside its shutdown
            # stack, which also completes any pending query callbacks.
            watched.on_failed = lambda: QTimer.singleShot(0, lambda: verdict("failed"))
        if getattr(watched, "failed", False):
            QTimer.singleShot(0, lambda: verdict("failed"))
        else:
            QTimer.singleShot(6000, probe)

    @staticmethod
    def _probe_face(check: Any, verdict: Any) -> None:
        """Ask the page how it is getting on, and never raise doing it.

        The widget may have been rebuilt or destroyed between one poll and the
        next, and an exception here would come from a timer with nothing to
        catch it.
        """
        try:
            check(verdict)
        except Exception:
            pass

    def _build_face(self) -> Any:
        """ORION's avatar for THIS window.

        When the native renderer owns the face (Mark XXIII, the default), this
        returns a NativeFacePanel — an adapter that will host the scene
        renderer in portrait framing once app.py has one to give it. ORION is
        drawn ONCE, by that renderer, and this window shows his face at full
        size on the primary monitor while the Command Deck occupies the
        second.

        It previously returned the 2-D HologramFace here and left app.py to
        hide the whole column, on the reasoning that ORION was visible inside
        the swarm anyway. That is what the user actually saw at startup: a
        flat voxel face and no 3-D avatar anywhere.

        Order: the software head when explicitly asked for (ORION_HOLO_HEAD=1),
        then the SCREEN face — the default form, built before anything that
        touches WebEngine so the common startup loads no Chromium face process
        and no Three.js page at all — then the orb, then the classic WebEngine
        face where a separate window makes it possible, then the native GL
        panel (ORION_NATIVE_FACE=0 to skip), then the 2-D rig.

        Kept as a seam for a second reason: the 3-D WebEngine face cannot
        survive a native window recreation under the offscreen QPA platform,
        so headless tests that exercise window-flag changes (overlay mode)
        substitute the 2-D fallback here."""
        # An explicit ORION_HOLO_HEAD=1 WINS. It is the escape hatch for a
        # machine where WebEngine will not start, and an escape hatch that
        # only applies once the thing you are escaping has already failed is
        # not one — it also made the test suite non-deterministic, because it
        # could not choose the face it needed.
        if holo_head_enabled():
            try:
                from .holo_head import HoloHeadPanel
                return _wear_orion_colours(HoloHeadPanel())
            except Exception:
                pass    # numpy or the face asset missing: fall through

        # The user's choice comes before every automatic fallback below. A
        # human face is not always what you want in the room — shown to
        # somebody who did not ask to be looked at by one, it lands somewhere
        # between striking and unsettling — so the screen face is the default
        # and the orb is a first-class form rather than a degraded mode, and
        # asking for either is honoured exactly.
        from ..appearance import face_form
        form = face_form()
        if form == "screen":
            # ORION's screen face: pure QPainter on the face pipeline's worker
            # thread. It needs no WebGL, so a WebGL fallback flag does not
            # apply to it, and nothing on this path imports face3d.
            try:
                from .screen_face import ScreenFacePanel
                return _wear_orion_colours(ScreenFacePanel())
            except Exception as exc:
                from ..startup_report import say

                say(f"[ORION] FACE: the screen face could not be built ({exc}) "
                    f"- using the classic face instead.")
        webgl_ok = (webengine_face_possible()
                    and os.getenv("ORION_FACE_FALLBACK", "") != "orb")
        if form == "orb" or not webgl_ok and os.getenv("ORION_FACE_FALLBACK") == "orb":
            # His orb. The Three.js face holds its own orb form — the same
            # scene, so switching back to the face is a morph, not a reload —
            # and the software-painted orb stands in when WebGL cannot run.
            if webgl_ok:
                try:
                    from .face3d import QuantumFace3D
                    if QuantumFace3D.available:
                        face = _new_webgl_face()
                        face.set_form("orb")
                        return _wear_orion_colours(face)
                except Exception:
                    pass
            try:
                from .hud import CentralHud
                return _wear_orion_colours(CentralHud())
            except Exception:
                pass    # the orb could not be built; fall through to a face

        # Otherwise ORION's classic face: the quantum orb that holds the
        # screen while he comes up, then materialises into him.
        if webgl_ok:
            try:
                from .face3d import QuantumFace3D
                if QuantumFace3D.available:
                    return _wear_orion_colours(_new_webgl_face())
                raise RuntimeError("WebEngine is unavailable")
            except Exception:
                # A failed local loader must not replace an offline avatar
                # with another renderer needing a GL context or the internet.
                os.environ["ORION_FACE_FALLBACK"] = "orb"
                from .hud import CentralHud
                return _wear_orion_colours(CentralHud())
        if holo_head_enabled():
            try:
                from .holo_head import HoloHeadPanel
                return _wear_orion_colours(HoloHeadPanel())
            except Exception:
                pass    # numpy or the face asset missing: fall through
        if native_face_enabled():
            from .native_face import NativeFacePanel
            return _wear_orion_colours(NativeFacePanel())
        return _wear_orion_colours(HologramFace())

    # ── the face, behind one stable indirection ──────────────────────────────

    def _face_set_state(self, state: Any) -> None:
        self.face_state = str(state or "STANDBY").upper()
        # Keep the snapshot current for a rebuilt view or compact overlay.
        # Once attached, the behaviour controller is the only state writer.
        if getattr(self, "avatar", None) is None:
            self._to_face("set_state", state)

    def _face_set_amplitude(self, value: Any) -> None:
        self._to_face("set_amplitude", value)

    def _face_set_speaking(self, active: Any) -> None:
        self._to_face("set_speaking", active)

    def _face_set_viseme(self, payload: Any) -> None:
        """Forward a lip-sync viseme to the live face (Mark XXVI). A no-op on
        faces that do not implement set_viseme, via _to_face's guard."""
        if not isinstance(payload, dict):
            return
        if "open" in payload:
            # A posture measured from the audio's formants. There is no viseme
            # ID to give here — the spectrum yields openness and width, not a
            # name — so it is forwarded as numbers.
            self._to_face("set_viseme", float(payload.get("open", 0.0)),
                          float(payload.get("width", 0.0)),
                          float(payload.get("closure", 0.0)))
        else:
            self._to_face("set_viseme", payload.get("viseme", "sil"),
                          float(payload.get("weight", 1.0)))

    def _face_apply_emotion(self, name: Any, params: Any = None) -> None:
        self._to_face("apply_emotion", name, params)

    def _to_face(self, method: str, *args: Any) -> None:
        face = getattr(self, "face", None)
        fn = getattr(face, method, None)
        if fn is None:
            return
        try:
            fn(*args)
        except RuntimeError:
            pass   # the widget was destroyed mid-swap; the rebuild re-applies

    def _choose_face_form(self, form: str) -> None:
        """Menu handler: switch form and leave the ticks consistent."""
        self.set_face_form(form)
        self._sync_form_menu()

    def _form_button_toggled(self, button: Any, checked: bool) -> None:
        """A form button (Screen Face, Classic Face, Orb) became the selected
        one, however it was pressed."""
        if not checked or getattr(self, "_syncing_form", False):
            return
        for form, candidate in getattr(self, "_form_buttons", {}).items():
            if candidate is button:
                self._choose_face_form(form)
                return

    def _sync_form_menu(self) -> None:
        """Tick whichever form is actually in effect. Never raises: the menu
        is a convenience and must not be able to stop the window building."""
        self._syncing_form = True      # reflecting state is not choosing it
        try:
            from ..appearance import face_form

            current = face_form()
            for name, action in getattr(self, "_form_actions", {}).items():
                action.setChecked(name == current)
            button = getattr(self, "_form_buttons", {}).get(current)
            if button is not None:
                button.setChecked(True)     # the group unticks the other
        except Exception:
            pass
        finally:
            self._syncing_form = False

    #: The Three.js renderer's own names for the two forms it can morph
    #: between. The screen face is not one of them: it is a different widget.
    _WEBGL_FORMS = {"classic": "face", "orb": "orb"}

    def set_face_form(self, form: Any) -> str:
        """Put ORION in his screen face, classic face or orb, now, and
        remember it.

        Returns the form in effect, or "" if *form* named none of them.
        Applied live: somebody asking for the orb because a friend has just
        walked in needs it gone now, not after a restart.
        """
        from ..appearance import face_form, form_label, set_face_form

        chosen = set_face_form(form)
        if not chosen:
            return ""
        if chosen != "orb" and self._orb_hands_enabled:
            self.set_orb_hand_control(False)
        if chosen == "classic" and os.environ.pop("ORION_FACE_FALLBACK", None):
            # He fell back to the orb because WebGL would not start. Being
            # asked for the classic face is a reason to try again, not to
            # refuse.
            self._face_retried = False
        face = getattr(self, "face", None)
        current = self._current_face_form()
        if chosen != current:
            if (hasattr(face, "set_form") and chosen in self._WEBGL_FORMS
                    and current in self._WEBGL_FORMS):
                # The Three.js face morphs between its face and its orb in
                # place: no new page, no reload, no flash — the change the
                # room sees is him folding into the orb (or out of it).
                face.set_form(self._WEBGL_FORMS[chosen])
            else:
                self._swap_face()
        avatar = getattr(self, "avatar", None)
        if avatar is not None and hasattr(avatar, "set_form"):
            # The behaviour engine knows "face", "orb" and "auto"; the screen
            # face runs it on "auto" so its dormancy channel still reports
            # standby (the screen face rests on it).
            avatar.set_form(self._WEBGL_FORMS.get(chosen, "auto"))
        self.bus.log.emit(f"GUI: ORION is now wearing his {form_label(chosen)}.")
        return face_form()

    def _current_face_form(self) -> str:
        """Which form the widget on screen actually IS — not what is stored.

        Asked to switch, the stored value has already been written, so
        comparing against it would always say "no change needed" and the
        screen would never move.
        """
        face = getattr(self, "face", None)
        kind = type(face).__name__
        if kind == "ScreenFacePanel":
            return "screen"
        form = getattr(face, "form", None)
        if form == "orb":
            return "orb"
        if form == "face":
            return "classic"
        return "orb" if kind == "CentralHud" else "classic"

    @staticmethod
    def _retire_face(old: Any, layout: Any) -> None:
        """Take a face off screen for good. A face drawn by a face host is
        asked to quit first: deleting the widget alone left its process (and
        ~350 MB) running until ORION exited, because the writer thread keeps
        the Python object alive (Mark XXXV W1.2 review, 11 Oct 2026)."""
        old._orion_retired = True
        old._orion_health_watch = None
        if hasattr(old, "on_failed"):
            old.on_failed = None
        shutdown = getattr(old, "shutdown", None)
        if callable(shutdown) and hasattr(old, "host_wids"):
            try:
                shutdown()
            except Exception:
                pass
        # A software face renders on a worker thread (face_pipeline). The
        # widget gets no close event here, only deleteLater, so its worker is
        # stopped explicitly — otherwise it went on drawing a face nobody
        # could see until ORION exited.
        stop_pipeline = getattr(old, "stop_pipeline", None)
        if callable(stop_pipeline):
            try:
                stop_pipeline()
            except Exception:
                pass
        try:
            layout.removeWidget(old)
            old.setParent(None)
            old.deleteLater()
        except RuntimeError:
            pass

    def _swap_face(self) -> None:
        """Rebuild the face widget in place, whatever kind it is.

        _rebuild_face exists for the same job but returns early unless the
        current face is a WebEngine one, because its purpose is recovering
        from a native window recreation. Switching forms has to work in both
        directions, including orb -> face where the OLD widget is not
        WebEngine at all.
        """
        old = getattr(self, "face", None)
        container = getattr(self, "face_container", None)
        if old is None or container is None:
            return
        layout = container.layout()
        if layout is None:
            return
        self._retire_face(old, layout)
        self.face = self._build_face()
        self._watch_face_health()
        layout.addWidget(self.face, 1)
        if (self._orb_hands_enabled and
            (not hasattr(self.face, "set_orb_hands") or
             not getattr(self.face, "supports_orb_hands", True))):
            self.set_orb_hand_control(False)
        self._to_face("set_state", getattr(self, "face_state", "STANDBY"))
        self._to_face("set_orb_camera_enabled", self._orb_hands_enabled)
        renderer = getattr(self, "face_renderer", None)
        if renderer is not None and hasattr(self.face, "attach_renderer"):
            try:
                self.face.attach_renderer(renderer)
            except Exception:
                pass
        avatar = getattr(self, "avatar", None)
        if avatar is not None and hasattr(avatar, "attach"):
            try:
                avatar.attach(self.face)
            except Exception:
                pass

    def _rebuild_face(self) -> None:
        """Replace the face widget after the native window was recreated.

        Overlay mode calls setWindowFlags, which forces Qt to destroy and
        recreate the native window — and a QWebEngineView cannot survive that.
        Coming back from the overlay left ORION's 3-D face rendering pitch
        black. Nothing recovers it in place, so the widget is rebuilt and its
        state re-applied. Costs a page load, and only when the flags change.
        """
        old = getattr(self, "face", None)
        if old is None or not self._face_is_webengine():
            return          # the native and 2-D rigs both survive recreation
        layout = self.face_container.layout()
        self._retire_face(old, layout)
        self.face = self._build_face()
        self._watch_face_health()
        layout.addWidget(self.face, 1)
        if (self._orb_hands_enabled and
            (not hasattr(self.face, "set_orb_hands") or
             not getattr(self.face, "supports_orb_hands", True))):
            self.set_orb_hand_control(False)
        # Re-apply what he was doing, or he comes back in whatever state a
        # freshly constructed avatar defaults to.
        self._to_face("set_state", getattr(self, "face_state", "STANDBY"))
        self._to_face("set_orb_camera_enabled", self._orb_hands_enabled)
        renderer = getattr(self, "face_renderer", None)
        if renderer is not None and hasattr(self.face, "attach_renderer"):
            self.face.attach_renderer(renderer)
        avatar = getattr(self, "avatar", None)
        if avatar is not None and hasattr(avatar, "attach"):
            try:
                avatar.attach(self.face)
            except Exception:
                pass

    def _build_ui(self) -> None:
        root   = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(SPACE.MD, SPACE.MD, SPACE.MD, SPACE.MD)
        layout.setSpacing(SPACE.MD)

        self.header_frame = self._build_header()
        layout.addWidget(self.header_frame)

        self.face = self._build_face()

        # Face-only shell: LOG/MEMORY/TELEMETRY, the inner-voice thought
        # stream and the calendar/geo panel all live on the Command Deck
        # (OPS page) — there is nothing left to switch between here.
        #
        # Mark XXI, Track A1: the one thing the user explicitly asked to see
        # alongside the face, in the SAME window, is the swarm — ORION's
        # live 3D map of himself. A collapsible rail (own SwarmDeckView
        # instance, attached late once app.py has real backends — see
        # attach_swarm_view) sits beside the face in a QSplitter. It starts
        # hidden so this window's construction/geometry/startup behaviour is
        # completely unchanged unless the user opts in (toggle_swarm_rail,
        # Ctrl+Shift+S) — the dual-monitor Command Deck layout in app.py's
        # _apply_startup_layout is untouched, and the Command Deck keeps its
        # own independent SWARM page exactly as before.
        face_container = QWidget()
        face_layout = QVBoxLayout(face_container)
        face_layout.setContentsMargins(0, 0, 0, 0)
        face_layout.addWidget(self.face, 1)
        # Held so _rebuild_face can swap the widget after overlay mode has
        # forced the native window to be recreated.
        self.face_container = face_container

        self.swarm_rail = QWidget()
        swarm_rail_layout = QVBoxLayout(self.swarm_rail)
        swarm_rail_layout.setContentsMargins(0, 0, 0, 0)
        self._swarm_rail_placeholder = QLabel("")
        self._swarm_rail_placeholder.setObjectName("mutedLabel")
        self._swarm_rail_placeholder.setWordWrap(True)
        swarm_rail_layout.addWidget(self._swarm_rail_placeholder)
        self.swarm_view: Any = None
        self.swarm_rail.hide()

        # Mark XXII, Phase 5: the panel the Command Deck is docked into when
        # the unified shell is active. Built empty and hidden always, so the
        # two-window layout is completely unaffected — nothing is reserved,
        # nothing is shown, until attach_deck_inline is actually called.
        self.deck_inline: Any = None
        self.deck_panel = QWidget()
        deck_layout = QVBoxLayout(self.deck_panel)
        deck_layout.setContentsMargins(0, 0, 0, 0)
        self._deck_placeholder = QLabel("")
        self._deck_placeholder.setObjectName("mutedLabel")
        self._deck_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        deck_layout.addWidget(self._deck_placeholder)
        self.deck_panel.hide()

        # The face must never be squeezed out of existence. A QSplitter lets a
        # child collapse to zero width by default, and with the deck docked
        # beside it that is exactly what happened: the face reached 0px, its
        # WebGL surface reported "Framebuffer is incomplete: Attachment has
        # zero size" on every frame, and ORION's face — the entire point of
        # the unified shell — was simply absent from the window.
        face_container.setMinimumWidth(360)
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.addWidget(face_container)
        self.content_splitter.addWidget(self.swarm_rail)
        self.content_splitter.addWidget(self.deck_panel)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 2)
        self.content_splitter.setStretchFactor(2, 5)
        # The HUD wraps the splitter rather than replacing it. Everything
        # already built — the face, the swarm rail, the docked deck — stays
        # where it was and keeps working as it did; the HUD only adds a
        # telemetry rail down the left, the log and somewhere to type down
        # the right, and status along the bottom. The face is passed through
        # as the centre widget and is never wrapped, restyled or reparented
        # beyond being placed.
        from .hud_shell import HudShell, hud_stylesheet

        self.hud = HudShell(self.content_splitter)
        self.hud.setStyleSheet(hud_stylesheet())
        layout.addWidget(self.hud, 1)
        self._wire_hud()
        self._populate_controls()

        # The compact orb is its own window now (gui/orb_overlay.py), with its
        # own restore / shut-down chips; nothing in this window changes shape.
        self.setCentralWidget(root)

        # ── keyboard shortcuts ────────────────────────────────────────────────
        # (Ctrl+Return "send manual message" lived here only because the LOG
        # view's input line used to be on this window; that view is now a
        # Command Deck page and keeps its own shortcut there.)
        fullscreen_action = QAction(self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        self.addAction(fullscreen_action)

        overlay_action = QAction(self)
        overlay_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        overlay_action.triggered.connect(self.toggle_overlay_mode)
        self.addAction(overlay_action)

        dashboard_action = QAction(self)
        dashboard_action.setShortcut(QKeySequence("Ctrl+D"))
        dashboard_action.triggered.connect(self.toggle_dashboard)
        self.addAction(dashboard_action)

        centre_action = QAction(self)
        centre_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        centre_action.triggered.connect(self.toggle_command_centre)
        self.addAction(centre_action)

        # Mission page (14 dockable panels) was previously reachable only by
        # opening the deck — which always lands on Widgets or Command Centre
        # — then swiping/clicking left. It's the richest page and deserves
        # its own shortcut, same pattern as the two above.
        mission_action = QAction(self)
        mission_action.setShortcut(QKeySequence("Ctrl+M"))
        mission_action.triggered.connect(self.toggle_mission_deck)
        self.addAction(mission_action)

        mute_action = QAction(self)
        mute_action.setShortcut(QKeySequence("Ctrl+Shift+M"))
        mute_action.triggered.connect(self._toggle_voice_mute)
        self.addAction(mute_action)

        workbench_action = QAction(self)
        workbench_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        workbench_action.triggered.connect(self.open_electronics_workbench)
        self.addAction(workbench_action)

        # §1.2 — Ctrl+K opens the fuzzy command palette over every dispatcher tool.
        palette_action = QAction(self)
        palette_action.setShortcut(QKeySequence("Ctrl+K"))
        palette_action.triggered.connect(self.open_command_palette)
        self.addAction(palette_action)
        # The Command Deck's visible Search bar opens the same palette from the
        # other window, via the bus (the two windows do not hold references to
        # each other). Queued: the palette is a modal dialog (palette.exec()),
        # and a modal loop run inline inside an asyncio task re-enters the
        # other tasks ("Cannot enter into task" - see gui_command below).
        try:
            self.bus.open_palette.connect(self.open_command_palette,
                                          Qt.ConnectionType.QueuedConnection)
        except Exception:
            pass

        pause_action = QAction(self)
        pause_action.setShortcut(QKeySequence("Ctrl+Space"))
        pause_action.triggered.connect(self._toggle_pause)
        self.addAction(pause_action)

        # Stop (cancel the current request) — distinct from pause: it scraps the
        # in-flight turn entirely so the user can rephrase from scratch.
        stop_action = QAction(self)
        stop_action.setShortcut(QKeySequence("Ctrl+Shift+Space"))
        stop_action.triggered.connect(self._stop_turn)
        self.addAction(stop_action)

        quit_action = QAction(self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self._quit)
        self.addAction(quit_action)

    # ── the HUD ──────────────────────────────────────────────────────────────

    def _wire_hud(self) -> None:
        """Feed the HUD from signals that already existed.

        Nothing here is a new source of truth. The telemetry loop, the log and
        the content results were all already being emitted; before the HUD
        they reached a second window, or nothing at all.
        """
        self.bus.telemetry_sample.connect(self.hud.rail.apply_sample)
        self.bus.log.connect(self.hud.log.append)
        from .. import session_log
        session_log.attach(self.bus)        # no-op unless orion.py enabled it
        self.bus.mic_enabled.connect(self.hud.command.set_mic)

        self.hud.command.submitted.connect(self._hud_submit)
        self.hud.command.interrupted.connect(self._stop_turn)
        self.hud.command.mic_toggled.connect(self._hud_set_mic)
        self.hud.drop.dropped.connect(self._hud_files_dropped)

        # How long ORION has been active, on its own slow timer: a minute is
        # its smallest step, so the 0.75 s telemetry tick would be waste.
        self._hud_facts_timer = QTimer(self)
        self._hud_facts_timer.setInterval(15_000)
        self._hud_facts_timer.timeout.connect(self._refresh_hud_facts)
        self._hud_facts_timer.start()
        self._refresh_hud_facts()

    def open_controls(self) -> None:
        """The controls column, as a popover under the gear.

        A popover rather than a docked column because it is a list of things
        you do occasionally, and giving it permanent screen width would take
        that width from the face.
        """
        hud = getattr(self, "hud", None)
        if hud is None:
            return
        panel = hud.controls
        if panel.isVisible():
            panel.hide()
            return
        from .hud_shell import hud_stylesheet

        panel.setParent(self, Qt.WindowType.Popup)
        panel.setStyleSheet(hud_stylesheet(
            getattr(self, "_accent_override", "") or ""))
        panel.adjustSize()
        button = getattr(self, "controls_btn", None)
        if button is not None:
            corner = button.mapToGlobal(button.rect().bottomLeft())
            panel.move(corner.x(), corner.y() + 4)
        panel.show()
        panel.raise_()

    def _refresh_hud_facts(self) -> None:
        """'ORION has been active for 3 h 14 min' - since this ORION started."""
        try:
            import time as _time

            started = getattr(self, "_orion_started_at", None)
            if started is None:
                started = self._orion_started_at = psutil.Process().create_time()
            seconds = max(0, int(_time.time() - started))
            hours, minutes = seconds // 3600, (seconds % 3600) // 60
            days, hours = hours // 24, hours % 24
            # "2 days 3 h", "3 h 14 min", "32 min" - said, not "51:14".
            if days:
                active = f"{days} day{'s' if days != 1 else ''} {hours} h"
            elif hours:
                active = f"{hours} h {minutes} min"
            elif minutes:
                active = f"{minutes} min"
            else:
                active = ""
            self.hud.rail.set_active(active)
        except Exception:
            pass

    def _hud_submit(self, text: str) -> None:
        """Send typed text to ORION, the same way the deck's uplink does."""
        message = str(text or "").strip()
        if not message:
            return
        self.write_log(f"YOU: {message}")
        if self.worker is None:
            self.write_log("SYS: Live session not ready.")
            return
        try:
            from .. import background

            background.spawn(self.worker.submit_text(message))
        except Exception as exc:
            self.write_log(f"SYS: could not send that - {exc}")

    def _hud_set_mic(self, active: bool) -> None:
        self.bus.mic_enabled.emit(bool(active))
        self.write_log(f"SYS: microphone {'on' if active else 'muted'}.")

    def _hud_files_dropped(self, paths: Any) -> None:
        """Hand a dropped file to ORION.

        The widget neither reads nor interprets it — what a dropped file means
        is the worker's decision, not a drop zone's.
        """
        chosen = [str(p) for p in (paths or []) if str(p).strip()]
        if not chosen:
            return
        for path in chosen:
            self.write_log(f"YOU: (dropped) {path}")
        worker = self.worker
        review = getattr(worker, "review_file", None) if worker else None
        if review is None:
            self.write_log("SYS: nothing is ready to look at that yet.")
            return
        try:
            from .. import background, file_review

            async def review_all():
                # 30 Oct 2026 ("does he actually process them properly?"): he
                # read the file's whole path aloud and the conversation never
                # saw a word of it. Handed over, the review leads with what the
                # file is and an offer, and its content joins the conversation
                # before he speaks, so "yes, summarise it" can be answered.
                with file_review.handed_over(lambda note: _share_with_live(worker, note)):
                    for path in dict.fromkeys(chosen):
                        _note_dropped_file(worker, path, await review(path))
            background.spawn(review_all(), name="orion-dropped-files")
        except Exception as exc:
            self.write_log(f"SYS: could not open that file - {exc}")

    # ── the controls column ──────────────────────────────────────────────────

    def _populate_controls(self) -> None:
        """Every switch, named, in one column.

        These were a `\u2022\u2022\u2022` menu, a handful of keyboard shortcuts and
        several settings reachable only by asking ORION out loud. A control
        you cannot see is a control you do not have.
        """
        add = self.hud.controls.add
        add("remote", "Phone Pairing", self.open_remote_control,
            icon="phone", primary=True,
            hint="Pair a phone, and show the QR code to do it.")
        add("fullscreen", "Fullscreen  [F11]", self.toggle_fullscreen,
            icon="maximize")
        add("shortcut", "Create Desktop Shortcut", self.create_desktop_shortcut,
            icon="monitor", hint="Put ORION on the desktop and Start menu.")
        add("autostart", "Start at Login: \u2026", self.toggle_autostart,
            icon="login", hint="Start ORION when you log in.")
        add("customise", "Preferences", self.open_preferences,
            icon="sliders", hint="Name, voice, accent and audio devices.")
        add("brief", "Brief Me Now", self.run_morning_brief,
            icon="bulb", hint="Hear the latest briefing for this part of the day.")
        add("ptt", "Push to Talk: \u2026", self.toggle_push_to_talk,
            icon="keyboard", hint="Hold a key to talk, instead of always listening.")
        add("overlay", "Compact Face  [Ctrl+Shift+O]", self.toggle_overlay_mode,
            icon="compact", hint="Shrink ORION to his floating face.")
        add("audio", "Audio Devices", self.open_preferences,
            icon="volume-high", hint="Choose the microphone and the speakers.")
        add("memory", "Memory", self.open_memory_viewer,
            icon="database", hint="Read and delete what ORION remembers.")
        add("palette", "Command Palette  [Ctrl+K]", self.open_command_palette,
            icon="search")
        add("deck", "Command Deck  [Ctrl+D]", self.toggle_dashboard,
            icon="dashboard")
        self._refresh_control_states()

    def _refresh_control_states(self) -> None:
        """Make each toggle's label say what it currently is.

        "AUTO-START: OFF" rather than a checkbox whose meaning depends on
        which way round you read it. Each state is read back from the
        subsystem that owns it rather than remembered here, so a label cannot
        drift away from the truth.
        """
        hud = getattr(self, "hud", None)
        if hud is None:
            return

        state = "?"
        try:
            from ..autostart import status

            state = "ON" if status().enabled else "OFF"
        except Exception:
            pass
        hud.controls.set_state("autostart", f"Start at Login: {state.lower()}")

        state = "?"
        try:
            from ..push_to_talk import enabled

            state = "ON" if enabled() else "OFF"
        except Exception:
            pass
        hud.controls.set_state("ptt", f"Push to Talk: {state.lower()}")

    # ── what those controls needed ───────────────────────────────────────────

    def open_remote_control(self) -> None:
        """Show phone pairing in the Security Centre."""
        self._open_deck_page("SECURITY")

    def create_desktop_shortcut(self) -> None:
        """Put ORION on the desktop and in the Start menu."""
        try:
            from ..desktop_app import install_shortcuts

            made = install_shortcuts()
        except Exception as exc:
            self.write_log(f"SYS: could not create the shortcut - {exc}")
            return
        placed = [place for place, ok in (made or {}).items() if ok]
        if placed:
            self.write_log(f"SYS: shortcut created on the {' and '.join(placed)}.")
        else:
            self.write_log("SYS: no shortcut was created.")

    def toggle_autostart(self) -> None:
        """Start ORION with Windows, or stop doing that."""
        try:
            from .. import autostart

            state = autostart.status()
            if not state.supported:
                self.write_log(f"SYS: auto-start is unavailable - {state.detail}")
                return
            result = autostart.disable() if state.enabled else autostart.enable()
            self.write_log(
                f"SYS: auto-start {'off' if state.enabled else 'on'}"
                f"{' - ' + result.detail if result.detail else ''}.")
        except Exception as exc:
            self.write_log(f"SYS: auto-start unchanged - {exc}")
        self._refresh_control_states()

    def toggle_push_to_talk(self) -> None:
        """Hold a key to talk, rather than ORION always listening."""
        try:
            from .. import push_to_talk

            if push_to_talk.enabled():
                push_to_talk.disable()
                self.write_log("SYS: push-to-talk off — ORION is listening again.")
            else:
                push_to_talk.enable()
                self.write_log(
                    f"SYS: push-to-talk on — hold "
                    f"{push_to_talk.chord_label()} to speak.")
        except Exception as exc:
            self.write_log(f"SYS: push-to-talk unchanged - {exc}")
        self._refresh_control_states()

    def run_morning_brief(self) -> None:
        """Read this morning's briefing aloud, now.

        Deliberately an action rather than an ON/OFF switch. A toggle whose
        state nothing actually reads is worse than no toggle: it looks like it
        configures the briefing and does not.
        """
        if self.worker is None:
            self.write_log("SYS: Live session not ready.")
            return
        try:
            from .. import background
            from ..time_service import TIME

            # The part of the day it IS. This button used to send "Give me my
            # morning briefing." whatever the clock said, and ORION — quite
            # reasonably — announced a morning briefing at half past six at
            # night.
            period = TIME.greeting_period()
            background.spawn(self.worker.submit_text(f"Give me my {period} briefing."))
        except Exception as exc:
            self.write_log(f"SYS: could not start the briefing - {exc}")

    def _build_header(self) -> QFrame:
        frame  = QFrame()
        frame.setObjectName("headerFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(SPACE.LG, SPACE.SM + 2, SPACE.LG, SPACE.SM + 2)
        layout.setSpacing(SPACE.SM)

        # Mark XXXI: a wordmark and a separate mark tag, rather than one long
        # title — the version is legible at a glance and the name stays quiet.
        from ..constants import APP_MARK_TAG

        # No trailing dot (owner, 28 Oct 2026): the header reads
        # "O.R.I.O.N" and then the mark tag.
        self.title_label    = QLabel("O.R.I.O.N")
        self.title_label.setObjectName("brandMark")
        self.mark_label = QLabel(APP_MARK_TAG)
        self.mark_label.setObjectName("markTag")
        # Subtitle carries a build stamp so a running window is unmistakably the
        # current build (a stale instance shows a different / missing stamp).
        self.subtitle_label = QLabel(f"{APP_SUBTITLE}   ·   BUILD {APP_BUILD}")
        self.subtitle_label.setObjectName("subtitleLabel")
        self.title_label.setToolTip(f"{APP_NAME}\n{self.subtitle_label.text()}")
        self.mark_label.setToolTip(self.subtitle_label.text())
        self.subtitle_label.hide()
        # Mark XXXV: a hello beside the name, in the human tone - the window
        # greets the person in front of it rather than only naming itself.
        self.greeting_label = QLabel(_greeting_words(datetime.now()))
        self.greeting_label.setObjectName("greetingLabel")
        text_box = QHBoxLayout()
        text_box.setSpacing(SPACE.MD - 4)
        text_box.addWidget(self.title_label)
        text_box.addWidget(self.mark_label, 0, Qt.AlignmentFlag.AlignVCenter)
        text_box.addSpacing(SPACE.SM)
        text_box.addWidget(self.greeting_label, 0, Qt.AlignmentFlag.AlignVCenter)
        text_box.addWidget(self.subtitle_label)

        # ── Window controls — one compact icon cluster (was five loose buttons)
        # Plain geometric glyphs, not the emoji ⏸/⏹: Windows draws those as
        # blue picture-emoji, the only colour on screen that was not ORION's.
        self.pause_btn = QPushButton("Pause")
        from .line_icons import apply as apply_icon
        apply_icon(self.pause_btn, "pause")
        self.pause_btn.setAccessibleName("Pause speech and listening")
        self.pause_btn.setToolTip("Pause ORION's speech and listening (say 'pause'); "
                                  "say 'resume' or 'Orion' to zone back in.")
        self.pause_btn.clicked.connect(self._toggle_pause)

        # Stop = cancel the current request outright (Ctrl+Shift+Space, or say
        # "Orion cancel"/"never mind"), returning to listening so you can
        # rephrase.  Unlike Pause it does not preserve position.
        self.stop_btn = QPushButton("Stop")
        from .line_icons import apply as apply_icon
        apply_icon(self.stop_btn, "stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setToolTip("Cancel the current request and start over "
                                 "(Ctrl+Shift+Space; say 'Orion cancel' or "
                                 "'never mind').")
        self.stop_btn.clicked.connect(self._stop_turn)

        # Mute = silence ORION's VOICE only (Ctrl+Shift+M, or say "mute
        # yourself" / "unmute"). He keeps listening and working; his replies
        # still arrive as text. Not the PC's volume — that is its own control.
        self._voice_muted = False
        self.mute_btn = QPushButton("Mute")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setAccessibleName("Mute ORION's voice")
        self.mute_btn.setToolTip("Mute ORION's voice — he keeps listening and replies in "
                                 "text (Ctrl+Shift+M; say 'mute yourself' / 'unmute').")
        self.mute_btn.clicked.connect(self._toggle_voice_mute)

        self.dashboard_btn = QPushButton("")
        from .line_icons import apply as apply_icon
        apply_icon(self.dashboard_btn, "dashboard")
        self.dashboard_btn.setObjectName("iconButton")
        self.dashboard_btn.setToolTip("Command Deck → Widgets (Ctrl+D)")
        self.dashboard_btn.setAccessibleName('Widgets page')
        self.dashboard_btn.clicked.connect(self.toggle_dashboard)
        self.centre_btn = QPushButton("")
        from .line_icons import apply as apply_icon
        apply_icon(self.centre_btn, "gauge")
        self.centre_btn.setObjectName("iconButton")
        self.centre_btn.setToolTip("Command Deck → Command Centre (Ctrl+Shift+C)")
        self.centre_btn.setAccessibleName('Command Centre')
        self.centre_btn.clicked.connect(self.toggle_command_centre)
        # Mission (14 dockable panels) previously had no button/shortcut of
        # its own — the richest page on the deck, buried behind a swipe.
        self.mission_btn = QPushButton("")
        from .line_icons import apply as apply_icon
        apply_icon(self.mission_btn, "flag")
        self.mission_btn.setObjectName("iconButton")
        self.mission_btn.setToolTip("Command Deck → Mission (Ctrl+M)")
        self.mission_btn.setAccessibleName('Mission page')
        self.mission_btn.clicked.connect(self.toggle_mission_deck)
        # The fuzzy command palette (Ctrl+K, searches every dispatcher tool)
        # was hotkey-only and undiscoverable without a visible affordance.
        self.palette_btn = QPushButton("")
        from .line_icons import apply as apply_icon
        apply_icon(self.palette_btn, "search")
        self.palette_btn.setObjectName("iconButton")
        self.palette_btn.setToolTip("Command palette — search every tool (Ctrl+K)")
        self.palette_btn.setAccessibleName('Command palette')
        self.palette_btn.clicked.connect(self.open_command_palette)
        self.overlay_btn = QPushButton("")
        from .line_icons import apply as apply_icon
        apply_icon(self.overlay_btn, "compact")
        self.overlay_btn.setObjectName("iconButton")
        self.overlay_btn.setToolTip("Compact face (Ctrl+Shift+O; Esc to exit)")
        self.overlay_btn.setAccessibleName('Compact overlay')
        self.overlay_btn.clicked.connect(self.toggle_overlay_mode)
        self.fullscreen_btn = QPushButton("")
        from .line_icons import apply as apply_icon
        apply_icon(self.fullscreen_btn, "maximize")
        self.fullscreen_btn.setObjectName("iconButton")
        self.fullscreen_btn.setToolTip("Fullscreen (F11)")
        self.fullscreen_btn.setAccessibleName('Fullscreen')
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)

        # The controls popover's trigger. Everything that used to be buried
        # in a '•••' menu or a shortcut nobody could discover lives behind it.
        self.controls_btn = QPushButton("")
        from .line_icons import apply as apply_icon
        apply_icon(self.controls_btn, "sliders")
        self.controls_btn.setObjectName("iconButton")
        self.controls_btn.setToolTip("Controls — settings, pairing, shortcuts")
        self.controls_btn.setAccessibleName("Controls")
        self.controls_btn.clicked.connect(self.open_controls)

        self.workbench_btn = QPushButton("Vision Lab")
        self.workbench_btn.setObjectName("workbenchShortcut")
        self.workbench_btn.setToolTip("Scan anything with the camera, on a grid (Ctrl+Shift+E)")
        self.workbench_btn.clicked.connect(self.open_electronics_workbench)
        self.research_btn = QPushButton("Research")
        self.research_btn.setObjectName("workbenchShortcut")
        self.research_btn.setToolTip("The live research console — every search, page and note")
        self.research_btn.setAccessibleName("Research Console")
        self.research_btn.clicked.connect(lambda: self._open_deck_page("RESEARCH"))

        # Screen Face | Classic Face | Orb, visible. It was two items buried in
        # an "Appearance" submenu of the ••• menu, which is not where anybody
        # looks when a visitor has just been unsettled by a face on the screen.
        self.form_toggle = QFrame()
        self.form_toggle.setObjectName("segmented")
        form_row = QHBoxLayout(self.form_toggle)
        form_row.setContentsMargins(3, 3, 3, 3)
        form_row.setSpacing(2)
        self._form_buttons: dict[str, QPushButton] = {}
        # One exclusive group, followed through `buttonToggled` rather than
        # each button's `clicked`: a mouse click emits both, but Windows
        # accessibility (Narrator, Voice Access, UI Automation) toggles a
        # checkable button WITHOUT a click. With `clicked` alone, toggling
        # "Show ORION's face" that way changed nothing and left both lit.
        self._form_group = QButtonGroup(self.form_toggle)
        self._form_group.setExclusive(True)
        # Only the forms this build offers (appearance.ENABLED_FORMS - the
        # Screen Face alone in Mark XXXVII). The classic face and the orb stay
        # defined and their code paths stay in the tree, but get no button.
        from ..appearance import enabled_forms
        _offered = enabled_forms()
        for form, label, tip, phrase in (
            ("screen", "Screen Face", "Show ORION's screen face",
             "switch to your screen face"),
            ("classic", "Classic Face", "Show ORION's classic 3-D face",
             "switch to your classic face"),
            ("orb", "Orb", "Show ORION as his orb", "switch to your orb form"),
        ):
            if form not in _offered:
                continue
            button = QPushButton(label)
            button.setObjectName("formItem")
            button.setCheckable(True)
            button.setToolTip(f"{tip} (or say \"{phrase}\")")
            button.setAccessibleName(tip)
            self._form_group.addButton(button)
            form_row.addWidget(button)
            self._form_buttons[form] = button
        self._form_group.buttonToggled.connect(self._form_button_toggled)
        if len(self._form_buttons) < 2:
            # One form is nothing to choose between: no lone "Screen Face"
            # pill in the header. (Built and hidden, so the attribute exists.)
            self.form_toggle.hide()
        self.compact_btn = QPushButton("Compact")
        from .line_icons import apply as apply_icon
        apply_icon(self.compact_btn, "compact")
        self.compact_btn.setObjectName("headerAction")
        self.compact_btn.setToolTip("Shrink ORION to his floating face (Ctrl+Shift+O)")
        self.compact_btn.setAccessibleName("Compact face")
        self.compact_btn.clicked.connect(self.toggle_overlay_mode)
        self.more_btn = QToolButton()
        self.more_btn.setText("•••")
        self.more_btn.setObjectName("workspaceMenu")
        self.more_btn.setAccessibleName("More ORION controls")
        self.more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        more_menu = QMenu(self.more_btn)
        for label, callback in (
            ("Mission    Ctrl+M", self.toggle_mission_deck),
            ("Widgets    Ctrl+D", self.toggle_dashboard),
            ("Command Centre    Ctrl+Shift+C", self.toggle_command_centre),
            ("Research Console", lambda: self._open_deck_page("RESEARCH")),
            ("Compact Face    Ctrl+Shift+O", self.toggle_overlay_mode),
            ("Fullscreen    F11", self.toggle_fullscreen),
        ):
            more_menu.addAction(label).triggered.connect(callback)

        # Which form ORION wears. One checkable entry per form rather than a
        # toggle, so the menu says what he is currently in as well as what he
        # could be — somebody reaching for this is usually in a hurry with
        # another person in the room.
        #
        # Only offered forms appear (appearance.ENABLED_FORMS - the Screen
        # Face alone in Mark XXXVII). With nothing to choose between, the
        # Appearance submenu is not built at all, and the orb-only hand
        # control entries go with it; the action object still exists (detached,
        # disabled) so code that checks it keeps working.
        self._form_actions = {}
        self.orb_hands_action = QAction("Camera Hand Control", self)
        self.orb_hands_action.setCheckable(True)
        self.orb_hands_action.setEnabled(False)
        self.orb_hands_action.setToolTip(
            "Move the orb with one open hand; pull two open hands apart to scatter it")
        self.orb_hands_action.triggered.connect(self.set_orb_hand_control)
        if len(_offered) > 1 or "orb" in _offered:
            more_menu.addSeparator()
            appearance = more_menu.addMenu("Appearance")
            for form, label in (("screen", "Screen Face"), ("classic", "Classic Face"),
                                ("orb", "Orb")):
                if form not in _offered:
                    continue
                action = appearance.addAction(label)
                action.setCheckable(True)
                action.triggered.connect(
                    lambda _checked=False, chosen=form: self._choose_face_form(chosen))
                self._form_actions[form] = action
            self._sync_form_menu()
            if "orb" in _offered:
                appearance.addAction(self.orb_hands_action)
                help_action = appearance.addAction("Orb Controls…")
                help_action.triggered.connect(self._show_orb_controls)
        self.more_btn.setMenu(more_menu)

        self.control_cluster = QFrame()
        self.control_cluster.setObjectName("controlCluster")
        cluster_row = QHBoxLayout(self.control_cluster)
        cluster_row.setContentsMargins(SPACE.XS, SPACE.XS, SPACE.XS, SPACE.XS)
        cluster_row.setSpacing(SPACE.XS)
        for button in (self.mission_btn, self.dashboard_btn, self.centre_btn,
                       self.overlay_btn, self.fullscreen_btn):
            button.setParent(self.control_cluster)
            button.hide()
        for button in (self.workbench_btn, self.research_btn, self.palette_btn,
                       self.controls_btn, self.more_btn):
            cluster_row.addWidget(button)

        # Explicit shutdown — closing this window or Ctrl+Q stops ORION fully.
        self.quit_btn = QPushButton("")
        from .line_icons import apply as apply_icon
        apply_icon(self.quit_btn, "power")
        self.quit_btn.setObjectName("quitButton")
        self.quit_btn.setToolTip("Shut ORION down completely (Ctrl+Q)")
        self.quit_btn.setAccessibleName("Shut ORION down")
        self.quit_btn.clicked.connect(self._quit)

        # ── Status chips — grouped on the right ───────────────────────────────
        self.voice_led = QLabel("VOICE ○")
        self.voice_led.setObjectName("voiceLed")
        self.voice_led.setProperty("speaking", "false")
        self.voice_led.hide()
        self.state_label = QLabel(_state_words("INITIALISING"))
        self.state_label.setObjectName("stateLabel")
        self.clock_label  = QLabel(_clock_words(datetime.now()))
        self.clock_label.setObjectName("clockLabel")

        # Three zones: who (left), the voice controls (centre), where to go
        # and the time (right). Everything used to sit at the two ends of one
        # stretch, so the name and the greeting bunched up on the left and a
        # third of the header stood empty before the first button. The two
        # side zones take equal shares of the width, which keeps the voice
        # controls in the true middle of the header.
        left_zone = QHBoxLayout()
        left_zone.setSpacing(0)
        left_zone.addLayout(text_box)
        left_zone.addStretch(1)

        voice_zone = QHBoxLayout()
        voice_zone.setSpacing(SPACE.SM)
        voice_zone.addWidget(self.pause_btn)
        voice_zone.addWidget(self.stop_btn)
        voice_zone.addWidget(self.mute_btn)
        voice_zone.addSpacing(SPACE.SM)
        voice_zone.addWidget(self.form_toggle)
        voice_zone.addWidget(self.compact_btn)

        # Spacing set on the row rather than spacers between the items, so a
        # label _fit_header hides takes its gap with it.
        right_zone = QHBoxLayout()
        right_zone.setSpacing(SPACE.MD)
        right_zone.addStretch(1)
        right_zone.addWidget(self.control_cluster)
        right_zone.addSpacing(SPACE.SM)
        right_zone.addWidget(self.state_label)
        right_zone.addWidget(self.clock_label)
        right_zone.addWidget(self.quit_btn)

        layout.addLayout(left_zone, 1)
        layout.addSpacing(SPACE.LG)
        layout.addLayout(voice_zone, 0)
        layout.addSpacing(SPACE.LG)
        layout.addLayout(right_zone, 1)
        # The two gaps either side of the voice controls: _fit_header closes
        # them up before it starts hiding things on a narrow window.
        self._header_gaps = [layout.itemAt(i).spacerItem()
                             for i in (1, 3)]
        # The header must never set the window's minimum width: Qt's answer
        # to a row wider than its window was to grow the window past the
        # screen, or to shave every button ("Vision Lab" to "ision La").
        # _fit_header hides what the row cannot hold instead. (An explicit
        # minimum also stops the layout imposing its own, so the height is
        # pinned to the row's natural height by the size policy.)
        frame.setMinimumWidth(SPACE.LG * 10)
        frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self._tick_clock)
        self.clock_timer.start()
        return frame

    def _tick_clock(self) -> None:
        """Header clock — idle while this window is hidden (the overlay
        toggle hides the whole window rather than closing it)."""
        if not self.isVisible():
            return
        now = datetime.now()
        self.clock_label.setText(_clock_words(now))
        greeting = getattr(self, "greeting_label", None)
        if greeting is not None:
            identity = self._identity()
            text = _greeting_words(now, identity.user_name() if identity else "")
            if greeting.text() != text:
                greeting.setText(text)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        # After the header has its new width, not before it.
        QTimer.singleShot(0, self._fit_header)

    def _header_give_way(self) -> list[tuple[str, list[Any]]]:
        """What the header gives up on a narrow window, first to last.

        A hello is the one thing in the row nobody needs, so it goes first;
        then the gaps round the voice controls close up; then the clock (the
        taskbar has one), the state words (the face shows his state) and the
        two shortcuts the deck's sidebar also has. The voice controls, the
        tools and Shut Down never give way.
        """
        return [
            ("greeting", [self.greeting_label]),
            ("gaps", []),
            ("clock", [self.clock_label]),
            ("state", [self.state_label]),
            ("shortcuts", [self.workbench_btn, self.research_btn]),
        ]

    def _fit_header(self) -> None:
        """Fit the header row to the window by giving things up in order.

        The row carries the wordmark, the Mark tag, the greeting, the voice
        controls, the shortcut cluster, the state and the clock - about
        1500 px written out in Title Case with the gaps between its three
        zones. Qt answers a row that is too wide by shaving every button,
        which clipped "Vision Lab" to "ision La" and then to an empty pill,
        so instead the row gives things up (_header_give_way) until what is
        left fits, and takes them back as the window widens.
        """
        greeting = getattr(self, "greeting_label", None)
        header = getattr(self, "header_frame", None)
        layout = header.layout() if header is not None else None
        if greeting is None or layout is None:
            return
        steps = self._header_give_way()
        gaps = [gap for gap in getattr(self, "_header_gaps", []) if gap is not None]

        def give_up(upto: int) -> None:
            for index, (name, widgets) in enumerate(steps):
                keep = index >= upto
                if name == "gaps":
                    for gap in gaps:
                        gap.changeSize(SPACE.LG if keep else SPACE.XS, 0)
                    side = SPACE.LG if keep else SPACE.MD
                    margins = layout.contentsMargins()
                    if margins.left() != side:
                        layout.setContentsMargins(side, margins.top(), side, margins.bottom())
                    continue
                for widget in widgets:
                    if widget.isHidden() == keep:
                        widget.setVisible(keep)

        available = header.width()
        for upto in range(len(steps) + 1):
            give_up(upto)
            layout.invalidate()
            layout.activate()
            if layout.minimumSize().width() <= available:
                break

    # ── display modes ─────────────────────────────────────────────────────────

    def toggle_fullscreen(self) -> None:
        """F11 — immersive full-monitor mode (exits overlay first if active)."""
        if self._overlay_active:
            self.exit_overlay_mode()
        if self.isFullScreen():
            self.showNormal()
            self.bus.log.emit("SYS: fullscreen disengaged.")
        else:
            self.showFullScreen()
            self.bus.log.emit("SYS: fullscreen engaged - press F11 or Esc to exit.")

    def _toggle_deck_page(self, page: str, label: str) -> None:
        """Show/hide the Command Deck on a named page.

        dashboard/command_centre/mission are three different KEYBOARD ENTRY
        POINTS into the exact same swipeable UnifiedDashboard instance
        (attach_dashboard and attach_command_centre both point at it, see
        app.py) — one shared implementation so toggle_dashboard/
        toggle_command_centre/toggle_mission_deck can never drift from each
        other, and so re-pressing a different shortcut while the deck is
        already open always jumps straight to that shortcut's page rather
        than just raising whatever page happens to be showing.
        """
        deck = self.dashboard or self.command_centre
        if deck is None:
            self.bus.log.emit("SYS: Command Deck not attached.")
            return
        if self.deck_is_inline:
            # Unified shell: the deck is a panel in this window, so "toggle"
            # means collapse/expand the panel — never show()/hide() on what
            # would otherwise be a top-level window, which in embedded mode
            # would either do nothing or briefly tear the deck out of the
            # splitter. isHidden() rather than isVisible() for the same reason
            # as the swarm rail: isVisible() also reports False whenever this
            # whole window is minimised, which would invert the toggle.
            if self.deck_panel.isHidden():
                self.deck_panel.show()
            if hasattr(deck, "show_page_named"):
                deck.show_page_named(page)
            self._collapse_rail_for_deck(page)
            self.bus.log.emit(f"SYS: Command Deck → {label}.")
            return
        if hasattr(deck, "show_page_named") and deck.isVisible():
            # A separate deck window that is MINIMISED still reports isVisible()
            # True, and raise_() alone will not restore it — so the button looked
            # dead (it navigated a window the user couldn't see). Un-minimise
            # first, then bring it forward, so pressing a deck button always has
            # a visible effect even on a second monitor.
            if getattr(deck, "isMinimized", lambda: False)():
                deck.showNormal()
            deck.show_page_named(page)
            self._collapse_rail_for_deck(page)
            deck.raise_()
            deck.activateWindow()
            self.bus.log.emit(f"SYS: Command Deck → {label}.")
            return
        if deck.isVisible():
            deck.hide()
        else:
            deck.show()
            if hasattr(deck, "show_page_named"):
                deck.show_page_named(page)
            self._collapse_rail_for_deck(page)
            deck.raise_()
            deck.activateWindow()
        self.bus.log.emit(f"SYS: Command Deck → {label}.")

    def toggle_dashboard(self) -> None:
        """Ctrl+D — show/hide the Command Deck on the Widgets page."""
        self._toggle_deck_page("WIDGETS", "Widgets")

    def toggle_mission_deck(self) -> None:
        """Ctrl+M — show/hide the Command Deck on the Mission page."""
        self._toggle_deck_page("MISSION", "Mission")

    def open_electronics_workbench(self) -> None:
        """Open the camera lab without activating any capture hardware."""
        self._toggle_deck_page("WORKBENCH", "Vision Lab")

    def open_camera_lab_live(self) -> bool:
        """Open the camera lab AND start the live feed.

        "Show me the camera" and "what can you see" are different requests and
        only the second one was reachable: asking to see the camera ran the
        vision tool, which grabs one frame, analyses it and describes it in
        words. That answers a question the user did not ask.

        This is the other one — the lab open, the feed running, ORION's view
        on screen. It starts hardware, so it is deliberately separate from the
        toolbar button, which opens the page and leaves the camera alone.

        Returns whether the feed actually started.
        """
        self._toggle_deck_page("WORKBENCH", "Vision Lab")
        workbench = self._deck_page_widget("WORKBENCH")
        if workbench is None or not hasattr(workbench, "start_camera"):
            self.bus.log.emit("GUI: the camera lab is not available.")
            return False
        try:
            workbench.start_camera()
        except Exception as exc:
            self.bus.log.emit(f"GUI: the camera would not start - {exc}")
            return False
        # start_camera() returns None whether it opened a device or gave up —
        # it reports "CAMERA UNAVAILABLE" on its own canvas and returns. So the
        # only honest answer comes from the flag it sets on the way through;
        # returning True regardless would be this method claiming a success it
        # has no evidence for.
        started = bool(getattr(workbench, "_camera_enabled", False))
        if not started:
            self.bus.log.emit(
                "GUI: the camera lab is open but no camera started — "
                "check the camera service.")
        return started

    def _deck_page_widget(self, name: str) -> Any:
        """The widget behind a deck page, or None. The deck owns the page list;
        this window only ever borrows from it.

        Goes through page_widget() so a page that has not been opened yet is
        BUILT rather than handed back as its placeholder — starting the camera
        on a placeholder would silently do nothing.
        """
        deck = getattr(self, "dashboard", None) or getattr(self, "command_centre", None)
        if deck is None:
            return None
        getter = getattr(deck, "page_widget", None)
        if callable(getter):
            try:
                return getter(name)
            except Exception:
                pass
        for page_name, widget in getattr(deck, "_pages", None) or []:
            if page_name.upper() == name.upper():
                return widget
        return None

    def _face_draws_itself(self) -> bool:
        """True when this window's face renders ORION on its own.

        The Three.js avatar and the 2-D rig both do; NativeFacePanel does not,
        because it is an adapter that has to be handed a scene renderer."""
        return not hasattr(getattr(self, "face", None), "attach_renderer")

    def _face_is_webengine(self) -> bool:
        """True when this window's face is the Three.js avatar, which cannot
        survive a QOpenGLWidget appearing beside it."""
        return type(getattr(self, "face", None)).__name__ in {"QuantumFace3D", "RemoteFace"}

    #: Deck pages that must not share the screen with the swarm rail.
    #:
    #: SWARM was the original reason — the deck's page and this rail rendered
    #: the same graph from the same backends, so opening one beside the other
    #: showed two identical swarms.
    #:
    #: BRAIN inherits the rule for a different and more important reason.
    #: It replaced the SWARM page and is a full WebGL scene with its own
    #: Chromium render process; the rail is another. Leaving both live is
    #: precisely the GPU contention that starved the globe's Cesium context
    #: and broke it. One heavy scene at a time.
    _RAIL_CONFLICT_PAGES = ("SWARM", "BRAIN")

    def _collapse_rail_for_deck(self, page: str) -> None:
        """Never leave two heavy 3-D surfaces on screen at once.

        Opening the deck on any page in _RAIL_CONFLICT_PAGES collapses the
        rail. Every other page (CHESS, LOG, …) is a perfectly reasonable
        thing to have open beside it, so the rail is left alone for those.
        """
        upper = str(page or "").upper()
        if not any(name in upper for name in self._RAIL_CONFLICT_PAGES):
            return
        if not self.swarm_rail.isHidden():
            self.swarm_rail.hide()
            self.bus.log.emit("SYS: swarm rail collapsed — the deck is showing "
                              "the same graph (Ctrl+Shift+S to bring it back).")

    def toggle_swarm_rail(self) -> None:
        """Retired in Mark XXXI, with the "neural subsystem map" it showed.

        The rail put a second 3-D scene beside the face — the same live
        architecture the Command Deck's BRAIN page draws — and a GL surface in
        this window takes the WebGL face down besides. Asking for it now opens
        the BRAIN page instead, so the request still lands somewhere useful.
        """
        if not self.swarm_rail.isHidden():
            self.swarm_rail.hide()
        self.bus.log.emit("SYS: the subsystem map was retired — ORION's live "
                          "architecture is on the Command Deck's BRAIN page.")
        self._open_deck_page("BRAIN")

    def _toggle_pause(self) -> None:
        """Pause/resume ORION from the header button."""
        if self.worker is not None:
            self.worker.toggle_pause()
        else:
            self.bus.log.emit("SYS: worker not ready for pause.")

    def _stop_turn(self) -> None:
        """Cancel ORION's current request outright so the user can rephrase."""
        if self.worker is not None and hasattr(self.worker, "cancel_turn"):
            self.worker.cancel_turn()
        else:
            self.bus.log.emit("SYS: worker not ready to cancel.")

    # ── autonomous self-GUI navigation ────────────────────────────────────────

    def _open_deck_page(self, name: str) -> None:
        """Show the swipeable deck on a named page (widgets/toolkit/studio/
        command/globe/diagnostics).  The dashboard and command centre share one
        deck, so either reference works."""
        deck = getattr(self, "dashboard", None) or getattr(self, "command_centre", None)
        if deck is None:
            self.bus.log.emit("GUI: deck not attached.")
            return
        if self.deck_is_inline:
            # Embedded: bring THIS window forward, not the deck — raising an
            # embedded child is meaningless and activateWindow() on it would
            # target the wrong top-level.
            if self.deck_panel.isHidden():
                self.deck_panel.show()
            if hasattr(deck, "show_page_named") and name:
                if not deck.show_page_named(name):
                    self.bus.log.emit(f"GUI: no deck page matches {name!r}.")
            self._collapse_rail_for_deck(name)
            self.raise_()
            self.activateWindow()
            return
        if not deck.isVisible():
            deck.show()
        if hasattr(deck, "show_page_named") and name:
            deck.show_page_named(name)
        self._collapse_rail_for_deck(name)
        deck.raise_()
        deck.activateWindow()

    def _deck_has_page(self, name: str) -> bool:
        """Whether the deck has a page by this name."""
        deck = getattr(self, "dashboard", None) or getattr(self, "command_centre", None)
        wanted = str(name or "").strip().lower()
        if deck is None or not wanted or not hasattr(deck, "page_names"):
            return False
        try:
            return any(wanted in str(page).lower() for page in deck.page_names())
        except Exception:
            return False

    def _open_deck_zone(self, zone: str) -> None:
        """Navigate the Command Deck to a ZONE, given its name.

        The bridge that turns the swarm into ORION's navigation surface:
        opening a cluster node in the graph lands on that zone's first real
        page. Zone-to-page resolution is asked of the deck itself
        (ZONE_PAGES), never duplicated here, so the two can never disagree
        about which pages a zone owns."""
        deck = getattr(self, "dashboard", None) or getattr(self, "command_centre", None)
        if deck is None:
            self.bus.log.emit("GUI: deck not attached.")
            return
        zone_name = str(zone or "").strip().upper()
        pages = getattr(deck, "ZONE_PAGES", {}).get(zone_name, ())
        if not pages:
            # An empty zone is a real state (see UnifiedDashboard's own notes),
            # so say so rather than silently doing nothing.
            self.bus.log.emit(f"GUI: zone '{zone_name}' has no pages yet.")
            return
        self._open_deck_page(pages[0])
        self.bus.log.emit(f"GUI: swarm → {zone_name} zone ({pages[0]}).")

    def _deck_query(self, payload: dict) -> dict:
        """Answer interface_control's deck read / find / press request with a
        receipt (gui/deck_reader.py). On the GUI thread, like every widget it
        touches. A press is carried out AFTER the receipt goes back: a button
        that opens a dialog must not hold the tool call open behind it."""
        receipt: dict[str, Any] = {"request_id": payload.get("request_id"), "ok": False}
        deck = getattr(self, "dashboard", None) or getattr(self, "command_centre", None)
        if deck is None or not hasattr(deck, "page_widget"):
            receipt["error"] = "The Command Deck is not open yet."
            return receipt
        from . import deck_reader

        query = str(payload.get("query") or "")
        target = str(payload.get("target") or "").strip()
        page_word = str(payload.get("page") or "").strip()
        page = deck.resolve_page_name(page_word) if page_word else None
        if page_word and page is None:
            receipt["error"] = (f"There is no deck page matching {page_word!r}. Pages: "
                                + ", ".join(deck.page_names()) + ".")
            return receipt
        if query == "pages":
            current = deck.current_page_name()
            names = [f"{deck.display_name(n)}{' (on screen now)' if n == current else ''}"
                     for n in deck.page_names()]
            receipt.update(ok=True, text="Command Deck pages: " + ", ".join(names) + ".")
        elif query == "read":
            name = page or deck.current_page_name()
            if not name:
                receipt["error"] = "No Command Deck page is on screen; name one to read."
                return receipt
            self._open_deck_page(name)
            ok, text = deck_reader.read_page(deck, name)
            receipt.update(ok=ok, text=text, page=name)
        elif query == "find":
            ok, text, best = deck_reader.find(deck, target)
            if best:
                self._open_deck_page(best)
            receipt.update(ok=ok, text=text, page=best)
        elif query == "press":
            ok, text, button, name = deck_reader.locate(deck, target, page)
            receipt.update(ok=ok, text=text, page=name)
            if ok and button is not None:
                self._open_deck_page(name)
                QTimer.singleShot(0, button.click)
                self.bus.log.emit(f"GUI: pressed {target!r} on {name}.")
        else:
            receipt["error"] = f"Unknown deck request {query!r}."
        if not receipt["ok"] and "error" not in receipt:
            receipt["error"] = receipt.get("text") or "That did not work."
        return receipt

    def _on_gui_command(self, payload: Any) -> None:
        """Handle a self-navigation request from ORION (emitted on the bus by
        the interface_control tool).  Runs on the GUI thread."""
        try:
            if not isinstance(payload, dict):
                return
            action = str(payload.get("action") or "").lower().strip()
            target = str(payload.get("target") or "").strip()
            low    = target.lower()
            if action == "presentation":
                try:
                    self._apply_presentation(payload.get("presentation") or {})
                    receipt = {"request_id": payload.get("request_id"), "ok": True}
                except Exception as exc:
                    receipt = {"request_id": payload.get("request_id"), "ok": False,
                               "error": str(exc)[:200]}
                self.bus.gui_result.emit(receipt)
                return
            if action == "deck_query":
                try:
                    receipt = self._deck_query(payload)
                except Exception as exc:
                    receipt = {"request_id": payload.get("request_id"), "ok": False,
                               "error": f"The deck could not be read - {str(exc)[:160]}"}
                self.bus.gui_result.emit(receipt)
                return
            # Gemini sometimes emits a generic Command Centre navigation as a
            # trailing companion call after the chess tool.  A successful chess
            # handoff is authoritative for a brief window: do not let that
            # stale generic navigation cover the board it just opened.
            generic_command_centre = (
                action in {"command", "command_centre", "centre"}
                or (action in {"page", "deck", "show_page", "open_page", "open"}
                    and low in {"command", "command centre", "command_centre"})
            )
            if (generic_command_centre
                    and time.monotonic() < getattr(self, "_chess_navigation_until", 0.0)):
                self.bus.log.emit("GUI: ignored stale Command Centre navigation after chess handoff.")
                return
            if action in {"view", "show_view", "select_view"}:
                if low in {"log", "memory", "telemetry"}:
                    self._open_deck_page(low)
                else:
                    # Face is the only thing left on this window — nothing to
                    # switch to, so just bring it to the front.
                    self.raise_()
                    self.activateWindow()
            elif action in {"page", "deck", "show_page", "open_page", "open"}:
                self._open_deck_page(target)
            elif action in {"dashboard", "widgets"}:
                self._open_deck_page("widgets")
            elif action in {"toolkit"}:
                self._open_deck_page("toolkit")
            elif action in {"research", "research_console"}:
                self._open_deck_page("RESEARCH")
            elif action in {"command_centre", "centre", "command"}:
                self._open_deck_page("command")
            elif action in {"face_form", "appearance", "form"}:
                applied = self.set_face_form(target)
                self._sync_form_menu()
                if not applied:
                    from ..appearance import SCREEN_ONLY_REPLY, form_unavailable
                    self.bus.log.emit(
                        f"GUI: {SCREEN_ONLY_REPLY}" if form_unavailable(target) else
                        f"GUI: {target!r} is not one of ORION's forms (Screen Face).")
            elif action in {"face_design", "design"}:
                # Mark XXXIX: the Screen Face's designs (appearance.FACE_DESIGNS).
                from ..appearance import design_label, set_face_design
                from .screen_face import apply_design
                chosen = set_face_design(target)
                if chosen:
                    apply_design(chosen)
                    self.bus.log.emit(f"GUI: ORION's face is now the {design_label(chosen)} design.")
                else:
                    self.bus.log.emit(f"GUI: {target!r} is not one of ORION's face designs.")
            elif action in {"show_dock", "hide_dock", "dock"}:
                # The docked Screen Face is not on screen at start: this is how
                # it is asked for. app._start_windows_integration installs the
                # control; without it (a bare window in a test) there is none.
                control = getattr(self, "_dock_control", None)
                if callable(control):
                    control(False if action == "hide_dock" or low in {
                        "hide", "off", "away", "close"} else True)
                else:
                    self.bus.log.emit("GUI: the dock is not available yet.")
            elif action in {"camera", "camera_lab", "webcam", "live_camera"}:
                # Deliberately NOT the vision tool: that captures one frame and
                # describes it. This shows the live feed.
                self.open_camera_lab_live()
            elif action == "chess":
                self._chess_navigation_until = time.monotonic() + 3.0
                self._open_deck_page("CHESS")
            elif action in {"diagnostics"}:
                self._open_deck_page("diagnostics")
            elif action in {"preferences", "settings", "options"}:
                self.open_preferences()
            elif action in {"what_you_remember", "stored_memory",
                            "memory_viewer", "forget_something"}:
                self.open_memory_viewer()
            elif action in {"results", "findings", "what_you_found",
                            "content", "articles"}:
                self.open_content_panel()
            elif action in {"clipboard", "clipboard_on", "watch_clipboard"}:
                self.enable_clipboard_intelligence(True)
            elif action in {"clipboard_off", "stop_clipboard"}:
                self.enable_clipboard_intelligence(False)
            elif action == "globe":
                self._open_deck_page("globe")
                if target:
                    self.bus.globe_request.emit(target)
            elif action in {"refresh_environment", "refresh"}:
                panel = self.environment_panel
                if panel is not None:
                    background.spawn(panel.refresh_environment_widgets())
                else:
                    self.bus.log.emit("GUI: environment panel not attached.")
            elif action in {"overlay", "compact", "compact_orb", "minimise_to_orb",
                            "compact_face", "minimise_to_face", "mini_face"}:
                self.toggle_overlay_mode()
            elif action == "standby":
                if not self._overlay_active:
                    self.enter_overlay_mode()
            elif action in {"wake", "resume_standby"}:
                if self._overlay_active:
                    self.exit_overlay_mode()
            elif action in {"swarm"}:
                self.toggle_swarm_rail()
            elif action in {"fullscreen"}:
                self.toggle_fullscreen()
            elif action in {"pause"}:
                self._toggle_pause()
            elif action in {"stop", "cancel"}:
                self._stop_turn()
            elif self._deck_has_page(action):
                # "Take me to chess" arrives as action="chess" with no target,
                # and chess is not one of the named actions above — so it fell
                # through to "unknown" and the deck opened on whatever page it
                # was already showing. Any action that names a real page IS a
                # navigation request.
                self._open_deck_page(action)
            else:
                self.bus.log.emit(f"GUI: unknown interface action '{action}'.")
                return
            self.bus.log.emit(f"GUI: {action}{(' → ' + target) if target else ''}.")
        except Exception as exc:
            self.bus.log.emit(f"GUI: interface command failed - {exc}")

    def _on_paused(self, paused: bool) -> None:
        self.pause_btn.setText("Resume" if paused else "Pause")
        from .line_icons import apply
        apply(self.pause_btn, "play" if paused else "pause")

    def _apply_presentation(self, payload: dict) -> None:
        """Apply presentation controls on the GUI thread with an acknowledgement."""
        from ..constants import C
        operation = payload.get("operation")
        hud = getattr(self, "hud", None)
        if operation == "display":
            panel = self._ensure_content_panel()
            if panel is None:
                raise ValueError("The content display is unavailable.")
            panel.show_presentation(payload)
            panel.show()
        elif operation == "ui_theme":
            if not self.apply_accent(payload["accent"]):
                raise ValueError("That accent has insufficient contrast. Choose a brighter colour.")
        elif operation in {"ui_reactor", "ui_orbit"}:
            # The reactor and the image orbit live on the 3-D orb. The screen
            # face has neither, so switching to the orb comes FIRST and the
            # renderer is asked again afterwards.
            self.set_face_form("orb")
            apply = getattr(self.face, "set_presentation", None)
            if not callable(apply):
                from ..appearance import SCREEN_ONLY_REPLY, form_enabled
                raise ValueError(
                    "These orb controls need the 3-D orb renderer, "
                    "which is not available on this machine."
                    if form_enabled("orb") else
                    f"These orb controls are not part of this version. {SCREEN_ONLY_REPLY}")
            apply(payload)
        elif operation == "ui_chrome":
            if hud is None:
                raise ValueError("The desktop HUD is unavailable.")
            for name in ("telemetry", "transcript"):
                if name in payload:
                    getattr(hud, "rail" if name == "telemetry" else "log").setVisible(payload[name])
            timer = getattr(self, "_chrome_timer", None)
            if timer is None:
                timer = self._chrome_timer = QTimer(hud.rail)
                timer.setSingleShot(True)
                timer.timeout.connect(lambda: (hud.rail.show(), hud.log.show()))
            timer.start(30_000)
        elif operation == "ui_effect":
            from ..appearance import motion_profile
            apply = getattr(self.face, "set_presentation", None)
            if callable(apply) and motion_profile().name != "reduced":
                apply(payload)
            else:
                self.bus.banner.emit("Display updated", 1500)
        elif operation == "ui_screen":
            if payload["target"] in {"panels", "all"}:
                panel = getattr(self, "_content_panel", None)
                if panel is not None:
                    panel.clear()
            if payload["target"] in {"transcript", "all"} and hud is not None:
                hud.log.clear()
        elif operation == "ui_reset":
            self.apply_accent(C.PRI)
            timer = getattr(self, "_chrome_timer", None)
            if timer is not None:
                timer.stop()
            if hud is not None:
                hud.rail.show()
                hud.log.show()
            apply = getattr(self.face, "set_presentation", None)
            if callable(apply):
                apply(payload)
        else:
            raise ValueError("Unknown presentation operation.")

    def toggle_command_centre(self) -> None:
        """Ctrl+Shift+C — show the Command Deck on the Command Centre page."""
        self._toggle_deck_page("COMMAND", "Command Centre")

    def open_command_palette(self) -> None:
        """Ctrl+K — fuzzy-search every dispatcher tool AND every Command Deck
        page in one list (Mark XX design-spec §8: before this, Ctrl+K only
        indexed the ~100 dispatcher tools, so a user who half-remembered a
        page name — "the chess thing", "the ROAS calculator" — had no search
        path once it left the tab bar). Choosing a tool prefills the message
        box (never auto-fires it; the user still sends); choosing a page
        navigates the deck straight there."""
        # Imported lazily so the heavy dispatcher module isn't pulled in at GUI
        # construction time (and to keep the strict-downward import graph clean).
        from ..dispatcher import TOOL_DECLARATIONS
        from .command_palette import (
            CommandPalette, build_action_commands, build_commands,
            build_page_commands,
        )

        commands = []
        # Real, executable commands come FIRST — these are the entries that
        # actually do the thing they are named after (brief §20).
        router = getattr(self, "command_router", None)
        if router is not None:
            commands += build_action_commands(router.palette_entries())
        deck = getattr(self, "dashboard", None) or getattr(self, "command_centre", None)
        if deck is not None and hasattr(deck, "page_names"):
            commands += build_page_commands(deck.page_names())
        commands += build_commands(TOOL_DECLARATIONS)

        palette = CommandPalette(commands, self)
        palette.command_chosen.connect(self._on_palette_command_chosen)
        palette.exec()

    def attach_command_router(self, router: Any) -> None:
        """Give the palette a router so its entries can actually execute."""
        self.command_router = router

    def _on_palette_command_chosen(self, kind: str, name: str) -> None:
        from .command_palette import KIND_ACTION, KIND_PAGE
        if kind == KIND_PAGE:
            self._open_deck_page(name)
        elif kind == KIND_ACTION:
            self._run_palette_action(name)
        else:
            self._prefill_command(name)

    def _run_palette_action(self, command_id: str) -> None:
        """Execute a palette command on ORION's event loop.

        The GUI thread never awaits and never touches asyncio directly — it
        hands the coroutine to the loop and returns immediately, so a command
        that takes ten seconds cannot freeze the interface.
        """
        from ..command_router import run_soon
        router = getattr(self, "command_router", None)
        if router is None:
            self.bus.log.emit(f"SYS: no command router; '{command_id}' not run.")
            return
        command = router.get(command_id)
        if command is None:
            self.bus.log.emit(f"SYS: unknown command '{command_id}'.")
            return
        if command.destructive and not self._confirm_command(command):
            self.bus.log.emit(f"SYS: '{command_id}' cancelled by the user.")
            return
        self.bus.banner.emit(f"RUNNING: {command.title.upper()}", 2)
        worker = getattr(self, "worker", None)
        loop = getattr(worker, "_loop", None) if worker is not None else None
        if loop is None:
            try:
                import asyncio
                loop = asyncio.get_running_loop()   # qasync: running in Qt slots
            except RuntimeError:
                loop = None
        run_soon(router.run(command_id), loop)

    def _confirm_command(self, command: Any) -> bool:
        """Ask before anything that can change something."""
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Confirm Command",
            f"{command.title}\n\n{command.subtitle}\n\n"
            f"Subsystem: {command.subsystem}\n\nRun it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return reply == QMessageBox.StandardButton.Yes

    def _prefill_command(self, name: str) -> None:
        line = getattr(self.log_view, "input_line", None) if self.log_view is not None else None
        if line is None:
            self.bus.log.emit(f"SYS: command palette chose '{name}' (no input to prefill).")
            return
        self._open_deck_page("log")
        line.setText(name)
        line.setFocus()

    # ── the compact orb ─────────────────────────────────────────────────────

    def _ensure_orb_overlay(self) -> Any:
        """The floating orb window, built on first use and then kept.

        A SEPARATE top-level window (gui/orb_overlay.py). This window is only
        hidden while it is up — never re-flagged — so the face in it is never
        destroyed, never reloaded and never loses its WebGL context. See that
        module for why changing this window's flags was the root of every
        compact-overlay fault.
        """
        overlay = getattr(self, "_orb_overlay", None)
        if overlay is not None:
            return overlay
        from .orb_overlay import OrbOverlay

        # Parented to this window so it lives and dies with it (a parentless
        # top-level outliving its owner is a crash at exit), but flagged as a
        # window of its own: hiding this one leaves the orb on screen —
        # measured on the Windows platform, not assumed.
        overlay = OrbOverlay(self)
        overlay.restore_requested.connect(self.exit_overlay_mode)
        overlay.quit_requested.connect(self._quit)
        overlay.set_state(getattr(self, "face_state", "STANDBY"))
        # State and loudness, and - for the Screen Face it now shows - speech,
        # emotion and lip-sync, so the small face is as alive as the big one.
        overlay.connect_bus(self.bus)
        self._orb_overlay = overlay
        return overlay

    def toggle_overlay_mode(self) -> None:
        if self._overlay_active:
            self.exit_overlay_mode()
        else:
            self.enter_overlay_mode()

    def enter_overlay_mode(self) -> None:
        """Shrink ORION to his floating face (the orb, for the orb or classic
        face). He keeps listening and working.

        The main window is hidden, not changed, and remembered exactly as it
        was (maximised, full-screen or its normal geometry) for the way back.
        """
        if self._overlay_active:
            return
        if self._orb_hands_enabled:
            self.set_orb_hand_control(False)
        self._overlay_active = True
        self._overlay_restore = ("fullscreen" if self.isFullScreen()
                                 else "maximised" if self.isMaximized()
                                 else "normal")
        self._saved_geometry = self.saveGeometry()
        screen = self.screen() or QGuiApplication.primaryScreen()
        try:
            overlay = self._ensure_orb_overlay()
            overlay.place_on(screen)
            overlay.show()
            overlay.raise_()
            # Focus goes with him, so Esc restores straight away — the user
            # has just pressed Compact in this window, so nothing else loses it.
            overlay.activateWindow()
        except Exception as exc:
            # No orb means no way back from a hidden window: stay as we are.
            self._overlay_active = False
            self.bus.log.emit(f"SYS: the compact face could not open - {exc}")
            return
        self.hide()
        self.bus.log.emit("SYS: compact face engaged - double-click it, press Esc, "
                          "use the ⤢ chip or Ctrl+Shift+O to restore.")

    def exit_overlay_mode(self) -> None:
        """Bring the full window back exactly as it was left."""
        if not self._overlay_active:
            return
        self._overlay_active = False
        overlay = getattr(self, "_orb_overlay", None)
        if overlay is not None:
            overlay.hide()
        if self._saved_geometry is not None:
            self.restoreGeometry(self._saved_geometry)
        mode = getattr(self, "_overlay_restore", "normal")
        if mode == "fullscreen":
            self.showFullScreen()
        elif mode == "maximised":
            self.showMaximized()
        else:
            self.showNormal()
        self.raise_()
        self.activateWindow()
        self.bus.log.emit("SYS: compact face disengaged.")

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Escape:
            if self._overlay_active:
                self.exit_overlay_mode()
                return
            if self.isFullScreen():
                self.showNormal()
                return
        super().keyPressEvent(event)

    # ── shutdown ────────────────────────────────────────────────────────────────

    def _toggle_voice_mute(self, *_args: Any) -> None:
        self.bus.voice_mute_request.emit(not self._voice_muted)

    def _on_voice_muted(self, muted: bool) -> None:
        """Keep the button honest whichever way the mute happened (button,
        shortcut or spoken)."""
        self._voice_muted = bool(muted)
        button = getattr(self, "mute_btn", None)
        if button is None:
            return
        button.setChecked(self._voice_muted)
        button.setText("Unmute" if self._voice_muted else "Mute")
        button.setAccessibleName("Unmute ORION's voice" if self._voice_muted
                                 else "Mute ORION's voice")

    def _quit(self) -> None:
        """Shut ORION down completely — every window and background task."""
        self.bus.log.emit("SYS: shutdown requested by user.")
        self.bus.request_shutdown.emit()

    def _tray_active(self) -> bool:
        tray = getattr(self, "_tray", None)
        return bool(tray is not None and getattr(tray, "active", False))

    def _mark_shutting_down(self) -> None:
        self._shutting_down = True
        self._cancel_background_plugin_prompts()

    def closeEvent(self, event: Any) -> None:
        """
        Closing the core window shuts the whole assistant down — UNLESS ORION is
        resident in the system tray, in which case closing merely HIDES him to
        the tray so he stays available like a real OS assistant. An explicit
        Quit (the ⏻ button or the tray's "Quit ORION") sets ``_shutting_down``
        first, so it always brings the whole assistant down cleanly.

        Without either behaviour ORION would keep running headless after the
        main window closed — which is why `py orion.py` from a terminal once
        appeared to 'never quit'.
        """
        self._cancel_background_plugin_prompts()
        if self._orb_hands_enabled:
            self.set_orb_hand_control(False)
        if not getattr(self, "_shutting_down", False) and self._tray_active():
            # Minimise to the tray instead of quitting.
            event.ignore()
            self.hide()
            tray = getattr(self, "_tray", None)
            notify = getattr(tray, "_on_safety", None)
            if notify is not None and not getattr(self, "_tray_hint_shown", False):
                self._tray_hint_shown = True
                try:
                    tray._tray.showMessage(
                        APP_NAME,
                        "Still here — I'm in the tray. Right-click me to return "
                        "or quit.", tray._tray.MessageIcon.Information, 5000)
                except Exception:
                    pass
            return
        self.bus.request_shutdown.emit()
        super().closeEvent(event)

    # ── bus connections ───────────────────────────────────────────────────────

    def _connect_bus(self) -> None:
        # bus.log is already connected inside LogConsoleView.__init__;
        # connecting it here as well duplicated every console line.
        # Any shutdown request (quit button, tray "Quit", spoken "shut down")
        # marks a real quit so closeEvent stops hiding-to-tray and lets it happen.
        self._shutting_down = False
        self.bus.request_shutdown.connect(self._mark_shutting_down)
        self.bus.state.connect(self.set_state)
        # Through the window, never bound straight to the face object: the
        # face has to be REBUILT whenever overlay mode changes the window
        # flags (see _rebuild_face), and connections bound to the old object
        # would keep driving a destroyed widget.
        self.bus.state.connect(self._face_set_state)
        self.bus.amplitude.connect(self._face_set_amplitude)
        self.bus.banner.connect(self._on_banner)
        # What ORION found, on its way to a surface it can be clicked from.
        self.bus.content_results.connect(self._on_content_results)
        self.bus.speaking.connect(self._face_set_speaking)
        self.bus.viseme.connect(self._face_set_viseme)     # Mark XXVI lip-sync
        # Queued: a "stopped" sample is emitted on this thread by the
        # gesture_control tool (inside its task), and handling it stops the
        # orb camera tracker (a thread join of up to 2 s) and re-enters
        # set_orb_hand_control while it is still running. Frame samples
        # already arrive queued from the gesture thread.
        self.bus.orb_gesture.connect(self._on_orb_gesture, Qt.ConnectionType.QueuedConnection)
        self.bus.speaking.connect(self._on_speaking_changed)
        self.bus.paused.connect(self._on_paused)
        # Self-GUI navigation: ORION drives his own interface over the bus
        # (queued connection → runs on the GUI thread, safe from the worker).
        # Queued, never inline. Tools emit this from inside an asyncio task on
        # this thread, and a handler that waits on Windows (opening a camera
        # through Media Foundation, the shell opening a folder) pumps the
        # message queue: qasync then steps OTHER tasks inside the running one,
        # "Cannot enter into task", and those tasks never resume. On 8 Oct the
        # camera opened that way and the Live receive loop died silently -
        # ten minutes of no transcripts and no replies. REPRODUCED: inline,
        # a ticking task stopped for good; queued, it kept ticking (49/s).
        self.bus.gui_command.connect(self._on_gui_command, Qt.ConnectionType.QueuedConnection)
        # Destructive-action confirmation (Section 10): a guarded OS power action
        # or file deletion surfaces here as an accessible dialog; only the user's
        # approval releases it, via the deterministic dispatcher callback.
        # Queued for the same reason as gui_command: the dialog runs a modal
        # loop (box.exec()), and it is emitted from inside the voice task.
        self.bus.confirm_action.connect(self._on_confirm_action, Qt.ConnectionType.QueuedConnection)
        # Mark X.7: the EmotionStateManager broadcasts full rendering
        # parameter sets; the face morphs to whatever arrives.
        self.bus.emotion_changed.connect(self._face_apply_emotion)
        self.bus.voice_muted.connect(self._on_voice_muted)
        # NOT connected to QApplication.quit. It used to be, which stopped the
        # event loop the moment shutdown was requested — while app.py's
        # teardown still had ~45 steps to run on that loop. run_application()
        # ends the loop itself once teardown has returned.

    def _ensure_content_panel(self) -> Any:
        """The panel that holds what ORION found. Created once, kept filled.

        It is populated whether or not it is on screen, so opening it shows
        this morning's stories rather than an empty rectangle — and it does NOT
        raise itself when results arrive. A window that appears over your work
        in the middle of a briefing is not a feature.

        It sits under the face, in the HUD. It used to be a separate window,
        on the reasoning that the Core Window was deliberately face-only and a
        scrolling list beneath the avatar would undo that on the quiet. The
        Core Window is a HUD now — telemetry down one side, log down the other
        — so that reasoning has expired, and a result you have to open a
        second window to read is a result you do not read.
        """
        panel = getattr(self, "_content_panel", None)
        if panel is not None:
            try:
                panel.count           # touch it; raises if Qt deleted it
                return panel
            except RuntimeError:
                pass
        try:
            from .content_panel import ContentPanel
        except Exception as exc:
            self.bus.log.emit(f"GUI: content panel unavailable - {exc}")
            return None
        panel = ContentPanel()
        panel.setWindowTitle("O.R.I.O.N. — What He Found")
        self._content_panel = panel
        hud = getattr(self, "hud", None)
        if hud is not None:
            # A strip under the face, not a window. Bounded in height so a
            # long list of results cannot squeeze the face down to nothing —
            # the panel scrolls instead.
            panel.setMaximumHeight(CONTENT_STRIP_HEIGHT)
            panel.hide()
            hud.add_below_centre(panel)
        else:
            panel.resize(460, 620)
        return panel

    def _on_content_results(self, rows: Any) -> None:
        panel = self._ensure_content_panel()
        if panel is None:
            return
        try:
            added = panel.show_results(rows)
        except Exception:
            return
        if added:
            # A glance down at what landed, the same wordless acknowledgement
            # the banner gets. The panel itself stays where it is.
            self._to_face("glance", 0.0, -0.6, 1.0)

    def open_content_panel(self) -> None:
        """Show the strip of results under the face.

        It is docked now, so this reveals it in place. Raising and activating
        it belonged to the separate window it used to be, and calling those on
        a child widget steals focus from whatever you were typing into.
        """
        panel = self._ensure_content_panel()
        if panel is None:
            return
        panel.show()

    def open_memory_viewer(self) -> None:
        """Show everything ORION has stored, and let the user delete any of it.

        The Command Deck already had a MEMORY panel, but it showed COUNTS —
        "identity:2 personal:5 knowledge:1207". A number is not transparency.
        If an assistant keeps a file on someone, that person has to be able to
        read it and cross things out, and `records()` and `forget()` had both
        been available the whole time with nothing to reach them.
        """
        try:
            from .memory_viewer import MemoryViewer
        except Exception as exc:
            self.bus.log.emit(f"GUI: memory viewer unavailable - {exc}")
            return
        matrix = getattr(self.memory, "matrix", None) or self.memory
        existing = getattr(self, "_memory_viewer", None)
        if existing is not None:
            try:
                existing.refresh()
                existing.show()
                existing.raise_()
                return
            except RuntimeError:
                pass          # the previous one was closed and deleted
        viewer = MemoryViewer(matrix, self)
        self._memory_viewer = viewer
        viewer.show()

    def enable_clipboard_intelligence(self, on: bool = True) -> bool:
        """Watch the clipboard and offer translate / summarise / explain / fix.

        Off unless asked for, because a watcher nobody asked for is a watcher.
        Every capture goes through SpillageGuard first and anything holding a
        credential is skipped silently — not announced, because "I noticed your
        password" tells the user their clipboard is being read in the most
        alarming way available. Nothing leaves the machine until a button is
        pressed; showing the panel costs nothing.
        """
        if not on:
            watcher = getattr(self, "_clipboard_watcher", None)
            if watcher is not None:
                watcher.enabled = False
            self.bus.log.emit("GUI: clipboard suggestions off.")
            return False
        try:
            from .clipboard_panel import ClipboardPanel, ClipboardWatcher
        except Exception as exc:
            self.bus.log.emit(f"GUI: clipboard panel unavailable - {exc}")
            return False

        watcher = getattr(self, "_clipboard_watcher", None)
        if watcher is None:
            panel = ClipboardPanel(self)
            panel.requested.connect(self._on_clipboard_request)
            watcher = ClipboardWatcher(panel, enabled=True)
            if not watcher.attach():
                self.bus.log.emit("GUI: no clipboard available on this platform.")
                return False
            self._clipboard_panel = panel
            self._clipboard_watcher = watcher
        watcher.enabled = True
        self.bus.log.emit(
            "GUI: clipboard suggestions on — credentials are never offered.")
        return True

    def _on_clipboard_request(self, instruction: str, text: str) -> None:
        """A button was pressed: now, and only now, the text goes to ORION."""
        worker = getattr(self, "worker", None)
        if worker is None or not hasattr(worker, "submit_text"):
            self.bus.log.emit("GUI: nothing to send the clipboard to.")
            return
        try:
            from .. import background

            background.spawn(worker.submit_text(f"{instruction}\n\n{text}"))
        except Exception as exc:
            self.bus.log.emit(f"GUI: clipboard request failed - {exc}")

    def _identity(self) -> Any:
        """ORION's IdentityManager, once the voice worker is attached."""
        dispatcher = getattr(self.worker, "dispatcher", None) if self.worker is not None else None
        return getattr(dispatcher, "identity", None)

    def open_preferences(self) -> None:
        """Audio devices, identity and accent colour, in one place.

        Reachable by voice ("open preferences") as well as from the interface,
        because the single most common reason to want it is that ORION cannot
        hear you — and clicking is awkward when the thing you are configuring
        is how you talk to him.

        Applying is deliberately narrow: the audio devices go through
        audio_devices.set_device so the existing verify-and-narrate path runs,
        and the accent recolours the live face immediately. Anything this does
        not know how to apply is simply reported rather than silently dropped.
        """
        try:
            from .preferences import PreferencesDialog
        except Exception as exc:
            self.bus.log.emit(f"GUI: preferences unavailable - {exc}")
            return

        identity = self._identity()
        dialog = PreferencesDialog(self, accent=getattr(self, "_accent_override", ""),
                                   user_name=identity.user_name() if identity else "",
                                   messaging_provider=lambda: getattr(
                                       getattr(getattr(self, "worker", None), "dispatcher", None), "messaging", None))
        # The Brain selector saves as it changes; the running assistant
        # applies it at once (Live closed for the local brain, Ollama started).
        brain_changed = getattr(dialog, "brain_mode_changed", None)
        if brain_changed is not None:
            brain_changed.connect(lambda mode: OrionCoreWindow._apply_brain_mode(self, mode))
        # Reply pause, expressive voice and the online voice live in the
        # channel's own setup: refresh it (resuming the conversation) once he
        # is quiet, so the change is heard now rather than at the next channel.
        voice_changed = getattr(dialog, "voice_setting_changed", None)
        if voice_changed is not None:
            voice_changed.connect(lambda key, _value: OrionCoreWindow._refresh_live_channel(self, key))
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        chosen = dialog.values()

        if identity is not None:
            try:
                if identity.set_user_name(chosen.get("user_name", "")):
                    name = identity.user_name()
                    self.bus.log.emit(f"ORION: I'll call you {name}." if name
                                      else "ORION: I'll stop using your name.")
                    self._tick_clock()
            except Exception as exc:
                self.bus.log.emit(f"GUI: could not save your name - {exc}")

        try:
            from .. import audio_devices

            for kind in ("input", "output"):
                wanted = chosen.get(f"{kind}_device", "")
                if wanted == audio_devices.configured_spec(kind):
                    continue
                # Over the bus, not audio_devices.set_device() directly.
                # set_device only PERSISTS the choice — it does not reopen the
                # streams — so picking a microphone here used to do nothing
                # until ORION was restarted, which is the opposite of what
                # somebody opening this dialog needs: they are usually here
                # BECAUSE he cannot hear them. The bus handler persists it and
                # reopens the live stream, on the thread that owns it, and
                # says which device it landed on.
                self.bus.audio_device_request.emit(kind, wanted)
        except Exception as exc:
            self.bus.log.emit(f"AUDIO: could not apply device choice - {exc}")

        accent = chosen.get("accent", "")
        if accent:
            self.apply_accent(accent)

        motion = chosen.get("motion", "")
        if motion:
            try:
                from .. import appearance
                if motion != appearance.motion_profile().name:
                    applied = appearance.set_motion_profile(motion)
                    if applied:
                        self.bus.log.emit(
                            f"GUI: motion preference set to "
                            f"{appearance.MOTION_PROFILES[applied].label}.")
            except Exception as exc:
                self.bus.log.emit(f"GUI: could not apply the motion preference - {exc}")

        design = chosen.get("face_design", "")
        if design:
            try:
                from .. import appearance
                from .screen_face import apply_design
                if design != appearance.face_design():
                    applied = appearance.set_face_design(design)
                    if applied:
                        apply_design(applied)
                        self.bus.log.emit(f"GUI: ORION's face is now the "
                                          f"{appearance.design_label(applied)} design.")
            except Exception as exc:
                self.bus.log.emit(f"GUI: could not apply the face design - {exc}")

    def _apply_brain_mode(self, mode: str) -> None:
        """Apply a brain mode Preferences has already saved."""
        apply = getattr(getattr(self, "worker", None), "apply_brain_mode", None)
        if not callable(apply):
            self.bus.log.emit(f"BRAIN: {mode} saved; it applies from the next start.")
            return
        try:
            _ok, message = apply(mode, persist=False)
            self.bus.log.emit(f"BRAIN: {message}")
        except Exception as exc:
            self.bus.log.emit(f"BRAIN: could not apply the brain mode - {exc}")

    def _refresh_live_channel(self, key: str) -> None:
        """A voice setting the live channel reads at setup was saved."""
        if key not in ("reply_pause", "expressive", "cloud_voice"):
            return
        refresh = getattr(getattr(self, "worker", None), "request_channel_refresh", None)
        if callable(refresh):
            refresh(f"new {key.replace('_', ' ')} setting")

    def apply_accent(self, accent: str) -> bool:
        """Recolour the whole shell, and ORION's face with it.

        The stylesheet is an f-string built once at import, so the crimson is
        baked in by the time anything is on screen. Rather than making the
        palette mutable — which would mean auditing every module that captured
        a colour into a local, and would still repaint nothing — a recoloured
        stylesheet is derived by substitution and handed to Qt, which repaints
        the tree itself.

        A chosen colour becomes a FAMILY. Swapping only the primary leaves
        every hover state crimson, which reads as a bug rather than a theme.
        """
        try:
            from .theming import is_readable, themed_stylesheet
        except Exception as exc:
            self.bus.log.emit(f"GUI: theming unavailable - {exc}")
            return False

        if not is_readable(accent):
            # Not a hard refusal — it is the user's interface — but a colour
            # that vanishes into the void black is worth saying so about
            # rather than silently applying and leaving them with an
            # unreadable shell.
            self.bus.log.emit(
                f"GUI: {accent} has too little contrast against the background "
                f"to read comfortably — keeping the current accent.")
            return False

        self._accent_override = accent
        try:
            self.setStyleSheet(themed_stylesheet(accent))
        except Exception as exc:
            self.bus.log.emit(f"GUI: could not restyle - {exc}")
            return False

        # The HUD sets its OWN stylesheet on itself, and a child's sheet wins
        # over an inherited one for its own rules — so without this the shell
        # turned to the new accent and the HUD stayed crimson, which reads as
        # a half-finished theme rather than a theme.
        hud = getattr(self, "hud", None)
        if hud is not None:
            try:
                from .hud_shell import hud_stylesheet

                hud.setStyleSheet(hud_stylesheet(accent))
            except Exception as exc:
                self.bus.log.emit(f"GUI: HUD kept its old accent - {exc}")

        # The Command Deck is a SEPARATE top-level window. It is handed a copy
        # of this window's stylesheet once, at startup, so a Qt stylesheet set
        # here never reaches it — every one of its twenty-odd pages would have
        # stayed crimson. Both attach points hold the same object; restyle it
        # once.
        for name in ("dashboard", "command_centre"):
            deck = getattr(self, name, None)
            if deck is None or deck is self:
                continue
            try:
                deck.setStyleSheet(themed_stylesheet(accent))
            except Exception as exc:
                self.bus.log.emit(f"GUI: the deck kept its old accent - {exc}")
            break

        # The face takes its colours from the caller each frame, so this lands
        # on the very next repaint with nothing to invalidate.
        face = getattr(self, "face", None)
        if face is not None and hasattr(face, "set_palette_colours"):
            try:
                from .style import C

                face.set_palette_colours(accent, C.ACCENT)
            except Exception:
                pass
        self.bus.log.emit(f"GUI: accent is now {accent}.")
        return True

    def _on_banner(self, text: str, _secs: int = 0) -> None:
        """The retired HUD owned the banner surface; in the face-first shell the
        banner is shown on the face's own label so that feedback is preserved."""
        if hasattr(self.face, "set_label"):
            try:
                self.face.set_label(str(text))
            except Exception:
                pass
        # A glance down at whatever just appeared — a wordless "that landed".
        # People look at a thing before they comment on it, and a face that
        # announces something while staring straight ahead reads as a recording
        # rather than as attention. Costs nothing: the gaze is already animated,
        # this only chooses where it goes for a moment.
        self._to_face("glance", 0.0, -0.65, 1.2)

    def _on_confirm_action(self, payload: Any) -> None:
        """Raise an accessible confirmation dialog for a guarded destructive
        action and release it ONLY on the user's explicit approval.

        The token arrives over the bus (never through the model). Approval routes
        to the deterministic dispatcher callback; declining cancels the pending
        action so nothing is executed."""
        if not isinstance(payload, dict):
            return
        token = str(payload.get("token") or "")
        intent = str(payload.get("intent") or "")
        message = str(payload.get("message") or "Confirm this action?")
        dispatcher = getattr(self.worker, "dispatcher", None) if self.worker is not None else None
        if not token or dispatcher is None:
            self.bus.log.emit("SYS: confirmation prompt could not be routed (no token/dispatcher).")
            return
        if payload.get("background") is True:
            self._show_background_plugin_confirmation(payload, dispatcher)
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Confirm Destructive Action")
        box.setText(message)
        box.setInformativeText("This cannot be undone easily. Approve only if you are sure.")
        approve = box.addButton("Confirm", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(box.buttons()[-1])  # default to Cancel (safe)
        # Keyboard navigable and screen-reader labelled by QMessageBox defaults.
        box.exec()
        if box.clickedButton() is approve:
            self.bus.log.emit(f"SYS: user confirmed '{intent or 'action'}'.")
            try:
                if intent == "file_cleanup" and hasattr(dispatcher, "confirm_cleanup"):
                    # Recycling checks every path under each target (recycle_bin
                    # .refusal) and moves it through the shell - seconds for a big
                    # folder, on the thread that is also the event loop (audit 6
                    # Oct 2026, A02 review). Done on a worker; the outcome is
                    # logged when it lands.
                    async def _release(tok: str = token) -> None:
                        try:
                            done = await asyncio.to_thread(dispatcher.confirm_cleanup, tok)
                            self.bus.log.emit(f"SYS: {done.text}")
                        except Exception as exc:
                            self.bus.log.emit(f"SYS: confirmed action failed - {exc}")

                    if background.spawn(_release(), name="cleanup-release") is not None:
                        self.bus.log.emit("SYS: cleaning up in the background…")
                        result = None
                    else:                       # no running loop: do it here
                        result = dispatcher.confirm_cleanup(token)
                elif hasattr(dispatcher, "confirm_system_action"):
                    result = dispatcher.confirm_system_action(token)
                else:
                    result = None
                if result is not None:
                    self.bus.log.emit(f"SYS: {result.text}")
            except Exception as exc:
                self.bus.log.emit(f"SYS: confirmed action failed - {exc}")
        else:
            self.bus.log.emit(f"SYS: user cancelled '{intent or 'action'}'.")
            try:
                if intent == "file_cleanup" and hasattr(dispatcher, "cancel_cleanup"):
                    dispatcher.cancel_cleanup(token)
                elif hasattr(dispatcher, "cancel_system_action"):
                    dispatcher.cancel_system_action(token)
            except Exception:
                pass

    def _cancel_background_plugin_prompts(self) -> None:
        for state in list(getattr(self, "_background_plugin_prompts", {}).values()):
            state["finish"](reason="host_closed")

    def _show_background_plugin_confirmation(self, payload: dict, dispatcher: Any) -> None:
        """Offer a token-bound approval without a modal loop or desktop focus.

        The guard owns the action. GUI payload arguments are checked against
        that record; only a deep copy of the bound arguments is previewed.
        Clicking Approve releases the token through the existing dispatcher,
        never by replaying GUI arguments or granting foreground permission.
        """
        from ..system_guard import ActionIntent, PendingAction, SystemActionGuard

        token = payload.get("token")
        guard = getattr(dispatcher, "system_guard", None)
        def cancel() -> None:
            if type(token) is not str or not token:
                return  # Never turn a malformed token into cancel-all.
            try:
                dispatcher.cancel_system_action(token)
            except Exception:
                pass
        # A message on one of ORION's own channels (message_channels) is
        # approved the same way as a plugin's: non-modal, token-bound.
        channel_message = payload.get("intent") == "channel_message"
        if ((payload.get("intent") != "plugin_action" and not channel_message)
                or type(token) is not str or not token or not isinstance(guard, SystemActionGuard)):
            cancel()
            return
        guard._sweep()
        pending = guard._pending.get(token)
        if not isinstance(pending, PendingAction) or pending.intent is not ActionIntent.SEND_MESSAGE:
            cancel()
            return
        if channel_message:
            mismatch = (not isinstance(pending.payload.get("channel"), str) or not pending.payload["channel"]
                        or payload.get("channel") != pending.payload["channel"]
                        or payload.get("to") != pending.payload.get("to")
                        or payload.get("body") != pending.payload.get("message"))
        else:
            mismatch = (not isinstance(pending.payload.get("plugin"), str) or not pending.payload["plugin"]
                        or not isinstance(pending.payload.get("arguments"), dict)
                        or payload.get("plugin") != pending.payload["plugin"]
                        or payload.get("arguments") != pending.payload["arguments"])
        if mismatch:
            cancel()
            return
        try:
            expires = float(pending.created_at + pending.ttl)
            if not math.isfinite(expires) or pending.expired(guard._clock()):
                raise ValueError("Expired approval")
            bound = deepcopy(pending.payload)
            if channel_message:
                preview = (f"Channel: {bound['channel']}\nTo: {bound.get('to') or '(default)'}\n\n"
                           f"{bound.get('message') or ''}")
            else:
                preview = json.dumps({"plugin": bound["plugin"], "arguments": bound["arguments"]},
                                     ensure_ascii=False, indent=2, allow_nan=False)
            # Refuse an oversized preview instead of showing truncated approval.
            if len(preview.encode("utf-8")) > 256 * 1024:
                raise ValueError("Approval preview exceeds its display limit")
        except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
            cancel()
            return
        prompts = getattr(self, "_background_plugin_prompts", None)
        if prompts is None:
            prompts = self._background_plugin_prompts = {}
        if token in prompts:
            return  # A repeated signal must not create a second release button.
        if len(prompts) >= 4 or getattr(self, "_shutting_down", False):
            cancel()
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setTextFormat(Qt.TextFormat.PlainText)
        if channel_message:
            box.setWindowTitle("Approve Message")
            box.setText(str(payload.get("message") or "Send this message?"))
            box.setInformativeText("Nothing is sent unless you approve. This request expires "
                                   "automatically; Cancel leaves it unsent.")
        else:
            box.setWindowTitle("Approve Background Plugin Action")
            box.setText(f"Allow {bound['plugin']} to perform this exact action?")
            box.setInformativeText("Review the full arguments under Show Details before approving. "
                                   "This request expires automatically. Cancel leaves the action unstarted.")
        box.setDetailedText(preview)
        approve = box.addButton("Approve", QMessageBox.ButtonRole.AcceptRole)
        reject = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(reject)
        box.setEscapeButton(reject)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setModal(False)
        box.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        box.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        timer = QTimer(box)
        timer.setInterval(500)
        host_ref = weakref.ref(self)
        state: dict[str, Any] = {"box": box, "timer": timer, "done": False}

        def finish(*, approved: bool = False, reason: str = "cancelled") -> None:
            if state["done"]:
                return
            state["done"] = True
            prompts.pop(token, None)
            try:
                timer.stop()
                box.hide()
                box.deleteLater()
            except RuntimeError:
                pass
            host = host_ref()
            alive = host is not None and not sip.isdeleted(host)
            if alive:
                try:
                    host.destroyed.disconnect(host_destroyed)
                except (TypeError, RuntimeError):
                    pass  # Qt may already be destroying the host.
            valid = (guard._pending.get(token) is pending and pending.payload == bound
                     and not pending.expired(guard._clock()))
            if approved and alive and valid and not getattr(host, "_shutting_down", False):
                try:
                    result = dispatcher.confirm_system_action(token)
                    host.bus.log.emit(f"SYS: {result.text}")
                except Exception as exc:
                    cancel()
                    try:
                        host.bus.log.emit(f"SYS: confirmed plugin action failed ({type(exc).__name__}).")
                    except RuntimeError:
                        pass
            else:
                cancel()

        def expire() -> None:
            if guard._pending.get(token) is not pending or pending.expired(guard._clock()):
                finish(reason="expired")
        def choose(clicked: Any) -> None:
            if clicked is approve:
                finish(approved=True)
            elif clicked is reject:
                finish()
        def host_destroyed(*_args: Any) -> None:
            finish(reason="host_closed")
        state["finish"] = finish
        prompts[token] = state
        box.buttonClicked.connect(choose)
        box.finished.connect(lambda *_args: finish())
        box.destroyed.connect(lambda *_args: finish(reason="closed"))
        self.destroyed.connect(host_destroyed)
        timer.timeout.connect(expire)
        timer.start()
        box.show()

    def _on_speaking_changed(self, active: bool) -> None:
        """Voice-activity LED driven by the SpeechQueueManager."""
        self.voice_led.setText("VOICE ●" if active else "VOICE ○")
        self.voice_led.setProperty("speaking", "true" if active else "false")
        style = self.voice_led.style()
        if style is not None:
            style.unpolish(self.voice_led)
            style.polish(self.voice_led)

        # Face-only shell: the face animates its own speaking envelope via
        # bus.speaking → face.set_speaking; there is nothing else on this
        # window to switch.

    # ── state / logging ───────────────────────────────────────────────────────

    def write_log(self, message: str) -> None:
        self.bus.log.emit(message)

    def set_state(self, state: str) -> None:
        self.active_state       = str(state).upper()
        self.telemetry["state"] = self.active_state
        self.state_label.setText(_state_words(self.active_state))

    # ── telemetry snapshot ────────────────────────────────────────────────────

    def telemetry_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self.telemetry)
        snapshot["clock"] = datetime.now().strftime("%H:%M:%S")
        if self.worker is not None:
            snapshot["queue_depth"]    = self.worker.out_queue.qsize()
            snapshot["live_connected"] = self.worker.connected
            if hasattr(self.worker, "router"):
                snapshot["providers"] = self.worker.router.provider_snapshot()
        else:
            snapshot["queue_depth"]    = 0
            snapshot["live_connected"] = False
        return snapshot

    # ── telemetry loop (async, 0.75 s interval) ───────────────────────────────

    async def start_telemetry(self) -> None:
        # Shared with the command centre's system strip, which used to read
        # the same three counters on its own timer. net_io_counters costs
        # 7.5 ms here because it enumerates eight adapters, and both loops
        # were paying it on the thread that also paints the face.
        from ..system_metrics import prime, sample
        from .. import gpu_stats

        prime()
        while True:
            try:
                await asyncio.sleep(0.75)
                reading = sample()
                # NVML is cheap; the nvidia-smi fallback is not, so keep both
                # off the qasync/audio thread. gpu_stats caches the answer for
                # a second and returns a clean unavailable result on machines
                # without an NVIDIA adapter.
                gpu = await asyncio.to_thread(gpu_stats.sample)
                gpu_value = (float(gpu.get("util", 0.0))
                             if gpu.get("available") else None)
                temp_value = (float(gpu.get("temp_c", 0.0))
                              if gpu.get("available") else None)
                temp_percent = (float(gpu.get("temp_percent", temp_value))
                                if temp_value is not None else None)
                self.telemetry.update({
                    "cpu":         float(reading.cpu),
                    "ram":         float(reading.ram),
                    "net_bps":     float(reading.bytes_per_second),
                    "net_percent": float(reading.network),
                    "gpu":         gpu_value,
                    "gpu_name":    str(gpu.get("name") or "n/a"),
                    "temperature": temp_value,
                    "temperature_percent": temp_percent,
                    "state":       self.active_state,
                    "updated_at":  utc_stamp(),
                })
                # The TELEMETRY deck page (self-wired to bus.telemetry_sample)
                # renders the real-time graphs; this window no longer holds a
                # direct reference to that view.
                self.bus.telemetry_sample.emit(dict(self.telemetry))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.write_log(f"TEL: telemetry loop recovered - {exc}")
                await asyncio.sleep(1.0)


# Legacy alias — external references to OrionMainWindow keep working.
OrionMainWindow = OrionCoreWindow
