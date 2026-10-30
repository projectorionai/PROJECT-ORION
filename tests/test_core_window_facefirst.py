"""
Face-only shell: the Core Window shows nothing but ORION's face.  LOG,
MEMORY, TELEMETRY, the inner-voice thought stream and the calendar/geo panel
all moved to the Command Deck (see test_ops_deck_inner_voice_environment.py
for the panels that now live there).

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ORION_REMOTE_ACCESS", "0")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _window(tmp_path):
    from orion_core.bus import OrionBus
    from orion_core.memory import MemoryAgent, OrionMemoryMatrix
    from orion_core.gui.core_window import OrionCoreWindow

    bus = OrionBus()
    matrix = OrionMemoryMatrix(tmp_path / "core.db", tmp_path, bus)
    memory = MemoryAgent(matrix, bus)
    return OrionCoreWindow(bus, memory)


class _StubDeck:
    """Minimal stand-in for the UnifiedDashboard the Core Window talks to."""

    def __init__(self) -> None:
        self.shown_pages: list[str] = []
        self._visible = False

    def isVisible(self) -> bool:
        return self._visible

    def show(self) -> None:
        self._visible = True

    def hide(self) -> None:
        self._visible = False

    def show_page_named(self, name: str) -> None:
        self.shown_pages.append(name)

    def raise_(self) -> None:
        pass

    def activateWindow(self) -> None:
        pass


def test_only_the_face_is_on_the_window(_app, tmp_path):
    win = _window(tmp_path)
    assert win.centralWidget() is not None
    # Nothing left to switch between — the old view-switcher/stack/left panel/
    # mini-orb machinery is gone entirely.
    for gone in ("stack", "left_panel", "mini_orb", "_view_buttons",
                 "view_switcher", "thought_box", "calendar_widget",
                 "location_label", "weather_label"):
        assert not hasattr(win, gone), f"{gone} should not exist on the face-only shell"
    assert hasattr(win, "face")


def test_environment_panel_and_log_view_attach_late(_app, tmp_path):
    win = _window(tmp_path)
    assert win.environment_panel is None
    assert win.log_view is None

    class _StubEnvironment:
        pass

    class _StubLog:
        pass

    env = _StubEnvironment()
    log = _StubLog()
    win.attach_environment_panel(env)
    win.attach_log_view(log)
    assert win.environment_panel is env
    assert win.log_view is log


def test_gui_command_view_log_memory_telemetry_open_deck_pages(_app, tmp_path):
    win = _window(tmp_path)
    deck = _StubDeck()
    win.attach_dashboard(deck)
    for target in ("log", "memory", "telemetry"):
        win._on_gui_command({"action": "view", "target": target})
    assert deck.shown_pages == ["log", "memory", "telemetry"]


def test_gui_command_view_face_raises_the_window_instead_of_switching(_app, tmp_path, monkeypatch):
    win = _window(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(win, "raise_", lambda: calls.append("raise"))
    monkeypatch.setattr(win, "activateWindow", lambda: calls.append("activate"))
    win._on_gui_command({"action": "view", "target": "face"})
    assert calls == ["raise", "activate"]


def test_gui_command_refresh_environment_routes_to_attached_panel(_app, tmp_path):
    import asyncio

    win = _window(tmp_path)

    class _StubEnvironment:
        def __init__(self):
            self.refreshed = False

        async def refresh_environment_widgets(self):
            self.refreshed = True
            return True

    env = _StubEnvironment()
    win.attach_environment_panel(env)

    async def _run():
        # asyncio.create_task() needs a running loop, so drive the handler
        # from inside one, then yield so the created task actually executes.
        win._on_gui_command({"action": "refresh_environment", "target": ""})
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert env.refreshed


def test_gui_command_refresh_environment_without_panel_logs_cleanly(_app, tmp_path):
    win = _window(tmp_path)
    received: list[str] = []
    win.bus.log.connect(received.append)
    win._on_gui_command({"action": "refresh_environment", "target": ""})
    assert any("environment panel not attached" in m for m in received)


def test_prefill_command_without_log_view_logs_cleanly(_app, tmp_path):
    win = _window(tmp_path)
    received: list[str] = []
    win.bus.log.connect(received.append)
    win._prefill_command("some_tool")
    assert any("no input to prefill" in m for m in received)


def test_prefill_command_with_log_view_fills_and_opens_deck(_app, tmp_path):
    from PyQt6.QtWidgets import QLineEdit

    win = _window(tmp_path)
    deck = _StubDeck()
    win.attach_dashboard(deck)

    class _StubLog:
        def __init__(self):
            self.input_line = QLineEdit()

    win.attach_log_view(_StubLog())
    win._prefill_command("security_recon")
    assert win.log_view.input_line.text() == "security_recon"
    assert deck.shown_pages == ["log"]


def test_write_log_emits_on_the_bus(_app, tmp_path):
    win = _window(tmp_path)
    received: list[str] = []
    win.bus.log.connect(received.append)
    win.write_log("SYS: hello")
    assert "SYS: hello" in received


# ── swarm rail (Mark XXI, Track A1/A2) ───────────────────────────────────────

def test_swarm_rail_exists_and_starts_collapsed(_app, tmp_path):
    win = _window(tmp_path)
    assert hasattr(win, "swarm_rail")
    assert win.swarm_rail.isHidden() is True
    assert win.swarm_view is None


def test_attach_swarm_view_docks_the_widget_and_replaces_the_placeholder(_app, tmp_path):
    from PyQt6.QtWidgets import QLabel

    win = _window(tmp_path)
    stub = QLabel("stand-in swarm view")
    win.attach_swarm_view(stub)
    assert win.swarm_view is stub
    assert stub.parent() is win.swarm_rail
    # the "waiting for ORION to finish booting" placeholder is out of the
    # layout (removeWidget) and scheduled for deletion (deleteLater — Qt
    # only actually deletes it on the next event-loop pass, so its .parent()
    # isn't cleared synchronously; the layout membership is what matters).
    assert win.swarm_rail.layout().indexOf(win._swarm_rail_placeholder) == -1
    assert win.swarm_rail.layout().indexOf(stub) != -1


def test_the_subsystem_map_cannot_be_opened_any_more(_app, tmp_path):
    """Mark XXXI retired the "neural subsystem map" rail. Asking for it must
    neither open a second 3-D scene beside the face nor do nothing: it says
    where the architecture now lives and goes there."""
    win = _window(tmp_path)
    deck = _StubDeck()
    win.attach_dashboard(deck)
    said: list[str] = []
    win.bus.log.connect(said.append)
    win.toggle_swarm_rail()
    assert win.swarm_rail.isHidden() is True
    assert deck.shown_pages == ["BRAIN"]
    assert any("BRAIN" in line for line in said)


def test_gui_command_swarm_opens_the_brain_page_instead(_app, tmp_path):
    win = _window(tmp_path)
    deck = _StubDeck()
    win.attach_dashboard(deck)
    win._on_gui_command({"action": "swarm"})
    assert win.swarm_rail.isHidden() is True
    assert "BRAIN" in deck.shown_pages


def test_the_swarm_shortcut_is_gone(_app, tmp_path):
    from PyQt6.QtGui import QKeySequence

    win = _window(tmp_path)
    shortcuts = [a.shortcut() for a in win.actions()]
    assert QKeySequence("Ctrl+Shift+S") not in shortcuts


def test_command_deck_swarm_page_is_unaffected_by_attaching_a_core_rail_view(_app, tmp_path):
    """attach_swarm_view must dock a SEPARATE instance, never reparent the
    Command Deck's own page widget out from under it."""
    from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

    win = _window(tmp_path)
    deck_page_widget = QLabel("the Command Deck's page")
    deck_container = QWidget()
    QVBoxLayout(deck_container).addWidget(deck_page_widget)

    rail_widget = QLabel("a different, second instance")
    win.attach_swarm_view(rail_widget)

    assert deck_page_widget.parent() is deck_container   # never touched
    assert rail_widget.parent() is win.swarm_rail


# ── the compact orb: a separate window, the face never touched ───────────────

def test_the_compact_orb_is_its_own_window_and_the_face_survives(_app, tmp_path):
    """The old overlay re-flagged THIS window, which destroyed and rebuilt the
    face — and a rebuilt WebGL page in a translucent tool window often never
    started, so ORION came back as the old software head, or as black.

    Now this window is only hidden; the face object is the very same one
    before, during and after."""
    win = _window(tmp_path)
    win.show()
    face = win.face
    flags = win.windowFlags()
    win.enter_overlay_mode()
    try:
        overlay = win._orb_overlay
        assert overlay is not None and overlay.isVisible()
        assert overlay.isWindow(), "the orb must be its own top-level window"
        assert not win.isVisible(), "the full window should step aside"
        assert win.face is face, "the face was rebuilt"
        assert win.windowFlags() == flags, "this window's flags were changed"
    finally:
        win.exit_overlay_mode()
    assert win.isVisible() and not win._orb_overlay.isVisible()
    assert win.face is face
    assert win.windowFlags() == flags


def test_every_way_back_from_the_orb_works(_app, tmp_path):
    """Double-click, Esc, the chip and the menu all emit restore_requested;
    the window is wired to it. "I can't uncompact it anymore" must be
    impossible."""
    win = _window(tmp_path)
    win.show()
    win.enter_overlay_mode()
    win._orb_overlay.restore_requested.emit()
    assert win._overlay_active is False
    assert win.isVisible()
    win.enter_overlay_mode()
    win._orb_overlay.restore_btn.click()
    assert win._overlay_active is False
    win.enter_overlay_mode()
    win.toggle_overlay_mode()
    assert win._overlay_active is False


def test_the_orb_follows_orions_state(_app, tmp_path):
    win = _window(tmp_path)
    win.show()
    win.enter_overlay_mode()
    try:
        win.bus.state.emit("SPEAKING")
        assert win._orb_overlay.orb.state_name == "SPEAKING"
    finally:
        win.exit_overlay_mode()


def test_compact_shows_the_screen_face_he_is_wearing(_app, tmp_path, monkeypatch):
    """26 Oct 2026, the owner: pressing Compact showed the orb - a form Mark
    XXXVII had switched off. It shows the face he is wearing."""
    from orion_core import appearance
    from orion_core.gui.screen_face import ScreenFacePanel
    monkeypatch.setattr(appearance, "face_form", lambda: "screen")
    win = _window(tmp_path)
    win.show()
    win.enter_overlay_mode()
    try:
        overlay = win._orb_overlay
        assert overlay.form == "screen" and isinstance(overlay.orb, ScreenFacePanel)
        assert overlay.width() > overlay.height(), "the visor's proportions, not a disc"
        win.bus.state.emit("LISTENING")
        assert overlay.orb.state_name == "LISTENING"
    finally:
        win.exit_overlay_mode()


def test_the_compact_face_speaks_emotes_and_moves_its_mouth(_app):
    from orion_core.gui.orb_overlay import OrbOverlay

    class Face:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return lambda *args: self.calls.append((name, args))

    overlay = OrbOverlay(form="screen")
    panel, face = overlay.orb, Face()
    overlay.orb = face
    try:
        overlay.set_speaking(True)
        overlay.apply_emotion("happy", {"intensity": 0.8})
        overlay.set_viseme_payload({"open": 0.5, "width": 0.2, "closure": 0.0})
        overlay.set_viseme_payload({"viseme": "aa", "weight": 0.7})
        overlay.set_viseme_payload("not a payload")
    finally:
        overlay.orb = panel
        overlay.deleteLater()
    assert face.calls == [("set_speaking", (True,)),
                          ("apply_emotion", ("happy", {"intensity": 0.8})),
                          ("set_viseme", (0.5, 0.2, 0.0)),
                          ("set_viseme", ("aa", 0.7))]


def test_the_orb_is_kept_for_the_orb_and_classic_faces(_app, monkeypatch):
    from orion_core import appearance
    from orion_core.gui import orb_overlay
    monkeypatch.setattr(appearance, "face_form", lambda: "classic")
    assert orb_overlay.compact_form() == "orb"
    overlay = orb_overlay.OrbOverlay()
    try:
        assert isinstance(overlay.orb, orb_overlay.CompactOrb)
        assert overlay.width() == overlay.height() == orb_overlay.ORB_SIZE
        overlay.set_speaking(True)                  # the orb has none of these:
        overlay.apply_emotion("happy")              # each is a quiet no-op
    finally:
        overlay.deleteLater()


def test_a_maximised_window_comes_back_maximised(_app, tmp_path):
    win = _window(tmp_path)
    win.showMaximized()
    win.enter_overlay_mode()
    win.exit_overlay_mode()
    assert win.isMaximized()


def test_initial_face_health_watch_is_installed_after_build(_app, tmp_path, monkeypatch):
    from orion_core.gui.core_window import OrionCoreWindow
    watched = []
    monkeypatch.setattr(OrionCoreWindow, "_watch_face_health", lambda self: watched.append(self.face))
    window = _window(tmp_path)
    assert watched == [window.face]


def _health_watch_fixture(monkeypatch, *, states=None, answer=True, failed=False):
    from types import SimpleNamespace
    from orion_core.gui import core_window

    timers, queries, actions = [], [], []
    remaining = list(states or ["alive"])

    class Face:
        on_failed = None

        def __init__(self):
            self.failed = failed

        def check_state(self, callback):
            queries.append(callback)
            if answer:
                callback(remaining.pop(0) if remaining else "alive")

    face = Face()
    owner = SimpleNamespace(face=face, isVisible=lambda: True, _shutting_down=False,
                            bus=SimpleNamespace(log=SimpleNamespace(emit=lambda _line: None)),
                            _probe_face=core_window.OrionCoreWindow._probe_face)

    def replace(action):
        actions.append(action)
        owner.face = Face()

    owner._rebuild_face = lambda: replace("retry")
    owner._swap_face = lambda: replace("software")
    monkeypatch.setattr(core_window, "QTimer", SimpleNamespace(
        singleShot=lambda delay, callback: timers.append((delay, callback))))

    def run_timer(delay):
        for index, (when, callback) in enumerate(timers):
            if when == delay:
                timers.pop(index)
                callback()
                return
        raise AssertionError(f"no queued timer at {delay} ms")

    core_window.OrionCoreWindow._watch_face_health(owner)
    return owner, face, actions, queries, run_timer


def test_a_startup_host_failure_triggers_recovery_without_waiting_for_a_probe(monkeypatch):
    owner, face, actions, _queries, timer = _health_watch_fixture(monkeypatch, failed=True)
    timer(0)
    assert actions == ["retry"]
    assert owner.face is not face


def test_face_health_keeps_watching_after_the_first_healthy_frame(monkeypatch):
    owner, face, actions, queries, timer = _health_watch_fixture(monkeypatch, states=["alive", "failed"])
    timer(6000)
    assert len(queries) == 1 and not actions
    timer(4000)
    assert len(queries) == 2 and actions == ["retry"]
    assert owner.face is not face


def test_late_host_failure_callback_recovers_a_previously_healthy_face(monkeypatch):
    _owner, face, actions, _queries, timer = _health_watch_fixture(monkeypatch)
    timer(6000)
    face.on_failed()
    timer(0)
    assert actions == ["retry"]


def test_unanswered_face_queries_have_a_bounded_recovery(monkeypatch):
    owner, face, actions, queries, timer = _health_watch_fixture(monkeypatch, answer=False)
    timer(6000)
    timer(2000)
    timer(4000)
    timer(2000)
    timer(4000)
    timer(2000)
    assert len(queries) == 3 and actions == ["retry"]
    # A reply from the discarded query cannot condemn its replacement.
    queries[0]("failed")
    assert actions == ["retry"] and owner.face is not face


def test_face_retirement_cannot_recover_recursively_from_shutdown_queries(monkeypatch):
    from types import SimpleNamespace
    from orion_core.gui.core_window import OrionCoreWindow
    _owner, face, actions, queries, timer = _health_watch_fixture(monkeypatch, answer=False)
    timer(6000)
    face.host_wids = {}
    face.shutdown = lambda: queries[0]("failed")
    face.setParent = lambda _parent: None
    face.deleteLater = lambda: None
    OrionCoreWindow._retire_face(face, SimpleNamespace(removeWidget=lambda _widget: None))
    timer(2000)
    assert actions == [] and face._orion_retired


def test_face_failure_after_one_retry_uses_software_without_changing_saved_form(monkeypatch):
    monkeypatch.setenv("ORION_FACE_FALLBACK", "")
    owner, face, actions, _queries, timer = _health_watch_fixture(monkeypatch, failed=True)
    owner._face_retried = True
    timer(0)
    assert actions == ["software"]
    assert os.environ["ORION_FACE_FALLBACK"] == "orb"
    assert owner.face is not face


def test_shutting_down_ignores_pending_face_failure(monkeypatch):
    owner, _face, actions, _queries, timer = _health_watch_fixture(monkeypatch, answer=False)
    timer(6000)
    owner._shutting_down = True
    timer(2000)
    assert actions == []


def test_broken_local_webgl_uses_a_software_orb_instead_of_unattached_native_gl(monkeypatch):
    from orion_core import appearance
    from orion_core.gui import core_window, hud

    class SoftwareOrb:
        pass

    def broken():
        raise RuntimeError("the local avatar assets cannot be served")

    monkeypatch.setenv("ORION_FACE_FALLBACK", "")
    monkeypatch.setattr(core_window, "holo_head_enabled", lambda: False)
    monkeypatch.setattr(core_window, "webengine_face_possible", lambda: True)
    monkeypatch.setattr(core_window, "_new_webgl_face", broken)
    # The classic (WebGL) face is the one that can break this way; the
    # screen face needs no WebGL at all.
    monkeypatch.setattr(appearance, "face_form", lambda: "classic")
    monkeypatch.setattr(hud, "CentralHud", SoftwareOrb)
    assert isinstance(core_window.OrionCoreWindow._build_face(object()), SoftwareOrb)
    assert appearance.face_form() == "classic"


def test_failed_local_handler_draws_a_live_software_orb_without_a_browser(_app, monkeypatch):
    from types import SimpleNamespace
    from orion_core.gui import face3d
    from orion_core.gui.hud import CentralHud
    discarded = []

    class UnloadedView:
        loadFinished = SimpleNamespace(connect=lambda _callback: None)
        renderProcessTerminated = SimpleNamespace(connect=lambda _callback: None)

        def page(self):
            return SimpleNamespace(setBackgroundColor=lambda _colour: None)

        def deleteLater(self):
            discarded.append(True)

    monkeypatch.setattr(face3d, "WEBENGINE_OK", True)
    monkeypatch.setattr(face3d, "QWebEngineView", UnloadedView, raising=False)
    face = face3d.QuantumFace3D()
    monkeypatch.setattr(face, "_install_asset_handler", lambda: None)
    face._ensure_built()
    assert isinstance(face._software_fallback, CentralHud)
    assert face.view is None and discarded == [True]
    face.set_form("face")
    face.set_avatar_frame({"js_state": "SPEAKING", "amplitude": 0.7})
    assert face.form == "face"
    assert face._software_fallback.state_name == "SPEAKING"
    assert face._software_fallback.target_amplitude == 0.7
    face.set_state("OFFLINE")
    face.set_amplitude(0.2)
    assert face._software_fallback.state_name == "OFFLINE"
    assert face._software_fallback.target_amplitude == 0.2
