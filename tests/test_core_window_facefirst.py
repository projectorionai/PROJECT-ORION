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


def test_a_maximised_window_comes_back_maximised(_app, tmp_path):
    win = _window(tmp_path)
    win.showMaximized()
    win.enter_overlay_mode()
    win.exit_overlay_mode()
    assert win.isMaximized()
