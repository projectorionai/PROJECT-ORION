"""
Tests for the unified shell — face and Command Deck in ONE window
(Mark XXII, Phase 5).

The two-window layout is the long-standing default and the riskiest thing to
break, so the first duty here is proving the unified path is opt-in and that
nothing about the existing path changed when it is off.

Uses a real UnifiedDashboard (with cheap QLabel pages) rather than a stub,
because the whole claim being tested is that a REAL deck survives being
reparented into the core window with its state intact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ORION_REMOTE_ACCESS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from orion_core.bus import OrionBus  # noqa: E402
from orion_core.gui import core_window as core_window_mod  # noqa: E402
# NOT imported from orion_core.app: importing the composition root drags in
# the whole system and leaves the process unable to survive a native window
# being recreated under offscreen Qt, which crashed unrelated window tests in
# THIS file and in test_core_window_facefirst.py. The flag lives in
# core_window.py for exactly that reason.
from orion_core.gui.core_window import OrionCoreWindow, unified_shell_enabled  # noqa: E402
from orion_core.gui.unified_dashboard import UnifiedDashboard  # noqa: E402
from orion_core.memory import MemoryAgent, OrionMemoryMatrix  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_webengine_face(monkeypatch):
    """Build the 2-D HologramFace instead of the WebEngine QuantumFace3D.

    OrionCoreWindow normally builds a real QWebEngineView for the avatar. Under
    the offscreen QPA platform that Chromium view does not survive the native
    window being recreated, which is exactly what overlay mode's
    setWindowFlags does — the process dies with STATUS_STACK_BUFFER_OVERRUN
    before any assertion runs. Verified as an environment limitation, not a
    product defect: the same sequence is fine with the 2-D face, and the
    crash reproduces with no deck attached at all, so it is unrelated to the
    unified shell being tested here.

    HologramFace is the documented fallback OrionCoreWindow already uses
    wherever QtWebEngine is absent, so this exercises a real supported path
    rather than a fiction — and the deck-docking logic under test does not
    care which widget the face is.
    """
    monkeypatch.setattr(OrionCoreWindow, "_build_face",
                        lambda self: core_window_mod.HologramFace())
    yield


def _window(tmp_path) -> OrionCoreWindow:
    bus = OrionBus()
    matrix = OrionMemoryMatrix(tmp_path / "core.db", tmp_path, bus)
    return OrionCoreWindow(bus, MemoryAgent(matrix, bus))


def _deck(bus) -> UnifiedDashboard:
    return UnifiedDashboard(bus, [("MISSION", QLabel("mission")),
                                  ("WIDGETS", QLabel("widgets")),
                                  ("SWARM", QLabel("swarm"))])


# ── the flag ─────────────────────────────────────────────────────────────────

def test_two_windows_are_the_default(monkeypatch):
    """Merging them was technically sound and looked cluttered — a face, a swarm
    rail and a twenty-page deck competing in one frame, with the deck's SWARM
    page and the rail drawing the same graph twice. Two windows give each
    surface a monitor and a job, so the shell is opt-IN again."""
    monkeypatch.delenv("ORION_UNIFIED_SHELL", raising=False)
    assert unified_shell_enabled() is False


def test_the_flag_accepts_the_usual_truthy_spellings(monkeypatch):
    for value in ("1", "true", "on", "YES"):
        monkeypatch.setenv("ORION_UNIFIED_SHELL", value)
        assert unified_shell_enabled() is True, value


def test_an_explicit_false_restores_the_two_window_layout(monkeypatch):
    for value in ("0", "false", "off", "NO"):
        monkeypatch.setenv("ORION_UNIFIED_SHELL", value)
        assert unified_shell_enabled() is False, value


def test_an_unrecognised_value_does_not_silently_merge_the_windows(monkeypatch):
    """Only the explicit truthy spellings turn the single-window shell on."""
    monkeypatch.setenv("ORION_UNIFIED_SHELL", "nonsense")
    assert unified_shell_enabled() is False


# ── the face must never be squeezed out ──────────────────────────────────────

def test_the_face_can_never_be_collapsed_to_zero_width(_app, tmp_path):
    """It reached 0px once the deck was docked beside it, which killed its
    WebGL surface ("Attachment has zero size") and removed ORION's face —
    the entire point of the unified shell — from the window."""
    win = _window(tmp_path)
    assert win.content_splitter.childrenCollapsible() is False
    win.attach_deck_inline(_deck(win.bus))
    assert win.content_splitter.sizes()[0] >= 300


def test_the_face_keeps_a_real_share_of_the_window(_app, tmp_path):
    """Measured after a REAL layout pass.

    QSplitter rescales setSizes to its own current width, so on an unshown
    window the numbers reflect a pre-layout width and the face's 360px
    minimum can dominate the whole (bogus) total. Showing the window is safe
    here only because the autouse fixture swaps the WebEngine face for the
    2-D one — see its docstring."""
    win = _window(tmp_path)
    win.resize(1800, 1000)
    win.attach_deck_inline(_deck(win.bus))
    win.show()
    QApplication.processEvents()
    face_width, _rail, deck_width = win.content_splitter.sizes()
    total = face_width + deck_width
    assert total > 800                       # a genuine layout happened
    assert 0.15 <= face_width / total <= 0.45
    assert deck_width > face_width           # the deck still owns the work area
    win.hide()


def test_the_face_pane_has_a_hard_minimum_width(_app, tmp_path):
    win = _window(tmp_path)
    win.attach_deck_inline(_deck(win.bus))
    face_container = win.content_splitter.widget(0)
    assert face_container.minimumWidth() >= 360


# ── nothing changes when it is off ───────────────────────────────────────────

def test_the_deck_panel_is_hidden_and_empty_by_default(_app, tmp_path):
    """Nothing is reserved or shown until attach_deck_inline is called."""
    win = _window(tmp_path)
    assert win.deck_is_inline is False
    assert win.deck_panel.isHidden() is True
    assert win.deck_inline is None


def test_toggling_a_page_without_a_deck_logs_cleanly(_app, tmp_path):
    win = _window(tmp_path)
    received: list[str] = []
    win.bus.log.connect(received.append)
    win.toggle_dashboard()
    assert any("not attached" in m for m in received)


# ── docking ──────────────────────────────────────────────────────────────────

def test_attaching_the_deck_moves_its_body_into_this_window(_app, tmp_path):
    """The deck's CENTRAL WIDGET is docked, never the QMainWindow itself —
    nesting one QMainWindow in another crashed this window natively when
    overlay mode changed its flags."""
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_deck_inline(deck)
    assert win.deck_is_inline is True
    assert win.deck_panel.isHidden() is False
    # the deck's stack is now a descendant of this window
    assert win.isAncestorOf(deck.stack)
    # and no QMainWindow was nested inside another
    assert deck.parent() is not win.deck_panel


def test_the_deck_keeps_all_its_pages_through_the_move(_app, tmp_path):
    """Reparented, not rebuilt — every page survives."""
    win = _window(tmp_path)
    deck = _deck(win.bus)
    before = list(deck.page_names())
    win.attach_deck_inline(deck)
    assert list(deck.page_names()) == before


def test_the_deck_keeps_its_current_page_through_the_move(_app, tmp_path):
    win = _window(tmp_path)
    deck = _deck(win.bus)
    deck.show_page_named("SWARM")
    index = deck.stack.currentIndex()
    win.attach_deck_inline(deck)
    assert deck.stack.currentIndex() == index


def test_the_placeholder_is_removed_once_the_deck_arrives(_app, tmp_path):
    win = _window(tmp_path)
    win.attach_deck_inline(_deck(win.bus))
    assert win._deck_placeholder is None


def test_docking_is_logged(_app, tmp_path):
    win = _window(tmp_path)
    received: list[str] = []
    win.bus.log.connect(received.append)
    win.attach_deck_inline(_deck(win.bus))
    assert any("unified shell" in m for m in received)


# ── navigation in the embedded case ──────────────────────────────────────────

def test_a_shortcut_switches_the_embedded_decks_page(_app, tmp_path):
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_deck_inline(deck)
    win.attach_dashboard(deck)
    win.toggle_mission_deck()
    assert "MISSION" in deck.page_names()[deck.stack.currentIndex()]


def test_toggling_re_expands_a_collapsed_deck_panel(_app, tmp_path):
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_deck_inline(deck)
    win.attach_dashboard(deck)
    win.deck_panel.hide()
    win.toggle_dashboard()
    assert win.deck_panel.isHidden() is False


def test_toggling_never_detaches_the_embedded_deck(_app, tmp_path):
    """Toggling must not tear the deck body out of the splitter or turn any
    of it back into a floating window."""
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_deck_inline(deck)
    win.attach_dashboard(deck)
    for _ in range(3):
        win.toggle_dashboard()
        win.toggle_mission_deck()
    assert win.isAncestorOf(deck.stack)
    assert deck.stack.isWindow() is False


def test_self_navigation_switches_the_embedded_page(_app, tmp_path):
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_deck_inline(deck)
    win.attach_dashboard(deck)
    win._on_gui_command({"action": "page", "target": "SWARM"})
    assert "SWARM" in deck.page_names()[deck.stack.currentIndex()]


def test_self_navigation_raises_this_window_not_the_child(_app, tmp_path, monkeypatch):
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_deck_inline(deck)
    win.attach_dashboard(deck)
    calls: list[str] = []
    monkeypatch.setattr(win, "raise_", lambda: calls.append("raise"))
    monkeypatch.setattr(win, "activateWindow", lambda: calls.append("activate"))
    win._open_deck_page("WIDGETS")
    assert calls == ["raise", "activate"]


# ── overlay mode ─────────────────────────────────────────────────────────────

def test_overlay_mode_steps_the_whole_window_aside_deck_included(_app, tmp_path):
    """The compact orb is its own window; this one, deck and all, is hidden
    whole and comes back untouched — nothing inside it is re-laid out."""
    win = _window(tmp_path)
    win.attach_deck_inline(_deck(win.bus))
    win.show()
    win.enter_overlay_mode()
    try:
        assert not win.isVisible()
        assert win._orb_overlay.isVisible()
        assert win.deck_panel.isHidden() is False
    finally:
        win.exit_overlay_mode()
    assert win.isVisible()


def test_leaving_overlay_mode_restores_the_deck(_app, tmp_path):
    win = _window(tmp_path)
    win.attach_deck_inline(_deck(win.bus))
    win.enter_overlay_mode()
    win.exit_overlay_mode()
    assert win.deck_panel.isHidden() is False


def test_a_deck_the_user_had_collapsed_stays_collapsed_after_overlay(_app, tmp_path):
    """Restoring it unconditionally would override the user's own choice."""
    win = _window(tmp_path)
    win.attach_deck_inline(_deck(win.bus))
    win.deck_panel.hide()
    win.enter_overlay_mode()
    win.exit_overlay_mode()
    assert win.deck_panel.isHidden() is True


def test_overlay_mode_without_an_embedded_deck_is_safe(_app, tmp_path):
    win = _window(tmp_path)
    win.enter_overlay_mode()
    win.exit_overlay_mode()          # must not raise


# ── the retired swarm rail ─────────────────────────────────────

def test_the_retired_swarm_rail_leaves_the_embedded_deck_alone(_app, tmp_path):
    win = _window(tmp_path)
    win.attach_deck_inline(_deck(win.bus))
    win.toggle_swarm_rail()
    assert win.swarm_rail.isHidden() is True     # the map is retired
    assert win.deck_panel.isHidden() is False
