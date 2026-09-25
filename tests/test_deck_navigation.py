"""
Tests for the Command Deck navigation consolidation (Mark X.14):

toggle_dashboard / toggle_command_centre / toggle_mission_deck previously
each hand-rolled the same show/hide/jump-to-page logic against what is
actually the SAME UnifiedDashboard instance (attach_dashboard and
attach_command_centre both point at it, see app.py) — three copies that
could drift from each other, and pressing a second shortcut while the deck
was already open on a different page didn't reliably jump to the new one.
Consolidated into _toggle_deck_page(); these tests pin that it still does
the right thing for each entry point, and that Mission (previously
buttonless/shortcutless) now has one.

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


class _FakeDeck:
    """Stands in for UnifiedDashboard: records what page it was told to
    show and whether it's visible, without needing a real window to show."""

    def __init__(self):
        self.shown_pages: list[str] = []
        self._visible = False
        self.raised = 0
        self.activated = 0

    def isVisible(self):
        return self._visible

    def show(self):
        self._visible = True

    def hide(self):
        self._visible = False

    def show_page_named(self, name):
        self.shown_pages.append(name)

    def raise_(self):
        self.raised += 1

    def activateWindow(self):
        self.activated += 1

    def page_names(self):
        return ["MISSION", "WIDGETS", "CHESS"]


def test_dashboard_and_command_centre_and_mission_share_one_deck_instance(_app, tmp_path):
    win = _window(tmp_path)
    deck = _FakeDeck()
    win.attach_dashboard(deck)
    win.attach_command_centre(deck)
    assert win.dashboard is deck
    assert win.command_centre is deck

    win.toggle_dashboard()
    assert deck.shown_pages[-1] == "WIDGETS"
    assert deck.isVisible()

    win.toggle_command_centre()
    assert deck.shown_pages[-1] == "COMMAND"
    assert deck.isVisible()          # still open, jumped to the new page — not closed

    win.toggle_mission_deck()
    assert deck.shown_pages[-1] == "MISSION"
    assert deck.isVisible()


def test_toggling_the_same_page_again_closes_the_deck(_app, tmp_path):
    win = _window(tmp_path)
    deck = _FakeDeck()
    win.attach_dashboard(deck)
    win.attach_command_centre(deck)

    win.toggle_dashboard()          # opens on WIDGETS
    assert deck.isVisible()
    deck._visible = False           # simulate the deck reporting closed
    win.toggle_dashboard()
    assert deck.isVisible()         # re-opened


def test_missing_deck_logs_instead_of_raising(_app, tmp_path):
    win = _window(tmp_path)
    win.toggle_dashboard()          # no deck attached — must not raise
    win.toggle_mission_deck()


# ── unified Command Palette routing (Mark XX design-spec §8) ────────────────

def test_palette_page_choice_navigates_the_deck(_app, tmp_path):
    win = _window(tmp_path)
    deck = _FakeDeck()
    win.attach_dashboard(deck)

    win._on_palette_command_chosen("page", "CHESS")
    assert deck.shown_pages[-1] == "CHESS"
    assert deck.isVisible()


def test_a_chess_tool_handoff_opens_the_chess_board_not_command_centre(_app, tmp_path):
    win = _window(tmp_path)
    deck = _FakeDeck()
    win.attach_dashboard(deck)

    win._on_gui_command({"action": "chess", "target": ""})

    assert deck.shown_pages == ["CHESS"]
    assert deck.isVisible()


def test_stale_command_centre_navigation_cannot_cover_a_chess_board(_app, tmp_path):
    win = _window(tmp_path)
    deck = _FakeDeck()
    win.attach_dashboard(deck)
    win._chess_navigation_until = float("inf")

    win._on_gui_command({"action": "command", "target": ""})

    assert deck.shown_pages == []


def test_standby_ui_request_enters_the_compact_orb(_app, tmp_path, monkeypatch):
    win = _window(tmp_path)
    entered: list[bool] = []
    monkeypatch.setattr(win, "enter_overlay_mode", lambda: entered.append(True))
    win._overlay_active = False

    win._on_gui_command({"action": "standby", "target": ""})

    assert entered == [True]


def test_palette_tool_choice_does_not_navigate_the_deck(_app, tmp_path):
    win = _window(tmp_path)
    deck = _FakeDeck()
    win.attach_dashboard(deck)

    win._on_palette_command_chosen("tool", "breach_check")
    # A tool prefills the message box (or logs that there's nowhere to
    # prefill into) — it must never also jump the deck to a page.
    assert deck.shown_pages == []


def test_open_command_palette_indexes_deck_pages_alongside_tools(_app, tmp_path, monkeypatch):
    win = _window(tmp_path)
    deck = _FakeDeck()
    win.attach_dashboard(deck)

    from orion_core.gui import command_palette as command_palette_module

    captured: dict = {}

    class _NullSignal:
        def connect(self, fn):
            pass

    class _RecordingPalette:
        def __init__(self, commands, parent=None):
            captured["commands"] = commands
            self.command_chosen = _NullSignal()

        def exec(self):
            pass

    monkeypatch.setattr(command_palette_module, "CommandPalette", _RecordingPalette)
    win.open_command_palette()

    names = {c.name for c in captured["commands"]}
    assert "CHESS" in names            # a deck page
    assert any("_" in n or n.islower() for n in names)   # a dispatcher tool


def test_mission_and_palette_buttons_exist_with_distinguishing_tooltips(_app, tmp_path):
    win = _window(tmp_path)
    assert hasattr(win, "mission_btn")
    assert hasattr(win, "palette_btn")
    assert "Mission" in win.mission_btn.toolTip()
    assert "Ctrl+M" in win.mission_btn.toolTip()
    assert "Ctrl+K" in win.palette_btn.toolTip()
    # The Ctrl+D/Ctrl+Shift+C tooltips must make the shared-deck relationship
    # explicit rather than reading as two separate destinations.
    assert win.dashboard_btn.toolTip().startswith("Command Deck")
    assert win.centre_btn.toolTip().startswith("Command Deck")
