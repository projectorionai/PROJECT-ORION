"""
Swarm-first navigation: the Command Deck's tab bars are retired and the graph
becomes ORION's only navigation surface (Mark XXII).

The critical property, and the reason most of these exist: with the tab bars
gone, ANY capability that is not reachable from the swarm becomes unreachable
entirely. So "every page has a node" and "there is always a way back" are not
nice-to-haves here — they are what stops the UI trapping the user.
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
from PyQt6.QtCore import QEvent, Qt  # noqa: E402
from PyQt6.QtGui import QKeyEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from orion_core.gui.swarm_view import SwarmDeckView  # noqa: E402
from orion_core.gui.unified_dashboard import UnifiedDashboard  # noqa: E402
from orion_core.swarm.model import NodeKind  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _Signal:
    def __init__(self): self._slots = []
    def connect(self, slot): self._slots.append(slot)
    def emit(self, *a, **k):
        for s in self._slots:
            s(*a, **k)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def _deck(bus=None) -> UnifiedDashboard:
    return UnifiedDashboard(bus or _StubBus(), [
        ("MISSION", QLabel("mission")), ("BRAIN", QLabel("brain")),
        ("GLOBE", QLabel("globe")), ("OPS", QLabel("ops")),
        ("MEMORY", QLabel("memory")), ("CHESS", QLabel("chess")),
    ])


def _page_map(deck) -> dict[str, str]:
    zone_of = {p: z for z, ps in UnifiedDashboard.ZONE_PAGES.items() for p in ps}
    return {name: zone_of.get(name, "SYSTEM") for name in deck.page_names()}


# ── the deck's own navigation is retired ─────────────────────────────────────

def test_the_zone_row_is_hidden(_app):
    deck = _deck()
    deck.set_swarm_navigation(True)
    assert deck._zone_frame.isVisible() is False


def test_the_whole_page_row_is_hidden_not_just_its_middle(_app):
    """Leaving the chevrons behind would be a tab bar with the labels taken
    away, rather than an absence of one."""
    deck = _deck()
    deck.set_swarm_navigation(True)
    for widget in deck._page_bar_widgets:
        assert widget.isVisible() is False


def test_it_lands_on_the_swarm(_app):
    deck = _deck()
    deck.set_swarm_navigation(True)
    assert "BRAIN" in deck.page_names()[deck.stack.currentIndex()]


def test_the_tab_bars_can_be_restored(_app):
    deck = _deck()
    deck.set_swarm_navigation(True)
    deck.set_swarm_navigation(False)
    assert deck._zone_frame.isHidden() is False
    assert deck.swarm_navigation is False


# ── nothing is lost ──────────────────────────────────────────────────────────

def test_every_page_still_exists(_app):
    """Only the duplicate chrome goes; the pages themselves are untouched."""
    deck = _deck()
    before = list(deck.page_names())
    deck.set_swarm_navigation(True)
    assert list(deck.page_names()) == before


def test_programmatic_navigation_still_works(_app):
    deck = _deck()
    deck.set_swarm_navigation(True)
    deck.show_page_named("OPS")
    assert "OPS" in deck.page_names()[deck.stack.currentIndex()]


def test_the_command_palette_still_sees_every_page(_app):
    """Ctrl+K remains a full escape hatch, tab bars or not."""
    deck = _deck()
    deck.set_swarm_navigation(True)
    assert len(deck.page_names()) == 6


# ── you can always get back ──────────────────────────────────────────────────

def test_a_home_button_appears(_app):
    deck = _deck()
    deck.set_swarm_navigation(True)
    assert deck._swarm_home_btn.isHidden() is False


def test_the_home_button_returns_to_the_swarm(_app):
    deck = _deck()
    deck.set_swarm_navigation(True)
    deck.show_page_named("OPS")
    deck._swarm_home_btn.click()
    assert "BRAIN" in deck.page_names()[deck.stack.currentIndex()]


def test_escape_returns_to_the_swarm(_app):
    deck = _deck()
    deck.set_swarm_navigation(True)
    deck.show_page_named("MEMORY")
    deck.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                 Qt.KeyboardModifier.NoModifier))
    assert "BRAIN" in deck.page_names()[deck.stack.currentIndex()]


def test_escape_returns_to_the_brain_whatever_the_navigation_mode(_app):
    # Mark XXXII: Esc is "home" from every page. It used to work only in the
    # legacy swarm mode, so from most pages it did nothing at all.
    deck = _deck()
    deck.set_swarm_navigation(False)
    deck.show_page_named("OPS")
    deck.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                 Qt.KeyboardModifier.NoModifier))
    assert "BRAIN" in deck.page_names()[deck.stack.currentIndex()]


def test_the_home_button_is_hidden_when_tab_bars_are_showing(_app):
    deck = _deck()
    deck.set_swarm_navigation(False)
    assert deck._swarm_home_btn.isVisible() is False


# ── every capability is reachable from the graph ─────────────────────────────

def test_every_deck_page_has_a_node_in_the_swarm(_app):
    """THE property that makes retiring the tab bars safe: a page with no
    node would become unreachable, not merely harder to find."""
    deck = _deck()
    view = SwarmDeckView(_StubBus())
    view.attach_deck_pages(_page_map(deck))
    ids = view.compose_typed().node_ids()
    for name in deck.page_names():
        assert f"page:{name}" in ids, name


def test_an_orphan_page_is_still_reachable(_app):
    """CHESS belongs to no zone by design."""
    deck = _deck()
    view = SwarmDeckView(_StubBus())
    view.attach_deck_pages(_page_map(deck))
    assert "page:CHESS" in view.compose_typed().node_ids()


def test_every_page_node_carries_an_open_action(_app):
    deck = _deck()
    view = SwarmDeckView(_StubBus())
    view.attach_deck_pages(_page_map(deck))
    from orion_core.swarm.inspect import build_report

    for node in view.compose_typed().nodes:
        if node.kind is NodeKind.PAGE:
            assert build_report(node).action_target == node.label


def test_opening_every_page_node_navigates_the_real_deck(_app):
    """End to end, for every page: click its node, land on that page."""
    deck = _deck()
    view = SwarmDeckView(_StubBus(), on_open_page=deck.show_page_named)
    view.attach_deck_pages(_page_map(deck))
    deck.set_swarm_navigation(True)
    for name in deck.page_names():
        view._on_swarm_event({"id": f"page:{name}", "kind": "page",
                              "label": name, "cluster": "SYSTEM"})
        view.inspector_action.click()
        assert name in deck.page_names()[deck.stack.currentIndex()], name
