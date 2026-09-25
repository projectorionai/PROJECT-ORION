"""Native Command Deck overview and persistent navigation regressions."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget

from orion_core.bus import OrionBus
from orion_core.gui.command_overview import CommandOverview
from orion_core.gui.unified_dashboard import UnifiedDashboard


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_overview_updates_only_live_values_and_routes_actions():
    app = _app()
    bus = OrionBus()
    opened = []
    view = CommandOverview(bus, on_open_page=opened.append)
    view.attach_deck_pages({"RESEARCH": "RESEARCH", "LOG": "MONITORING"})
    bus.state.emit("PROCESSING")
    bus.connection_state.emit({"provider": "Gemini", "state": "connected"})
    bus.telemetry_sample.emit({"cpu": 27.3, "ram": 61.8})
    view.pulse(to_id="research", tool="web_search", ok=True)
    app.processEvents()
    assert view.mode.text() == "PROCESSING"
    assert "Gemini" in view.link.text()
    assert view.host.text() == "CPU 27%   ·   RAM 62%"
    assert "web_search" in view.route.text()
    assert view._quick_links["RESEARCH"].isEnabled()
    assert not view._quick_links["MISSION"].isEnabled()
    view._quick_links["RESEARCH"].click()
    assert opened == ["RESEARCH"]


def test_split_pane_keeps_every_page_visible_in_navigation():
    _app()
    bus = OrionBus()
    deck = UnifiedDashboard(bus, [(name, QWidget()) for name in
                                  ("BRAIN", "RESEARCH", "GLOBE", "LOG", "CHESS")])
    assert set(deck._sidebar_buttons) == set(deck.page_names())
    assert deck._sidebar.isVisibleTo(deck) or not deck.isVisible()
    deck._sidebar_buttons["GLOBE"].click()
    assert deck.stack.currentIndex() == deck.page_names().index("GLOBE")
    assert deck._sidebar_buttons["GLOBE"].isChecked()
    assert deck.display_name("BRAIN") == "Brain"
