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


def test_overview_shows_the_text_model_route_gpu_and_sound_alerts():
    app = _app()
    bus = OrionBus()
    view = CommandOverview(bus)
    bus.telemetry_sample.emit({"cpu": 10.0, "ram": 20.0, "gpu": 55.4})
    bus.dashboard_event.emit("provider_route", {
        "provider": "groq", "local": False, "latency_s": 1.234, "fallbacks": 1})
    app.processEvents()
    assert view.host.text() == "CPU 10%   ·   RAM 20%   ·   GPU 55%"
    assert view.model_route.text() == "groq  ·  cloud  ·  1.2 s  ·  after 1 fallback(s)"

    # A breaker opening on the provider shown says so; another provider's
    # does not overwrite it.
    bus.dashboard_event.emit("provider_breaker", {"provider": "ollama", "cooldown_s": 45})
    app.processEvents()
    assert view.model_route.text().startswith("groq  ·  cloud")
    bus.dashboard_event.emit("provider_breaker", {"provider": "groq", "cooldown_s": 90.0})
    app.processEvents()
    assert view.model_route.text() == "groq  ·  cooling 90 s"

    view.show()
    bus.dashboard_event.emit("sound_alert", {"category": "smoke alarm", "score": 0.7})
    app.processEvents()
    assert view.hearing.text().startswith("Smoke alarm  ·  ")
    assert "SOUND: smoke alarm heard" in view.activity.text()
    view.hide()


def test_overview_ignores_malformed_dashboard_payloads():
    app = _app()
    bus = OrionBus()
    view = CommandOverview(bus)
    for channel in ("provider_route", "provider_breaker", "sound_alert"):
        bus.dashboard_event.emit(channel, "not a dict")
    bus.dashboard_event.emit("provider_route", {"provider": "p", "latency_s": "n/a"})
    app.processEvents()
    assert view.hearing.text() == "No sound alerts"
    assert view.model_route.text() == "p  ·  cloud"
