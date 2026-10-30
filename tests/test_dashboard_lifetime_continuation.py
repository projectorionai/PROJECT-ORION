"""Real Qt fade ownership, plus deterministic late-callback delivery."""
from types import SimpleNamespace

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QPropertyAnimation
from PyQt6.QtWidgets import QApplication, QLabel

from orion_core.bus import OrionBus
import orion_core.gui.unified_dashboard as dashboard


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def dispose(deck):
    if not sip.isdeleted(deck):
        sip.delete(deck)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_interrupted_fades_release_animations_and_old_effect(app):
    first, second = QLabel("first"), QLabel("second")
    deck = dashboard.UnifiedDashboard(OrionBus(), [("WIDGETS", first), ("TASKS", second)])
    try:
        for index in range(20):
            deck._fade_in(first if index % 2 else second)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert len(deck.findChildren(QPropertyAnimation)) == 1
        assert second.graphicsEffect() is None
        assert first.graphicsEffect() is not None
    finally:
        dispose(deck)


def test_completed_fade_releases_animation_and_effect(app):
    page = QLabel("page")
    deck = dashboard.UnifiedDashboard(OrionBus(), [("WIDGETS", page)])
    try:
        deck._fade_in(page)
        animation = deck._anim
        animation.setCurrentTime(animation.duration())
        assert deck._anim is None
        assert page.graphicsEffect() is None
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert deck.findChildren(QPropertyAnimation) == []
    finally:
        dispose(deck)


class CapturedAnimation:
    """Capture the production completion slot without dispatching through Qt."""
    def __init__(self, *_):
        self.finished = SimpleNamespace(connect=lambda callback: setattr(self, "callback", callback))

    def __getattr__(self, _):
        return lambda *_args: None


@pytest.mark.parametrize("delete_owner", [False, True])
def test_late_completion_tolerates_deleted_page_or_owner(app, monkeypatch, delete_owner):
    monkeypatch.setattr(dashboard, "QPropertyAnimation", CapturedAnimation)
    page = QLabel("page")
    deck = dashboard.UnifiedDashboard(OrionBus(), [("WIDGETS", page)])
    try:
        deck._fade_in(page)
        callback = deck._anim.callback
        sip.delete(deck if delete_owner else page)
        callback()
    finally:
        dispose(deck)


def test_stale_completion_preserves_new_fade(app, monkeypatch):
    monkeypatch.setattr(dashboard, "QPropertyAnimation", CapturedAnimation)
    page = QLabel("page")
    deck = dashboard.UnifiedDashboard(OrionBus(), [("WIDGETS", page)])
    try:
        deck._fade_in(page)
        previous = deck._anim
        deck._fade_in(page)
        current = deck._anim
        effect = page.graphicsEffect()
        previous.callback()
        assert deck._anim is current
        assert page.graphicsEffect() is effect
    finally:
        dispose(deck)


def test_native_surface_keeps_instant_transition(app):
    page = QLabel("synthetic native page")
    deck = dashboard.UnifiedDashboard(OrionBus(), [("GLOBE", page)])
    try:
        deck._fade_in(page)
        assert deck._anim is None
        assert page.graphicsEffect() is None
    finally:
        dispose(deck)


def test_deferred_rail_restore_runs_and_dies_with_deck(app, monkeypatch, tmp_path):
    from orion_core.gui import deck_favourites
    preferences = tmp_path / "deck.json"
    preferences.write_text('{"rail": true}', encoding="utf-8")
    monkeypatch.setattr(deck_favourites, "DECK_PREFERENCES_PATH", preferences)
    deck = dashboard.UnifiedDashboard(OrionBus(), [("WIDGETS", QLabel("page"))])
    try:
        app.processEvents()
        assert deck.sidebar_rail is True
    finally:
        dispose(deck)
    deck = dashboard.UnifiedDashboard(OrionBus(), [("WIDGETS", QLabel("page"))])
    timer = deck._rail_restore_timer
    dispose(deck)
    assert sip.isdeleted(timer)
    app.processEvents()
