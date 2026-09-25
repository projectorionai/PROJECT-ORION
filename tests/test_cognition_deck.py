"""
COGNITION deck page (Mark XXVI, Phase 1).

The page is a *view* over the existing study/focus engines, so it is tested at two
levels: the pure view-model reducers (no Qt), and the widget itself rendered
head-less into an image buffer with a state-probe hook — the visual-CI method for
a QPainter surface.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from orion_core.focus import FocusEngine, FocusStore  # noqa: E402
from orion_core.gui.cognition_deck import (  # noqa: E402
    CognitionDeckView,
    FocusPanel,
    FocusRing,
    StudyPanel,
    focus_view,
    insight_view,
    study_view,
)
from orion_core.study import StudyEngine, StudyStore  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def engines(tmp_path):
    study = StudyEngine(StudyStore(tmp_path / "s.db"))
    focus = FocusEngine(FocusStore(tmp_path / "f.db"))
    return study, focus


def _bus():
    from orion_core.bus import OrionBus
    return OrionBus()


# ── pure reducers (no Qt) ─────────────────────────────────────────────────────

def test_focus_view_when_unavailable():
    vm = focus_view(None)
    assert vm["available"] is False and vm["active"] is False


def test_focus_view_reflects_a_running_block(engines):
    _study, focus = engines
    focus.start("revise action potentials", preset="pomodoro")
    vm = focus_view(focus)
    assert vm["active"] is True
    assert vm["label"] == "revise action potentials"
    assert 0.0 <= vm["progress"] <= 1.0
    assert vm["planned"] == 25


def test_study_view_counts_due_and_mastery(engines):
    study, _focus = engines
    study.add("What is myelin?", "an insulating sheath", deck="Neuro")
    vm = study_view(study)
    assert vm["available"] is True
    assert vm["due"] >= 1
    assert vm["total"] >= 1
    assert any(d["deck"] == "Neuro" for d in vm["decks"])


def test_insight_view_combines_both(engines):
    study, focus = engines
    study.add("q", "a")
    vm = insight_view(study, focus)
    assert "focus" in vm and "study" in vm and "focus_stats" in vm


# ── the widget, rendered head-less ────────────────────────────────────────────

def test_the_page_renders_a_real_frame(qapp, engines):
    from PyQt6.QtGui import QColor, QImage

    study, focus = engines
    study.add("Resting potential?", "-70 mV", deck="Neuro")
    focus.start("write results", preset="deep")

    view = CognitionDeckView(_bus(), study=study, focus=focus)
    view.resize(760, 620)
    view.refresh_all()

    img = QImage(760, 620, QImage.Format.Format_RGB32)
    img.fill(QColor("#0b0f14"))
    view.render(img)
    colours = {img.pixel(x, y) for x in range(0, 760, 40) for y in range(0, 620, 40)}
    assert len(colours) > 1, "the page rendered as a flat fill"

    probe = view.__probe_state__()
    assert probe["study"]["due"] >= 1
    assert probe["focus"]["active"] is True


def test_a_hidden_page_never_refreshes(qapp):
    view = CognitionDeckView(_bus())
    calls: list[int] = []
    view.refresh_all = lambda: calls.append(1)   # type: ignore[method-assign]
    view._tick()                                  # not shown → isVisible() False
    assert calls == []


def test_the_focus_ring_skips_a_degenerate_frame(qapp, monkeypatch):
    ring = FocusRing()
    monkeypatch.setattr(ring, "width", lambda: 2)
    monkeypatch.setattr(ring, "height", lambda: 2)
    # Returns before constructing a QPainter — the black-flicker guard.
    assert ring.paintEvent(None) is None


# ── interaction ───────────────────────────────────────────────────────────────

def test_starting_a_block_from_the_focus_panel(qapp, engines):
    _study, focus = engines
    panel = FocusPanel(_bus(), focus)
    panel.intention.setText("revise")
    panel._start()
    active = focus.active()
    assert active is not None and active.label == "revise"


def test_the_study_review_and_grade_flow(qapp, engines):
    study, _focus = engines
    study.add("Q?", "A", deck="D")
    panel = StudyPanel(_bus(), study)
    panel._next()
    assert panel._card is not None
    panel._reveal()
    assert panel._revealed is True
    panel._grade(5)
    assert study.store.all()[0].reviews == 1


def test_panels_survive_missing_engines(qapp):
    # Constructed before an engine is attached — must not crash, just disable.
    view = CognitionDeckView(_bus(), study=None, focus=None)
    view.refresh_all()
    assert view.__probe_state__()["study"]["available"] is False


# ── registration ──────────────────────────────────────────────────────────────

def test_cognition_is_registered_without_growing_the_zone_taxonomy():
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    assert "COGNITION" in UnifiedDashboard.ZONE_PAGES["OPERATIONS"]
    assert len(UnifiedDashboard.ZONE_ORDER) == 12   # still twelve zones


def test_the_app_builds_and_registers_the_page():
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8")
    assert "CognitionDeckView(" in src
    assert '("COGNITION", cognition_deck)' in src
