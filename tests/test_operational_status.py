"""
Operational status (Mark XXVI, §14) — the single "what is ORION doing?" surface.

The composer is pure (state→voice, offline handling, guarded field reads); the
strip is rendered head-less to prove it paints a real, non-black frame carrying
the status.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from orion_core.operational_status import OpStatus, compose, from_context  # noqa: E402


# ── pure composer ─────────────────────────────────────────────────────────────

def test_state_maps_to_a_voice_word():
    assert compose(state="SPEAKING").voice == "speaking"
    assert compose(state="LISTENING").voice == "listening"
    assert compose(state="PROCESSING").voice == "processing"
    assert compose(state="STANDBY").voice == "standby"
    assert compose(state="whatever").voice == "idle"


def test_offline_forces_offline_mode():
    assert compose(online=False).mode == "offline"
    assert compose(mode="B", online=True).mode == "offline"
    assert compose(online=True, mode="cloud").mode == "cloud"


def test_fields_are_bounded_and_normalised():
    s = compose(activity="x" * 200, focus="f" * 200, health="online")
    assert len(s.activity) <= 60 and len(s.focus) <= 40
    assert s.health == "ONLINE"


def test_line_and_dict_render_the_context():
    s = compose(online=True, state="SPEAKING", activity="researching",
                focus="thesis", mission="Neuro")
    line = s.line()
    assert "speaking" in line and "researching" in line and "thesis" in line
    d = s.as_dict()
    assert d["voice"] == "speaking" and d["mission"] == "Neuro"


# ── from_context: guarded live reads ──────────────────────────────────────────

def test_from_context_reads_focus_and_mission_and_health():
    from types import SimpleNamespace
    focus = SimpleNamespace(active=lambda: SimpleNamespace(label="deep work"))
    mission = SimpleNamespace(active=lambda: SimpleNamespace(title="Build Demo Game"))
    health = SimpleNamespace(overall=lambda: "ONLINE")
    s = from_context(state="SPEAKING", online=True, focus_engine=focus,
                     mission_engine=mission, health_model=health)
    assert s.focus == "deep work"
    assert s.mission == "Build Demo Game"
    assert s.health == "ONLINE"


def test_from_context_degrades_one_field_not_the_whole_strip():
    class _Boom:
        def active(self):
            raise RuntimeError("subsystem down")
    # A faulting focus engine must not take the snapshot down.
    s = from_context(state="LISTENING", focus_engine=_Boom())
    assert s.focus == "" and s.voice == "listening"


# ── the strip (rendered head-less) ────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_strip_renders_a_non_black_frame(qapp):
    from PyQt6.QtGui import QColor, QImage
    from orion_core.gui.status_strip import StatusStrip
    w = StatusStrip()
    w.resize(700, 32)
    w.set_status(compose(online=True, state="SPEAKING", activity="researching",
                         mission="Neuro"))
    img = QImage(700, 32, QImage.Format.Format_RGB32)
    img.fill(QColor("#000000"))
    w.render(img)
    colours = {img.pixel(x, y) for x in range(0, 700, 20) for y in range(0, 32, 8)}
    assert len(colours) > 1, "the strip rendered flat"
    assert w.__probe_state__()["voice"] == "speaking"


def test_the_strip_skips_a_degenerate_frame(qapp, monkeypatch):
    from orion_core.gui.status_strip import StatusStrip
    w = StatusStrip()
    monkeypatch.setattr(w, "width", lambda: 2)
    monkeypatch.setattr(w, "height", lambda: 2)
    assert w.paintEvent(None) is None      # returns before constructing a QPainter


def test_health_colour_tracks_state(qapp):
    from orion_core.gui.status_strip import _health_colour, _OK, _WARN, _OFF
    assert _health_colour(compose(online=True, health="ONLINE")).name() == _OK.name()
    assert _health_colour(compose(online=True, health="DEGRADED")).name() == _WARN.name()
    assert _health_colour(compose(online=False)).name() == _OFF.name()


def test_the_deck_shows_and_updates_the_status_strip(qapp):
    from PyQt6.QtWidgets import QWidget
    from orion_core.bus import OrionBus
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    bus = OrionBus()
    deck = UnifiedDashboard(bus, [("BRAIN", QWidget()), ("CHESS", QWidget())])
    assert hasattr(deck, "status_strip")
    bus.state.emit("SPEAKING")
    assert deck.status_strip.__probe_state__()["voice"] == "speaking"
    bus.state.emit("LISTENING")
    assert deck.status_strip.__probe_state__()["voice"] == "listening"


# ── the live provider (mission dicts, connectivity, guarded) ────────────────

def test_from_context_reads_a_mission_dict():
    from types import SimpleNamespace
    engine = SimpleNamespace(current=lambda: {"title": "Build Demo Game"})
    assert from_context(state="", mission_engine=engine).mission == "Build Demo Game"


def test_from_context_still_accepts_an_active_style_engine():
    from types import SimpleNamespace
    engine = SimpleNamespace(active=lambda: SimpleNamespace(name="Neuro"))
    assert from_context(state="", mission_engine=engine).mission == "Neuro"


def test_the_deck_uses_a_status_provider_when_given_one(qapp):
    from PyQt6.QtWidgets import QWidget
    from orion_core.bus import OrionBus
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    deck = UnifiedDashboard(OrionBus(), [("BRAIN", QWidget())])
    deck.attach_status_provider(
        lambda state: compose(state=state, online=False, focus="thesis",
                              mission="Neuro"))
    deck.bus.state.emit("SPEAKING")
    probe = deck.status_strip.__probe_state__()
    assert probe["voice"] == "speaking"
    assert probe["focus"] == "thesis" and probe["mission"] == "Neuro"
    assert probe["online"] is False


def test_a_faulting_provider_never_blanks_the_strip(qapp):
    from PyQt6.QtWidgets import QWidget
    from orion_core.bus import OrionBus
    from orion_core.gui.unified_dashboard import UnifiedDashboard

    def _boom(_state):
        raise RuntimeError("engine down")

    deck = UnifiedDashboard(OrionBus(), [("BRAIN", QWidget())])
    deck.attach_status_provider(_boom)
    deck.bus.state.emit("LISTENING")
    assert deck.status_strip.__probe_state__()["voice"] == "listening"


def test_the_app_wires_the_provider_with_the_real_apis():
    src = (Path(__file__).resolve().parents[1] / "orion_core"
           / "app.py").read_text(encoding="utf-8")
    assert "deck.attach_status_provider(" in src
    assert "monitor.is_online()" in src, "connectivity exposes is_online(), not .online"
