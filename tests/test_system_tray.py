"""
System-tray presence (Mark XXII) — ORION resident in the notification area.

The tray itself needs a real desktop, so the tests exercise the pure helpers and
the no-tray degradation path, and confirm the close-to-tray wiring is in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import system_tray as st  # noqa: E402


# ── tooltip helper ───────────────────────────────────────────────────────────

def test_tooltip_names_orion_and_the_state():
    assert "O.R.I.O.N" in st.tooltip_for_state("LISTENING")
    assert "listening" in st.tooltip_for_state("LISTENING")
    assert "thinking" in st.tooltip_for_state("PROCESSING")
    assert "speaking" in st.tooltip_for_state("SPEAKING")


def test_tooltip_handles_blank_and_unknown_states():
    assert st.tooltip_for_state("") .endswith("ready")
    assert "custom" in st.tooltip_for_state("CUSTOM").lower()


def test_tray_available_returns_a_bool():
    assert isinstance(st.tray_available(), bool)


# ── no-tray degradation ──────────────────────────────────────────────────────

class _Signal:
    def __init__(self):
        self.calls = []
    def emit(self, *a):
        self.calls.append(a)
    def connect(self, *a):
        pass


class _Bus:
    def __init__(self):
        self.request_shutdown = _Signal()
        self.state = _Signal()
        self.safety_alert = _Signal()


class _Window:
    def __init__(self):
        self.shown = 0
        self.raised = 0
    def isMinimized(self):
        return False
    def show(self):
        self.shown += 1
    def raise_(self):
        self.raised += 1
    def activateWindow(self):
        pass


@pytest.fixture()
def no_tray(monkeypatch):
    monkeypatch.setattr(st, "tray_available", lambda: False)


def test_without_a_tray_it_is_inactive_and_safe(no_tray):
    tray = st.OrionTray(app=object(), window=_Window(), bus=_Bus())
    assert tray.active is False
    # every method is a safe no-op
    tray.show(); tray.hide(); tray.show_window(); tray.overlay()


def test_quit_requests_shutdown(no_tray):
    bus = _Bus()
    tray = st.OrionTray(app=object(), window=_Window(), bus=bus)
    tray.quit()
    assert bus.request_shutdown.calls        # a shutdown was requested


def test_show_window_restores_and_raises(no_tray):
    window = _Window()
    tray = st.OrionTray(app=object(), window=window, bus=_Bus())
    tray.show_window()
    assert window.shown == 1 and window.raised == 1


# ── close-to-tray wiring ──────────────────────────────────────────────────────
# Read the sources from disk rather than importing core_window / app: both pull
# in QtWebEngine at import time, which destabilises the offscreen Qt platform for
# every test that runs afterwards.

_ROOT = Path(__file__).resolve().parents[1] / "orion_core"


def test_close_event_hides_to_tray_when_resident():
    src = (_ROOT / "gui" / "core_window.py").read_text(encoding="utf-8")
    # closes to the tray only when resident AND not an explicit quit
    assert "_tray_active()" in src
    assert "_shutting_down" in src
    assert "event.ignore()" in src
    # the shutdown flag is wired so every quit path still shuts down
    assert "self.bus.request_shutdown.connect(self._mark_shutting_down)" in src


def test_app_makes_orion_resident_when_a_tray_exists():
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert "OrionTray(" in src
    assert "setQuitOnLastWindowClosed(False)" in src
