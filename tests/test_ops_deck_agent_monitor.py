"""
Tests for AgentMonitorPanel's enriched rendering — it used to show only
name/focus (a static roster with no bus wiring), which meant there was
nothing to see even once agents could look things up. It now renders the
calls/last_active fields AgentManager.describe() surfaces.

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.gui.ops_deck import AgentMonitorPanel  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _StubBus:
    def __getattr__(self, name):
        class _Sig:
            def connect(self, *a, **k):
                pass
        return _Sig()


class _StubAgents:
    def __init__(self, rows):
        self._rows = rows

    def describe(self):
        return self._rows


def test_panel_shows_not_yet_consulted_when_zero_calls(_app):
    agents = _StubAgents([
        {"name": "coding", "title": "Coding", "focus": "software", "calls": 0, "last_active": ""},
    ])
    panel = AgentMonitorPanel(_StubBus(), agents)
    text = panel.listing.item(0).text()
    assert "coding" in text
    assert "not yet consulted" in text


def test_panel_shows_call_count_and_last_active(_app):
    agents = _StubAgents([
        {"name": "coding", "title": "Coding", "focus": "software",
         "calls": 3, "last_active": "2026-08-01T09:00:00Z"},
    ])
    panel = AgentMonitorPanel(_StubBus(), agents)
    text = panel.listing.item(0).text()
    assert "3 call(s)" in text
    assert "2026-08-01T09:00:00Z" in text


def test_panel_handles_missing_agents_gracefully(_app):
    panel = AgentMonitorPanel(_StubBus(), None)
    assert panel.listing.item(0).text() == "Agent manager not available."
