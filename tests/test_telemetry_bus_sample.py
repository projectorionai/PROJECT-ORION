"""
TelemetryView's CPU/RAM/network graphs used to be pushed into directly by the
Core Window's start_telemetry() loop (a direct widget reference the Core
Window no longer holds now that it's face-only).  The view now self-wires to
bus.telemetry_sample instead, the same self-wiring pattern every other deck
view already uses for its own bus signals.

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


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def test_telemetry_sample_updates_the_graphs(_app):
    from orion_core.bus import OrionBus
    from orion_core.gui.views import TelemetryView

    bus = OrionBus()
    view = TelemetryView(bus)
    bus.telemetry_sample.emit({"cpu": 42.5, "ram": 61.0, "net_percent": 12.0})
    assert view.cpu_graph.value == pytest.approx(42.5)
    assert view.ram_graph.value == pytest.approx(61.0)
    assert view.net_graph.value == pytest.approx(12.0)


def test_telemetry_sample_ignores_non_dict_payloads(_app):
    from orion_core.bus import OrionBus
    from orion_core.gui.views import TelemetryView

    bus = OrionBus()
    view = TelemetryView(bus)
    bus.telemetry_sample.emit("not a dict")  # must not raise
    assert view.cpu_graph.value == 0.0
