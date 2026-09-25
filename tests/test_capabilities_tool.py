"""
Tests for the capabilities_tool dispatcher method (dispatch_productivity.py)
— confirms it threads self.telemetry through to registries.report() so the
"Unhealthy modules" line (Track G) actually appears via the voice/text tool
path, not only when calling registries.report() directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dispatcher import OrionDispatcher
from orion_core.registries import SystemRegistries


class _StubHealth:
    def __init__(self, rows):
        self._rows = rows

    def snapshot(self):
        return list(self._rows)


class _StubTelemetry:
    def __init__(self, rows):
        self.health = _StubHealth(rows)


def _dispatcher() -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.registries = None
    d.telemetry = None
    return d


def test_capabilities_tool_reports_unavailable_with_no_registries():
    d = _dispatcher()
    result = d.capabilities_tool({})
    assert not result.ok


def test_capabilities_tool_passes_telemetry_through_to_the_report():
    d = _dispatcher()
    d.registries = SystemRegistries()
    d.registries.modules.register("dispatcher", object(), "router")
    d.telemetry = _StubTelemetry([{"name": "dispatcher", "status": "DOWN"}])
    result = d.capabilities_tool({})
    assert result.ok
    assert "Unhealthy modules: dispatcher" in result.text


def test_capabilities_tool_works_with_no_telemetry_attached():
    d = _dispatcher()
    d.registries = SystemRegistries()
    d.registries.modules.register("dispatcher", object(), "router")
    result = d.capabilities_tool({})
    assert result.ok
    assert "Unhealthy" not in result.text
