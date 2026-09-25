"""
The SUBSYSTEM REGISTRY panel in the Command Centre.

ModuleRegistry has carried role/dependencies/version (plus health
cross-referenced from Telemetry) for every registered subsystem, but had no
GUI surface at all — it was only reachable as a text blob through the
`capabilities` dispatcher tool. This panel renders it.

These tests cover the rendering layer only; the registry logic itself is
already covered by test_registries_health_deps.py, so the stubs here are
deliberately thin. CommandCentreWindow's real __init__ wants seven live
subsystems, so the window is built by bypassing it and initialising only the
Qt base — the panel methods touch nothing but self.dispatcher/self.telemetry.

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
from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from orion_core.gui.command_centre import CommandCentreWindow  # noqa: E402
from orion_core.registries import SystemRegistries  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _StubHealth:
    def __init__(self, rows):
        self._rows = rows

    def snapshot(self):
        return list(self._rows)


class _StubTelemetry:
    def __init__(self, rows=()):
        self.health = _StubHealth(rows)


class _StubDispatcher:
    def __init__(self, registries=None):
        if registries is not None:
            self.registries = registries


def _window(dispatcher, telemetry):
    window = CommandCentreWindow.__new__(CommandCentreWindow)
    QMainWindow.__init__(window)
    window.dispatcher = dispatcher
    window.telemetry = telemetry
    # In the real window the grid layout owns this frame; here nothing would,
    # so Qt would delete it (and its children) the moment it fell out of
    # scope. Parent it to the window to keep the panel alive for the test.
    frame = window._build_capabilities_panel()
    frame.setParent(window)
    return window


def _registries():
    registries = SystemRegistries()
    registries.modules.register(
        "vision", object(), "screen capture, OCR and visual understanding",
        dependencies=["desktop"], version="20.0.0")
    registries.modules.register(
        "dispatcher", object(), "the tool-call router",
        dependencies=["memory", "vision"], version="20.0.0")
    return registries


# ── construction ─────────────────────────────────────────────────────────────

def test_panel_starts_empty(_app):
    window = _window(_StubDispatcher(_registries()), _StubTelemetry())
    assert window.capabilities_table.rowCount() == 0


def test_panel_has_the_five_expected_columns(_app):
    window = _window(_StubDispatcher(_registries()), _StubTelemetry())
    headers = [window.capabilities_table.horizontalHeaderItem(i).text()
               for i in range(window.capabilities_table.columnCount())]
    assert headers == ["Module", "Role", "Health", "Depends on", "Version"]


# ── rendering ────────────────────────────────────────────────────────────────

def test_refresh_renders_one_row_per_registered_module(_app):
    window = _window(_StubDispatcher(_registries()), _StubTelemetry())
    window._refresh_capabilities()
    assert window.capabilities_table.rowCount() == 2


def test_rows_are_sorted_by_module_name(_app):
    window = _window(_StubDispatcher(_registries()), _StubTelemetry())
    window._refresh_capabilities()
    assert window.capabilities_table.item(0, 0).text() == "dispatcher"
    assert window.capabilities_table.item(1, 0).text() == "vision"


def test_role_and_version_are_rendered(_app):
    window = _window(_StubDispatcher(_registries()), _StubTelemetry())
    window._refresh_capabilities()
    assert "tool-call router" in window.capabilities_table.item(0, 1).text()
    assert window.capabilities_table.item(0, 4).text() == "20.0.0"


def test_dependencies_are_comma_joined(_app):
    window = _window(_StubDispatcher(_registries()), _StubTelemetry())
    window._refresh_capabilities()
    assert window.capabilities_table.item(0, 3).text() == "memory, vision"


def test_a_module_with_no_dependencies_renders_an_empty_cell(_app):
    registries = SystemRegistries()
    registries.modules.register("bus", object(), "the signal bus")
    window = _window(_StubDispatcher(registries), _StubTelemetry())
    window._refresh_capabilities()
    assert window.capabilities_table.item(0, 3).text() == ""


def test_health_is_cross_referenced_from_telemetry(_app):
    telemetry = _StubTelemetry([
        {"name": "vision", "status": "DEGRADED"},
        {"name": "dispatcher", "status": "OK"},
    ])
    window = _window(_StubDispatcher(_registries()), telemetry)
    window._refresh_capabilities()
    assert window.capabilities_table.item(0, 2).text() == "OK"          # dispatcher
    assert window.capabilities_table.item(1, 2).text() == "DEGRADED"    # vision


def test_a_module_with_no_heartbeat_shows_unknown(_app):
    window = _window(_StubDispatcher(_registries()), _StubTelemetry())
    window._refresh_capabilities()
    assert window.capabilities_table.item(0, 2).text() == "UNKNOWN"


def test_the_note_summarises_registered_and_available_counts(_app):
    registries = _registries()
    registries.modules.register("missing", None, "a service that never came up")
    window = _window(_StubDispatcher(registries), _StubTelemetry())
    window._refresh_capabilities()
    note = window.capabilities_note.text()
    assert "3 subsystem(s) registered" in note
    assert "2 available" in note


def test_refresh_is_repeatable_without_duplicating_rows(_app):
    window = _window(_StubDispatcher(_registries()), _StubTelemetry())
    window._refresh_capabilities()
    window._refresh_capabilities()
    assert window.capabilities_table.rowCount() == 2


# ── defensive ────────────────────────────────────────────────────────────────

def test_refresh_is_safe_when_the_dispatcher_has_no_registries(_app):
    window = _window(_StubDispatcher(), _StubTelemetry())
    window._refresh_capabilities()   # must not raise
    assert window.capabilities_table.rowCount() == 0
    assert "not available" in window.capabilities_note.text().lower()


def test_refresh_is_safe_when_the_registry_lookup_raises(_app):
    class _BrokenModules:
        def all_described(self, _telemetry):
            raise RuntimeError("registry exploded")

    class _BrokenRegistries:
        modules = _BrokenModules()

    window = _window(_StubDispatcher(_BrokenRegistries()), _StubTelemetry())
    window._refresh_capabilities()   # must not raise
    assert window.capabilities_table.rowCount() == 0
