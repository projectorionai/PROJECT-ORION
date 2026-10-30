"""
Tests for the status-colour consolidation (Mark XX design-spec §9, High
Impact item 5): the audit found the OK/DEGRADED/DOWN triad independently
defined three times with three different RGB values — C.GOOD/WARN/BAD,
command_centre.py's own inline #39ff88/#ffcc44/#ff4d4d, and
diagnostics_centre.py's own #2ecc71/#f1c40f pair (plus a fourth, differently
-keyed PASS/WARN/FAIL triad in the same file, and a fifth hardcoded #2ecc71
literal in a crash panel). All five now resolve to the same three constants.

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
from PyQt6.QtGui import QColor  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.bus import OrionBus
from orion_core.constants import C
from orion_core.gui.command_centre import CommandCentreWindow
from orion_core.gui.diagnostics_centre import _CrashPanel, _HealthChecksPanel, _HealthPanel


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _StubHealth:
    def snapshot(self):
        return [
            {"name": "cognition", "status": "OK", "detail": "fine"},
            {"name": "forge", "status": "DEGRADED", "detail": "slow"},
            {"name": "sentinel", "status": "DOWN", "detail": "offline"},
        ]


class _StubTelemetry:
    health = _StubHealth()

    def tick_history(self):
        pass


# ── command_centre.py ────────────────────────────────────────────────────────

def test_command_centre_health_table_uses_the_shared_colour_tokens(_app):
    window = CommandCentreWindow(OrionBus(), _StubTelemetry(), None, None, None, None, None)
    window._refresh_health()
    expected = {"OK": C.GOOD, "DEGRADED": C.WARN, "DOWN": C.BAD}
    for row in range(window.health_table.rowCount()):
        status = window.health_table.item(row, 1).text()
        colour = window.health_table.item(row, 1).foreground().color()
        assert colour == QColor(expected[status]), f"{status} row uses the wrong colour"


def test_command_centre_scans_audio_devices_off_the_event_loop(_app, monkeypatch):
    """Owner's 17 Oct log: query_devices() blocked the event loop 325 ms from
    this page. The scan runs on a worker; the combos fill from its signal."""
    import threading
    import time
    from orion_core import audio_devices
    scanned_on: list[str] = []
    devices = [{"name": "Desk mic", "max_input_channels": 1, "max_output_channels": 0},
               {"name": "Monitor", "max_input_channels": 0, "max_output_channels": 2}]

    def fake_devices():
        scanned_on.append(threading.current_thread().name)
        return devices
    monkeypatch.setattr(audio_devices, "_devices", fake_devices)
    monkeypatch.setattr(audio_devices, "resolve_effective", lambda kind: 0 if kind == "input" else 1)
    window = CommandCentreWindow(OrionBus(), _StubTelemetry(), None, None, None, None, None)
    window._timer.stop()
    window._populate_audio_devices()
    end = time.monotonic() + 5
    while (window.input_combo.count() < 2 or window.output_combo.count() < 2
           or len(scanned_on) < 2) and time.monotonic() < end:
        _app.processEvents()
        time.sleep(0.01)
    assert scanned_on and threading.main_thread().name not in scanned_on
    assert window.input_combo.currentText() == "[0] Desk mic"
    assert window.output_combo.currentText() == "[1] Monitor"


# ── diagnostics_centre.py ────────────────────────────────────────────────────

def test_diagnostics_health_panel_uses_the_shared_colour_tokens(_app):
    panel = _HealthPanel()
    panel.render([
        {"name": "cognition", "status": "OK", "detail": ""},
        {"name": "forge", "status": "DEGRADED", "detail": ""},
        {"name": "sentinel", "status": "DOWN", "detail": ""},
    ])
    html = panel.body.text()
    assert f"color:{C.GOOD}" in html
    assert f"color:{C.WARN}" in html
    assert f"color:{C.BAD}" in html
    # The panel's own former triad must be gone.
    assert "#2ecc71" not in html
    assert "#f1c40f" not in html


def test_diagnostics_health_checks_panel_uses_the_shared_colour_tokens(_app):
    panel = _HealthChecksPanel()
    panel.render_checks("12:00:00", fails=1, warns=1, checks=[
        {"name": "imports", "status": "PASS", "detail": ""},
        {"name": "disk", "status": "WARN", "detail": ""},
        {"name": "network", "status": "FAIL", "detail": ""},
    ])
    html = panel.checks_label.text()
    assert f"color:{C.GOOD}" in html
    assert f"color:{C.WARN}" in html
    assert f"color:{C.BAD}" in html
    assert "#2ecc71" not in html
    assert "#f1c40f" not in html


def test_diagnostics_health_checks_panel_forge_section_uses_shared_tokens(_app):
    panel = _HealthChecksPanel()
    panel.render_forge({
        "live": 3, "session_ok": 2, "session_total": 3,
        "lessons": {"lessons": 5, "unresolved": 1},
        "quarantine": [{"tool": "x"}],
        "contract_failures": [{"name": "y"}],
    })
    html = panel.forge_label.text()
    assert f"color:{C.WARN}" in html    # quarantine
    assert f"color:{C.BAD}" in html     # contract failures


def test_diagnostics_crash_panel_all_clear_uses_shared_good_token(_app):
    panel = _CrashPanel()
    panel.render([])
    assert f"color:{C.GOOD}" in panel.body.text()
    assert "#2ecc71" not in panel.body.text()


def test_all_three_files_now_agree_on_the_same_good_colour_value(_app):
    # The whole point of the consolidation: one RGB value for "good"
    # everywhere it's used, not three.
    cc = CommandCentreWindow(OrionBus(), _StubTelemetry(), None, None, None, None, None)
    cc._refresh_health()
    ok_row = next(r for r in range(cc.health_table.rowCount())
                  if cc.health_table.item(r, 1).text() == "OK")
    cc_good = cc.health_table.item(ok_row, 1).foreground().color()

    dp = _HealthPanel()
    dp.render([{"name": "x", "status": "OK", "detail": ""}])
    assert f"color:{C.GOOD}" in dp.body.text()
    assert cc_good == QColor(C.GOOD)
