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


def test_the_process_list_is_read_off_the_gui_thread(_app, monkeypatch):
    """28 Oct 2026: process_iter took 330-360 ms on the GUI thread every five
    seconds while the Load tab was open; now a thread reads, the loop fills."""
    import asyncio
    import threading

    from orion_core.bus import OrionBus
    from orion_core.gui.views import TelemetryView

    view = TelemetryView(OrionBus())
    readers: list[str] = []

    def read():
        readers.append(threading.current_thread().name)
        return [("4", "orion.exe", "1.0", "2.0"), ("8", "python.exe", "0.5", "1.0")]

    monkeypatch.setattr(view, "_read_processes", read)

    async def run():
        main = threading.current_thread().name
        view._refresh_processes()
        view._refresh_processes()                     # one read at a time
        for _ in range(100):
            await asyncio.sleep(0.01)
            if not view._proc_busy:
                break
        return main

    main = asyncio.run(run())
    assert readers and readers[0] != main and len(readers) == 1
    assert view.proc_table.rowCount() == 2
    assert view.proc_table.item(0, 1).text() == "orion.exe"
    view._refresh_processes()                         # no loop: still fills, in place
    assert view.proc_table.rowCount() == 2


# ── the activity log (audit 6 Oct 2026, A10) ─────────────────────────────────

def _log_view(_app):
    from orion_core.bus import OrionBus
    from orion_core.gui.views import LogConsoleView
    view = LogConsoleView(OrionBus())
    view.resize(700, 430)
    view.show()
    for i in range(180):
        view.write_log(f"event {i}")
    _app.processEvents()
    return view


def test_a_new_log_line_leaves_a_reader_where_they_are(_app):
    view = _log_view(_app)
    try:
        bar = view.log_box.verticalScrollBar()
        assert bar.maximum() > 0
        bar.setValue(0)
        view.write_log("background update while reading")
        _app.processEvents()
        assert bar.value() == 0
    finally:
        view.close()
        view.deleteLater()


def test_a_reader_at_the_end_keeps_following_the_log(_app):
    view = _log_view(_app)
    try:
        bar = view.log_box.verticalScrollBar()
        bar.setValue(bar.maximum())
        before = bar.maximum()
        for i in range(20):
            view.write_log(f"live {i}")
        _app.processEvents()
        assert bar.maximum() > before and bar.value() == bar.maximum()
    finally:
        view.close()
        view.deleteLater()


def test_a_reader_mid_log_with_a_selection_keeps_both(_app):
    view = _log_view(_app)
    try:
        bar = view.log_box.verticalScrollBar()
        cursor = view.log_box.textCursor()
        cursor.setPosition(100)
        cursor.setPosition(140, cursor.MoveMode.KeepAnchor)
        view.log_box.setTextCursor(cursor)               # (scrolls to show it)
        selected = view.log_box.textCursor().selectedText()
        bar.setValue(bar.maximum() // 2)
        middle = bar.value()
        for i in range(5):
            view.write_log(f"background {i}")
        _app.processEvents()
        assert bar.value() == middle
        assert view.log_box.textCursor().selectedText() == selected
    finally:
        view.close()
        view.deleteLater()


def test_pruning_at_the_retention_cap_keeps_the_reader_on_their_text(_app):
    from orion_core.bus import OrionBus
    from orion_core.gui.views import LogConsoleView
    view = LogConsoleView(OrionBus())
    view.resize(700, 430)
    view.show()
    try:
        view.log_box.setMaximumBlockCount(60)
        for i in range(60):
            view.write_log(f"event {i}")
        _app.processEvents()
        bar = view.log_box.verticalScrollBar()
        bar.setValue(20)
        first_visible = view.log_box.firstVisibleBlock().text()
        for i in range(5):                                 # prunes 5 from the top
            view.write_log(f"late {i}")
        _app.processEvents()
        assert bar.value() == 15
        assert view.log_box.firstVisibleBlock().text() == first_visible
    finally:
        view.close()
        view.deleteLater()


def test_the_reactor_gauge_follows_the_sample_and_marks_a_missing_gpu(_app):
    from orion_core.bus import OrionBus
    from orion_core.gui.hud_instruments import ReactorGauge
    from orion_core.gui.views import TelemetryView

    bus = OrionBus()
    view = TelemetryView(bus)
    gauge = view.reactor
    bus.telemetry_sample.emit({"cpu": 34.0, "gpu": None, "ram": 72.0, "net_percent": 8.0})
    assert gauge.target["cpu"] == 34.0 and gauge.target["ram"] == 72.0
    assert gauge.available["gpu"] is False and "GPU" not in gauge.accessibleDescription()
    for _ in range(60):
        gauge._advance()
    assert abs(gauge.shown["ram"] - 72.0) < 0.5 and not gauge._animating()
    standalone = ReactorGauge()
    standalone.set_sample({"cpu": 250.0})
    assert standalone.target["cpu"] == 100.0             # clamped
    standalone.resize(500, 230)
    standalone.grab()


def test_the_memory_page_draws_its_categories_as_a_constellation(_app):
    from PyQt6.QtCore import QPointF

    from orion_core.gui.hud_instruments import MemoryConstellation
    from orion_core.gui.views import MemoryMatrixView

    class Memory:
        def records(self, query="", limit=100):
            return []

        def category_counts(self):
            return {"knowledge": 120, "pack_ai": 4, "personal": 9, "long_term": 22}

    view = MemoryMatrixView(Memory())
    view.refresh()
    stars = view.constellation
    assert [name for name, _count in stars.categories][:2] == ["knowledge", "long_term"]
    stars.resize(900, 240)
    stars.grab()
    box = stars._box()
    brightest = QPointF(box.left() + 0.48 * box.width(), box.top() + 0.10 * box.height())
    assert stars.star_at(brightest) == ("knowledge", 120)
    from orion_core.gui.hud_instruments import _category_name
    assert _category_name("pack_ai") == "Pack AI"
    empty = MemoryConstellation()
    empty.resize(400, 200)
    empty.set_categories([])
    empty.grab()
