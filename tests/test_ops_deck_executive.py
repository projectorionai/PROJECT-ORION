"""
Tests for ExecutivePanel (Mark X.14) — ExecutiveCore (decision/priority/
strategy engine) was constructed and registered with the dispatcher but had
no GUI panel, zero visibility beyond chatting with ORION about it. Same
category of gap the Diagnostics Centre panel had before it was fixed.

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.data import ToolResult  # noqa: E402
from orion_core.gui.ops_deck import ExecutivePanel, OperationsDeckView  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _StubBus:
    def __getattr__(self, name):
        class _Sig:
            def connect(self, *a, **k):
                pass
        return _Sig()


class _StubExecutiveCore:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.focus_calls = 0

    async def focus(self):
        self.focus_calls += 1
        if self._error is not None:
            raise self._error
        return self._result


def test_panel_shows_not_available_when_executive_core_is_none(_app):
    panel = ExecutivePanel(_StubBus(), None)
    assert "not available" in panel.summary.text().lower()


def test_refresh_async_renders_focus_text(_app):
    core = _StubExecutiveCore(result=ToolResult("Priority queue:\n  1. [task] ship it"))
    panel = ExecutivePanel(_StubBus(), core)
    asyncio.run(panel._refresh_async())
    assert "Priority queue" in panel.summary.text()


def test_refresh_async_handles_backend_failure_gracefully(_app):
    core = _StubExecutiveCore(error=RuntimeError("cognition snapshot failed"))
    panel = ExecutivePanel(_StubBus(), core)
    asyncio.run(panel._refresh_async())
    assert "unavailable" in panel.summary.text().lower()
    assert "cognition snapshot failed" in panel.summary.text()


def test_refresh_without_running_loop_does_not_raise(_app):
    """refresh() (the sync Qt-slot entry point) must degrade cleanly when
    called with no running asyncio loop, e.g. during widget construction in
    a plain test — not a qasync-driven app."""
    core = _StubExecutiveCore(result=ToolResult("fine"))
    panel = ExecutivePanel(_StubBus(), core)  # __init__ calls self.refresh()
    assert core.focus_calls == 0  # no loop was running, so no task ran


def test_operations_deck_view_wires_executive_panel(_app):
    view = OperationsDeckView(_StubBus(), executive_core=_StubExecutiveCore(result=ToolResult("x")))
    assert hasattr(view, "executive_panel")
    assert isinstance(view.executive_panel, ExecutivePanel)


def test_operations_deck_view_defaults_executive_core_to_none(_app):
    view = OperationsDeckView(_StubBus())
    assert view.executive_panel.executive_core is None
    assert "not available" in view.executive_panel.summary.text().lower()
