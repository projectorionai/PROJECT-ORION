"""
Tests for the display_info dispatcher tool's set_name action (dispatch_
vision.py) — the voice/text-reachable surface for naming a monitor
contextually (Track I, Mark XX architectural-audit pass).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dispatcher import OrionDispatcher


class _StubDisplay:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._summary = "Displays: *[0] Primary Monitor 1920x1080 @ (0,0) scale 100%"

    def set_custom_name(self, device, label):
        from orion_core.data import ToolResult
        self.calls.append((device, label))
        return ToolResult(f"'{device}' will now be called '{label}'.")

    def refresh(self):
        pass

    def summary(self):
        return self._summary


def _dispatcher(display=None) -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.display = display
    return d


def test_display_info_reports_unavailable_with_no_display_manager():
    d = _dispatcher(None)
    result = d.display_info({})
    assert not result.ok


def test_display_info_default_action_returns_the_summary():
    display = _StubDisplay()
    d = _dispatcher(display)
    result = d.display_info({})
    assert result.ok
    assert "Primary Monitor" in result.text


def test_display_info_set_name_calls_set_custom_name():
    display = _StubDisplay()
    d = _dispatcher(display)
    result = d.display_info({
        "action": "set_name", "device": "\\\\.\\DISPLAY1", "label": "Coding Monitor"})
    assert result.ok
    assert display.calls == [("\\\\.\\DISPLAY1", "Coding Monitor")]
    assert "Coding Monitor" in result.text


def test_display_info_rename_alias_also_works():
    display = _StubDisplay()
    d = _dispatcher(display)
    result = d.display_info({
        "action": "rename", "monitor": "\\\\.\\DISPLAY2", "name": "Gaming Monitor"})
    assert result.ok
    assert display.calls == [("\\\\.\\DISPLAY2", "Gaming Monitor")]
