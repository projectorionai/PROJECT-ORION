"""
Tests for the agent consult panel surfacing its evidence trail (Mark XX
design-spec §6, High Impact item 6): AgentManager.dispatch() now returns a
ToolResult.evidence list when the specialist looked things up via its
toolbelt — WidgetDashboardWindow's CONSULT panel renders it as a "Sources
consulted (N)" strip instead of only ever showing prose, which is what
happened before even though the grounding work was already being done.

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

from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.gui.dashboard import WidgetDashboardWindow


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _StubOutlook:
    available = False


class _StubNotion:
    available = False


class _StubAgentManager:
    def __init__(self, result: ToolResult) -> None:
        self._result = result
        self.calls: list[tuple[str, str]] = []

    def describe(self):
        return [{"name": "coding", "title": "Coding Agent", "focus": "software engineering",
                 "calls": 0, "last_active": ""}]

    async def dispatch(self, request, agent_name="auto"):
        self.calls.append((request, agent_name))
        return self._result


def _window(result: ToolResult) -> WidgetDashboardWindow:
    return WidgetDashboardWindow(
        OrionBus(), _StubAgentManager(result), _StubOutlook(), _StubNotion(), None)


def test_consult_renders_the_evidence_trail_when_present(_app):
    result = ToolResult(
        "[Coding Agent]\nUse a context manager here.", ok=True,
        evidence=[
            {"tool": "memory_search", "ok": True, "summary": "memory_search(...) 3 hits"},
            {"tool": "knowledge_graph", "ok": False, "summary": "knowledge_graph(...) FAILED — offline"},
        ],
    )
    window = _window(result)

    async def _run():
        window.agent_input.setText("how should I structure this")
        window._consult_agent()
        await asyncio.sleep(0)   # let the created task run
        await asyncio.sleep(0)

    asyncio.run(_run())
    text = window.agent_output.toPlainText()
    assert "Sources consulted (2)" in text
    assert "✓" in text and "memory_search" in text
    assert "✗" in text and "knowledge_graph" in text


def test_consult_shows_no_sources_strip_when_evidence_is_none(_app):
    result = ToolResult("[Coding Agent]\nanswer from persona alone.", ok=True, evidence=None)
    window = _window(result)

    async def _run():
        window.agent_input.setText("what is a decorator")
        window._consult_agent()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert "Sources consulted" not in window.agent_output.toPlainText()
