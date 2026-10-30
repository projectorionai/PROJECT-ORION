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


# ── the Markets chart (Mark XXXX.4) ────────────────────────────────────────────
#
# Owner, 28 Oct 2026: the stocks, ETFs and the FTSE "must represent as an
# amazing graph". The Today panel draws them from the session's prices.

def _yahoo(symbol, closes, *, currency="GBP", previous=100.0):
    start = 1_793_174_400
    return {"meta": {"symbol": symbol, "regularMarketPrice": closes[-1], "currency": currency,
                     "chartPreviousClose": previous, "regularMarketTime": start + 300 * len(closes),
                     "currentTradingPeriod": {"regular": {"start": start, "end": start + 30_600}}},
            "timestamp": [start + 300 * i for i in range(len(closes))],
            "indicators": {"quote": [{"close": closes}]}}


def test_the_session_prices_are_read_with_gaps_dropped_and_pence_shown_in_pounds():
    from orion_core import markets
    result = _yahoo("ISF.L", [1040.0, None, 1044.4], currency="GBp", previous=1035.0)
    times, points = markets.series(result)
    assert points == (1040.0, 1044.4) and len(times) == 2
    quote = markets.parse("ISF.L", "iShares FTSE 100", result["meta"])
    assert quote.previous == 1035.0 and quote.period is not None
    assert markets.price_text(quote) == "£10.44"
    index = markets.parse("^FTSE", "FTSE 100", _yahoo("^FTSE", [9876.5])["meta"])
    assert markets.price_text(index) == "9,876.50"


def test_the_owner_can_replace_a_watched_group(tmp_path):
    from orion_core import markets
    (tmp_path / markets.WATCH_FILE).write_text('{"Stocks": [["TSLA", "Tesla"]]}', encoding="utf-8")
    groups = markets.watchlist(tmp_path)
    assert groups["Stocks"] == (("TSLA", "Tesla"),)
    assert groups["Indices"][0] == ("^FTSE", "FTSE 100")
    (tmp_path / markets.WATCH_FILE).write_text("not json", encoding="utf-8")
    assert markets.watchlist(tmp_path) == markets.WATCH


def test_the_market_chart_draws_a_group_and_switches_on_a_click(_app):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    from orion_core import markets
    from orion_core.gui.market_chart import MarketChart, change_text, tone

    def quote(symbol, label, closes, previous):
        result = _yahoo(symbol, closes, previous=previous)
        made = markets.parse(symbol, label, result["meta"])
        made.times, made.points = markets.series(result)
        return made

    groups = {"Indices": [quote("^FTSE", "FTSE 100", [100, 101, 102], 100.0),
                          quote("^GSPC", "S&P 500", [100, 99, 98], 100.0)],
              "ETFs": [quote("VUSA.L", "Vanguard S&P 500", [50, 51], 50.0)]}
    chart = MarketChart()
    chart.resize(1100, 360)
    chart.set_quotes(groups)
    assert chart.selected().symbol == "^FTSE"
    assert change_text(chart.selected()).startswith("▲")
    assert tone(groups["Indices"][1]) != tone(groups["Indices"][0])   # falling is crimson
    chart.grab()                                     # draws the still layer, records targets
    assert not chart._plot.isEmpty()
    targets = {(kind, value): rect for rect, kind, value in chart._hits}
    assert ("symbol", "^GSPC") in targets and ("group", "ETFs") in targets

    def click(rect):
        chart.mousePressEvent(QMouseEvent(
            QMouseEvent.Type.MouseButtonPress, QPointF(rect.center()), QPointF(rect.center()),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))

    click(targets[("symbol", "^GSPC")])
    assert chart.selected().label == "S&P 500"
    assert "S&P 500" in chart.accessibleDescription()
    click(targets[("group", "ETFs")])
    assert chart.group == "ETFs" and chart.selected().symbol == "VUSA.L"
    chart.set_quotes({})
    assert chart.status == "Markets Unavailable" and chart.selected() is None
    chart.grab()


def test_a_heat_map_tile_opens_that_markets_chart(_app):
    from orion_core import markets

    def quote(symbol, label, closes, previous):
        result = _yahoo(symbol, closes, previous=previous)
        made = markets.parse(symbol, label, result["meta"])
        made.times, made.points = markets.series(result)
        return made

    window = _window(ToolResult("unused"))
    groups = {"Indices": [quote("^FTSE", "FTSE 100", [100, 101], 100.0)],
              "Stocks": [quote("NVDA", "Nvidia", [100, 97], 100.0),
                         quote("AAPL", "Apple", [100, 100.5], 100.0)]}
    window.market_chart.set_quotes(groups)
    window.market_heat.set_quotes(groups)
    heat = window.market_heat
    heat.resize(900, 260)
    window.market_view_buttons["Heat Map"].click()
    assert window.market_stack.currentWidget() is heat
    heat.grab()
    nvidia = next(rect for rect, _group, symbol in heat._tiles if symbol == "NVDA")
    apple = next(rect for rect, _group, symbol in heat._tiles if symbol == "AAPL")
    assert nvidia.width() > apple.width()                  # the bigger move, the wider tile
    assert heat.tile_at(nvidia.center()) == ("Stocks", "NVDA")
    heat.on_pick("Stocks", "NVDA")
    assert window.market_stack.currentWidget() is window.market_chart
    assert window.market_chart.selected().symbol == "NVDA"
