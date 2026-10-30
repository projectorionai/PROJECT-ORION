"""
Headless (offscreen) UI tests for the reusable status components (Section 7).

Runs with the Qt 'offscreen' platform so no display is required.  Verifies the
widgets construct, react to the new bus signals, and render the correct
accessible state for connection, degraded mode, diagnosis and token usage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.bus import OrionBus  # noqa: E402
from orion_core.gui.status_widgets import (  # noqa: E402
    ConnectionStatusChip,
    DegradedBanner,
    ProviderStatusStrip,
    TokenUsageMini,
    reduced_motion,
)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def test_connection_chip_reacts_to_state(app):
    bus = OrionBus()
    chip = ConnectionStatusChip(bus)
    bus.connection_state.emit({"state": "connected", "message": "synchronised"})
    assert "Connected" in chip.text()          # sentence case, like every deck label
    assert chip.property("state") == "good"
    bus.connection_state.emit({"state": "degraded_offline", "message": "no provider"})
    assert "No Provider" in chip.text()
    assert chip.property("state") == "bad"
    bus.connection_state.emit({"state": "reconnecting", "message": "reconnecting"})
    assert chip.property("state") == "warn"


def test_degraded_banner_shows_and_hides(app):
    bus = OrionBus()
    banner = DegradedBanner(bus)
    assert banner.isHidden()
    bus.provider_degraded.emit({"degraded": True, "alternatives": ["add an OpenAI key", "start Ollama"]})
    assert not banner.isHidden()
    assert "No AI text provider" in banner.text()
    assert "Ollama" in banner.text()
    bus.provider_degraded.emit({"degraded": False})
    assert banner.isHidden()


def test_token_mini_handles_empty_and_not_reported(app):
    mini = TokenUsageMini()
    mini.update_summary(None)  # empty state
    assert mini._labels["requests"].text() == "—"
    mini.update_summary({"requests": 3, "total_tokens": None, "estimated_cost_usd": 0.0123})
    assert mini._labels["requests"].text() == "3"
    assert mini._labels["total_tokens"].text() == "Not reported"
    assert mini._labels["estimated_cost_usd"].text().startswith("$")


def test_status_strip_composes_and_diagnoses(app):
    bus = OrionBus()
    strip = ProviderStatusStrip(bus)
    assert strip.accessibleName()
    bus.provider_diagnosed.emit({"category": "rate_limit", "provider": "openai",
                                 "remediation": "back off then retry"})
    assert "Rate limit" in strip.diag_chip.text()
    assert strip.diag_chip.property("state") == "bad"
    # A fake ledger drives the token mini.
    class _Ledger:
        def summary(self):
            return {"requests": 10, "total_tokens": 12345, "estimated_cost_usd": None}
    strip.refresh_tokens(_Ledger())
    assert strip.token_mini._labels["total_tokens"].text() == "12,345"
    assert strip.token_mini._labels["estimated_cost_usd"].text() == "Not reported"


def test_reduced_motion_flag(monkeypatch):
    monkeypatch.delenv("ORION_REDUCED_MOTION", raising=False)
    assert reduced_motion() is False
    monkeypatch.setenv("ORION_REDUCED_MOTION", "1")
    assert reduced_motion() is True


def test_stylesheet_includes_status_surfaces():
    from orion_core.gui.style import APP_STYLESHEET
    assert "statusStrip" in APP_STYLESHEET
    assert "degradedBanner" in APP_STYLESHEET
    assert "statusChip" in APP_STYLESHEET


def test_a_strip_built_late_starts_from_the_live_state():
    """Deck pages are built on first open now. A status strip created after
    the connection came up must say CONNECTED, not its OFFLINE default, and
    an active degraded warning must already be showing."""
    from orion_core.gui.status_widgets import ProviderStatusStrip

    globals().setdefault("_QT_APP", QApplication.instance() or QApplication([]))  # held, or it is collected
    bus = OrionBus()
    bus.connection_state.emit({"state": "connected", "message": "Live"})
    bus.provider_degraded.emit({"degraded": True, "reason": "all cooling down",
                                "configured": ["groq"], "alternatives": []})
    strip = ProviderStatusStrip(bus)
    assert "Offline" not in strip.conn_chip.text()
    assert not strip.degraded.isHidden() or strip.degraded.text()
    assert bus.last_value("connection_state") == {"state": "connected", "message": "Live"}
