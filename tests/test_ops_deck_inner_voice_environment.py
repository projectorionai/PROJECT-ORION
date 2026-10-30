"""
Tests for the Operations environment panel. Reflections have their own useful
home in Cognition; Operations must not mount a duplicate notes panel.

Headless (offscreen Qt) — no display required.  Network calls are mocked;
these never touch a real geolocation/weather provider.
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

from orion_core.gui import ops_deck  # noqa: E402
from orion_core.gui.ops_deck import (  # noqa: E402
    EnvironmentPanel,
    OperationsDeckView,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _Signal:
    def __init__(self) -> None:
        self._slots = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def emit(self, *args) -> None:
        for slot in self._slots:
            slot(*args) if args else slot()


class _StubBus:
    def __init__(self) -> None:
        self.thought_delta = _Signal()
        self.thought = _Signal()
        self.log = _Signal()

    def __getattr__(self, name):
        return _Signal()


# ── EnvironmentPanel ─────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeGetContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, by_prefix: dict) -> None:
        self._by_prefix = by_prefix

    def get(self, url, *a, **k):
        for prefix, response in self._by_prefix.items():
            if url.startswith(prefix):
                return _FakeGetContext(response)
        raise AssertionError(f"unexpected URL requested: {url}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_refresh_environment_widgets_success(_app, monkeypatch):
    geo_payload = {
        "latitude": 51.5, "longitude": -0.12,
        "city": "London", "region": "England", "country_name": "United Kingdom",
    }
    weather_payload = {"current": {
        "temperature_2m": 18.0, "relative_humidity_2m": 55,
        "wind_speed_10m": 12.0, "weather_code": 1,
    }}
    session = _FakeSession({
        "https://ipapi.co": _FakeResponse(200, geo_payload),
        "https://api.open-meteo.com": _FakeResponse(200, weather_payload),
    })
    monkeypatch.setattr(ops_deck, "ClientSession", lambda *a, **k: session)

    panel = EnvironmentPanel(_StubBus())
    ok = asyncio.run(panel.refresh_environment_widgets())
    assert ok is True
    assert "London" in panel.location_label.text()
    assert "18.0" in panel.weather_label.text()


def test_refresh_environment_widgets_falls_back_to_second_geo_provider(_app, monkeypatch):
    weather_payload = {"current": {
        "temperature_2m": 10.0, "relative_humidity_2m": 80,
        "wind_speed_10m": 5.0, "weather_code": 3,
    }}

    class _FirstProviderFailsSession(_FakeSession):
        def get(self, url, *a, **k):
            if url.startswith("https://ipapi.co"):
                return _FakeGetContext(_FakeResponse(429, {}))
            return super().get(url, *a, **k)

    session = _FirstProviderFailsSession({
        "https://ipwho.is": _FakeResponse(200, {
            "latitude": 40.7, "longitude": -74.0,
            "city": "New York", "region": "NY", "country": "USA",
        }),
        "https://api.open-meteo.com": _FakeResponse(200, weather_payload),
    })
    monkeypatch.setattr(ops_deck, "ClientSession", lambda *a, **k: session)

    panel = EnvironmentPanel(_StubBus())
    ok = asyncio.run(panel.refresh_environment_widgets())
    assert ok is True
    assert "New York" in panel.location_label.text()


def test_refresh_environment_widgets_reports_failure_when_all_providers_fail(_app, monkeypatch):
    class _AlwaysFailSession:
        def get(self, url, *a, **k):
            return _FakeGetContext(_FakeResponse(500, {}))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(ops_deck, "ClientSession", lambda *a, **k: _AlwaysFailSession())

    panel = EnvironmentPanel(_StubBus())
    ok = asyncio.run(panel.refresh_environment_widgets())
    assert ok is False
    assert "unavailable" in panel.location_label.text().lower()
    assert "unavailable" in panel.weather_label.text().lower()


# ── OperationsDeckView wiring ────────────────────────────────────────────────

def test_operations_keeps_environment_without_duplicate_reflections(_app):
    view = OperationsDeckView(_StubBus())
    assert not hasattr(view, "inner_voice_panel")
    assert hasattr(view, "environment_panel")
    assert isinstance(view.environment_panel, EnvironmentPanel)
