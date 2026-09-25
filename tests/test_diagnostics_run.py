"""
Tests for DiagnosticsEngine.run() — the periodic autonomous self-diagnostic
loop (Mark X.14) that keeps the Diagnostics Centre panel fresh without the
user asking for it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.diagnostics import DiagnosticsEngine


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def test_run_ticks_repeatedly_on_the_configured_interval(monkeypatch):
    engine = DiagnosticsEngine(_StubBus(), memory=None)
    monkeypatch.setattr(engine, "FIRST_TICK_DELAY_S", 0)
    monkeypatch.setattr(engine, "INTERVAL_S", 0)
    calls: list[int] = []

    async def _fake_run_full():
        calls.append(1)
        if len(calls) >= 3:
            raise asyncio.CancelledError()

    monkeypatch.setattr(engine, "run_full", _fake_run_full)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.run())
    assert len(calls) == 3


def test_run_survives_a_failed_tick_and_keeps_going(monkeypatch):
    """A single bad run_full() (e.g. a transient DB lock) must not kill the
    autonomous loop — the whole point is the panel stays fresh indefinitely."""
    engine = DiagnosticsEngine(_StubBus(), memory=None)
    monkeypatch.setattr(engine, "FIRST_TICK_DELAY_S", 0)
    monkeypatch.setattr(engine, "INTERVAL_S", 0)
    calls: list[int] = []

    async def _flaky_run_full():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("transient failure")
        if len(calls) >= 4:
            raise asyncio.CancelledError()

    monkeypatch.setattr(engine, "run_full", _flaky_run_full)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.run())
    assert len(calls) == 4  # the RuntimeError at call 2 did not stop the loop


def test_run_waits_for_the_first_tick_delay(monkeypatch):
    engine = DiagnosticsEngine(_StubBus(), memory=None)
    monkeypatch.setattr(engine, "INTERVAL_S", 0)
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def _tracking_sleep(seconds):
        slept.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _tracking_sleep)

    async def _fake_run_full():
        raise asyncio.CancelledError()

    monkeypatch.setattr(engine, "run_full", _fake_run_full)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.run())
    assert slept[0] == engine.FIRST_TICK_DELAY_S
