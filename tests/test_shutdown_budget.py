"""Shutdown has a time budget, and the pieces have to add up.

Three bounded waits sit on the shutdown path, and a watchdog behind them all
force-exits the process if it takes too long:

    _await_farewell(limit)      let ORION finish the goodbye
    + 1.3s tail                 the output buffer draining after "idle"
    + 2 x STOP_JOIN_SECONDS     waiting for the audio threads to leave native code
    ------------------------------------------------------------------
    must stay under _arm_shutdown_watchdog(grace)

These were independent numbers in three files, and nothing related them. Making
the farewell wait work (it had been returning instantly on a missing method)
and then joining the audio threads pushed the worst legitimate shutdown to
25.3s against a 25s grace — the watchdog would have killed a shutdown that was
behaving, mid-goodbye, and looked exactly like the hang it exists to catch.

Offline: reads defaults, runs nothing.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import app  # noqa: E402
from orion_core.audio import SpeechQueueManager  # noqa: E402

#: The settle time _await_farewell sleeps once the pipeline reports idle.
TAIL_SECONDS = 1.3


def _default(func, name: str) -> float:
    return float(inspect.signature(func).parameters[name].default)


def _worst_legitimate_shutdown() -> float:
    return (_default(app._await_farewell, "limit")
            + TAIL_SECONDS
            + 2 * SpeechQueueManager.STOP_JOIN_SECONDS)


def test_a_working_shutdown_finishes_before_the_watchdog_fires():
    grace = _default(app._arm_shutdown_watchdog, "grace")
    worst = _worst_legitimate_shutdown()
    assert worst < grace, (
        f"a legitimate shutdown can take {worst:.1f}s but the watchdog "
        f"force-exits at {grace:.1f}s — it would kill a working shutdown "
        f"mid-goodbye")


def test_there_is_real_margin_not_a_coincidence():
    """Teardown does other work too: closing databases, cancelling tasks,
    stopping a dozen services. The farewell and the audio joins must not eat
    the entire budget."""
    grace = _default(app._arm_shutdown_watchdog, "grace")
    assert grace - _worst_legitimate_shutdown() >= 3.0


def test_the_tail_this_arithmetic_assumes_is_really_there():
    """If the tail changes and this constant does not, the sum is fiction."""
    source = inspect.getsource(app._await_farewell)
    assert f"asyncio.sleep({TAIL_SECONDS})" in source


def test_the_joins_are_short_enough_to_be_spent_twice():
    """stop() waits for both audio threads, so the bound is paid twice."""
    assert SpeechQueueManager.STOP_JOIN_SECONDS <= 3.0


# ── the budget only means anything if teardown actually gets to run ─────────
#
# Every quit path ends in QApplication.quit(), which stops the qasync loop.
# Until 2026-09-23 that happened the moment shutdown was requested, so the
# teardown this file budgets was abandoned a few steps in. A plain asyncio
# loop's stop() is the same event from run_until_complete()'s point of view.

async def _teardown(loop, steps, stops=1):
    steps.append("begin")
    for _ in range(stops):
        loop.stop()                      # what QApplication.quit() does to qasync
        await asyncio.sleep(0)
    steps.append("end")
    return True


def _drive(stops, requested):
    loop = asyncio.new_event_loop()
    steps: list[str] = []
    try:
        return app._run_until_torn_down(
            loop, _teardown(loop, steps, stops), requested=lambda: requested), steps
    finally:
        loop.close()


def test_teardown_finishes_when_the_loop_is_stopped_under_it():
    result, steps = _drive(stops=1, requested=True)
    assert result is True and steps == ["begin", "end"]


def test_an_unrequested_stop_is_still_a_fault():
    with pytest.raises(RuntimeError):
        _drive(stops=1, requested=False)


def test_resuming_is_bounded():
    with pytest.raises(RuntimeError):
        _drive(stops=10, requested=True)


def test_the_quit_button_no_longer_stops_the_loop_itself():
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    assert "request_shutdown.connect(_app.quit)" not in source
