"""
Tests for the non-destructive resource-pressure monitor (Section 8), using
simulated readings.  Verifies: momentary spikes are ignored, elevated pressure
keeps the task running (with an after-task note), critical pressure checkpoints
and sheds background work while keeping the task, OS-instability asks the user,
offenders are identified without exposing command lines, and there is NO way to
terminate another process.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.resource_monitor import (
    PressureLevel,
    ProcessInfo,
    RecommendedAction,
    ResourceMonitor,
    ResourceReading,
    ResourceThresholds,
    identify_offenders,
)


def _monitor():
    ticks = {"t": 0.0}
    th = ResourceThresholds(
        elevated_cpu=85, critical_cpu=95, elevated_mem=85, critical_mem=95,
        elevated_hold_s=20, critical_hold_s=10, instability_swap=90,
    )
    mon = ResourceMonitor(thresholds=th, clock=lambda: ticks["t"])
    return mon, ticks


def _reading(cpu=10.0, mem=30.0, swap=0.0, t=0.0):
    return ResourceReading(cpu_percent=cpu, mem_percent=mem, swap_percent=swap, timestamp=t)


def test_nominal_continues():
    mon, _ = _monitor()
    a = mon.observe(_reading(cpu=20, mem=40, t=0))
    assert a.level is PressureLevel.NOMINAL
    assert a.action is RecommendedAction.CONTINUE
    assert a.preserve_task


def test_momentary_spike_is_ignored():
    mon, _ = _monitor()
    # A single high reading has not held for the window → not engaged.
    a = mon.observe(_reading(cpu=99, mem=40, t=0))
    assert a.engaged is False
    assert a.action is RecommendedAction.CONTINUE  # task runs on
    assert a.preserve_task


def test_sustained_elevated_notes_but_continues():
    mon, _ = _monitor()
    mon.observe(_reading(cpu=88, mem=40, t=0))
    a = mon.observe(_reading(cpu=88, mem=40, t=25))   # held past 20s window
    assert a.level is PressureLevel.ELEVATED
    assert a.engaged
    assert a.action is RecommendedAction.CONTINUE_AND_NOTE
    assert a.preserve_task
    # The event is queued to mention after the task finishes.
    notes = mon.drain_notifications()
    assert notes and notes[0]["level"] == "elevated"


def test_sustained_critical_checkpoints_and_sheds_but_keeps_task():
    mon, _ = _monitor()
    mon.observe(_reading(cpu=97, mem=50, t=0))
    a = mon.observe(_reading(cpu=97, mem=50, t=11))   # held past 10s window
    assert a.level is PressureLevel.CRITICAL
    assert a.action is RecommendedAction.CHECKPOINT_AND_REDUCE
    assert a.preserve_task                            # task NOT abandoned
    assert "self-improvement ticks" in a.shed


def test_instability_risk_asks_user():
    mon, _ = _monitor()
    # Physical memory near-full with heavy swap → genuine OS instability risk.
    mon.observe(_reading(cpu=50, mem=97, swap=95, t=0))
    a = mon.observe(_reading(cpu=50, mem=97, swap=95, t=11))
    assert a.level is PressureLevel.INSTABILITY_RISK
    assert a.action is RecommendedAction.CHECKPOINT_AND_ASK
    assert a.preserve_task is False                   # user must decide


def test_ram_and_swap_are_distinct_metrics():
    mon, _ = _monitor()
    mon.observe(_reading(cpu=10, mem=97, swap=95, t=0))
    a = mon.observe(_reading(cpu=10, mem=97, swap=95, t=11))
    assert "physical memory (RAM)" in a.affected_resources
    assert "swap" in a.affected_resources            # not merged with RAM


def test_offender_identified_without_command_line():
    procs = [
        ProcessInfo(pid=101, name="video_encoder", cpu_percent=180.0, mem_percent=12.0),
        ProcessInfo(pid=102, name="browser", cpu_percent=40.0, mem_percent=8.0),
    ]
    top = identify_offenders(procs, "cpu", limit=1)
    assert top[0]["name"] == "video_encoder"
    assert "cmdline" not in top[0] and "cmd" not in top[0]  # no args exposed


def test_engaged_assessment_carries_offender():
    mon, _ = _monitor()
    procs = [ProcessInfo(pid=1, name="renderer", cpu_percent=200.0, mem_percent=5.0)]
    mon.observe(_reading(cpu=97, mem=40, t=0))
    a = mon.observe(_reading(cpu=97, mem=40, t=11), processes=procs)
    assert a.offender and a.offender["name"] == "renderer"


def test_monitor_has_no_process_termination_capability():
    mon, _ = _monitor()
    for verb in ("kill", "terminate", "suspend", "throttle", "stop_process", "close_process"):
        assert not hasattr(mon, verb), f"monitor must not expose {verb}"
