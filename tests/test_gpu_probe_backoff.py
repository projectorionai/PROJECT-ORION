"""
The GPU meters must never launch a process on the event loop.

History
-------
`CommandCentre._gpu` shelled out to nvidia-smi on the Qt thread, which under
qasync is also the asyncio event loop that paints the face and services the
audio callback. First its cache check was backwards (a machine with no NVIDIA
GPU spawned a doomed process once a second); once fixed it still ran
nvidia-smi every five seconds — MEASURED at ~80 ms per call, a visible hitch
each time, with a 2 s timeout at worst.

It now reads `gpu_stats.latest()`: NVML in-process when available (0.02 ms),
otherwise the last reading the background telemetry loop took. No subprocess
is ever launched by these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import gpu_stats  # noqa: E402


class _Probe:
    """The real method, on an object with nothing else attached."""

    from orion_core.gui.command_centre import CommandCentreWindow as _CC

    _gpu = _CC._gpu


@pytest.fixture
def no_processes(monkeypatch):
    launched = []
    monkeypatch.setattr(gpu_stats.subprocess, "run",
                        lambda cmd, **kw: launched.append(cmd))
    return launched


@pytest.fixture
def fresh_state(monkeypatch):
    monkeypatch.setattr(gpu_stats, "_state", {
        "init": True, "ok": True, "handle": None, "name": "", "backend": "nvidia-smi",
        "exe": "nvidia-smi", "at": 0.0, "cache": None})
    return gpu_stats._state


def test_the_gui_thread_never_launches_nvidia_smi(no_processes, fresh_state):
    for _ in range(20):
        assert _Probe()._gpu() == ("n/a", 0.0)
    assert no_processes == []


def test_the_gui_thread_shows_the_background_reading(no_processes, fresh_state):
    fresh_state["cache"] = {"available": True, "util": 41.0}
    assert _Probe()._gpu() == ("41%", 41.0)
    assert no_processes == []


def test_nvml_is_sampled_directly(monkeypatch, fresh_state):
    fresh_state["backend"] = "nvml"
    monkeypatch.setattr(gpu_stats, "sample", lambda: {"available": True, "util": 12.0})
    assert _Probe()._gpu() == ("12%", 12.0)


def test_a_machine_without_a_gpu_reads_na(no_processes, monkeypatch):
    monkeypatch.setattr(gpu_stats, "_state", {
        "init": True, "ok": False, "handle": None, "name": "", "backend": "",
        "exe": None, "at": 0.0, "cache": None})
    assert _Probe()._gpu() == ("n/a", 0.0)
    assert no_processes == []


def test_the_probe_never_raises(monkeypatch):
    def _explode():
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(gpu_stats, "latest", _explode)
    assert _Probe()._gpu() == ("n/a", 0.0)


def test_the_nvidia_smi_fallback_is_sampled_rarely(monkeypatch, fresh_state):
    calls = []

    class _Result:
        returncode = 0
        stdout = "RTX,41,1000,8000,50\n"
        stderr = ""

    monkeypatch.setattr(gpu_stats.subprocess, "run",
                        lambda cmd, **kw: (calls.append(cmd), _Result())[1])
    clock = [100.0]
    monkeypatch.setattr(gpu_stats.time, "monotonic", lambda: clock[0])
    for _ in range(4):
        gpu_stats.sample()
        clock[0] += 1.0
    assert len(calls) == 1, "nvidia-smi must not be relaunched every second"
    clock[0] += gpu_stats._SMI_INTERVAL_S
    gpu_stats.sample()
    assert len(calls) == 2
