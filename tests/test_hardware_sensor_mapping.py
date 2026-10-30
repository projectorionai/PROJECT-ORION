import sys
from types import SimpleNamespace

from orion_core import hardware_telemetry as hardware


def test_windows_worker_initialises_com_and_selects_ryzen_die_sensor(monkeypatch):
    if hardware.os.name != "nt":
        import pytest
        pytest.skip("Windows WMI mapping")
    import win32com.client
    calls = []
    monkeypatch.setitem(sys.modules, "pythoncom", SimpleNamespace(
        CoInitialize=lambda: calls.append("initialise"),
        CoUninitialize=lambda: calls.append("release")))
    sensors = [SimpleNamespace(Name="Core #1", Identifier="/amdcpu/0/temperature/1", Value=51),
               SimpleNamespace(Name="Core (Tctl/Tdie)", Identifier="/amdcpu/0/temperature/0", Value=56.25)]
    monkeypatch.setattr(win32com.client, "GetObject", lambda _path: SimpleNamespace(ExecQuery=lambda _query: sensors))
    assert hardware._cpu_temperature() == 56.2
    assert calls == ["initialise", "release"]


def test_absent_gpu_has_no_fabricated_zero_load_or_vram(monkeypatch):
    from orion_core import gpu_stats
    monkeypatch.setattr(hardware, "_last", None)
    monkeypatch.setattr(gpu_stats, "sample", lambda: {"available": False})
    monkeypatch.setattr(hardware, "_cpu_temperature", lambda: None)
    result = hardware.sample()["gpu"]
    assert result["load_percent"] is None
    assert result["vram_used_mb"] is None
