"""Bounded host CPU and GPU readings for the paired mobile dashboard.

CPU temperature on Windows needs a hardware monitor such as LibreHardwareMonitor
publishing WMI sensors. Missing sensors are reported as null, never fabricated.
The gateway calls this sampler in a worker so WMI and GPU drivers cannot stall
the asyncio event loop.
"""

from __future__ import annotations

import os
import math
import platform
import threading
import time
from typing import Any


_lock = threading.Lock()
_last_at = 0.0
_last: dict[str, Any] | None = None


def _cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg
            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:
            pass
    return platform.processor() or "CPU"


def _cpu_temperature() -> float | None:
    if os.name == "nt":
        com = None
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            com = pythoncom
            for namespace in ("LibreHardwareMonitor", "OpenHardwareMonitor"):
                try:
                    service = win32com.client.GetObject(
                        rf"winmgmts:\\.\root\{namespace}")
                    sensors = service.ExecQuery(
                        "SELECT Name, SensorType, Value, Identifier FROM Sensor "
                        "WHERE SensorType = 'Temperature'")
                    readings = []
                    for sensor in sensors:
                        name = str(sensor.Name).lower()
                        identifier = str(sensor.Identifier).lower()
                        if "cpu" not in identifier:
                            continue
                        value = float(sensor.Value)
                        if not math.isfinite(value) or not -20 <= value <= 150:
                            continue
                        # Ryzen sensors are named Core (Tctl/Tdie), rather than
                        # CPU Package. Prefer that die/package reading to cores.
                        rank = 0 if any(label in name for label in ("tctl", "tdie", "package")) else 1
                        readings.append((rank, value))
                    if readings:
                        return round(min(readings, key=lambda reading: reading[0])[1], 1)
                except Exception:
                    continue
        except Exception:
            pass
        finally:
            if com is not None:
                com.CoUninitialize()
        return None
    try:
        import psutil
        for group in psutil.sensors_temperatures().values():
            for sensor in group:
                label = str(sensor.label).lower()
                if "package" in label or "cpu" in label:
                    return round(float(sensor.current), 1)
    except Exception:
        pass
    return None


def sample() -> dict[str, Any]:
    """Return a cached, JSON-safe reading with explicit unavailable fields."""
    global _last_at, _last
    # A WMI sensor can stall beyond the HTTP deadline. Never queue a new
    # thread behind that sampler for every one-second dashboard refresh.
    if not _lock.acquire(blocking=False):
        if _last is not None:
            return {**_last, "stale": True}
        return {
            "sampled_at": None,
            "stale": True,
            "cpu": {"name": "CPU", "load_percent": None,
                    "temperature_c": None},
            "gpu": {"available": False, "name": "GPU unavailable",
                    "load_percent": None, "vram_used_mb": None,
                    "vram_total_mb": None, "temperature_c": None},
            "memory": {"used_percent": None},
        }
    try:
        now = time.monotonic()
        if _last is not None and now - _last_at < 2.0:
            return dict(_last)
        import psutil
        from . import gpu_stats

        cpu_load = float(psutil.cpu_percent(interval=None))
        ram = psutil.virtual_memory()
        gpu = gpu_stats.sample()
        reading = {
            "sampled_at": int(time.time() * 1000),
            "cpu": {
                "name": _cpu_name(),
                "load_percent": round(cpu_load, 1),
                "temperature_c": _cpu_temperature(),
            },
            "gpu": {
                "available": bool(gpu.get("available")),
                "name": str(gpu.get("name") or "GPU unavailable"),
                "load_percent": float(gpu.get("util") or 0) if gpu.get("available") else None,
                "vram_used_mb": int(gpu.get("mem_used_mb") or 0) if gpu.get("available") else None,
                "vram_total_mb": int(gpu.get("mem_total_mb") or 0) if gpu.get("available") else None,
                "temperature_c": gpu.get("temp_c") if gpu.get("available") else None,
            },
            "memory": {"used_percent": round(float(ram.percent), 1)},
        }
        _last, _last_at = reading, now
        return dict(reading)
    finally:
        _lock.release()
