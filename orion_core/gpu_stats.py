"""
GPU telemetry — lightweight, cached NVIDIA utilisation via NVML or nvidia-smi.

NVML is preferred because it is in-process and cheap.  ``nvidia-smi`` is the
fallback already present with the NVIDIA driver on Windows, so a missing
optional ``pynvml`` wheel no longer leaves the first-window GPU/TMP meters at
N/A.  Both paths are cached behind a short interval; callers may sample from a
worker thread without turning the GUI/audio loop into a subprocess launcher.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import time
from typing import Any

_state: dict[str, Any] = {"init": False, "ok": False, "handle": None, "name": "",
                         "backend": "", "exe": None, "at": 0.0, "cache": None}
_MIN_INTERVAL_S = 1.0
#: MEASURED: an nvidia-smi sample is ~80 ms and a fresh 28 MB process, where
#: NVML in-process is 0.02 ms. The fallback is therefore refreshed far less
#: often — a meter a few seconds old beats a process launched every second on
#: a machine already short of memory, while the face is rendering.
_SMI_INTERVAL_S = 5.0
_SMI_TIMEOUT_S = 0.75


def _empty() -> dict[str, Any]:
    return {"available": False, "name": "n/a", "util": 0.0,
            "mem_percent": 0.0, "mem_used_mb": 0, "temp_c": 0,
            "temp_percent": 0.0}


def _ensure_init() -> bool:
    if _state["init"]:
        return _state["ok"]
    _state["init"] = True
    try:
        import pynvml
        pynvml.nvmlInit()
        _state["handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(_state["handle"])
        _state["name"] = name.decode() if isinstance(name, bytes) else str(name)
        _state["backend"] = "nvml"
        _state["ok"] = True
    except Exception:
        # pynvml is intentionally optional.  The driver utility is a reliable
        # fallback on Windows and costs nothing until a reading is requested.
        exe = shutil.which("nvidia-smi")
        if exe:
            _state["exe"] = exe
            _state["backend"] = "nvidia-smi"
            _state["ok"] = True
        else:
            _state["ok"] = False
    return _state["ok"]


def latest() -> dict[str, Any]:
    """A reading that is safe to take on the GUI thread.

    NVML answers in microseconds, so it is sampled; the nvidia-smi fallback
    launches a process, so the GUI thread only ever gets the last reading a
    background sampler (the core window's telemetry loop) already took.
    """
    if _ensure_init() and _state["backend"] == "nvml":
        return sample()
    return _state["cache"] or _empty()


def sample() -> dict[str, Any]:
    """Return GPU utilisation, memory and temperature for the first adapter.

    ``temp_percent`` is a display-friendly normalisation of the GPU's Celsius
    reading against a 100°C ceiling.  The raw ``temp_c`` remains available to
    callers that need the physical unit.
    """
    if not _ensure_init():
        return _empty()
    now = time.monotonic()
    interval = _MIN_INTERVAL_S if _state["backend"] == "nvml" else _SMI_INTERVAL_S
    if _state["cache"] is not None and (now - _state["at"]) < interval:
        return _state["cache"]
    try:
        if _state["backend"] == "nvml":
            import pynvml
            h = _state["handle"]
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            out = {
                "available": True, "name": _state["name"],
                "util": float(util.gpu),
                "mem_percent": round(mem.used / mem.total * 100.0, 1) if mem.total else 0.0,
                "mem_used_mb": int(mem.used / (1024 * 1024)),
                "temp_c": int(temp),
            }
        else:
            proc = subprocess.run(
                [_state["exe"] or "nvidia-smi",
                 "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=_SMI_TIMEOUT_S,
                check=False,
                # ORION is a windowed application.  Without this flag Windows
                # can create a console flash every time the optional fallback
                # probe runs (the NVML path never launches a child process).
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                raise RuntimeError(proc.stderr.strip() or "nvidia-smi returned no data")
            row = next(csv.reader(io.StringIO(proc.stdout)), [])
            if len(row) < 5:
                raise RuntimeError("nvidia-smi returned an incomplete row")
            name, util, used, total, temp = (item.strip() for item in row[:5])
            temp_c = int(float(temp))
            total_mb = float(total)
            used_mb = float(used)
            out = {
                "available": True, "name": name or "NVIDIA GPU",
                "util": float(util),
                "mem_percent": round(used_mb / total_mb * 100.0, 1) if total_mb else 0.0,
                "mem_used_mb": int(used_mb), "temp_c": temp_c,
            }
        out["temp_percent"] = min(100.0, max(0.0, float(out["temp_c"])))
    except Exception:
        out = _empty()
    _state["cache"] = out
    _state["at"] = now
    return out
