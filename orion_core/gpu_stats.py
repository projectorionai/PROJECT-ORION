"""
GPU telemetry — lightweight, cached NVIDIA utilisation via NVML (pynvml).

Adds GPU visibility to ORION's diagnostics without a heavy dependency: NVML is
the driver's own management library, queried once and cached behind a short
interval so sampling never touches a hot path.  Degrades cleanly to
``available=False`` on machines without an NVIDIA GPU (or without pynvml), so
the panel simply shows "GPU: n/a".
"""

from __future__ import annotations

import time
from typing import Any

_state: dict[str, Any] = {"init": False, "ok": False, "handle": None, "name": "",
                         "at": 0.0, "cache": None}
_MIN_INTERVAL_S = 1.0


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
        _state["ok"] = True
    except Exception:
        _state["ok"] = False
    return _state["ok"]


def sample() -> dict[str, Any]:
    """Return {available, name, util, mem_percent, mem_used_mb, temp_c}."""
    if not _ensure_init():
        return {"available": False, "name": "n/a", "util": 0.0,
                "mem_percent": 0.0, "mem_used_mb": 0, "temp_c": 0}
    now = time.monotonic()
    if _state["cache"] is not None and (now - _state["at"]) < _MIN_INTERVAL_S:
        return _state["cache"]
    try:
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
    except Exception:
        out = {"available": False, "name": "n/a", "util": 0.0,
               "mem_percent": 0.0, "mem_used_mb": 0, "temp_c": 0}
    _state["cache"] = out
    _state["at"] = now
    return out
