"""
DisplayTopologyManager (Phase 9) — multi-monitor + high-DPI intelligence.

Every autonomous action that touches a coordinate goes through here first, so
the control layer and the verification engine speak one coordinate language
regardless of how many monitors are attached, their resolutions, their
per-monitor DPI scaling, or their physical arrangement.

Coordinate spaces
-----------------
    virtual   — the OS virtual desktop: physical pixels, origin at the
                primary monitor's top-left, monitors can sit at negative x/y.
    monitor   — (monitor_index, local_x, local_y): pixels relative to one
                monitor's top-left.

Windows is made **per-monitor DPI aware** at construction (best effort) so the
pixel coordinates we compute match what ``SendInput`` and ``mss`` actually
use — the classic source of "clicks land in the wrong place on a scaled
second monitor".

The topology is cached and refreshed on demand (``refresh``) or automatically
when it goes stale, so hot-plugging a monitor or changing resolution is picked
up without a restart.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Monitor:
    index: int
    x: int
    y: int
    width: int
    height: int
    is_primary: bool = False
    scale: float = 1.0          # DPI scale factor (1.0 = 100%, 1.5 = 150%)
    name: str = ""

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def centre(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, vx: int, vy: int) -> bool:
        return self.x <= vx < self.right and self.y <= vy < self.bottom

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "is_primary": self.is_primary, "scale": round(self.scale, 3),
            "name": self.name,
        }


@dataclass
class Topology:
    monitors: list[Monitor] = field(default_factory=list)
    captured_at: float = 0.0

    @property
    def primary(self) -> Optional[Monitor]:
        for m in self.monitors:
            if m.is_primary:
                return m
        return self.monitors[0] if self.monitors else None

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Virtual-desktop bounding box (min_x, min_y, max_x, max_y)."""
        if not self.monitors:
            return (0, 0, 0, 0)
        return (
            min(m.x for m in self.monitors),
            min(m.y for m in self.monitors),
            max(m.right for m in self.monitors),
            max(m.bottom for m in self.monitors),
        )


class DisplayTopologyManager:
    """Enumerates monitors, tracks DPI, and translates between coordinate spaces."""

    REFRESH_AFTER_S = 5.0

    def __init__(self, telemetry: Any | None = None) -> None:
        self.telemetry = telemetry
        self._topology = Topology()
        self._make_dpi_aware()
        self.refresh()

    # ── DPI awareness ─────────────────────────────────────────────────────────

    def _make_dpi_aware(self) -> None:
        """Best-effort per-monitor-v2 DPI awareness so pixels are physical."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            # PROCESS_PER_MONITOR_DPI_AWARE = 2; ignore failure if already set.
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    # ── enumeration ───────────────────────────────────────────────────────────

    def refresh(self) -> Topology:
        monitors = self._enumerate_windows() if sys.platform == "win32" else []
        if not monitors:
            monitors = self._enumerate_screeninfo()
        if not monitors:
            monitors = [Monitor(0, 0, 0, 1920, 1080, True, 1.0, "fallback")]
        self._topology = Topology(monitors=monitors, captured_at=time.monotonic())
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("display.monitors", float(len(monitors)))
            self.telemetry.log.info(
                "DISPLAY",
                f"topology: {len(monitors)} monitor(s)",
                primary=self._topology.primary.name if self._topology.primary else "?",
            )
        return self._topology

    def _enumerate_windows(self) -> list[Monitor]:
        """Enumerate via Win32 with per-monitor DPI (authoritative on Windows)."""
        try:
            import ctypes
            import ctypes.wintypes as wintypes

            user32 = ctypes.windll.user32
            shcore = None
            try:
                shcore = ctypes.windll.shcore
            except Exception:
                shcore = None

            monitors: list[Monitor] = []

            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

            MONITORINFOF_PRIMARY = 1

            class MONITORINFOEXW(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                            ("rcWork", RECT), ("dwFlags", wintypes.DWORD),
                            ("szDevice", wintypes.WCHAR * 32)]

            MonitorEnumProc = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(RECT), ctypes.c_double,
            )

            index_box = [0]

            def _callback(hmon: Any, hdc: Any, lprc: Any, data: Any) -> bool:
                info = MONITORINFOEXW()
                info.cbSize = ctypes.sizeof(MONITORINFOEXW)
                if not user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                    return True
                rc = info.rcMonitor
                scale = 1.0
                if shcore is not None:
                    try:
                        dpi_x = wintypes.UINT()
                        dpi_y = wintypes.UINT()
                        # MDT_EFFECTIVE_DPI = 0
                        if shcore.GetDpiForMonitor(hmon, 0, ctypes.byref(dpi_x),
                                                   ctypes.byref(dpi_y)) == 0:
                            scale = round(dpi_x.value / 96.0, 4)
                    except Exception:
                        scale = 1.0
                monitors.append(Monitor(
                    index=index_box[0],
                    x=rc.left, y=rc.top,
                    width=rc.right - rc.left, height=rc.bottom - rc.top,
                    is_primary=bool(info.dwFlags & MONITORINFOF_PRIMARY),
                    scale=scale,
                    name=str(info.szDevice) or f"monitor-{index_box[0]}",
                ))
                index_box[0] += 1
                return True

            user32.EnumDisplayMonitors(None, None, MonitorEnumProc(_callback), 0)
            # Order so the primary is index 0-ish and layout reads left→right.
            monitors.sort(key=lambda m: (not m.is_primary, m.x, m.y))
            for i, m in enumerate(monitors):
                m.index = i
            return monitors
        except Exception:
            return []

    def _enumerate_screeninfo(self) -> list[Monitor]:
        try:
            from screeninfo import get_monitors  # type: ignore
            out: list[Monitor] = []
            for i, m in enumerate(get_monitors()):
                out.append(Monitor(
                    index=i, x=int(m.x), y=int(m.y),
                    width=int(m.width), height=int(m.height),
                    is_primary=bool(getattr(m, "is_primary", i == 0)),
                    scale=1.0, name=str(getattr(m, "name", f"monitor-{i}")),
                ))
            return out
        except Exception:
            return []

    # ── access ────────────────────────────────────────────────────────────────

    def topology(self, auto_refresh: bool = True) -> Topology:
        if auto_refresh and (
            not self._topology.monitors
            or (time.monotonic() - self._topology.captured_at) > self.REFRESH_AFTER_S
        ):
            self.refresh()
        return self._topology

    def monitors(self) -> list[Monitor]:
        return list(self.topology().monitors)

    def primary(self) -> Optional[Monitor]:
        return self.topology().primary

    def monitor_at(self, vx: int, vy: int) -> Optional[Monitor]:
        for m in self.topology().monitors:
            if m.contains(vx, vy):
                return m
        return self.primary()

    def monitor_by_index(self, index: int) -> Optional[Monitor]:
        for m in self.topology().monitors:
            if m.index == index:
                return m
        return self.primary()

    # ── coordinate translation ────────────────────────────────────────────────

    def to_virtual(self, monitor_index: int, local_x: int, local_y: int) -> tuple[int, int]:
        """(monitor-local) → virtual-desktop coordinates."""
        mon = self.monitor_by_index(monitor_index)
        if mon is None:
            return (local_x, local_y)
        return (mon.x + int(local_x), mon.y + int(local_y))

    def to_monitor(self, vx: int, vy: int) -> tuple[int, int, int]:
        """virtual → (monitor_index, local_x, local_y)."""
        mon = self.monitor_at(vx, vy)
        if mon is None:
            return (0, vx, vy)
        return (mon.index, vx - mon.x, vy - mon.y)

    def clamp_to_desktop(self, vx: int, vy: int) -> tuple[int, int]:
        """Clamp a virtual coordinate into the visible desktop bounding box."""
        min_x, min_y, max_x, max_y = self.topology().bounds
        return (max(min_x, min(max_x - 1, int(vx))), max(min_y, min(max_y - 1, int(vy))))

    def normalise(self, monitor_index: int, nx: float, ny: float) -> tuple[int, int]:
        """Fractional (0..1) position on a monitor → virtual pixel coordinates."""
        mon = self.monitor_by_index(monitor_index)
        if mon is None:
            return (0, 0)
        return (mon.x + int(max(0.0, min(1.0, nx)) * (mon.width - 1)),
                mon.y + int(max(0.0, min(1.0, ny)) * (mon.height - 1)))

    # ── cursor ────────────────────────────────────────────────────────────────

    def cursor_position(self) -> tuple[int, int]:
        if sys.platform == "win32":
            try:
                import ctypes

                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                pt = POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                return (pt.x, pt.y)
            except Exception:
                pass
        try:
            import pyautogui  # type: ignore
            pos = pyautogui.position()
            return (int(pos.x), int(pos.y))
        except Exception:
            return (0, 0)

    # ── reporting ─────────────────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        topo = self.topology()
        cx, cy = self.cursor_position()
        return {
            "monitors": [m.to_dict() for m in topo.monitors],
            "bounds": topo.bounds,
            "cursor": {"x": cx, "y": cy, "monitor": self.to_monitor(cx, cy)[0]},
            "primary_index": topo.primary.index if topo.primary else None,
        }

    def summary(self) -> str:
        parts = []
        for m in self.topology().monitors:
            flag = "*" if m.is_primary else " "
            parts.append(
                f"{flag}[{m.index}] {m.width}x{m.height} @ ({m.x},{m.y}) "
                f"scale {int(m.scale * 100)}%"
            )
        return "Displays: " + "; ".join(parts)
