"""
SentinelAgent — ambient system monitoring with proactive spoken alerts.

This is the JARVIS "Sir, power is at fifteen percent" faculty.  A background
loop samples the host every few seconds and, when something crosses a
threshold, ORION *says* it (through the proactive-voice channel) and raises a
HUD banner — rather than the telemetry sitting silently in the Command Centre.

Watched signals:
    • CPU sustained load           • RAM pressure
    • Low disk space               • Battery low / charger removed
    • A new heavyweight process appearing (basic situational awareness)

Every alert type has its own cooldown so ORION warns once and then stays quiet
until the situation resolves and recurs — no nagging.  Thresholds are gentle
by default and the whole agent can be disabled with ORION_SENTINEL=0.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import psutil

from .bus import OrionBus
from .data import ToolResult


class SentinelAgent:
    CPU_ALERT = 92.0            # sustained %
    RAM_ALERT = 90.0            # %
    DISK_ALERT = 92.0           # % used on the system drive
    BATTERY_ALERT = 20.0        # %
    BATTERY_CRITICAL = 10.0     # %
    COOLDOWN = 300.0            # seconds between repeats of the same alert
    SAMPLE_INTERVAL = 6.0

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.enabled = os.getenv("ORION_SENTINEL", "1").strip().lower() not in {"0", "false", "no", "off"}
        self._last_alert: dict[str, float] = {}
        self._cpu_streak = 0
        self._known_procs: set[int] = set()
        self._was_plugged: bool | None = None
        self._stop = asyncio.Event()

    # ── alerting ──────────────────────────────────────────────────────────────

    def _alert(self, key: str, message: str, critical: bool = False) -> None:
        now = time.monotonic()
        if now - self._last_alert.get(key, 0.0) < self.COOLDOWN:
            return
        self._last_alert[key] = now
        self.bus.banner.emit(f"⚠ {message}", 4 if critical else 3)
        self.bus.speak_request.emit(f"Sir, {message}.")
        if self.telemetry is not None:
            self.telemetry.metrics.incr(f"sentinel.alert.{key}")

    # ── sampling ──────────────────────────────────────────────────────────────

    def _sample(self) -> None:
        # CPU — require a short sustained streak so a momentary spike is ignored.
        cpu = psutil.cpu_percent(interval=None)
        self._cpu_streak = self._cpu_streak + 1 if cpu >= self.CPU_ALERT else 0
        if self._cpu_streak >= 3:
            self._alert("cpu", f"the processor has been at {cpu:.0f} percent for a sustained period")
            self._cpu_streak = 0

        ram = psutil.virtual_memory().percent
        if ram >= self.RAM_ALERT:
            self._alert("ram", f"memory usage is high at {ram:.0f} percent")

        try:
            disk = psutil.disk_usage(os.path.abspath(os.sep)).percent
            if disk >= self.DISK_ALERT:
                self._alert("disk", f"the system drive is nearly full at {disk:.0f} percent")
        except Exception:
            pass

        self._check_battery()
        self._check_new_processes()

    def _check_battery(self) -> None:
        try:
            battery = psutil.sensors_battery()
        except Exception:
            battery = None
        if battery is None:
            return
        plugged = bool(battery.power_plugged)
        pct = float(battery.percent)
        if self._was_plugged is True and not plugged:
            self._alert("unplugged", f"the charger has been removed; battery at {pct:.0f} percent")
        self._was_plugged = plugged
        if not plugged:
            if pct <= self.BATTERY_CRITICAL:
                self._alert("battery_crit", f"battery is critically low at {pct:.0f} percent — "
                            "please connect the charger", critical=True)
            elif pct <= self.BATTERY_ALERT:
                self._alert("battery_low", f"battery is running low at {pct:.0f} percent")

    def _check_new_processes(self) -> None:
        try:
            current = {p.pid for p in psutil.process_iter(["pid"])}
        except Exception:
            return
        if not self._known_procs:
            self._known_procs = current
            return
        new = current - self._known_procs
        self._known_procs = current
        # Only flag a genuinely heavyweight newcomer (RAM > 300 MB) to avoid noise.
        for pid in list(new)[:20]:
            try:
                proc = psutil.Process(pid)
                rss = proc.memory_info().rss / (1024 * 1024)
                if rss > 300:
                    self._alert(f"proc_{proc.name()}",
                                f"a new heavyweight process started — {proc.name()} "
                                f"using {rss:.0f} megabytes")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # ── on-demand status ──────────────────────────────────────────────────────

    def status(self) -> ToolResult:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        try:
            disk = psutil.disk_usage(os.path.abspath(os.sep)).percent
        except Exception:
            disk = 0.0
        parts = [f"CPU {cpu:.0f}%", f"RAM {ram:.0f}%", f"disk {disk:.0f}%"]
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                parts.append(f"battery {battery.percent:.0f}%"
                             + (" (charging)" if battery.power_plugged else ""))
        except Exception:
            pass
        verdict = "All systems nominal, sir." if (cpu < self.CPU_ALERT and ram < self.RAM_ALERT
                                                  and disk < self.DISK_ALERT) else "Some systems need attention, sir."
        return ToolResult(f"Situation report: {', '.join(parts)}. {verdict}")

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.bus.log.emit(f"SENTINEL: monitoring {'enabled' if self.enabled else 'disabled'}.")

    # ── background loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        if self.telemetry is not None:
            self.telemetry.health.register("sentinel")
        psutil.cpu_percent(interval=None)  # prime the CPU meter
        try:
            while not self._stop.is_set():
                if self.enabled:
                    try:
                        await asyncio.to_thread(self._sample)
                    except Exception as exc:
                        self.bus.log.emit(f"SENTINEL: recovered - {str(exc).splitlines()[0][:100]}")
                if self.telemetry is not None:
                    self.telemetry.health.beat("sentinel", "OK", "watching" if self.enabled else "off")
                await asyncio.sleep(self.SAMPLE_INTERVAL)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()
