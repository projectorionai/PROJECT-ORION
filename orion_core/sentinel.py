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


# Everyday interactive and Windows shell processes are not actionable security
# signals. Sentinel should surface unfamiliar heavyweight programs, not narrate
# normal desktop activity every time an application creates a helper process.
_ROUTINE_PROCESS_NAMES = frozenset({
    "agent", "applicationframehost", "battle.net", "battlenet", "blizzard",
    "chatgpt", "chrome", "claude", "cmd", "code", "conhost", "discord",
    "discordcanary", "discordptb", "dwm", "explorer", "firefox", "msedge",
    "msedgewebview2", "notepad", "obs", "obs64", "onedrive", "powershell",
    "runtimebroker", "searchhost", "shellexperiencehost", "sihost", "slack",
    "spotify", "startmenuexperiencehost", "steam", "teams", "terminal",
    "textinputhost", "widgetservice", "widgets", "windows terminal",
})


def is_routine_process(process_name: str) -> bool:
    """Whether a process name is normal desktop background activity.

    Both executable names (``msedge.exe``) and bare names (``msedge``) match;
    unknown programs remain eligible for a heavyweight-process alert.
    """

    normalised = str(process_name or "").strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    stem = normalised.removesuffix(".exe")
    return normalised in _ROUTINE_PROCESS_NAMES or stem in _ROUTINE_PROCESS_NAMES


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

    def _alert(self, key: str, message: str, critical: bool = False,
               kind: str = "") -> None:
        """Record an alert, and speak it ONLY if it earns being spoken.

        Every alert used to go straight to the voice channel, which is how a
        new heavyweight process — including ORION's own llama-server — got
        narrated aloud. The banner and the log still get everything; the
        ProactivePolicy decides what is worth interrupting a person for.
        """
        now = time.monotonic()
        if now - self._last_alert.get(key, 0.0) < self.COOLDOWN:
            return
        self._last_alert[key] = now
        self.bus.banner.emit(f"⚠ {message}", 4 if critical else 3)

        from .proactive_policy import POLICY, Urgency
        decision = POLICY.should_speak(
            kind or key, Urgency.CRITICAL if critical else None)
        spoken = f"{message[:1].upper()}{message[1:]}."
        if decision.speak and decision.urgency >= Urgency.CRITICAL:
            # Danger takes the override channel. speak_request lands on
            # announce(), which returns early on quiet_mode and paused — so a
            # CRITICAL finding the policy had just ruled "spoken regardless of
            # state" was being dropped by the delivery layer immediately after.
            self.bus.safety_alert.emit(spoken)
        elif decision.speak:
            self.bus.speak_request.emit(spoken)
        else:
            self.bus.log.emit(f"SENTINEL: {message} (not spoken — {decision.reason})")
        if self.telemetry is not None:
            self.telemetry.metrics.incr(f"sentinel.alert.{key}")

    # ── sampling ──────────────────────────────────────────────────────────────

    def _sample(self) -> None:
        # CPU — require a short sustained streak so a momentary spike is ignored.
        cpu = psutil.cpu_percent(interval=None)
        self._cpu_streak = self._cpu_streak + 1 if cpu >= self.CPU_ALERT else 0
        if self._cpu_streak >= 3:
            self._alert("cpu", f"the processor has been at {cpu:.0f} percent for a sustained period",
                        kind="cpu_high")
            self._cpu_streak = 0

        ram = psutil.virtual_memory().percent
        if ram >= self.RAM_ALERT:
            self._alert("ram", f"memory usage is high at {ram:.0f} percent",
                        kind="ram_high")

        try:
            disk = psutil.disk_usage(os.path.abspath(os.sep)).percent
            if disk >= self.DISK_ALERT:
                # A full system drive genuinely stops the user working.
                self._alert("disk", f"the system drive is nearly full at {disk:.0f} percent",
                            kind="disk_full")
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
            self._alert("unplugged", f"the charger has been removed; battery at {pct:.0f} percent",
                        kind="task_due")
        self._was_plugged = plugged
        if not plugged:
            if pct <= self.BATTERY_CRITICAL:
                self._alert("battery_crit", f"battery is critically low at {pct:.0f} percent — "
                            "please connect the charger", critical=True,
                            kind="battery_critical")
            elif pct <= self.BATTERY_ALERT:
                self._alert("battery_low", f"battery is running low at {pct:.0f} percent",
                            kind="task_due")

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
        # Only flag unfamiliar, genuinely heavyweight newcomers. Common apps
        # such as browsers, chat clients and launchers can legitimately spawn
        # several helpers and should never become unsolicited narration.
        for pid in list(new)[:20]:
            try:
                proc = psutil.Process(pid)
                rss = proc.memory_info().rss / (1024 * 1024)
                name = proc.name()
                if rss > 300 and not is_routine_process(name):
                    # AMBIENT by policy: a process starting is not dangerous
                    # and does not affect the user. Banner + log only — this is
                    # the exact narration the user asked to stop.
                    self._alert(f"proc_{name}",
                                f"an unfamiliar heavyweight process started — {name} "
                                f"using {rss:.0f} megabytes",
                                kind="process_started")
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
        verdict = "All systems nominal." if (cpu < self.CPU_ALERT and ram < self.RAM_ALERT
                                                  and disk < self.DISK_ALERT) else "Some systems need attention."
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
