"""
Proactive cybersecurity monitoring (improvement #17).

``SecuritySentinel`` watches the host for situational-security changes and warns
ORION's user proactively (spoken + banner), read-only and privacy-respecting —
it inspects local system state via ``psutil`` and never touches the network or
transmits anything.

Watched signals:
    • new listening sockets on non-loopback interfaces (a service just opened
      a port to the outside world);
    • a process with a suspicious name pattern (miners, remote-access tools,
      obvious script-runner spawns from temp directories);
    • new external/removable drives being mounted;
    • unusually many new processes appearing at once (a possible fork storm).

It complements the general ``SentinelAgent`` (which watches performance/battery)
by focusing on security posture.  Every alert type has a cooldown so ORION
warns once per genuine change.  Disable with ORION_SECURITY_WATCH=0.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

import psutil

from .bus import OrionBus
from .data import ToolResult

# Process-name patterns worth flagging if newly seen (heuristic, low-noise).
SUSPICIOUS_NAME_RE = re.compile(
    r"(?i)\b(xmrig|minerd|cgminer|ngrok|anydesk|teamviewer|rclone|nc\.exe|"
    r"ncat|mimikatz|psexec|cryptolocker|ransom)\b"
)
# Directories a legitimate installed program rarely runs from.
SUSPICIOUS_PATH_RE = re.compile(r"(?i)\\temp\\|\\tmp\\|\\appdata\\local\\temp\\|/tmp/")


class SecuritySentinel:
    COOLDOWN = 600.0
    SAMPLE_INTERVAL = 20.0
    PROCESS_SURGE = 40      # new processes in one interval → possible fork storm

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.enabled = os.getenv("ORION_SECURITY_WATCH", "1").strip().lower() not in {"0", "false", "no", "off"}
        self._last_alert: dict[str, float] = {}
        self._known_ports: set[int] = set()
        self._known_procs: set[int] = set()
        self._known_drives: set[str] = set()
        self._baseline = False
        self._stop = asyncio.Event()

    # ── alerting ──────────────────────────────────────────────────────────────

    def _alert(self, key: str, message: str) -> None:
        now = time.monotonic()
        if now - self._last_alert.get(key, 0.0) < self.COOLDOWN:
            return
        self._last_alert[key] = now
        self.bus.banner.emit(f"🛡 {message}", 4)
        self.bus.speak_request.emit(f"Security note, sir: {message}.")
        if self.telemetry is not None:
            self.telemetry.metrics.incr(f"security.alert.{key.split(':')[0]}")

    # ── sampling ──────────────────────────────────────────────────────────────

    def _sample(self) -> None:
        self._check_ports()
        self._check_processes()
        self._check_drives()

    def _check_ports(self) -> None:
        try:
            conns = psutil.net_connections(kind="inet")
        except Exception:
            return
        listening: set[int] = set()
        details: dict[int, str] = {}
        for c in conns:
            if c.status == psutil.CONN_LISTEN and c.laddr:
                ip = c.laddr.ip
                if ip not in {"127.0.0.1", "::1", "localhost"}:
                    listening.add(c.laddr.port)
                    details[c.laddr.port] = ip
        if not self._baseline:
            self._known_ports = listening
            return
        for port in sorted(listening - self._known_ports):
            self._alert(f"port:{port}",
                        f"a service is now listening on port {port} "
                        f"({details.get(port, '?')}) — reachable beyond this machine")
        self._known_ports = listening

    def _check_processes(self) -> None:
        try:
            procs = list(psutil.process_iter(["pid", "name", "exe"]))
        except Exception:
            return
        current = {p.info["pid"] for p in procs}
        if not self._baseline:
            self._known_procs = current
            return
        new = current - self._known_procs
        if len(new) > self.PROCESS_SURGE:
            self._alert("surge", f"{len(new)} new processes started at once — "
                        "worth a glance in case something is spawning uncontrolled")
        for p in procs:
            if p.info["pid"] not in new:
                continue
            name = str(p.info.get("name") or "")
            exe = str(p.info.get("exe") or "")
            if SUSPICIOUS_NAME_RE.search(name) or SUSPICIOUS_NAME_RE.search(exe):
                self._alert(f"proc:{name}", f"a process named '{name}' started, which "
                            "matches a remote-access or mining pattern")
            elif exe and SUSPICIOUS_PATH_RE.search(exe):
                self._alert(f"temp:{name}", f"'{name}' is running from a temp directory — "
                            "unusual for trusted software")
        self._known_procs = current

    def _check_drives(self) -> None:
        try:
            parts = psutil.disk_partitions(all=False)
        except Exception:
            return
        removable: set[str] = set()
        for part in parts:
            opts = (part.opts or "").lower()
            if "removable" in opts or "cdrom" in opts:
                removable.add(part.device)
        if not self._baseline:
            self._known_drives = removable
            return
        for drive in sorted(removable - self._known_drives):
            self._alert(f"drive:{drive}", f"an external drive was connected at {drive}")
        self._known_drives = removable

    # ── on-demand report ──────────────────────────────────────────────────────

    def status(self) -> ToolResult:
        try:
            listening = sorted({
                c.laddr.port for c in psutil.net_connections(kind="inet")
                if c.status == psutil.CONN_LISTEN and c.laddr
                and c.laddr.ip not in {"127.0.0.1", "::1"}
            })
        except Exception:
            listening = []
        proc_count = len(psutil.pids())
        verdict = "no obvious security concerns" if not listening else \
            f"{len(listening)} externally-listening port(s)"
        return ToolResult(
            f"Security posture: {proc_count} processes running; "
            f"externally-reachable ports: {listening or 'none'}. {verdict}, sir."
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.bus.log.emit(f"SECURITY: monitoring {'enabled' if self.enabled else 'disabled'}.")

    # ── background loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        if self.telemetry is not None:
            self.telemetry.health.register("security")
        try:
            while not self._stop.is_set():
                if self.enabled:
                    try:
                        await asyncio.to_thread(self._sample)
                        self._baseline = True   # first pass establishes the baseline
                    except Exception as exc:
                        self.bus.log.emit(f"SECURITY: recovered - {str(exc).splitlines()[0][:100]}")
                if self.telemetry is not None:
                    self.telemetry.health.beat("security", "OK",
                                               "watching" if self.enabled else "off")
                await asyncio.sleep(self.SAMPLE_INTERVAL)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()
