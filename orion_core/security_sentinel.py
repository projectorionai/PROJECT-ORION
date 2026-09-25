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
import subprocess
import sys
import time
from typing import Any

import psutil

from .bus import OrionBus
from .data import ToolResult


def format_network_dashboard(firewall: str, listening: list[dict[str, Any]],
                             established: int, startup: list[str]) -> str:
    """
    Render the network-security dashboard as a spoken/printed report.  Pure
    function (no I/O) so the format is unit-testable independent of the host.
    """
    lines: list[str] = ["Network security dashboard:"]
    fw_note = {
        "on": "Firewall: ON across all profiles.",
        "off": "Firewall: OFF on every profile — that is a real exposure; turn it back on.",
        "partial": "Firewall: PARTIAL — at least one profile is disabled.",
    }.get(firewall, "Firewall: status unavailable.")
    lines.append(f"  • {fw_note}")
    if listening:
        top = ", ".join(
            f"{e['port']}/{e.get('proc') or '?'}" for e in listening[:8])
        lines.append(f"  • {len(listening)} externally-listening port(s): {top}"
                     + ("…" if len(listening) > 8 else ""))
    else:
        lines.append("  • No externally-listening ports — nothing is reachable from outside.")
    lines.append(f"  • {established} established outbound/active connection(s).")
    if startup:
        head = ", ".join(startup[:6]) + ("…" if len(startup) > 6 else "")
        lines.append(f"  • {len(startup)} auto-start entr(y/ies): {head}")
    else:
        lines.append("  • No auto-start program entries found.")
    return "\n".join(lines)

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
        self.bus.speak_request.emit(f"Security note: {message}.")
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
            f"externally-reachable ports: {listening or 'none'}. {verdict}."
        )

    # ── network security dashboard (read-only, on demand) ─────────────────────

    def _firewall_status_sync(self) -> str:
        """'on' | 'off' | 'partial' | 'unknown' — Windows netsh, safe elsewhere."""
        if not sys.platform.startswith("win"):
            return "unknown"
        try:
            out = subprocess.run(
                ["netsh", "advfirewall", "show", "allprofiles", "state"],
                capture_output=True, text=True, timeout=8.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
        except Exception:
            return "unknown"
        states = [m.group(1).lower()
                  for m in re.finditer(r"(?im)^\s*State\s+(\w+)", out)]
        if not states:
            return "unknown"
        on = sum(1 for s in states if s == "on")
        if on == len(states):
            return "on"
        if on == 0:
            return "off"
        return "partial"

    def _listening_services_sync(self) -> list[dict[str, Any]]:
        """Externally-listening ports with the owning process name."""
        out: list[dict[str, Any]] = []
        try:
            conns = psutil.net_connections(kind="inet")
        except Exception:
            return out
        for c in conns:
            if c.status != psutil.CONN_LISTEN or not c.laddr:
                continue
            if c.laddr.ip in {"127.0.0.1", "::1", "localhost"}:
                continue
            proc = ""
            if c.pid:
                try:
                    proc = psutil.Process(c.pid).name()
                except Exception:
                    proc = ""
            out.append({"port": c.laddr.port, "ip": c.laddr.ip, "pid": c.pid, "proc": proc})
        out.sort(key=lambda e: e["port"])
        return out

    def _established_count_sync(self) -> int:
        try:
            return sum(1 for c in psutil.net_connections(kind="inet")
                       if c.status == psutil.CONN_ESTABLISHED)
        except Exception:
            return 0

    def _startup_entries_sync(self) -> list[str]:
        """Registry Run keys + Startup folder (read-only). Windows only."""
        entries: list[str] = []
        if not sys.platform.startswith("win"):
            return entries
        try:
            import winreg  # type: ignore
            runs = [
                (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
            ]
            for hive, path in runs:
                try:
                    with winreg.OpenKey(hive, path) as key:
                        i = 0
                        while True:
                            try:
                                name, _value, _type = winreg.EnumValue(key, i)
                            except OSError:
                                break
                            entries.append(str(name))
                            i += 1
                except OSError:
                    continue
        except Exception:
            return entries
        try:
            appdata = os.getenv("APPDATA") or ""
            if appdata:
                folder = os.path.join(
                    appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
                if os.path.isdir(folder):
                    for fn in os.listdir(folder):
                        if not fn.lower().startswith("desktop.ini"):
                            entries.append(os.path.splitext(fn)[0])
        except Exception:
            pass
        return entries

    def network_dashboard(self) -> ToolResult:
        """Assemble the full network-security posture (blocking; call via to_thread)."""
        firewall = self._firewall_status_sync()
        listening = self._listening_services_sync()
        established = self._established_count_sync()
        startup = self._startup_entries_sync()
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("security.network_dashboard")
            except Exception:
                pass
        return ToolResult(format_network_dashboard(firewall, listening, established, startup))

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
