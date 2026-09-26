"""
Windows Defender awareness (Security Sentinel Expansion — antivirus).

``AntivirusMonitor`` lets ORION answer "is there malware on my PC?" and "is my
antivirus up to date?" by talking to the antivirus that is *already installed*
on the machine — Microsoft Defender — through its scriptable PowerShell
interface (``Get-MpComputerStatus``, ``Get-MpThreatDetection``, ``Get-MpThreat``,
``Start-MpScan``). This module deliberately does **not** implement its own
malware-detection/heuristics engine: hand-rolling one would be both largely
ineffective (no signature/behavioural corpus, no telemetry feed, no update
pipeline) and risky (anything that inspects untrusted bytes is itself an
attack surface). Real detection is left entirely to Defender; ORION only
reports what Defender already knows and can trigger the same safe, built-in
quick scan a user could run from Windows Security.

Safety invariants (enforced by design, covered by tests):

    • Strictly defensive / read-mostly. The only state-changing action this
      module performs is triggering a **quick scan**
      (``Start-MpScan -ScanType QuickScan``) — itself a built-in, ordinary
      Defender operation. Nothing here disables protections, adds
      exclusions, deletes/quarantines/restores files, or executes any file —
      that remains Defender's job, and the user's, through Windows Security.
    • Every PowerShell script this module runs is a **fixed string authored
      in this file**, never assembled by interpolating caller-supplied text.
      No argument, filename, or identifier coming from the user or the model
      ever reaches a command line here, so there is no injection surface —
      the same "argument list, never shell=True with string interpolation"
      discipline as ``docker_control.py`` and ``security_sentinel.py``'s
      ``netsh`` call.
    • No network calls of any kind. Everything is local IPC to the Defender
      service via PowerShell/WMI. This is a stricter guarantee than
      ``breach_monitor.py``'s "network only on explicit request" — this
      module never touches the network, full stop.
    • Never scans, probes, or acts on any other host. That is
      ``security_recon.py``'s separate, authorization-gated territory; this
      module only ever asks Defender about *this* machine's own state.
    • Degrades gracefully everywhere it might not work: not on Windows,
      PowerShell missing, the Defender module/cmdlets absent (e.g. a
      third-party antivirus has taken over real-time protection and removed
      Defender's cmdlets), or the service not responding all produce a
      clear, actionable message — never a crash, and never a false "clean"
      result manufactured to paper over a failed check.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .bus import OrionBus
from .data import ToolResult

# Microsoft's own "signatures are out of date" threshold is 7 days.
STALE_SIGNATURE_DAYS = 7

_TIMEOUT_STATUS = 15.0
_TIMEOUT_THREATS = 15.0
_TIMEOUT_QUICK_SCAN = 600.0   # a Defender quick scan commonly runs several minutes

_SEVERITY_IDS = {0: "unknown", 1: "low", 2: "moderate", 4: "high", 5: "severe"}
_THREAT_STATUS_IDS = {
    1: "detected", 2: "cleaned", 3: "quarantined", 4: "removed",
    5: "allowed by user", 6: "blocked",
}

_UNAVAILABLE_MESSAGE = (
    "Windows Defender isn't reachable from here — either this isn't Windows, "
    "PowerShell's Defender module isn't present (a third-party antivirus may have "
    "taken over real-time protection), or the Defender service isn't responding. "
    "ORION has no malware-scanning engine of its own by design; it only reports "
    "what the OS's own antivirus knows."
)

# Fixed, non-interpolated PowerShell scripts. No caller-supplied value is ever
# spliced into these — see the module docstring's injection-surface note.
_STATUS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$s = Get-MpComputerStatus
[PSCustomObject]@{
    RealTimeProtectionEnabled    = $s.RealTimeProtectionEnabled
    AntivirusEnabled             = $s.AntivirusEnabled
    AntispywareEnabled           = $s.AntispywareEnabled
    BehaviorMonitorEnabled       = $s.BehaviorMonitorEnabled
    NISEnabled                   = $s.NISEnabled
    AntivirusSignatureVersion    = $s.AntivirusSignatureVersion
    AntivirusSignatureAge        = $s.AntivirusSignatureAge
    AntivirusSignatureLastUpdated = if ($s.AntivirusSignatureLastUpdated) { $s.AntivirusSignatureLastUpdated.ToString('o') } else { $null }
    QuickScanAge                 = $s.QuickScanAge
    FullScanAge                  = $s.FullScanAge
    AMEngineVersion               = $s.AMEngineVersion
    AMProductVersion              = $s.AMProductVersion
} | ConvertTo-Json -Compress
""".strip()

_THREATS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$d = Get-MpThreatDetection | Select-Object -First 25 ThreatID, ProcessName, ActionSuccess, ThreatStatusID, Resources, @{N='InitialDetectionTime';E={ if ($_.InitialDetectionTime) { $_.InitialDetectionTime.ToString('o') } else { $null } }}
$t = Get-MpThreat | Select-Object -First 50 ThreatID, ThreatName, SeverityID
[PSCustomObject]@{ Detections = @($d); Threats = @($t) } | ConvertTo-Json -Compress -Depth 4
""".strip()

_SCAN_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
try {
    Start-MpScan -ScanType QuickScan
    [PSCustomObject]@{ Success = $true; Error = $null } | ConvertTo-Json -Compress
} catch {
    [PSCustomObject]@{ Success = $false; Error = $_.Exception.Message } | ConvertTo-Json -Compress
}
""".strip()


# ── data model ──────────────────────────────────────────────────────────────

@dataclass
class ProtectionStatus:
    """Parsed ``Get-MpComputerStatus`` snapshot."""

    realtime_protection: bool | None = None
    antivirus_enabled: bool | None = None
    antispyware_enabled: bool | None = None
    behavior_monitor_enabled: bool | None = None
    nis_enabled: bool | None = None
    signature_version: str = ""
    signature_age_days: int | None = None
    signature_last_updated: str = ""
    quick_scan_age_days: int | None = None
    full_scan_age_days: int | None = None
    engine_version: str = ""
    product_version: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """True when real-time protection is on and definitions aren't stale."""
        if self.realtime_protection is not True:
            return False
        if self.signature_age_days is not None and self.signature_age_days > STALE_SIGNATURE_DAYS:
            return False
        return True


@dataclass
class ThreatDetection:
    """One entry from Defender's detection history (``Get-MpThreatDetection``),
    enriched with the friendly name/severity from ``Get-MpThreat`` when available."""

    threat_id: str
    name: str = "Unknown threat"
    severity: str = "unknown"
    process_name: str = ""
    resources: list[str] = field(default_factory=list)
    detected_at: str = ""
    action_success: bool | None = None
    status: str = "detected"


@dataclass
class ScanOutcome:
    """Result of a ``run_quick_scan_sync`` call."""

    triggered: bool
    completed: bool
    scan_type: str
    message: str
    threats_found: int = 0
    threats: list[ThreatDetection] = field(default_factory=list)
    duration_s: float = 0.0


def severity_for_id(severity_id: Any) -> str:
    """Map Defender's numeric SeverityID to a plain-language band. Pure/testable."""
    try:
        return _SEVERITY_IDS.get(int(severity_id), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def threat_status_for_id(status_id: Any) -> str:
    """Map Defender's numeric ThreatStatusID to a plain-language word. Pure/testable."""
    try:
        return _THREAT_STATUS_IDS.get(int(status_id), "detected")
    except (TypeError, ValueError):
        return "detected"


# ── parsing (pure functions — unit-tested directly against mocked stdout) ──────

def parse_protection_status(stdout: str) -> ProtectionStatus | None:
    """Parse ``_STATUS_SCRIPT``'s JSON output. Returns None on anything unparsable
    so callers treat it the same as "Defender didn't answer"."""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        return None

    def _bool(key: str) -> bool | None:
        v = data.get(key)
        return v if isinstance(v, bool) else None

    # Get-MpComputerStatus reports a scan "age in days" as UInt32.MaxValue
    # (4294967295) when that scan type has never run — not an actual day
    # count. Left unguarded this printed as "Last full scan: 4294967295
    # day(s) ago" on any machine that has never run a full scan, which is
    # every default Windows Defender install (only quick scans run
    # automatically). Treated the same as "never recorded".
    _NEVER_SCANNED = 4294967295

    def _int(key: str) -> int | None:
        v = data.get(key)
        if v is None:
            return None
        try:
            parsed = int(v)
        except (TypeError, ValueError):
            return None
        return None if parsed == _NEVER_SCANNED else parsed

    return ProtectionStatus(
        realtime_protection=_bool("RealTimeProtectionEnabled"),
        antivirus_enabled=_bool("AntivirusEnabled"),
        antispyware_enabled=_bool("AntispywareEnabled"),
        behavior_monitor_enabled=_bool("BehaviorMonitorEnabled"),
        nis_enabled=_bool("NISEnabled"),
        signature_version=str(data.get("AntivirusSignatureVersion") or ""),
        signature_age_days=_int("AntivirusSignatureAge"),
        signature_last_updated=str(data.get("AntivirusSignatureLastUpdated") or ""),
        quick_scan_age_days=_int("QuickScanAge"),
        full_scan_age_days=_int("FullScanAge"),
        engine_version=str(data.get("AMEngineVersion") or ""),
        product_version=str(data.get("AMProductVersion") or ""),
        raw=data,
    )


def parse_threats(stdout: str) -> list[ThreatDetection]:
    """Parse ``_THREATS_SCRIPT``'s combined Detections+Threats JSON. Pure/testable."""
    text = (stdout or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    detections = data.get("Detections") or []
    if isinstance(detections, dict):
        detections = [detections]
    threats_meta = data.get("Threats") or []
    if isinstance(threats_meta, dict):
        threats_meta = [threats_meta]

    names: dict[str, tuple[str, str]] = {}
    for t in threats_meta:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("ThreatID") or "").strip()
        if not tid:
            continue
        names[tid] = (str(t.get("ThreatName") or "Unknown threat"),
                      severity_for_id(t.get("SeverityID")))

    out: list[ThreatDetection] = []
    for d in detections:
        if not isinstance(d, dict):
            continue
        tid = str(d.get("ThreatID") or "").strip()
        name, sev = names.get(tid, ("Unknown threat", "unknown"))
        resources = d.get("Resources") or []
        if isinstance(resources, str):
            resources = [resources]
        elif not isinstance(resources, list):
            resources = []
        action_success = d.get("ActionSuccess")
        out.append(ThreatDetection(
            threat_id=tid,
            name=name,
            severity=sev,
            process_name=str(d.get("ProcessName") or ""),
            resources=[str(r) for r in resources][:10],
            detected_at=str(d.get("InitialDetectionTime") or ""),
            action_success=action_success if isinstance(action_success, bool) else None,
            status=threat_status_for_id(d.get("ThreatStatusID")),
        ))
    return out


def parse_scan_trigger(stdout: str) -> tuple[bool, str]:
    """Parse ``_SCAN_SCRIPT``'s JSON. Returns (success, error_message). Pure/testable."""
    text = (stdout or "").strip()
    if not text:
        return False, "no response from Start-MpScan"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False, text[:300]
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        return False, "unexpected response shape"
    return bool(data.get("Success")), str(data.get("Error") or "")


# ── formatting (pure functions, mirrors security_sentinel.format_network_dashboard) ─

def format_status(st: ProtectionStatus) -> str:
    lines = ["Windows Defender status:"]
    if st.realtime_protection is True:
        lines.append("  • Real-time protection: ON")
    elif st.realtime_protection is False:
        lines.append("  • Real-time protection: OFF — this machine is not actively "
                     "protected against new threats right now")
    else:
        lines.append("  • Real-time protection: unknown")
    if st.signature_age_days is not None:
        freshness = ("current" if st.signature_age_days <= STALE_SIGNATURE_DAYS else
                     f"STALE — {st.signature_age_days} day(s) old, updates may be failing")
        lines.append(f"  • Virus definitions: {st.signature_version or '?'} ({freshness})")
    else:
        lines.append(f"  • Virus definitions: {st.signature_version or 'unknown'}")
    if st.quick_scan_age_days is not None:
        lines.append("  • Last quick scan: today" if st.quick_scan_age_days <= 0 else
                     f"  • Last quick scan: {st.quick_scan_age_days} day(s) ago")
    else:
        lines.append("  • Last quick scan: never recorded")
    if st.full_scan_age_days is not None:
        lines.append("  • Last full scan: today" if st.full_scan_age_days <= 0 else
                     f"  • Last full scan: {st.full_scan_age_days} day(s) ago")
    else:
        lines.append("  • Last full scan: never recorded")
    extras = []
    if st.antispyware_enabled is False:
        extras.append("anti-spyware OFF")
    if st.behavior_monitor_enabled is False:
        extras.append("behaviour monitoring OFF")
    if st.nis_enabled is False:
        extras.append("network inspection OFF")
    if extras:
        lines.append(f"  • Also off: {', '.join(extras)}")
    return "\n".join(lines)


def format_threats(threats: list[ThreatDetection]) -> str:
    if not threats:
        return "No recently detected threats — Defender's detection history is clean."
    lines = [f"{len(threats)} recently detected threat(s):"]
    for t in threats:
        where = t.resources[0] if t.resources else (t.process_name or "?")
        lines.append(f"  • {t.name} ({t.severity}) — {t.status}, {where}")
    return "\n".join(lines)


def format_scan_outcome(outcome: ScanOutcome) -> str:
    if not outcome.triggered or not outcome.completed:
        return outcome.message
    header = f"{outcome.message} ({outcome.duration_s:.0f}s)."
    if not outcome.threats:
        return header
    return header + "\n" + format_threats(outcome.threats)


# ── subprocess plumbing (argument list, never shell=True; bounded timeouts) ────

def is_windows() -> bool:
    return sys.platform.startswith("win")


def _powershell_exe() -> str | None:
    for candidate in ("powershell", "pwsh"):
        if shutil.which(candidate):
            return candidate
    return None


def is_available() -> bool:
    """Cheap platform/PATH pre-check. Does NOT guarantee Defender itself will
    answer — that's only known after an actual call, since the Defender
    PowerShell module may be absent even on a Windows box with PowerShell."""
    return is_windows() and _powershell_exe() is not None


def _run_ps(script: str, timeout: float) -> subprocess.CompletedProcess:
    """
    Run one fixed PowerShell script via a plain argument list — never
    shell=True, never a string built from caller-supplied text (see module
    docstring). Mirrors the subprocess discipline in docker_control._run and
    security_sentinel._firewall_status_sync: capture_output, text mode,
    bounded timeout, no console flash on Windows.
    """
    exe = _powershell_exe()
    if exe is None:
        raise FileNotFoundError("no powershell.exe/pwsh found on PATH")
    return subprocess.run(
        [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, timeout=timeout, shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def get_protection_status_sync(timeout: float = _TIMEOUT_STATUS) -> ProtectionStatus | None:
    """Blocking. Call via ``asyncio.to_thread`` from async code. Returns None
    when Defender can't be reached for any reason (not Windows, PowerShell
    missing, Defender module absent, service not responding, timeout)."""
    if not is_windows():
        return None
    try:
        completed = _run_ps(_STATUS_SCRIPT, timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return parse_protection_status(completed.stdout)


def get_recent_threats_sync(timeout: float = _TIMEOUT_THREATS) -> list[ThreatDetection]:
    """Blocking. Call via ``asyncio.to_thread`` from async code. Returns []
    both when Defender is unreachable and when it genuinely has no recent
    detections — callers that need to distinguish should check
    ``get_protection_status_sync`` first."""
    if not is_windows():
        return []
    try:
        completed = _run_ps(_THREATS_SCRIPT, timeout)
    except (subprocess.TimeoutExpired, OSError):
        return []
    if completed.returncode != 0:
        return []
    return parse_threats(completed.stdout)


def run_quick_scan_sync(timeout: float = _TIMEOUT_QUICK_SCAN) -> ScanOutcome:
    """
    Blocking — ``Start-MpScan -ScanType QuickScan`` does not return until the
    scan finishes, which is commonly several minutes. Call via
    ``asyncio.to_thread`` from async code, with a caller-side awareness that
    this may take a while.

    If our own subprocess call times out first, Defender's scan is NOT
    cancelled — ``Start-MpScan`` hands the request to the Defender service
    over WMI, so killing our waiting PowerShell process does not stop the
    scan running server-side; we simply report that we stopped waiting.
    """
    if not is_windows():
        return ScanOutcome(triggered=False, completed=False, scan_type="quick",
                           message="Windows Defender scanning is only available on Windows.")
    start = time.monotonic()
    try:
        completed = _run_ps(_SCAN_SCRIPT, timeout)
    except subprocess.TimeoutExpired:
        return ScanOutcome(
            triggered=True, completed=False, scan_type="quick",
            message=(f"The quick scan is still running after {timeout:.0f}s — Defender "
                     "continues it in the background regardless of this timeout; check "
                     "Windows Security for the result, or ask again shortly."),
            duration_s=time.monotonic() - start)
    except OSError as exc:
        return ScanOutcome(triggered=False, completed=False, scan_type="quick",
                           message=f"Could not start PowerShell to trigger the scan: {exc}")
    duration = time.monotonic() - start
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[:400]
        return ScanOutcome(triggered=False, completed=False, scan_type="quick",
                           message=f"Quick scan could not be started: {detail or 'unknown error'}",
                           duration_s=duration)
    ok, err = parse_scan_trigger(completed.stdout)
    if not ok:
        return ScanOutcome(triggered=False, completed=False, scan_type="quick",
                           message=f"Quick scan could not be started: {err or 'unknown error'}",
                           duration_s=duration)
    # Defender's own detection history after the scan — note this reflects
    # overall history, not strictly "only found by this run" (the cmdlets
    # don't expose a clean per-scan filter).
    threats = get_recent_threats_sync()
    message = ("Quick scan finished — no threats found." if not threats else
               f"Quick scan finished — {len(threats)} threat(s) in Defender's detection "
               "history (may include items from before this scan).")
    return ScanOutcome(triggered=True, completed=True, scan_type="quick", message=message,
                       threats_found=len(threats), threats=threats, duration_s=duration)


def proactive_check_sync(timeout: float = _TIMEOUT_STATUS) -> list[tuple[str, str]]:
    """
    Cheap, read-only health check meant for SecuritySentinel's periodic loop:
    flags real-time protection being off or definitions gone stale. Returns
    (alert_key, message) pairs, or [] when healthy — and also [] when Defender
    is unreachable, deliberately: plenty of machines run a third-party
    antivirus instead of Defender, so silence is the right default rather
    than nagging every cycle about a Defender that was never meant to be
    active. Never triggers a scan, never touches the network.
    """
    st = get_protection_status_sync(timeout)
    if st is None:
        return []
    alerts: list[tuple[str, str]] = []
    if st.realtime_protection is False:
        alerts.append(("av:realtime_off",
                       "Windows Defender real-time protection is OFF — this machine "
                       "is not actively protected right now"))
    if st.signature_age_days is not None and st.signature_age_days > STALE_SIGNATURE_DAYS:
        alerts.append(("av:stale_defs",
                       f"Windows Defender virus definitions are {st.signature_age_days} "
                       "day(s) old — updates may be failing"))
    return alerts


def describe() -> str:
    """One-line availability summary for a diagnostics panel, matching
    docker_control.describe()'s shape."""
    if not is_windows():
        return "Windows Defender integration is only available on Windows."
    if _powershell_exe() is None:
        return "PowerShell not found on PATH — Defender status can't be queried."
    st = get_protection_status_sync()
    if st is None:
        return ("PowerShell is available, but Defender's PowerShell module didn't "
                "respond — it may be absent (a third-party antivirus may own real-time "
                "protection) or the Defender service may be stopped.")
    return "Windows Defender is reachable and reporting status." if st.is_healthy() else \
        "Windows Defender is reachable, but protection is degraded — see antivirus status."


# ── async facade used by the dispatcher (dispatch_files.py) ────────────────────

class AntivirusMonitor:
    """
    On-request (+ light proactive) Windows Defender awareness: protection
    status, a quick scan, and recently detected threats — no custom detection
    logic of any kind, entirely delegated to the OS's own antivirus. See the
    module docstring for the full safety-invariant list.
    """

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry

    def _incr(self, metric: str) -> None:
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr(metric)
            except Exception:
                pass

    async def status(self) -> ToolResult:
        st = await asyncio.to_thread(get_protection_status_sync)
        self._incr("security.antivirus.status")
        if st is None:
            return ToolResult(_UNAVAILABLE_MESSAGE, ok=False)
        if not st.is_healthy():
            self.bus.banner.emit("🛡 Windows Defender protection is degraded", 5)
        return ToolResult(format_status(st))

    async def recent_threats(self, limit: int = 10) -> ToolResult:
        threats = await asyncio.to_thread(get_recent_threats_sync)
        self._incr("security.antivirus.threats")
        limit = max(1, min(int(limit or 10), 25))
        return ToolResult(format_threats(threats[:limit]))

    async def quick_scan(self, timeout: float = _TIMEOUT_QUICK_SCAN) -> ToolResult:
        self.bus.log.emit("ANTIVIRUS: quick scan requested")
        outcome = await asyncio.to_thread(run_quick_scan_sync, timeout)
        self._incr("security.antivirus.quick_scan")
        if outcome.threats_found:
            self.bus.banner.emit(
                f"🛡 Defender quick scan found {outcome.threats_found} threat(s)", 6)
            self.bus.speak_request.emit(
                f"Security alert: Windows Defender's quick scan found "
                f"{outcome.threats_found} threat(s) in its detection history. "
                f"Please review Windows Security.")
        return ToolResult(format_scan_outcome(outcome), ok=outcome.triggered)

    def describe(self) -> str:
        return describe()
