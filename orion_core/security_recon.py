"""
security_recon.py — real cybersecurity/pentesting execution tooling.

Everything that touches a network target routes through
cyber_curriculum.gate_security_action(AUTHORIZED_LAB, ...), which is
default-deny: loopback is always allowed, anything else must already be in
the authorized_targets.py store, and the gate NEVER permits an INTRUSIVE
activity regardless of scope — this subsystem automates reconnaissance and
analysis against targets the user has explicitly authorized (their own lab,
a CTF box they've been given, etc), never real-world intrusion.

craft_packet is deliberately narrow — a single ICMP echo, not a generic
arbitrary-packet builder and not loopable — so it can never become a
flood/DoS primitive.

Graceful degradation: python-nmap needs the real `nmap` binary installed
separately (detected via shutil.which); scapy on Windows needs the Npcap
driver installed separately for raw-socket operations. Neither gets
auto-installed here — a pip package is a safe, reversible action; a
kernel-mode driver install is a materially different one (see
social_automation.py's module docstring for the same reasoning applied to
Playwright/Chrome).
"""

from __future__ import annotations

import asyncio
import shutil
import time
from typing import Any

from .authorized_targets import (
    add_authorized_target,
    list_authorized_targets,
    load_authorized_targets,
    remove_authorized_target,
)
from .bus import OrionBus
from .cyber_curriculum import SecurityActivityClass, gate_security_action
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation

_NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"


class SecurityReconService:
    """Real recon/scanning tooling for the user's own authorized targets."""

    # Published CVE records are effectively static over a working session —
    # re-querying NVD for the same ID minutes apart buys nothing but latency
    # (and burns the public API's rate limit). Per-process and in-memory by
    # design: nothing here is worth persisting across restarts.
    _CVE_CACHE_TTL_S: float = 3600.0

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._cve_cache: dict[str, tuple[float, ToolResult]] = {}

    def _safe_target(self, target: str) -> str:
        return SecuritySanitiser.guard_text(str(target or ""), "security_recon.target").strip()

    # ── scope management ────────────────────────────────────────────────────

    def authorize_target(self, target: str) -> ToolResult:
        target = str(target or "").strip()
        if not target:
            return ToolResult("Give me a target to authorize.", ok=False)
        targets = add_authorized_target(target)
        self.bus.log.emit(f"SECURITY_RECON: authorized '{target}' for testing.")
        return ToolResult(
            f"Authorized '{target}' for security testing. "
            f"Current scope: {', '.join(sorted(targets)) or 'none'}."
        )

    def revoke_target(self, target: str) -> ToolResult:
        target = str(target or "").strip()
        if not target:
            return ToolResult("Give me a target to revoke.", ok=False)
        targets = remove_authorized_target(target)
        return ToolResult(
            f"Revoked '{target}'. Current scope: {', '.join(sorted(targets)) or 'none'}."
        )

    def list_authorized(self) -> ToolResult:
        targets = list_authorized_targets()
        if not targets:
            return ToolResult(
                "No targets are authorized for security testing "
                "(loopback is always allowed)."
            )
        return ToolResult("Authorized targets: " + ", ".join(targets))

    def _check_scope(self, target: str) -> tuple[bool, str]:
        scope = load_authorized_targets()
        return gate_security_action(SecurityActivityClass.AUTHORIZED_LAB, target, scope)

    # ── recon ────────────────────────────────────────────────────────────────

    async def scan_host(self, target: str, ports: str = "") -> ToolResult:
        try:
            target = self._safe_target(target)
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        if not target:
            return ToolResult("Give me a target to scan.", ok=False)
        ok, reason = self._check_scope(target)
        if not ok:
            return ToolResult(f"Scan refused: {reason}", ok=False)
        if shutil.which("nmap") is None:
            return ToolResult(
                "nmap isn't installed on this host — install it from nmap.org, "
                "then try again.", ok=False,
            )
        try:
            import nmap
            scanner = nmap.PortScanner()
            self.bus.log.emit(f"SECURITY_RECON: scanning '{target}'...")
            result = await asyncio.to_thread(scanner.scan, target, ports or None)
        except Exception as exc:
            return ToolResult(f"Scan failed: {exc}", ok=False)
        return ToolResult(self._format_scan(target, result))

    @staticmethod
    def _format_scan(target: str, result: dict[str, Any]) -> str:
        hosts = (result or {}).get("scan") or {}
        if not hosts:
            return f"No results for '{target}' — host may be down or filtering probes."
        lines = [f"Scan results for {target}:"]
        for host, info in hosts.items():
            state = (info.get("status") or {}).get("state", "unknown")
            lines.append(f"  host {host}: {state}")
            for proto in ("tcp", "udp"):
                for port, meta in sorted((info.get(proto) or {}).items()):
                    lines.append(
                        f"    {proto}/{port}: {meta.get('state')} "
                        f"{meta.get('name', '')} {meta.get('product', '')}".rstrip()
                    )
        return "\n".join(lines)

    async def packet_capture(self, target: str, count: int = 10, timeout: float = 10.0) -> ToolResult:
        try:
            target = self._safe_target(target)
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        if not target:
            return ToolResult("Give me a target to capture traffic for.", ok=False)
        ok, reason = self._check_scope(target)
        if not ok:
            return ToolResult(f"Capture refused: {reason}", ok=False)
        try:
            from scapy.all import sniff
        except Exception as exc:
            return ToolResult(f"scapy is unavailable: {exc}", ok=False)

        count = max(1, min(200, int(count or 10)))
        timeout = max(1.0, min(60.0, float(timeout or 10.0)))

        def _capture() -> list[str]:
            packets = sniff(filter=f"host {target}", count=count, timeout=timeout)
            return [p.summary() for p in packets]

        try:
            self.bus.log.emit(f"SECURITY_RECON: capturing traffic for '{target}'...")
            summaries = await asyncio.to_thread(_capture)
        except Exception as exc:
            return ToolResult(
                f"Packet capture failed: {exc}. On Windows this needs the Npcap "
                "driver installed from npcap.com.", ok=False,
            )
        if not summaries:
            return ToolResult(f"No packets captured for '{target}' within {timeout:.0f}s.")
        return ToolResult(
            f"Captured {len(summaries)} packet(s) for {target}:\n" + "\n".join(summaries)
        )

    async def craft_packet(self, target: str) -> ToolResult:
        """Sends exactly ONE ICMP echo request and reports the reply."""
        try:
            target = self._safe_target(target)
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        if not target:
            return ToolResult("Give me a target to ping.", ok=False)
        ok, reason = self._check_scope(target)
        if not ok:
            return ToolResult(f"Packet send refused: {reason}", ok=False)
        try:
            from scapy.all import ICMP, IP, sr1
        except Exception as exc:
            return ToolResult(f"scapy is unavailable: {exc}", ok=False)

        def _ping() -> Any:
            return sr1(IP(dst=target) / ICMP(), timeout=3, verbose=0)

        try:
            self.bus.log.emit(f"SECURITY_RECON: sending ICMP echo to '{target}'...")
            reply = await asyncio.to_thread(_ping)
        except Exception as exc:
            return ToolResult(
                f"Packet send failed: {exc}. On Windows this needs the Npcap "
                "driver installed from npcap.com.", ok=False,
            )
        if reply is None:
            return ToolResult(f"No reply from {target} within 3s (host down, or blocking ICMP).")
        return ToolResult(f"Reply from {target}: {reply.summary()}")

    # ── CVE lookup (public, read-only — no scope gate needed) ─────────────────

    async def cve_lookup(self, query: str) -> ToolResult:
        query = str(query or "").strip()
        if not query:
            return ToolResult("Give me a CVE ID or a product/keyword to search.", ok=False)
        cache_key = query.upper()
        cached = self._cve_cache.get(cache_key)
        if cached is not None and (time.monotonic() - cached[0]) < self._CVE_CACHE_TTL_S:
            return cached[1]
        params = (
            {"cveId": query} if query.upper().startswith("CVE-")
            else {"keywordSearch": query}
        )
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _NVD_API, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        return ToolResult(f"NVD lookup failed: HTTP {resp.status}", ok=False)
                    data = await resp.json()
        except Exception as exc:
            return ToolResult(f"CVE lookup failed: {exc}", ok=False)
        vulns = data.get("vulnerabilities") or []
        if not vulns:
            result = ToolResult(f"No CVE results for '{query}'.")
        else:
            lines = [f"CVE results for '{query}':"]
            for item in vulns[:5]:
                cve = item.get("cve") or {}
                cve_id = cve.get("id", "?")
                descs = cve.get("descriptions") or []
                desc = next((d.get("value", "") for d in descs if d.get("lang") == "en"), "")
                severity = self._extract_severity(cve.get("metrics") or {})
                lines.append(f"  {cve_id} [{severity or 'n/a'}]: {desc[:200]}")
            result = ToolResult("\n".join(lines))
        # Only successful lookups are cached — a transient network failure or
        # an HTTP error returns early above, so it can never get stuck cached.
        self._cve_cache[cache_key] = (time.monotonic(), result)
        return result

    @staticmethod
    def _extract_severity(metrics: dict[str, Any]) -> str:
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            entries = metrics.get(key)
            if entries:
                data = entries[0].get("cvssData", {}) or {}
                return data.get("baseSeverity", "") or entries[0].get("baseSeverity", "")
        return ""
