"""
Reachable-endpoint discovery for the remote uplink (#12 — frictionless connect).

The phone should never make the user type an IP. The desktop knows where it can
be reached and hands that list to the paired device, which then auto-connects to
whichever endpoint answers — preferring the Tailscale name, because it resolves
to the same machine at home *and* on mobile data (Tailscale is a WireGuard mesh,
so there's no NAT to punch and no port to forward).

Endpoint kinds, in the order the phone should try them:
  1. ``tailscale``     — the MagicDNS name (e.g. orion-pc.tailXXXX.ts.net). Works
                         everywhere the phone has Tailscale up; the primary path.
  2. ``tailscale-ip``  — the 100.x.y.z address, as a fallback if MagicDNS is off.
  3. ``lan``           — a local 192.168/10.x address: lowest latency at home,
                         useless away from it.

Everything degrades gracefully: no Tailscale installed → just LAN endpoints; no
network at all → an empty list. Stdlib only, and the detection seams are
injectable so this unit-tests without touching real sockets or subprocesses.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from typing import Any, Callable


# ── Tailscale ─────────────────────────────────────────────────────────────────

def tailscale_exe() -> str | None:
    """Locate the Tailscale CLI, honouring an explicit override first."""
    override = os.getenv("ORION_TAILSCALE_PATH")
    if override and os.path.exists(override):
        return override
    found = shutil.which("tailscale")
    if found:
        return found
    if sys.platform == "win32":
        for base in (os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                     os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")):
            candidate = os.path.join(base, "Tailscale", "tailscale.exe")
            if os.path.exists(candidate):
                return candidate
    elif sys.platform == "darwin":
        for candidate in (
            "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            "/usr/local/bin/tailscale",
        ):
            if os.path.exists(candidate):
                return candidate
    return None


def tailscale_status_json(runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Return ``tailscale status --json`` parsed, or {} if unavailable."""
    exe = tailscale_exe()
    if not exe:
        return {}
    run = runner or subprocess.run
    try:
        proc = run([exe, "status", "--json"], capture_output=True, text=True,
                   timeout=6.0)
    except Exception:
        return {}
    out = getattr(proc, "stdout", "") or ""
    if not out.strip():
        return {}
    try:
        data = json.loads(out)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_tailscale_self(status: dict[str, Any]) -> dict[str, Any]:
    """Pull this node's MagicDNS name + Tailscale IPs out of a status dict.

    Only reports when the backend is actually up (``BackendState == 'Running'``),
    so a signed-out or stopped Tailscale never yields a dead endpoint."""
    if not isinstance(status, dict):
        return {"dns": "", "ips": []}
    state = str(status.get("BackendState") or "")
    if state and state != "Running":
        return {"dns": "", "ips": []}
    self_node = status.get("Self") or {}
    if not isinstance(self_node, dict):
        return {"dns": "", "ips": []}
    dns = str(self_node.get("DNSName") or "").rstrip(".")
    ips = [str(ip) for ip in (self_node.get("TailscaleIPs") or []) if ip]
    return {"dns": dns, "ips": ips}


# ── LAN ───────────────────────────────────────────────────────────────────────

def lan_ipv4_addresses() -> list[str]:
    """Best-effort list of this host's private IPv4 addresses (no loopback)."""
    found: list[str] = []
    # The "connect a UDP socket to a public IP" trick reveals the primary
    # outbound interface address without sending a packet.
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            primary = probe.getsockname()[0]
            if primary and not primary.startswith("127."):
                found.append(primary)
        finally:
            probe.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if addr and not addr.startswith("127.") and addr not in found:
                found.append(addr)
    except Exception:
        pass
    return found


# ── assembly ──────────────────────────────────────────────────────────────────

def _url(scheme: str, host: str, port: int) -> str:
    # Bracket IPv6 literals; Tailscale/LAN here are IPv4/DNS, but be safe.
    hostpart = f"[{host}]" if ":" in host else host
    return f"{scheme}://{hostpart}:{int(port)}/"


def detect_endpoints(
    port: int,
    https: bool = True,
    *,
    tailscale: dict[str, Any] | None = None,
    lan: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the ordered endpoint list the phone auto-connects through.

    ``tailscale`` (a ``parse_tailscale_self`` result) and ``lan`` (a list of
    IPv4 strings) may be injected for tests; otherwise they're detected live.
    """
    scheme = "https" if https else "http"
    ts = tailscale if tailscale is not None else parse_tailscale_self(tailscale_status_json())
    lan_ips = lan if lan is not None else lan_ipv4_addresses()

    endpoints: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(kind: str, host: str, priority: int) -> None:
        host = str(host or "").strip()
        if not host or host in seen:
            return
        seen.add(host)
        endpoints.append({
            "kind": kind, "host": host, "port": int(port),
            "https": bool(https), "url": _url(scheme, host, port),
            "priority": priority,
        })

    if ts.get("dns"):
        add("tailscale", ts["dns"], 10)
    for ip in ts.get("ips", []):
        add("tailscale-ip", ip, 20)
    for ip in lan_ips:
        add("lan", ip, 30)

    endpoints.sort(key=lambda e: e["priority"])
    return endpoints
