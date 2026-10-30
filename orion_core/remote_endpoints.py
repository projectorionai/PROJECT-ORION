"""
Reachable-endpoint discovery for the remote uplink (#12 — frictionless connect).

The phone should never make the user type an IP. In Tailscale Serve mode the
desktop advertises only its HTTPS MagicDNS name on port 443. Explicit legacy
direct-bind deployments may still advertise LAN and Tailscale IP fallbacks.

Legacy direct-bind endpoint kinds, in the order the phone should try them:
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

import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from typing import Any, Callable
from urllib.parse import urlsplit


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
        proc = run([exe, "status", "--json"], capture_output=True, text=True, encoding="utf-8", errors="replace",
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

_LAN_NETWORKS = tuple(ipaddress.ip_network(cidr) for cidr in
                      ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))


def _is_lan_ipv4(value: str) -> bool:
    try:
        address = ipaddress.IPv4Address(value)
    except (ipaddress.AddressValueError, ValueError):
        return False
    return any(address in network for network in _LAN_NETWORKS)

def lan_ipv4_addresses() -> list[str]:
    """Best-effort RFC 1918 LAN addresses, excluding Tailscale's CGNAT range."""
    found: list[str] = []
    # The "connect a UDP socket to a public IP" trick reveals the primary
    # outbound interface address without sending a packet.
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            primary = probe.getsockname()[0]
            if _is_lan_ipv4(primary):
                found.append(primary)
        finally:
            probe.close()
    except Exception:
        pass
    for addr in _host_addresses():
        if _is_lan_ipv4(addr) and addr not in found:
            found.append(addr)
    return found


#: The longest the host-name lookup below may hold its caller.
HOST_LOOKUP_TIMEOUT_S = 1.5


def _host_addresses(timeout: float = HOST_LOOKUP_TIMEOUT_S) -> list[str]:
    """IPv4 addresses for this PC's own host name, or [] if that takes too long.

    getaddrinfo(gethostname()) is normally instant, but it goes through the
    resolver: on 20 Oct it held ORION's startup for 17.9 s (the event loop,
    via the remote uplink's "open this on your phone" line) while DNS timed
    out. The lookup runs on a daemon thread and is abandoned after *timeout*;
    the primary-interface probe above already found the address that matters.
    """
    result: list[str] = []
    done = threading.Event()

    def lookup() -> None:
        try:
            result.extend(info[4][0] for info in
                          socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=lookup, name="orion-host-lookup", daemon=True).start()
    return list(result) if done.wait(timeout) else []


# ── assembly ──────────────────────────────────────────────────────────────────

def _url(scheme: str, host: str, port: int) -> str:
    # Bracket IPv6 literals; Tailscale/LAN here are IPv4/DNS, but be safe.
    hostpart = f"[{host}]" if ":" in host else host
    return f"{scheme}://{hostpart}:{int(port)}/"


def public_https_endpoint(value: str) -> dict[str, Any] | None:
    """Validate the named HTTPS origin used by a dedicated reverse proxy.

    Only a DNS hostname on the standard HTTPS port is advertised. A path,
    credentials, or a local address would send paired phones to the wrong
    destination, so these are rejected rather than silently corrected.
    """
    try:
        parsed = urlsplit(str(value or "").strip())
        host = (parsed.hostname or "").lower().rstrip(".")
        labels = host.split(".")
        valid_host = (len(host) <= 253 and len(labels) >= 2 and all(
            1 <= len(label) <= 63 and
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
            for label in labels))
        try:
            ipaddress.ip_address(host)
            valid_host = False
        except ValueError:
            pass
        if (parsed.scheme != "https" or not valid_host or parsed.port not in (None, 443)
                or parsed.username or parsed.password or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment):
            return None
        return {"kind": "reverse-proxy", "host": host, "port": 443,
                "https": True, "url": _url("https", host, 443), "priority": 10}
    except ValueError:
        return None


def detect_endpoints(
    port: int,
    https: bool = True,
    *,
    serve: bool = False,
    tailnet_only: bool = False,
    tailscale: dict[str, Any] | None = None,
    lan: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the ordered endpoint list the phone auto-connects through.

    ``tailscale`` (a ``parse_tailscale_self`` result) and ``lan`` (a list of
    IPv4 strings) may be injected for tests; otherwise they're detected live.
    """
    scheme = "https" if https or serve else "http"
    ts = tailscale if tailscale is not None else parse_tailscale_self(tailscale_status_json())
    lan_ips = [] if serve or tailnet_only else (lan if lan is not None else lan_ipv4_addresses())

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
        add("tailscale-serve" if serve else "tailscale", ts["dns"], 10)
    if serve:
        # Serve terminates TLS on 443. A raw 100.x address or loopback port
        # bypasses that TLS endpoint and can collide with mobile CGNAT routes.
        for endpoint in endpoints:
            endpoint.update(port=443, https=True,
                            url=_url("https", endpoint["host"], 443))
        return endpoints
    if tailnet_only:
        # The direct TLS listener is bound to this node's Tailscale IP. The
        # phone uses the named HTTPS origin so certificate pinning stays stable.
        return endpoints
    for ip in ts.get("ips", []):
        add("tailscale-ip", ip, 20)
    for ip in lan_ips:
        if _is_lan_ipv4(ip):
            add("lan", ip, 30)

    endpoints.sort(key=lambda e: e["priority"])
    return endpoints
