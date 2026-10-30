"""
Home Wi-Fi link for the companion app: the phone reaches ORION directly on the
local network, with no Tailscale and no tunnel.

Owner, 29 Oct 2026: "I want the app to run via my PC so I don't have to use
tailscale ... my PC is pretty much always on in the day ... always connected
to wifi."

What it is
----------
A second HTTPS listener of the same uplink application, bound to the PC's
private LAN address (default port 8765). It serves exactly the routes, the
pairing and the token checks the main uplink does; nothing is reachable here
that is not reachable through Tailscale Serve already.

Why it is no weaker
-------------------
* TLS always. The certificate is ORION's own self-signed one
  (config/remote_tls), so the phone cannot use the public CA system for it.
  Instead the phone PINS it: the SHA-256 of the exact certificate is handed to
  an already-paired phone only inside an authenticated /v1/auth/token or
  /v1/auth/pair answer, and a phone pairing over Wi-Fi for the first time must
  type the certificate code the PC shows beside the pairing code. A device on
  the same Wi-Fi presenting any other certificate gets no token, no request.
* Private peers only. The listener answers 403 to any client whose address is
  not RFC 1918, link-local or IPv6 unique-local, whatever the firewall says.
  Tailscale's 100.64/10 range is not "the LAN" and is refused here.
* A private-profile-only Windows Firewall rule (``remoteip=localsubnet``),
  never public: on a café Wi-Fi set to Public the port stays closed.
* Same pairing-code rate limits, same 15-minute access tokens, same
  per-device revocation.

Off with ``ORION_REMOTE_LAN=0``; the port is ``ORION_REMOTE_LAN_PORT``.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import os
import re
import socket
import subprocess
from typing import Any, Callable

#: The LAN port. 8765 was the uplink's historic direct port, so a phone that
#: once used it already has the right number in its address box.
LAN_PORT_DEFAULT = 8765
FIREWALL_RULE = "ORION Phone LAN (HTTPS)"
#: How often the bound address is re-checked (DHCP renewals, a new router).
WATCH_S = 60.0

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_PRIVATE_V4 = tuple(ipaddress.ip_network(c) for c in
                    ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"))
_PRIVATE_V6 = tuple(ipaddress.ip_network(c) for c in ("fc00::/7", "fe80::/10"))
_TAILNET = ipaddress.ip_network("100.64.0.0/10")


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off", ""}


def lan_port() -> int:
    """The LAN listener's port; a bad value raises rather than guessing."""
    raw = os.getenv("ORION_REMOTE_LAN_PORT", "").strip()
    port = int(raw) if raw else LAN_PORT_DEFAULT
    if not 1024 <= port <= 65535:
        raise ValueError("ORION_REMOTE_LAN_PORT must be between 1024 and 65535")
    return port


def lan_wanted(host: str, exposure: str, main_port: int) -> bool:
    """Whether this uplink should also listen on the home network.

    Not for a cloud container or a reverse proxy (there is no home network),
    and not when the main listener is itself bound to the LAN already (the
    legacy direct mode): one listener per interface."""
    if not _flag("ORION_REMOTE_LAN", "1"):
        return False
    if exposure in {"reverse-proxy", "platform-proxy"}:
        return False
    if host in _LOOPBACK:
        pass
    else:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        if address not in _TAILNET:
            return False           # 0.0.0.0 or a LAN IP: already on the LAN
    try:
        return lan_port() != int(main_port)
    except ValueError:
        return False


def is_private_peer(value: Any) -> bool:
    """True for a client on the local network: RFC 1918, link-local or IPv6
    unique-local (an IPv4-mapped IPv6 address is judged as its IPv4).
    Tailscale's CGNAT range, loopback and every public address are False."""
    try:
        address = ipaddress.ip_address(str(value or "").split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if isinstance(address, ipaddress.IPv4Address):
        return any(address in net for net in _PRIVATE_V4)
    return any(address in net for net in _PRIVATE_V6)


def certificate_sha256(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def display_code(sha256_hex: str) -> str:
    """The certificate code a person compares or types: the first 64 bits of
    the fingerprint in four groups, e.g. ``1A2B-3C4D-5E6F-7A8B``. Forging a
    certificate whose fingerprint starts the same is a 2^64 search."""
    head = "".join(ch for ch in str(sha256_hex).upper() if ch in "0123456789ABCDEF")[:16]
    return "-".join(head[i:i + 4] for i in range(0, len(head), 4))


def lan_endpoint(ip: str, port: int, sha256_hex: str) -> dict[str, Any]:
    """The endpoint entry a paired phone learns. Priority 5 puts it ahead of
    Tailscale (10): on the home Wi-Fi the phone tries the PC directly first,
    and away from home that address simply does not answer."""
    return {"kind": "lan", "host": ip, "port": int(port), "https": True,
            "url": f"https://{ip}:{int(port)}/", "priority": 5,
            "cert_sha256": str(sha256_hex).lower()}


def firewall_rule_args(port: int) -> list[str]:
    """netsh arguments for the inbound rule: Private profile only, and only
    from addresses on the PC's own subnets. Never the Public profile."""
    return ["netsh", "advfirewall", "firewall", "add", "rule",
            f"name={FIREWALL_RULE}", "dir=in", "action=allow", "protocol=TCP",
            f"localport={int(port)}", "profile=private", "remoteip=localsubnet",
            "description=ORION companion app on the home network (HTTPS, paired devices)"]


def _rule_has_port(show_stdout: str, port: int) -> bool:
    """Whether a shown rule names *port*. Read as numbers anywhere in the text,
    not by the "LocalPort" label, which Windows translates."""
    return str(int(port)) in re.findall(r"\b\d{2,5}\b", str(show_stdout or ""))


def _rule_profiles(show_stdout: str) -> str:
    for line in str(show_stdout or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "profiles":
            return value.strip().lower()
    return ""


#: Adapter names that are not the home network: virtual switches (WSL,
#: Hyper-V), VPNs and tunnels, VM host adapters, Bluetooth.
_VIRTUAL_ADAPTER = re.compile(
    r"(?i)vethernet|wsl|hyper-v|virtualbox|vmware|vmnet|\btap\b|\btun\b|vpn|tailscale|"
    r"zerotier|wireguard|wg\d|loopback|bluetooth|npcap|docker")


def home_lan_ip(candidates: Callable[[], list[str]] | None = None,
                adapters: Callable[[], dict[str, list[str]]] | None = None,
                up: Callable[[], set[str]] | None = None) -> str:
    """The PC's address on its real Wi-Fi or Ethernet adapter, or "".

    The first private address used to win: with a VPN up that was the VPN's
    10.x address, and with the Wi-Fi down a WSL or Hyper-V 172.x address -
    advertised to the phone, and its peers passed the private-peer check
    (review, 29 Oct 2026)."""
    if candidates is None:
        from .remote_endpoints import lan_ipv4_addresses as candidates
    if adapters is None or up is None:
        import psutil
        if adapters is None:
            def adapters() -> dict[str, list[str]]:
                return {name: [a.address for a in addrs if a.family == socket.AF_INET]
                        for name, addrs in psutil.net_if_addrs().items()}
        if up is None:
            def up() -> set[str]:
                return {name for name, stats in psutil.net_if_stats().items() if stats.isup}
    try:
        by_ip = {ip: name for name, ips in adapters().items() for ip in ips}
        live = up()
        for ip in candidates():
            name = by_ip.get(ip, "")
            if name and name in live and not _VIRTUAL_ADAPTER.search(name):
                return ip
    except Exception:
        return ""
    return ""


class LanLink:
    """The home-network listener of one :class:`RemoteGateway`."""

    def __init__(self, gateway: Any, *,
                 lan_ip: Callable[[], str] | None = None,
                 on_windows: Callable[[], bool] | None = None) -> None:
        self.gateway = gateway
        self._lan_ip = lan_ip or home_lan_ip
        self._on_windows = on_windows
        #: Which addresses may be bound (default: private ones). A seam for
        #: tests, separate from the request guard on purpose.
        self._bindable: Callable[[str], bool] | None = None
        self.port = 0
        self.ip = ""
        self.sha256 = ""
        self._site: Any = None
        self._context: Any = None
        self._watch: asyncio.Task | None = None
        self._failed: tuple[str, int, str] | None = None   # the last bind failure, logged once

    # ── what the rest of the gateway asks ────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._site is not None and bool(self.ip and self.sha256)

    def endpoint(self) -> dict[str, Any] | None:
        return lan_endpoint(self.ip, self.port, self.sha256) if self.active else None

    def pairing_hint(self) -> str:
        """The line the Security Centre shows beside a pairing code."""
        if not self.active:
            return ""
        return (f"On this Wi-Fi use https://{self.ip}:{self.port} with certificate code "
                f"{display_code(self.sha256)}.")

    def is_lan_request(self, request: Any) -> bool:
        if not self.active:
            return False
        try:
            sockname = request.transport.get_extra_info("sockname")
        except Exception:
            return False
        return bool(sockname) and int(sockname[1]) == self.port and str(sockname[0]) == self.ip

    def middleware(self) -> Any:
        """aiohttp middleware: the LAN listener answers private peers only."""
        from aiohttp import web

        @web.middleware
        async def lan_guard(request: Any, handler: Any) -> Any:
            if self.is_lan_request(request) and not is_private_peer(request.remote):
                return web.json_response(
                    {"ok": False, "error": "the home-network link serves local devices only"},
                    status=403)
            return await handler(request)

        return lan_guard

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self, runner: Any, web: Any, tls: Callable[[], tuple[Any, bytes]]) -> bool:
        """Bind the LAN listener beside the main one. Never raises: a failure
        here must not take the Tailscale or tunnel path down with it."""
        gateway = self.gateway
        log = gateway.bus.log.emit
        exposure = os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower()
        if not lan_wanted(gateway.host, exposure, gateway.port):
            return False
        try:
            self.port = lan_port()
            self._context, der = await asyncio.to_thread(tls)
            self.sha256 = certificate_sha256(der)
        except Exception as exc:
            log(f"REMOTE: home Wi-Fi link off - certificate unavailable ({exc}).")
            return False
        bound = await self._bind(runner, web)
        if bound:
            await self.ensure_firewall()
        if self._watch is None:
            self._watch = asyncio.create_task(self._watch_address(runner, web),
                                              name="orion-remote-lan-watch")
        return bound

    async def _bind(self, runner: Any, web: Any) -> bool:
        log = self.gateway.bus.log.emit
        ip = await asyncio.to_thread(self._lan_ip)
        if not (self._bindable or is_private_peer)(ip):
            if self.ip:
                log("REMOTE: home Wi-Fi link paused - this PC has no private network address.")
            await self._unbind()
            return False
        if ip == self.ip and self._site is not None:
            return True
        await self._unbind()
        site = web.TCPSite(runner, ip, self.port, ssl_context=self._context)
        try:
            await site.start()
        except Exception as exc:
            # A site that failed to start stays registered with the runner:
            # the watcher's retry every minute added another one (and another
            # log line) while the port was taken (review, 29 Oct 2026).
            try:
                await site.stop()
            except Exception:
                pass
            failure = (ip, self.port, type(exc).__name__)
            if failure != self._failed:
                self._failed = failure
                log(f"REMOTE: home Wi-Fi link could not listen on {ip}:{self.port} ({exc}).")
            return False
        self._failed = None
        self._site, self.ip = site, ip
        self.gateway._endpoints_cache = None
        log(f"REMOTE: home Wi-Fi link at https://{ip}:{self.port}/ (certificate code "
            f"{display_code(self.sha256)}). Paired phones on this network connect "
            "directly; Tailscale is only needed away from home.")
        return True

    async def _unbind(self) -> None:
        site, self._site = self._site, None
        self.ip = ""
        if site is not None:
            self.gateway._endpoints_cache = None
            try:
                await site.stop()
            except Exception:
                pass

    async def _watch_address(self, runner: Any, web: Any) -> None:
        """Follow the PC's LAN address: a DHCP renewal or a new router moves
        it, and the phone re-learns the new one (or finds it by its pinned
        certificate)."""
        while True:
            await asyncio.sleep(WATCH_S)
            if getattr(self.gateway, "_closing", False):
                return
            try:
                await self._bind(runner, web)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    def detach(self) -> asyncio.Task | None:
        """Gateway shutdown: cancel the address watcher and forget the site
        WITHOUT stopping it - the runner's cleanup stops every site it owns,
        and a second stop of the same site raises inside that cleanup."""
        watch, self._watch = self._watch, None
        self._site, self.ip = None, ""
        if watch is not None:
            watch.cancel()
        return watch

    async def stop(self) -> None:
        watch, self._watch = self._watch, None
        if watch is not None:
            watch.cancel()
            try:
                await watch
            except (asyncio.CancelledError, Exception):
                pass
        await self._unbind()

    # ── Windows Firewall ─────────────────────────────────────────────────────

    async def ensure_firewall(self) -> None:
        """Make sure the Private-only rule exists. Idempotent; never raises;
        never touches the Public profile. Adding a rule needs Administrator,
        so a refusal logs the exact command to run once."""
        on_windows = self._on_windows
        if on_windows is None:
            from . import remote as _remote
            on_windows = _remote._on_windows
        if not on_windows() or not _flag("ORION_REMOTE_FIREWALL", "1"):
            return
        log = self.gateway.bus.log.emit

        def _run(args: list[str]) -> Any:
            return subprocess.run(
                args, capture_output=True, text=True, errors="replace", timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        try:
            show = await asyncio.to_thread(
                _run, ["netsh", "advfirewall", "firewall", "show", "rule",
                       f"name={FIREWALL_RULE}"])
            if show.returncode == 0 and FIREWALL_RULE in (show.stdout or ""):
                profiles = _rule_profiles(show.stdout)
                if "public" in profiles or profiles == "any":
                    log(f"REMOTE: the firewall rule '{FIREWALL_RULE}' also allows Public "
                        "networks. Restrict it to Private in Windows Defender Firewall.")
                if not _rule_has_port(show.stdout, self.port):
                    # ORION_REMOTE_LAN_PORT changed: the rule still opened the old port.
                    update = await asyncio.to_thread(
                        _run, ["netsh", "advfirewall", "firewall", "set", "rule",
                               f"name={FIREWALL_RULE}", "new", f"localport={self.port}"])
                    if update.returncode != 0:
                        log(f"REMOTE: the firewall rule '{FIREWALL_RULE}' opens another port. "
                            "Run once in an elevated PowerShell: netsh advfirewall firewall set "
                            f'rule name="{FIREWALL_RULE}" new localport={self.port}')
                return
            args = firewall_rule_args(self.port)
            add = await asyncio.to_thread(_run, args)
            if add.returncode == 0:
                log(f"REMOTE: firewall rule added for the home Wi-Fi link (TCP {self.port}, "
                    "Private networks, local subnet only).")
            else:
                log("REMOTE: the home Wi-Fi link needs one firewall rule (Administrator). "
                    "Run once in an elevated PowerShell: netsh advfirewall firewall add rule "
                    f'name="{FIREWALL_RULE}" dir=in action=allow protocol=TCP '
                    f"localport={self.port} profile=private remoteip=localsubnet")
        except Exception:
            pass


__all__ = [
    "FIREWALL_RULE", "LAN_PORT_DEFAULT", "LanLink", "certificate_sha256",
    "display_code", "firewall_rule_args", "home_lan_ip", "is_private_peer", "lan_endpoint",
    "lan_port", "lan_wanted",
]
