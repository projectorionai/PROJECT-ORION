"""
ConnectivityMonitor (Phase 1) — the switch between MODE A (cloud-enhanced) and
MODE B (fully offline).

A single cheap TCP probe to well-known hosts decides whether the internet is
reachable, cached for a few seconds so hot paths never block.  The router asks
``is_online()``; a background refresh keeps it current and emits a bus event on
every transition so the GUI can show which mode ORION is running in.

No DNS-only checks (captive portals lie) — we open a raw socket to port 443 on
resilient anycast hosts and treat a completed handshake as "online".
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any

# Anycast hosts that answer on 443 almost everywhere; first success wins.
_PROBE_HOSTS = (
    ("1.1.1.1", 443),          # Cloudflare
    ("8.8.8.8", 443),          # Google
    ("208.67.222.222", 443),   # OpenDNS
)


class ConnectivityMonitor:
    CACHE_TTL = 6.0
    REFRESH_INTERVAL = 15.0

    def __init__(self, bus: Any | None = None, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._online = self._probe()
        self._checked_at = time.monotonic()
        self._stop = asyncio.Event()

    # ── synchronous probe ─────────────────────────────────────────────────────

    def _probe(self, timeout: float = 1.2) -> bool:
        for host, port in _PROBE_HOSTS:
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except OSError:
                continue
        return False

    def is_online(self, force: bool = False) -> bool:
        """Cached connectivity — refreshes at most every CACHE_TTL seconds."""
        if force or (time.monotonic() - self._checked_at) > self.CACHE_TTL:
            self._update(self._probe())
        return self._online

    def _update(self, online: bool) -> None:
        self._checked_at = time.monotonic()
        if online != self._online:
            self._online = online
            if self.telemetry is not None:
                self.telemetry.metrics.gauge("net.online", 1.0 if online else 0.0)
                self.telemetry.log.info("NET", f"connectivity {'restored' if online else 'lost'}",
                                        mode="MODE A" if online else "MODE B")
            if self.bus is not None:
                try:
                    self.bus.dashboard_event.emit("connectivity", online)
                    self.bus.banner.emit(
                        "ONLINE — cloud intelligence available" if online
                        else "OFFLINE — running on local intelligence", 2)
                except RuntimeError:
                    pass
        else:
            self._online = online

    # ── background refresh ────────────────────────────────────────────────────

    async def run(self) -> None:
        if self.telemetry is not None:
            self.telemetry.health.register("connectivity")
        try:
            while not self._stop.is_set():
                online = await asyncio.to_thread(self._probe)
                self._update(online)
                if self.telemetry is not None:
                    self.telemetry.health.beat(
                        "connectivity", "OK", "online" if online else "offline")
                await asyncio.sleep(self.REFRESH_INTERVAL)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()

    def mode(self) -> str:
        return "MODE A (cloud-enhanced)" if self._online else "MODE B (fully offline)"
