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
import os
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
    #: Failed probes in a row before ORION believes the internet has gone.
    #: One round of three timed-out connects was enough to switch the whole
    #: mode: on 7 Oct at 21:32 a routine channel rotation found "no live audio
    #: provider" (and no text fallback either) and the voice was gone for five
    #: seconds, with no provider fault logged anywhere. Coming back online
    #: still takes a single success.
    OFFLINE_AFTER = 2
    #: How soon a failed probe is checked again, so a real outage is still
    #: noticed within a few seconds.
    RECHECK_INTERVAL = 2.0

    def __init__(self, bus: Any | None = None, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._offline_only = os.getenv("ORION_MODE", "").strip().lower() == "offline"
        # Desktop and headless boot both construct this object inside an
        # already running event loop. A synchronous three-host probe can hold
        # that loop for several seconds on a mobile or captive connection.
        # Their run() task performs the first probe in a worker thread.
        if self._offline_only:
            self._online = False
            self._checked_at = time.monotonic()
            self._has_result = True
        else:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                self._online = self._probe()
                self._checked_at = time.monotonic()
                self._has_result = True
            else:
                self._online = False
                self._checked_at = 0.0
                self._has_result = False
        self._stop = asyncio.Event()

    # ── synchronous probe ─────────────────────────────────────────────────────

    def _probe(self, timeout: float = 1.2) -> bool:
        if self._offline_only:
            return False
        for host, port in _PROBE_HOSTS:
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except OSError:
                continue
        return False

    def is_online(self, force: bool = False) -> bool:
        """Cached connectivity — refreshes at most every CACHE_TTL seconds.

        The probe is a blocking TCP connect (up to 1.2 s per host), and this is
        read from the event loop by the deck's status strip and the health
        model: MEASURED as a 261 ms stall. With a loop running, a stale answer
        is refreshed in a worker thread and the cached one returned now; the
        background monitor (run) keeps it current anyway.
        """
        if self._offline_only:
            return False
        stale = force or (time.monotonic() - self._checked_at) > self.CACHE_TTL
        if not stale:
            return self._online
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            self._update(self._probe())
        elif not getattr(self, "_refreshing", False):
            self._refreshing = True

            def _done(future: Any) -> None:
                self._refreshing = False
                try:
                    self._update(bool(future.result()))
                except Exception:
                    pass

            loop.run_in_executor(None, self._probe).add_done_callback(
                lambda future: loop.call_soon_threadsafe(_done, future))
        return self._online

    def cached_online(self) -> bool:
        """Last known connectivity only: safe for audio and GUI hot paths."""
        return not self._offline_only and self._has_result and self._online

    def _update(self, online: bool) -> None:
        self._checked_at = time.monotonic()
        self._has_result = True
        if online:
            self._misses = 0
        elif self._online:
            self._misses = getattr(self, "_misses", 0) + 1
            if self._misses < self.OFFLINE_AFTER:
                # Not yet: one miss is a busy moment as often as an outage.
                # Marked stale so the next reader re-probes rather than
                # trusting this for CACHE_TTL.
                self._checked_at = 0.0
                return
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
                online = False if self._offline_only else await asyncio.to_thread(self._probe)
                self._update(online)
                if self.telemetry is not None:
                    self.telemetry.health.beat(
                        "connectivity", "OK", "online" if online else "offline")
                await asyncio.sleep(self.RECHECK_INTERVAL if getattr(self, "_misses", 0)
                                    and self._online else self.REFRESH_INTERVAL)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()

    def mode(self) -> str:
        if not self._has_result:
            return "CHECKING CONNECTIVITY"
        return "MODE A (cloud-enhanced)" if self._online else "MODE B (fully offline)"
