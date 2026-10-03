"""
Unattended-node care: an external heartbeat and scheduled backups (brief §4.2).

A headless node has nobody watching it. The brief asked for two cheap
insurances before the node's data becomes load-bearing, and neither existed:

* **Health pings.** ``/api/health`` answers when asked, but nothing asks; a
  crashed node was noticed when the phone stopped answering. With
  ``ORION_HEALTHCHECK_URL`` set (a healthchecks.io check, an Uptime Kuma push
  monitor, or anything that alerts on a missed GET), the node pings it every
  ``ORION_HEALTHCHECK_MINUTES`` (default 5). The monitoring service raises the
  alarm on silence, which is the one failure the node cannot report itself.
* **Scheduled backups.** With ``ORION_NODE_BACKUP_DIR`` set, the node writes a
  BackupManager archive there every ``ORION_NODE_BACKUP_HOURS`` (default 24):
  consistent SQLite snapshots, credentials withheld unless
  ``ORION_BACKUP_SECRETS=1``, eight archives kept.

Both are off unless configured, and neither can stop the node: a failed ping
or backup is logged once per change of state, not every interval.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .utils import first_line


def _float_env(name: str, default: float, low: float) -> float:
    try:
        return max(low, float(os.getenv(name, "") or default))
    except ValueError:
        return default


class _Periodic:
    """A stoppable loop that runs *step* every *interval* seconds."""

    def __init__(self, interval: float, first_delay: float = 0.0) -> None:
        self.interval = float(interval)
        self.first_delay = float(first_delay)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> bool:
        """Wait *seconds*; True when stopped meanwhile."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, seconds))
            return True
        except asyncio.TimeoutError:
            return False

    async def step(self) -> None:          # pragma: no cover - overridden
        raise NotImplementedError

    async def run(self) -> None:
        if self.first_delay and await self._sleep(self.first_delay):
            return
        while not self._stop.is_set():
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass                       # step() reports its own failures
            if await self._sleep(self.interval):
                return


class HealthPinger(_Periodic):
    """GET a monitoring URL on a schedule; the monitor alerts on silence."""

    def __init__(self, url: str, bus: Any, interval: float = 300.0,
                 fetch: Optional[Callable[[str], Awaitable[int]]] = None) -> None:
        super().__init__(interval)
        self.url = url
        self.bus = bus
        self._fetch = fetch or self._get
        self._failing = False
        self.pings = 0

    @classmethod
    def from_env(cls, bus: Any) -> Optional["HealthPinger"]:
        url = os.getenv("ORION_HEALTHCHECK_URL", "").strip()
        if not url.lower().startswith(("https://", "http://")):
            return None
        minutes = _float_env("ORION_HEALTHCHECK_MINUTES", 5.0, 1.0)
        return cls(url, bus, interval=minutes * 60.0)

    @staticmethod
    async def _get(url: str) -> int:
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"User-Agent": "ORION-node"}) as response:
                return response.status

    async def step(self) -> None:
        try:
            status = await self._fetch(self.url)
            ok = 200 <= int(status) < 300
            problem = f"HTTP {status}"
        except Exception as exc:
            ok, problem = False, first_line(exc, 120)
        if ok:
            self.pings += 1
            if self._failing:
                self.bus.log.emit("NODE: health ping reaching the monitor again.")
            self._failing = False
        elif not self._failing:
            self._failing = True
            self.bus.log.emit(f"NODE: health ping failed ({problem}); the monitor "
                              "will alert if this continues.")


class ScheduledBackup(_Periodic):
    """Write a BackupManager archive on a schedule."""

    def __init__(self, manager: Any, bus: Any, interval: float = 86400.0,
                 first_delay: float = 600.0) -> None:
        # The first archive waits ten minutes: a node just started has nothing
        # new to save, and start-up is the busiest I/O moment it has.
        super().__init__(interval, first_delay=first_delay)
        self.manager = manager
        self.bus = bus
        self._failing = False
        self.made = 0

    @classmethod
    def from_env(cls, bus: Any) -> Optional["ScheduledBackup"]:
        folder = os.getenv("ORION_NODE_BACKUP_DIR", "").strip()
        if not folder:
            return None
        from .backup_manager import BackupManager
        hours = _float_env("ORION_NODE_BACKUP_HOURS", 24.0, 1.0)
        manager = BackupManager(bus, destination=Path(folder).expanduser())
        return cls(manager, bus, interval=hours * 3600.0)

    async def step(self) -> None:
        result = await self.manager.backup("scheduled node backup")
        if result.ok:
            self.made += 1
            if self._failing:
                self.bus.log.emit("NODE: scheduled backups are working again.")
            self._failing = False
            self.bus.log.emit(f"NODE: {first_line(result.text, 200)}")
        elif not self._failing:
            self._failing = True
            self.bus.log.emit(f"NODE: scheduled backup failed - {first_line(result.text, 200)}")
