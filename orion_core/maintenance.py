"""
Database housekeeping (Mark X.12 §5.2).

ORION's SQLite stores accumulate write-ahead-log/-shm siblings and free pages
over time (the ingestion store alone reached ~5 MB with a ~4 MB WAL). Nothing
previously checkpointed or compacted them. This low-priority background task
walks every ``*.db`` under the config directory on a slow cadence (weekly by
default) and, for each store:

  * ``PRAGMA wal_checkpoint(TRUNCATE)`` — folds the write-ahead log back into the
    main file and truncates the WAL (the cheap, always-safe win);
  * ``VACUUM``                          — reclaims free pages and defragments.

Design notes:
  * Runs entirely off the event loop (``asyncio.to_thread``) — it never holds
    the loop that audio/voice depend on.
  * Each store is swept independently on its own short-lived connection: a lock
    or error on one is logged and skipped, never aborting the rest. SQLite's own
    locking guarantees no corruption from touching a live store; the worst case
    is a transient "database is locked" which simply retries next cycle.
  * Discovery is by glob, so a newly-added store is swept automatically without
    editing this file.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .bus import OrionBus
from .constants import CONFIG_DIR

_WEEK_SECONDS = 7 * 24 * 60 * 60
_BUSY_TIMEOUT_MS = 5_000


@dataclass
class HousekeepingReport:
    """Outcome of one sweep across every store."""

    swept: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)   # name -> reason
    bytes_before: int = 0
    bytes_after: int = 0

    @property
    def reclaimed(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)


class DatabaseHousekeeper:
    """Periodic WAL-checkpoint + VACUUM sweep over the config SQLite stores."""

    def __init__(self, bus: OrionBus, config_dir: Path | None = None,
                 *, interval_seconds: float = _WEEK_SECONDS) -> None:
        self.bus = bus
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.interval_seconds = float(interval_seconds)
        self._stop = asyncio.Event()

    # ── discovery ───────────────────────────────────────────────────────────────

    def _stores(self) -> list[Path]:
        return sorted(p for p in self.config_dir.glob("*.db") if p.is_file())

    # ── per-store work (blocking; always called via a worker thread) ─────────────

    def _sweep_one(self, path: Path) -> tuple[bool, str]:
        """Checkpoint + VACUUM a single store. Returns (ok, detail)."""
        try:
            conn = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_MS / 1000.0)
        except sqlite3.Error as exc:
            return False, f"connect-failed({exc})"
        # Autocommit so VACUUM (which cannot run inside a transaction) is legal.
        conn.isolation_level = None
        done: list[str] = []
        try:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                done.append("checkpoint")
            except sqlite3.Error as exc:
                done.append(f"checkpoint-failed({exc})")
            try:
                conn.execute("VACUUM")
                done.append("vacuum")
            except sqlite3.Error as exc:
                done.append(f"vacuum-failed({exc})")
        finally:
            conn.close()
        ok = "checkpoint" in done or "vacuum" in done
        return ok, "+".join(done)

    #: How often to fold the write-ahead logs back into their databases.
    #: Far more often than the weekly VACUUM, and for a different reason —
    #: see checkpoint_all().
    CHECKPOINT_INTERVAL_SECONDS = 300.0

    def _checkpoint_one(self, path: Path) -> tuple[bool, str]:
        """Fold one store's WAL back into it. Cheap; no VACUUM."""
        try:
            conn = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_MS / 1000.0)
        except sqlite3.Error as exc:
            return False, f"connect-failed({exc})"
        conn.isolation_level = None
        try:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return True, "checkpoint"
        except sqlite3.Error as exc:
            return False, f"checkpoint-failed({exc})"
        finally:
            conn.close()

    def checkpoint_all(self) -> int:
        """Collapse every store's WAL into its database. Returns how many.

        This is a DURABILITY measure, not a performance one, and it is why it
        runs on minutes rather than the VACUUM's weekly cadence.

        In WAL mode a store is THREE files — .db, .db-wal, .db-shm — and only
        the first is meaningful on its own. ORION's config directory lives
        inside a synced OneDrive folder, which uploads those three
        independently and can restore them independently. A .db recovered
        without its matching WAL is a database as it stood at the last
        checkpoint, and everything since is simply gone: ORION's amnesia,
        with no error anywhere because SQLite opened a perfectly valid file.

        Checkpointing often keeps the .db current and the WAL near empty, so
        there is only ever one file that matters. It does not defeat WAL's
        write speed: writes still batch into the log in between.
        """
        done = 0
        for path in self._stores():
            ok, _detail = self._checkpoint_one(path)
            done += bool(ok)
        return done

    def sweep(self) -> HousekeepingReport:
        """Synchronous one-shot sweep across all stores (unit-test entry point)."""
        report = HousekeepingReport()
        for path in self._stores():
            report.bytes_before += path.stat().st_size if path.exists() else 0
            ok, detail = self._sweep_one(path)
            report.bytes_after += path.stat().st_size if path.exists() else 0
            if ok:
                report.swept.append(path.name)
            else:
                report.skipped[path.name] = detail
        return report

    # ── async surface ────────────────────────────────────────────────────────────

    async def run_once(self) -> HousekeepingReport:
        report = await asyncio.to_thread(self.sweep)
        summary = (
            f"HOUSEKEEPING: swept {len(report.swept)} store(s), reclaimed "
            f"~{report.reclaimed / 1024.0:.0f} KB"
        )
        if report.skipped:
            details = ", ".join(f"{k} ({v})" for k, v in report.skipped.items())
            summary += f"; skipped {len(report.skipped)}: {details}"
        self.bus.log.emit(summary)
        return report

    def stop(self) -> None:
        self._stop.set()

    async def run_checkpoints(self) -> None:
        """Fold the write-ahead logs back in, every few minutes.

        Separate from the sweep below because it answers a different question.
        VACUUM is about disk space and can wait a week; this is about ORION's
        memory surviving, and a week is exactly how much of it was at risk —
        see checkpoint_all().
        """
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.CHECKPOINT_INTERVAL_SECONDS)
                    return
                except asyncio.TimeoutError:
                    pass
                try:
                    await asyncio.to_thread(self.checkpoint_all)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.bus.log.emit(f"HOUSEKEEPING: checkpoint pass failed - {exc}")
        except asyncio.CancelledError:
            raise

    async def run(self) -> None:
        """Background loop: sweep every ``interval_seconds`` until stopped.

        The first sweep is deferred by one interval — a freshly launched process
        has just-opened, already-compact stores, so there is nothing to reclaim
        at boot and no reason to compete with startup for I/O.
        """
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                    return  # stop() was signalled during the wait
                except asyncio.TimeoutError:
                    pass    # interval elapsed — time to sweep
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:   # never let a maintenance error kill the loop
                    self.bus.log.emit(f"HOUSEKEEPING: sweep failed — {exc}")
        except asyncio.CancelledError:
            return
