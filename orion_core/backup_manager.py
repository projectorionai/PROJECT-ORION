"""
Local backup & synchronisation manager (improvement #30).

``BackupManager`` snapshots ORION's important state — the config directory
(provider settings, integrations, pipeline, knowledge packs), the SQLite memory
database, and the knowledge corpus manifest — into a single timestamped ``.zip``
archive.

Because this machine's project already lives under OneDrive, the default backup
destination is a ``ORION_Backups`` folder in the user's OneDrive root, so the
archive can be picked up by the user's configured sync client. The receipt
identifies the destination; it does not attest cloud delivery. A different
destination can be supplied.

Backups are pruned to a retention count, and ``restore`` unpacks a chosen (or
the latest) archive into a safe restore folder for the user to inspect — it
never overwrites live files without the user copying them back deliberately.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import stat
import tempfile
import time
import uuid
import zipfile
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .atomic_io import _replace
from .constants import BASE_DIR, CONFIG_DIR
from .data import ToolResult
from .db import apply_pragmas
from .utils import first_line


class BackupManager:
    RETENTION = 8

    def __init__(self, bus: OrionBus, telemetry: Any | None = None,
                 destination: Optional[Path] = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._observer_failures: set[str] = set()
        self.destination = destination or self._default_destination()
        try:
            self.destination.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.destination = BASE_DIR / "backups"
            self.destination.mkdir(parents=True, exist_ok=True)

    def _default_destination(self) -> Path:
        one_drive = os.getenv("OneDrive") or os.getenv("OneDriveConsumer")
        if one_drive and Path(one_drive).is_dir():
            return Path(one_drive) / "ORION_Backups"
        return BASE_DIR / "backups"

    # ── backup ────────────────────────────────────────────────────────────────

    async def backup(self, note: str = "") -> ToolResult:
        return await asyncio.to_thread(self._backup_sync, note)

    def _backup_sync(self, note: str) -> ToolResult:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive = self.destination / f"orion_backup_{stamp}_{uuid.uuid4().hex[:8]}.zip"
        added = 0
        try:
            if self.destination.resolve().is_relative_to(CONFIG_DIR.resolve()):
                raise ValueError("The backup destination must be outside the configuration directory")
            # Publish only the closed archive. A failed read must not silently
            # produce a successful but incomplete backup or prune older ones.
            with tempfile.TemporaryDirectory(prefix=".backup-", dir=self.destination) as staging:
                temporary = Path(staging) / "archive.zip"
                files = self._config_files()
                databases = {p for p in files if self._is_sqlite(p)}
                sidecars = {Path(str(p) + suffix) for p in databases
                            for suffix in ("-wal", "-shm", "-journal")}
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zf:
                    for path in files:
                        if path in sidecars:
                            continue
                        arcname = "config/" + path.relative_to(CONFIG_DIR).as_posix()
                        if path in databases:
                            snapshot = Path(staging) / f"database-{added}.sqlite3"
                            self._snapshot_sqlite(path, snapshot)
                            zf.write(snapshot, arcname)
                        else:
                            zf.write(path, arcname)
                        added += 1
                    if note.strip():
                        zf.writestr("BACKUP_NOTE.txt", note.strip())
                size_mb = temporary.stat().st_size / (1024 * 1024)
                _replace(str(temporary), archive)
        except Exception as exc:
            return ToolResult(f"Backup failed: {first_line(exc)}", ok=False)
        retention_error = ""
        try:
            self._prune()
        except OSError as exc:
            # The closed archive has already been published. A maintenance
            # failure cannot turn it into an apparent failed backup.
            retention_error = first_line(exc)
            self._observer_failed("retention")
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("backup.created")
                self.telemetry.metrics.gauge("backup.size_mb", size_mb)
            except Exception:
                self._observer_failed("metrics")
        try:
            self.bus.dashboard_event.emit("backup", {"archive": str(archive), "size_mb": round(size_mb, 2)})
        except Exception:
            self._observer_failed("dashboard")
        note = " Retention cleanup was unavailable; existing archives were kept." if retention_error else ""
        return ToolResult(
            f"Backup complete — {added} file(s), {size_mb:.1f} MB → "
            f"'{archive.name}' in {self.destination}.{note}",
            evidence={"archive": str(archive), "files": added, "size_mb": size_mb,
                      "retention_error": retention_error},
        )

    def _observer_failed(self, hook: str) -> None:
        if hook not in self._observer_failures:
            self._observer_failures.add(hook)
            logging.getLogger(__name__).warning(
                "Backup completed, but %s could not finish", hook, exc_info=True)

    @staticmethod
    def _config_files() -> list[Path]:
        """Enumerate completely or fail; pathlib glob suppresses scan errors."""
        if not stat.S_ISDIR(CONFIG_DIR.stat().st_mode):
            raise ValueError("The configuration path is not a directory")

        def failed_scan(error: OSError) -> None:
            raise error

        files = []
        for parent, _directories, names in os.walk(CONFIG_DIR, onerror=failed_scan):
            for name in names:
                path = Path(parent) / name
                if stat.S_ISREG(path.stat().st_mode) and not (
                        "knowledge_corpus" in path.parts and path.suffix == ".md"):
                    files.append(path)
        return sorted(files)

    @staticmethod
    def _is_sqlite(path: Path) -> bool:
        with path.open("rb") as stream:
            return stream.read(16) == b"SQLite format 3\0"

    @staticmethod
    def _snapshot_sqlite(source: Path, destination: Path) -> None:
        deadline = time.monotonic() + 30.0
        def progress(_status: int, _remaining: int, _total: int) -> None:
            if time.monotonic() >= deadline:
                raise TimeoutError("SQLite snapshot exceeded its 30-second budget")
        with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True,
                                     timeout=5)) as reader:
            # The read-only source cannot change journal mode; the shared
            # helper tolerates that refusal and still applies cache/timeouts.
            apply_pragmas(reader)
            with closing(sqlite3.connect(destination)) as writer:
                apply_pragmas(writer)
                writer.execute("PRAGMA synchronous=FULL")
                reader.backup(writer, pages=256, progress=progress, sleep=0.05)

    def _prune(self) -> None:
        archives = self._archives()
        for old in archives[:-self.RETENTION]:
            try:
                old.unlink()
            except OSError:
                continue

    # ── listing + restore ─────────────────────────────────────────────────────

    def _archives(self) -> list[Path]:
        """Read the location completely; glob can hide scan failures as empty."""
        with os.scandir(self.destination) as entries:
            return sorted(Path(entry.path) for entry in entries
                          if entry.name.startswith("orion_backup_") and entry.name.endswith(".zip")
                          and stat.S_ISREG(entry.stat().st_mode))

    def list_backups(self) -> ToolResult:
        try:
            archives = self._archives()[::-1]
        except OSError as exc:
            return ToolResult(f"Backup location unavailable: {first_line(exc)}", ok=False)
        if not archives:
            return ToolResult(f"No backups yet. Destination: {self.destination}.")
        lines = [f"Backups in {self.destination}:"]
        for a in archives[:12]:
            try:
                info = a.stat()
                size = info.st_size / (1024 * 1024)
                when = datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError as exc:
                return ToolResult(f"Could not read backup '{a.name}': {first_line(exc)}", ok=False)
            lines.append(f"- {a.name}  ({size:.1f} MB, {when})")
        return ToolResult("\n".join(lines))

    async def restore(self, archive_name: str = "") -> ToolResult:
        return await asyncio.to_thread(self._restore_sync, archive_name)

    def _restore_sync(self, archive_name: str) -> ToolResult:
        try:
            archives = self._archives()[::-1]
        except OSError as exc:
            return ToolResult(f"Backup location unavailable: {first_line(exc)}", ok=False)
        if not archives:
            return ToolResult("There are no backups to restore.", ok=False)
        if archive_name:
            matches = [a for a in archives if a.name == archive_name]
            if not matches:
                matches = [a for a in archives if archive_name in a.name]
            if len(matches) != 1:
                return ToolResult("No unique backup matches that name; choose one from the backup list.", ok=False)
            archive = matches[0]
        else:
            archive = archives[0]
        restore_root = BASE_DIR / "restore"
        restore_dir = restore_root / f"{archive.stem}_{uuid.uuid4().hex[:8]}"
        try:
            restore_root.mkdir(parents=True, exist_ok=True)
            # A fresh directory preserves edits to earlier reviewed restores
            # and avoids traversing leftover entries during extraction.
            with tempfile.TemporaryDirectory(prefix=".unpack-", dir=restore_root) as staging:
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(staging)
                _replace(staging, restore_dir)
        except Exception as exc:
            return ToolResult(f"Restore failed: {first_line(exc)}", ok=False)
        try:
            where = restore_dir.relative_to(BASE_DIR)
        except ValueError:          # restored outside ORION's own folder: say it in full
            where = restore_dir
        return ToolResult(
            f"Unpacked '{archive.name}' into {where}. "
            "I've left it there for you to review and copy back deliberately — I won't "
            "overwrite live settings automatically."
        )
