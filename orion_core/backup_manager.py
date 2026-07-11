"""
Local backup & synchronisation manager (improvement #30).

``BackupManager`` snapshots ORION's important state — the config directory
(provider settings, integrations, pipeline, knowledge packs), the SQLite memory
database, and the knowledge corpus manifest — into a single timestamped ``.zip``
archive.

Because this machine's project already lives under OneDrive, the default backup
destination is a ``ORION_Backups`` folder in the user's OneDrive root, so the
archive is picked up by cloud sync automatically (that is the "cloud sync" this
environment can honestly provide).  A different destination can be supplied.

Backups are pruned to a retention count, and ``restore`` unpacks a chosen (or
the latest) archive into a safe restore folder for the user to inspect — it
never overwrites live files without the user copying them back deliberately.
"""

from __future__ import annotations

import asyncio
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import BASE_DIR, CONFIG_DIR
from .data import ToolResult
from .utils import first_line


class BackupManager:
    RETENTION = 8

    def __init__(self, bus: OrionBus, telemetry: Any | None = None,
                 destination: Optional[Path] = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = self.destination / f"orion_backup_{stamp}.zip"
        added = 0
        try:
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                # config/ (settings, integrations, pipeline, packs, and the memory
                # DB + its -wal/-shm siblings, which live under config/) — skip
                # only the regenerable 50 MB corpus shards.
                seen: set[str] = set()
                if CONFIG_DIR.is_dir():
                    for path in CONFIG_DIR.rglob("*"):
                        if not path.is_file():
                            continue
                        if "knowledge_corpus" in path.parts and path.suffix == ".md":
                            continue  # the 50 MB corpus is regenerable; don't bloat backups
                        arcname = str(path.relative_to(BASE_DIR))
                        if arcname in seen:
                            continue
                        try:
                            zf.write(path, arcname)
                            seen.add(arcname)
                            added += 1
                        except OSError:
                            continue
                if note.strip():
                    zf.writestr("BACKUP_NOTE.txt", note.strip())
        except Exception as exc:
            return ToolResult(f"Backup failed: {first_line(exc)}", ok=False)
        size_mb = archive.stat().st_size / (1024 * 1024)
        self._prune()
        if self.telemetry is not None:
            self.telemetry.metrics.incr("backup.created")
            self.telemetry.metrics.gauge("backup.size_mb", size_mb)
        self.bus.dashboard_event.emit("backup", {"archive": str(archive), "size_mb": round(size_mb, 2)})
        where = "OneDrive (will cloud-sync)" if "OneDrive" in str(self.destination) else str(self.destination)
        return ToolResult(
            f"Backup complete, sir — {added} file(s), {size_mb:.1f} MB → "
            f"'{archive.name}' in {where}."
        )

    def _prune(self) -> None:
        archives = sorted(self.destination.glob("orion_backup_*.zip"))
        for old in archives[:-self.RETENTION]:
            try:
                old.unlink()
            except OSError:
                continue

    # ── listing + restore ─────────────────────────────────────────────────────

    def list_backups(self) -> ToolResult:
        archives = sorted(self.destination.glob("orion_backup_*.zip"), reverse=True)
        if not archives:
            return ToolResult(f"No backups yet, sir. Destination: {self.destination}.")
        lines = [f"Backups in {self.destination}:"]
        for a in archives[:12]:
            size = a.stat().st_size / (1024 * 1024)
            when = datetime.fromtimestamp(a.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- {a.name}  ({size:.1f} MB, {when})")
        return ToolResult("\n".join(lines))

    async def restore(self, archive_name: str = "") -> ToolResult:
        return await asyncio.to_thread(self._restore_sync, archive_name)

    def _restore_sync(self, archive_name: str) -> ToolResult:
        archives = sorted(self.destination.glob("orion_backup_*.zip"), reverse=True)
        if not archives:
            return ToolResult("There are no backups to restore, sir.", ok=False)
        archive = next((a for a in archives if archive_name and archive_name in a.name), archives[0])
        restore_dir = BASE_DIR / "restore" / archive.stem
        restore_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(restore_dir)
        except Exception as exc:
            return ToolResult(f"Restore failed: {first_line(exc)}", ok=False)
        return ToolResult(
            f"Unpacked '{archive.name}' into {restore_dir.relative_to(BASE_DIR)}, sir. "
            "I've left it there for you to review and copy back deliberately — I won't "
            "overwrite live settings automatically."
        )
