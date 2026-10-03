"""
Local backup & synchronisation manager (improvement #30).

``BackupManager`` snapshots ORION's important state — the config directory
(provider settings, integrations, pipeline, knowledge packs) and every SQLite
store in it — into a single timestamped ``.zip`` archive.

Because this machine's project already lives under OneDrive, the default backup
destination is a ``ORION_Backups`` folder in the user's OneDrive root, so the
archive is picked up by cloud sync automatically (that is the "cloud sync" this
environment can honestly provide).  A different destination can be supplied.

Three properties the archive keeps (audit 2026-10-03):

* **It works wherever the config lives.** Entries are named ``config/…``
  relative to ``CONFIG_DIR``, not to the project. ``tools/move_config.py`` and
  ``ORION_CONFIG_DIR`` move the config out of the synced project folder — the
  recommended setup — and every backup then failed with "is not in the subpath".
* **Databases are consistent.** Each SQLite store is copied with SQLite's own
  backup API (a read lock, page by page), never as raw ``.db``/``-wal``/``-shm``
  files that a running ORION is writing to — the same rule
  ``deploy/sync_state.sh`` follows with ``.backup``.
* **Credentials stay home.** API keys, the uplink's token-signing secret and TLS
  key, bot/Twilio tokens, MCP credentials, the plugin vault and logged-in
  browser sessions are left out: the default destination is a cloud-synced
  folder, which is exactly where ``constants._resolve_key_store`` says keys must
  not go. ``ORION_BACKUP_SECRETS=1`` includes them for a destination you trust.

Backups are pruned to a retention count, and ``restore`` unpacks a chosen (or
the latest) archive into a safe restore folder for the user to inspect — it
never overwrites live files without the user copying them back deliberately.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import BASE_DIR, CONFIG_DIR
from .data import ToolResult
from .utils import first_line


#: Files that hold live credentials, by name.
SECRET_NAMES = frozenset({
    "api_keys.json",          # provider keys
    "remote_secret.key",      # signs every uplink access token
    "telephony.json",         # Twilio credentials
    "messaging.json",         # Discord/Telegram bot tokens
    "mcp_servers.json",       # MCP server env: GitHub/Notion/Postgres tokens
    "plugin_vault.json",      # sealed plugin secrets
    "plugin_vault.key",       # the key that unseals them off Windows
})
#: Credential-bearing suffixes (private keys, keystores).
SECRET_SUFFIXES = frozenset({".key", ".pem", ".pfx", ".p12", ".keystore"})
#: Directories that are a logged-in session (cookies, OAuth tokens).
SESSION_DIRS = frozenset({"browser_profile", "social_profile", "mcp_google_drive"})
#: Directories that are regenerable caches or downloads, not state.
CACHE_DIRS = frozenset({"tts_cache", "ingest_cache", "models", "__pycache__"})
#: SQLite sidecars; the backup API folds their contents into the snapshot.
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")
SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})


def _include_secrets() -> bool:
    return os.getenv("ORION_BACKUP_SECRETS", "").strip().lower() in {"1", "true", "yes", "on"}


def classify(relative: Path) -> str:
    """What a config file is, for the archive: ``keep``, ``secret`` or ``skip``.

    *relative* is the path inside the config directory.
    """
    parts = set(relative.parts[:-1])
    name = relative.name
    if parts & CACHE_DIRS:
        return "skip"
    if name.endswith(SQLITE_SIDECARS):
        return "skip"
    if "knowledge_corpus" in parts and relative.suffix == ".md":
        return "skip"            # the 50 MB corpus is regenerable
    if parts & SESSION_DIRS:
        return "secret"
    if name in SECRET_NAMES or relative.suffix.lower() in SECRET_SUFFIXES:
        return "secret"
    return "keep"


def _snapshot_sqlite(source: Path, target: Path) -> None:
    """A consistent copy of a live SQLite database at *target*.

    An ordinary connection, not a ``mode=ro`` URI: a URI breaks on Windows
    drive letters and on '?', '#' or '%' in a path, and a read-only open of a
    WAL database fails when its -shm is absent. The backup API itself only
    ever takes read locks on the source."""
    src = sqlite3.connect(str(source), timeout=10)
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


class BackupManager:
    RETENTION = 8

    def __init__(self, bus: OrionBus, telemetry: Any | None = None,
                 destination: Optional[Path] = None,
                 config_dir: Optional[Path] = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
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
        # Written under a temporary name and renamed only when complete, so a
        # failure never leaves a truncated archive that list/restore offer.
        partial = archive.with_name(archive.name + ".part")
        include_secrets = _include_secrets()
        added = 0
        unreadable = 0
        withheld: list[str] = []
        try:
            with tempfile.TemporaryDirectory(prefix="orion-backup-") as scratch, \
                    zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as zf:
                if self.config_dir.is_dir():
                    for path in sorted(self.config_dir.rglob("*")):
                        if not path.is_file():
                            continue
                        relative = path.relative_to(self.config_dir)
                        kind = classify(relative)
                        if kind == "skip":
                            continue
                        if kind == "secret" and not include_secrets:
                            withheld.append(relative.as_posix())
                            continue
                        arcname = "config/" + relative.as_posix()
                        try:
                            if path.suffix.lower() in SQLITE_SUFFIXES:
                                copy = Path(scratch) / f"{added}.db"
                                try:
                                    _snapshot_sqlite(path, copy)
                                    zf.write(copy, arcname)
                                except sqlite3.DatabaseError:
                                    zf.write(path, arcname)   # not a database after all
                            else:
                                zf.write(path, arcname)
                            added += 1
                        except OSError:
                            unreadable += 1     # locked or vanished: skip it, say so
                            continue
                if note.strip():
                    zf.writestr("BACKUP_NOTE.txt", note.strip())
                if withheld:
                    zf.writestr("WITHHELD_SECRETS.txt", (
                        "These files hold credentials or logged-in sessions and were\n"
                        "left out of this archive. Re-enter keys after a restore, or\n"
                        "set ORION_BACKUP_SECRETS=1 to include them next time.\n\n"
                        + "\n".join(withheld) + "\n"))
            if unreadable and not added:
                raise OSError(f"none of the {unreadable} file(s) could be read")
            os.replace(partial, archive)
        except Exception as exc:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            return ToolResult(f"Backup failed: {first_line(exc)}", ok=False)
        size_mb = archive.stat().st_size / (1024 * 1024)
        self._prune()
        if self.telemetry is not None:
            self.telemetry.metrics.incr("backup.created")
            self.telemetry.metrics.gauge("backup.size_mb", size_mb)
        self.bus.dashboard_event.emit("backup", {"archive": str(archive), "size_mb": round(size_mb, 2)})
        where = "OneDrive (will cloud-sync)" if "OneDrive" in str(self.destination) else str(self.destination)
        kept_back = (f" {len(withheld)} credential file(s) were left out; re-enter keys "
                     "after a restore." if withheld else "")
        if unreadable:
            kept_back += f" {unreadable} file(s) could not be read and are missing."
        return ToolResult(
            f"Backup complete — {added} file(s), {size_mb:.1f} MB → "
            f"'{archive.name}' in {where}.{kept_back}"
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
            return ToolResult(f"No backups yet. Destination: {self.destination}.")
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
            return ToolResult("There are no backups to restore.", ok=False)
        archive = next((a for a in archives if archive_name and archive_name in a.name), archives[0])
        restore_dir = BASE_DIR / "restore" / archive.stem
        restore_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(restore_dir)
        except Exception as exc:
            return ToolResult(f"Restore failed: {first_line(exc)}", ok=False)
        return ToolResult(
            f"Unpacked '{archive.name}' into {restore_dir.relative_to(BASE_DIR)}. "
            "I've left it there for you to review and copy back deliberately — I won't "
            "overwrite live settings automatically."
        )
