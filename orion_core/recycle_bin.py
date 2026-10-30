"""
The Windows Recycle Bin, without a third-party package.

Deleting a user's file used to be final unless `send2trash` happened to be
installed - and it was never in the build environment, so the packaged app's
"move to the recycle bin" was always a permanent delete. `file_controller`
deleted outright and kept a text copy for undo, which corrupted any binary
file it "restored".

`send_to_recycle_bin` uses the shell's own FO_DELETE + FOF_ALLOWUNDO and then
CHECKS the bin: a drive without a Recycle Bin (network shares, some USB
sticks) makes the shell delete permanently without saying so, and that must
not be reported as recoverable. `restore_from_recycle_bin` reads the bin's
`$I` records (original path) and moves the matching `$R` item back - no COM,
no localised verb names.

Checking afterwards only makes the report honest; the item is already gone.
With FOF_NOCONFIRMATION the shell destroys instead of recycling, silently,
whenever the bin cannot take the item - measured 6 Oct 2026: a folder holding
ONE path of 260+ characters (MAX_PATH; 259 recycled, 260 destroyed the whole
folder) - a Gradle build output under this project's long OneDrive path was
lost that way. `refusal` names those cases BEFORE anything is touched: a long
path inside, an item bigger than the bin, a bin set to delete immediately,
or a drive with no bin at all.
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
from pathlib import Path

_FO_DELETE = 0x0003
_FOF_SILENT = 0x0004
_FOF_NOCONFIRMATION = 0x0010
_FOF_ALLOWUNDO = 0x0040
_FOF_NOERRORUI = 0x0400

MAX_PATH = 260                 # any path this long inside an item: destroyed, not recycled
_DRIVE_FIXED = 3
_BIN_SETTINGS = r"Software\Microsoft\Windows\CurrentVersion\Explorer\BitBucket\Volume"


def available() -> bool:
    return sys.platform == "win32"


def _bin_settings(target: Path) -> tuple[int | None, bool]:
    """(the bin's capacity in bytes or None if unknown, delete-immediately)
    for the volume holding *target* - the shell's own per-volume settings."""
    import ctypes
    import winreg
    name = ctypes.create_unicode_buffer(64)
    if not ctypes.windll.kernel32.GetVolumeNameForVolumeMountPointW(target.anchor, name, 64):
        return None, False
    volume = name.value
    guid = volume[volume.find("{"):volume.find("}") + 1]
    if not guid:
        return None, False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"{_BIN_SETTINGS}\{guid}") as key:
            def value(field):
                try:
                    return int(winreg.QueryValueEx(key, field)[0])
                except (OSError, ValueError, TypeError):
                    return None
            capacity, nuke = value("MaxCapacity"), value("NukeOnDelete")
    except OSError:
        return None, False                    # never configured: the default bin applies
    return (capacity * 1024 * 1024 if capacity else None), bool(nuke)


def _too_long_or_large(target: Path, capacity: int | None) -> str | None:
    """Walk the item once: the first path the bin cannot hold, or a total
    size over its capacity."""
    if len(str(target)) >= MAX_PATH:
        return f"its path is {len(str(target))} characters (Windows' limit is {MAX_PATH - 1})"
    if not target.is_dir():
        size = target.stat().st_size
        return (f"it is larger than the Recycle Bin ({size // 2**20} MB)"
                if capacity and size > capacity else None)
    unreadable: list[str] = []
    total = 0
    for folder, dirs, files in os.walk(target, onerror=lambda error: unreadable.append(str(error))):
        for name in dirs + files:
            inside = os.path.join(folder, name)
            if len(inside) >= MAX_PATH:
                return (f"it holds a path of {len(inside)} characters "
                        f"(Windows' limit is {MAX_PATH - 1}): {inside}")
        for name in files:
            try:
                total += os.lstat(os.path.join(folder, name)).st_size
            except OSError as error:
                unreadable.append(str(error))
        if capacity and total > capacity:
            return f"it is larger than the Recycle Bin (over {capacity // 2**20} MB)"
        if unreadable:
            return f"part of it could not be read ({unreadable[0]})"
    return None


def refusal(path: str | os.PathLike) -> str | None:
    """Why the shell would DESTROY *path* rather than recycle it, or None
    when the Recycle Bin can take it. Touches nothing."""
    if not available():
        return "there is no Recycle Bin on this system"
    target = Path(path).resolve()
    if not target.exists():
        return "it does not exist"
    import ctypes
    if ctypes.windll.kernel32.GetDriveTypeW(target.anchor) != _DRIVE_FIXED:
        return f"{target.anchor} is a network or removable drive without a Recycle Bin"
    capacity, nuke = _bin_settings(target)
    if nuke:
        return f"the Recycle Bin on {target.anchor} is set to delete files immediately"
    return _too_long_or_large(target, capacity)


def send_to_recycle_bin(path: str | os.PathLike) -> bool:
    """Move a file or folder to the Recycle Bin. True only when it is gone
    from its place AND found in the bin; False without touching it when the
    bin could not take it (`refusal` says why)."""
    if not available():
        return False
    target = Path(path).resolve()
    if not target.exists() or refusal(target) is not None:
        return False
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    operation = SHFILEOPSTRUCTW()
    operation.wFunc = _FO_DELETE
    operation.pFrom = str(target) + "\0"          # double-NUL terminated list
    operation.fFlags = _FOF_ALLOWUNDO | _FOF_NOCONFIRMATION | _FOF_SILENT | _FOF_NOERRORUI
    try:
        failed = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    except Exception:
        return False
    if failed or operation.fAnyOperationsAborted or target.exists():
        return False
    return find_in_recycle_bin(target) is not None


def _original_path(record: Path) -> str | None:
    """The original full path stored in a `$I` record (Vista+ v1, Win10 v2)."""
    try:
        data = record.read_bytes()
    except OSError:
        return None
    if len(data) < 24:
        return None
    version = struct.unpack_from("<q", data, 0)[0]
    if version == 2 and len(data) >= 28:
        length = struct.unpack_from("<i", data, 24)[0]
        raw = data[28:28 + 2 * max(0, length)]
    elif version == 1:
        raw = data[24:24 + 520]
    else:
        return None
    return raw.decode("utf-16-le", errors="ignore").split("\0", 1)[0] or None


def _bin_folders(drive: str) -> list[Path]:
    root = Path(drive + "\\$Recycle.Bin")
    try:
        return [entry for entry in root.iterdir() if entry.is_dir()]
    except OSError:
        return []


def find_in_recycle_bin(path: str | os.PathLike) -> tuple[Path, Path] | None:
    """(the `$R` item, its `$I` record) for the most recently deleted item
    that came from *path*, or None."""
    if not available():
        return None
    wanted = os.path.normcase(str(Path(path).resolve()))
    drive = os.path.splitdrive(wanted)[0]
    if not drive:
        return None
    newest: tuple[float, Path, Path] | None = None
    for folder in _bin_folders(drive):
        try:
            records = list(folder.glob("$I*"))
        except OSError:
            continue
        for record in records:
            original = _original_path(record)
            if original is None or os.path.normcase(original) != wanted:
                continue
            item = record.with_name("$R" + record.name[2:])
            if not item.exists():
                continue
            try:
                stamp = record.stat().st_mtime
            except OSError:
                continue
            if newest is None or stamp > newest[0]:
                newest = (stamp, item, record)
    return None if newest is None else (newest[1], newest[2])


def restore_from_recycle_bin(path: str | os.PathLike) -> bool:
    """Put the most recently recycled copy of *path* back where it was.
    Refuses (False) when something now occupies that path."""
    target = Path(path).resolve()
    if target.exists():
        return False
    found = find_in_recycle_bin(target)
    if found is None:
        return False
    item, record = found
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item), str(target))
    except OSError:
        return False
    try:
        record.unlink()
    except OSError:
        pass
    return True
