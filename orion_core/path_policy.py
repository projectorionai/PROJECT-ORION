"""
Where ORION may change files - one table, and one honest reason per refusal.

9 Oct 2026: "could you delete that folder" ("Subi Dropshipping", inside the
owner's OneDrive business folder). The owner confirmed, and it still failed:
the only rule was "inside ORION's own folder or nowhere"
(``dispatcher._ensure_write_safe``: "write path outside workspace"), so
nothing a person calls "my files" could ever be changed - and ORION then made
up a policy of his own to explain the refusal. This module names the places
people mean, keeps the places nobody should touch protected, and gives every
refusal its exact reason and what would allow it.

    zone       where                                      changing / deleting
    ---------  -----------------------------------------  ---------------------------------
    workspace  ORION's own output folders inside his      as before: immediate and undoable
               folder (research, exports, reports,        (a folder with contents: confirmed
               projects, conversations ...)               on screen first)
    user       Desktop, Documents, Downloads, Pictures,   NEW files and folders: immediate
               Music, Videos, the OneDrive roots, and     and undoable. Changing or deleting
               folders on the PC's other drives           what is already there: confirmed
                                                          on screen (token-bound), deletes
                                                          to the Recycle Bin only
    protected  Windows, Program Files, ProgramData,       never - the reason, and who can
               drive roots, other users' profiles,
               ORION's own code, settings and memory,
               the built app (dist)
    outside    anything else (AppData, the profile        never - the reason, and where the
               root, unknown places)                      file could go instead

Reading is not governed here: ORION may read whatever the signed-in user can.
Nothing in this module changes anything; it only answers "may I, and if not,
why not".
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Zone(str, Enum):
    WORKSPACE = "workspace"
    USER = "user"
    PROTECTED = "protected"
    OUTSIDE = "outside"


#: The places a person means by "my files", as the policy names them.
USER_AREAS_TEXT = ("your Desktop, Documents, Downloads, Pictures, Music and Videos "
                   "folders, your OneDrive, and folders on the PC's other drives")


@dataclass(frozen=True)
class Verdict:
    """Whether ORION may change *path*, and the plain reason when not."""

    zone: Zone
    path: Path
    #: Where the path is, as a noun phrase: "inside your OneDrive (OneDrive -
    #: Contoso)", "inside Windows' own system folder".
    area: str
    #: Why that area is closed (empty for an open one).
    reason: str = ""
    #: What would allow it - said every time, so a refusal is never a dead end.
    remedy: str = ""

    @property
    def writable(self) -> bool:
        return self.zone in (Zone.WORKSPACE, Zone.USER)

    def refusal(self, doing: str = "change") -> str:
        """The exact sentence a refusal reports, unchanged, to the user."""
        why = f", {self.reason}" if self.reason else ""
        return (f"I can't {doing} {self.path}: it is {self.area}{why}. "
                f"Nothing was changed. {self.remedy}").strip()


# ── the places ───────────────────────────────────────────────────────────────

def _norm(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def _within(path_n: str, root_n: str) -> bool:
    """True when normalised *path_n* is *root_n* or inside it."""
    if not root_n:
        return False
    if path_n == root_n:
        return True
    return path_n.startswith(root_n.rstrip("\\/") + os.sep)


def resolve(path: str | os.PathLike) -> Path:
    """The real location: links and junctions followed, so a shortcut folder
    inside Documents that points into Windows is judged as Windows."""
    raw = os.path.expandvars(os.path.expanduser(os.fspath(path)))
    if not raw or "\0" in raw:
        raise ValueError("empty or invalid path")
    # Do not fall back to an apparently safe spelling when a junction cannot
    # be resolved. Classification catches the error and fails closed.
    return Path(raw).resolve(strict=False)


#: (how it is said, app_resolver's spoken key, the profile sub-folder name)
_KNOWN = (("Desktop", "desktop folder", "Desktop"), ("Documents", "documents", "Documents"),
          ("Downloads", "downloads", "Downloads"), ("Pictures", "pictures folder", "Pictures"),
          ("Music", "music folder", "Music"), ("Videos", "videos folder", "Videos"))


def onedrive_roots() -> list[Path]:
    """Every OneDrive this user syncs (personal and work or school)."""
    found: list[Path] = []
    for variable in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        value = os.environ.get(variable, "").strip()
        if value:
            found.append(Path(value))
    try:
        for entry in Path.home().iterdir():
            if entry.name.lower().startswith("onedrive") and entry.is_dir():
                found.append(entry)
    except OSError:
        pass
    unique: dict[str, Path] = {}
    for root in found:
        unique.setdefault(_norm(root), root)
    return list(unique.values())


def known_folders() -> list[tuple[str, Path]]:
    """(name, path) of the user's own folders - the registry's (OneDrive
    redirects Documents, Desktop and Pictures) and the profile's own."""
    try:
        from .app_resolver import known_folder
    except Exception:          # pragma: no cover - app_resolver always imports
        known_folder = None
    home = Path.home()
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for label, spoken, plain in _KNOWN:
        candidates = []
        if known_folder is not None:
            try:
                candidates.append(known_folder(spoken))
            except Exception:
                pass
        candidates.append(home / plain)
        for candidate in candidates:
            if candidate is None:
                continue
            key = _norm(candidate)
            if key in seen or not Path(candidate).is_dir():
                continue
            seen.add(key)
            out.append((label, Path(candidate)))
    return out


def user_roots() -> list[tuple[str, Path]]:
    """(how to say it, root) for every user area in the table above, except
    the other drives (see data_drives)."""
    roots = [(f"your {label} folder", path) for label, path in known_folders()]
    for root in onedrive_roots():
        roots.append((f"your OneDrive ({root.name})", root))
    return roots


def data_drives() -> list[Path]:
    """Roots of this PC's other local drives - fixed or removable, never the
    Windows drive, never a network share."""
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        import string
        system = (os.environ.get("SystemDrive") or "C:")[:2].upper()
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        roots: list[Path] = []
        for index, letter in enumerate(string.ascii_uppercase):
            if not mask & (1 << index) or f"{letter}:" == system:
                continue
            root = f"{letter}:\\"
            if ctypes.windll.kernel32.GetDriveTypeW(root) in (2, 3):   # removable, fixed
                roots.append(Path(root))
        return roots
    except Exception:
        return []


#: Folders at the top of ANY drive that belong to Windows or to installed
#: programs, never to the user.
ROOT_SYSTEM_NAMES = frozenset({
    "windows", "program files", "program files (x86)", "programdata", "$recycle.bin",
    "system volume information", "recovery", "perflogs", "boot", "efi", "config.msi",
    "msocache", "$winreagent", "$windows.~bt", "$windows.~ws", "$sysreset", "windowsapps",
    "documents and settings", "onedrivetemp", "$getcurrent", "esd",
})


def _system_places() -> list[tuple[str, str]]:
    """(normalised path, how to say it) of Windows' own folders."""
    places: list[tuple[str, str]] = []
    for variable, said in (("SystemRoot", "inside Windows' own system folder"),
                           ("windir", "inside Windows' own system folder"),
                           ("ProgramFiles", "inside Program Files, where installed programs live"),
                           ("ProgramFiles(x86)", "inside Program Files (x86), where installed programs live"),
                           ("ProgramW6432", "inside Program Files, where installed programs live"),
                           ("ProgramData", "inside ProgramData, where installed programs keep their data"),
                           ("ALLUSERSPROFILE", "inside ProgramData, where installed programs keep their data")):
        value = os.environ.get(variable, "").strip()
        if value:
            places.append((_norm(value), said))
    return places


# ── ORION's own folder ───────────────────────────────────────────────────────

#: Top-level entries of ORION's folder that are his code, his settings, his
#: build or another tool's working copy - never "files" to change by request.
ORION_SOURCE_DIRS = frozenset({
    "orion_core", "tests", "tools", "launcher", "android", "assets", "docs",
    "requirements", "config", "dist", "build", "release", "releases", "deploy",
    "engines", "skills", ".git", ".github", ".venv", "venv", ".neural-debug-env",
    ".coordination", ".audit_tmp", ".test-runs", ".pytest_cache",
    ".ruff_cache", ".runtime-cache", "__pycache__", ".agents", ".codex", ".aws",
})
#: Files at the top of ORION's folder that are code, build or documentation.
ORION_SOURCE_SUFFIXES = frozenset({
    ".py", ".pyw", ".pyd", ".pyi", ".spec", ".toml", ".cfg", ".ini", ".bat", ".cmd",
    ".ps1", ".sh", ".exe", ".dll", ".lock", ".yml", ".yaml", ".md", ".prev",
})
ORION_SOURCE_NAMES = frozenset({
    "license", "notice", ".gitignore", ".dockerignore", ".gitattributes", ".env",
    ".env.example", "requirements.txt", "config_location.txt",
})

_ORION_REMEDY = ("That is ORION's own program, settings or memory: ask me to change "
                 "the setting itself, or edit the source yourself.")


def orion_roots() -> list[Path]:
    """ORION's own folder - and, for the built app, the checkout it was built
    from (…\\dist\\ORION -> …), which is his source too. Deepest first."""
    from . import constants
    base = Path(constants.BASE_DIR)
    roots = [base]
    try:
        checkout = base.parent.parent
        if base.parent.name.lower() == "dist" and (checkout / "orion_core").is_dir():
            roots.append(checkout)
    except Exception:
        pass
    return roots


def _orion_private() -> list[tuple[str, str]]:
    """(normalised path, how to say it) of ORION's settings, memory and code
    wherever they live - CONFIG_DIR can sit outside his folder."""
    from . import constants
    places = [(_norm(constants.CONFIG_DIR), "ORION's own settings and memory")]
    try:
        parent = Path(constants.CONFIG_DIR).parent
        if parent.name.lower() == "orion" and _norm(parent) != _norm(Path.home()):
            places.append((_norm(parent), "ORION's own data folder"))
    except Exception:
        pass
    places.append((_norm(constants.PACKAGE_DIR), "ORION's own program code"))
    places.append((_norm(constants.LAUNCHER_PATH), "ORION's own launcher"))
    resource = getattr(constants, "RESOURCE_DIR", None)
    if getattr(sys, "frozen", False):
        if resource:
            places.append((_norm(resource), "ORION's own program files"))
        places.append((_norm(sys.executable), "ORION's own program"))
    return places


def _inside_orion(path: Path, path_n: str, roots: list[Path]) -> Verdict | None:
    for root in roots:
        root_n = _norm(root)
        if not _within(path_n, root_n):
            continue
        if path_n == root_n:
            return Verdict(Zone.PROTECTED, path, "ORION's own folder",
                           "which holds his program and everything he keeps", _ORION_REMEDY)
        relative = Path(os.path.relpath(path_n, root_n))
        top = relative.parts[0].lower()
        if (top in ORION_SOURCE_DIRS or top.startswith("_internal")
                or top.startswith("orion.exe")):
            return Verdict(Zone.PROTECTED, path,
                           f"part of ORION's own program ({relative.parts[0]})",
                           "which a file request must never change", _ORION_REMEDY)
        if len(relative.parts) == 1 and not path.is_dir() and (
                Path(top).suffix in ORION_SOURCE_SUFFIXES or top in ORION_SOURCE_NAMES):
            return Verdict(Zone.PROTECTED, path,
                           f"one of ORION's own program files ({relative.parts[0]})",
                           "which a file request must never change", _ORION_REMEDY)
        return Verdict(Zone.WORKSPACE, path, "in ORION's own workspace")
    return None


# ── the judgement ────────────────────────────────────────────────────────────

_SYSTEM_REMEDY = ("Windows protects it and ORION never changes system folders; if it "
                  "really must change, do it yourself in File Explorer (Windows will "
                  "ask for administrator approval).")
_OUTSIDE_REMEDY = (f"I can change files in {USER_AREAS_TEXT}. Move it into one of "
                   "those, or change it yourself in File Explorer.")


def classify(path: str | os.PathLike) -> Verdict:
    """Which zone *path* is in - after following links - and why."""
    try:
        real = resolve(path)
        real_n = _norm(real)
    except Exception:
        shown = Path(os.fspath(path))
        return Verdict(Zone.PROTECTED, shown, "a path I could not resolve safely", "",
                       "Give me the full path of the item, as File Explorer shows it.")

    anchor = Path(real.anchor) if real.anchor else None
    if anchor is not None and real_n == _norm(anchor):
        return Verdict(Zone.PROTECTED, real, f"the whole of drive {real.drive or anchor}",
                       "and ORION never changes a drive itself", _SYSTEM_REMEDY)

    if os.name == "nt" and ":" in str(real)[len(real.drive):]:
        return Verdict(Zone.PROTECTED, real, "an NTFS alternate data stream", "",
                       "Give me the path of the ordinary file instead.")

    # ORION himself: settings, memory and code wherever they are kept.
    # A junction-selected config or checkout is still private at its real
    # destination. Compare both sides in the same resolved namespace; otherwise
    # its direct Documents/OneDrive spelling was accepted as an ordinary file.
    try:
        private = [(_norm(resolve(place_n)), said)
                   for place_n, said in _orion_private()]
        program_roots = [resolve(root) for root in orion_roots()]
    except (OSError, ValueError, RuntimeError):
        return Verdict(Zone.PROTECTED, real,
                       "a location whose protected ORION paths could not be resolved safely", "",
                       "Check ORION's configuration and program locations before changing files.")
    for place_n, said in private:
        if _within(real_n, place_n):
            return Verdict(Zone.PROTECTED, real, said,
                           "which a file request must never change", _ORION_REMEDY)
    inside = _inside_orion(real, real_n, program_roots)
    if inside is not None:
        return inside

    # A permitted parent must not provide a route around protected descendants.
    for place_n, said in private + [(_norm(root), "ORION's own folder")
                                   for root in program_roots]:
        if _within(place_n, real_n):
            return Verdict(Zone.PROTECTED, real, f"a folder containing {said}",
                           "which a file request must never change", _ORION_REMEDY)

    for place_n, said in _system_places():
        if _within(real_n, place_n):
            return Verdict(Zone.PROTECTED, real, said, "", _SYSTEM_REMEDY)
    if anchor is not None:
        parts = real.parts
        if len(parts) > 1 and (parts[1].lower() in ROOT_SYSTEM_NAMES
                               or parts[1].startswith("$")):
            return Verdict(Zone.PROTECTED, real,
                           f"inside {parts[0]}{parts[1]}, which belongs to Windows",
                           "", _SYSTEM_REMEDY)

    # Profiles: only this user's own, and within it only the user areas.
    home = Path.home()
    home_n = _norm(home)
    profiles_n = _norm(home.parent)
    if _within(real_n, profiles_n) and not _within(real_n, home_n):
        if real_n == profiles_n:
            return Verdict(Zone.PROTECTED, real, "the folder that holds every user's profile",
                           "", _SYSTEM_REMEDY)
        owner = Path(os.path.relpath(real_n, profiles_n)).parts[0]
        return Verdict(Zone.PROTECTED, real, f"inside another user's profile ({owner})",
                       "", f"Sign in to Windows as {owner} to change it.")

    for said, root in user_roots():
        root_n = _norm(resolve(root))
        if real_n == root_n:
            return Verdict(Zone.PROTECTED, real, f"the whole of {said}",
                           "which holds all the files in that area",
                           "Choose the particular file or subfolder you want changed.")
        if _within(real_n, root_n):
            return Verdict(Zone.USER, real, f"inside {said}")
    for drive in data_drives():
        drive_n = _norm(drive)
        if _within(real_n, drive_n) and real_n != drive_n:
            return Verdict(Zone.USER, real, f"on drive {drive.drive or drive}")

    if real_n == home_n:
        return Verdict(Zone.OUTSIDE, real, "your whole user profile",
                       "which also holds Windows' settings for you", _OUTSIDE_REMEDY)
    if _within(real_n, home_n):
        top = Path(os.path.relpath(real_n, home_n)).parts[0]
        if top.lower() == "appdata":
            return Verdict(Zone.OUTSIDE, real, "inside AppData",
                           "where programs keep their own settings", _OUTSIDE_REMEDY)
        return Verdict(Zone.OUTSIDE, real, f"in {top} in your profile",
                       "which is not one of the folders ORION may change", _OUTSIDE_REMEDY)
    return Verdict(Zone.OUTSIDE, real, "outside the folders ORION may change", "",
                   _OUTSIDE_REMEDY)


# ── what a delete would take ─────────────────────────────────────────────────

@dataclass(frozen=True)
class TreeSize:
    files: int
    folders: int
    bytes: int
    complete: bool = True

    def describe(self) -> str:
        """'37 files in 5 folders (412.3 MB)', 'at least ...' when cut short."""
        from .file_search import human_size
        lead = "" if self.complete else "at least "
        files = f"{lead}{self.files:,} file{'s' if self.files != 1 else ''}"
        folders = (f" in {self.folders:,} folder{'s' if self.folders != 1 else ''}"
                   if self.folders else "")
        return f"{files}{folders} ({lead}{human_size(self.bytes)})"


def measure(path: str | os.PathLike, budget_s: float = 8.0) -> TreeSize:
    """How much is under *path*: files, sub-folders and bytes, time-boxed so a
    huge tree reports "at least" rather than holding the request."""
    target = Path(path)
    if not target.is_dir():
        try:
            return TreeSize(1, 0, target.stat().st_size, True)
        except OSError:
            return TreeSize(1, 0, 0, False)
    deadline = time.monotonic() + max(0.0, budget_s)
    files = folders = size = 0
    errors: list[OSError] = []
    complete = True
    for base, dirnames, filenames in os.walk(target, onerror=errors.append):
        folders += len(dirnames)
        kept = []
        for name in dirnames:
            child = Path(base) / name
            try:
                if child.is_symlink() or getattr(child, "is_junction", lambda: False)():
                    complete = False
                    continue
            except OSError:
                complete = False
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            files += 1
            try:
                size += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                complete = False
        if time.monotonic() > deadline:
            return TreeSize(files, folders, size, False)
    return TreeSize(files, folders, size, complete and not errors)


__all__ = [
    "ORION_SOURCE_DIRS", "ROOT_SYSTEM_NAMES", "TreeSize", "USER_AREAS_TEXT", "Verdict",
    "Zone", "classify", "data_drives", "known_folders", "measure", "onedrive_roots",
    "orion_roots", "resolve", "user_roots",
]
