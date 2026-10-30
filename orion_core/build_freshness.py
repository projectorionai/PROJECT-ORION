"""
Is this standalone ORION.exe older than the source it was built from?

The Desktop and Start-menu shortcuts launch ``dist/ORION/ORION.exe``. Nothing
rebuilt it when the source moved on, and nothing said so: on 3 Oct 2026 the
installed build dated from 25 Sep, so a week of fixes — the Live tool-call
cancellation among them — never reached the app being used, and the same
"Requested entity was not found" drop and retired-model walk came back in the
Live journal as if the fixes had failed.

build_standalone.py now writes ``build_info.json`` into the bundle: when the
build was made and the newest source file it contained. At start-up a frozen
build that still sits inside its project (``<project>/dist/ORION/ORION.exe``)
compares that with the project's ``orion_core`` and says plainly when source
has changed since. A build with no project beside it — a copy on another
machine — has nothing to compare with and stays silent.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .constants import BASE_DIR, RESOURCE_DIR

BUILD_INFO_NAME = "build_info.json"
#: Copying an export can nudge mtimes; a minute of slack keeps that quiet.
_SLACK_S = 60.0


def newest_source(root: Path) -> tuple[float, int]:
    """(newest mtime, file count) over ``root/orion_core/**/*.py``."""
    newest, count = 0.0, 0
    for path in (root / "orion_core").rglob("*.py"):
        try:
            newest = max(newest, path.stat().st_mtime)
            count += 1
        except OSError:
            continue
    return newest, count


def module_hashes(root: Path) -> dict[str, str]:
    """sha256 of every ``root/orion_core/**/*.py``, by path inside orion_core.

    Mark XXXX.4: the installed app holds 5 of its 539 modules as files (the rest
    are packed in PyInstaller's archive), so change_awareness, scanning files,
    reported "no source files changed" after nearly every update. The build
    records each module's hash here and the installed ORION compares those."""
    import hashlib
    package = root / "orion_core"
    hashes: dict[str, str] = {}
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            hashes[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return hashes


def build_info(root: Path) -> dict[str, Any]:
    """What build_standalone.py records about the source it is packaging."""
    newest, count = newest_source(root)
    version = ""
    try:
        from . import __version__ as version  # type: ignore[attr-defined]
    except Exception:
        pass
    return {"built_at": time.time(), "source_newest": newest,
            "source_files": count, "version": str(version or ""),
            "modules": module_hashes(root)}


def read_build_info(resource_dir: Path = RESOURCE_DIR) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(resource_dir) / BUILD_INFO_NAME).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def project_root(base_dir: Path = BASE_DIR) -> Path | None:
    """The source checkout a ``<project>/dist/ORION/ORION.exe`` was built in."""
    candidate = Path(base_dir).parent.parent
    if (candidate / "orion_core").is_dir() and (candidate / "orion.py").is_file():
        return candidate
    return None


def assess(info: dict[str, Any], root: Path) -> dict[str, Any] | None:
    """None when the build is current; otherwise how far the source has moved."""
    try:
        built_from = float(info.get("source_newest") or 0.0)
    except (TypeError, ValueError):
        return None
    if built_from <= 0:
        return None
    changed: list[str] = []
    newest = 0.0
    for path in (root / "orion_core").rglob("*.py"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > built_from + _SLACK_S:
            changed.append(path.relative_to(root).as_posix())
            newest = max(newest, mtime)
    if not changed:
        return None
    try:
        built_at = float(info.get("built_at") or built_from)
    except (TypeError, ValueError):
        built_at = built_from
    return {"changed": len(changed), "examples": sorted(changed)[:5],
            "built_at": built_at, "newest_change": newest}


def _day(stamp: float) -> str:
    return time.strftime("%d %b %Y %H:%M", time.localtime(stamp))


def describe(stale: dict[str, Any]) -> str:
    return (f"This ORION build is from {_day(stale['built_at'])}, but "
            f"{stale['changed']} source file(s) have changed since (newest "
            f"{_day(stale['newest_change'])}) — fixes made after the build are not "
            "in the app you are running. Rebuild with build_standalone.py from a "
            "clean source export to pick them up.")


def check(bus: Any = None) -> str | None:
    """Run at start-up on a worker thread. Returns the warning, if any.
    Never raises; a source run or a build without its project is silent."""
    try:
        if not getattr(sys, "frozen", False):
            return None
        info = read_build_info()
        root = project_root()
        if info is None or root is None:
            return None
        stale = assess(info, root)
        if stale is None:
            return None
        message = describe(stale)
        if bus is not None:
            try:
                bus.log.emit("BUILD: " + message + " Changed, e.g.: "
                             + ", ".join(stale["examples"]))
                bus.banner.emit("⚠ This ORION build is older than its source — "
                                "see the log", 12000)
            except Exception:
                pass
        return message
    except Exception:
        return None
