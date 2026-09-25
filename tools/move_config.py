"""
Move ORION's writable data out of a synced folder.

Why
---
ORION keeps everything he writes in ``config/`` — nineteen SQLite stores,
conversation episodes, the knowledge graph, his voice profile. On this machine
that lives inside OneDrive, and live SQLite inside a sync client is a genuinely
bad combination.

Every store runs in WAL mode, so each database is three files: ``.db``,
``.db-wal`` and ``.db-shm``. A sync client treats those as three unrelated
files and uploads, restores and conflict-resolves them independently. A ``.db``
recovered without its matching WAL is the database as it stood at the last
checkpoint, and everything since is gone — silently, because SQLite opened a
perfectly valid file. Measured on 2026-09-21: 8.3 MB in the checkout's logs and
a further 9.8 MB in the installed app's, none of it yet in any database.

Frequent checkpointing narrows that window. Moving the data out of the synced
folder closes it.

What this does
--------------
Copies (never moves) the directory to a location outside the sync root,
verifies the copy file by file, checks that every database still opens and
answers a query, and only then writes the pointer file that redirects ORION.
The original is left exactly where it was. Nothing is deleted, ever — if the
new location turns out to be wrong, delete the pointer and ORION goes back to
using the original.

Usage
-----
    python tools/move_config.py --plan                  # show what would happen
    python tools/move_config.py --app dist/ORION        # do it for the installed app
    python tools/move_config.py --app dist/ORION --revert

ORION must not be running: copying a database out from under a live writer is
how you manufacture the corruption this exists to prevent.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

CONFIG_POINTER_NAME = "config_location.txt"


def store_python() -> bool:
    r"""Whether this interpreter is the Microsoft Store build of Python.

    It matters here for one reason: a Store app's writes under
    ``AppData\Local`` are silently REDIRECTED into a per-package sandbox at
    ``AppData\Local\Packages\<package>\LocalCache\Local\``. Python then
    reads back its own redirected view and sees a perfect copy, while every
    other program — including ORION.exe, which is an ordinary binary — looks at
    the real path and finds nothing there.

    That is not a hypothetical. The first run of this tool reported a verified
    1.27 GB copy that no other process could see, because the verification ran
    inside the same sandbox that received it.
    """
    return "WindowsApps" in sys.base_prefix


def is_virtualised(destination: Path) -> bool:
    """Whether *destination* would be redirected out from under us."""
    if not store_python():
        return False
    local = (os.getenv("LOCALAPPDATA") or "").lower()
    return bool(local) and str(destination).lower().startswith(local)


def default_destination() -> Path:
    r"""Somewhere outside the sync folder that nothing will redirect.

    Not ``AppData\Local``, which is the obvious answer and the wrong one
    here: see store_python(). The user profile root is plain, visible, needs
    no administrator, and is subject to neither OneDrive nor Store
    virtualisation.
    """
    return Path.home() / "ORION" / "config"


def looks_synced(path: Path) -> str:
    """A reason *path* looks like it is inside a sync client's folder, or ""."""
    text = str(path).lower()
    for marker in ("onedrive", "dropbox", "google drive", "icloud", "nextcloud"):
        if marker in text:
            return marker
    return ""


def orion_is_running() -> bool:
    try:
        import psutil
    except Exception:
        return False
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info.get("name") or "").lower() == "orion.exe":
                return True
        except Exception:
            continue
    return False


# ── verification ─────────────────────────────────────────────────────────────

def compare_trees(source: Path, destination: Path) -> list[str]:
    """Every file under *source* that did not arrive intact. Empty is good."""
    problems: list[str] = []
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        mirrored = destination / relative
        if not mirrored.is_file():
            problems.append(f"missing: {relative}")
            continue
        try:
            if item.stat().st_size != mirrored.stat().st_size:
                problems.append(f"size differs: {relative}")
        except OSError as exc:
            problems.append(f"unreadable: {relative} ({exc})")
    return problems


def databases_open(directory: Path) -> list[str]:
    """Every .db that will not open and answer a query."""
    broken: list[str] = []
    for store in sorted(directory.glob("*.db")):
        try:
            connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
            try:
                connection.execute(
                    "SELECT count(*) FROM sqlite_master").fetchone()
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    broken.append(f"{store.name}: integrity_check failed")
            finally:
                connection.close()
        except Exception as exc:
            broken.append(f"{store.name}: {exc}")
    return broken


def checkpoint(directory: Path) -> int:
    """Fold every WAL back in before copying, so the copy is self-contained."""
    done = 0
    for store in sorted(directory.glob("*.db")):
        try:
            connection = sqlite3.connect(str(store), timeout=10.0)
            connection.isolation_level = None
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.close()
            done += 1
        except Exception:
            continue
    return done


# ── the move ─────────────────────────────────────────────────────────────────

def move(app_dir: Path, destination: Path, dry_run: bool = False) -> int:
    source = app_dir / "config"
    pointer = app_dir / CONFIG_POINTER_NAME

    if not source.is_dir():
        print(f"No config directory at {source}")
        return 1
    marker = looks_synced(source)
    print(f"  source      : {source}")
    print(f"  destination : {destination}")
    print(f"  sync folder : {marker or 'no — nothing to fix'}")
    total = sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    files = sum(1 for f in source.rglob("*") if f.is_file())
    print(f"  size        : {total / 1024 / 1024:,.0f} MB in {files:,} files")

    if dry_run:
        print("\n  --plan only; nothing was copied.")
        return 0
    if orion_is_running():
        print("\n  ORION is running. Copying a database out from under a live "
              "writer is how you manufacture the corruption this prevents.")
        return 1
    if is_virtualised(destination):
        print(f"\n  {destination} is under AppData\\Local and this is the "
              "Microsoft Store build of Python, whose writes there are "
              "redirected into a private sandbox. The copy would appear to "
              "succeed and ORION.exe would never see it. Choose a destination "
              "outside AppData (the default now is "
              f"{Path.home() / 'ORION' / 'config'}).")
        return 1
    if destination.exists() and any(destination.iterdir()):
        print(f"\n  {destination} already exists and is not empty. Refusing to "
              "overwrite — move it aside or choose another destination.")
        return 1

    print(f"\n  checkpointing {checkpoint(source)} store(s) so the copy is "
          "self-contained…")
    destination.mkdir(parents=True, exist_ok=True)
    print("  copying…")
    shutil.copytree(source, destination, dirs_exist_ok=True)

    print("  verifying every file…")
    problems = compare_trees(source, destination)
    if problems:
        print(f"  COPY INCOMPLETE ({len(problems)}): " + "; ".join(problems[:5]))
        print("  The pointer was NOT written. The original is untouched.")
        return 1

    print("  confirming the copy is visible outside this process…")
    unseen = _visible_to_others(destination)
    if unseen:
        print(f"  THE COPY IS NOT WHERE IT APPEARS TO BE: {unseen}")
        print("  The pointer was NOT written. The original is untouched.")
        return 1

    print("  opening every database…")
    broken = databases_open(destination)
    if broken:
        print(f"  DATABASES UNREADABLE ({len(broken)}): " + "; ".join(broken[:5]))
        print("  The pointer was NOT written. The original is untouched.")
        return 1

    pointer.write_text(str(destination), encoding="utf-8")
    print(f"\n  done. {pointer.name} now points at the new location.")
    print("  The original is still there, untouched, as a backup — delete it "
          "yourself once you are satisfied.")
    return 0


def _visible_to_others(destination: Path) -> str:
    """"" if another process can see *destination*, else why not.

    Asks ``cmd.exe`` — a program with no stake in this interpreter's
    redirection — to count the files. If Python can see a copy that cmd
    cannot, the copy is in a sandbox and the whole move is an illusion.
    """
    probe = destination / ".orion-visibility-probe"
    try:
        probe.write_text("probe", encoding="utf-8")
    except Exception as exc:
        return f"could not write a probe file ({exc})"
    try:
        result = subprocess.run(
            ["cmd", "/c", "if", "exist", str(probe), "echo", "SEEN"],
            capture_output=True, text=True, timeout=30)
        if "SEEN" not in (result.stdout or ""):
            return (f"another process cannot see {probe} — this interpreter's "
                    "writes are being redirected")
    except Exception as exc:
        return f"could not run the visibility probe ({exc})"
    finally:
        try:
            probe.unlink()
        except Exception:
            pass
    return ""


def revert(app_dir: Path) -> int:
    pointer = app_dir / CONFIG_POINTER_NAME
    if not pointer.is_file():
        print("  No pointer file; ORION is already using the original.")
        return 0
    target = pointer.read_text(encoding="utf-8").strip()
    pointer.unlink()
    print(f"  Pointer removed. ORION goes back to {app_dir / 'config'}.")
    print(f"  The moved copy is still at {target} — delete it yourself.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default=".",
                        help="the directory holding config/ (e.g. dist/ORION)")
    parser.add_argument("--to", default=None, help="destination directory")
    parser.add_argument("--plan", action="store_true",
                        help="show what would happen, copy nothing")
    parser.add_argument("--revert", action="store_true",
                        help="remove the pointer and use the original again")
    args = parser.parse_args(argv)

    app_dir = Path(args.app).resolve()
    if args.revert:
        return revert(app_dir)
    destination = Path(args.to).resolve() if args.to else default_destination()
    return move(app_dir, destination, dry_run=args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
