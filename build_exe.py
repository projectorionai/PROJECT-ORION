"""
Build ORION.exe from launcher/orion_app.py.

Run:  python build_exe.py

Produces ORION.exe in the project root — a real executable you can double-click,
pin to the taskbar, and add to startup. It launches the app with the project's
own Python, detached and windowless.

This builds the LAUNCHER, not a frozen bundle of the whole application; see
launcher/orion_app.py for why. PyInstaller is installed on demand if absent.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LAUNCHER = BASE / "launcher" / "orion_app.py"
ICON = BASE / "assets" / "orion.ico"


def _ensure_pyinstaller() -> bool:
    try:
        import PyInstaller  # noqa: F401
        return True
    except Exception:
        print("PyInstaller not found — installing it…")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyinstaller"],
            capture_output=True, text=True)
        if result.returncode != 0:
            print("Could not install PyInstaller:\n" + result.stderr[-800:])
            return False
        return True


def _ensure_icon() -> None:
    if ICON.exists():
        return
    try:
        from orion_core import desktop_app
        desktop_app.ensure_icon()
    except Exception:
        pass


def build() -> int:
    if not LAUNCHER.is_file():
        print(f"Launcher source missing: {LAUNCHER}")
        return 1
    if not _ensure_pyinstaller():
        return 1
    _ensure_icon()

    work = BASE / "build" / "_exe"
    dist = BASE / "build" / "_dist"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--windowed", "--name", "ORION",
        "--distpath", str(dist), "--workpath", str(work),
        "--specpath", str(work), "--noconfirm",
    ]
    if ICON.exists():
        cmd += ["--icon", str(ICON)]
    cmd.append(str(LAUNCHER))

    print("Building ORION.exe … (first build downloads/bundles the bootloader)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Build failed:\n" + result.stdout[-1500:] + "\n" + result.stderr[-1500:])
        return 1

    built = dist / "ORION.exe"
    if not built.is_file():
        print(f"Build reported success but {built} is missing.")
        return 1
    target = BASE / "ORION.exe"
    try:
        shutil.copy2(built, target)
    except OSError as exc:
        print(f"Built {built} but could not copy it to the project root: {exc}")
        return 1
    print(f"Done — {target} ({target.stat().st_size // 1024} KB).")
    print("Pin it to the taskbar, or run desktop_app.install_shortcuts() to add "
          "Desktop + Start-menu entries pointing at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
