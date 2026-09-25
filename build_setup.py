"""
Build ORION-Setup.exe — the self-installing package.

Run:  python build_setup.py

Produces two double-clickable executables in the project root:

    ORION-Setup.exe      installs ORION onto this PC — Desktop + Start-menu
                         shortcuts, an Add/Remove Programs entry (so it shows in
                         Windows "Installed apps" and uninstalls cleanly), and
                         optionally starts with Windows.
    ORION-Uninstall.exe  removes all of the above (never your project files).

Both are thin, windowless launchers over orion_core/installer.py — the same
deliberate choice as ORION.exe (build_exe.py): the app runs in place under its
own interpreter, which is what actually works, so the installer REGISTERS it
rather than copying a fragile 500 MB freeze into Program Files.

The build also produces ORION.exe first (via build_exe.py) if it is missing, so
the shortcuts the installer creates point at a real launcher.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SETUP_SRC = BASE / "build" / "_setup_src"
DIST = BASE / "build" / "_setup_dist"

# The two tiny entry points. Each just calls into the tested installer module.
_SETUP_MAIN = """\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orion_core import installer
raise SystemExit(installer.main([]))
"""
_UNINSTALL_MAIN = """\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orion_core import installer
raise SystemExit(installer.main(["--uninstall"]))
"""


def _ensure_pyinstaller() -> bool:
    try:
        import PyInstaller  # noqa: F401
        return True
    except Exception:
        print("PyInstaller not found — installing it…")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("Could not install PyInstaller:\n" + r.stderr[-800:])
            return False
        return True


def _ensure_launcher() -> None:
    if (BASE / "ORION.exe").exists():
        return
    print("ORION.exe missing — building the launcher first…")
    try:
        import build_exe
        build_exe.build()
    except Exception as exc:
        print(f"(could not pre-build ORION.exe: {exc} — shortcuts may not work "
              "until you run build_exe.py)")


def _icon_arg() -> list[str]:
    icon = BASE / "assets" / "orion.ico"
    if not icon.exists():
        try:
            from orion_core import desktop_app
            desktop_app.ensure_icon()
        except Exception:
            pass
    return ["--icon", str(icon)] if icon.exists() else []


def _build_one(name: str, source: str) -> int:
    SETUP_SRC.mkdir(parents=True, exist_ok=True)
    src = SETUP_SRC / f"{name}.py"
    src.write_text(source, encoding="utf-8")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--windowed", "--noconfirm", "--clean",
        "--name", name,
        "--distpath", str(DIST),
        "--workpath", str(BASE / "build" / "_setup_work"),
        "--specpath", str(BASE / "build" / "_setup_spec"),
        # the installer imports orion_core.* at runtime — bundle it
        "--paths", str(BASE),
        "--hidden-import", "orion_core.installer",
        "--hidden-import", "orion_core.desktop_app",
        "--hidden-import", "orion_core.autostart",
        "--hidden-import", "orion_core.constants",
        *_icon_arg(),
        str(src),
    ]
    print(f"Building {name}.exe …")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"PyInstaller failed for {name} (exit {result.returncode}).")
        return result.returncode
    built = DIST / f"{name}.exe"
    if built.exists():
        target = BASE / f"{name}.exe"
        try:
            import shutil
            shutil.copy2(built, target)
            print(f"  → {target}")
        except OSError as exc:
            print(f"  (built but could not copy to project root: {exc})")
    return 0


def build() -> int:
    if not _ensure_pyinstaller():
        return 1
    _ensure_launcher()
    rc = _build_one("ORION-Setup", _SETUP_MAIN)
    if rc != 0:
        return rc
    rc = _build_one("ORION-Uninstall", _UNINSTALL_MAIN)
    if rc != 0:
        return rc
    print("\nDone. Double-click ORION-Setup.exe to install ORION onto this PC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
