"""
ORION.exe — the real executable.

  "He's also still not a running app in the background like wallpaper_engine.exe
   ... can we convert him into an exe file ... and then put him in startup"

This is the source of ORION.exe. It is a LAUNCHER, deliberately: freezing the
whole application (PyQt6, QtWebEngine, Torch, mediapipe, Vosk data, …) into one
standalone binary is a 500 MB+, fragile build that regularly breaks at runtime
in ways that are hard to see — the wrong trade for a machine that already has
the working interpreter installed. Instead this tiny exe finds the project's own
Python and starts ORION with it, detached and windowless.

What that buys, which a .lnk shortcut did not:
  * a genuine .exe Windows will pin to the taskbar directly (pinning a pythonw
    shortcut is the flaky path — "I can't pin the app" was exactly that);
  * a real process to put in startup;
  * a background app with no console, like any other .exe.

It sits in the project root so it can find ``.venv\\Scripts\\pythonw.exe`` and
``orion.py`` beside it. Built with build_exe.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _base_dir() -> Path:
    """The project root — where orion.py and .venv live.

    When frozen, the exe sits in the project root, so its own location is the
    base. Run as a script, it is one level down in launcher/.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _interpreter(base: Path) -> str:
    """The windowless interpreter to run ORION with."""
    for candidate in (
        base / ".venv" / "Scripts" / "pythonw.exe",
        base / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return "pythonw"          # fall back to whatever is on PATH


def _standalone(base: Path) -> Path | None:
    """The real standalone ORION (build_standalone.py), if it is built.

    Preferred over running orion.py: under the Microsoft Store Python the
    interpreter is a PACKAGED app, and Windows attributes every window it
    opens to that package — so ORION started through this launcher showed in
    the taskbar and Task Manager as "Python 3.13", with Python's icon, whatever
    ORION set. The standalone ORION.exe IS the process, with ORION's own name
    and icon.
    """
    candidate = base / "dist" / "ORION" / "ORION.exe"
    return candidate if candidate.is_file() else None


def main() -> int:
    base = _base_dir()
    standalone = _standalone(base)
    if standalone is not None:
        flags = getattr(subprocess, "DETACHED_PROCESS", 0)
        try:
            subprocess.Popen([str(standalone)], cwd=str(standalone.parent),
                             creationflags=flags, close_fds=True)
            return 0
        except OSError:
            pass            # fall through to running the source
    script = base / "orion.py"
    if not script.is_file():
        # A GUI message rather than a silent failure — the exe has no console.
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, f"ORION could not find orion.py next to the app "
                   f"(looked in {base}).", "ORION", 0x10)
        except Exception:
            pass
        return 1
    flags = (getattr(subprocess, "DETACHED_PROCESS", 0)
             | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        subprocess.Popen([_interpreter(base), str(script)],
                         cwd=str(base), creationflags=flags, close_fds=True)
    except OSError as exc:
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, f"ORION could not start: {exc}", "ORION", 0x10)
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
