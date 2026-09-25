"""
Starting ORION with Windows.

"I want ORION to be able to turn on when my PC starts up."

Registered under ``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run``
rather than a Scheduled Task or a Startup-folder shortcut, deliberately:

  * HKCU needs no administrator rights, so enabling it never raises a UAC
    prompt and never needs ORION to be running elevated;
  * it is the list Task Manager's Startup tab shows, so the user can see it and
    switch it off without asking ORION anything — a program that can start
    itself with the machine should be visible where people look for that;
  * a Startup-folder shortcut is a file in a OneDrive-synced tree on this
    machine, which would mean ORION registering itself on every other machine
    signed into the same account.

Three details that decide whether this actually works:

  * **which interpreter.** ORION does not necessarily run under .venv — on this
    machine the real runtime is a separate WindowsApps Python — so the command
    is built from ``sys.executable`` as it is at registration time, not from a
    guess about where Python lives.
  * **pythonw, not python.** ``python.exe`` at logon opens a console window
    that sits behind the UI for the whole session. ``pythonw.exe`` is the same
    interpreter without one.
  * **quoting.** The project path here contains spaces (and an en dash), so
    both the interpreter and the script are quoted. An unquoted Run value
    silently launches nothing at all.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

#: The name the entry appears under in Task Manager's Startup tab.
RUN_KEY_NAME = "ORION"
RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"


@dataclass
class AutostartState:
    """What autostart is currently doing, in terms a person can act on."""

    supported: bool
    enabled: bool
    command: str = ""
    detail: str = ""

    def describe(self) -> str:
        if not self.supported:
            return f"Starting with the machine is not supported here - {self.detail}"
        if self.enabled:
            return ("I'll start automatically when you sign in to Windows. "
                    "You can turn that off in Task Manager's Startup tab, or "
                    "just tell me to stop starting with the PC.")
        return "I don't start automatically at the moment."


def entry_script() -> Path:
    """The script Windows should run: orion.py at the project root."""
    return Path(__file__).resolve().parents[1] / "orion.py"


def launch_interpreter() -> str:
    """The interpreter to launch with — windowed, if one exists beside it.

    ``sys.executable`` is the interpreter ORION is ACTUALLY running under,
    which is the only reliable source: this project has both a .venv and a
    separate WindowsApps Python, and picking the wrong one produces an entry
    that starts something that immediately fails on a missing import.
    """
    current = Path(sys.executable)
    windowed = current.with_name(current.name.replace("python", "pythonw"))
    if "python" in current.name.lower() and windowed.exists():
        return str(windowed)
    return str(current)


def best_launch_exe() -> "Path | None":
    """The best real executable to start ORION by, or None to fall back to the
    interpreter. Preference order is chosen for the taskbar identity the user
    actually sees:

      1. the frozen standalone (build_standalone.py -> dist/ORION/ORION.exe) —
         it IS the process, with ORION's own name and icon and no python at all;
      2. the launcher ORION.exe (build_exe.py) — a real .exe, but it spawns
         pythonw, so the running process still shows as "Python 3.13". Fallback.

    A Run-key entry pointing at (2) while (1) exists is exactly why ORION "still
    opens as Python 3.13 on startup": is_stale() detects the mismatch against
    this preference and ensure_current() upgrades the entry to the standalone.
    """
    root = entry_script().parent
    standalone = root / "dist" / "ORION" / "ORION.exe"
    if standalone.exists():
        return standalone
    launcher = root / "ORION.exe"
    if launcher.exists():
        return launcher
    return None


def startup_command() -> str:
    """The exact command line to register. Quoted — the paths have spaces.

    Prefers a real ORION executable (the standalone first, see best_launch_exe)
    so the logon start is ORION's own process rather than a bare "Python".
    Falls back to launching orion.py with the windowless interpreter.
    """
    exe = best_launch_exe()
    if exe is not None:
        return f'"{exe}"'
    return f'"{launch_interpreter()}" "{entry_script()}"'


def interpreter_can_run_orion(interpreter: str, timeout: float = 25.0) -> bool:
    """Can this interpreter actually start ORION?

    Worth the couple of seconds it costs, because the failure it prevents is
    invisible: this project has more than one Python (a .venv and the system
    one), they do not have the same packages installed, and an entry pointing
    at the wrong one produces a machine that boots, runs the Run key, fails on
    an import, and shows the user nothing at all. Registering something that
    does not work is worse than not registering it.

    PyQt6 is the probe because it is the one import ORION cannot start without
    and the one most likely to be missing from a bare interpreter.
    """
    import subprocess
    windowed = Path(interpreter)
    # A pythonw.exe cannot report back — it has no console — so the probe runs
    # against its console twin.
    console = windowed.with_name(windowed.name.replace("pythonw", "python"))
    probe = console if console.exists() else windowed
    try:
        finished = subprocess.run(
            [str(probe), "-c", "import PyQt6"],
            capture_output=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return False
    return finished.returncode == 0


def _open_run_key(write: bool = False):
    import winreg
    access = winreg.KEY_SET_VALUE if write else winreg.KEY_READ
    return winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, access)


def status() -> AutostartState:
    """Whether ORION is registered to start with Windows."""
    if sys.platform != "win32":
        return AutostartState(supported=False, enabled=False,
                              detail="this is only a Windows facility")
    try:
        import winreg
    except ImportError:
        return AutostartState(supported=False, enabled=False,
                              detail="the registry is not reachable")
    try:
        with _open_run_key() as key:
            value, _ = winreg.QueryValueEx(key, RUN_KEY_NAME)
    except FileNotFoundError:
        return AutostartState(supported=True, enabled=False)
    except OSError as exc:
        return AutostartState(supported=False, enabled=False, detail=str(exc))
    return AutostartState(supported=True, enabled=True, command=str(value))


def enable() -> AutostartState:
    """Register ORION to start at sign-in. Idempotent, and self-correcting.

    Rewritten even when an entry already exists: the interpreter or the project
    path can move (a venv rebuilt, the folder renamed), and a stale entry fails
    silently at logon with nothing to indicate why.
    """
    current = status()
    if not current.supported:
        return current
    command = startup_command()
    script = entry_script()
    if not script.is_file():
        return AutostartState(
            supported=False, enabled=False,
            detail=f"I can't find my own entry point at {script}")
    # A real ORION executable carries its own interpreter resolution, so skip
    # the PyQt probe when one exists — the probe guards the raw-interpreter path
    # only.
    exe = best_launch_exe()
    interpreter = launch_interpreter()
    if exe is None and not interpreter_can_run_orion(interpreter):
        return AutostartState(
            supported=False, enabled=False,
            detail=(f"{interpreter} can't import PyQt6, so an entry pointing at "
                    "it would fail silently at sign-in. Nothing was registered."))
    try:
        import winreg
        with _open_run_key(write=True) as key:
            winreg.SetValueEx(key, RUN_KEY_NAME, 0, winreg.REG_SZ, command)
    except OSError as exc:
        return AutostartState(supported=False, enabled=False, detail=str(exc))
    refreshed = "refreshed" if current.enabled else "set up"
    return AutostartState(supported=True, enabled=True, command=command,
                          detail=f"{refreshed} to run {command}")


def disable() -> AutostartState:
    """Stop starting with Windows. Not being registered is already success."""
    current = status()
    if not current.supported:
        return current
    if not current.enabled:
        return AutostartState(supported=True, enabled=False,
                              detail="I wasn't starting automatically anyway")
    try:
        import winreg
        with _open_run_key(write=True) as key:
            winreg.DeleteValue(key, RUN_KEY_NAME)
    except FileNotFoundError:
        pass
    except OSError as exc:
        return AutostartState(supported=True, enabled=True, detail=str(exc))
    return AutostartState(supported=True, enabled=False,
                          detail="removed from Windows startup")


def is_stale() -> bool:
    """True when registered, but pointing somewhere that no longer exists.

    Worth knowing separately from enabled/disabled: this is the state where the
    user believes ORION starts with the machine and it silently does not.
    """
    current = status()
    if not current.enabled:
        return False
    return current.command.strip() != startup_command().strip()


def ensure_current() -> AutostartState:
    """Repair a stale entry, without turning autostart on if it is off."""
    if is_stale():
        return enable()
    return status()


def launched_at_startup() -> bool:
    """Whether THIS run was started by Windows rather than by the user.

    Set by the registered command's environment on nothing at present — kept as
    the single place to answer the question, since a logon start reasonably
    wants a quieter greeting than a start the user asked for.
    """
    return os.getenv("ORION_LAUNCHED_AT_STARTUP", "").strip().lower() in {
        "1", "true", "yes"}


__all__ = [
    "RUN_KEY_NAME", "RUN_KEY_PATH", "AutostartState", "best_launch_exe",
    "disable", "enable", "ensure_current", "entry_script", "is_stale",
    "launch_interpreter", "launched_at_startup", "startup_command", "status",
]
