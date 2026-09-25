"""
The self-installer (Mark XXII) — make ORION a real installed application.

    "Instead of ORION opening up just in Python, can we make him an actual
     application — an installable .exe that actually installs itself onto the
     PC?"

ORION already runs as a real background .exe (``ORION.exe``, built by
build_exe.py — a launcher that starts the app windowless and detached, pinnable
to the taskbar). What made it still feel like "just Python" was that nothing
*registered* it with Windows: no entry in Add/Remove Programs, no clean
uninstaller, no one-click install step.

This closes that gap without the fragile parts. It deliberately does NOT copy
the project into Program Files or freeze a 500 MB bundle — the app runs in place
under its own interpreter (which is exactly what works today), and a naive copy
would risk launching it under the wrong Python. Instead it:

  * builds the desktop + Start-menu shortcuts (via desktop_app, already tested);
  * writes a proper **Add/Remove Programs** entry under
    ``HKCU\\...\\Uninstall\\ORION`` — DisplayName, icon, version, publisher and a
    working UninstallString — so ORION appears in "Installed apps" and uninstalls
    cleanly like any other program;
  * optionally registers itself to start with Windows.

Everything that talks to the registry is injectable, so the entry construction
and the install/uninstall flow are unit-tested without writing a single real
registry key.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .constants import APP_BUILD, BASE_DIR
from . import desktop_app

# HKEY_CURRENT_USER so no administrator elevation is needed — a per-user install,
# which is what a single-user desktop assistant wants.
UNINSTALL_SUBKEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\ORION"
DISPLAY_NAME = "O.R.I.O.N."
PUBLISHER = "Project ORION"


def _version() -> str:
    raw = str(APP_BUILD or "").strip()
    # APP_BUILD may be "Mark XXII" or similar; keep a numeric-ish version too.
    return raw or "22.0"


def uninstall_command() -> str:
    """The command Windows runs when the user clicks Uninstall.

    Uses the same windowless interpreter the launcher prefers, invoking this
    module's ``--uninstall`` entry point. Quoted for paths with spaces.
    """
    interp = desktop_app._launch_target()[0] if hasattr(desktop_app, "_launch_target") else sys.executable
    # Prefer a real pythonw so the uninstaller has no console flash.
    pyw = _pythonw()
    return f'"{pyw}" -m orion_core.installer --uninstall'


def _pythonw() -> str:
    for candidate in (
        BASE_DIR / ".venv" / "Scripts" / "pythonw.exe",
        BASE_DIR / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return str(candidate)
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    return str(pyw if pyw.exists() else exe)


def uninstall_entries(*, install_location: Path | None = None,
                      icon_path: Path | None = None,
                      version: str | None = None) -> dict[str, Any]:
    """The Add/Remove Programs values, as a plain dict (pure + testable)."""
    install_location = install_location or BASE_DIR
    icon = icon_path or (desktop_app.ICON_PATH if hasattr(desktop_app, "ICON_PATH")
                         else install_location / "assets" / "orion.ico")
    exe = getattr(desktop_app, "EXE_PATH", install_location / "ORION.exe")
    return {
        "DisplayName": DISPLAY_NAME,
        "DisplayVersion": version or _version(),
        "Publisher": PUBLISHER,
        "DisplayIcon": str(icon),
        "InstallLocation": str(install_location),
        "UninstallString": uninstall_command(),
        "QuietUninstallString": uninstall_command(),
        "DisplayName_exe": str(exe),
        "NoModify": 1,
        "NoRepair": 1,
        "EstimatedSize": 250000,     # KB, approximate — for the size column
    }


# ── registry access (injectable) ─────────────────────────────────────────────

class RegistryWriter:
    """Thin wrapper over winreg so install/uninstall can be tested with a fake."""

    def write(self, subkey: str, values: dict[str, Any]) -> None:
        import winreg
        key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, subkey, 0,
                                 winreg.KEY_WRITE)
        try:
            for name, value in values.items():
                if name == "DisplayName_exe":
                    continue          # internal, not a real registry value
                if isinstance(value, int):
                    winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)
                else:
                    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, str(value))
        finally:
            winreg.CloseKey(key)

    def delete(self, subkey: str) -> None:
        import winreg
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)
        except FileNotFoundError:
            pass


@dataclass
class InstallReport:
    shortcuts: dict[str, bool]
    registered: bool
    startup: bool
    icon: str

    def describe(self) -> str:
        made = [k for k, ok in self.shortcuts.items() if ok]
        bits = [f"shortcuts: {', '.join(made) or 'none'}"]
        bits.append("registered in Add/Remove Programs" if self.registered
                    else "registry entry skipped")
        if self.startup:
            bits.append("starts with Windows")
        return "ORION installed — " + "; ".join(bits) + "."


def install(*, add_to_startup: bool = False,
            writer: RegistryWriter | None = None) -> InstallReport:
    """Register ORION as an installed application on this PC."""
    writer = writer or RegistryWriter()
    icon = desktop_app.ensure_icon()
    shortcuts = desktop_app.install_shortcuts()
    registered = True
    try:
        writer.write(UNINSTALL_SUBKEY, uninstall_entries())
    except Exception:
        registered = False
    startup = False
    if add_to_startup:
        try:
            from . import autostart
            startup = bool(autostart.enable())
        except Exception:
            startup = False
    return InstallReport(shortcuts=shortcuts, registered=registered,
                         startup=startup, icon=str(icon or ""))


def uninstall(*, writer: RegistryWriter | None = None) -> str:
    """Remove ORION's shortcuts, registry entry and startup registration.

    The app files themselves are left in place — they are the user's project, not
    something this installer copied in, so removing them is not ours to do.
    """
    writer = writer or RegistryWriter()
    removed: list[str] = []
    try:
        _remove_shortcuts()
        removed.append("shortcuts")
    except Exception:
        pass
    try:
        writer.delete(UNINSTALL_SUBKEY)
        removed.append("Add/Remove Programs entry")
    except Exception:
        pass
    try:
        from . import autostart
        autostart.disable()
        removed.append("startup entry")
    except Exception:
        pass
    return "ORION uninstalled — removed " + ", ".join(removed) + \
           ". Your project files were left untouched."


def _remove_shortcuts() -> None:
    """Delete the Desktop and Start-menu shortcuts install_shortcuts created."""
    name = getattr(desktop_app, "SHORTCUT_NAME", "O.R.I.O.N.") + ".lnk"
    for folder in (desktop_app._desktop_dir(), desktop_app._start_menu_dir()):
        if folder is None:
            continue
        link = Path(folder) / name
        try:
            if link.exists():
                link.unlink()
        except OSError:
            pass


# ── CLI entry point ──────────────────────────────────────────────────────────

def _confirm(message: str, title: str = "ORION") -> bool:
    """A GUI yes/no when there's no console (the setup .exe is windowless)."""
    try:
        import ctypes
        # MB_YESNO | MB_ICONQUESTION
        return ctypes.windll.user32.MessageBoxW(0, message, title, 0x24) == 6
    except Exception:
        return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--uninstall" in argv:
        if not _confirm("Remove ORION from this PC? Your project files stay put."):
            return 1
        message = uninstall()
    else:
        startup = "--startup" in argv or "--with-startup" in argv
        report = install(add_to_startup=startup)
        message = report.describe()
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, "ORION", 0x40)  # MB_ICONINFO
    except Exception:
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["install", "uninstall", "uninstall_entries", "uninstall_command",
           "RegistryWriter", "InstallReport", "UNINSTALL_SUBKEY"]
