"""
One authoritative answer to "which browser does ORION open?".

"ORION must use Microsoft Edge or msedge.exe as the main browser it goes to -
the default browser must be used each time."

Two requirements that only look like one. "The default browser each time" means
ORION must not have a browser of his own choosing at all — he should land
wherever the user's own links land, with their profile, their signed-in
sessions, their extensions. "Edge / msedge.exe" is what that resolves to on
this machine, and is the right answer when the default cannot be determined.

Why ``webbrowser.open`` alone was not enough
--------------------------------------------
It is documented as using the default browser, and on Windows it usually does —
but not reliably, and the ways it goes wrong are all silent:

  * a ``BROWSER`` environment variable overrides everything, and anything from
    a shell profile to a stray tool can set it;
  * its Windows backend calls ``os.startfile``, which honours the *file*
    association for whatever it is handed rather than the *http/https protocol*
    association — those are separate registry keys and can disagree;
  * inside a frozen or embedded interpreter its registration table can come up
    empty, at which point it fails silently and nothing opens at all.

So the default is resolved explicitly, from the key Windows itself uses for
"which browser handles https" (``UrlAssociations\\https\\UserChoice``), and
everything falls back through Edge before it reaches ``webbrowser``.

Every lookup is cached: this is asked once per link opened, it involves registry
reads and filesystem probes, and the answer changes about once a year.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from functools import lru_cache
from pathlib import Path, PureWindowsPath

#: Set ORION_BROWSER_PATH to override everything below — an explicit
#: instruction from the user always outranks anything inferred.
ENV_OVERRIDE = "ORION_BROWSER_PATH"


def _edge_candidates() -> list[Path]:
    pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    return [
        Path(pf86) / "Microsoft/Edge/Application/msedge.exe",
        Path(pf) / "Microsoft/Edge/Application/msedge.exe",
        Path(local) / "Microsoft/Edge/Application/msedge.exe",
    ]


@lru_cache(maxsize=1)
def edge_path() -> str | None:
    """msedge.exe, if this machine has it."""
    if sys.platform == "darwin":
        mac = Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")
        return str(mac) if mac.exists() else None
    if sys.platform != "win32":
        return shutil.which("microsoft-edge") or shutil.which("microsoft-edge-stable")
    for candidate in _edge_candidates():
        if candidate.exists():
            return str(candidate)
    return shutil.which("msedge")


def _command_from_progid(progid: str) -> str | None:
    """The executable a Windows ProgId launches, from its shell open command."""
    try:
        import winreg
    except ImportError:
        return None
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            key_path = rf"SOFTWARE\Classes\{progid}\shell\open\command"
            with winreg.OpenKey(root, key_path) as key:
                command, _ = winreg.QueryValueEx(key, "")
        except OSError:
            continue
        return _executable_from_command(str(command))
    return None


def _executable_from_command(command: str) -> str | None:
    """Pull the program out of a registry shell command string.

    These are not plain paths — they look like
    ``"C:\\Program Files\\...\\msedge.exe" --single-argument %1`` — so the
    quoted form is taken first, and the unquoted form is truncated at the first
    switch rather than at the first space (Windows paths are full of spaces).
    """
    text = str(command or "").strip()
    if not text:
        return None
    if text.startswith('"'):
        closing = text.find('"', 1)
        if closing > 1:
            candidate = text[1:closing]
            return candidate if Path(candidate).exists() else None
        return None
    for marker in (" --", " -", " /", " %"):
        cut = text.find(marker)
        if cut > 0:
            text = text[:cut]
            break
    text = text.strip()
    return text if Path(text).exists() else None


@lru_cache(maxsize=1)
def default_browser_path() -> str | None:
    """The browser Windows itself uses for https links, or None.

    Read from UserChoice, which is the association the user actually set — not
    the file association ``os.startfile`` consults, which is a different key and
    can disagree with it.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None
    try:
        key_path = (r"SOFTWARE\Microsoft\Windows\Shell\Associations"
                    r"\UrlAssociations\https\UserChoice")
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            progid, _ = winreg.QueryValueEx(key, "ProgId")
    except OSError:
        return None
    return _command_from_progid(str(progid))


@lru_cache(maxsize=1)
def browser_path() -> str | None:
    """The executable ORION opens links with, in the user's stated order."""
    override = os.getenv(ENV_OVERRIDE, "").strip().strip('"')
    if override and Path(override).exists():
        return override
    return default_browser_path() or edge_path()


@lru_cache(maxsize=1)
def browser_name() -> str:
    """A human name for the browser in use — for what ORION says out loud."""
    path = browser_path()
    if not path:
        return "your default browser"
    # PureWindowsPath splits on both separators, so the name is right for a
    # Windows path wherever this runs (the registry hands back C:\...).
    stem = PureWindowsPath(path).stem
    return {
        "msedge": "Microsoft Edge",
        "chrome": "Chrome",
        "firefox": "Firefox",
        "brave": "Brave",
        "opera": "Opera",
        "vivaldi": "Vivaldi",
    }.get(stem.lower(), stem)


def open_url(url: str) -> bool:
    """Open *url* in the user's browser. Returns whether anything was launched.

    Every URL ORION opens goes through here, so there is exactly one place that
    decides, and one place to change if the answer ever needs to.
    """
    target = str(url or "").strip()
    if not target:
        return False
    path = browser_path()
    if path:
        try:
            # Detached: ORION must not hold a handle on a browser the user will
            # keep open for hours, and must not die with it either.
            subprocess.Popen(
                [path, target],
                creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                close_fds=True,
            )
            return True
        except (OSError, ValueError):
            pass          # fall through — a missing browser is not fatal
    try:
        return bool(webbrowser.open(target))
    except Exception:
        return False


def reset_cache() -> None:
    """Forget the resolved browser — for tests, and after the user changes it."""
    for cached in (edge_path, default_browser_path, browser_path, browser_name):
        cached.cache_clear()


__all__ = [
    "ENV_OVERRIDE", "browser_name", "browser_path", "default_browser_path",
    "edge_path", "open_url", "reset_cache",
]
