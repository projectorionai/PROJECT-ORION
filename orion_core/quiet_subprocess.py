"""
No console windows flashing up while ORION works.

  "When ORION's opened in CMD, cmd keeps randomly popping up and I want that to
   stop since it's annoying"

Those flashes are child processes. ORION shells out constantly — pip when the
forge installs a dependency, Stockfish when chess opens, ffmpeg for audio, git,
nmap, and more — and on Windows a console child launched without an explicit
"no window" instruction gets its own console window, which appears, steals
focus for an instant, and vanishes. Thirty-odd call sites do this, and several
are in third-party libraries (python-chess starting Stockfish, pip itself), so
fixing them one by one would be incomplete the moment a library spawns
something.

So it is fixed in ONE place. ``subprocess.Popen`` is the funnel every other
spawn helper (`run`, `call`, `check_output`, …) goes through, including the
ones inside libraries, so wrapping its constructor to default to CREATE_NO_WINDOW
covers all of them at once — ORION's own code and its dependencies' — with no
call site needing to change.

Why this is safe
----------------
ORION has no child process that is meant to show a console. Every one is a tool
whose output is captured or ignored. The wrapper therefore only ever SUPPRESSES
a window; it never removes one that was wanted. And it steps aside completely
the moment a caller expresses any console intent of its own — an explicit
``creationflags``, a ``startupinfo``, or asking for a new console — so nothing
that deliberately wants a terminal is affected.
"""

from __future__ import annotations

import subprocess
import sys

#: CREATE_NO_WINDOW: the child runs with no console window at all. 0 off-Windows
#: so the wrapper is a no-op there rather than a special case at every guard.
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if sys.platform == "win32" else 0

#: Flags that mean "the caller has an opinion about the console" — if any is
#: set we leave the call exactly as it was.
_CONSOLE_INTENT = 0
if sys.platform == "win32":
    _CONSOLE_INTENT = (
        getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "DETACHED_PROCESS", 0)
    )

_installed = False
_original_init = None


def quiet_kwargs(existing: dict | None = None) -> dict:
    """kwargs that hide the console, merged onto *existing*. For call sites that
    prefer to be explicit rather than rely on the global wrapper."""
    kwargs = dict(existing or {})
    if sys.platform != "win32":
        return kwargs
    if "creationflags" not in kwargs and "startupinfo" not in kwargs:
        kwargs["creationflags"] = CREATE_NO_WINDOW
    return kwargs


def install() -> None:
    """Make every Popen on Windows default to a hidden console. Idempotent.

    Wraps ``Popen.__init__`` rather than replacing ``Popen``: subclasses,
    ``run``, ``call`` and library code all construct the same class, so the one
    hook reaches every path.
    """
    global _installed, _original_init
    if _installed or sys.platform != "win32":
        _installed = True
        return
    _original_init = subprocess.Popen.__init__

    def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Stand aside if the caller already said anything about the console.
        flags = kwargs.get("creationflags", 0) or 0
        if not (flags & _CONSOLE_INTENT) and "startupinfo" not in kwargs:
            kwargs["creationflags"] = flags | CREATE_NO_WINDOW
        return _original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _init  # type: ignore[assignment]
    _installed = True


def uninstall() -> None:
    """Restore the original Popen — for tests, so the wrapper cannot leak
    across cases and hide a genuine assertion about creationflags."""
    global _installed, _original_init
    if _original_init is not None:
        subprocess.Popen.__init__ = _original_init  # type: ignore[assignment]
        _original_init = None
    _installed = False


def is_installed() -> bool:
    return _installed


__all__ = ["CREATE_NO_WINDOW", "install", "is_installed", "quiet_kwargs",
           "uninstall"]
