"""
One ORION at a time.

Nothing prevented a second ORION from starting. Not a lock, not a mutex, not a
check of any kind — so every launch started a whole new assistant: its own
audio capture on the same microphone, its own live session burning its own
tokens, its own writers against the same SQLite files. And because a running
ORION can restart himself, one that respawns before the old process has gone
leaves two, and two that each do it leave four. That is what replicating looks
like from the outside, and it is not malice, it is a missing guard.

Why a kernel object rather than a lock file
-------------------------------------------
A lock file has to be deleted to be released, which means a crash, a kill, or
a power cut leaves a file claiming ORION is running when he is not — and then
he will not start at all, which is a worse failure than the one being fixed.
Every workaround for that (a PID inside the file, a timestamp, a liveness
probe) is another thing to get wrong.

A named mutex has no such problem: Windows releases it when the process ends,
however it ends. There is no stale state to clean up because there is no state
on disk. On other platforms an abstract-namespace socket has exactly the same
property — the kernel reclaims it on exit — and a POSIX file lock is the
fallback, which `flock` also releases on process death.

What the second instance does
-----------------------------
It does not die silently. A user who double-clicks the icon and sees nothing
happen will double-click it again. So the second instance raises the window of
the first and then exits, which is what every other desktop application does
and what the person actually meant.
"""

from __future__ import annotations

import sys

import os
from dataclasses import dataclass
from typing import Any

#: The name the lock is claimed under. Global\ so it spans terminal-server
#: sessions: two ORIONs under one login are the problem, and two under
#: different logins on the same machine would still fight over the same audio
#: device and the same databases.
MUTEX_NAME = "Global\\ORION-single-instance-9f3c"

#: The same identity for the POSIX fallback.
SOCKET_NAME = "\0orion-single-instance-9f3c"

#: Where the file-lock fallback lives when nothing better is available.
LOCK_FILENAME = "orion.lock"


@dataclass
class Claim:
    """The outcome of trying to be the one ORION."""

    granted: bool
    reason: str = ""
    #: Kept so the operating system does not reclaim the handle early. A mutex
    #: whose handle is garbage-collected is a mutex that stops holding.
    handle: Any = None

    def __bool__(self) -> bool:
        return self.granted


def _claim_windows() -> Claim | None:
    """Claim a named mutex. None if this is not Windows or ctypes is absent."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    ERROR_ALREADY_EXISTS = 183
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL,
                                      wintypes.LPCWSTR)
    kernel32.CreateMutexW.restype = wintypes.HANDLE

    handle = kernel32.CreateMutexW(None, True, MUTEX_NAME)
    last_error = ctypes.get_last_error()
    if not handle:
        # Could not create it at all — do not block a launch over a failure to
        # check. Starting twice is bad; refusing to start ever is worse.
        return Claim(True, "the lock could not be created; not enforcing")
    if last_error == ERROR_ALREADY_EXISTS:
        # Close it. A named mutex lives while ANY handle to it is open, so a
        # refused claim that keeps one props the object up after its owner has
        # gone — and then nothing can ever claim it again. That failure is
        # worse than the one this module exists to prevent, because it is
        # silent: the process exits cleanly, having done nothing.
        kernel32.CloseHandle(handle)
        return Claim(False, "another ORION already holds the lock")
    return Claim(True, "lock held", handle)


def _claim_posix() -> Claim | None:
    """Bind an abstract-namespace socket, which the kernel frees on exit."""
    if os.name == "nt":
        return None
    try:
        import socket

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(SOCKET_NAME)
        return Claim(True, "lock held", sock)
    except OSError as exc:
        import errno

        if exc.errno == errno.EADDRINUSE:
            try:
                sock.close()        # same reason as the Windows branch above
            except Exception:
                pass
            return Claim(False, "another ORION already holds the lock")
        try:
            sock.close()
        except Exception:
            pass
        return None
    except Exception:
        return None


def _claim_lockfile(directory: Any = None) -> Claim:
    """Last resort: an OS-level lock on a file.

    Still not a "delete me when done" lock — `flock` and `msvcrt.locking` are
    both released by the kernel when the process ends, so a crash does not
    leave ORION unable to start. The file itself remaining is harmless.
    """
    from pathlib import Path

    try:
        from .constants import CONFIG_DIR

        root = Path(directory) if directory else CONFIG_DIR
    except Exception:
        root = Path(directory) if directory else Path.cwd()
    try:
        root.mkdir(parents=True, exist_ok=True)
        handle = open(root / LOCK_FILENAME, "a+b")
    except OSError as exc:
        return Claim(True, f"no lock file available ({exc}); not enforcing")

    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return Claim(False, "another ORION already holds the lock file")
    except Exception as exc:
        handle.close()
        return Claim(True, f"the lock could not be taken ({exc}); not enforcing")
    return Claim(True, "lock held", handle)


#: The claim this process is holding, if any.
#:
#: Kept here rather than only on the caller because releasing it must not
#: depend on being able to reach whichever module called claim(). app.py used
#: to do `import orion` to find it — and orion.py is the entry script, so it
#: is __main__ and that import builds a SECOND copy of the module, re-running
#: the guard and raising SystemExit out of a `except Exception` that cannot
#: catch it. This module is imported once, under one name, frozen or not.
_HELD: "Claim | None" = None


def held() -> "Claim | None":
    """The claim this process holds, or None."""
    return _HELD


def release() -> bool:
    """Give up the lock, so a successor process can take it.

    Called before respawning on restart. Without it the successor claims a
    mutex still held by a process that is on its way out, is refused, and the
    restart becomes a shutdown.

    Returns whether a handle was actually closed. Never raises: this runs
    while ORION is already shutting down, and an exception here would replace
    a restart with a traceback.
    """
    global _HELD

    claim_held, _HELD = _HELD, None
    handle = getattr(claim_held, "handle", None)
    if handle is None:
        return False
    try:
        if os.name == "nt":
            import ctypes

            ctypes.WinDLL("kernel32").CloseHandle(int(handle))
        else:
            handle.close()
        return True
    except Exception:
        return False


def claim(directory: Any = None) -> Claim:
    """Try to become the one running ORION.

    Returns a granted claim to hold for the life of the process, or a refused
    one meaning another ORION is already running.

    A failure to *check* never blocks a launch. Starting twice is a bad day;
    an assistant that refuses to start at all because a lock could not be
    created is a worse one, and much harder to diagnose.
    """
    global _HELD

    for attempt in (_claim_windows, _claim_posix):
        result = attempt()
        if result is not None:
            if result.granted and result.handle is not None:
                _HELD = result
            return result
    result = _claim_lockfile(directory)
    if result.granted and result.handle is not None:
        _HELD = result
    return result


def raise_existing_window(title_contains: str = "O.R.I.O.N") -> bool:
    """Bring the ORION that is already running to the front.

    A second instance that exits silently looks like a launch that did
    nothing, and the person double-clicks again — which is precisely how one
    stray process becomes several. Showing them the window they asked for is
    both the correct behaviour and the thing that stops the pile-up.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    SW_RESTORE = 9
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _param):
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if title_contains.lower() in buffer.value.lower():
                if user32.IsWindowVisible(hwnd):
                    found.append(hwnd)
                    return False
        return True

    try:
        user32.EnumWindows(each, 0)
    except Exception:
        return False
    if not found:
        return False
    try:
        user32.ShowWindow(found[0], SW_RESTORE)
        user32.SetForegroundWindow(found[0])
        return True
    except Exception:
        return False


def enforce(*, on_refused: Any = None) -> Claim:
    """Claim the lock, and if refused, surface the running ORION and exit.

    Called once, as early in start-up as possible — before audio devices are
    opened, before a live session is dialled, and certainly before anything
    writes to the databases. A second instance that gets as far as opening the
    microphone has already taken it from the first.
    """
    result = claim()
    if result.granted:
        return result

    raised = raise_existing_window()
    message = ("O.R.I.O.N. is already running."
               + (" Bringing his window to the front." if raised else
                  " Look for him in the system tray."))
    if on_refused is not None:
        try:
            on_refused(message)
        except Exception:
            pass
    else:
        _notify_refused(f"[ORION] {message}")
    return result


def _notify_refused(line: str) -> None:
    """Tell the person, on stderr.

    Deliberately not the startup report's say(), which prints to stdout. This
    process is about to exit, and whoever launched it may be READING its
    stdout: the desktop launcher's preflight runs exactly that way, parsing a
    line of JSON back from a subprocess that imports orion.py. Announcing
    there put an English sentence in front of the JSON and the parse failed,
    so a perfectly healthy launcher reported a broken one whenever ORION
    happened to be open.

    A diagnostic belongs on the diagnostic stream. Never raises: under
    pythonw there is no console at all and sys.stderr is None, and a refused
    launch must not turn into a traceback.
    """
    stream = getattr(sys, "stderr", None)
    if stream is None:
        return
    try:
        print(line, file=stream, flush=True)
    except Exception:
        pass


__all__ = [
    "LOCK_FILENAME", "MUTEX_NAME", "SOCKET_NAME",
    "Claim", "claim", "enforce", "held", "raise_existing_window",
    "release",
]
