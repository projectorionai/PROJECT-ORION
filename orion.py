"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  O.R.I.O.N.  Mark XXV  —  Open Resolution Intelligence Overt Network         ║
║  Launcher  |  python orion.py  |  Windows / Linux                            ║
║                                                                              ║
║  The Mark X architecture lives in the orion_core/ package:                   ║
║  modular services and managers (audio, vision, agents, memory, providers,   ║
║  Outlook, Notion, briefing, dual-window GUI).  This file is intentionally   ║
║  a thin shim so `python orion.py` keeps working exactly as it always has.   ║
║                                                                              ║
║  Current capabilities: docs/CAPABILITIES_2026-09-18.md                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import time


_LAUNCH_STARTED_AT = time.perf_counter()


def _force_utf8_console() -> None:
    """Make stdout/stderr UTF-8 before anything can write to them.

    ORION's log lines carry '✓', '✗', '—' and similar.  Under the desktop GUI
    those land in a Qt widget and are fine, but a console run on Windows gets a
    cp1252 stream and the first tick mark raises UnicodeEncodeError — killing
    whatever was logging rather than the thing it was reporting on.  Headless
    and server runs write to a console by definition, so this is not optional
    there.  ``errors="replace"`` guarantees output can never be the thing that
    crashes the process.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass          # a redirected or already-wrapped stream: leave it


_force_utf8_console()


def _relaunch_in_background() -> None:
    """Re-launch ORION under pythonw.exe (no console) and let the console exit.

    "make ORION a background app maybe?" — this is the way to do it without a
    console window at all. python.exe keeps a console attached for its whole
    life; pythonw.exe is the same interpreter with none. So when asked for
    background mode we relaunch ourselves under pythonw, detached, and the
    original console process exits, leaving ORION running with no terminal.

    Opt-in (``--background`` or ORION_BACKGROUND=1) rather than automatic,
    because a console is exactly what you want while developing — this must not
    take the logs away from someone who is watching them. Guarded against
    relaunching itself forever with an environment marker.
    """
    if sys.platform != "win32" or os.getenv("ORION_RELAUNCHED") == "1":
        return
    wants = (os.getenv("ORION_BACKGROUND", "").strip().lower() in {"1", "true", "yes", "on"}
             or "--background" in sys.argv[1:])
    if not wants:
        return
    import subprocess
    from pathlib import Path

    current = Path(sys.executable)
    windowed = current.with_name(current.name.replace("python", "pythonw"))
    if "python" not in current.name.lower() or not windowed.exists():
        return          # already windowless, or no pythonw beside this python
    argv = [a for a in sys.argv if a != "--background"]
    env = dict(os.environ, ORION_RELAUNCHED="1")
    try:
        subprocess.Popen(
            [str(windowed)] + argv, env=env,
            creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0),
            close_fds=True)
    except OSError:
        return          # could not relaunch — fall through and run in console
    print("O.R.I.O.N. is starting in the background (no console). "
          "This window can be closed.")
    raise SystemExit(0)


_relaunch_in_background()


def _claim_single_instance() -> object:
    """Refuse to be the second ORION.

    Before anything else: before the GUI, before a microphone is opened,
    before a live session is dialled, before a single database is touched. A
    second instance that gets as far as opening the microphone has already
    taken it from the first.

    Nothing used to stop a second ORION at all — so a launch while one was
    running started a whole second assistant, with its own audio capture on
    the same device, its own live session spending its own tokens, and its own
    writers against the same SQLite files. And because ORION can restart
    himself, one that respawned before the old process had gone left two.

    The claim is returned and held for the life of the process. Letting it be
    garbage-collected would release the lock while ORION is still running.
    """
    try:
        from orion_core.single_instance import enforce
    except Exception:
        return None                 # never block a launch over the guard
    claim = enforce()
    if not claim.granted:
        raise SystemExit(0)
    return claim


# Only when this file is being RUN. Importing it is not starting a second
# ORION -- the desktop launcher's preflight imports it to check dependencies,
# and diagnostics do the same -- but the claim ran at import time, so any of
# that raised SystemExit(0) the moment a real ORION was already up. The
# preflight then produced no output at all and a perfectly healthy launcher
# looked broken, which read as five unrelated flaky tests because it only
# happened while ORION was open.
#
# Kept HERE rather than moved down to the __main__ block at the foot of the
# file: a refused second instance should exit before the whole application
# graph is imported, not after.
_INSTANCE_CLAIM = _claim_single_instance() if __name__ == "__main__" else None


def _headless_requested() -> bool:
    """Headless (cloud/server) mode: no GUI, brain + remote uplink only."""
    if os.getenv("ORION_HEADLESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return any(arg in {"--headless", "--server", "--cloud"} for arg in sys.argv[1:])


def _require_modules(*names: str) -> None:
    """Check installation without executing the heavy runtime packages.

    The desktop deliberately warms GenAI, HTTP and audio after first paint.
    Importing them here merely to check installation used to bypass that
    policy, adding over a second before the window could exist. ``find_spec``
    checks availability; the actual imports still validate each package when
    its service is warmed or used. Missing namespace parents count as absent.
    """
    from importlib.util import find_spec

    missing: list[str] = []
    for name in names:
        try:
            if find_spec(name) is None:
                missing.append(name)
        except (ImportError, AttributeError, ValueError):
            missing.append(name)
    if missing:
        raise ModuleNotFoundError("Missing packages: " + ", ".join(missing))


if _headless_requested():
    # Lighter gate: the cloud node needs no GUI, audio or screen-grab stack.
    try:
        _require_modules("aiohttp", "qasync", "PyQt6.QtCore")
    except Exception as import_error:  # pragma: no cover - env dependent
        print("O.R.I.O.N. headless node cannot initialise. Missing dependency:")
        print(f"  {import_error}")
        print("Install: pip install -r deploy/requirements-server.txt")
        raise SystemExit(1)
    from orion_core.server import main
else:
    # Full desktop gate: fail with an actionable message instead of a stack trace.
    try:
        _require_modules(
            "aiohttp", "mss", "psutil", "qasync", "sounddevice",
            "google.genai", "PIL.Image", "PyQt6.QtWidgets",
        )
    except Exception as import_error:
        print("O.R.I.O.N. cannot initialise. Missing runtime dependency:")
        print(f"  {import_error}")
        print(
            "Install: pip install PyQt6 qasync aiohttp sounddevice "
            "google-genai pillow mss psutil\n"
            "Optional (recommended): pip install pyttsx3 pywin32 pytesseract vosk pypdf\n"
            "Cloud/headless node instead? Run: python orion.py --headless"
        )
        raise SystemExit(1)
    from orion_core.app import main

if __name__ == "__main__":
    if _headless_requested():
        main()
    else:
        main(started_at=_LAUNCH_STARTED_AT)
