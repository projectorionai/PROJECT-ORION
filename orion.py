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

# Mark XXXX.4: OpenMP threads (faster-whisper's ctranslate2) sleep as soon as a
# transcription ends instead of spinning: 40% less CPU, same 0.63 s latency.
os.environ.setdefault("KMP_BLOCKTIME", "0")
# OpenBLAS (numpy's, and scipy's own copy) commits a ~32 MB buffer per thread
# the moment it loads, one thread per logical core. MEASURED 29 Oct 2026 on the
# owner's 12-thread Ryzen: `import numpy` committed +363 MB private with +11 MB
# resident; at 4 threads +106 MB. ORION's BLAS work (recall over 20k x 384
# vectors, intent-network batches, 512x512 products) ran as fast at 4 threads
# as at 12 (0.32 vs 0.49 ms, 1.67 vs 1.96 ms, 0.64 vs 0.48 ms). Must be set
# before anything imports numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")


def _confine_link_launch(argv: list) -> list:
    """A launch carrying an orion:// link carries the link and nothing else.

    Windows runs `ORION.exe "%1"` for a link, and %1 is a URL a web page
    chose. Should a browser ever pass quotes through unescaped, the rest of
    that URL would arrive as extra arguments - `--forge-verify x.py`,
    `--mcp-server` - and be obeyed by the mode dispatch below. So the extra
    arguments are dropped before anything reads them; windows_shell then
    parses the link against its allowlist.
    """
    links = [a for a in argv[1:]
             if isinstance(a, str) and a.strip().lower().startswith("orion:")]
    return [argv[0], *links] if links and argv else argv


if __name__ == "__main__":
    sys.argv[:] = _confine_link_launch(sys.argv)

# Local operation is a launch choice, before provider or GUI bootstrap runs.
# Headless children inherit the same routing policy without changing saved keys.
if __name__ == "__main__" and "--offline" in sys.argv[1:]:
    os.environ["ORION_MODE"] = "offline"

# PyInstaller must dispatch multiprocessing children BEFORE claiming the UI
# singleton or importing Qt/audio. spawn imports this file as __mp_main__.
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

def _headless(run: "object") -> int:
    """Run a headless child mode; a failure becomes a traceback on fd 2 and
    exit code 1.

    The packaged app is a windowed executable, and PyInstaller shows a dialog
    box for any unhandled exception in one - so a background child of ORION
    (an MCP server, a page host, the Forge verifier) that failed to import
    popped "Unhandled exception in script" onto the owner's screen (11 Oct
    2026). Its parent reads the pipe and logs the failure instead. (Each mode
    keeps a plain import inside its function so PyInstaller still finds it.)
    """
    try:
        return run()  # type: ignore[operator]
    except SystemExit:
        raise
    except BaseException:
        import traceback
        text = traceback.format_exc()
        try:
            os.write(2, text.encode("utf-8", "replace"))
        except OSError:
            pass
        return 1


def _report_desktop_boot_failure(exc: BaseException) -> None:
    """Persist a traceback when the GUI fails before its crash reporter starts.

    The normal app installs CrashReporter only after most of startup. An error
    while QApplication or the first window is being created therefore used to
    leave a session header, but no traceback in the diagnostic files.
    """
    import traceback

    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    summary = f"Desktop startup failed: {type(exc).__name__}: {exc}"
    try:
        from orion_core import session_log
        summary = session_log.redact(summary)
        detail = session_log.redact(detail)
        session_log.write(f"CRASH: {summary}\n{detail}")
        session_log.flush(timeout=2.0)
    except BaseException:
        pass
    try:
        from orion_core.crash_reporter import CrashReporter
        CrashReporter(None).report("python", summary, detail)
    except BaseException:
        pass
    try:
        os.write(2, f"ORION could not start: {summary}\n".encode("utf-8", "replace"))
    except OSError:
        pass


def _argv_after(flag: str) -> list[str]:
    return sys.argv[sys.argv.index(flag) + 1:]


def _mcp_server() -> int:
    from orion_core.mcp_bundled import run
    return run(_argv_after("--mcp-server"))


def _face_host() -> int:
    from orion_core.gui.face_host import run_host
    return run_host()


def _page_host() -> int:
    from orion_core.gui.page_host import run_host
    rest = _argv_after("--page-host")
    return run_host(rest[0] if rest else "")


def _forge_verify() -> int:
    from orion_core.forge_runtime import run_script
    return run_script(_argv_after("--forge-verify"))


# The packaged app runs its own bundled MCP servers (`ORION.exe --mcp-server
# orion_world`): no window, no singleton, no Qt - see orion_core/mcp_bundled.
if __name__ == "__main__" and "--mcp-server" in sys.argv[1:]:
    raise SystemExit(_headless(_mcp_server))

# ORION's face in its own process (`ORION.exe --face-host`), embedded in the
# Core Window so Python stalls cannot stutter it - see orion_core/gui/face_host.
if __name__ == "__main__" and "--face-host" in sys.argv[1:]:
    raise SystemExit(_headless(_face_host))

# The globe and the Brain instruments the same way (`ORION.exe --page-host
# web|brain|face`) - see orion_core/gui/page_host.
if __name__ == "__main__" and "--page-host" in sys.argv[1:]:
    raise SystemExit(_headless(_page_host))

# The Forge verifies new tools with the installed app's own interpreter
# (`ORION.exe --forge-verify script.py`) - see orion_core/forge_runtime.
if __name__ == "__main__" and "--forge-verify" in sys.argv[1:]:
    raise SystemExit(_headless(_forge_verify))

# Forged tools may import packages the Forge installed for them.
if __name__ == "__main__":
    try:
        from orion_core.forge_runtime import add_packages_to_path as _forge_packages
        _forge_packages()
    except Exception:
        pass


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


if __name__ == "__main__":
    if "--game-updater-scheduled" in sys.argv[1:]:
        # A scheduled launcher operation needs neither an ORION window nor
        # the UI singleton; it may run while the assistant is already open.
        from orion_core.jv_plugins.scheduled import main as scheduled_game_update
        raise SystemExit(scheduled_game_update())
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

    A refused launch that carries a shell command (Explorer's "Ask ORION",
    a jump-list task, an orion:// link) hands it to the running ORION before
    exiting - enforce() does that through windows_shell.forward_launch.
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


if __name__ == "__mp_main__":
    pass
elif _headless_requested():
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
        # The log lines also go to config/diagnostics/session.log (a real run
        # only: tests build windows without passing through here).
        try:
            from orion_core import session_log
            session_log.enable()
        except Exception:
            pass
        # ORION speaks in Charon even when the Live channel is down (voice_charon).
        try:
            from orion_core import voice_charon
            voice_charon.enable()
        except Exception:
            pass
        # Shell opens run off the event-loop thread (see pump_guard).
        try:
            from orion_core import pump_guard
            pump_guard.install()
        except Exception:
            pass
        try:
            main(started_at=_LAUNCH_STARTED_AT)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            _report_desktop_boot_failure(exc)
            raise SystemExit(1) from None
