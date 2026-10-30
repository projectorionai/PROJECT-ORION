"""Keep Windows' message pumping off ORION's event-loop thread.

ORION's asyncio loop IS the Qt GUI thread (qasync). Some Windows calls wait by
pumping that thread's message queue - ``os.startfile`` (ShellExecute) is the
common one. When that happens inside an asyncio task, Qt dispatches nested
events, qasync steps OTHER tasks inside the running one, and each of those
fails with "Cannot enter into task" and never resumes. On 8 Oct 2026 that is
how the stall detector lost its heartbeat (23:32:09, the moment find_files
opened the research folder) and, via the camera, how the Live voice went
silent for ten minutes.

:func:`install` wraps ``os.startfile`` once, so every caller - ORION's own
and its libraries' - is covered: on the GUI thread the call runs on a short
worker thread (whose own queue does the pumping) and is waited for without
pumping. Elsewhere it runs as before.
"""

from __future__ import annotations

import os
import threading
from typing import Any

#: How long the GUI thread waits for the shell before letting it finish alone.
WAIT_S = 10.0

_original: Any = None


def _run_with_com(call: Any, box: dict[str, BaseException]) -> None:
    try:
        try:
            import pythoncom  # type: ignore[import-not-found]
            pythoncom.CoInitialize()     # shell extensions are COM objects
        except Exception:
            pass
        call()
    except BaseException as exc:         # re-raised on the caller's thread
        box["error"] = exc


def _startfile_off_the_gui(path: Any, *args: Any, **kwargs: Any) -> Any:
    if threading.current_thread() is not threading.main_thread():
        return _original(path, *args, **kwargs)
    box: dict[str, BaseException] = {}
    worker = threading.Thread(
        target=_run_with_com, args=(lambda: _original(path, *args, **kwargs), box),
        name="orion-startfile", daemon=True)
    worker.start()
    worker.join(WAIT_S)
    if "error" in box:
        raise box["error"]
    return None


def install() -> bool:
    """Wrap os.startfile (Windows only, once). True when installed."""
    global _original
    if _original is not None or not hasattr(os, "startfile"):
        return False
    _original = os.startfile
    os.startfile = _startfile_off_the_gui      # type: ignore[attr-defined]
    return True


__all__ = ["install", "WAIT_S"]
