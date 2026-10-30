"""
Away mode: the PC stays ready for the owner's phone while he is out.

TORVIS calls it Remote Mode ("Stays ready"); the owner asked on 28 Oct 2026
for everything TORVIS has. While away mode is on:

  * Windows is kept from sleeping (SetThreadExecutionState with
    ES_SYSTEM_REQUIRED, held by one thread for as long as the mode lasts) -
    the screen may still turn off, so nothing is lit for nothing;
  * after LOCK_AFTER_S with no keyboard or mouse input, the PC locks
    (LockWorkStation), once, so a PC left awake is never left open.

Turning it off, or ORION closing, gives sleep back to Windows: the execution
state belongs to the thread, and the thread ends.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Any, Callable

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
#: No input for this long while away, and the PC locks (seconds).
LOCK_AFTER_S = 600.0
POLL_S = 15.0


class _LastInput(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def idle_seconds() -> float:
    """Seconds since the last keyboard or mouse input (0 off Windows)."""
    try:
        info = _LastInput()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        now = ctypes.windll.kernel32.GetTickCount() & 0xFFFFFFFF
        return ((now - info.dwTime) & 0xFFFFFFFF) / 1000.0
    except (AttributeError, OSError):
        return 0.0


def _keep_awake(on: bool) -> bool:
    try:
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except (AttributeError, OSError):
        return False


def _lock() -> bool:
    try:
        return bool(ctypes.windll.user32.LockWorkStation())
    except (AttributeError, OSError):
        return False


class AwayMode:
    def __init__(self, log: Callable[[str], Any] | None = None, *,
                 idle: Callable[[], float] = idle_seconds,
                 keep_awake: Callable[[bool], bool] = _keep_awake,
                 lock: Callable[[], bool] = _lock, poll_s: float = POLL_S,
                 lock_after_s: float = LOCK_AFTER_S) -> None:
        self.log = log or (lambda _line: None)
        self._idle, self._keep_awake, self._lock = idle, keep_awake, lock
        self.poll_s, self.lock_after_s = poll_s, lock_after_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.since = 0.0
        self.locked_at = 0.0

    @property
    def on(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> str:
        if self.on:
            return "Away mode is already on."
        self._stop.clear()
        self.since, self.locked_at = time.time(), 0.0
        self._thread = threading.Thread(target=self._run, name="orion-away-mode", daemon=True)
        self._thread.start()
        minutes = int(self.lock_after_s // 60)
        return (f"Away mode on: this PC stays awake for your phone, and locks after {minutes} "
                "minutes without input.")

    def stop(self) -> str:
        if not self.on:
            return "Away mode is off."
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return "Away mode off: Windows can sleep again."

    def status(self) -> str:
        if not self.on:
            return "Away mode is off."
        hours = (time.time() - self.since) / 3600
        locked = " The PC has locked itself." if self.locked_at else ""
        return f"Away mode has been on for {hours:.1f} h.{locked}"

    def _run(self) -> None:
        # The execution state belongs to THIS thread: set here, cleared here.
        if not self._keep_awake(True):
            self.log("AWAY: Windows refused to stay awake; away mode is not keeping it up.")
        try:
            while not self._stop.wait(self.poll_s):
                if not self.locked_at and self._idle() >= self.lock_after_s:
                    if self._lock():
                        self.locked_at = time.time()
                        self.log("AWAY: no input for a while - the PC is locked.")
                elif self.locked_at and self._idle() < self.poll_s:
                    self.locked_at = 0.0          # he is back: lock again next time he leaves
        finally:
            self._keep_awake(False)


__all__ = ["AwayMode", "LOCK_AFTER_S", "idle_seconds"]
