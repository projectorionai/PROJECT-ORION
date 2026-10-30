"""
ORION's presence in Windows - the pieces that need the operating system's ears.

Phase 1 of the Windows integration (Mark XXXVII.1), PRESENCE lane. The SHELL
lane registers ORION with Windows (Explorer menu, orion:// links, jump list);
this lane makes ORION notice and answer Windows:

  * ``GlobalHotkey``   - one chord, from any window, that reaches ORION.
  * ``SessionWatcher`` - the session locking and unlocking, the machine going
                         to sleep and waking, as events.
  * ``user_busy_state`` / ``should_stay_quiet`` - whether the person is in a
                         full-screen game, a presentation or Do Not Disturb.
  * ``ToastCenter``    - Windows notifications with buttons whose clicks come
                         back as single-use, expiring, verb-checked actions.
  * ``start_windows_presence`` - starts the above per the stored switches and
                         hands the lead one handle to hold and stop.

Three rules shape all of it.

1. THE QT THREAD IS THE EVENT LOOP. Under qasync, blocking the GUI thread stops
   the audio callback and the Live socket. So every Win32 wait (a message loop
   for the hotkey, another for the session window, the WinRT call that shows a
   toast) lives on its own daemon thread, and what it finds crosses back to the
   GUI thread through one queued Qt signal before any ORION code runs.

2. NOTHING HERE RAISES INTO ORION. A hotkey that Windows refuses, a machine with
   no WinRT, a shell that will not report its state: each is reported in words
   (``describe()``, a ``ToastReceipt``) and ORION carries on without it.

3. EVERY WIN32 / WINRT CALL IS INJECTABLE. ``user32``, ``kernel32``, ``shell32``,
   the message-window factory and the toast backend are constructor arguments,
   so the tests drive the whole of this module with fakes and never register a
   real hotkey, open a real window or show a real notification.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import secrets
import sys
import threading
import time
import weakref
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Sequence

from PyQt6.QtCore import QCoreApplication, QObject, Qt, QThread, pyqtSignal, pyqtSlot

from . import windows_settings
from .desktop_app import APP_USER_MODEL_ID

log = logging.getLogger("orion.windows_presence")

#: A callable that runs another callable on the Qt thread.
Dispatch = Callable[[Callable[[], None]], None]

IS_WINDOWS = sys.platform == "win32"

# ── Win32 constants (documented where they are used) ─────────────────────────

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN = 0x1, 0x2, 0x4, 0x8
MOD_NOREPEAT = 0x4000
WM_QUIT = 0x0012
WM_HOTKEY = 0x0312
WM_USER = 0x0400
PM_NOREMOVE = 0
WM_POWERBROADCAST = 0x0218
WM_WTSSESSION_CHANGE = 0x02B1
WTS_SESSION_LOCK, WTS_SESSION_UNLOCK = 0x7, 0x8
PBT_APMSUSPEND = 0x4
PBT_APMRESUMESUSPEND = 0x7          # resumed by the user (follows the automatic one)
PBT_APMRESUMEAUTOMATIC = 0x12       # the machine woke - always sent
NOTIFY_FOR_THIS_SESSION = 0
HWND_MESSAGE = -3
ERROR_HOTKEY_ALREADY_REGISTERED = 1409

#: Events the SessionWatcher reports, in the words ORION uses for them.
SESSION_EVENTS = ("locked", "unlocked", "suspending", "resumed")

DEFAULT_CHORD = "win+alt+o"
FALLBACK_CHORD = "ctrl+alt+o"
#: Mutes and unmutes the microphone from any window (Mark XXXX). On the
#: owner's PC both of the first two were taken, every launch since 26 Oct
#: 2026, and the microphone had no hotkey; the third is rarely claimed.
MIC_CHORD = "win+alt+m"
MIC_FALLBACK_CHORD = ("ctrl+alt+m", "ctrl+alt+shift+m")


# ══════════════════════════════════════════════════════════════════════════════
# CROSSING TO THE QT THREAD
# ══════════════════════════════════════════════════════════════════════════════

class _QtDispatcher(QObject):
    """Runs callables on the thread that owns it - always the GUI thread.

    One queued signal is the whole mechanism: emitting from any thread is
    safe, and the receiver's slot runs from the GUI thread's own event loop.
    """

    _post = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self._post.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @pyqtSlot(object)
    def _run(self, fn: Any) -> None:
        try:
            fn()
        except Exception:
            # A failing consumer must not take the dispatcher (and with it the
            # hotkey, the session watch and every toast action) down.
            log.exception("a callback queued to the Qt thread raised")


_dispatcher: _QtDispatcher | None = None
_dispatcher_app: Any = None
_dispatcher_lock = threading.Lock()


def _qt_dispatcher() -> _QtDispatcher | None:
    global _dispatcher, _dispatcher_app
    app = QCoreApplication.instance()
    if app is None:
        return None
    with _dispatcher_lock:
        current = _dispatcher
        if current is not None and _dispatcher_app is app:
            try:
                from PyQt6 import sip

                if not sip.isdeleted(current):
                    return current
            except Exception:
                pass
        created = _QtDispatcher()
        created.moveToThread(app.thread())
        _dispatcher, _dispatcher_app = created, app
        return created


def post_to_qt(fn: Callable[[], None]) -> None:
    """Run *fn* on the Qt thread, later. With no QCoreApplication (a script, a
    test) it runs at once, on the calling thread."""
    dispatcher = _qt_dispatcher()
    if dispatcher is None:
        fn()
        return
    dispatcher._post.emit(fn)


def call_on_qt(fn: Callable[[], None]) -> None:
    """Run *fn* on the Qt thread: now if already there, otherwise queued."""
    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() == app.thread():
        fn()
        return
    post_to_qt(fn)


# ══════════════════════════════════════════════════════════════════════════════
# CHORDS
# ══════════════════════════════════════════════════════════════════════════════

class ChordError(ValueError):
    """A chord string that cannot be a global hotkey."""


_MODIFIERS = {
    "win": MOD_WIN, "windows": MOD_WIN, "super": MOD_WIN, "meta": MOD_WIN,
    "ctrl": MOD_CONTROL, "control": MOD_CONTROL, "ctl": MOD_CONTROL,
    "alt": MOD_ALT, "shift": MOD_SHIFT,
}

#: Printed in this order: Win, Ctrl, Alt, Shift.
_MODIFIER_LABELS = ((MOD_WIN, "Win"), (MOD_CONTROL, "Ctrl"),
                    (MOD_ALT, "Alt"), (MOD_SHIFT, "Shift"))

_NAMED_KEYS = {
    "space": (0x20, "Space"), "enter": (0x0D, "Enter"), "return": (0x0D, "Enter"),
    "tab": (0x09, "Tab"), "esc": (0x1B, "Esc"), "escape": (0x1B, "Esc"),
    "backspace": (0x08, "Backspace"), "insert": (0x2D, "Insert"),
    "ins": (0x2D, "Insert"), "delete": (0x2E, "Delete"), "del": (0x2E, "Delete"),
    "home": (0x24, "Home"), "end": (0x23, "End"),
    "pageup": (0x21, "PageUp"), "pagedown": (0x22, "PageDown"),
    "left": (0x25, "Left"), "up": (0x26, "Up"),
    "right": (0x27, "Right"), "down": (0x28, "Down"),
    "pause": (0x13, "Pause"), "printscreen": (0x2C, "PrintScreen"),
}


@dataclass(frozen=True)
class Chord:
    """A parsed hotkey: Win32 modifier flags and a virtual-key code."""

    modifiers: int
    vk: int
    key_label: str

    @property
    def label(self) -> str:
        parts = [name for flag, name in _MODIFIER_LABELS if self.modifiers & flag]
        return "+".join(parts + [self.key_label])

    @property
    def text(self) -> str:
        """The settings spelling, e.g. "win+alt+o"."""
        return self.label.lower()


def parse_chord(text: str) -> Chord:
    """"win+alt+o" -> Chord. Raises ChordError for anything unusable.

    At least one modifier is required: a bare letter as a GLOBAL hotkey would
    swallow that letter in every application. Exactly one key. Letters, digits,
    F1-F24 and the named keys above are accepted; punctuation is not, because
    which key carries it depends on the keyboard layout.
    """
    parts = [part.strip().lower() for part in re.split(r"[+\s]+", str(text or "")) if part.strip()]
    if not parts:
        raise ChordError("the chord is empty")
    modifiers = 0
    key: tuple[int, str] | None = None
    for part in parts:
        if part in _MODIFIERS:
            flag = _MODIFIERS[part]
            if modifiers & flag:
                raise ChordError(f"{part!r} is repeated")
            modifiers |= flag
            continue
        if key is not None:
            raise ChordError("only one non-modifier key is allowed")
        if len(part) == 1 and "a" <= part <= "z":
            key = (ord(part.upper()), part.upper())
        elif len(part) == 1 and "0" <= part <= "9":
            key = (ord(part), part)
        elif re.fullmatch(r"f([1-9]|1\d|2[0-4])", part):
            number = int(part[1:])
            key = (0x70 + number - 1, f"F{number}")
        elif part in _NAMED_KEYS:
            key = _NAMED_KEYS[part]
        else:
            raise ChordError(f"{part!r} is not a key ORION can bind")
    if key is None:
        raise ChordError("the chord has no key, only modifiers")
    if not modifiers:
        raise ChordError("a global hotkey needs at least one modifier (Win, Ctrl, Alt or Shift)")
    return Chord(modifiers, key[0], key[1])


# ══════════════════════════════════════════════════════════════════════════════
# REAL WIN32 LIBRARIES (private instances, so prototypes set here never leak
# into anything else that uses ctypes.windll)
# ══════════════════════════════════════════════════════════════════════════════

_LRESULT = ctypes.c_ssize_t
_libs: dict[str, Any] = {}
_libs_lock = threading.Lock()


def _load_lib(name: str) -> Any:
    if not IS_WINDOWS:
        return None
    with _libs_lock:
        if name in _libs:
            return _libs[name]
        lib: Any = None
        try:
            lib = ctypes.WinDLL(name, use_last_error=True)
            _declare(name, lib)
        except Exception:
            lib = None
        _libs[name] = lib
        return lib


def _declare(name: str, lib: Any) -> None:
    """Argument and return types: without them a 64-bit WPARAM/LPARAM/HWND is
    silently truncated to a C int."""
    if name == "user32":
        P = ctypes.POINTER
        lib.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        lib.RegisterHotKey.restype = wintypes.BOOL
        lib.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        lib.UnregisterHotKey.restype = wintypes.BOOL
        lib.GetMessageW.argtypes = [P(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        lib.GetMessageW.restype = wintypes.BOOL
        lib.PeekMessageW.argtypes = [P(wintypes.MSG), wintypes.HWND, wintypes.UINT,
                                     wintypes.UINT, wintypes.UINT]
        lib.PeekMessageW.restype = wintypes.BOOL
        lib.DispatchMessageW.argtypes = [P(wintypes.MSG)]
        lib.DispatchMessageW.restype = _LRESULT
        lib.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT,
                                           wintypes.WPARAM, wintypes.LPARAM]
        lib.PostThreadMessageW.restype = wintypes.BOOL
        lib.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        lib.DefWindowProcW.restype = _LRESULT
        lib.DestroyWindow.argtypes = [wintypes.HWND]
        lib.DestroyWindow.restype = wintypes.BOOL
        lib.CreateWindowExW.argtypes = [
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
        lib.CreateWindowExW.restype = wintypes.HWND
        lib.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        lib.UnregisterClassW.restype = wintypes.BOOL
    elif name == "kernel32":
        lib.GetCurrentThreadId.argtypes = []
        lib.GetCurrentThreadId.restype = wintypes.DWORD
        lib.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        lib.GetModuleHandleW.restype = wintypes.HMODULE
    elif name == "wtsapi32":
        lib.WTSRegisterSessionNotification.argtypes = [wintypes.HWND, wintypes.DWORD]
        lib.WTSRegisterSessionNotification.restype = wintypes.BOOL
        lib.WTSUnRegisterSessionNotification.argtypes = [wintypes.HWND]
        lib.WTSUnRegisterSessionNotification.restype = wintypes.BOOL
    elif name == "shell32":
        lib.SHQueryUserNotificationState.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.SHQueryUserNotificationState.restype = ctypes.c_long


def _last_error() -> int:
    try:
        return int(ctypes.get_last_error())
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# THE GLOBAL HOTKEY
# ══════════════════════════════════════════════════════════════════════════════

class GlobalHotkey:
    """One chord, registered with Windows, pressed from any window.

    ``RegisterHotKey`` - not a keyboard hook - because Windows itself decides
    when the chord is down, so it needs no elevation, never sees what else is
    typed, and costs nothing while idle. (Push-to-talk polls key state instead
    because it needs the RELEASE too; this needs only the press.)

    The registration belongs to the thread that makes it, so a dedicated daemon
    thread registers the chord, then blocks in ``GetMessageW`` until a
    ``WM_HOTKEY`` arrives or ``stop()`` posts ``WM_QUIT``. ``MOD_NOREPEAT``
    keeps a held chord from firing again and again.

    If Windows refuses the requested chord (another program owns it) the
    fallback is tried and ``used_fallback`` says so; if both are refused the
    hotkey is simply absent and ``describe()`` says why.
    """

    def __init__(self, callback: Callable[[], None], *, chord: str = DEFAULT_CHORD,
                 fallback: str | Sequence[str] | None = FALLBACK_CHORD,
                 dispatch: Dispatch | None = None,
                 user32: Any = None, kernel32: Any = None, hotkey_id: int = 0x4F52,
                 start_timeout: float = 3.0, min_interval: float = 0.25) -> None:
        self._callback = callback
        self._requested_text = str(chord or "")
        self._fallback_text = fallback
        self._dispatch = dispatch or post_to_qt
        self._user32 = user32
        self._kernel32 = kernel32
        self._id = int(hotkey_id)
        self._start_timeout = float(start_timeout)
        self._min_interval = float(min_interval)

        self.requested: Chord | None = None
        self.active_chord: Chord | None = None
        self.used_fallback = False
        self.error = ""
        self._parse_error = ""
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._last_fired = 0.0
        self._abandoned = False

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self.active_chord is not None)

    @property
    def label(self) -> str:
        """The chord that is actually live ("" when none)."""
        return self.active_chord.label if self.active_chord else ""

    def describe(self) -> str:
        if self.active_chord is not None and self.running:
            if self.used_fallback and self.requested is not None:
                return (f"Global hotkey: {self.active_chord.label} works in any window "
                        f"({self.requested.label} is taken by Windows or another app).")
            if self.used_fallback:
                return (f"Global hotkey: {self.active_chord.label} works in any window "
                        f"(the chosen chord was not usable: {self._parse_error}).")
            return f"Global hotkey: {self.active_chord.label} works in any window."
        if self.error:
            return f"Global hotkey unavailable: {self.error}"
        return "Global hotkey is not running."

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Register the chord. True when one is live. Never raises."""
        if self.running:
            return True
        self.error = ""
        self._parse_error = ""
        try:
            self.requested = parse_chord(self._requested_text)
        except ChordError as exc:
            self.requested = None
            self._parse_error = f"{self._requested_text!r}: {exc}"
            self.error = self._parse_error
        candidates: list[Chord] = [self.requested] if self.requested else []
        fallbacks = ([self._fallback_text] if isinstance(self._fallback_text, str)
                     else list(self._fallback_text or ()))
        for text in fallbacks:
            try:
                fallback = parse_chord(text)
                if fallback not in candidates:
                    candidates.append(fallback)
            except ChordError:
                pass
        if not candidates:
            return False
        if self._user32 is None:
            self._user32 = _load_lib("user32")
        if self._kernel32 is None:
            self._kernel32 = _load_lib("kernel32")
        if self._user32 is None or self._kernel32 is None:
            self.error = self.error or "global hotkeys are a Windows feature"
            return False

        self._ready.clear()
        self._stop.clear()
        self._abandoned = False
        self.active_chord = None
        self.used_fallback = False
        self._thread = threading.Thread(target=self._run, args=(candidates,),
                                        name="orion-hotkey", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._start_timeout):
            self._abandoned = True              # the thread unregisters itself if it ever wakes
            self.error = self.error or "Windows did not answer the registration in time"
            return False
        return self.active_chord is not None

    def stop(self) -> None:
        """Unregister and end the thread. Safe to call twice, or before start."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            try:
                if self._user32 is not None and self._thread_id:
                    self._user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass
            thread.join(timeout=2.0)
        self.active_chord = None

    # ── the message-loop thread ──────────────────────────────────────────────

    def _run(self, candidates: Sequence[Chord]) -> None:
        user32 = self._user32
        msg = wintypes.MSG()
        registered = False
        try:
            self._thread_id = int(self._kernel32.GetCurrentThreadId())
            # Peek once so the thread owns a message queue before anyone can
            # post WM_QUIT to it.
            user32.PeekMessageW(ctypes.pointer(msg), None, WM_USER, WM_USER, PM_NOREMOVE)
            refused: list[str] = []
            for index, chord in enumerate(candidates):
                if user32.RegisterHotKey(None, self._id, chord.modifiers | MOD_NOREPEAT, chord.vk):
                    self.active_chord = chord
                    self.used_fallback = chord is not self.requested
                    registered = True
                    break
                code = _last_error()
                reason = ("already taken by another program"
                          if code == ERROR_HOTKEY_ALREADY_REGISTERED
                          else f"Windows refused it (error {code})" if code else "Windows refused it")
                refused.append(f"{chord.label}: {reason}")
                # Every chord tried, not only the last: "Ctrl+Alt+M is taken"
                # hid that Win+Alt+M had been refused as well.
                self.error = "; ".join(refused)
            if registered:
                self.error = ""
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()
        if not registered:
            return
        try:
            while not self._stop.is_set() and not self._abandoned:
                result = user32.GetMessageW(ctypes.pointer(msg), None, 0, 0)
                if result <= 0:                  # 0 = WM_QUIT, -1 = error
                    break
                if msg.message == WM_HOTKEY and msg.wParam == self._id:
                    self._fire()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                user32.UnregisterHotKey(None, self._id)     # the registering thread must do this
            except Exception:
                pass
            self.active_chord = None

    def _fire(self) -> None:
        now = time.monotonic()
        if now - self._last_fired < self._min_interval:
            return
        self._last_fired = now
        callback = self._callback

        def deliver() -> None:
            try:
                callback()
            except Exception:
                log.exception("the hotkey callback raised")

        try:
            self._dispatch(deliver)
        except Exception:
            log.exception("the hotkey could not be handed to the Qt thread")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION AND POWER EVENTS
# ══════════════════════════════════════════════════════════════════════════════

def map_session_message(msg: int, wparam: int) -> str | None:
    """The ORION event for a window message, or None when it is not one.

      WM_WTSSESSION_CHANGE  LOCK -> locked, UNLOCK -> unlocked
      WM_POWERBROADCAST     APMSUSPEND -> suspending;
                            APMRESUMEAUTOMATIC / APMRESUMESUSPEND -> resumed
    """
    if msg == WM_WTSSESSION_CHANGE:
        return {WTS_SESSION_LOCK: "locked", WTS_SESSION_UNLOCK: "unlocked"}.get(int(wparam))
    if msg == WM_POWERBROADCAST:
        return {PBT_APMSUSPEND: "suspending", PBT_APMRESUMEAUTOMATIC: "resumed",
                PBT_APMRESUMESUSPEND: "resumed"}.get(int(wparam))
    return None


class _MessageWindow:
    """A message-only window (invisible, no taskbar, no paint) that Windows
    sends session and power notifications to. All of it runs on ONE thread: the
    one that calls ``open()``, ``run()`` and ``close()``."""

    def __init__(self, handler: Callable[[int, int, int, int], int | None]) -> None:
        self._handler = handler
        self._hwnd: Any = None
        self._thread_id = 0
        self._class_name = f"OrionSessionWatcher{id(self):x}"
        self._hinstance: Any = None
        self._proc: Any = None          # kept alive: Windows holds a bare pointer to it
        self._wts_registered = False
        self._user32 = _load_lib("user32")
        self._kernel32 = _load_lib("kernel32")
        self._wtsapi32 = _load_lib("wtsapi32")

    def open(self) -> bool:
        user32, kernel32 = self._user32, self._kernel32
        if user32 is None or kernel32 is None:
            return False
        WNDPROC = ctypes.WINFUNCTYPE(_LRESULT, wintypes.HWND, wintypes.UINT,
                                     wintypes.WPARAM, wintypes.LPARAM)

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                        ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                        ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HANDLE),
                        ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HANDLE),
                        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

        handler = self._handler
        default_proc = user32.DefWindowProcW

        def proc(hwnd: Any, msg: int, wparam: int, lparam: int) -> int:
            try:
                result = handler(int(msg), int(wparam), int(lparam), hwnd)
            except Exception:
                log.exception("the session window procedure raised")
                result = None
            if result is None:
                return default_proc(hwnd, msg, wparam, lparam)
            return result

        self._proc = WNDPROC(proc)
        self._thread_id = int(kernel32.GetCurrentThreadId())
        self._hinstance = kernel32.GetModuleHandleW(None)
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
        user32.RegisterClassW.restype = wintypes.ATOM
        window_class = WNDCLASSW()
        window_class.lpfnWndProc = self._proc
        window_class.hInstance = self._hinstance
        window_class.lpszClassName = self._class_name
        if not user32.RegisterClassW(ctypes.byref(window_class)):
            return False
        self._hwnd = user32.CreateWindowExW(
            0, self._class_name, "ORION Session Watcher", 0, 0, 0, 0, 0,
            wintypes.HWND(HWND_MESSAGE), None, self._hinstance, None)
        if not self._hwnd:
            user32.UnregisterClassW(self._class_name, self._hinstance)
            return False
        if self._wtsapi32 is not None:
            self._wts_registered = bool(self._wtsapi32.WTSRegisterSessionNotification(
                self._hwnd, NOTIFY_FOR_THIS_SESSION))
        return True

    def run(self) -> None:
        user32 = self._user32
        msg = wintypes.MSG()
        while True:
            result = user32.GetMessageW(ctypes.pointer(msg), None, 0, 0)
            if result <= 0:
                break
            user32.DispatchMessageW(ctypes.pointer(msg))

    def post_quit(self) -> None:
        if self._user32 is not None and self._thread_id:
            self._user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

    def close(self) -> None:
        user32 = self._user32
        try:
            if self._wts_registered and self._wtsapi32 is not None:
                self._wtsapi32.WTSUnRegisterSessionNotification(self._hwnd)
        except Exception:
            pass
        try:
            if self._hwnd:
                user32.DestroyWindow(self._hwnd)
            user32.UnregisterClassW(self._class_name, self._hinstance)
        except Exception:
            pass
        self._hwnd = None


class SessionWatcher:
    """Reports the session locking, unlocking, and the machine sleeping/waking.

    ``callback(event)`` receives one of SESSION_EVENTS, on the Qt thread.
    ``handle_message`` is the whole of the decision logic, separate from the
    Win32 plumbing so it can be driven directly: the window procedure only
    forwards to it.
    """

    def __init__(self, callback: Callable[[str], None], *, dispatch: Dispatch | None = None,
                 window_factory: Callable[[Callable[..., Any]], Any] | None = None,
                 start_timeout: float = 3.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_resumed_at: float | None = None
        self._callback = callback
        self._dispatch = dispatch or post_to_qt
        self._factory = window_factory or _MessageWindow
        self._start_timeout = float(start_timeout)
        self._thread: threading.Thread | None = None
        self._window: Any = None
        self._ready = threading.Event()
        self._opened = False
        self.last_event = ""
        self.locked = False
        self.error = ""

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._opened)

    def describe(self) -> str:
        if self.running:
            return "Session watch: ORION notices lock, unlock, sleep and wake."
        return f"Session watch unavailable: {self.error}" if self.error else "Session watch is not running."

    # ── the decision, window-procedure shaped ────────────────────────────────

    def handle_message(self, msg: int, wparam: int = 0, lparam: int = 0, hwnd: Any = None) -> int | None:
        """Returns the window-procedure result, or None to let Windows default
        it. Anything that is not a session/power event is left alone."""
        event = map_session_message(msg, wparam)
        if event is None:
            return None
        if event == "resumed":
            # Windows sends RESUMEAUTOMATIC and, if the user then touches the
            # machine, RESUMESUSPEND a moment later: one wake, one event. By
            # TIME, not "previous event", because a machine on modern standby
            # may never report the suspend in between.
            now = self._clock()
            if self._last_resumed_at is not None and now - self._last_resumed_at < 10.0:
                return 1
            self._last_resumed_at = now
        self.last_event = event
        if event == "locked":
            self.locked = True
        elif event == "unlocked":
            self.locked = False
        callback = self._callback

        def deliver(name: str = event) -> None:
            try:
                callback(name)
            except Exception:
                log.exception("the session-event callback raised")

        try:
            self._dispatch(deliver)
        except Exception:
            log.exception("a session event could not be handed to the Qt thread")
        return 1 if msg == WM_POWERBROADCAST else 0         # TRUE: broadcast accepted

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if self.running:
            return True
        self.error = ""
        self._ready.clear()
        self._opened = False
        self._thread = threading.Thread(target=self._run, name="orion-session", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._start_timeout):
            self.error = self.error or "Windows did not answer in time"
            return False
        return self._opened

    def stop(self) -> None:
        thread, self._thread = self._thread, None
        window = self._window
        if thread is not None and thread.is_alive() and window is not None:
            try:
                window.post_quit()
            except Exception:
                pass
            thread.join(timeout=2.0)
        self._opened = False

    def _run(self) -> None:
        window: Any = None
        try:
            window = self._factory(self.handle_message)
            self._window = window
            self._opened = bool(window.open())
            if not self._opened:
                self.error = "the message window could not be created"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._opened = False
        finally:
            self._ready.set()
        if not self._opened or window is None:
            return
        try:
            window.run()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                window.close()
            except Exception:
                pass
            self._opened = False


# ══════════════════════════════════════════════════════════════════════════════
# IS THE PERSON BUSY?
# ══════════════════════════════════════════════════════════════════════════════

#: SHQueryUserNotificationState -> ORION's word.
#:   1 QUNS_NOT_PRESENT            screen saver, locked, fast user switch -> busy
#:   2 QUNS_BUSY                   a full-screen application              -> fullscreen_app
#:   3 QUNS_RUNNING_D3D_FULL_SCREEN a full-screen Direct3D game            -> fullscreen_app
#:   4 QUNS_PRESENTATION_MODE      presentation settings applied          -> presentation
#:   5 QUNS_ACCEPTS_NOTIFICATIONS  nothing in the way                     -> available
#:   6 QUNS_QUIET_TIME             Do Not Disturb / Focus                 -> quiet_hours
#:   7 QUNS_APP                    a full-screen Store app                -> fullscreen_app
BUSY_STATES = ("available", "busy", "fullscreen_app", "presentation", "quiet_hours", "unknown")
_QUNS = {1: "busy", 2: "fullscreen_app", 3: "fullscreen_app", 4: "presentation",
         5: "available", 6: "quiet_hours", 7: "fullscreen_app"}

#: States in which ORION should not interrupt (speak, pop up, greet).
QUIET_STATES = frozenset({"busy", "fullscreen_app", "presentation", "quiet_hours"})

#: States in which a docked face gets out of the way.
HIDE_DOCK_STATES = frozenset({"fullscreen_app", "presentation"})


def busy_state_from_quns(code: Any) -> str:
    try:
        return _QUNS.get(int(code), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def user_busy_state(shell32: Any = None) -> str:
    """One of BUSY_STATES. Never raises; "unknown" when Windows will not say."""
    try:
        lib = shell32 if shell32 is not None else _load_lib("shell32")
        if lib is None:
            return "unknown"
        out = ctypes.c_int(0)
        result = lib.SHQueryUserNotificationState(ctypes.pointer(out))
        if result != 0:                         # a failed HRESULT
            return "unknown"
        state = busy_state_from_quns(out.value)
        if out.value == 2 and shell32 is None and not foreground_covers_monitor():
            # QUNS_BUSY is reported far more often than a full-screen app is
            # running: measured 24 Oct on the owner's PC with only ordinary
            # maximised windows open. Taken at its word, the dock hid and
            # ORION stayed quiet all day. Believe it only when the window in
            # front really fills its monitor.
            return "available"
        return state
    except Exception:
        return "unknown"


def foreground_covers_monitor(user32: Any = None) -> bool:
    """True when the foreground window fills its whole monitor (taskbar
    included) and is not the desktop or one of ORION's own windows. Never
    raises; False when Windows will not say."""
    try:
        from ctypes import wintypes
        lib = user32 if user32 is not None else ctypes.windll.user32
        hwnd = lib.GetForegroundWindow()
        if not hwnd:
            return False
        name = ctypes.create_unicode_buffer(64)
        lib.GetClassNameW(hwnd, name, 64)
        if name.value in {"Progman", "WorkerW", "Shell_TrayWnd"}:
            return False
        if lib.IsZoomed(hwnd):
            # Maximised is not full-screen. On a monitor with no taskbar a
            # maximised window covers it too; real full-screen apps (games,
            # video, F11) are borderless windows sized to the monitor.
            return False
        pid = wintypes.DWORD(0)
        lib.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == os.getpid():
            return False
        rect = wintypes.RECT()
        if not lib.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False

        class _MonitorInfo(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        monitor = lib.MonitorFromWindow(hwnd, 2)          # MONITOR_DEFAULTTONEAREST
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        if not monitor or not lib.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return False
        screen = info.rcMonitor
        return (rect.left <= screen.left and rect.top <= screen.top
                and rect.right >= screen.right and rect.bottom >= screen.bottom)
    except Exception:
        return False


def should_stay_quiet(state: str | None = None, *, shell32: Any = None) -> bool:
    """True when ORION should hold his tongue: the person has asked for that
    (``quiet_when_busy``) AND Windows says they are busy. Safety alerts do not
    consult this - they are for danger, not for conversation."""
    try:
        if not windows_settings.get("quiet_when_busy"):
            return False
    except Exception:
        return False
    return (state or user_busy_state(shell32)) in QUIET_STATES


# ══════════════════════════════════════════════════════════════════════════════
# TOASTS WITH BUTTONS
# ══════════════════════════════════════════════════════════════════════════════

TOAST_SCHEME = "orion://toast"
MAX_ACTIONS = 5                     # Windows shows at most five buttons
MAX_PENDING = 32
_VERB = re.compile(r"[A-Za-z0-9_.-]{1,32}\Z")
_XML_BAD = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _xml_text(text: Any, limit: int) -> str:
    """Text safe for an XML text node or attribute: control characters that XML
    1.0 forbids are dropped, the five markup characters are escaped."""
    clean = _XML_BAD.sub("", str(text or "")).strip()
    if len(clean) > limit:
        clean = clean[:limit - 1].rstrip() + "…"
    return (clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def toast_url(token: str, verb: str) -> str:
    return f"{TOAST_SCHEME}?t={token}&a={verb}"


def build_toast_xml(title: str, body: str, actions: Sequence[tuple[str, str]],
                    token: str | None, default_verb: str | None = None) -> str:
    """ToastGeneric XML. Every piece of text is escaped; buttons activate by
    PROTOCOL, so a click starts ``orion://toast?t=<token>&a=<verb>`` - the shell
    lane turns that into an action for the running ORION."""
    launch = ""
    if token and default_verb:
        launch = (f' launch="{_xml_text(toast_url(token, default_verb), 200)}"'
                  ' activationType="protocol"')
    parts = [f"<toast{launch}><visual><binding template=\"ToastGeneric\">",
             f"<text>{_xml_text(title, 120)}</text>"]
    if body:
        parts.append(f"<text>{_xml_text(body, 400)}</text>")
    parts.append("</binding></visual>")
    if token and actions:
        parts.append("<actions>")
        for label, verb in actions[:MAX_ACTIONS]:
            parts.append(
                f'<action content="{_xml_text(label, 40)}" '
                f'arguments="{_xml_text(toast_url(token, verb), 200)}" '
                f'activationType="protocol"/>')
        parts.append("</actions>")
    parts.append("</toast>")
    return "".join(parts)


class ToastDisabled(RuntimeError):
    """Windows has notifications for ORION switched off. Not retried as a
    balloon: the person's choice in Windows settings stands."""


class ToastUnavailable(RuntimeError):
    """Windows cannot show a toast for ORION right now (for instance his
    identity is not registered). The message says why; a tray balloon is used."""


#: HRESULT 0x80070490 - what the notification platform answers when the
#: AppUserModelID has no Start-menu shortcut carrying it.
_E_NOT_FOUND = 0x80070490


class WinrtToastBackend:
    """The real thing: WinRT ToastNotificationManager for ORION's AUMID."""

    def __init__(self, aumid: str = APP_USER_MODEL_ID) -> None:
        self._aumid = aumid

    def available(self) -> tuple[bool, str]:
        if not IS_WINDOWS:
            return False, "Windows notifications need Windows"
        try:
            import winrt.windows.data.xml.dom  # noqa: F401
            import winrt.windows.ui.notifications  # noqa: F401
        except Exception as exc:
            return False, f"the WinRT notification packages are missing ({type(exc).__name__})"
        return True, ""

    def show(self, xml: str, *, tag: str, expires_s: float) -> None:
        from winrt.windows.data.xml.dom import XmlDocument
        from winrt.windows.ui.notifications import (
            NotificationSetting, ToastNotification, ToastNotificationManager)

        notifier = ToastNotificationManager.create_toast_notifier_with_id(self._aumid)
        setting = self._read_setting(notifier)
        if setting is not None and setting != NotificationSetting.ENABLED:
            raise ToastDisabled("notifications for ORION are switched off in Windows settings")
        document = XmlDocument()
        document.load_xml(xml)
        toast = ToastNotification(document)
        try:
            toast.tag = tag[:64]
            toast.group = "orion"
            toast.expiration_time = datetime.now(timezone.utc) + timedelta(seconds=expires_s)
        except Exception:
            pass                                # cosmetic: grouping and auto-removal
        notifier.show(toast)

    def _read_setting(self, notifier: Any) -> Any:
        """The notifier's NotificationSetting, or None when it will not say.

        "Element not found" is the platform's definite answer that this
        AppUserModelID is not registered - a toast would be dropped silently,
        so it is reported as unavailable instead of pretended to succeed.
        """
        try:
            return notifier.setting
        except OSError as exc:
            code = (getattr(exc, "winerror", None) or 0) & 0xFFFFFFFF
            if code == _E_NOT_FOUND:
                raise ToastUnavailable(
                    "Windows does not know ORION's notification identity yet - his "
                    "Start-menu shortcut needs refreshing") from exc
            return None
        except Exception:
            return None


@dataclass
class ToastReceipt:
    """What happened to one ``ToastCenter.show``. Filled in as it happens: a
    toast is shown on a worker thread, so ``via`` may still say "toast" while
    the delivery is in flight and flip to "balloon" if Windows refuses it."""

    token: str | None = None
    via: str = "pending"            # "toast" | "balloon" | "none"
    delivered: bool = False
    buttons: bool = False
    reason: str = ""

    def summary(self) -> str:
        if self.via == "toast":
            return "Windows notification" + (" with buttons" if self.buttons else "")
        if self.via == "balloon":
            return f"tray balloon, without buttons ({self.reason})"
        if self.via == "none":
            return f"not shown ({self.reason})"
        return "pending"


@dataclass
class _Pending:
    verbs: frozenset
    callback: Callable[[str], None]
    expires: float


class ToastCenter:
    """Actionable Windows notifications whose button clicks come back safely.

    Why tokens. A button starts ``orion://toast?t=<token>&a=<verb>``, a URL any
    program (or web page) on the machine could also start. So a click does
    nothing by itself: ORION only acts when the token is one it issued, for
    exactly this notification, has not yet been used, has not expired, and the
    verb is one of those the notification actually offered. The token is 24
    random bytes, single-use, and forgotten after ``expires_s``.
    """

    def __init__(self, *, dispatch: Dispatch | None = None, backend: Any = None,
                 shortcut_check: Callable[[], bool] | None = None, tray: Any = None,
                 run_async: Callable[[Callable[[], None]], None] | None = None,
                 say: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._dispatch = dispatch or call_on_qt
        self._backend = backend if backend is not None else WinrtToastBackend()
        self._shortcut_check = shortcut_check or _start_menu_shortcut_exists
        self._tray = tray
        self._run_async = run_async or _run_in_thread
        self._say = say
        self._clock = clock
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()
        self.last_receipt: ToastReceipt | None = None

    def set_tray(self, tray: Any) -> None:
        """The tray icon whose balloon is the fallback (``OrionTray``)."""
        self._tray = tray

    # ── showing ──────────────────────────────────────────────────────────────

    def show(self, title: str, body: str = "", actions: Iterable[tuple[str, str]] = (),
             on_action: Callable[[str], None] | None = None, *, expires_s: float = 600.0,
             default_verb: str | None = None) -> ToastReceipt:
        """Show a notification. ``actions`` is ``[(label, verb), ...]`` (verbs are
        letters, digits and ``_ . -``); ``on_action(verb)`` runs on the Qt thread,
        once, when a button (or the toast itself, for ``default_verb``) is
        clicked. Returns at once with a receipt; never raises."""
        receipt = ToastReceipt()
        self.last_receipt = receipt
        try:
            return self._show(receipt, title, body, actions, on_action, expires_s, default_verb)
        except Exception as exc:                      # a notification must never break a caller
            log.exception("showing a notification failed")
            receipt.via, receipt.delivered = "none", False
            receipt.reason = f"{type(exc).__name__}"
            return receipt

    def _show(self, receipt: ToastReceipt, title: str, body: str,
              actions: Iterable[tuple[str, str]], on_action: Any, expires_s: float,
              default_verb: str | None) -> ToastReceipt:
        title_text = str(title or "").strip() or "ORION"
        body_text = str(body or "").strip()
        offered = _clean_actions(actions) if callable(on_action) else []
        default = default_verb if (callable(on_action) and default_verb
                                   and _VERB.match(str(default_verb))) else None
        why_not = self._why_not_a_toast()
        if why_not:
            return self._balloon(receipt, title_text, body_text, why_not)

        token = None
        if offered or default:
            verbs = {verb for _label, verb in offered} | ({default} if default else set())
            token = self._issue(verbs, on_action, expires_s)
        xml = build_toast_xml(title_text, body_text, offered, token, default)
        receipt.token = token
        receipt.via = "toast"
        receipt.buttons = bool(offered)

        def deliver() -> None:
            self._deliver(receipt, xml, token, expires_s, title_text, body_text)

        self._run_async(deliver)
        return receipt

    def _why_not_a_toast(self) -> str:
        """"" when a real toast can be shown, otherwise the reason in words."""
        try:
            if not windows_settings.get("toasts"):
                return "notifications with buttons are switched off in Preferences"
        except Exception:
            pass
        ok, reason = self._backend.available()
        if not ok:
            return reason or "Windows notifications are unavailable"
        try:
            if not self._shortcut_check():
                return ("ORION has no Start-menu shortcut yet, and Windows only shows "
                        "notifications for apps that have one")
        except Exception:
            return "ORION's Start-menu shortcut could not be checked"
        return ""

    def _deliver(self, receipt: ToastReceipt, xml: str, token: str | None,
                 expires_s: float, title: str, body: str) -> None:
        """Worker thread: the WinRT call, then the fallback if it fails."""
        try:
            self._backend.show(xml, tag=(token or secrets.token_urlsafe(6))[:16],
                               expires_s=float(expires_s))
            receipt.delivered = True
        except ToastDisabled as exc:
            self.revoke(token)
            receipt.via, receipt.delivered, receipt.buttons = "none", False, False
            receipt.reason = str(exc)
            self._tell(f"[ORION] NOTIFY: {exc}.")
        except ToastUnavailable as exc:
            self.revoke(token)
            self._balloon(receipt, title, body, str(exc))
        except Exception as exc:
            self.revoke(token)
            self._balloon(receipt, title, body,
                          f"Windows would not show the notification ({type(exc).__name__})")

    def _balloon(self, receipt: ToastReceipt, title: str, body: str, reason: str) -> ToastReceipt:
        """The fallback: a plain tray balloon - no buttons, and it says so."""
        receipt.buttons = False
        receipt.token = None
        tray = self._tray
        if tray is None:
            receipt.via, receipt.delivered = "none", False
            receipt.reason = f"{reason}; no tray icon to fall back to"
            self._tell(f"[ORION] NOTIFY: {receipt.summary()}.")
            return receipt
        receipt.via = "balloon"
        receipt.reason = reason

        def show_balloon() -> None:
            try:
                notify = getattr(tray, "notify", None)
                receipt.delivered = bool(notify(title, body)) if callable(notify) else False
            except Exception:
                receipt.delivered = False

        try:
            self._dispatch(show_balloon)
        except Exception:
            receipt.delivered = False
        self._tell(f"[ORION] NOTIFY: using a tray balloon without buttons - {reason}.")
        return receipt

    def _tell(self, text: str) -> None:
        log.info(text)
        if self._say is not None:
            try:
                self._say(text)
            except Exception:
                pass

    # ── tokens ───────────────────────────────────────────────────────────────

    def _issue(self, verbs: set[str], callback: Callable[[str], None], expires_s: float) -> str:
        token = secrets.token_urlsafe(24)
        lifetime = max(1.0, min(float(expires_s or 600.0), 24 * 3600.0))
        with self._lock:
            self._prune_locked()
            while len(self._pending) >= MAX_PENDING:
                del self._pending[next(iter(self._pending))]          # forget the oldest
            self._pending[token] = _Pending(frozenset(verbs), callback, self._clock() + lifetime)
        return token

    def _prune_locked(self) -> None:
        now = self._clock()
        for token in [t for t, entry in self._pending.items() if entry.expires <= now]:
            del self._pending[token]

    def revoke(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._pending.pop(token, None)

    def pending_count(self) -> int:
        with self._lock:
            self._prune_locked()
            return len(self._pending)

    def handle_action(self, token: Any, verb: Any) -> bool:
        """A click arrived (from ``orion://toast?...``). True when it was a
        valid, first use of a live token with an offered verb - and the stored
        callback has been handed to the Qt thread. Anything else changes nothing
        and returns False."""
        if not isinstance(token, str) or not isinstance(verb, str):
            return False
        if not token or len(token) > 64 or not _VERB.match(verb):
            return False
        with self._lock:
            self._prune_locked()
            entry = self._pending.get(token)
            if entry is None or verb not in entry.verbs:
                return False
            del self._pending[token]                    # single use, decided under the lock
        callback = entry.callback

        def run() -> None:
            try:
                callback(verb)
            except Exception:
                log.exception("a notification action callback raised")

        try:
            self._dispatch(run)
        except Exception:
            log.exception("a notification action could not be handed to the Qt thread")
            return False
        return True

    def describe(self) -> str:
        why = self._why_not_a_toast()
        if not why:
            return "Notifications: Windows notifications with buttons."
        return f"Notifications: tray balloons without buttons ({why})."


def _clean_actions(actions: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    clean: list[tuple[str, str]] = []
    for item in actions or ():
        try:
            label, verb = item
        except (TypeError, ValueError):
            continue
        label, verb = str(label or "").strip(), str(verb or "")
        if label and _VERB.match(verb) and all(verb != v for _l, v in clean):
            clean.append((label, verb))
        if len(clean) >= MAX_ACTIONS:
            break
    return clean


def _run_in_thread(fn: Callable[[], None]) -> None:
    """WinRT is called off the GUI thread: the GUI thread is an STA, and a
    projection call from there can fail with a changed-apartment error."""
    threading.Thread(target=fn, name="orion-toast", daemon=True).start()


_shortcut_cache: tuple[float, bool] = (0.0, False)


def _start_menu_shortcut_exists() -> bool:
    """Windows shows an unpackaged app's notifications only if a Start-menu
    shortcut carrying its AppUserModelID exists. desktop_app writes that."""
    global _shortcut_cache
    stamp, value = _shortcut_cache
    if time.monotonic() - stamp < 30.0 and stamp:
        return value
    try:
        from . import desktop_app

        value = bool(desktop_app.status().get("start_menu_shortcut"))
    except Exception:
        value = False
    _shortcut_cache = (time.monotonic(), value)
    return value


def shortcut_aumid() -> str | None:
    """The AppUserModelID stamped on ORION's Start-menu shortcut, or None.
    Uses COM, so call it from a worker thread, never the Qt thread."""
    try:
        import pythoncom
        from win32com.propsys import propsys, pscon
        from win32com.shell import shell

        from . import desktop_app

        folder = desktop_app._start_menu_dir()
        link_path = folder / f"{desktop_app.SHORTCUT_NAME}.lnk" if folder else None
        if link_path is None or not link_path.exists():
            return None
        pythoncom.CoInitialize()
        try:
            link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None,
                                              pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
            link.QueryInterface(pythoncom.IID_IPersistFile).Load(str(link_path), 0)
            store = link.QueryInterface(propsys.IID_IPropertyStore)
            return str(store.GetValue(pscon.PKEY_AppUserModel_ID).GetValue() or "") or None
        finally:
            pythoncom.CoUninitialize()
    except Exception:
        return None


def ensure_toast_identity(*, read_aumid: Callable[[], str | None] | None = None,
                          install: Callable[[], Any] | None = None) -> str:
    """Make sure Windows can match ORION's notifications to him.

    Toasts for an unpackaged app are shown only when a Start-menu shortcut
    carries the app's AppUserModelID. A shortcut written by an older build can
    carry a different one (this checkout found "Owner.ORION.Assistant" where
    the code says "ProjectORION.Assistant"), in which case Windows drops every
    toast. This re-stamps the shortcut when it does not match - the same call
    desktop_app already uses to install it. Returns "ok", "refreshed" or
    "failed: <why>". Call from a worker thread; it touches the Start menu, so
    it is opt-in (``start_windows_presence(refresh_identity=True)``).
    """
    try:
        current = (read_aumid or shortcut_aumid)()
        if current == APP_USER_MODEL_ID:
            return "ok"
        if install is None:
            from . import desktop_app

            def install() -> Any:
                return desktop_app.install_shortcuts(desktop=False, start_menu=True)

        result = install()
        refreshed = bool(result.get("start_menu")) if isinstance(result, dict) else bool(result)
        global _shortcut_cache
        _shortcut_cache = (0.0, False)                      # look again, do not trust the old answer
        return "refreshed" if refreshed else "failed: the shortcut could not be written"
    except Exception as exc:
        return f"failed: {type(exc).__name__}"


# ══════════════════════════════════════════════════════════════════════════════
# THE HANDLE THE LEAD HOLDS
# ══════════════════════════════════════════════════════════════════════════════

_active: "weakref.ReferenceType[PresenceHandle] | None" = None


def active_hotkey_text() -> str:
    """For Preferences: what the running hotkey is, or "" when none is."""
    handle = _active() if _active is not None else None
    if handle is None or handle.hotkey is None:
        return ""
    return handle.hotkey.describe()


class PresenceHandle:
    """Everything start_windows_presence started, and how to stop it."""

    def __init__(self, bus: Any, toasts: ToastCenter, hotkey: GlobalHotkey | None,
                 watcher: SessionWatcher | None, palette: Any = None) -> None:
        self.bus = bus
        self.toasts = toasts
        self.hotkey = hotkey
        self.watcher = watcher
        self.mic_hotkey: GlobalHotkey | None = None
        self._palette = _palette_tuple(palette)
        self._docks: list[Any] = []
        self._stopped = False
        self._user_session_callback: Callable[[str], None] | None = None
        self.locked = False

    # ── state ────────────────────────────────────────────────────────────────

    def busy_state(self) -> str:
        return user_busy_state()

    def quiet(self) -> bool:
        """Should ORION stay quiet right now? (setting AND Windows state)"""
        return should_stay_quiet()

    @property
    def greet_on_unlock(self) -> bool:
        return bool(windows_settings.get("greet_on_unlock"))

    @property
    def pause_hearing_when_locked(self) -> bool:
        return bool(windows_settings.get("pause_hearing_when_locked"))

    def describe(self) -> list[str]:
        lines = [self.hotkey.describe() if self.hotkey else "Global hotkey is switched off."]
        lines.append(self.watcher.describe() if self.watcher else "Session watch is not running.")
        lines.append(self.toasts.describe())
        return lines

    # ── the dock ─────────────────────────────────────────────────────────────

    def make_dock(self, *, on_activate: Callable[[], None] | None = None,
                  on_show: Callable[[], None] | None = None,
                  on_face_form: Callable[[str], None] | None = None,
                  on_hide: Callable[[], None] | None = None,
                  face_form: str = "", parent: Any = None, **options: Any) -> Any:
        """A ScreenDock wired to this handle's palette and busy check. Not
        shown: call ``dock.show_dock()`` when the owner asks for it (or when
        ``dock_enabled`` and ``dock_on_startup`` both say so at start-up)."""
        from .gui.screen_dock import ScreenDock

        dock = ScreenDock(parent, edge=str(windows_settings.get("dock_edge")),
                          busy_state=self.busy_state, on_activate=on_activate,
                          on_show=on_show, on_face_form=on_face_form, on_hide=on_hide,
                          face_form=face_form, **options)
        if self._palette is not None:
            dock.set_palette_colours(*self._palette)
        self._docks.append(weakref.ref(dock))
        return dock

    def dock_factory(self) -> Callable[..., Any]:
        """``make_dock`` as a plain callable, for code that takes a factory."""
        return self.make_dock

    def set_palette(self, primary: Any, accent: Any, background: Any = None) -> None:
        """Recolour the handle and every dock it made (the owner's accent)."""
        self._palette = _palette_tuple((primary, accent, background))
        for dock in self._live_docks():
            try:
                dock.set_palette_colours(*self._palette)
            except Exception:
                pass

    def _live_docks(self) -> list[Any]:
        docks = [ref() for ref in self._docks]
        self._docks = [ref for ref, dock in zip(self._docks, docks) if dock is not None]
        return [dock for dock in docks if dock is not None]

    # ── lifecycle ────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Unregister the hotkey, close the session window, stop every dock's
        render thread. Idempotent; never raises."""
        global _active
        if self._stopped:
            return
        self._stopped = True
        for part in (self.hotkey, self.mic_hotkey, self.watcher):
            try:
                if part is not None:
                    part.stop()
            except Exception:
                log.exception("stopping a presence part failed")
        for dock in self._live_docks():
            try:
                dock.stop()
            except Exception:
                pass
        if _active is not None and _active() is self:
            _active = None


def _palette_tuple(palette: Any) -> tuple | None:
    if palette is None:
        return None
    if isinstance(palette, dict):
        palette = (palette.get("primary"), palette.get("accent"), palette.get("background"))
    try:
        items = tuple(palette)
    except TypeError:
        return None
    if len(items) < 2 or items[0] is None or items[1] is None:
        return None
    return (items[0], items[1], items[2] if len(items) > 2 else None)


def _say(bus: Any, text: str) -> None:
    log.info(text)
    try:
        bus.log.emit(text)
    except Exception:
        pass


def start_windows_presence(bus: Any, *, on_hotkey: Callable[[], None],
                           on_session_event: Callable[[str], None], palette: Any = None,
                           on_mic_hotkey: Callable[[], None] | None = None,
                           dispatch: Dispatch | None = None,
                           hotkey_factory: Callable[..., Any] | None = None,
                           watcher_factory: Callable[..., Any] | None = None,
                           toast_backend: Any = None,
                           refresh_identity: bool = False) -> PresenceHandle:
    """Start the hotkey and the session watch according to the stored switches.

    Call from the Qt thread. ``on_hotkey()`` and ``on_session_event(event)``
    (event in SESSION_EVENTS) are always called on the Qt thread. Whatever
    cannot start is reported on ``bus.log`` and left out; the handle is
    returned regardless, so ORION never fails to start over this.
    """
    global _active
    toasts = ToastCenter(dispatch=dispatch, backend=toast_backend, say=lambda text: _say(bus, text))
    handle = PresenceHandle(bus, toasts, None, None, palette)

    def hotkey_pressed() -> None:
        try:
            on_hotkey()
        except Exception:
            log.exception("the hotkey handler raised")

    def session_event(event: str) -> None:
        if event == "locked":
            handle.locked = True
        elif event == "unlocked":
            handle.locked = False
        try:
            on_session_event(event)
        except Exception:
            log.exception("the session-event handler raised")

    try:
        if windows_settings.get("hotkey_enabled"):
            factory = hotkey_factory or GlobalHotkey
            hotkey = factory(hotkey_pressed, chord=str(windows_settings.get("hotkey")),
                             dispatch=dispatch)
            handle.hotkey = hotkey
            hotkey.start()
            _say(bus, "[ORION] HOTKEY: " + hotkey.describe())
    except Exception as exc:
        _say(bus, f"[ORION] HOTKEY: could not start ({type(exc).__name__}).")
    try:
        if on_mic_hotkey is not None and windows_settings.get("hotkey_enabled"):
            def mic_pressed() -> None:
                try:
                    on_mic_hotkey()
                except Exception:
                    log.exception("the microphone hotkey handler raised")
            factory = hotkey_factory or GlobalHotkey
            mic = factory(mic_pressed, chord=MIC_CHORD, fallback=MIC_FALLBACK_CHORD,
                          dispatch=dispatch, hotkey_id=0x4D55)
            handle.mic_hotkey = mic
            mic.start()
            _say(bus, "[ORION] HOTKEY: microphone - " + mic.describe())
    except Exception as exc:
        _say(bus, f"[ORION] HOTKEY: microphone hotkey could not start ({type(exc).__name__}).")
    try:
        if IS_WINDOWS or watcher_factory is not None:
            factory = watcher_factory or SessionWatcher
            watcher = factory(session_event, dispatch=dispatch)
            handle.watcher = watcher
            watcher.start()
            _say(bus, "[ORION] SESSION: " + watcher.describe())
    except Exception as exc:
        _say(bus, f"[ORION] SESSION: could not start ({type(exc).__name__}).")
    if refresh_identity and windows_settings.get("toasts"):
        def refresh() -> None:
            outcome = ensure_toast_identity()
            if outcome != "ok":
                _say(bus, f"[ORION] NOTIFY: notification identity {outcome}.")

        threading.Thread(target=refresh, name="orion-toast-identity", daemon=True).start()
    _active = weakref.ref(handle)
    return handle


__all__ = [
    "BUSY_STATES", "Chord", "ChordError", "DEFAULT_CHORD", "FALLBACK_CHORD",
    "GlobalHotkey", "HIDE_DOCK_STATES", "PresenceHandle", "QUIET_STATES",
    "SESSION_EVENTS", "SessionWatcher", "ToastCenter", "ToastDisabled",
    "ToastReceipt", "ToastUnavailable", "WinrtToastBackend", "active_hotkey_text", "build_toast_xml",
    "busy_state_from_quns", "call_on_qt", "ensure_toast_identity", "map_session_message",
    "parse_chord", "post_to_qt", "shortcut_aumid", "should_stay_quiet",
    "start_windows_presence", "toast_url", "user_busy_state",
]
