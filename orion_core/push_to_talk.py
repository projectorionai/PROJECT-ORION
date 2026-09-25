"""
Push-to-talk — hold a key, speak, release.

Why ORION needs one
-------------------
The wake word is hands-free, which is the right default and the wrong tool in a
meeting, in a noisy room, with a television on, or when you simply would rather
not say a name out loud. A key you hold is faster than a wake word, never
mishears, and — the part that matters most for privacy — means the microphone
is CLOSED the rest of the time rather than listening for a trigger.

It is also the natural way to talk to ORION while another window has focus,
which is most of the time: he is an assistant to the work, not the work.

How it works, and what it costs
-------------------------------
No new dependency. ORION already ships ``keyboard``, but that package needs
elevation for global hooks on Windows and installs a low-level hook that has to
be serviced promptly — a poor trade for reading two key states.

* **Windows** — ``GetAsyncKeyState`` through ``ctypes``, polled from one small
  daemon thread. Deliberately NOT ``RegisterHotKey``, which reports a press and
  not a release; push-to-talk needs both, and it needs to work while another
  application has focus. Polling two virtual-key codes thirty times a second is
  a rounding error of CPU and needs no message loop, no elevation and no hook.

* **macOS / Linux** — there is no dependency-free way to read global key state
  without accessibility permissions, so the chord is bound inside ORION's own
  window instead. ``scope`` reports which of the two you got, and the caller is
  expected to SAY so in the log rather than implying a global binding that does
  not exist.

Nothing here raises. If the platform hook cannot be installed the object still
constructs, reports ``scope == "window"``, and the window-level shortcut carries
it — a push-to-talk that quietly does nothing is worse than one that admits it
is only active when ORION has focus.
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Callable, Iterable

#: Ctrl+Space. Free in most desktop environments, and the same finger shape on
#: every keyboard layout — which matters, because ORION is not an English-only
#: assistant and a chord chosen around a QWERTY letter is not portable.
DEFAULT_CHORD: tuple[str, ...] = ("ctrl", "space")

#: Windows virtual-key codes for the chord names accepted here.
_VK = {
    "ctrl": 0x11, "shift": 0x10, "alt": 0x12, "space": 0x20,
    "capslock": 0x14, "insert": 0x2D, "scrolllock": 0x91, "pause": 0x13,
    "f8": 0x77, "f9": 0x78, "f10": 0x79, "f12": 0x7B,
}

#: Qt key-sequence spelling for the same names, for the windowed fallback.
_QT_NAME = {
    "ctrl": "Ctrl", "shift": "Shift", "alt": "Alt", "space": "Space",
    "capslock": "CapsLock", "insert": "Ins", "scrolllock": "ScrollLock",
    "pause": "Pause", "f8": "F8", "f9": "F9", "f10": "F10", "f12": "F12",
}

_POLL_HZ = 30.0

#: How long the chord must be held before it counts as speech. Stops a stray
#: brush of Ctrl+Space from opening the microphone for a few milliseconds.
_DEBOUNCE_S = 0.06


def chord_label(chord: Iterable[str] = DEFAULT_CHORD) -> str:
    """Human spelling of a chord, for the log and the settings UI."""
    return "+".join(_QT_NAME.get(part, str(part).title()) for part in chord)


def qt_key_sequence(chord: Iterable[str] = DEFAULT_CHORD) -> str:
    """The same chord as a Qt shortcut string, e.g. "Ctrl+Space"."""
    return "+".join(_QT_NAME.get(part, str(part).title()) for part in chord)


class PushToTalk:
    """Watches a key chord and reports hold/release.

    ``on_press`` fires once when the chord goes down and has been held past the
    debounce; ``on_release`` fires once when it comes back up. Both run on the
    poll thread, so a caller that touches Qt must marshal onto the GUI thread —
    ORION's bus already does that for every other cross-thread signal.
    """

    def __init__(self, on_press: Callable[[], None],
                 on_release: Callable[[], None],
                 *, chord: Iterable[str] = DEFAULT_CHORD,
                 poll_hz: float = _POLL_HZ) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self.chord = tuple(str(part).lower() for part in chord)
        self._interval = 1.0 / max(1.0, float(poll_hz))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._held = False
        self._down_since: float | None = None
        self._user32 = None

        if platform.system() == "Windows" and all(p in _VK for p in self.chord):
            try:
                import ctypes

                self._user32 = ctypes.windll.user32       # type: ignore[attr-defined]
            except Exception:
                self._user32 = None

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def scope(self) -> str:
        """"global" when the chord works while any window has focus, otherwise
        "window". The caller should report this honestly rather than letting the
        user assume the stronger guarantee."""
        return "global" if self._user32 is not None else "window"

    @property
    def held(self) -> bool:
        return self._held

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def describe(self) -> str:
        if self.scope == "global":
            return (f"Push-to-talk: hold {chord_label(self.chord)} "
                    f"(works in any window).")
        return (f"Push-to-talk: hold {chord_label(self.chord)} "
                f"while ORION's window has focus — this platform has no "
                f"dependency-free global key state.")

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Begin polling. Returns True when a GLOBAL watch was established."""
        if self._user32 is None or self.running:
            return self.running and self.scope == "global"
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="orion-ptt",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        if self._held:
            # Never leave the microphone latched open because the watch stopped
            # while the key happened to be down.
            self._held = False
            self._fire(self._on_release)

    # ── the windowed fallback ────────────────────────────────────────────────

    def bind_to_window(self, widget) -> bool:
        """Bind the chord inside a Qt window, for platforms with no global read.

        Qt delivers key presses and auto-repeat but a plain QShortcut gives no
        release, so this watches the widget's key events directly.
        """
        try:
            from PyQt6.QtCore import QEvent, QObject, Qt

            wanted_key = Qt.Key.Key_Space if "space" in self.chord else None
            modifier = (Qt.KeyboardModifier.ControlModifier
                        if "ctrl" in self.chord else
                        Qt.KeyboardModifier.AltModifier)
            owner = self

            class _ChordFilter(QObject):
                def eventFilter(self, obj, event):          # noqa: N802
                    kind = event.type()
                    if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                        if event.isAutoRepeat():
                            return False
                        matches = (wanted_key is None or event.key() == wanted_key)
                        if matches and (event.modifiers() & modifier):
                            if kind == QEvent.Type.KeyPress:
                                owner._set_held(True)
                            else:
                                owner._set_held(False)
                    return False

            self._filter = _ChordFilter(widget)
            widget.installEventFilter(self._filter)
            return True
        except Exception:
            return False

    # ── internals ────────────────────────────────────────────────────────────

    def _fire(self, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            # A failing consumer must never kill the watch — the user would be
            # left with a microphone that silently stopped responding.
            pass

    def _set_held(self, held: bool) -> None:
        if held == self._held:
            return
        self._held = held
        self._fire(self._on_press if held else self._on_release)

    def _chord_down(self) -> bool:
        if self._user32 is None:
            return False
        try:
            # The high bit of GetAsyncKeyState is "currently down". The low bit
            # is "pressed since last call" and is deliberately NOT used: it is
            # consumed by the read, so two pollers would steal it from each
            # other and it says nothing about whether the key is still held.
            return all(self._user32.GetAsyncKeyState(_VK[part]) & 0x8000
                       for part in self.chord)
        except Exception:
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            down = self._chord_down()
            now = time.monotonic()
            if down and not self._held:
                if self._down_since is None:
                    self._down_since = now
                elif now - self._down_since >= _DEBOUNCE_S:
                    self._set_held(True)
            elif not down:
                self._down_since = None
                if self._held:
                    self._set_held(False)
            self._stop.wait(self._interval)


# ──────────────────────────────────────────────────────────────────────────────
# THE CAPTURE GATE
# ──────────────────────────────────────────────────────────────────────────────
#
# A push-to-talk that nothing consults is dead code, and ORION already had one
# of those: bus.viseme was declared, subscribed to and fully implemented for a
# whole release without a single producer. So the gate is reached from the
# audio path directly rather than waiting for a settings screen to exist.
#
# OFF by default. The wake word stays the default way in, because a key that
# has to be held is a change to how the assistant is used and should be chosen,
# not inherited. While it is off this function is a constant and costs nothing.

_enabled = False
_watcher: "PushToTalk | None" = None


def enabled() -> bool:
    return _enabled


def gate_allows_capture() -> bool:
    """True when the microphone may be streamed.

    With push-to-talk off this is always True and the caller behaves exactly as
    before. With it on, the microphone is CLOSED unless the chord is held —
    which is the privacy property that makes the feature worth having: nothing
    leaves the machine while you are not holding the key.
    """
    if not _enabled:
        return True
    return bool(_watcher is not None and _watcher.held)


def enable(on_press=None, on_release=None, *, chord=DEFAULT_CHORD) -> "PushToTalk":
    """Turn push-to-talk on and start watching the chord."""
    global _enabled, _watcher
    disable()
    _watcher = PushToTalk(on_press or (lambda: None),
                          on_release or (lambda: None), chord=chord)
    _watcher.start()
    _enabled = True
    return _watcher


def disable() -> None:
    global _enabled, _watcher
    _enabled = False
    watcher, _watcher = _watcher, None
    if watcher is not None:
        try:
            watcher.stop()
        except Exception:
            # Teardown must not raise. The flag is already cleared above, so
            # the gate is open again whatever happened to the watcher.
            pass


def active_watcher() -> "PushToTalk | None":
    return _watcher


__all__ = ["DEFAULT_CHORD", "PushToTalk", "active_watcher", "chord_label",
           "disable", "enable", "enabled", "gate_allows_capture",
           "qt_key_sequence"]
