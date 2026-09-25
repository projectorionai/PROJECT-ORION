"""
Follow the audio device the user is actually using.

ORION resolved his microphone and speaker once, at startup, and held them for
the life of the process. Plug in headphones an hour later and Windows moves its
default while ORION carries on talking to the speakers that are no longer being
listened to — and listening to a microphone that is no longer being spoken
into. Nothing fails; he simply stops being able to hear you, and you stop being
able to hear him.

This watches for that and moves with it.

Two rules, and the second is the important one
----------------------------------------------
**Follow the system default.** When Windows changes which device is default —
headphones in, headset out, a monitor's speakers waking up — ORION goes there
too, because that is by definition where the user is.

**Never override a device the user chose.** If they have pinned a specific
microphone, that is a decision, and a decision outranks a default. Following
would silently undo it the next time anything changed. A pinned device that has
been UNPLUGGED is different again: that is not a decision any more, it is a
dead end, so ORION falls back to the default rather than staying deaf.

Cost
----
PortAudio caches its device list, so noticing a change means re-initialising
it. That is not free, and it is why this polls on a slow cadence and off the
event loop rather than checking constantly. The user replugs a headset a few
times a day; a five-second gap between that and ORION noticing costs nothing.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import audio_devices

#: How often to look. Slow on purpose — see the module docstring.
POLL_SECONDS = 5.0


@dataclass(frozen=True)
class DefaultDevices:
    """The system's current default input and output, by endpoint ID.

    Identified by ID rather than name or index: indices are renumbered as
    hardware comes and goes, and two identical headsets share a name. An
    endpoint ID is stable and unique.
    """

    input_name: str = ""
    output_name: str = ""

    def differs_from(self, other: "DefaultDevices") -> list[str]:
        changed = []
        if self.input_name != other.input_name:
            changed.append("input")
        if self.output_name != other.output_name:
            changed.append("output")
        return changed


def _default_endpoint_id(capture: bool) -> str:
    """The Windows endpoint ID of the current default device, or "".

    Read straight from Core Audio through COM, and deliberately NOT through
    sounddevice. PortAudio caches its device list at initialisation, so the
    obvious way to notice a new device is to call ``sd._terminate()`` and
    ``sd._initialize()`` — which is what the first version of this did, and it
    crashed ORION with STATUS_HEAP_CORRUPTION inside a minute. Tearing
    PortAudio down while the capture and playback streams are open destroys
    the state those streams are still using. The fault dump named this
    function.

    Asking the operating system is both safer and more direct: it is read
    only, it touches nothing ORION is using, and the endpoint ID is exactly
    the identity we want. An ID is also better than a friendly name — names
    are duplicated across identical hardware, IDs are not.
    """
    if sys.platform != "win32":
        return ""
    import ctypes
    from ctypes import POINTER, byref, c_void_p, c_wchar_p

    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                    ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_byte * 8)]

        def __init__(self, text: str) -> None:
            super().__init__()
            ole32 = ctypes.WinDLL("ole32")
            if ole32.CLSIDFromString(c_wchar_p(text), byref(self)) != 0:
                raise OSError(f"bad GUID {text}")

    CLSID_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
    IID_ENUMERATOR = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
    CLSCTX_ALL = 23
    E_RENDER, E_CAPTURE, ROLE_CONSOLE = 0, 1, 0

    ole32 = ctypes.WinDLL("ole32")
    ole32.CoInitializeEx(None, 0)                       # COINIT_MULTITHREADED
    enumerator = c_void_p()
    device = c_void_p()
    try:
        if ole32.CoCreateInstance(byref(_GUID(CLSID_ENUMERATOR)), None,
                                  CLSCTX_ALL, byref(_GUID(IID_ENUMERATOR)),
                                  byref(enumerator)) != 0:
            return ""
        # IMMDeviceEnumerator::GetDefaultAudioEndpoint is vtable slot 4
        vtable = ctypes.cast(enumerator, POINTER(POINTER(c_void_p)))[0]
        get_default = ctypes.WINFUNCTYPE(
            ctypes.c_long, c_void_p, ctypes.c_int, ctypes.c_int,
            POINTER(c_void_p))(vtable[4])
        flow = E_CAPTURE if capture else E_RENDER
        if get_default(enumerator, flow, ROLE_CONSOLE, byref(device)) != 0:
            return ""                                   # no device of that kind
        # IMMDevice::GetId is vtable slot 5
        device_vtable = ctypes.cast(device, POINTER(POINTER(c_void_p)))[0]
        get_id = ctypes.WINFUNCTYPE(
            ctypes.c_long, c_void_p, POINTER(c_wchar_p))(device_vtable[5])
        out = c_wchar_p()
        if get_id(device, byref(out)) != 0:
            return ""
        try:
            return str(out.value or "")
        finally:
            ole32.CoTaskMemFree(out)
    except Exception:
        return ""
    finally:
        for handle in (device, enumerator):
            if handle:
                try:
                    release = ctypes.cast(
                        handle, POINTER(POINTER(c_void_p)))[0][2]
                    ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(release)(handle)
                except Exception:
                    pass


def read_defaults(refresh: bool = True) -> DefaultDevices:
    """The current default input and output, by Windows endpoint ID.

    ``refresh`` is accepted and ignored: there is nothing to refresh any more,
    because nothing is cached. It stays so existing callers do not break.
    """
    return DefaultDevices(_default_endpoint_id(capture=True),
                          _default_endpoint_id(capture=False))


def pinned(kind: str) -> str:
    """The device the user chose for *kind*, or "" if they chose none."""
    try:
        import os

        env = os.getenv("ORION_AUDIO_INPUT" if kind == "input"
                        else "ORION_AUDIO_OUTPUT", "").strip()
        if env:
            return env
        saved = audio_devices._load_config().get(kind)
        return str(saved).strip() if saved else ""
    except Exception:
        return ""


def pin_is_live(kind: str) -> bool:
    """Whether the user's pinned device is still plugged in.

    A pin that no longer resolves is not a preference being honoured, it is
    ORION talking to a device that is not there.
    """
    spec = pinned(kind)
    if not spec:
        return False
    try:
        return audio_devices._match(spec, kind) is not None
    except Exception:
        return False


@dataclass
class AudioFollower:
    """Moves ORION onto the current default device when it changes."""

    bus: Any = None
    #: Called as switch(kind) -> str|None to actually move the device.
    switch: Callable[[str], Any] | None = None
    poll_seconds: float = POLL_SECONDS
    _last: DefaultDevices = field(default_factory=DefaultDevices)
    _stop: asyncio.Event | None = None
    _started_at: float = 0.0

    def prime(self) -> DefaultDevices:
        """Record the starting state without acting on it."""
        self._last = read_defaults()
        self._started_at = time.monotonic()
        return self._last

    def check_once(self) -> list[str]:
        """Look once. Returns the kinds that were actually switched."""
        current = read_defaults()
        changed = current.differs_from(self._last)
        self._last = current
        moved: list[str] = []
        for kind in changed:
            if pinned(kind) and pin_is_live(kind):
                # A decision outranks a default. Say so once so the user is
                # not left wondering why ORION ignored the change.
                self._log(f"AUDIO: the default {kind} changed, but you have "
                          f"'{pinned(kind)}' pinned — staying on it.")
                continue
            name = current.input_name if kind == "input" else current.output_name
            if self._apply(kind):
                moved.append(kind)
                self._log(f"AUDIO: following the system default {kind} — "
                          f"now {name or 'the default device'}.")
        return moved

    def _apply(self, kind: str) -> bool:
        """Move ORION onto the current default for *kind*.

        Goes through the same bus request the user's own device picker uses,
        for two reasons: that handler runs on the thread the audio streams
        were opened on, and it already knows how to reopen them. Reaching into
        the streams from this watcher's worker thread would be a second,
        less-tested way to do the same thing.

        The spec is the word "default" rather than the device's name, which
        matters: passing a name would PIN that device, and the next time the
        user plugged something in ORION would refuse to follow because he
        would think a choice had been made.
        """
        if self.switch is not None:
            try:
                self.switch(kind)
                return True
            except Exception as exc:
                self._log(f"AUDIO: could not follow the default {kind} - {exc}")
                return False
        bus = self.bus
        if bus is None:
            return False
        try:
            bus.audio_device_request.emit(kind, "default")
            return True
        except Exception as exc:
            self._log(f"AUDIO: could not follow the default {kind} - {exc}")
            return False

    def _log(self, message: str) -> None:
        bus = self.bus
        if bus is None:
            return
        try:
            bus.log.emit(message)
        except Exception:
            pass

    async def run(self) -> None:
        """Poll until stopped. Never lets one bad look end the loop."""
        self._stop = asyncio.Event()
        await asyncio.to_thread(self.prime)
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(),
                                           timeout=self.poll_seconds)
                    return
                except asyncio.TimeoutError:
                    pass
                try:
                    await asyncio.to_thread(self.check_once)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._log(f"AUDIO: device watch recovered - {exc}")
        except asyncio.CancelledError:
            raise

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()


__all__ = [
    "POLL_SECONDS", "AudioFollower", "DefaultDevices",
    "pin_is_live", "pinned", "read_defaults",
]
