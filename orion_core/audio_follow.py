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

A monitor's speakers are not where the user went
------------------------------------------------
When a Bluetooth headset switches off, Windows moves the default to the
monitor's speakers — the only output left — and that move is right to follow.
But the monitor ALSO grabs the default when it wakes while the headset is still
on, and following THAT sends ORION's voice to speakers nobody is listening to.
So a move onto a display endpoint is declined while the previous device is
still connected, and followed the moment it is not (including later, when the
headset is switched off while the monitor already holds the default).

Cost
----
PortAudio caches its device list, so a device that appeared or vanished is
invisible to it until it is re-initialised — and a stale list makes every open
fail ("device ID out of range"), which left ORION mute for 14 minutes on 20 Oct.
Re-initialising is not free and is only safe with every stream closed, so it is
done by audio_devices.PortAudioReinit (which closes both streams first), once,
from this worker thread, immediately BEFORE ORION is asked to reopen — and this
polls on a slow cadence and off the event loop rather than checking constantly.
The user replugs a headset a few times a day; a five-second gap between that
and ORION noticing costs nothing.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
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


def desktop_output_state() -> tuple[str, bool, int] | None:
    """(name, muted, volume %) of Windows' default output, or None.

    Read only. "My desktop keeps randomly muting" (10 Oct) had no trace to go
    on: ORION never sets the default device or mutes it unasked, and nothing
    recorded what the desktop's output was doing. The watcher logs changes."""
    if sys.platform != "win32":
        return None
    try:
        import comtypes  # type: ignore[import-not-found]
        try:
            comtypes.CoInitialize()          # a pool thread, not the importer's
        except OSError:
            pass
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        device = AudioUtilities.GetSpeakers()
        endpoint = getattr(device, "EndpointVolume", None)
        if endpoint is None:                 # older pycaw: a raw IMMDevice
            from ctypes import POINTER, cast
            endpoint = cast(device.Activate(IAudioEndpointVolume._iid_, 23, None),
                            POINTER(IAudioEndpointVolume))
        name = str(getattr(device, "FriendlyName", "") or "the default output")
        return name, bool(endpoint.GetMute()), int(round(endpoint.GetMasterVolumeLevelScalar() * 100))
    except Exception:
        return None


#: Friendly-name fragments of an output that belongs to a display (HDMI,
#: DisplayPort, a monitor's built-in speakers), lower-case.
DISPLAY_ENDPOINT_HINTS = (
    "nvidia high definition audio", "amd high definition audio",
    "intel(r) display audio", "intel display audio",
    "hdmi", "displayport", "display port",
)


def is_display_endpoint(name: str) -> bool:
    """Whether *name* is the speakers of a display rather than of a headset or
    desk speakers (e.g. "XV252Q (NVIDIA High Definition Audio)")."""
    low = " ".join(str(name or "").split()).casefold()
    return any(hint in low for hint in DISPLAY_ENDPOINT_HINTS)


def endpoint_is_active(endpoint_id: str) -> bool | None:
    """Whether Windows reports endpoint *endpoint_id* as plugged in and enabled.

    True: present. False: switched off, unplugged, disabled or removed.
    None: cannot tell (not Windows, no pycaw) — callers treat that as "follow",
    which is what ORION did before this check existed.

    Read only, by endpoint ID, so it is cheap enough for every poll. NOT
    PortAudio's list: that is exactly what is stale after a headset switches off.
    """
    if sys.platform != "win32" or not endpoint_id:
        return None
    try:
        import comtypes  # type: ignore[import-not-found]
        try:
            comtypes.CoInitialize()
        except OSError:
            pass
        from pycaw.pycaw import AudioUtilities
        enumerator = AudioUtilities.GetDeviceEnumerator()
        try:
            device = enumerator.GetDevice(endpoint_id)
        except Exception as exc:
            hresult = (exc.args[0] if getattr(exc, "args", None) else 0) & 0xFFFFFFFF
            return False if hresult == 0x80070490 else None      # E_NOTFOUND
        return int(device.GetState()) == 1                      # DEVICE_STATE_ACTIVE
    except Exception:
        return None


def describe_output_change(before: tuple[str, bool, int] | None,
                           after: tuple[str, bool, int] | None) -> str:
    """One log line for a change in the desktop's output, or ''."""
    if before is None or after is None or before == after:
        return ""
    (name0, muted0, vol0), (name1, muted1, vol1) = before, after
    if name0 != name1:
        return f"AUDIO: the desktop's sound moved from {name0} to {name1} (Windows default output)."
    if muted1 != muted0:
        return f"AUDIO: the desktop was {'muted' if muted1 else 'unmuted'} ({name1})."
    if abs(vol1 - vol0) >= 10 or (vol1 == 0) != (vol0 == 0):
        return f"AUDIO: desktop volume {vol0}% -> {vol1}% ({name1})."
    return ""


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
    _output: Any = None          # (name, muted, volume %) of the desktop output
    #: (endpoint ID, friendly name) of the output ORION is deliberately staying
    #: on while Windows' default sits on a display's speakers; None otherwise.
    _staying: Any = None
    #: Where the owner's listening habit is kept (seconds ORION has spent on
    #: each output); None keeps no habit and never writes (tests, embedding).
    habit_path: Path | None = None
    _habit: dict[str, float] = field(default_factory=dict)
    _habit_unsaved: float = 0.0

    #: Listening time on a display's speakers that makes them the owner's
    #: own choice rather than a monitor waking up and taking the default.
    HABIT_MIN_SECONDS = 600.0
    HABIT_SAVE_EVERY_S = 60.0

    def prime(self) -> DefaultDevices:
        """Record the starting state without acting on it."""
        self._last = read_defaults()
        self._output = desktop_output_state()
        self._started_at = time.monotonic()
        self._habit = self._load_habit()
        return self._last

    # ── the owner's listening habit (25 Oct 2026) ────────────────────────────
    #
    # 11:50:36: Windows moved the default to the XV252Q monitor; ORION stayed
    # on the HyperX headset because its wireless dongle still reported
    # "connected" - with the headset off. The owner could not hear him, said
    # "can you do a restart because I can't hear you", and only the restart
    # (which takes the default) brought the voice back. The monitor is where
    # he listens (the whole of 24 Oct's session ran on it). Staying off a
    # display is right when a waking monitor steals the default from a
    # headset in use, wrong when the display is the owner's own speakers: the
    # time ORION has spent on each output tells the two apart.

    def _load_habit(self) -> dict[str, float]:
        path = self.habit_path
        if path is None:
            return {}
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return {str(k): float(v) for k, v in (data.get("seconds") or {}).items()
                    if isinstance(v, (int, float))}
        except (OSError, ValueError, AttributeError):
            return {}

    def _note_listening(self, name: str, seconds: float) -> None:
        if not name or self.habit_path is None or seconds <= 0:
            return
        self._habit[name] = float(self._habit.get(name, 0.0)) + float(seconds)
        self._habit_unsaved += seconds
        if self._habit_unsaved >= self.HABIT_SAVE_EVERY_S:
            self._habit_unsaved = 0.0
            try:
                from .atomic_io import atomic_write_text
                top = dict(sorted(self._habit.items(), key=lambda kv: -kv[1])[:12])
                atomic_write_text(Path(self.habit_path), json.dumps({"seconds": top}, indent=1),
                                  encoding="utf-8")
            except Exception:
                pass

    def usual_output(self, display_name: str, other_name: str) -> bool:
        """True when the owner listens on *display_name* at least as much as
        on *other_name* (and for long enough to call it a habit)."""
        display = float(self._habit.get(display_name, 0.0))
        return display >= self.HABIT_MIN_SECONDS and display >= float(self._habit.get(other_name, 0.0))

    def check_once(self) -> list[str]:
        """Look once. Returns the kinds that were actually switched."""
        before_output = getattr(self, "_output", None)
        output = desktop_output_state()
        note = describe_output_change(before_output, output)
        if note:
            self._log(note)
        if output is not None:
            self._output = output
        # The output ORION speaks on: the one it stays on, else the default.
        self._note_listening((self._staying or ("", ""))[1] or (output[0] if output else ""),
                             self.poll_seconds)
        previous = self._last
        current = read_defaults()
        changed = current.differs_from(previous)
        self._last = current
        self._recheck_stay(changed, output)
        movers: list[tuple[str, str]] = []           # (kind, friendly name)
        for kind in changed:
            if pinned(kind) and pin_is_live(kind):
                # A decision outranks a default. Say so once so the user is
                # not left wondering why ORION ignored the change.
                self._log(f"AUDIO: the default {kind} changed, but you have "
                          f"'{pinned(kind)}' pinned — staying on it.")
                continue
            name = self._friendly_name(kind, output)
            if kind == "output":
                held = self._staying or (previous.output_name,
                                         before_output[0] if before_output else "")
                if self._stay_off_display(held[0], held[1], name):
                    continue
            movers.append((kind, name))
        if movers:
            self._refresh_portaudio(movers)
        moved: list[str] = []
        for kind, name in movers:
            if self._apply(kind):
                moved.append(kind)
                self._log(f"AUDIO: following the system default {kind} — "
                          f"now {name}.")
        return moved

    @staticmethod
    def _friendly_name(kind: str, output: tuple[str, bool, int] | None) -> str:
        """The device's name as a person knows it, never its endpoint ID.

        The default output's name was already read from Core Audio by
        desktop_output_state(); the input's is read the same way here."""
        name = output[0] if kind == "output" and output else ""
        if not name:
            name = audio_devices._system_default_name(kind) or ""
        return name or f"the default {kind} device"

    def _stay_off_display(self, previous_id: str, previous_name: str,
                          new_name: str) -> bool:
        """True when ORION should NOT follow Windows' default onto *new_name*.

        Declines only a move onto a display's speakers while the device ORION
        was on is still connected. Anything unknown (no name, no way to ask
        Windows) follows, as ORION always did. When it stays, it also tells
        audio_devices to keep resolving "the default" to that device, so a
        later reopen (a write fault, the preferences dialog) does not wander
        onto the monitor either."""
        if (not is_display_endpoint(new_name) or not previous_id or not previous_name
                or is_display_endpoint(previous_name)
                or endpoint_is_active(previous_id) is not True):
            self._staying = None
            audio_devices.hold_default("output", None)
            return False
        if self.usual_output(new_name, previous_name):
            # The owner's own speakers, not a monitor waking up.
            if self._staying is not None:
                audio_devices.hold_default("output", None)
            self._staying = None
            self._log(f"AUDIO: following the default onto {new_name} - that is where "
                      f"you usually listen, even though {previous_name} is still connected.")
            return False
        if self._staying is None or self._staying[0] != previous_id:
            self._log(f"AUDIO: Windows moved the default output to {new_name}, a "
                      f"display's speakers, but {previous_name} is still connected — "
                      "staying on it. (Tell me to use the monitor if you want the "
                      "voice there.)")
        self._staying = (previous_id, previous_name)
        audio_devices.hold_default("output", previous_name)
        return True

    def _recheck_stay(self, changed: list[str], output: Any) -> None:
        """While staying put, notice the moment the device we stayed on goes.

        The headset can switch off AFTER the monitor took the default, so no
        default change arrives to prompt a move; and it may still have looked
        connected on the poll where the default first moved. Either way, once
        it is gone the monitor is the only output and ORION follows it."""
        if self._staying is None or "output" in changed:
            return
        if endpoint_is_active(self._staying[0]) is False:
            self._log(f"AUDIO: {self._staying[1]} is no longer connected — "
                      "following the system default output instead.")
            changed.append("output")

    def _refresh_portaudio(self, movers: list[tuple[str, str]]) -> None:
        """Make PortAudio's device list match Windows BEFORE anything reopens.

        PortAudio only learns of a device that appeared or vanished when it is
        re-initialised; reopening first is what failed 799 times on 20 Oct. One
        refresh serves every kind that moved (a headset moves input and output
        together). Forced: a real default change must be reflected even a
        moment after the last attempt. The coordinator closes the renderer and
        the microphone first and abandons the refresh, logging why, if either
        will not let go — in which case ORION is moved exactly as it was before
        this existed.
        """
        reason = " and ".join(f"the default {kind} changed to {name}"
                              for kind, name in movers)
        try:
            audio_devices.request_reinit(reason, force=True)
        except Exception as exc:
            self._log(f"AUDIO: could not refresh the device list - {exc}")

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
    "DISPLAY_ENDPOINT_HINTS", "POLL_SECONDS", "AudioFollower", "DefaultDevices",
    "endpoint_is_active", "is_display_endpoint", "pin_is_live", "pinned",
    "read_defaults",
]
