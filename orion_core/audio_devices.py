"""
Audio device selection (Phase 8 follow-up) — pick and see ORION's ears/voice.

Previously the capture and playback streams used PortAudio's *default* input
and output with no way to inspect or change them, so "I can't hear ORION"
usually meant his voice was being played to the wrong device (a monitor with
no speakers, a disconnected headset) with no visibility into which.

This module gives ORION explicit, persistent device control:

    list_devices()            — every input/output device with its index.
    describe()                — which mic and speaker are in effect right now.
    resolve(kind)             — the device index to hand PortAudio (or None
                                for the system default), from env then config.
    set_device(kind, spec)    — choose a device by index or name fragment;
                                persisted to config/audio.json.

Selection precedence: environment (``ORION_AUDIO_INPUT`` /
``ORION_AUDIO_OUTPUT``) overrides the saved config, which overrides the system
default.  A spec may be a device index (``"3"``) or a case-insensitive name
fragment (``"headphones"``).  Nothing here raises — a bad spec logs and falls
back to the default so audio never fails to start.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import sys
import threading
import time
import weakref
from threading import RLock
from typing import Any, Callable, Iterator

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

AUDIO_CONFIG_PATH = CONFIG_DIR / "audio.json"

# In-memory record of the device actually applied to the live stream this
# session (as opposed to what's merely saved to config) — set by the audio
# engine itself whenever it reopens a stream, read by resolve_effective() so
# the GUI picker reflects reality even when a live swap hasn't been persisted.
_live: dict[str, int | None] = {}
_suggestion_pending_until = 0.0
_suggestion_lock = RLock()

#: A device ORION has been told to STAY on even though Windows' default moved
#: (see audio_follow's display-endpoint policy), by friendly name. Consulted by
#: system_default_index() so a later reopen — a write fault, a preferences
#: click — does not quietly wander onto the endpoint the follower declined.
_held_default: dict[str, str] = {}


def hold_default(kind: str, name: str | None) -> None:
    """Keep resolving the "default" *kind* device to *name* (None releases it)."""
    kind = "input" if str(kind).lower().startswith("in") else "output"
    if name:
        _held_default[kind] = str(name)
    else:
        _held_default.pop(kind, None)


# ── short-lived streams must not be open while PortAudio is torn down ────────
#
# The renderer and the microphone are the two streams ORION holds for its whole
# life, and the re-initialisation coordinator at the bottom of this file asks
# both to let go. probe() and transport_works() are the other places this
# module opens a stream — for a few hundred milliseconds, on whatever worker
# thread asked — and a re-init that landed in that window would terminate
# PortAudio under them, which is the same crash. They announce themselves here
# instead: the coordinator waits for the count to reach zero and, while it is
# running, new ones wait for it to finish.

_gate = threading.Condition()
_transient_streams = 0
_reinit_running = False
#: True only between sd._terminate() and the end of sd._initialize(): PortAudio
#: has no state then, so even a device QUERY from another thread must wait.
_portaudio_down = False


def _wait_portaudio_up(wait: float = 6.0) -> None:
    """Block (bounded) while PortAudio is mid re-initialisation; ~100-500 ms."""
    if _portaudio_down:                    # unlocked fast path: the normal case
        with _gate:
            _gate.wait_for(lambda: not _portaudio_down, timeout=wait)


@contextlib.contextmanager
def stream_guard(wait: float = 6.0) -> Iterator[None]:
    """Hold this for as long as a stream opened by the caller exists."""
    global _transient_streams
    with _gate:
        # Bounded: a coordinator that never finishes must not freeze a probe.
        _gate.wait_for(lambda: not _reinit_running, timeout=wait)
        _transient_streams += 1
    try:
        yield
    finally:
        with _gate:
            _transient_streams -= 1
            _gate.notify_all()


def _holds_portaudio_stream(func: Callable[..., Any]) -> Callable[..., Any]:
    """Run *func* (which opens and closes its own stream) inside stream_guard()."""
    @functools.wraps(func)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        with stream_guard():
            return func(*args, **kwargs)
    return guarded


def _load_config() -> dict[str, Any]:
    try:
        data = json.loads(AUDIO_CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_config(cfg: dict[str, Any]) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(AUDIO_CONFIG_PATH, json.dumps(cfg, indent=2), encoding="utf-8")
    except OSError:
        pass


def configured_spec(kind: str) -> str:
    """Saved device choice for settings, without resolving or probing hardware.

    ``device_name`` is a display label (``[16] Microphone``), not the persisted
    device identity. Keeping that label out of settings also preserves a name
    fragment or an unplugged endpoint when the dialog is accepted unchanged.
    Environment overrides and the device actually opened remain runtime state.
    """
    kind = "input" if str(kind).lower().startswith("in") else "output"
    saved = _load_config().get(kind)
    spec = "" if saved is None else str(saved).strip()
    return "" if spec.lower() in {"default", "reset", "auto"} else spec


def _devices() -> list[dict[str, Any]]:
    _wait_portaudio_up()
    try:
        import sounddevice as sd
        return list(sd.query_devices())
    except Exception:
        return []


def _system_default_name(kind: str) -> str | None:
    """Read Windows' current multimedia endpoint, on the audio owner's thread.

    PortAudio caches its default device at initialization. Reading the Windows
    endpoint avoids reopening that old default after a headset/output change.
    Only this call's COM initialization is balanced; other users own theirs.
    """
    if sys.platform != "win32" or kind not in {"input", "output"}:
        return None
    com = None
    initialized = False
    endpoint = device = None
    try:
        import comtypes as com
        com.CoInitialize()
        initialized = True
        from pycaw.pycaw import AudioUtilities

        endpoint = (AudioUtilities.GetMicrophone() if kind == "input"
                    else AudioUtilities.GetSpeakers())
        device = endpoint
        if endpoint is not None and not hasattr(endpoint, "FriendlyName"):
            # GetMicrophone returns an IMMDevice in current pycaw releases;
            # GetSpeakers already returns its AudioDevice wrapper.
            device = AudioUtilities.CreateDevice(endpoint)
        name = str(getattr(device, "FriendlyName", "") or "").strip()
        return name or None
    except Exception:
        return None
    finally:
        # Release references to COM interfaces before leaving this apartment.
        endpoint = device = None
        if initialized:
            try:
                com.CoUninitialize()
            except Exception:
                pass


def system_default_index(kind: str) -> int | None:
    """Map the current Windows endpoint to a safe cached PortAudio index.

    Call on the thread opening the stream, never the event loop. This does not
    refresh/reinitialize PortAudio, open a device, or change the OS default.
    Exact names and unambiguous MME-truncated prefixes are the only matches;
    absent or ambiguous endpoints retain the ordinary ``None`` default fallback.

    The cached list itself is refreshed elsewhere — PortAudioReinit, below,
    which only ever does so with every stream closed. A name held by
    hold_default() (a device the follower chose to stay on) wins over Windows'.
    """
    if kind not in {"input", "output"}:
        return None
    name = _held_default.get(kind) or _system_default_name(kind)
    if not name:
        return None
    target = " ".join(name.split()).casefold()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    try:
        import sounddevice as sd
        records = _devices()
        _wait_portaudio_up()
        api_names = [str(api["name"]).casefold() for api in sd.query_hostapis()]
    except Exception:
        return None

    compatible = []
    for index, record in enumerate(records):
        try:
            label = " ".join(str(record.get("name", "")).split()).casefold()
            if int(record.get(key, 0)) <= 0 or not label or _is_pseudo(label):
                continue
        except (TypeError, ValueError):
            continue
        compatible.append((index, label, record))

    order = list(_PREFERRED_HOST_APIS)
    measured = _chosen_api.get(kind)
    if measured in order:
        order.remove(measured)
        order.insert(0, measured)

    candidates = []
    for index, label, record in compatible:
        if label != target:
            # MME's 31-character limit can leave 30 after a trailing space is
            # stripped. Short fragments such as "Speakers" must never guess.
            if not (30 <= len(label) <= 31 and target.startswith(label)):
                continue
            if any(other != target and len(other) > len(label) and other.startswith(label)
                   for _other_index, other, _other_record in compatible):
                continue
        try:
            api = api_names[int(record["hostapi"])]
        except (KeyError, TypeError, ValueError, IndexError):
            api = ""
        rank = next((position for position, token in enumerate(order) if token in api), len(order))
        candidates.append((rank, index))

    if not candidates:
        return None
    best = min(rank for rank, _index in candidates)
    winners = [index for rank, index in candidates if rank == best]
    # Two identical endpoint names within one API cannot be distinguished by
    # this read-only mapping. Keep PortAudio's default instead of guessing.
    return winners[0] if len(winners) == 1 else None


def _match(spec: str, kind: str) -> int | None:
    """Resolve a spec (index or name fragment) to a device index of *kind*."""
    spec = str(spec or "").strip()
    if not spec:
        return None
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = _devices()
    # Exact index.
    if spec.isdigit():
        idx = int(spec)
        if 0 <= idx < len(devices) and devices[idx].get(channel_key, 0) > 0:
            return idx
        return None
    # Case-insensitive name fragment, first device of the right kind.
    low = spec.lower()
    for idx, dev in enumerate(devices):
        if dev.get(channel_key, 0) > 0 and low in str(dev.get("name", "")).lower():
            return idx
    return _match_loose(low, kind, devices, channel_key)


# ── what people call their devices ───────────────────────────────────────────
#
# Windows names a headset's output "Speakers (HyperX Cloud III S Wireless)" and
# a monitor's "XV252Q (NVIDIA High Definition Audio)" (this PC), so "switch to
# my headphones" or "use the monitor" never matched a name fragment. When the
# literal fragment fails, the request is read for the KIND of device.

_DEVICE_FILLER = frozenset({
    "my", "the", "a", "an", "to", "use", "switch", "change", "set", "please", "audio",
    "sound", "device", "output", "input", "through", "on", "onto", "over", "via", "play",
    "using", "instead", "now", "orion", "can", "you", "from"})
_HEADSET_WORDS = frozenset({"headphones", "headphone", "headset", "earphones", "earphone",
                            "earbuds", "buds", "cans", "airpods", "gaming"})
_HEADSET_HINTS = ("headset", "headphone", "earphone", "earbud", "buds", "airpods", "hyperx",
                  "cloud", "arctis", "steelseries", "razer", "corsair", "jabra", "hands-free",
                  "wh-", "bose", "logitech g", "astro", "wireless")
_DISPLAY_WORDS = frozenset({"monitor", "screen", "display", "tv", "television", "hdmi"})
_DISPLAY_HINTS = ("hdmi", "displayport", "display audio", "nvidia high definition",
                  "amd high definition", "intel(r) display")
_WEBCAM_WORDS = frozenset({"webcam", "camera", "cam"})
_WEBCAM_HINTS = ("webcam", "camera", "c920", "c922", "brio", "cam")
#: Software devices: a request for "speakers" means hardware.
_VIRTUAL_HINTS = ("voice changer", "virtual", "streaming", "mfdriver", "sound mapper",
                  "primary sound", "cable", "voicemeeter", "steam streaming")


def _match_loose(low: str, kind: str, devices: list[dict[str, Any]],
                 channel_key: str) -> int | None:
    """Best device of *kind* for a loosely worded request, or None."""
    words = [w for w in re.findall(r"[a-z0-9-]+", low) if w not in _DEVICE_FILLER]
    if not words:
        return None
    candidates = [(idx, str(dev.get("name", "")).lower()) for idx, dev in enumerate(devices)
                  if dev.get(channel_key, 0) > 0 and str(dev.get("name", "")).strip()]
    real = [(i, n) for i, n in candidates if not any(h in n for h in _VIRTUAL_HINTS)] or candidates

    def first_with(hints: tuple[str, ...], pool: list[tuple[int, str]]) -> int | None:
        for idx, name in pool:
            if any(hint in name for hint in hints):
                return idx
        return None

    wanted = set(words)
    if wanted & _HEADSET_WORDS:
        return first_with(_HEADSET_HINTS, real)
    if wanted & _DISPLAY_WORDS:
        return first_with(_DISPLAY_HINTS, real)
    if wanted & _WEBCAM_WORDS:
        return first_with(_WEBCAM_HINTS, real)
    # "Microphone" / "speakers" alone: the user's own hardware first.
    if wanted <= {"mic", "microphone", "speakers", "speaker", "headset"}:
        for fragment in PREFERRED_DEVICES.get(kind, ()):
            idx = first_with((fragment,), real)
            if idx is not None:
                return idx
        if kind == "output":
            # Plain speakers: a "Speakers (...)" device that is not a headset.
            plain = [(i, n) for i, n in real if n.startswith("speakers")
                     and not any(h in n for h in _HEADSET_HINTS)]
            if plain:
                return plain[0][0]
        return real[0][0] if real else None
    # Every remaining word in the name ("hyperx cloud", "fifine mic").
    phrase_hits = [idx for idx, name in real if all(w in name for w in words)]
    if phrase_hits:
        return phrase_hits[0]
    # A slip of the tongue or the keyboard ("fifne", "hyper x").
    import difflib
    for idx, name in real:
        tokens = re.findall(r"[a-z0-9-]+", name.replace(" x ", "x "))
        if all(difflib.get_close_matches(w, tokens, n=1, cutoff=0.8) for w in words):
            return idx
    return None


def resolve(kind: str) -> int | None:
    """Device index for *kind* ('input'/'output'), or None for the default."""
    env = os.getenv("ORION_AUDIO_INPUT" if kind == "input" else "ORION_AUDIO_OUTPUT", "")
    if env.strip():
        idx = _match(env, kind)
        if idx is not None:
            return idx
    saved = _load_config().get(kind)
    if saved is not None:
        idx = _match(str(saved), kind)
        if idx is not None:
            return idx
    return None      # PortAudio default


def note_live_device(kind: str, index: int | None) -> None:
    """Record the device the live stream just actually opened on.

    Called by the audio engine itself (AudioPlaybackThread / MicrophoneEngine)
    right after a reopen succeeds — never by anything that's only requesting a
    change, so this always reflects reality, not intent."""
    kind = "input" if str(kind).lower().startswith("in") else "output"
    _live[kind] = index


def resolve_effective(kind: str) -> int | None:
    """The device index actually in effect right now, live override first.

    Differs from resolve() in that a live swap (e.g. via the GUI picker or a
    round-robin switch) shows up immediately here even before/without being
    persisted to config — resolve() alone would keep reporting the old saved
    device until the next restart."""
    kind = "input" if str(kind).lower().startswith("in") else "output"
    if kind in _live:
        return _live[kind]
    return resolve(kind)


def next_index(kind: str, current: int | None) -> tuple[int, str] | None:
    """The next device of *kind* after *current*, for round-robin switching.

    Returns (index, name), or None when there is only one (or zero) device of
    that kind — nothing to switch to."""
    kind = "input" if str(kind).lower().startswith("in") else "output"
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = _devices()
    candidates = [i for i, dev in enumerate(devices) if dev.get(channel_key, 0) > 0]
    if len(candidates) <= 1:
        return None
    if current in candidates:
        pos = (candidates.index(current) + 1) % len(candidates)
    else:
        pos = 0
    index = candidates[pos]
    return index, str(devices[index].get("name", "?"))


def device_name(kind: str) -> str:
    idx = resolve_effective(kind)
    devices = _devices()
    try:
        if idx is None:
            import sounddevice as sd
            default = sd.default.device[0 if kind == "input" else 1]
            idx = default if isinstance(default, int) and default >= 0 else None
        if idx is not None and 0 <= idx < len(devices):
            return f"[{idx}] {devices[idx].get('name', '?')}"
    except Exception:
        pass
    return "system default"


def set_device(kind: str, spec: str) -> str:
    """Persist a device choice; returns a human-readable confirmation.

    Persists the SPEC itself (a name fragment, e.g. "fifine"), not the index
    it currently resolves to.  Windows renumbers a device's port label as
    it's replugged or re-enumerated at boot (observed on this machine: the
    same physical Fifine microphone moved from index [16] labelled
    "2- fifine Microphone" to the same index later labelled "3- fifine
    Microphone") — an index saved once can silently point at a DIFFERENT
    device after a renumbering. A name fragment re-matches fresh against the
    current device list every time resolve() is called, so it keeps finding
    the right physical device regardless of how Windows re-numbers it. An
    explicit numeric spec (the user asking for "device 16" specifically) is
    still honoured literally — only name fragments get this protection,
    which is exactly the case that needs it.
    """
    kind = "input" if str(kind).lower().startswith("in") else "output"
    spec = str(spec).strip()
    if spec.lower() in {"default", "reset", "auto", ""}:
        cfg = _load_config()
        cfg.pop(kind, None)
        _save_config(cfg)
        return f"{kind.capitalize()} device reset to the system default."
    idx = _match(spec, kind)
    if idx is None:
        return (f"No {kind} device matches '{spec}'. Use 'audio_devices list' "
                "to see the options.")
    cfg = _load_config()
    name = _devices()[idx].get("name", "?")
    # A bare number is a PortAudio index, which moves when a device is plugged
    # in or out: "change to fifine" was saved as "4". Keep the name instead.
    cfg[kind] = name if spec.isdigit() and name != "?" else spec
    _save_config(cfg)
    return (f"{kind.capitalize()} device set to [{idx}] {name}. "
            "It takes effect the next time ORION starts (or restart the voice).")


#: The devices ORION should return to whenever he wakes from standby, in
#: preference order.  Name fragments, never indices: Windows renumbers a
#: device's port label between boots (the same physical Fifine has been seen at
#: "2- fifine Microphone" and "3- fifine Microphone"), so a saved index can
#: silently point at a different device after a re-enumeration.
#:
#: These are the user's stated hardware.  Neither is guaranteed to be plugged
#: in — on the machine this was written on, NEITHER was present — so every
#: consumer must treat "not found" as an ordinary outcome and say so out loud
#: rather than failing or pretending it succeeded.
PREFERRED_DEVICES: dict[str, tuple[str, ...]] = {
    "output": ("xrocker", "x-rocker", "x rocker"),
    "input": ("fifine",),
}


class DeviceCheck:
    """The result of asking for a device and actually proving it works.

    ``found``      the name fragment matched a device in the current list
    ``opened``     a real stream was opened on it (the only proof that counts)
    ``index``      the resolved device index, or None for the system default
    ``name``       what the device is actually called right now
    ``detail``     a human sentence for the log and the health panel
    """

    __slots__ = ("kind", "requested", "found", "opened", "index", "name", "detail")

    def __init__(self, kind: str, requested: str, found: bool, opened: bool,
                 index: int | None, name: str, detail: str) -> None:
        self.kind = kind
        self.requested = requested
        self.found = found
        self.opened = opened
        self.index = index
        self.name = name
        self.detail = detail

    @property
    def ok(self) -> bool:
        return self.opened

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"DeviceCheck({self.kind}, requested={self.requested!r}, "
                f"found={self.found}, opened={self.opened}, "
                f"index={self.index}, name={self.name!r})")


@_holds_portaudio_stream
def probe(kind: str, index: int | None) -> tuple[bool, str]:
    """Open and immediately close a stream on *index*.  Proof, not assumption.

    ``resolve()`` only tells you a NAME matched an entry in PortAudio's device
    table.  It says nothing about whether the device can actually be opened —
    a disconnected Bluetooth headset, a device held exclusively by another
    application, and a stale entry for unplugged hardware all still appear in
    the list.  Standby recovery was "restoring" devices it had never once
    verified, which is why ORION could come back apparently healthy and be
    unable to speak or hear.

    Returns (ok, detail); never raises.
    """
    kind = "input" if str(kind).lower().startswith("in") else "output"
    try:
        import sounddevice as sd
        from .constants import CHANNELS, RECEIVE_SAMPLE_RATE, SEND_SAMPLE_RATE
    except Exception as exc:            # pragma: no cover - env dependent
        return False, f"audio stack unavailable ({exc})"
    try:
        if kind == "input":
            stream = sd.RawInputStream(
                samplerate=SEND_SAMPLE_RATE, channels=CHANNELS,
                dtype="int16", blocksize=0, device=index)
        else:
            stream = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE, channels=CHANNELS,
                dtype="int16", blocksize=0, device=index)
        try:
            stream.start()
            stream.stop()
        finally:
            stream.close()
        return True, "opened"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def verify(kind: str, spec: str = "", *, probe_device: bool = True) -> DeviceCheck:
    """Resolve *spec* for *kind* and prove the device opens.

    When *spec* is empty the preferred device for that kind is tried first
    (XRocker for output, Fifine for input), then whatever is configured, then
    the system default.  Falling back is always reported explicitly — a silent
    fallback to the default is exactly how "ORION lost the microphone" looked
    like nothing at all in the log.
    """
    kind = "input" if str(kind).lower().startswith("in") else "output"
    requested = str(spec or "").strip()
    candidates: list[tuple[str, int | None]] = []

    if requested:
        idx = _match(requested, kind)
        if idx is not None:
            candidates.append((requested, idx))
    else:
        for fragment in PREFERRED_DEVICES.get(kind, ()):
            idx = _match(fragment, kind)
            if idx is not None:
                candidates.append((fragment, idx))
                break
        configured = resolve(kind)
        if configured is not None and all(c[1] != configured for c in candidates):
            candidates.append((str(_load_config().get(kind) or configured), configured))

    found = bool(candidates)
    # The system default is always the last resort: ORION being audible on the
    # wrong speaker beats ORION being silent.
    candidates.append(("system default", None))

    for label, index in candidates:
        if not probe_device:
            name = device_name(kind) if index is None else _name_of(index)
            return DeviceCheck(kind, requested or label, found, True, index, name,
                               f"selected {name} (unverified)")
        ok, detail = probe(kind, index)
        name = device_name(kind) if index is None else _name_of(index)
        if ok:
            note = (f"{kind} verified on {name}" if index is not None
                    else f"{kind} verified on the system default ({name})")
            if requested and label != requested:
                note += f" — '{requested}' was not usable"
            elif not requested and index is None and found:
                note += " — the preferred device would not open"
            elif not requested and index is None and not found:
                preferred = "/".join(PREFERRED_DEVICES.get(kind, ())) or "preferred"
                note += f" — no {preferred} device is attached"
            return DeviceCheck(kind, requested or label, found, True, index, name, note)
        last = detail
    preferred = "/".join(PREFERRED_DEVICES.get(kind, ())) or "preferred"
    return DeviceCheck(
        kind, requested or preferred, found, False, None, "none",
        f"no {kind} device could be opened at all ({last})")


def _name_of(index: int | None) -> str:
    if index is None:
        return "system default"
    devices = _devices()
    if 0 <= index < len(devices):
        return f"[{index}] {devices[index].get('name', '?')}"
    return f"[{index}] (gone)"


def preferred_index(kind: str) -> int | None:
    """The index of the user's preferred device for *kind*, or None if absent.

    Absent is a normal answer — it means the hardware is not plugged in.
    """
    kind = "input" if str(kind).lower().startswith("in") else "output"
    for fragment in PREFERRED_DEVICES.get(kind, ()):
        idx = _match(fragment, kind)
        if idx is not None:
            return idx
    return None


def list_devices() -> str:
    devices = _devices()
    if not devices:
        return "No audio devices found (sounddevice/PortAudio unavailable)."
    ins, outs = [], []
    for idx, dev in enumerate(devices):
        name = str(dev.get("name", "?"))
        if dev.get("max_input_channels", 0) > 0:
            ins.append(f"  [{idx}] {name}")
        if dev.get("max_output_channels", 0) > 0:
            outs.append(f"  [{idx}] {name}")
    lines = ["INPUT (microphones):", *ins, "", "OUTPUT (speakers/headphones):", *outs]
    return "\n".join(lines)


def describe() -> str:
    return (f"ORION is listening on:  {device_name('input')}\n"
            f"ORION is speaking to:   {device_name('output')}")


def log_startup(bus: Any) -> None:
    """Emit the active devices at boot so the user can see input/output."""
    try:
        bus.log.emit(f"AUDIO: input (mic) = {device_name('input')}")
        bus.log.emit(f"AUDIO: output (voice) = {device_name('output')}")
    except Exception:
        pass


def microphone_suggestion(active: int | None) -> str:
    """Offer a closer microphone once; never select or open one here."""
    # Offline/live reconnects can briefly have overlapping startup threads.
    with _suggestion_lock:
        return _microphone_suggestion(active)


def _microphone_suggestion(active: int | None) -> str:
    global _suggestion_pending_until
    cfg = _load_config()
    if cfg.get("microphone_guidance") in {"offered", "declined"}:
        return ""
    devices = _devices()
    if active is None:
        try:
            import sounddevice as sd
            active = int(sd.default.device[0])
        except Exception:
            return ""
    if not isinstance(active, int) or not 0 <= active < len(devices):
        return ""
    current = str(devices[active].get("name", "")).lower()
    if not any(word in current for word in (*_WEBCAM_HINTS, "array")):
        return ""
    candidates = []
    for idx, record in enumerate(devices):
        name = str(record.get("name", ""))
        low = name.lower()
        if (idx == active or not record.get("max_input_channels", 0)
                or any(word in low for word in (*_VIRTUAL_HINTS, *_WEBCAM_HINTS, "array"))):
            continue
        headset = any(word in low for word in _HEADSET_HINTS)
        if headset or "usb" in low or "fifine" in low:
            candidates.append((not headset, -len(name), display_name(name)))
    if not candidates:
        return ""
    cfg["microphone_guidance"] = "offered"
    _save_config(cfg)
    _suggestion_pending_until = time.monotonic() + 90.0
    kind = "webcam" if any(w in current for w in _WEBCAM_HINTS) else "array"
    return (f"You're using the {kind} microphone. {min(candidates)[2]} is also connected; "
            "a microphone closer to you may help me hear names. Say 'switch microphone' "
            "to try another input, or 'no thanks' to keep this one.")


def decline_microphone_suggestion(text: str) -> bool:
    """Remember a narrow 'no' only while that suggestion is pending."""
    with _suggestion_lock:
        return _decline_microphone_suggestion(text)


def _decline_microphone_suggestion(text: str) -> bool:
    global _suggestion_pending_until
    if time.monotonic() >= _suggestion_pending_until:
        return False
    reply = str(text or "").lower().strip(" ,.!?")
    reply = re.sub(r"^(?:orion|ryan)[ ,]+", "", reply)
    if reply not in {"no", "no thanks", "no thank you", "keep this microphone", "keep this one"}:
        return False
    cfg = _load_config()
    cfg["microphone_guidance"] = "declined"
    _save_config(cfg)
    _suggestion_pending_until = 0.0
    return True


# ──────────────────────────────────────────────────────────────────────────────
# MEASURED TRANSPORT  —  does audio actually FLOW, not merely does it open
# ──────────────────────────────────────────────────────────────────────────────
#
# ``probe()`` above opens a stream and closes it, which is real proof that a
# device exists and is not held by another application. It is not proof that
# sound reaches the speaker.
#
# On Windows, PortAudio's DirectSound output is a silent sink: the stream opens,
# every write returns success in approximately zero time, and not one sample is
# ever heard. Nothing in the device record reports this — no capability flag, no
# error, no warning. An open-and-close probe passes it happily, and ORION would
# report healthy audio while saying nothing at all.
#
# The only way to tell is to time a write. A device consuming audio in real time
# takes about as long to accept two seconds of samples as it takes to play them;
# a sink that is swallowing them returns immediately:
#
#     write(2.0 s of silence) took        verdict
#     ------------------------------      -----------------------------
#     MME                    2.02 s       consumed in real time
#     DirectSound            0.00 s       swallowed instantly
#
# Silence is used so the probe is inaudible, and the result is cached per
# (host API, direction) because the property belongs to the API, not the device.
#
# The direction matters as much as the device: the probe must run in the SAME
# MODE the application ships. ORION writes to output with a blocking
# ``RawOutputStream.write`` (audio.py:340) and reads input through a
# ``RawInputStream`` CALLBACK (audio.py:2255). Probing input with a blocking
# read rejects a microphone that works perfectly in the app.

#: How long to exercise each direction. Output needs long enough that a real
#: device's buffering cannot hide the difference; input only needs enough
#: callbacks to be sure frames are arriving.
_TRANSPORT_SECONDS = {"output": 0.6, "input": 0.35}

#: A real device should take at least this fraction of real time to accept the
#: audio. Anything faster is not playing it.
_TRANSPORT_MIN_RATIO = 0.5

#: An input should deliver at least this fraction of the frames it promised.
_TRANSPORT_MIN_FRAMES = 0.2

_transport_cache: dict[tuple[str, int | None], tuple[bool, str]] = {}


def _host_api_of(index: int | None) -> str:
    """Name of the host API carrying *index*, or "" if it cannot be determined."""
    _wait_portaudio_up()
    try:
        import sounddevice as sd

        if index is None:
            return ""
        record = sd.query_devices(index)
        return str(sd.query_hostapis(record["hostapi"])["name"])
    except Exception:
        return ""


@_holds_portaudio_stream
def transport_works(kind: str, index: int | None,
                    *, use_cache: bool = True) -> tuple[bool, str]:
    """Measure whether audio actually moves through *index*.

    Returns (ok, detail) and never raises. See the note above for why opening a
    stream is not enough.
    """
    kind = "input" if str(kind).lower().startswith("in") else "output"
    key = (kind, index)
    if use_cache and key in _transport_cache:
        return _transport_cache[key]

    try:
        import sounddevice as sd

        from .constants import CHANNELS, RECEIVE_SAMPLE_RATE, SEND_SAMPLE_RATE
    except Exception as exc:                       # pragma: no cover - env
        return False, f"audio stack unavailable ({exc})"

    seconds = _TRANSPORT_SECONDS[kind]
    try:
        if kind == "output":
            rate = RECEIVE_SAMPLE_RATE
            stream = sd.RawOutputStream(samplerate=rate, channels=CHANNELS,
                                        dtype="int16", blocksize=1024,
                                        device=index)
            try:
                stream.start()
                quiet = bytes(int(rate * seconds) * 2 * CHANNELS)
                started = time.monotonic()
                stream.write(quiet)               # blocking: the shipped mode
                elapsed = time.monotonic() - started
            finally:
                try:
                    stream.stop()
                finally:
                    stream.close()
            ok = elapsed > seconds * _TRANSPORT_MIN_RATIO
            detail = (f"consumed {elapsed:.2f}s of {seconds:.2f}s"
                      if ok else
                      f"swallowed {seconds:.2f}s in {elapsed:.2f}s "
                      f"— opens but plays nothing")
        else:
            rate = SEND_SAMPLE_RATE
            delivered = [0]

            def _sink(_indata, frames, _time, _status) -> None:
                delivered[0] += int(frames)

            stream = sd.RawInputStream(samplerate=rate, channels=CHANNELS,
                                       dtype="int16", blocksize=1024,
                                       device=index, callback=_sink)
            try:
                stream.start()
                time.sleep(seconds)
            finally:
                try:
                    stream.stop()
                finally:
                    stream.close()
            expected = rate * seconds
            ok = delivered[0] > expected * _TRANSPORT_MIN_FRAMES
            detail = (f"delivered {delivered[0]} of ~{int(expected)} frames"
                      if ok else
                      f"delivered only {delivered[0]} of ~{int(expected)} frames")
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"

    result = (bool(ok), detail)
    _transport_cache[key] = result
    return result


# ──────────────────────────────────────────────────────────────────────────────
# A SHORT, HONEST DEVICE LIST
# ──────────────────────────────────────────────────────────────────────────────
#
# ``sd.query_devices()`` returns one row per device PER HOST API, not one per
# device. Measured on this machine: 68 rows, 29 of them inputs and 39 outputs,
# for 21 and 26 distinct names — the same microphone appearing under MME,
# DirectSound, WASAPI and WDM-KS with nothing to say which one to choose. That
# is not a choice, it is a quiz.
#
# Four of the rows are not hardware at all but aliases meaning "whatever the OS
# picks", which is already offered as the default entry.

_PSEUDO_DEVICES = (
    "sound mapper", "primary sound", "sysdefault", "default", "dmix",
    "dsnoop", "surround", "samplerate", "speexrate", "upmix", "vdownmix",
    "null", "pulse",
)


def _is_pseudo(name: str) -> bool:
    lowered = (name or "").lower()
    return any(token in lowered for token in _PSEUDO_DEVICES)


#: Windows hands back unresolved driver resource strings for some endpoints —
#: chiefly Bluetooth. The raw name of a perfectly ordinary headset is:
#:
#:     Headset (@System32\drivers\bthhfenum.sys,#2;%1 Hands-Free%0\r\n;(HyperX Cloud III S Wireless))
#:
#: Seven of the thirty-nine names on the development machine look like that.
#: The point of this module is a list a person can choose from, and a list of
#: driver paths is not one — the user cannot tell which line is their headset,
#: which is exactly the problem the deduplication was meant to solve.
#:
#: The human name is the last parenthesised group; everything before it is the
#: localisation reference Windows failed to resolve.
_RESOURCE_NAME = __import__("re").compile(r";\(([^()]+)\)\s*\)\s*$")


def display_name(name: str) -> str:
    """A device name fit to put in front of a person.

    Returns *name* unchanged when it is already readable — most are. Never
    raises, and never returns empty: an unreadable label is better than none.
    """
    raw = (name or "").strip()
    if not raw:
        return raw
    if "@" not in raw:
        return raw
    match = _RESOURCE_NAME.search(raw)
    if not match:
        # Unrecognised shape: strip the control characters at least, so the
        # combo box does not render a literal newline mid-entry.
        return " ".join(raw.split())
    human = " ".join(match.group(1).split())
    kind = raw.split("(", 1)[0].strip()      # "Headset", "Input", "Output"
    if kind and kind.lower() not in human.lower():
        return f"{human} ({kind.lower()})"
    return human or " ".join(raw.split())


def usable_devices(kind: str) -> list[tuple[int, str]]:
    """(index, name) for the devices worth offering, deduplicated by name.

    Ordered so the host APIs most likely to carry audio come first, then by
    index. Deduplication is by NAME rather than by index because the duplicates
    ARE the same physical device — keeping the first one that appears under a
    preferred API is the whole reduction.

    This does not open anything: it is safe to call on the GUI thread. Use
    ``probe()`` for "can it open" and ``transport_works()`` for "does sound
    actually move".
    """
    kind = "input" if str(kind).lower().startswith("in") else "output"
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    _wait_portaudio_up()
    try:
        import sounddevice as sd

        records = list(sd.query_devices())
        api_names = [str(a["name"]).lower() for a in sd.query_hostapis()]
    except Exception:
        return []

    # If a direction has already been measured, put its winning API first;
    # otherwise fall back to the starting order.
    measured = _chosen_api.get(kind)
    order = list(_PREFERRED_HOST_APIS)
    if measured in order:
        order.remove(measured)
        order.insert(0, measured)

    def rank(record: dict[str, Any]) -> int:
        try:
            api = api_names[int(record["hostapi"])]
        except Exception:
            return len(order)
        for position, token in enumerate(order):
            if token in api:
                return position
        return len(order)

    candidates = []
    for index, record in enumerate(records):
        try:
            if int(record.get(channel_key, 0)) <= 0:
                continue
            name = str(record.get("name", "")).strip()
        except Exception:
            continue
        if not name or _is_pseudo(name):
            continue
        candidates.append((rank(record), index, name))

    candidates.sort(key=lambda row: (row[0], row[1]))

    # MME truncates device names to 31 characters, so the API that actually
    # carries the audio is frequently not the one that can spell. The result is
    # that these arrive as two separate entries for one device:
    #
    #     "Digital Audio (S/PDIF) (High De"
    #     "Digital Audio (S/PDIF) (High Definition Audio Device)"
    #
    # Deduplicating on the exact string keeps both, which defeats the entire
    # point of the reduction: the user is handed two lines and no way to tell
    # that they are the same socket. A name that is a prefix of a longer one is
    # treated as the same device, and the longer spelling wins the label while
    # the higher-ranked API keeps the index that will actually be opened.
    full_names = sorted({name for _r, _i, name in candidates}, key=len, reverse=True)

    def canonical(name: str) -> str:
        if len(name) < 30:
            return name
        for longer in full_names:
            if longer != name and longer.startswith(name):
                return longer
        return name

    seen: set[str] = set()
    out: list[tuple[int, str]] = []
    for _rank, index, name in candidates:
        key = canonical(name)
        if key in seen:
            continue
        seen.add(key)
        out.append((index, key))
    return out


#: Host APIs worth trying, in order, by substring. A starting order only —
#: ``preferred_host_api`` MEASURES and the measurement wins.
#:
#: The order matters because the obvious one is wrong. Ranking by reputation
#: puts WASAPI first, and WASAPI in shared mode does not resample: against
#: 48 kHz hardware, ORION's 16 kHz input and 24 kHz output make EVERY open fail
#: with "Invalid sample rate". Measured here, on this machine:
#:
#:     host API        output (blocking write)   input (callback)
#:     -----------     ----------------------    ----------------
#:     MME             0.67 s  real              frames arriving
#:     DirectSound     0.00 s  SILENT SINK       frames arriving
#:     WASAPI          open fails (rate)         open fails (rate)
#:     WDM-KS          open fails (rate)         open fails (rate)
#:
#: DirectSound is the dangerous one: it opens, accepts every write instantly and
#: plays nothing, so an assistant that chose it would be mute while reporting
#: healthy audio. That is the whole reason this file measures instead of ranking.
_PREFERRED_HOST_APIS = ("mme", "directsound", "wasapi", "wdm-ks",
                        "core audio", "pulse", "pipewire", "jack", "alsa")

_chosen_api: dict[str, str | None] = {}


def preferred_host_api(kind: str) -> str | None:
    """The host API that actually carries audio for *kind*, measured once.

    Each direction chooses independently, because they genuinely differ: the
    same machine can need one API to hear and another to speak, and no amount
    of reasoning about API quality would produce that split.

    Returns the matching substring from ``_PREFERRED_HOST_APIS``, or None when
    nothing passed (the caller then falls back to the system default, which is
    what ORION did before any of this existed).
    """
    kind = "input" if str(kind).lower().startswith("in") else "output"
    if kind in _chosen_api:
        return _chosen_api[kind]

    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    chosen: str | None = None
    _wait_portaudio_up()
    try:
        import sounddevice as sd

        records = list(sd.query_devices())
        api_names = [str(a["name"]).lower() for a in sd.query_hostapis()]
        for token in _PREFERRED_HOST_APIS:
            for index, record in enumerate(records):
                try:
                    if int(record.get(channel_key, 0)) <= 0:
                        continue
                    if token not in api_names[int(record["hostapi"])]:
                        continue
                    if _is_pseudo(str(record.get("name", ""))):
                        continue
                except Exception:
                    continue
                ok, _detail = transport_works(kind, index)
                if ok:
                    chosen = token
                break                       # one probe per API per direction
            if chosen:
                break
    except Exception:
        chosen = None

    _chosen_api[kind] = chosen
    return chosen


def working_devices(kind: str, *, limit: int = 6) -> list[tuple[int, str]]:
    """``usable_devices`` filtered to those that actually carry audio.

    Measured, so it is slow — a few hundred milliseconds per device — and must
    never be called on the GUI thread or during startup. ``limit`` caps how many
    are exercised so a machine with twenty endpoints cannot stall a settings
    dialog; the cap is reported by the caller rather than hidden.
    """
    out: list[tuple[int, str]] = []
    for index, name in usable_devices(kind)[:max(1, limit)]:
        ok, _detail = transport_works(kind, index)
        if ok:
            out.append((index, name))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PORTAUDIO RE-INITIALISATION  —  noticing a device that appeared or vanished
# ──────────────────────────────────────────────────────────────────────────────
#
# PortAudio reads the machine's devices ONCE, when it is initialised, and every
# device index ORION ever holds is an index into that snapshot. On 20 Oct the
# owner's Bluetooth headset switched off; Windows moved its default output to
# the monitor; the monitor's index in the snapshot no longer matched Windows'
# own numbering, and every open failed with "A device ID has been used that is
# out of range for your system". The renderer retried once a second and logged
# each failure — 799 lines in fourteen minutes — and ORION stayed mute until the
# process was killed. Nothing ever refreshed the snapshot.
#
# Refreshing it means sd._terminate() then sd._initialize(), and that is the
# call audio_follow.py's history warns about: made while a stream is open, it
# frees the state the stream still points at, and the next touch of that stream
# is STATUS_HEAP_CORRUPTION (a fault dump named it inside a minute, 2026-09-20).
#
# So the only rule here is that PortAudio is never terminated while a stream
# exists. The coordinator below enforces it rather than hoping:
#
#   1. every owner of a long-lived stream (the voice renderer, the microphone)
#      is asked to close it at ITS next safe point — the coordinator never
#      touches a stream itself, so no stream is ever closed under its own user;
#   2. it waits, bounded, until each owner says it is closed AND no probe or
#      play()/rec() stream is open anywhere;
#   3. if anyone has not let go in time it ABANDONS the attempt, hands every
#      stream back, and says so — a mute ORION is recoverable, a crashed one
#      is not;
#   4. only then does it terminate and initialise, under a lock, on this one
#      thread, and let the owners reopen against the fresh list.

#: Fragments of the errors PortAudio gives when an index no longer matches the
#: machine (-9996 paInvalidDevice, -9999 host error "device ID ... out of
#: range", -9985 paDeviceUnavailable).
STALE_LIST_MARKERS = ("invalid device", "out of range", "-9996", "-9999", "-9985",
                      "device unavailable")


def looks_like_stale_device_list(error: object) -> bool:
    """Whether *error* reads like PortAudio holding an out-of-date device list."""
    text = str(error).lower()
    return any(marker in text for marker in STALE_LIST_MARKERS)


class FailureLog:
    """Say a repeating failure once, then count it instead of repeating it.

    The first occurrence of a message is emitted at once. Identical ones after
    that are only counted, and a summary is emitted when one arrives at least
    *interval* seconds after the last line. reset() — call it on recovery —
    reports whatever was still being held back, so a count is never lost.
    """

    def __init__(self, emit: Callable[[str], Any], interval: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._emit = emit
        self._interval = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: dict[str, list[Any]] = {}      # key -> [last line at, held back, text]

    def note(self, message: str, key: str | None = None) -> None:
        key = key or message
        now = self._clock()
        with self._lock:
            entry = self._seen.get(key)
            if entry is None:
                if len(self._seen) >= 16:          # never grow without bound
                    self._seen.clear()
                self._seen[key] = [now, 0, message]
                line = message
            else:
                entry[1] += 1
                entry[2] = message
                if now - entry[0] < self._interval:
                    return
                line = (f"{message} (repeated {entry[1]} more time"
                        f"{'s' if entry[1] != 1 else ''} in the last {int(now - entry[0])} s)")
                entry[0], entry[1] = now, 0
        self._emit(line)

    def reset(self) -> None:
        """The failure stopped: report what was held back and start afresh."""
        with self._lock:
            pending = [(held, text) for _at, held, text in self._seen.values() if held]
            self._seen.clear()
        for held, text in pending:
            self._emit(f"{text} (repeated {held} more time{'s' if held != 1 else ''} "
                       "before it recovered)")


def device_snapshot(index: int | None) -> tuple[str, int] | None:
    """(name, host API) of PortAudio device *index* as the list stands NOW.

    Indices are renumbered by a re-initialisation, so anything holding one
    across it — a microphone picked by round-robin, a switch still queued —
    keeps this instead and asks device_from_snapshot() afterwards.
    """
    if index is None:
        return None
    devices = _devices()
    if 0 <= index < len(devices):
        record = devices[index]
        try:
            return str(record.get("name", "")), int(record.get("hostapi", -1))
        except (TypeError, ValueError):
            return str(record.get("name", "")), -1
    return None


def device_from_snapshot(kind: str, snapshot: tuple[str, int] | None) -> int | None:
    """Where the device described by *snapshot* sits now, or None if it is gone."""
    if snapshot is None:
        return None
    name, api = snapshot
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    fallback = None
    for index, record in enumerate(_devices()):
        try:
            if int(record.get(channel_key, 0)) <= 0 or str(record.get("name", "")) != name:
                continue
            if int(record.get("hostapi", -1)) == api:
                return index                       # same name AND same host API
        except (TypeError, ValueError):
            continue
        if fallback is None:
            fallback = index
    return fallback


def _real_portaudio() -> Any | None:
    """The sounddevice module, or None when it must not be re-initialised.

    Never the real one under pytest: a test that happens to start the renderer
    on a machine with no usable device would otherwise enumerate (and tear
    down) the developer's actual audio hardware. Tests install a fake module,
    which has no ``_lib``, and that is allowed.
    """
    try:
        import sounddevice as sd
    except Exception:
        return None
    if "pytest" in sys.modules and hasattr(sd, "_lib"):
        return None
    return sd


class PortAudioReinit:
    """Re-initialise PortAudio only when every stream ORION holds is closed.

    Owners register as *participants*. A participant provides:

    ``reinit_release()``   ask it to close its stream at its next safe point and
                           stay closed (non-blocking; it is called from here)
    ``reinit_released()``  True once no PortAudio stream of its is open
    ``reinit_restore()``   it may open again — resolve() afresh
    ``reinit_log(text)``   where to write the one-line outcome
    ``reinit_label``       how to name it in that line ("the microphone")
    """

    #: How long the owners get to close before the attempt is abandoned.
    CLOSE_TIMEOUT = 3.0
    #: A request this soon after the last attempt is the same event, unless forced.
    MIN_INTERVAL = 3.0
    #: Polling step while waiting. Short: the wait is bounded by CLOSE_TIMEOUT.
    POLL = 0.02

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] = time.sleep,
                 portaudio: Callable[[], Any] | None = None,
                 close_timeout: float | None = None,
                 min_interval: float | None = None) -> None:
        self._clock = clock
        self._sleep = sleep
        self._portaudio = portaudio or _real_portaudio
        self.close_timeout = self.CLOSE_TIMEOUT if close_timeout is None else close_timeout
        self.min_interval = self.MIN_INTERVAL if min_interval is None else min_interval
        self._participants: "weakref.WeakSet[Any]" = weakref.WeakSet()
        self._lock = threading.Lock()              # one re-init at a time
        self._last_attempt = float("-inf")
        #: Optional override for where the outcome lines go.
        self.log: Callable[[str], Any] | None = None
        self._abort_log = FailureLog(self._emit, 60.0, clock)
        self.completed = 0
        self.abandoned = 0                         # owners never let go in time
        self.failed = 0                            # PortAudio itself refused

    def register(self, participant: Any) -> None:
        self._participants.add(participant)

    def unregister(self, participant: Any) -> None:
        self._participants.discard(participant)

    def _emit(self, message: str) -> None:
        sinks: list[Callable[[str], Any]] = []
        if self.log is not None:
            sinks.append(self.log)
        else:
            for participant in list(self._participants):
                sink = getattr(participant, "reinit_log", None)
                if callable(sink):
                    sinks.append(sink)
        for sink in sinks:
            try:
                sink(message)
                return                             # one line, one place
            except Exception:
                continue

    # -- entry points --------------------------------------------------------

    def request_reinit(self, reason: str, *, force: bool = False) -> bool:
        """Refresh PortAudio's device list. True only if it actually happened.

        Blocks for up to ``close_timeout`` plus the re-initialisation itself, so
        it belongs on a worker thread, never the event loop — and never on a
        thread that owns a participant's stream (that owner could not close it
        while waiting here); those use request_reinit_async().

        A request that arrives while one is running is covered by it. One that
        follows closely behind a finished attempt is dropped as the same event
        unless *force* (a real Windows default change always forces: the list
        must reflect it).
        """
        global _reinit_running
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if not force and self._clock() - self._last_attempt < self.min_interval:
                return False
            sd = self._portaudio()
            if sd is None:
                return False
            participants = list(self._participants)
            with _gate:
                _reinit_running = True             # new probes now wait
            try:
                return self._reinit(reason, sd, participants)
            finally:
                with _gate:
                    _reinit_running = False
                    _gate.notify_all()
                self._last_attempt = self._clock()
        finally:
            self._lock.release()

    def request_reinit_async(self, reason: str, *, force: bool = False) -> bool:
        """request_reinit() on a short-lived thread. False if one is already running."""
        if self._lock.locked():
            return False
        threading.Thread(target=self.request_reinit, args=(reason,),
                         kwargs={"force": force}, name="orion-audio-reinit",
                         daemon=True).start()
        return True

    # -- the attempt ---------------------------------------------------------

    def _reinit(self, reason: str, sd: Any, participants: list[Any]) -> bool:
        started = self._clock()
        for participant in participants:
            try:
                participant.reinit_release()
            except Exception:
                pass                               # still unproven closed below
        deadline = started + self.close_timeout
        while True:
            holders = self._holders(sd, participants)
            if not holders:
                break
            if self._clock() >= deadline:
                return self._abandon(reason, holders, participants)
            self._sleep(self.POLL)

        # Every stream is closed, the owners are parked and new probes are
        # held off: nothing can open until we are done.
        global _portaudio_down
        failure: Exception | None = None
        terminated = False
        with _gate:
            _portaudio_down = True                 # device queries wait, too
        try:
            sd._terminate()
            terminated = True
            sd._initialize()
        except Exception as exc:
            failure = exc
            if terminated:
                # Terminated but not restarted: PortAudio is DOWN, which is
                # worse than stale. One more try; the owners' own retry loops
                # (and the next request) take it from there.
                try:
                    sd._initialize()
                except Exception:
                    pass
        finally:
            with _gate:
                _portaudio_down = False
                _gate.notify_all()
        _transport_cache.clear()                   # keyed by indices that just moved
        _live.clear()
        for participant in participants:
            try:
                participant.reinit_restore()
            except Exception:
                pass
        if failure is not None:
            self.failed += 1
            self._emit(f"AUDIO: PortAudio could not be re-initialised ({reason}) - {failure}")
            return False
        self.completed += 1
        elapsed_ms = (self._clock() - started) * 1000.0
        self._emit(f"AUDIO: PortAudio re-initialised ({reason}) in {elapsed_ms:.0f} ms — "
                   f"{self._describe(sd)}.")
        return True

    def _holders(self, sd: Any, participants: list[Any]) -> list[str]:
        """Who still has a stream open (empty = safe to terminate)."""
        holders: list[str] = []
        for participant in participants:
            label = getattr(participant, "reinit_label", type(participant).__name__)
            try:
                if not participant.reinit_released():
                    holders.append(label)
            except Exception:
                holders.append(label)              # cannot prove it closed
        with _gate:
            if _transient_streams:
                holders.append("a device probe")
        if self._foreign_stream_active(sd):
            holders.append("a playback/recording stream")
        return holders

    @staticmethod
    def _foreign_stream_active(sd: Any) -> bool:
        """A stream started by sd.play()/sd.rec(), which nothing registers."""
        try:
            stream = sd.get_stream()
        except Exception:
            return False                           # none yet (or no such API)
        try:
            return bool(stream.active)
        except Exception:
            return False                           # closed: nothing to protect

    def _abandon(self, reason: str, holders: list[str], participants: list[Any]) -> bool:
        """Give every stream back and leave PortAudio exactly as it was."""
        for participant in participants:
            try:
                participant.reinit_restore()
            except Exception:
                pass
        self.abandoned += 1
        who = " and ".join(sorted(set(holders)))
        self._abort_log.note(
            f"AUDIO: PortAudio was NOT re-initialised ({reason}): {who} still held a "
            f"stream after {self.close_timeout:g} s, and terminating PortAudio under an "
            "open stream crashes the process. Everything was left as it was.",
            key="reinit-abandoned")
        return False

    def _describe(self, sd: Any) -> str:
        """'N devices listed; default output: X; default input: Y' by friendly name."""
        parts = []
        try:
            parts.append(f"{len(list(sd.query_devices()))} devices listed")
        except Exception:
            pass
        for kind in ("output", "input"):
            name = _system_default_name(kind)
            if not name:
                try:
                    name = str(sd.query_devices(kind=kind)["name"])
                except Exception:
                    name = ""
            if name:
                parts.append(f"default {kind}: {display_name(name)}")
        return "; ".join(parts) or "device list refreshed"


#: The one coordinator ORION uses. Owners register themselves on construction,
#: so nothing has to be wired in app.py.
COORDINATOR = PortAudioReinit()


def register_reinit_participant(participant: Any) -> None:
    """Make *participant* (see PortAudioReinit) part of every future re-init."""
    COORDINATOR.register(participant)


def request_reinit(reason: str, *, force: bool = False) -> bool:
    """Refresh PortAudio's device list now, from a worker thread. See the class."""
    return COORDINATOR.request_reinit(reason, force=force)


def request_reinit_async(reason: str, *, force: bool = False) -> bool:
    """request_reinit() on its own thread, for callers that own a stream."""
    return COORDINATOR.request_reinit_async(reason, force=force)
