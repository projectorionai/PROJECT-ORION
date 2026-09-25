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

import json
import os
import time
from typing import Any

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

AUDIO_CONFIG_PATH = CONFIG_DIR / "audio.json"

# In-memory record of the device actually applied to the live stream this
# session (as opposed to what's merely saved to config) — set by the audio
# engine itself whenever it reopens a stream, read by resolve_effective() so
# the GUI picker reflects reality even when a live swap hasn't been persisted.
_live: dict[str, int | None] = {}


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


def _devices() -> list[dict[str, Any]]:
    try:
        import sounddevice as sd
        return list(sd.query_devices())
    except Exception:
        return []


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
    cfg[kind] = spec
    _save_config(cfg)
    name = _devices()[idx].get("name", "?")
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
    try:
        import sounddevice as sd

        if index is None:
            return ""
        record = sd.query_devices(index)
        return str(sd.query_hostapis(record["hostapi"])["name"])
    except Exception:
        return ""


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
