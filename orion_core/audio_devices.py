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
from typing import Any

from .constants import CONFIG_DIR

AUDIO_CONFIG_PATH = CONFIG_DIR / "audio.json"


def _load_config() -> dict[str, Any]:
    try:
        data = json.loads(AUDIO_CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_config(cfg: dict[str, Any]) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
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


def device_name(kind: str) -> str:
    idx = resolve(kind)
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
    """Persist a device choice; returns a human-readable confirmation."""
    kind = "input" if str(kind).lower().startswith("in") else "output"
    if str(spec).strip().lower() in {"default", "reset", "auto", ""}:
        cfg = _load_config()
        cfg.pop(kind, None)
        _save_config(cfg)
        return f"{kind.capitalize()} device reset to the system default."
    idx = _match(spec, kind)
    if idx is None:
        return (f"No {kind} device matches '{spec}'. Use 'audio_devices list' "
                "to see the options.")
    cfg = _load_config()
    cfg[kind] = idx
    _save_config(cfg)
    name = _devices()[idx].get("name", "?")
    return (f"{kind.capitalize()} device set to [{idx}] {name}. "
            "It takes effect the next time ORION starts (or restart the voice).")


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
