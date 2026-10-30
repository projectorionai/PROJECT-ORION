"""
How far ORION reaches into Windows - one stored set of switches.

Phase 1 of the Windows integration (Mark XXXVII.1) gives ORION places in the
operating system itself: a global hotkey, a Screen Face docked on the screen
edge, "Ask ORION" in File Explorer, a taskbar jump list, orion:// links,
notifications with buttons, and awareness of lock, sleep and Do Not Disturb.
Each one is a switch here, so the owner can turn any of them off and the code
that registers it can be told to take it back out of Windows.

Stored beside the other preferences (CONFIG_DIR/windows_integration.json).
Reading order matches appearance.py: an environment override for a one-off
(ORION_WIN_<KEY>=0/1), then the stored choice, then the default.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .atomic_io import atomic_write_text
from .constants import CONFIG_DIR

SETTINGS_PATH = CONFIG_DIR / "windows_integration.json"

#: Every switch and its default. All on: the owner asked for ORION to be part
#: of Windows, and each piece is reversible from Preferences.
DEFAULTS: dict[str, Any] = {
    "start_in_tray_at_login": True, # sign-in starts the core in the tray, window on demand
    "hotkey_enabled": True,
    "hotkey": "win+alt+o",          # falls back to ctrl+alt+o if Windows refuses it
    "dock_enabled": True,           # the docked face is allowed at all
    "dock_on_startup": False,       # ...but it is NOT on screen at start; the owner asks
                                    # for it (tray "Show Dock", Preferences, "show your dock")
    "dock_edge": "right",           # right / left
    "explorer_menu": True,          # "Ask ORION" on files and folders
    "protocol": True,               # orion:// links
    "jump_list": True,              # taskbar right-click tasks
    "toasts": True,                 # Windows notifications with buttons
    "greet_on_unlock": True,
    "quiet_when_busy": True,        # Do Not Disturb, presentations, full-screen games
    "pause_hearing_when_locked": True,
}


def _load() -> dict[str, Any]:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get(key: str) -> Any:
    """The value in effect for *key* (unknown keys raise KeyError)."""
    default = DEFAULTS[key]
    override = os.getenv(f"ORION_WIN_{key.upper()}", "").strip().lower()
    if override and isinstance(default, bool):
        return override in {"1", "true", "on", "yes"}
    if override:
        return override
    stored = _load().get(key)
    if stored is None or type(stored) is not type(default):
        return default
    return stored


def snapshot() -> dict[str, Any]:
    return {key: get(key) for key in DEFAULTS}


def set_value(key: str, value: Any) -> bool:
    """Store *value* for *key*. Returns False (and stores nothing) for an
    unknown key or a value of the wrong type."""
    if key not in DEFAULTS or type(value) is not type(DEFAULTS[key]):
        return False
    data = _load()
    data[key] = value
    data.setdefault("schema", "orion.windows_integration.v1")
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(SETTINGS_PATH, json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


__all__ = ["DEFAULTS", "SETTINGS_PATH", "get", "set_value", "snapshot"]
