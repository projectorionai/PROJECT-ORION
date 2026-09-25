"""
Authorized-target store for security_recon.py.

A small, persistent set of hosts the user has explicitly said ORION may run
security_recon actions (scan_host, packet_capture, craft_packet) against —
the "authorized_scope" cyber_curriculum.validate_target_scope()/
gate_security_action() take. Loopback is always allowed regardless of this
store (see cyber_curriculum.py); everything else must be listed here first.

Adding a target is a direct, explicit user instruction ("authorize
192.168.1.50, that's my lab VM") — trusted the same way confirm=True is
already trusted elsewhere once the user has said so in conversation. There
is no extra gate on ADDING a target beyond that, since expanding scope is
exactly what the user is asking ORION to do when they say it.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

AUTHORIZED_TARGETS_PATH = CONFIG_DIR / "authorized_targets.json"

_lock = RLock()


def _normalise(target: str) -> str:
    return str(target or "").strip().lower()


def load_authorized_targets(path: Path | None = None) -> set[str]:
    """The current authorized-target set. Tolerates a missing or corrupt
    file (returns empty) — a bad config file must never crash a read."""
    target_path = path or AUTHORIZED_TARGETS_PATH
    with _lock:
        try:
            if not target_path.exists():
                return set()
            data = json.loads(target_path.read_text(encoding="utf-8"))
            targets = data.get("targets", []) if isinstance(data, dict) else []
            return {_normalise(t) for t in targets if _normalise(t)}
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            return set()


def _save(targets: set[str], path: Path | None = None) -> None:
    target_path = path or AUTHORIZED_TARGETS_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"targets": sorted(targets)}
    atomic_write_text(target_path, json.dumps(payload, indent=2), encoding="utf-8")


def add_authorized_target(target: str, path: Path | None = None) -> set[str]:
    """Add *target* to the authorized set; returns the updated set."""
    normalised = _normalise(target)
    with _lock:
        targets = load_authorized_targets(path)
        if normalised:
            targets.add(normalised)
        _save(targets, path)
        return targets


def remove_authorized_target(target: str, path: Path | None = None) -> set[str]:
    """Remove *target* from the authorized set; returns the updated set."""
    normalised = _normalise(target)
    with _lock:
        targets = load_authorized_targets(path)
        targets.discard(normalised)
        _save(targets, path)
        return targets


def list_authorized_targets(path: Path | None = None) -> list[str]:
    return sorted(load_authorized_targets(path))
