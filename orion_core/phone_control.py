"""Phone control: one master switch for everything the phone may do here, and
a record of what it did.

The remote gate (remote_capability.RemoteToolGate) already refuses, asks or
allows each tool a paired phone calls. Two things were missing:

  * a single switch that stops every phone request reaching this computer -
    "stop everything now" when a phone is lost, or simply while away - without
    unpairing it; and
  * a record the owner can read of what the phone made this computer do:
    the tool, when, which tier and what happened. Never the words or the
    arguments - the record must be safe to show and to keep.

The record is one JSON line per step in CONFIG_DIR/phone_activity.jsonl,
pruned to KEEP_DAYS. Both pieces are stdlib only and never raise into a
request: a broken file reads as "switched on, nothing recorded".
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR

SWITCH_PATH = CONFIG_DIR / "phone_control.json"
ACTIVITY_PATH = CONFIG_DIR / "phone_activity.jsonl"
#: How long the record of phone steps is kept.
KEEP_DAYS = 90
#: Read back at most this many lines; the file is pruned well before.
MAX_LINES = 20_000

OFF_REPLY = ("Phone control is switched off on the computer, so I won't act on "
             "anything from the phone. Switch it back on at the computer.")

_LOCK = threading.Lock()


def enabled(path: Path | None = None) -> bool:
    """Whether the phone may act on this computer (on unless switched off).

    Only a switch that was never set reads as on. A file that is locked or
    damaged reads as OFF (Mark XXXX.4): an antivirus or sync lock during a write
    must not let the phone through while the owner has switched it off; the
    owner switching it on again rewrites the file."""
    try:
        data = json.loads(Path(path or SWITCH_PATH).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("enabled") is not False


def set_enabled(on: bool, path: Path | None = None, *, now: float | None = None) -> str:
    """Switch phone control on or off; returns what to say."""
    target = Path(path or SWITCH_PATH)
    record = {"enabled": bool(on), "changed_at": float(now if now is not None else time.time())}
    with _LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(target)
    if on:
        return "Phone control is on again - the phone can ask me for things here."
    return ("Done - nothing from the phone reaches this computer now. "
            "Say 'phone control on' at the computer to allow it again.")


def record(tool: str, tier: str, outcome: str, *, device: str = "",
           path: Path | None = None, now: float | None = None) -> None:
    """Note one step a phone request took. No arguments, no words."""
    entry = {"at": round(float(now if now is not None else time.time()), 1),
             "tool": str(tool)[:60], "tier": str(tier)[:12], "outcome": str(outcome)[:24],
             "device": str(device)[-6:]}
    try:
        target = Path(path or ACTIVITY_PATH)
        with _LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def recent(days: float = 7.0, *, path: Path | None = None,
           now: float | None = None) -> list[dict[str, Any]]:
    """The recorded steps of the last *days*, oldest first."""
    since = float(now if now is not None else time.time()) - days * 86400.0
    try:
        lines = Path(path or ACTIVITY_PATH).read_text(encoding="utf-8").splitlines()[-MAX_LINES:]
    except OSError:
        return []
    found = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and float(entry.get("at") or 0) >= since:
            found.append(entry)
    return found


def prune(*, path: Path | None = None, now: float | None = None) -> int:
    """Drop steps older than KEEP_DAYS; returns how many were dropped."""
    target = Path(path or ACTIVITY_PATH)
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    keep = recent(KEEP_DAYS, path=target, now=now)
    if len(keep) == len(lines):
        return 0
    with _LOCK:
        tmp = target.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e) + "\n" for e in keep), encoding="utf-8")
        tmp.replace(target)
    return len(lines) - len(keep)


def summary(days: float = 7.0, *, path: Path | None = None, now: float | None = None) -> str:
    """What the phone made this computer do lately, in a few lines."""
    state = "on" if enabled() else "OFF"
    steps = recent(days, path=path, now=now)
    if not steps:
        return f"Phone control is {state}. Nothing from the phone in the last {days:g} days."
    ran = [s for s in steps if s.get("outcome") in ("ran", "approved")]
    asked = [s for s in steps if s.get("outcome") == "asked"]
    refused = [s for s in steps if s.get("outcome") in ("refused", "off")]
    lines = [f"Phone control is {state}. In the last {days:g} days the phone made "
             f"{len(steps)} request{'s' if len(steps) != 1 else ''}: {len(ran)} ran, "
             f"{len(asked)} waited for your approval, {len(refused)} refused."]
    for entry in steps[-8:]:
        when = datetime.fromtimestamp(float(entry["at"])).strftime("%a %H:%M")
        lines.append(f"{when} - {entry.get('tool')} ({entry.get('outcome')})")
    return "\n".join(lines)


__all__ = ["ACTIVITY_PATH", "KEEP_DAYS", "OFF_REPLY", "SWITCH_PATH", "enabled", "prune",
           "recent", "record", "set_enabled", "summary"]
