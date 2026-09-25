"""
Operational status (Mark XXVI, §14) — one honest answer to "what is ORION doing?"

ORION's live state is scattered across the bus (mode, connection, voice, control
activity), the engines (a running focus block, the active mission) and the health
model. This composes them into ONE small, structured snapshot so a single elegant
status surface can show it — mode, network, voice, the current activity, and any
focus/mission in play — instead of the user having to infer it.

Pure and dependency-light: ``compose(...)`` takes plain values and returns an
``OpStatus``; ``from_context(...)`` reads them off the live objects behind a
try/except each, so a missing subsystem degrades one field, never the strip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpStatus:
    mode: str = "cloud"          # "cloud" (MODE A) | "offline" (MODE B)
    online: bool = True
    voice: str = "idle"          # listening | speaking | processing | standby | paused | idle
    activity: str = ""           # the current tool/task, or "" when quiet
    focus: str = ""              # active focus-block label, or ""
    mission: str = ""            # active mission title, or ""
    health: str = "ONLINE"       # ONLINE | DEGRADED | OFFLINE | …

    def line(self) -> str:
        """A compact one-line summary — for a log, a tooltip, or the phone."""
        net = "cloud" if self.online else "offline"
        parts = [f"{net}", self.voice]
        if self.activity:
            parts.append(f"· {self.activity}")
        if self.focus:
            parts.append(f"· focus: {self.focus}")
        if self.mission:
            parts.append(f"· {self.mission}")
        return "  ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "online": self.online, "voice": self.voice,
                "activity": self.activity, "focus": self.focus,
                "mission": self.mission, "health": self.health}


_VOICE_FROM_STATE = {
    "LISTENING": "listening", "SPEAKING": "speaking", "PROCESSING": "processing",
    "CONNECTING": "processing", "INITIALISING": "processing",
    "STANDBY": "standby", "PAUSED": "paused",
}


def compose(*, mode: str = "cloud", online: bool = True, state: str = "",
            activity: str = "", focus: str = "", mission: str = "",
            health: str = "ONLINE") -> OpStatus:
    """Build an OpStatus from plain values (pure). ``state`` is ORION's pipeline
    state label (LISTENING/SPEAKING/…), mapped to a voice word."""
    voice = _VOICE_FROM_STATE.get(str(state or "").upper(), "idle")
    return OpStatus(
        mode=("offline" if not online or str(mode).upper() == "B" else "cloud"),
        online=bool(online), voice=voice, activity=str(activity or "").strip()[:60],
        focus=str(focus or "").strip()[:40], mission=str(mission or "").strip()[:40],
        health=str(health or "ONLINE").upper())


def from_context(*, state: str = "", online: bool = True, mode: str = "cloud",
                 focus_engine: Any = None, mission_engine: Any = None,
                 health_model: Any = None, activity: str = "") -> OpStatus:
    """Read the snapshot off the live objects, each behind its own guard so one
    missing subsystem degrades a single field rather than the whole strip."""
    focus_label = ""
    try:
        if focus_engine is not None:
            s = focus_engine.active()
            if s is not None:
                focus_label = s.label
    except Exception:
        pass
    mission_title = ""
    try:
        if mission_engine is not None:
            # MissionEngine exposes current(); other engines may expose active().
            getter = (getattr(mission_engine, "current", None)
                      or getattr(mission_engine, "active", None))
            m = getter() if callable(getter) else getter
            if isinstance(m, dict):
                mission_title = str(m.get("title") or m.get("name") or "")
            elif m:
                mission_title = (getattr(m, "title", None)
                                 or getattr(m, "name", None) or str(m))
    except Exception:
        pass
    health = "ONLINE"
    try:
        if health_model is not None:
            overall = getattr(health_model, "overall", None)
            if callable(overall):
                health = str(overall())
    except Exception:
        pass
    return compose(mode=mode, online=online, state=state, activity=activity,
                   focus=focus_label, mission=mission_title, health=health)


__all__ = ["OpStatus", "compose", "from_context"]
