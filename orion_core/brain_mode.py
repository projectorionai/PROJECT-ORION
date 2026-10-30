"""
Which brain answers - the cloud, the local model, or both.

ORION's conversation has always been cloud-first: Gemini Live for voice, the
provider router for text, and the rule-based LocalBrain only when everything
else failed. The local agent (local_agent.py) adds a real local model that can
choose and use ORION's tools, so the owner can choose how much of ORION runs
without any AI API:

    cloud   Today's behaviour. When the cloud text path fails, the local agent
            is tried before the LocalBrain rules.
    hybrid  Typed turns and offline-voice turns go to the local agent first.
            Open-ended, long-form or uncertain requests are handed to the cloud
            text model. Live voice still runs.
    local   No AI API calls at all: Live is not started, cloud text and cloud
            speech are not used. Local speech recognition -> local agent ->
            Kokoro speech. Ordinary internet tools (web search and the like)
            still work when the machine is online.

Stored beside the other preferences (CONFIG_DIR/brain_mode.json). Reading order
matches windows_settings.py: an environment override (ORION_BRAIN_MODE), then
the stored choice, then the default. ``ORION_MODE=offline`` (``--offline``)
means no cloud at all, so the effective mode is ``local`` whatever is stored.
Tests point SETTINGS_PATH at a temporary file (see conftest.py).
"""

from __future__ import annotations

import json
import os
from typing import Any

from .atomic_io import atomic_write_text
from .constants import CONFIG_DIR

SETTINGS_PATH = CONFIG_DIR / "brain_mode.json"

CLOUD = "cloud"
HYBRID = "hybrid"
LOCAL = "local"
MODES: tuple[str, ...] = (CLOUD, HYBRID, LOCAL)
DEFAULT_MODE = CLOUD

#: The names shown in Preferences. Names, so Title Case.
LABELS: dict[str, str] = {CLOUD: "Cloud", HYBRID: "Hybrid", LOCAL: "Local Only"}

#: One line each, for Preferences, the ai_mode report and spoken answers.
EXPLANATIONS: dict[str, str] = {
    CLOUD: "Cloud models answer as they always have; the local model steps in "
           "only when the cloud cannot.",
    HYBRID: "The local model handles everyday commands first and hands open-ended "
            "questions to the cloud; live voice keeps running.",
    LOCAL: "No AI service is used at all: local hearing, the local model and the "
           "local voice. Web tools still work when you are online.",
}

#: Every stored key and its default. ``local_model`` empty = the model Ollama
#: registration chose; ``keep_alive`` is how long the local model stays loaded
#: after a turn while hybrid or local mode is using it.
DEFAULTS: dict[str, Any] = {
    "brain_mode": DEFAULT_MODE,
    "local_model": "",
    "keep_alive": "10m",
}

_ALIASES = {
    "cloud": CLOUD, "online": CLOUD, "default": CLOUD, "normal": CLOUD,
    "hybrid": HYBRID, "mixed": HYBRID, "both": HYBRID,
    "local": LOCAL, "local_only": LOCAL, "local only": LOCAL, "localonly": LOCAL,
    "offline": LOCAL, "local brain": LOCAL, "local-only": LOCAL,
}


def normalise(value: Any) -> str | None:
    """A mode name from loose words ('Local Only', 'offline'), or None."""
    text = str(value or "").strip().lower().replace("_", " ")
    if not text:
        return None
    return _ALIASES.get(text) or _ALIASES.get(text.replace(" ", "_")) or (
        text if text in MODES else None)


def _load() -> dict[str, Any]:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def stored_mode() -> str:
    """The owner's saved choice (ignores the environment)."""
    return normalise(_load().get("brain_mode")) or DEFAULT_MODE


def mode() -> str:
    """The configured brain mode: ORION_BRAIN_MODE, then the stored choice."""
    override = normalise(os.getenv("ORION_BRAIN_MODE", ""))
    if override:
        return override
    return stored_mode()


def forced_offline() -> bool:
    """``--offline`` / ORION_MODE=offline: no cloud for this session."""
    return os.getenv("ORION_MODE", "").strip().lower() == "offline"


def effective_mode() -> str:
    """The mode that actually routes turns. A forced-offline session has no
    cloud to escalate to, so it behaves as ``local``."""
    if forced_offline():
        return LOCAL
    return mode()


def env_override_active() -> bool:
    return normalise(os.getenv("ORION_BRAIN_MODE", "")) is not None


def get(key: str) -> Any:
    """The value in effect for *key* (unknown keys raise KeyError)."""
    default = DEFAULTS[key]
    if key == "brain_mode":
        return mode()
    env = {"local_model": "ORION_LOCAL_AGENT_MODEL",
           "keep_alive": "ORION_LOCAL_AGENT_KEEP_ALIVE"}.get(key, "")
    override = os.getenv(env, "").strip() if env else ""
    if override:
        return override
    stored = _load().get(key)
    if stored is None or type(stored) is not type(default):
        return default
    return stored


def set_value(key: str, value: Any) -> bool:
    """Store *value* for *key*. False (nothing stored) for an unknown key or
    a value of the wrong type."""
    if key not in DEFAULTS:
        return False
    if key == "brain_mode":
        value = normalise(value)
        if value is None:
            return False
    elif type(value) is not type(DEFAULTS[key]):
        return False
    data = _load()
    data[key] = value
    data.setdefault("schema", "orion.brain_mode.v1")
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(SETTINGS_PATH, json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def set_mode(value: Any) -> str | None:
    """Save a brain mode; returns the saved mode or None if it was not valid."""
    chosen = normalise(value)
    if chosen is None or not set_value("brain_mode", chosen):
        return None
    return chosen


def describe(value: str | None = None) -> str:
    chosen = normalise(value) or effective_mode()
    return f"{LABELS[chosen]} - {EXPLANATIONS[chosen]}"


def snapshot() -> dict[str, Any]:
    return {"brain_mode": mode(), "effective": effective_mode(),
            "env_override": env_override_active(), "forced_offline": forced_offline(),
            "local_model": get("local_model"), "keep_alive": get("keep_alive")}


__all__ = ["CLOUD", "DEFAULTS", "DEFAULT_MODE", "EXPLANATIONS", "HYBRID", "LABELS", "LOCAL",
           "MODES", "SETTINGS_PATH", "describe", "effective_mode", "env_override_active",
           "forced_offline", "get", "mode", "normalise", "set_mode", "set_value", "snapshot",
           "stored_mode"]
