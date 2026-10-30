"""
Erase what ORION keeps about his owner - the right to erasure (GDPR art. 17).

Owner, 28 Oct 2026, before ORION goes open source: his data must stay his,
and "anything TORVIS has I want ORION to have" - TORVIS has "delete my
account and all my data".

Three steps, so nothing goes on a model's say-so:
  survey(scope)     read-only: what would go, and how much.
  schedule(scope)   only after the owner approves the on-screen confirmation
                    (system_guard DATA_ERASE): writes erase_pending.json.
  run_pending()     the first thing ORION does at start-up, before any store
                    is opened - a live database cannot be moved while open -
                    sends every listed item to the Recycle Bin, then the
                    marker. The owner's own rule: never deleted outright.

Scopes:
  personal    memories, conversations, logs, recordings, training data, his
              name and his location; keys, settings, paired devices and
              downloaded models stay, so ORION still starts and connects.
  everything  the whole config folder: a fresh install.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .constants import CONFIG_DIR

MARKER = "erase_pending.json"
SCOPES = ("personal", "everything")
#: What "personal" keeps: credentials, settings, pairings and model files -
#: nothing that records what the owner said, did or is. A file not named here
#: goes, so a store added later is erased by default.
KEEP_PERSONAL = frozenset({
    "api_keys.json", "remote_secret.key", "remote_devices.json", "appearance.json",
    "audio.json", "camera.json", "language.json", "mcp_servers.json", "mcp_catalogue.json",
    "plugins.json", "reporting.json", "windows_integration.json", "voice_settings.json",
    "models", "custom_tools", MARKER,
})


@dataclass
class EraseSurvey:
    scope: str
    paths: list[Path] = field(default_factory=list)
    total_bytes: int = 0

    def summary(self) -> str:
        size = self.total_bytes / (1024 * 1024)
        what = ("everything ORION keeps on this PC, settings and keys included"
                if self.scope == "everything" else
                "his memories, conversations, logs, recordings, training data, your name and "
                "your location (keys, settings, paired devices and models stay)")
        return (f"This erases {what}: {len(self.paths)} item(s), {size:.1f} MB, to the Recycle Bin, "
                "then ORION restarts.")


def _size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def survey(scope: str = "personal", config_dir: Path | None = None) -> EraseSurvey:
    scope = str(scope or "personal").strip().lower()
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
    folder = Path(config_dir or CONFIG_DIR)
    result = EraseSurvey(scope)
    if not folder.is_dir():
        return result
    for entry in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if entry.name == MARKER:
            continue
        if scope == "personal" and entry.name in KEEP_PERSONAL:
            continue
        result.paths.append(entry)
        result.total_bytes += _size(entry)
    return result


def schedule(scope: str, config_dir: Path | None = None) -> EraseSurvey:
    """Arm the erase for the next start (call only once the owner approved)."""
    plan = survey(scope, config_dir)
    folder = Path(config_dir or CONFIG_DIR)
    (folder / MARKER).write_text(json.dumps({"scope": plan.scope, "at": time.time()}),
                                 encoding="utf-8")
    return plan


def pending(config_dir: Path | None = None) -> str:
    """The scope waiting to run at the next start, or ''."""
    try:
        data = json.loads((Path(config_dir or CONFIG_DIR) / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    scope = str(data.get("scope") or "") if isinstance(data, dict) else ""
    return scope if scope in SCOPES else ""


def cancel(config_dir: Path | None = None) -> bool:
    marker = Path(config_dir or CONFIG_DIR) / MARKER
    if not marker.exists():
        return False
    from . import recycle_bin
    return bool(recycle_bin.send_to_recycle_bin(marker))


def run_pending(config_dir: Path | None = None, log: Callable[[str], Any] = print,
                recycle: Callable[[Path], bool] | None = None,
                refusal: Callable[[Path], str | None] | None = None) -> dict[str, Any]:
    """Carry out an approved erase. Start-up only, before stores open."""
    folder = Path(config_dir or CONFIG_DIR)
    scope = pending(folder)
    if not scope:
        return {"ran": False}
    if recycle is None or refusal is None:
        from . import recycle_bin
        recycle = recycle or recycle_bin.send_to_recycle_bin
        refusal = refusal or recycle_bin.refusal
    plan = survey(scope, folder)
    moved, kept = 0, []
    for path in plan.paths:
        why = refusal(path)
        if why:
            kept.append(f"{path.name}: {why}")
            continue
        if recycle(path):
            moved += 1
        else:
            kept.append(f"{path.name}: could not be moved")
    marker = folder / MARKER
    if not kept and marker.exists():
        recycle(marker)                      # done: no retry at the next start
    log(f"PRIVACY: erased {moved} item(s) ({scope}) to the Recycle Bin"
        + (f"; {len(kept)} could not be moved and the erase will try again next start: "
           + "; ".join(kept[:5]) if kept else "."))
    return {"ran": True, "scope": scope, "moved": moved, "kept": kept}


__all__ = ["KEEP_PERSONAL", "MARKER", "SCOPES", "EraseSurvey", "cancel", "pending", "run_pending",
           "schedule", "survey"]
