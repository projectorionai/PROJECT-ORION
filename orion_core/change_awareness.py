"""
Source change-awareness — ORION knowing which of his own files changed.

Whenever Claude (or anyone) edits ORION's source between runs, ORION should be
able to say *exactly* which of his "mains folder" modules changed, were added
or removed.  This is not the curated patch-notes prose (that is ``changelog``)
— it is the ground truth, computed from the code itself at the file level.

The tracker fingerprints every source file under his package directory
(``orion_core/``), persists the manifest, and on each boot diffs the current
tree against the last recorded manifest.  The diff — *added / modified /
removed* — is cached for the session so ORION can report it when asked, and the
new manifest becomes the baseline for next time.  A rolling journal of update
events is kept, so he can recount the last several updates, not just the latest.

Design notes
------------
* Everything is defensive: a corrupt store, an unreadable file or a missing
  directory degrades to "no changes" rather than raising — boot must never fail
  because ORION could not fingerprint himself.
* The very first run has no baseline; announcing all ~120 modules as "new"
  would be noise, so the first scan is recorded silently as the baseline and
  reported as an initial snapshot.
* Paths in the store use POSIX separators so a manifest is stable regardless of
  the host it was written on.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR, PACKAGE_DIR
from .atomic_io import atomic_write_text

# How many past update events to retain in the journal.
MAX_EVENTS = 40
# Directory names never fingerprinted (build artefacts, caches, venvs).
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
# File suffixes considered part of ORION's "mains" body.
_SOURCE_SUFFIXES = {".py"}


@dataclass
class ChangeSet:
    """The difference between two source snapshots."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    at: str = ""
    first_snapshot: bool = False
    total_files: int = 0

    @property
    def count(self) -> int:
        return len(self.added) + len(self.modified) + len(self.removed)

    @property
    def changed(self) -> bool:
        return self.count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "added": list(self.added),
            "modified": list(self.modified),
            "removed": list(self.removed),
            "count": self.count,
            "total_files": self.total_files,
            "first_snapshot": self.first_snapshot,
        }


class SourceChangeTracker:
    """Fingerprints ORION's own source tree and reports what changed."""

    def __init__(
        self,
        root: Path | None = None,
        store_path: Path | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else PACKAGE_DIR
        self.store_path = (
            Path(store_path) if store_path is not None
            else CONFIG_DIR / "source_awareness.json"
        )
        self.last_change: ChangeSet | None = None

    # ── fingerprinting ────────────────────────────────────────────────────────

    def _iter_files(self):
        for path in self.root.rglob("*"):
            if path.suffix not in _SOURCE_SUFFIXES:
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            yield path

    def _relpath(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _fingerprint(path: Path) -> dict[str, Any] | None:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        digest = hashlib.sha256(data).hexdigest()
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return {"hash": digest, "size": len(data), "mtime": round(mtime, 3)}

    def scan(self) -> dict[str, dict[str, Any]]:
        """A fresh manifest of the current source tree: relpath → fingerprint."""
        manifest: dict[str, dict[str, Any]] = {}
        for path in self._iter_files():
            fp = self._fingerprint(path)
            if fp is not None:
                manifest[self._relpath(path)] = fp
        return manifest

    # ── persistence ───────────────────────────────────────────────────────────

    def _load_store(self) -> dict[str, Any]:
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("manifest", {})
                data.setdefault("events", [])
                return data
        except (OSError, ValueError):
            pass
        return {"manifest": {}, "events": []}

    def _save_store(self, store: dict[str, Any]) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.store_path,
                json.dumps(store, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    # ── diffing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _diff(
        baseline: dict[str, dict[str, Any]],
        current: dict[str, dict[str, Any]],
    ) -> tuple[list[str], list[str], list[str]]:
        base_keys = set(baseline)
        cur_keys = set(current)
        added = sorted(cur_keys - base_keys)
        removed = sorted(base_keys - cur_keys)
        modified = sorted(
            k for k in (cur_keys & base_keys)
            if baseline[k].get("hash") != current[k].get("hash")
        )
        return added, modified, removed

    def detect(self, commit: bool = True) -> ChangeSet:
        """Compute what changed since the last recorded baseline.

        When *commit* is True the current tree becomes the new baseline and, if
        anything changed, an event is appended to the journal.  The result is
        cached on ``self.last_change``.
        """
        store = self._load_store()
        baseline = store.get("manifest") or {}
        current = self.scan()
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        if not baseline:
            # First ever run — adopt the current tree as the baseline silently.
            change = ChangeSet(at=now, first_snapshot=True, total_files=len(current))
            if commit:
                store["manifest"] = current
                self._save_store(store)
            self.last_change = change
            return change

        added, modified, removed = self._diff(baseline, current)
        change = ChangeSet(
            added=added, modified=modified, removed=removed,
            at=now, total_files=len(current),
        )
        if commit:
            store["manifest"] = current
            if change.changed:
                events = store.get("events") or []
                events.append(change.to_dict())
                store["events"] = events[-MAX_EVENTS:]
            self._save_store(store)
        self.last_change = change
        return change

    # ── history ───────────────────────────────────────────────────────────────

    def recent_events(self, limit: int = 5) -> list[dict[str, Any]]:
        store = self._load_store()
        events = store.get("events") or []
        return list(reversed(events[-max(1, limit):]))

    # ── rendering (first-person, spoken-friendly) ─────────────────────────────

    @staticmethod
    def _name_list(paths: list[str], cap: int = 12) -> str:
        # Speak the bare module names; a deep path keeps its tail for context.
        names = [p.split("/")[-1] if p.count("/") <= 1 else ".../" + p.split("/")[-1]
                 for p in paths]
        if len(names) <= cap:
            return ", ".join(names)
        shown = ", ".join(names[:cap])
        return f"{shown}, and {len(names) - cap} more"

    def render(self, change: ChangeSet | None) -> str:
        if change is None:
            change = self.detect(commit=False)
        if change.first_snapshot:
            return (
                f"I've just mapped my own source for the first time — "
                f"{change.total_files} modules fingerprinted as my baseline. "
                "From now on I'll know exactly which of my files change between "
                "updates."
            )
        if not change.changed:
            return "None of my source files have changed since I last checked."
        parts = [f"Since my last run, {change.count} of my files changed."]
        if change.modified:
            parts.append(f"Modified: {self._name_list(change.modified)}.")
        if change.added:
            parts.append(f"New: {self._name_list(change.added)}.")
        if change.removed:
            parts.append(f"Removed: {self._name_list(change.removed)}.")
        return " ".join(parts)

    def render_latest(self) -> str:
        """Report the change detected at boot (or a fresh read if none cached)."""
        return self.render(self.last_change)

    def render_history(self, limit: int = 5) -> str:
        events = self.recent_events(limit)
        if not events:
            return "I have no recorded update history yet."
        lines = ["My recent updates:"]
        for ev in events:
            when = ev.get("at", "?")
            count = ev.get("count", 0)
            bits = []
            if ev.get("modified"):
                bits.append(f"{len(ev['modified'])} modified")
            if ev.get("added"):
                bits.append(f"{len(ev['added'])} new")
            if ev.get("removed"):
                bits.append(f"{len(ev['removed'])} removed")
            summary = ", ".join(bits) if bits else "no file changes"
            lines.append(f"  • {when} — {count} files ({summary})")
        return "\n".join(lines)

    # ── boot integration ──────────────────────────────────────────────────────

    def log_startup(self, bus: Any) -> ChangeSet:
        """Run detection at boot, cache it, and emit a one-line summary."""
        try:
            change = self.detect(commit=True)
        except Exception:
            self.last_change = ChangeSet(total_files=0)
            return self.last_change
        try:
            if change.first_snapshot:
                bus.log.emit(
                    f"CHANGE: source baseline mapped — {change.total_files} modules "
                    "fingerprinted."
                )
            elif change.changed:
                bus.log.emit(
                    f"CHANGE: {change.count} source file(s) changed since last run "
                    f"({len(change.modified)} modified, {len(change.added)} new, "
                    f"{len(change.removed)} removed)."
                )
            else:
                bus.log.emit("CHANGE: no source files changed since last run.")
        except Exception:
            pass
        return change
