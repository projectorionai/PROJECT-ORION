"""
Pinned and recent Command Deck pages.

The deck has more than twenty pages in twelve zones. Search and the zone
headings find any of them, but most people use four or five, and those should
be one click from the top of the sidebar rather than a scroll away.

Pinned pages are a decision, so they persist until unpinned. Recent pages are
an observation, so they are simply the last few distinct pages opened — also
kept across restarts, because "the page I had open yesterday" is the most
common thing to want first thing in the morning.

Stored beside the other preferences, never inside the deck's own state, so a
page renamed or removed in an update drops out quietly instead of breaking the
sidebar.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable

from ..atomic_io import atomic_write_text
from ..constants import CONFIG_DIR

DECK_PREFERENCES_PATH = CONFIG_DIR / "deck_preferences.json"

#: How many recent pages the sidebar shows.
RECENT_LIMIT = 5
#: A ceiling on pins, so the section cannot push the zones out of sight.
PIN_LIMIT = 8


class DeckFavourites:
    def __init__(self, path: Path | None = None, *, write_in_background: bool = False) -> None:
        self.path = Path(path) if path is not None else DECK_PREFERENCES_PATH
        # The deck's own copy writes on a worker thread: the recent list
        # changes on every page switch, and the flushed write took 106 ms of
        # the GUI thread six page opens running (26 Oct 2026 log).
        self._background = bool(write_in_background)
        self._lock = threading.Lock()
        self._pending: str | None = None
        self._writing = False
        data = self._load()
        self.pinned: list[str] = self._clean(data.get("pinned"), PIN_LIMIT)
        self.recent: list[str] = self._clean(data.get("recent"), RECENT_LIMIT)
        # Sidebar sections the user folded away, and the icons-only rail.
        self.collapsed: list[str] = self._clean(data.get("collapsed"), 32)
        # Sections that start folded (More) but which the user has opened:
        # without this, opening one would be forgotten at the next start.
        self.expanded: list[str] = self._clean(data.get("expanded"), 32)
        self.rail: bool = bool(data.get("rail", False))

    @staticmethod
    def _clean(values: Any, limit: int) -> list[str]:
        out: list[str] = []
        for value in values if isinstance(values, list) else []:
            name = str(value or "").strip().upper()
            if name and name not in out:
                out.append(name)
        return out[:limit]

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self) -> None:
        text = json.dumps(
            {"schema": "orion.deck_preferences.v1", "pinned": self.pinned,
             "recent": self.recent, "collapsed": self.collapsed,
             "expanded": self.expanded, "rail": self.rail}, indent=2)
        if not self._background:
            self._write(text)
            return
        with self._lock:
            self._pending = text                 # only the newest is written
            if self._writing:
                return
            self._writing = True
        threading.Thread(target=self._drain, name="orion-deck-preferences", daemon=True).start()

    def _drain(self) -> None:
        while True:
            with self._lock:
                text, self._pending = self._pending, None
                if text is None:
                    self._writing = False
                    return
            self._write(text)

    def flush(self, timeout: float = 2.0) -> None:
        """Wait (briefly) for a background write to land."""
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._writing:
                    return
            time.sleep(0.01)

    def _write(self, text: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path, text, encoding="utf-8")
        except Exception:
            pass

    def restrict_to(self, pages: Iterable[str]) -> None:
        """Forget pages that no longer exist (renamed or removed in an update)."""
        known = set(pages)
        self.pinned = [n for n in self.pinned if n in known]
        self.recent = [n for n in self.recent if n in known]

    def is_pinned(self, name: str) -> bool:
        return name in self.pinned

    def toggle_pin(self, name: str) -> bool:
        """Pin or unpin *name*; returns whether it is now pinned."""
        if name in self.pinned:
            self.pinned.remove(name)
            pinned = False
        else:
            self.pinned = (self.pinned + [name])[-PIN_LIMIT:]
            pinned = True
        self._save()
        return pinned

    def is_collapsed(self, section: str, default: bool = False) -> bool:
        """Whether *section* is folded: the user's choice, else *default*."""
        section = str(section or "").strip().upper()
        if section in self.collapsed:
            return True
        if section in self.expanded:
            return False
        return bool(default)

    def set_collapsed(self, section: str, collapsed: bool) -> None:
        section = str(section or "").strip().upper()
        if not section:
            return
        was = (list(self.collapsed), list(self.expanded))
        if collapsed:
            if section not in self.collapsed:
                self.collapsed.append(section)
            if section in self.expanded:
                self.expanded.remove(section)
        else:
            if section in self.collapsed:
                self.collapsed.remove(section)
            if section not in self.expanded:
                self.expanded.append(section)
        if (self.collapsed, self.expanded) != was:
            self._save()

    def set_rail(self, on: bool) -> None:
        if bool(on) != self.rail:
            self.rail = bool(on)
            self._save()

    def visited(self, name: str) -> bool:
        """Record a visit; returns whether the recent list changed."""
        updated = ([name] + [n for n in self.recent if n != name])[:RECENT_LIMIT]
        if updated == self.recent:
            return False
        self.recent = updated
        self._save()
        return True


__all__ = ["DeckFavourites", "DECK_PREFERENCES_PATH", "RECENT_LIMIT", "PIN_LIMIT"]
