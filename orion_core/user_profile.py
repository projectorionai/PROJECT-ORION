"""
UserProfile — the operator's own names, kept out of a public source tree.

Why this exists
---------------
ORION's commerce advisor, creator suite and mission board are seeded with the
names of a *real* business and real projects.  That is exactly what makes them
useful day to day, and exactly what must not be committed once the repository
is public: a brand name, an agency name and a list of someone's personal
endeavours are personal data, not source code.

Deleting them is not the answer either — a clone that advises on "your brand"
is useless, and an operator who loses their own mission board to a repository
hygiene pass has been made worse off by a cleanup.

So the two concerns are separated the same way ``identity.py`` separates frozen
core identity from persisted preferences:

    * **the source** carries neutral, publishable defaults — what a stranger
      cloning the repository gets, and what the test suite asserts against;
    * **config/profile.json** carries the operator's real names.  ``config/``
      is ignored wholesale by .gitignore, so this file never leaves the
      machine, and ORION on this machine behaves exactly as it did before.

Nothing here is required.  With no profile.json the defaults apply and every
caller works unchanged, which is the property that makes this safe to adopt
across the codebase incrementally.

Dependency-light on purpose: no Qt, no bus, no networking, so it can be read
from a module-level constant during import without dragging the app in.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .constants import CONFIG_DIR

PROFILE_PATH = CONFIG_DIR / "profile.json"


#: Publishable defaults.  These are what the repository ships and what the test
#: suite pins; the operator's real values go in config/profile.json.
DEFAULTS: dict[str, Any] = {
    "brand": "ExampleStore",
    "brand_niche": "home products",
    "agency": "Creator Studio",
    "missions": [
        ["Build Demo Game",
         "Game development — design, build and ship Demo Game."],
        ["Develop ORION",
         "Evolve ORION into a full personal AI operating system."],
        ["Creator Studio",
         "Short-form content agency: creators, UGC, product research and growth."],
        ["Neuroscience Research",
         "Ongoing neuroscience study and research."],
        ["University Studies",
         "Coursework, deadlines and independent study."],
    ],
}

_cache: dict[str, Any] | None = None


def _mission_rows(value: Any) -> list[list[str]]:
    """Keep usable, unique text pairs from a hand-edited profile."""
    rows: list[list[str]] = []
    seen: set[str] = set()
    if not isinstance(value, (list, tuple)):
        return rows
    for row in value:
        if (not isinstance(row, (list, tuple)) or len(row) != 2
                or not all(isinstance(item, str) for item in row)):
            continue
        title, description = row[0].strip()[:80], row[1].strip()[:300]
        if title and title.casefold() not in seen:
            rows.append([title, description])
            seen.add(title.casefold())
    return rows


def reload() -> dict[str, Any]:
    """Re-read profile.json from disk, replacing the cache.

    Tests use this after writing a profile; normal runtime never needs it,
    because the file is read once per process.
    """
    global _cache
    merged = deepcopy(DEFAULTS)
    try:
        raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeDecodeError):
        # Absent, unreadable or malformed — the defaults are a complete,
        # working profile, so a broken override must never break startup.
        raw = {}
    if isinstance(raw, dict):
        for key in ("brand", "brand_niche", "agency"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                merged[key] = value.strip()[:300]
        if rows := _mission_rows(raw.get("missions")):
            merged["missions"] = rows
    _cache = merged
    return deepcopy(merged)


def profile() -> dict[str, Any]:
    """The merged profile: defaults overlaid with config/profile.json."""
    if _cache is None:
        return reload()
    return deepcopy(_cache)


def brand() -> str:
    """The operator's commerce brand (products sold direct to consumers)."""
    return str(profile().get("brand") or DEFAULTS["brand"])


def brand_niche() -> str:
    """What that brand sells, as a short phrase for prompt copy."""
    return str(profile().get("brand_niche") or DEFAULTS["brand_niche"])


def agency() -> str:
    """The operator's content/creator business."""
    return str(profile().get("agency") or DEFAULTS["agency"])


def missions() -> tuple[tuple[str, str], ...]:
    """Starter missions for a fresh mission board, as (title, description).

    Rows that are not a usable pair are skipped rather than raising, so one bad
    hand-edited entry cannot cost the operator the whole board.
    """
    return tuple((title, description) for title, description in profile()["missions"])


__all__ = [
    "DEFAULTS", "PROFILE_PATH", "agency", "brand", "brand_niche",
    "missions", "profile", "reload",
]
