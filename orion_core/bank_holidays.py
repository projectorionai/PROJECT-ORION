"""
UK bank holidays — from GOV.UK's own list, so ORION knows a day off is one.

GOV.UK publishes every bank holiday for England and Wales, Scotland and
Northern Ireland as JSON (https://www.gov.uk/bank-holidays.json, keyless,
including substitute days such as a Christmas that falls on a Saturday). The
list is fetched at most weekly and kept in config/bank_holidays.json; every
reader works from that file, so nothing here ever waits on the network except
refresh().

    division_for(region)   'england-and-wales' / 'scotland' / 'northern-ireland'
    holiday_on(day)        the title of the bank holiday on that date, or None
    next_holiday(after)    (title, date) of the next one
    refresh()              async: fetch when the cached copy is a week old
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR

URL = "https://www.gov.uk/bank-holidays.json"
CACHE_PATH = CONFIG_DIR / "bank_holidays.json"
MAX_AGE_S = 7 * 24 * 3600
DEFAULT_DIVISION = "england-and-wales"


def division_for(region: str = "") -> str:
    text = str(region or "").lower()
    if "scotland" in text:
        return "scotland"
    if "northern ireland" in text:
        return "northern-ireland"
    return DEFAULT_DIVISION


def _load(path: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads((path or CACHE_PATH).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def events(division: str = DEFAULT_DIVISION, path: Path | None = None) -> list[tuple[date, str]]:
    """Every cached (date, title) for the division, in date order."""
    block = (_load(path).get("data") or {}).get(division) or {}
    out = []
    for event in block.get("events") or []:
        try:
            title = str(event.get("title") or "Bank holiday").replace("’", "'")
            if "substitute" in str(event.get("notes") or "").lower():
                title += " (substitute day)"
            out.append((date.fromisoformat(str(event.get("date"))), title))
        except ValueError:
            continue
    return sorted(out)


def holiday_on(day: date, division: str = DEFAULT_DIVISION, path: Path | None = None) -> str | None:
    return next((title for when, title in events(division, path) if when == day), None)


def next_holiday(after: date, division: str = DEFAULT_DIVISION,
                 path: Path | None = None) -> tuple[str, date] | None:
    return next(((title, when) for when, title in events(division, path) if when >= after), None)


async def refresh(path: Path | None = None, *, session: Any = None) -> bool:
    """Fetch the list when the cached copy is missing or a week old. True if
    a fresh copy was written. Never raises."""
    target = path or CACHE_PATH
    saved = _load(target)
    if saved and time.time() - float(saved.get("fetched_at") or 0) < MAX_AGE_S:
        return False
    try:
        async def get(client: Any) -> dict[str, Any]:
            async with client.get(URL) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                return await response.json(content_type=None)
        if session is None:
            from aiohttp import ClientSession, ClientTimeout
            async with ClientSession(timeout=ClientTimeout(total=4.0, connect=2.0)) as own:
                data = await get(own)
        else:
            data = await get(session)
        if not isinstance(data, dict) or DEFAULT_DIVISION not in data:
            return False
        from .atomic_io import atomic_write_text
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, json.dumps({"fetched_at": time.time(), "data": data}),
                          encoding="utf-8")
        return True
    except Exception:
        return False


__all__ = ["CACHE_PATH", "division_for", "events", "holiday_on", "next_holiday", "refresh"]
