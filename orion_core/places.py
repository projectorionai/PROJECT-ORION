"""
Places — "find me a pizza place nearby", "where's the nearest pharmacy?".

The geo tool's 'nearby' lists towns; nothing could find a chemist, a cash
machine or a chippy. This asks OpenStreetMap through the Overpass API
(keyless; one query per question, an identifying User-Agent) for what is
around a point, nearest first, and says whether each is open now - but only
when its opening hours are written in a form read here with certainty
("Mo-Sa 09:00-18:00; Su 10:00-16:00", "24/7"); anything subtler (public
holidays, week numbers, "sunrise") is left unsaid rather than guessed.

    selectors(query)          everyday words -> Overpass tag filters
    search(query, lat, lon)   async; the places, nearest first
    open_now(hours, when)     True / False / None (can't tell)
    describe(found, ...)      the spoken answer
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "ORION-AI-OS (personal assistant; occasional nearby-place queries)"
RADII_M = (2500, 8000)          # widen once when the first ring finds too little
MAX_RESULTS = 40

#: What people say -> OpenStreetMap tags. Longest phrases are tried first.
_TAGS: dict[str, tuple[str, ...]] = {
    "pharmacy": ('["amenity"="pharmacy"]',), "chemist": ('["amenity"="pharmacy"]',),
    "petrol station": ('["amenity"="fuel"]',), "petrol": ('["amenity"="fuel"]',),
    "gas station": ('["amenity"="fuel"]',), "garage": ('["amenity"="fuel"]',),
    "supermarket": ('["shop"="supermarket"]',), "corner shop": ('["shop"="convenience"]',),
    "shop": ('["shop"="convenience"]', '["shop"="supermarket"]'),
    "cash machine": ('["amenity"="atm"]',), "cashpoint": ('["amenity"="atm"]',),
    "atm": ('["amenity"="atm"]',), "bank": ('["amenity"="bank"]',),
    "cafe": ('["amenity"="cafe"]',), "coffee": ('["amenity"="cafe"]',),
    "pub": ('["amenity"="pub"]',), "bar": ('["amenity"="bar"]',),
    "restaurant": ('["amenity"="restaurant"]',), "takeaway": ('["amenity"="fast_food"]',),
    "chippy": ('["cuisine"~"fish_and_chips",i]',), "fish and chips": ('["cuisine"~"fish_and_chips",i]',),
    "hospital": ('["amenity"="hospital"]',), "a&e": ('["amenity"="hospital"]["emergency"="yes"]',),
    "doctor": ('["amenity"="doctors"]',), "doctors": ('["amenity"="doctors"]',),
    "gp": ('["amenity"="doctors"]',), "dentist": ('["amenity"="dentist"]',),
    "vet": ('["amenity"="veterinary"]',), "police station": ('["amenity"="police"]',),
    "post office": ('["amenity"="post_office"]',), "post box": ('["amenity"="post_box"]',),
    "library": ('["amenity"="library"]',), "gym": ('["leisure"="fitness_centre"]',),
    "park": ('["leisure"="park"]',), "car park": ('["amenity"="parking"]',),
    "parking": ('["amenity"="parking"]',),
    "charging point": ('["amenity"="charging_station"]',),
    "ev charger": ('["amenity"="charging_station"]',),
    "charger": ('["amenity"="charging_station"]',),
    "toilet": ('["amenity"="toilets"]',), "toilets": ('["amenity"="toilets"]',),
    "train station": ('["railway"="station"]',), "station": ('["railway"="station"]',),
    "bus stop": ('["highway"="bus_stop"]',), "cinema": ('["amenity"="cinema"]',),
    "hairdresser": ('["shop"="hairdresser"]',), "barber": ('["shop"="hairdresser"]',),
    "launderette": ('["shop"="laundry"]',), "hotel": ('["tourism"="hotel"]',),
}
_CUISINES = ("pizza", "indian", "chinese", "italian", "thai", "kebab", "burger", "sushi",
             "mexican", "japanese", "turkish", "greek", "vietnamese", "korean", "chicken",
             "curry", "noodle", "vegan", "vegetarian", "fish")
_FILLER = re.compile(r"\b(the|a|an|some|nearest|closest|nearby|near me|near here|around here|"
                     r"local|good|decent|place|places|shop|shops|open|now|me|find|for)\b", re.I)


@dataclass(frozen=True)
class Found:
    name: str
    kind: str
    lat: float
    lon: float
    metres: float
    street: str = ""
    hours: str = ""


def selectors(query: str) -> tuple[str, ...]:
    """Overpass tag filters for what was asked for."""
    text = " " + re.sub(r"\s+", " ", str(query or "").lower().replace("’", "'")) + " "
    # A cuisine first: "indian takeaway" is Indian food, not any takeaway.
    if not re.search(r"\b(?:fish and chips|chippy)\b", text):
        for cuisine in _CUISINES:
            if re.search(rf"\b{cuisine}", text):
                tag = "curry|indian" if cuisine == "curry" else cuisine
                return (f'["cuisine"~"{tag}",i]',)
    for phrase in sorted(_TAGS, key=len, reverse=True):
        if f" {phrase} " in text or f" {phrase}s " in text:
            return _TAGS[phrase]
    words = re.sub(r"[^a-z0-9 &'-]", "", _FILLER.sub(" ", text)).split()
    if not words:
        return ()
    name = re.escape(" ".join(words)).replace('"', "")
    return (f'["name"~"{name}",i]',)


def _metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371008.8 * math.asin(min(1.0, math.sqrt(a)))


def parse(data: dict[str, Any], lat: float, lon: float) -> list[Found]:
    out: dict[tuple[str, int, int], Found] = {}
    for element in data.get("elements") or []:
        tags = element.get("tags") or {}
        point = element if "lat" in element else element.get("center") or {}
        try:
            plat, plon = float(point["lat"]), float(point["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        kind = (tags.get("cuisine") or tags.get("amenity") or tags.get("shop")
                or tags.get("leisure") or tags.get("railway") or tags.get("tourism") or "")
        name = str(tags.get("name") or tags.get("brand") or "").strip()
        if not name and kind in ("atm", "post_box", "toilets", "bus_stop", "parking",
                                 "charging_station"):
            name = kind.replace("_", " ")
        if not name:
            continue
        street = " ".join(p for p in (tags.get("addr:housenumber"), tags.get("addr:street")) if p)
        found = Found(name, str(kind).split(";")[0].replace("_", " "), plat, plon,
                      _metres(lat, lon, plat, plon), street, str(tags.get("opening_hours") or ""))
        key = (name.lower(), round(plat * 2000), round(plon * 2000))   # one shop drawn twice
        if key not in out or found.metres < out[key].metres:
            out[key] = found
    return sorted(out.values(), key=lambda f: f.metres)


async def search(query: str, lat: float, lon: float, session: Any = None) -> list[Found]:
    """Places matching *query* around (lat, lon), nearest first."""
    wanted = selectors(query)
    if not wanted:
        return []

    async def ask(client: Any, radius: int) -> list[Found]:
        body = "".join(f"nwr{s}(around:{radius},{lat:.5f},{lon:.5f});" for s in wanted)
        query_text = f"[out:json][timeout:15];({body});out center {MAX_RESULTS};"
        for attempt in range(2):
            async with client.post(OVERPASS, data={"data": query_text},
                                   headers={"User-Agent": USER_AGENT}) as response:
                if response.status in (429, 504) and attempt == 0:
                    # Overpass gives each address two query slots: busy now
                    # usually means free in a couple of seconds.
                    await asyncio.sleep(2.5)
                    continue
                if response.status != 200:
                    raise RuntimeError(f"the map service is busy (HTTP {response.status})")
                return parse(await response.json(content_type=None), lat, lon)
        return []

    async def run(client: Any) -> list[Found]:
        found: list[Found] = []
        try:
            for radius in RADII_M:
                found = await ask(client, radius)
                if found:              # widen only when the first ring found nothing
                    break
        except RuntimeError:
            # The public Overpass server is busy at times. Nominatim knows
            # amenity kinds ("pharmacy", "petrol station") though not
            # cuisines; ask it before giving up.
            found = await nominatim(client, query, lat, lon)
            if not found:
                raise
        return found

    if session is not None:
        return await run(session)
    from aiohttp import ClientSession, ClientTimeout
    async with ClientSession(timeout=ClientTimeout(total=20.0, connect=5.0)) as own:
        return await run(own)


NOMINATIM = "https://nominatim.openstreetmap.org/search"


async def nominatim(client: Any, query: str, lat: float, lon: float) -> list[Found]:
    """The fallback: Nominatim's bounded search (~6 km box), extratags for
    opening hours. [] on any failure."""
    words = " ".join(re.sub(r"[^a-z0-9 &'-]", " ", _FILLER.sub(" ", str(query).lower())).split())
    if not words:
        return []
    box = f"{lon - 0.08:.4f},{lat + 0.05:.4f},{lon + 0.08:.4f},{lat - 0.05:.4f}"
    params = {"format": "jsonv2", "addressdetails": "1", "extratags": "1", "limit": "20",
              "bounded": "1", "viewbox": box, "q": words}
    try:
        async with client.get(NOMINATIM, params=params,
                              headers={"User-Agent": USER_AGENT, "Accept-Language": "en"}) as response:
            if response.status != 200:
                return []
            rows = await response.json(content_type=None)
    except Exception:
        return []
    out = []
    for row in rows if isinstance(rows, list) else []:
        try:
            plat, plon = float(row["lat"]), float(row["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        address = row.get("address") or {}
        street = " ".join(p for p in (address.get("house_number"), address.get("road")) if p)
        out.append(Found(name, str(row.get("type") or "").replace("_", " "), plat, plon,
                         _metres(lat, lon, plat, plon), street,
                         str((row.get("extratags") or {}).get("opening_hours") or "")))
    return sorted(out, key=lambda f: f.metres)


_DAYS = ("mo", "tu", "we", "th", "fr", "sa", "su")
_RULE = re.compile(r"^(?P<days>(?:mo|tu|we|th|fr|sa|su)(?:\s*-\s*(?:mo|tu|we|th|fr|sa|su))?"
                   r"(?:\s*,\s*(?:mo|tu|we|th|fr|sa|su)(?:\s*-\s*(?:mo|tu|we|th|fr|sa|su))?)*)?\s*"
                   r"(?P<times>off|closed|\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}(?:\s*,\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})*)$")


def _day_set(spec: str) -> set[int]:
    days: set[int] = set()
    for part in spec.split(","):
        bits = [b.strip() for b in part.split("-")]
        if len(bits) == 1:
            days.add(_DAYS.index(bits[0]))
        else:
            start, end = _DAYS.index(bits[0]), _DAYS.index(bits[1])
            i = start
            while True:
                days.add(i)
                if i == end:
                    break
                i = (i + 1) % 7
    return days


def open_now(hours: str, when: datetime) -> bool | None:
    """Whether an OSM opening_hours string says open at *when*; None when it
    is missing or in a form not read here with certainty."""
    text = str(hours or "").strip().lower()
    if not text:
        return None
    if text == "24/7":
        return True
    if re.search(r"\bph\b", text):
        # "PH off" is on most UK shops: on an ordinary day it changes nothing.
        # On a bank holiday - or with no holiday list to check - don't guess.
        try:
            from .bank_holidays import events, holiday_on
            known = bool(events())
            holiday = holiday_on(when.date()) if known else None
        except Exception:
            known, holiday = False, None
        if not known or holiday:
            return None
        text = ";".join(r for r in text.split(";") if not re.search(r"\bph\b", r))
        if not text.strip():
            return None
    if re.search(r"\bsh\b|week|sunrise|sunset|dusk|dawn|\+|\"|\[|jan|feb|mar|apr|may|jun|jul|aug|"
                 r"sep|oct|nov|dec|easter", text):
        return None
    minute = when.hour * 60 + when.minute
    today, yesterday = when.weekday(), (when.weekday() - 1) % 7
    state: bool | None = False
    for rule in [r.strip() for r in text.split(";") if r.strip()]:
        m = _RULE.match(rule)
        if not m:
            return None
        days = _day_set(m.group("days")) if m.group("days") else set(range(7))
        times = m.group("times")
        if times in ("off", "closed"):
            if today in days:
                state = False
            continue
        for span in times.split(","):
            a, b = [t.strip() for t in span.split("-")]
            start = int(a[:-3]) * 60 + int(a[-2:])
            end = int(b[:-3]) * 60 + int(b[-2:])
            if end <= start:                      # past midnight: 17:00-03:00
                if (today in days and minute >= start) or (yesterday in days and minute < end):
                    state = True
            elif today in days and start <= minute < end:
                state = True
    return state


def _distance(metres: float) -> str:
    if metres < 400:
        return f"{round(metres / 10) * 10:.0f} metres"
    miles = metres / 1609.344
    return f"{miles:.1f} miles" if miles < 10 else f"{miles:.0f} miles"


def describe(found: list[Found], what: str, near: str, when: datetime | None = None,
             limit: int = 5) -> str:
    near_text = f" near {near}" if near else " nearby"
    if not found:
        return f"I couldn't find any {what}{near_text} on the map."
    when = when or datetime.now()
    lines = []
    for place in found[:limit]:
        where = ((f" at {place.street}" if place.street[:1].isdigit() else f" on {place.street}")
                 if place.street else "")
        state = open_now(place.hours, when)
        status = " — open now" if state is True else " — closed now" if state is False else ""
        lines.append(f"{place.name}{where}, {_distance(place.metres)}{status}")
    more = f" (and {len(found) - limit} more)" if len(found) > limit else ""
    return f"Nearest {what}{near_text}: " + "; ".join(lines) + more + "."


__all__ = ["Found", "describe", "open_now", "parse", "search", "selectors"]
