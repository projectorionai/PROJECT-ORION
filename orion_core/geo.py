"""
GeoIntelligenceEngine (Phase 8, Section D) — worldwide geospatial awareness.

The globe's old geocoder used Open-Meteo's search, which resolves major
cities but misses the towns, villages, districts and hamlets ORION is now
expected to find.  This engine gives ORION detailed geographical awareness of
the whole planet:

    geocode(query)              — resolve any place to coordinates + a rich
                                  address (country → region/state/county →
                                  city/town/village → district).  Backed by
                                  OpenStreetMap **Nominatim**, which indexes
                                  administrative units down to the hamlet.
    reverse(lat, lon)           — coordinates → the place that contains them.
    nearby(query, radius, kind) — every town/village/settlement within a
                                  radius, via the **Overpass** API over OSM
                                  ``place`` nodes.
    administrative(query)       — the admin hierarchy + boundary metadata.

All results are cached in a local SQLite spatial store
(``config/geo_cache.db``) so a place resolved once is instant and available
offline afterwards.  Network calls respect the Nominatim usage policy: a
descriptive User-Agent and a one-request-per-second floor, enforced here so
ORION is a good citizen of the free service.

Design: pure async service, no Qt beyond the bus log; every network call has
a hard timeout and degrades to the cache (then to ``None``) rather than
raising, so a geo lookup can never stall or crash a turn.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from urllib.parse import quote_plus

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from .lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .utils import first_line
from .db import apply_pragmas

GEO_CACHE_PATH = CONFIG_DIR / "geo_cache.db"
_USER_AGENT = "ORION-AI-OS/10.x (personal assistant; contact via desktop app)"
_NOMINATIM = "https://nominatim.openstreetmap.org"
_OVERPASS = "https://overpass-api.de/api/interpreter"

# OSM place=* values that count as populated settlements, coarse → fine.
_SETTLEMENT_KINDS = ("city", "town", "village", "hamlet", "suburb",
                     "quarter", "neighbourhood", "isolated_dwelling")

# UK postcodes are a special case: Nominatim's free-text search often resolves
# them coarsely (or to a nearby administrative centroid), and the Open-Meteo
# fallback doesn't understand postcodes at all — so a coastal postcode can
# land the globe camera out at sea.  ``postcodes.io`` is a free, keyless
# service that returns the EXACT centroid for any live UK postcode, so a
# postcode-shaped query is routed there first for pinpoint accuracy.
_POSTCODES_IO = "https://api.postcodes.io"
_UK_POSTCODE_RE = re.compile(
    r"^(GIR\s*0AA|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})$", re.IGNORECASE)


def looks_like_uk_postcode(query: str) -> bool:
    """True when *query* is shaped like a full UK postcode (with or without the
    internal space, any case) — e.g. 'EC1A 1BB', 'sw1a1aa', 'SW1A 1AA'."""
    return bool(_UK_POSTCODE_RE.match(str(query or "").strip()))


def normalise_uk_postcode(query: str) -> str:
    """Canonical UK postcode form: uppercase, a single space before the final
    three characters (the inward code).  'ec1a1bb' → 'EC1A 1BB'."""
    raw = re.sub(r"\s+", "", str(query or "")).upper()
    if len(raw) < 5:
        return raw
    return f"{raw[:-3]} {raw[-3:]}"


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    kind: str = ""                       # city/town/village/county/state…
    country: str = ""
    admin: dict[str, str] = field(default_factory=dict)   # region/county/…
    population: int | None = None
    display: str = ""

    def summary(self) -> str:
        bits = [self.name]
        for key in ("suburb", "city", "town", "village", "county",
                    "state", "region", "country"):
            value = self.admin.get(key)
            if value and value not in bits:
                bits.append(value)
        line = ", ".join(bits[:4])
        tail = f" [{self.kind}]" if self.kind else ""
        if self.population:
            tail += f", pop. {self.population:,}"
        return f"{line}{tail} — {self.lat:.4f}, {self.lon:.4f}"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lat": self.lat, "lon": self.lon,
                "kind": self.kind, "country": self.country,
                "admin": self.admin, "population": self.population,
                "display": self.display}


class GeoIntelligenceEngine:
    """Geocoding, reverse geocoding and nearby-settlement search over OSM."""

    MIN_INTERVAL_S = 1.05        # Nominatim policy: ≤ 1 request/second
    CACHE_TTL_DAYS = 90.0

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._lock = RLock()
        self._last_call = 0.0
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(GEO_CACHE_PATH, check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._conn)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS geo (
                query TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                cached_at REAL NOT NULL
            )""")
        self._conn.commit()
        if self.telemetry is not None:
            self.telemetry.health.register("geo")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── cache ─────────────────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, cached_at FROM geo WHERE query = ?",
                (key.lower(),)).fetchone()
        if row is None:
            return None
        if time.time() - row[1] > self.CACHE_TTL_DAYS * 86400.0:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def _cache_put(self, key: str, payload: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO geo(query, payload, cached_at) VALUES (?, ?, ?)",
                (key.lower(), json.dumps(payload), time.time()))
            self._conn.commit()

    # ── rate-limited fetch ────────────────────────────────────────────────────

    async def _get_json(self, url: str, timeout: float = 12.0) -> Any | None:
        # Enforce the one-request-per-second policy without blocking the loop.
        wait = self.MIN_INTERVAL_S - (time.monotonic() - self._last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()
        try:
            headers = {"User-Agent": _USER_AGENT, "Accept-Language": "en"}
            client_timeout = ClientTimeout(total=timeout, connect=5.0)
            async with ClientSession(timeout=client_timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return None
                    return await response.json(content_type=None)
        except Exception as exc:
            self.bus.log.emit(f"GEO: fetch failed - {first_line(exc, 80)}")
            return None

    # ── geocoding ─────────────────────────────────────────────────────────────

    async def geocode(self, query: str, limit: int = 1) -> list[Place]:
        """Resolve a place name (town, village, district…) worldwide."""
        query = str(query or "").strip()
        if not query:
            return []
        cache_key = f"geocode::{query}::{limit}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return [self._place_from_dict(d) for d in cached]
        # Pinpoint path: a UK postcode resolves through postcodes.io to its
        # exact centroid, so the globe flies to the real location rather than a
        # coarse administrative point (or, via the old fallback, open water).
        if looks_like_uk_postcode(query):
            place = await self._geocode_uk_postcode(query)
            if place is not None:
                self._cache_put(cache_key, [place.to_dict()])
                if self.telemetry is not None:
                    self.telemetry.metrics.incr("geo.geocode")
                    self.telemetry.health.beat("geo", "OK", f"postcode '{query[:12]}'")
                return [place]
            # else fall through to Nominatim, constrained to GB below.
        gb = "&countrycodes=gb" if looks_like_uk_postcode(query) else ""
        url = (f"{_NOMINATIM}/search?format=jsonv2&addressdetails=1&extratags=1"
               f"&accept-language=en{gb}&limit={max(1, min(10, limit))}&q={quote_plus(query)}")
        data = await self._get_json(url)
        if not data:
            return []
        places = [self._place_from_nominatim(item) for item in data]
        places = [p for p in places if p is not None]
        self._cache_put(cache_key, [p.to_dict() for p in places])
        if self.telemetry is not None:
            self.telemetry.metrics.incr("geo.geocode")
            self.telemetry.health.beat("geo", "OK", f"geocoded '{query[:30]}'")
        return places

    async def _geocode_uk_postcode(self, query: str) -> Place | None:
        """Resolve a UK postcode to its exact centroid via postcodes.io.

        Returns ``None`` (so the caller can fall back to Nominatim) when the
        postcode is unknown/terminated or the service is unreachable.
        """
        pc = normalise_uk_postcode(query)
        url = f"{_POSTCODES_IO}/postcodes/{quote_plus(pc)}"
        data = await self._get_json(url)
        if not isinstance(data, dict) or data.get("status") != 200:
            return None
        r = data.get("result") or {}
        lat, lon = r.get("latitude"), r.get("longitude")
        if lat is None or lon is None:
            return None
        name = r.get("postcode") or pc
        admin = {
            "postcode": name,
            "suburb": r.get("admin_ward") or "",
            "city": r.get("admin_district") or "",
            "town": r.get("admin_district") or "",
            "village": r.get("parish") or "",
            "county": r.get("admin_county") or "",
            "state": r.get("region") or r.get("country") or "",
            "region": r.get("region") or "",
            "country": r.get("country") or "",
        }
        display = ", ".join(x for x in (name, r.get("admin_district"),
                                        r.get("country")) if x)
        return Place(
            name=str(name), lat=float(lat), lon=float(lon), kind="postcode",
            country=str(r.get("country") or ""), admin=admin,
            display=display or str(name))

    async def reverse(self, lat: float, lon: float) -> Place | None:
        cache_key = f"reverse::{lat:.4f}::{lon:.4f}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return self._place_from_dict(cached) if cached else None
        url = (f"{_NOMINATIM}/reverse?format=jsonv2&addressdetails=1"
               f"&accept-language=en&lat={lat:.6f}&lon={lon:.6f}")
        data = await self._get_json(url)
        place = self._place_from_nominatim(data) if data else None
        self._cache_put(cache_key, place.to_dict() if place else {})
        return place

    async def nearby(self, query: str, radius_km: float = 50.0,
                     kinds: tuple[str, ...] = _SETTLEMENT_KINDS,
                     limit: int = 60) -> tuple[Place | None, list[Place]]:
        """Every settlement within *radius_km* of *query* (Overpass over OSM)."""
        anchors = await self.geocode(query, limit=1)
        if not anchors:
            return (None, [])
        anchor = anchors[0]
        radius_m = int(max(1.0, min(200.0, radius_km)) * 1000)
        cache_key = f"nearby::{anchor.lat:.3f}::{anchor.lon:.3f}::{radius_m}::{limit}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return (anchor, [self._place_from_dict(d) for d in cached])
        place_filter = "|".join(kinds)
        oql = (f"[out:json][timeout:25];"
               f"(node[\"place\"~\"^({place_filter})$\"]"
               f"(around:{radius_m},{anchor.lat:.5f},{anchor.lon:.5f}););"
               f"out {int(max(1, min(200, limit)))};")
        url = f"{_OVERPASS}?data={quote_plus(oql)}"
        data = await self._get_json(url, timeout=30.0)
        elements = (data or {}).get("elements") or []
        found: list[Place] = []
        for el in elements:
            tags = el.get("tags") or {}
            name = tags.get("name:en") or tags.get("name")
            if not name or el.get("lat") is None:
                continue
            pop = tags.get("population")
            found.append(Place(
                name=str(name), lat=float(el["lat"]), lon=float(el["lon"]),
                kind=str(tags.get("place") or ""), country=anchor.country,
                population=int(pop) if str(pop or "").isdigit() else None,
                display=str(name)))
        # Order by distance from the anchor, nearest first.
        found.sort(key=lambda p: self._haversine(anchor.lat, anchor.lon, p.lat, p.lon))
        self._cache_put(cache_key, [p.to_dict() for p in found])
        if self.telemetry is not None:
            self.telemetry.metrics.incr("geo.nearby")
        return (anchor, found)

    # ── tool-facing ───────────────────────────────────────────────────────────

    async def locate(self, query: str) -> ToolResult:
        """Spoken-friendly 'locate X' — resolves and flies the globe there."""
        places = await self.geocode(query, limit=1)
        if not places:
            return ToolResult(f"I couldn't locate '{query}' — unknown or offline.",
                              ok=False)
        p = places[0]
        try:
            self.bus.globe_request.emit(query)   # fly the on-screen globe there
        except Exception:
            pass
        return ToolResult(f"Located {p.summary()}")

    async def towns_near(self, query: str, radius_km: float = 50.0) -> ToolResult:
        anchor, found = await self.nearby(query, radius_km)
        if anchor is None:
            return ToolResult(f"I couldn't locate '{query}'.", ok=False)
        if not found:
            return ToolResult(f"No mapped settlements within {radius_km:.0f} km of "
                              f"{anchor.name}.")
        lines = [f"{len(found)} settlement(s) within {radius_km:.0f} km of "
                 f"{anchor.name}, {anchor.country} (nearest first):"]
        for p in found[:40]:
            dist = self._haversine(anchor.lat, anchor.lon, p.lat, p.lon)
            pop = f", pop. {p.population:,}" if p.population else ""
            lines.append(f"- {p.name} ({p.kind}, {dist:.0f} km{pop})")
        return ToolResult("\n".join(lines))

    async def describe(self, query: str) -> ToolResult:
        places = await self.geocode(query, limit=1)
        if not places:
            return ToolResult(f"No administrative record for '{query}'.", ok=False)
        p = places[0]
        lines = [f"{p.name} — {p.kind or 'place'}"]
        for key in ("suburb", "city", "town", "village", "county", "state",
                    "region", "country"):
            value = p.admin.get(key)
            if value:
                lines.append(f"  {key}: {value}")
        if p.population:
            lines.append(f"  population: {p.population:,}")
        lines.append(f"  coordinates: {p.lat:.4f}, {p.lon:.4f}")
        return ToolResult("\n".join(lines))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _place_from_nominatim(self, item: Any) -> Place | None:
        if not isinstance(item, dict) or item.get("lat") is None:
            return None
        addr = item.get("address") or {}
        extra = item.get("extratags") or {}
        postcode = addr.get("postcode") or ""
        # For a postcode/ZIP query Nominatim often returns an empty ``name`` but
        # carries the code in the address; surface it (with the settlement it
        # sits in, if any) so "SW1A 1AA" reads as a real place, not blank.
        name = (item.get("name") or addr.get("city") or addr.get("town")
                or addr.get("village") or addr.get("hamlet")
                or (f"{postcode}" if postcode and item.get("addresstype") == "postcode" else "")
                or postcode
                or str(item.get("display_name", "")).split(",")[0])
        pop = extra.get("population")
        admin = {
            "postcode": postcode,
            "suburb": addr.get("suburb") or addr.get("neighbourhood") or "",
            "city": addr.get("city") or "",
            "town": addr.get("town") or "",
            "village": addr.get("village") or addr.get("hamlet") or "",
            "county": addr.get("county") or "",
            "state": addr.get("state") or "",
            "region": addr.get("region") or addr.get("state_district") or "",
            "country": addr.get("country") or "",
        }
        return Place(
            name=str(name), lat=float(item["lat"]), lon=float(item["lon"]),
            kind=str(item.get("addresstype") or item.get("type") or ""),
            country=str(addr.get("country") or ""),
            admin={k: v for k, v in admin.items() if v},
            population=int(pop) if str(pop or "").isdigit() else None,
            display=str(item.get("display_name") or name))

    @staticmethod
    def _place_from_dict(d: dict[str, Any]) -> Place:
        return Place(name=d.get("name", ""), lat=d.get("lat", 0.0),
                     lon=d.get("lon", 0.0), kind=d.get("kind", ""),
                     country=d.get("country", ""), admin=d.get("admin", {}),
                     population=d.get("population"), display=d.get("display", ""))

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = (math.sin(dp / 2) ** 2
             + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
        return r * 2 * math.asin(min(1.0, math.sqrt(a)))
