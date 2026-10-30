"""
Journeys — how far, how long, and the way there.

The geo tool's vocabulary always promised "distance", "how far" and
"directions", but it could only locate places, list towns nearby and describe
them: "how far is Leeds from London" had nothing to answer with. This adds:

    distance     as the crow flies (exact, offline) and by road with a
                 driving time (the public OSRM router, keyless; its usage
                 policy asks for occasional use and an identifying agent,
                 which one spoken question is)
    directions   the route opened in Google Maps in the user's browser, by
                 car, on foot, by public transport or by bike

Miles first (UK), kilometres alongside. A road distance that cannot be had is
left out and said to be unavailable — never estimated from the straight line.
"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import quote_plus

OSRM_ROUTE = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false&steps=true"
USER_AGENT = "ORION-AI-OS (personal assistant; occasional journey queries)"
KM_PER_MILE = 1.609344
EARTH_KM = 6371.0088

MODES = {"driving": "driving", "drive": "driving", "car": "driving",
         "walking": "walking", "walk": "walking", "foot": "walking", "on foot": "walking",
         "transit": "transit", "public transport": "transit", "bus": "transit",
         "train": "transit", "cycling": "bicycling", "bike": "bicycling",
         "bicycle": "bicycling", "bicycling": "bicycling"}


def crow_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return EARTH_KM * 2 * math.asin(min(1.0, math.sqrt(a)))


def _miles_km(km: float) -> str:
    miles = km / KM_PER_MILE
    if km < 1.0:
        return f"{km * 1000:.0f} metres"
    fmt = (lambda v: f"{v:.1f}") if miles < 10 else (lambda v: f"{v:,.0f}")
    return f"{fmt(miles)} miles ({fmt(km)} km)"


def spoken_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60.0))
    if minutes < 1:
        return "under a minute"
    hours, mins = divmod(minutes, 60)
    if not hours:
        return f"{mins} minute{'s' if mins != 1 else ''}"
    text = f"{hours} hour{'s' if hours != 1 else ''}"
    return text + (f" {mins} minute{'s' if mins != 1 else ''}" if mins else "")


MAX_SNAP_M = 25_000


def _reached(data: dict[str, Any]) -> bool:
    """Whether the router actually routed between the places asked for.
    It snaps each end to the nearest road it can reach: New York became Cabo
    da Roca in Portugal and was reported as 1,420 miles by road. Each
    waypoint says how far it was moved; more than 25 km means no route."""
    if data.get("code") != "Ok":
        return False
    for point in data.get("waypoints") or []:
        try:
            if float(point.get("distance") or 0.0) > MAX_SNAP_M:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _uses_ferry(route: dict[str, Any]) -> bool:
    return any(step.get("mode") == "ferry" for leg in route.get("legs") or []
               for step in leg.get("steps") or [])


def describe(origin: str, destination: str, crow: float,
             road_km: float | None, road_s: float | None, *, road_tried: bool = True,
             no_road: bool = False, ferry: bool = False) -> str:
    """One answer to "how far is A from B". *road_tried* False (another
    continent) leaves the road out rather than calling it unavailable;
    *no_road* says the router answered and there is no road between them."""
    text = f"{destination} is {_miles_km(crow)} from {origin} as the crow flies"
    if road_km is not None and road_s is not None:
        text += (f"; by road{' and ferry' if ferry else ''} it's about {_miles_km(road_km)}, "
                 f"roughly {spoken_duration(road_s)} without traffic"
                 + (", including the crossing" if ferry else ""))
    elif no_road:
        text += "; there's no road route between them"
    elif road_tried:
        text += " (the road distance isn't available right now)"
    return text + "."


async def road(lat1: float, lon1: float, lat2: float, lon2: float,
               session: Any = None) -> tuple[float, float, bool] | bool | None:
    """(km, seconds, takes a ferry) by road; False when the router answered that there is
    no road between them; None when it could not be asked."""
    url = OSRM_ROUTE.format(lat1=f"{lat1:.5f}", lon1=f"{lon1:.5f}",
                            lat2=f"{lat2:.5f}", lon2=f"{lon2:.5f}")

    async def get(client: Any) -> tuple[float, float, bool] | bool | None:
        async with client.get(url, headers={"User-Agent": USER_AGENT}) as response:
            if response.status not in (200, 400):
                return None
            data = await response.json(content_type=None)
        routes = (data.get("routes") or []) if _reached(data) else []
        if not routes:
            return False if data.get("code") in ("Ok", "NoRoute", "NoSegment") else None
        return (float(routes[0]["distance"]) / 1000.0, float(routes[0]["duration"]),
                _uses_ferry(routes[0]))

    try:
        if session is not None:
            return await get(session)
        from aiohttp import ClientSession, ClientTimeout
        async with ClientSession(timeout=ClientTimeout(total=8.0, connect=4.0)) as own:
            return await get(own)
    except Exception:
        return None


OSRM_SHAPE = ("https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
              "?overview=full&geometries=geojson&steps=true")
MAX_ROUTE_POINTS = 1500


async def route_shape(lat1: float, lon1: float, lat2: float, lon2: float,
                      session: Any = None) -> tuple[float, float, list[list[float]], bool] | None:
    """(km, seconds, [[lon, lat], ...], takes a ferry) of the driving route - the line the
    Globe draws - or None. Thinned to MAX_ROUTE_POINTS, ends kept."""
    url = OSRM_SHAPE.format(lat1=f"{lat1:.5f}", lon1=f"{lon1:.5f}",
                            lat2=f"{lat2:.5f}", lon2=f"{lon2:.5f}")

    async def get(client: Any) -> tuple[float, float, list[list[float]], bool] | None:
        async with client.get(url, headers={"User-Agent": USER_AGENT}) as response:
            if response.status != 200:
                return None
            data = await response.json(content_type=None)
        routes = (data.get("routes") or []) if _reached(data) else []
        if not routes:
            return None
        coords = [[float(x), float(y)] for x, y in (routes[0].get("geometry") or {}).get("coordinates") or []]
        if len(coords) < 2:
            return None
        if len(coords) > MAX_ROUTE_POINTS:
            step = len(coords) / (MAX_ROUTE_POINTS - 1)
            coords = [coords[int(i * step)] for i in range(MAX_ROUTE_POINTS - 1)] + [coords[-1]]
        return (float(routes[0]["distance"]) / 1000.0, float(routes[0]["duration"]), coords,
                _uses_ferry(routes[0]))

    try:
        if session is not None:
            return await get(session)
        from aiohttp import ClientSession, ClientTimeout
        async with ClientSession(timeout=ClientTimeout(total=10.0, connect=4.0)) as own:
            return await get(own)
    except Exception:
        return None


def directions_url(origin: str, destination: str, mode: str = "driving") -> str:
    """Google Maps directions (the documented maps/dir URL, no key). An empty
    origin lets Maps start from the browser's own location."""
    travel = MODES.get(str(mode or "driving").strip().lower(), "driving")
    url = f"https://www.google.com/maps/dir/?api=1&destination={quote_plus(destination)}&travelmode={travel}"
    if origin:
        url += f"&origin={quote_plus(origin)}"
    return url


__all__ = ["MODES", "crow_km", "describe", "directions_url", "road", "route_shape", "spoken_duration"]
